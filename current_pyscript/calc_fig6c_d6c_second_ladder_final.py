#!/usr/bin/env python3
from pathlib import Path
import json
import traceback
import numpy as np
import pandas as pd

import TUfRG_instability as inst

# Reuse dataset-discovery utilities from the established D6a script.
from calc_fig6a_d6a_corrected_strict_precritical import (
    ROOTS,
    parse_params,
    find_case_roots,
    case_display_name,
)

# Reuse the audited Fig.6c/D6c source convention from the PACE basis audit script.
import check_fig6c_d6c_pace_basis_convention as d6csrc


OUTDIR = Path("fig6c_d6c_second_ladder_final")
OUTDIR.mkdir(parents=True, exist_ok=True)

Q_LABELS = ["Gamma", "M1", "M2", "M3"]


def preferred_last_snapshot(run_dir: Path):
    """
    Use the final FRG step.

    Priority:
      1. snapshots/final/O.dvg
      2. root O.dvg
      3. latest snapshots/iter_XXXX/O.dvg
    """
    final_dir = run_dir / "snapshots" / "final"
    if (final_dir / "O.dvg").exists():
        return final_dir, "final"

    if (run_dir / "O.dvg").exists():
        return run_dir, "root_final"

    snap_root = run_dir / "snapshots"
    if snap_root.exists():
        found = []
        for p in snap_root.iterdir():
            if p.is_dir() and p.name.startswith("iter_") and (p / "O.dvg").exists():
                try:
                    it = int(p.name.split("_")[1])
                except Exception:
                    it = -1
                found.append((it, p))
        if found:
            found.sort()
            p = found[-1][1]
            return p, p.name

    return None, None


def comp_key_int(R, ofrom, oto):
    return ((int(R[0]), int(R[1])), int(ofrom), int(oto))


def reverse_key_int(key):
    R, a, b = key
    return ((-int(R[0]), -int(R[1])), int(b), int(a))


def qmap_get(qmap, label):
    if label in qmap:
        return qmap[label]
    alt = label + "_nearest"
    if alt in qmap:
        return qmap[alt]
    if label == "Gamma" and "G" in qmap:
        return qmap["G"]
    raise KeyError(f"Cannot find q label {label}; qmap={qmap}")


def make_d6c_sources_on_O(O):
    """
    Construct D6c source vectors on the actual O.dvg TU basis.

    Global normalization:
        sum_q ||eta_q||^2 = 1

    For D6c:
        ||eta_Gamma||^2 = ||eta_M1||^2 = ||eta_M2||^2 = ||eta_M3||^2 = 1/4.
    """
    Rarr = np.asarray(O.mi_to_R, dtype=int)
    ofrom = np.asarray(O.mi_to_ofrom, dtype=int).reshape(-1)
    oto = np.asarray(O.mi_to_oto, dtype=int).reshape(-1)
    nff = len(ofrom)

    key_to_idx = {
        comp_key_int(Rarr[i], ofrom[i], oto[i]): i
        for i in range(nff)
    }

    terms = d6csrc.build_terms()

    etas = {}

    for qname, qred, _iq_expected in d6csrc.Q_LIST:
        eta = np.zeros(nff, dtype=np.complex128)

        for t in terms:
            sgn = int(t["sign"])

            R, a, b = t["key"]
            key = comp_key_int(R, d6csrc.SUB[a], d6csrc.SUB[b])
            rkey = reverse_key_int(key)

            if key not in key_to_idx:
                raise RuntimeError(f"missing D6c key {key} for {t}")
            if rkey not in key_to_idx:
                raise RuntimeError(f"missing D6c reverse key {rkey} for {t}")

            eta[key_to_idx[key]] += 1j * sgn * d6csrc.q_phase(qred, t["home"])
            eta[key_to_idx[rkey]] += -1j * sgn * d6csrc.q_phase(qred, t["reverse_home"])

        # Use Gamma label, not G.
        if qname == "G":
            qname = "Gamma"

        etas[qname] = eta

    total_norm2 = sum(float(np.vdot(e, e).real) for e in etas.values())
    if total_norm2 < 1e-14:
        raise RuntimeError("zero D6c source")

    gnorm = np.sqrt(total_norm2)
    etas = {q: e / gnorm for q, e in etas.items()}

    norm2 = {q: float(np.vdot(e, e).real) for q, e in etas.items()}
    total = sum(norm2.values())

    if abs(total - 1.0) > 1e-12:
        raise RuntimeError(f"bad total D6c source norm: {total}")

    for q in Q_LABELS:
        if abs(norm2[q] - 0.25) > 1e-12:
            raise RuntimeError(f"D6c q weight not 1/4 for {q}: {norm2}")

    diag = {
        "source_norm2_total": float(total),
        "source_norm2_Gamma": norm2["Gamma"],
        "source_norm2_M1": norm2["M1"],
        "source_norm2_M2": norm2["M2"],
        "source_norm2_M3": norm2["M3"],
        "source_nonzero_Gamma": int(np.count_nonzero(np.abs(etas["Gamma"]) > 1e-12)),
        "source_nonzero_M1": int(np.count_nonzero(np.abs(etas["M1"]) > 1e-12)),
        "source_nonzero_M2": int(np.count_nonzero(np.abs(etas["M2"]) > 1e-12)),
        "source_nonzero_M3": int(np.count_nonzero(np.abs(etas["M3"]) > 1e-12)),
    }

    return etas, diag


def hermitize(A):
    A = np.asarray(A, dtype=np.complex128)
    return 0.5 * (A + A.conj().T)


def psd_clip(A, rel_tol=1e-10):
    """
    Return PSD-clipped Hermitian matrix and diagnostics.
    """
    Ah = hermitize(A)
    w, U = np.linalg.eigh(Ah)

    max_abs = float(np.max(np.abs(w))) if len(w) else 0.0
    cutoff = rel_tol * max(max_abs, 1.0)

    neg_weight = float(np.sum(np.abs(w[w < -cutoff])))
    pos_weight = float(np.sum(np.abs(w[w > cutoff])))
    neg_ratio = neg_weight / max(pos_weight + neg_weight, 1e-300)

    w_clip = np.where(w > cutoff, w, 0.0)
    A_psd = (U * w_clip) @ U.conj().T

    return A_psd, {
        "chi0_eval_min_raw": float(np.min(w)),
        "chi0_eval_max_raw": float(np.max(w)),
        "chi0_neg_weight": neg_weight,
        "chi0_pos_weight": pos_weight,
        "chi0_neg_weight_ratio": neg_ratio,
        "chi0_clip_cutoff": float(cutoff),
        "chi0_rank_psd": int(np.count_nonzero(w_clip > 0)),
    }


def second_ladder_for_eta(eta, K_saved, L_saved):
    """
    Current convention:
        chi0  = -L_saved
        Gamma = -K_saved

    Expansion:
        chi = eta^dag chi0 eta
            + eta^dag chi0 Gamma chi0 eta
            + ...

    Return bare, second term, and bare+second.
    """
    eta = np.asarray(eta, dtype=np.complex128).reshape(-1)

    Lh_saved = hermitize(L_saved)
    Kh_saved = hermitize(K_saved)

    chi0_raw = -Lh_saved
    Gamma = -Kh_saved
    Gamma_h = hermitize(Gamma)

    chi0, chi0_diag = psd_clip(chi0_raw)

    g = chi0 @ eta

    chi_bare = np.vdot(eta, chi0 @ eta)
    chi_second = np.vdot(g, Gamma_h @ g)
    chi_bare_plus_second = chi_bare + chi_second

    # Dimensionless normalized correction, useful for comparing cases.
    denom = float(np.real(chi_bare))
    if abs(denom) > 1e-14:
        ratio_second_over_bare = float(np.real(chi_second) / denom)
        ratio_plus_over_bare = float(np.real(chi_bare_plus_second) / denom)
    else:
        ratio_second_over_bare = np.nan
        ratio_plus_over_bare = np.nan

    # Also record eigenvalues of B = chi0^{1/2} Gamma chi0^{1/2},
    # only as a diagnostic; we do NOT invert I-B here.
    w0, U0 = np.linalg.eigh(chi0)
    w0p = np.clip(w0, 0.0, None)
    sqrt_chi0 = (U0 * np.sqrt(w0p)) @ U0.conj().T
    B = hermitize(sqrt_chi0 @ Gamma_h @ sqrt_chi0)
    evals_B = np.linalg.eigvalsh(B)

    out = {
        "chi_bare_real": float(np.real(chi_bare)),
        "chi_bare_imag": float(np.imag(chi_bare)),
        "chi_second_real": float(np.real(chi_second)),
        "chi_second_imag": float(np.imag(chi_second)),
        "chi_bare_plus_second_real": float(np.real(chi_bare_plus_second)),
        "chi_bare_plus_second_imag": float(np.imag(chi_bare_plus_second)),
        "ratio_second_over_bare": ratio_second_over_bare,
        "ratio_plus_over_bare": ratio_plus_over_bare,
        "B_eval_min": float(np.min(evals_B)),
        "B_eval_max": float(np.max(evals_B)),
        "B_abs_eval_max": float(np.max(np.abs(evals_B))),
    }
    out.update(chi0_diag)
    return out


def compute_one(root_label, root_path, run_dir, snapshot_dir, snap_label):
    case_name = case_display_name(run_dir)
    params = parse_params(case_name)

    print(f"[case] {root_label} :: {case_name} :: {snap_label}", flush=True)

    data = inst.load_case(snapshot_dir)
    inst.assert_raw_channel_compatibility(data)

    qmap = inst.q_labels_to_indices(data, Q_LABELS)
    etas_by_q, source_diag = make_d6c_sources_on_O(data.O)

    rows = []

    for qlab in Q_LABELS:
        iq = qmap_get(qmap, qlab)
        eta = etas_by_q[qlab]

        kernels = inst.build_explicit_kernels(data, iq)

        for spec in kernels:
            if spec["family"] != "ph":
                continue
            if spec["sector"] not in ["ph-rho", "ph-sx", "ph-sy", "ph-sz"]:
                continue

            K_saved = np.asarray(spec["K"], dtype=np.complex128)
            L_saved = np.asarray(spec["L"], dtype=np.complex128)

            diag = second_ladder_for_eta(eta, K_saved, L_saved)

            rows.append({
                "root_label": root_label,
                "root_path": str(root_path),
                "case_name": case_name,
                "run_dir": str(run_dir),
                "snapshot_label": snap_label,
                "snapshot_dir": str(snapshot_dir),
                "q_label": qlab,
                "iq": int(iq),
                "sector": spec["sector"],
                "source": "D6c_fig6c_second_ladder_final",
                **source_diag,
                **diag,
                **params,
            })

    return rows


def make_summed(df):
    if len(df) == 0:
        return df

    keys = [
        "root_label", "case_name", "U", "V",
        "sector", "snapshot_label",
    ]

    out = df.groupby(keys).agg(
        chi_bare_sum=("chi_bare_real", "sum"),
        chi_second_sum=("chi_second_real", "sum"),
        chi_bare_plus_second_sum=("chi_bare_plus_second_real", "sum"),
        ratio_second_over_bare_mean=("ratio_second_over_bare", "mean"),
        B_eval_max_max=("B_eval_max", "max"),
        B_abs_eval_max_max=("B_abs_eval_max", "max"),
        chi0_neg_weight_ratio_max=("chi0_neg_weight_ratio", "max"),
        n_q=("q_label", "nunique"),
    ).reset_index()

    return out.sort_values(keys).reset_index(drop=True)


def write_outputs(rows, errors):
    df = pd.DataFrame(rows)
    df.to_csv(OUTDIR / "fig6c_d6c_second_ladder_final_all.csv", index=False)

    summed = make_summed(df)
    summed.to_csv(OUTDIR / "fig6c_d6c_second_ladder_final_summed.csv", index=False)

    if len(summed):
        compact = summed.pivot_table(
            index=[
                "root_label", "case_name", "U", "V", "snapshot_label",
                "B_eval_max_max", "B_abs_eval_max_max", "chi0_neg_weight_ratio_max",
            ],
            columns="sector",
            values="chi_second_sum",
            aggfunc="first",
        ).reset_index()
        compact.to_csv(OUTDIR / "fig6c_d6c_second_ladder_final_compact_second_only.csv", index=False)

        compact2 = summed.pivot_table(
            index=[
                "root_label", "case_name", "U", "V", "snapshot_label",
                "B_eval_max_max", "B_abs_eval_max_max", "chi0_neg_weight_ratio_max",
            ],
            columns="sector",
            values="chi_bare_plus_second_sum",
            aggfunc="first",
        ).reset_index()
        compact2.to_csv(OUTDIR / "fig6c_d6c_second_ladder_final_compact_bare_plus_second.csv", index=False)

    (OUTDIR / "errors.json").write_text(json.dumps(errors, indent=2))


def main():
    rows = []
    errors = []
    inventory = []

    for root_label, root_str in ROOTS:
        case_roots = find_case_roots(root_str)

        entry = {
            "root_label": root_label,
            "root_path": root_str,
            "n_cases": len(case_roots),
            "cases": [],
        }

        for rd in case_roots:
            snap, lab = preferred_last_snapshot(rd)
            entry["cases"].append({
                "case_name": case_display_name(rd),
                "run_dir": str(rd),
                "selected_snapshot": str(snap) if snap is not None else None,
                "selected_label": lab,
            })

        inventory.append(entry)

    (OUTDIR / "inventory.json").write_text(json.dumps(inventory, indent=2))

    print("=== Inventory ===", flush=True)
    for item in inventory:
        nsel = sum(1 for c in item["cases"] if c["selected_snapshot"])
        print(f"{item['root_label']:20s} {item['n_cases']:3d} cases, {nsel:3d} selected", flush=True)

    for root_label, root_str in ROOTS:
        for rd in find_case_roots(root_str):
            snap, lab = preferred_last_snapshot(rd)

            if snap is None:
                errors.append({
                    "root_label": root_label,
                    "run_dir": str(rd),
                    "error": "no usable final O.dvg found",
                })
                write_outputs(rows, errors)
                continue

            try:
                new_rows = compute_one(root_label, root_str, rd, snap, lab)
                rows.extend(new_rows)
                write_outputs(rows, errors)

            except Exception as e:
                print(f"[ERROR] {root_label} {rd} {snap}: {e}", flush=True)
                errors.append({
                    "root_label": root_label,
                    "run_dir": str(rd),
                    "snapshot_dir": str(snap),
                    "error": repr(e),
                    "traceback": traceback.format_exc(),
                })
                write_outputs(rows, errors)

    write_outputs(rows, errors)

    df = pd.DataFrame(rows)
    print("\n=== Done ===", flush=True)
    print("rows:", len(df), flush=True)
    print("errors:", len(errors), flush=True)
    print("outdir:", OUTDIR.resolve(), flush=True)

    if len(df):
        print("\n=== Source q weights ===", flush=True)
        print(
            df[[
                "source_norm2_Gamma",
                "source_norm2_M1",
                "source_norm2_M2",
                "source_norm2_M3",
            ]].drop_duplicates().to_string(index=False),
            flush=True,
        )

        print("\n=== Sector counts ===", flush=True)
        print(df["sector"].value_counts().to_string(), flush=True)

        print("\n=== q counts ===", flush=True)
        print(df["q_label"].value_counts().to_string(), flush=True)

        print("\n=== Compact second-only preview ===", flush=True)
        p = OUTDIR / "fig6c_d6c_second_ladder_final_compact_second_only.csv"
        if p.exists():
            print(pd.read_csv(p).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
