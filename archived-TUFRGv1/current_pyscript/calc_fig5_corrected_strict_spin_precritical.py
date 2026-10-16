#!/usr/bin/env python3
from pathlib import Path
import re
import json
import traceback
import numpy as np
import pandas as pd

import TUfRG_instability as inst


ROOTS = [
    ("nosoc", "nosoc"),
    ("lambda_nn_0p1", "lnn0p1_soc"),
    ("lambda_nnn_0p1", "lnnn0p1_soc"),
]

OUTDIR = Path("fig5_corrected_strict_spin_precritical")
OUTDIR.mkdir(parents=True, exist_ok=True)

SUB = {"A": 0, "B": 1, "C": 2}

UP_KEYS = [
    ((0, 0), SUB["A"], SUB["B"]),
    ((0, 0), SUB["B"], SUB["C"]),
    ((0, 0), SUB["C"], SUB["A"]),
]

DOWN_KEYS = [
    ((-1, 0), SUB["B"], SUB["A"]),
    ((0, -1), SUB["C"], SUB["A"]),
    ((1, -1), SUB["C"], SUB["B"]),
]


# ============================================================
# Corrected Fig.5 source
# ============================================================

def comp_key(R, ofrom, oto):
    return ((int(R[0]), int(R[1])), int(ofrom), int(oto))


def reverse_key(key):
    R, ofrom, oto = key
    return ((-int(R[0]), -int(R[1])), int(oto), int(ofrom))


def make_current_source(O, up_signs, down_signs):
    R = np.asarray(O.mi_to_R, dtype=int)
    ofrom = np.asarray(O.mi_to_ofrom, dtype=int)
    oto = np.asarray(O.mi_to_oto, dtype=int)
    nff = len(ofrom)

    key_to_idx = {
        comp_key(R[i], ofrom[i], oto[i]): i
        for i in range(nff)
    }

    eta = np.zeros(nff, dtype=np.complex128)

    for key_raw, J in zip(UP_KEYS, up_signs):
        key = comp_key(*key_raw)
        rkey = reverse_key(key)

        if key not in key_to_idx:
            raise RuntimeError(f"Missing UP key {key}")
        if rkey not in key_to_idx:
            raise RuntimeError(f"Missing reverse key {rkey} for {key}")

        i = key_to_idx[key]
        j = key_to_idx[rkey]

        eta[i] += 1j * float(J)
        eta[j] += -1j * float(J)

    for key_raw, J in zip(DOWN_KEYS, down_signs):
        key = comp_key(*key_raw)
        rkey = reverse_key(key)

        if key not in key_to_idx:
            raise RuntimeError(f"Missing DOWN key {key}")
        if rkey not in key_to_idx:
            raise RuntimeError(f"Missing reverse key {rkey} for {key}")

        i = key_to_idx[key]
        j = key_to_idx[rkey]

        eta[i] += 1j * float(J)
        eta[j] += -1j * float(J)

    nrm = np.linalg.norm(eta)
    if nrm < 1e-14:
        raise RuntimeError("zero source")

    eta /= nrm

    nonzero = int(np.count_nonzero(np.abs(eta) > 1e-12))
    if nonzero != 12:
        raise RuntimeError(f"bad source nonzero count: {nonzero}")

    max_herm_err = 0.0
    for i in range(nff):
        key = comp_key(R[i], ofrom[i], oto[i])
        rkey = reverse_key(key)
        j = key_to_idx.get(rkey, None)
        if j is not None:
            max_herm_err = max(max_herm_err, abs(eta[j] - np.conjugate(eta[i])))

    if max_herm_err > 1e-12:
        raise RuntimeError(f"Hermiticity failed: {max_herm_err}")

    return eta, {
        "eta_nonzero": nonzero,
        "eta_norm": float(np.linalg.norm(eta)),
        "eta_max_herm_err": float(max_herm_err),
    }


def make_fig5_sources(O):
    # Corrected signs:
    # Fig.5(a): same up/down flux
    # Fig.5(b): opposite up/down flux
    eta5a, d5a = make_current_source(
        O,
        up_signs=[+1, +1, +1],
        down_signs=[-1, +1, -1],
    )

    eta5b, d5b = make_current_source(
        O,
        up_signs=[+1, +1, +1],
        down_signs=[+1, -1, +1],
    )

    ov = np.vdot(eta5a, eta5b)
    if abs(ov) > 1e-12:
        raise RuntimeError(f"Fig5a/Fig5b not orthogonal: overlap={ov}")

    return {
        "fig5a": (eta5a, d5a),
        "fig5b": (eta5b, d5b),
    }


# ============================================================
# Strict susceptibility
#
# Physical convention:
#   chi0       = sign * L_saved
#   Gamma_irr = sign * K_saved
#
# For PH loop in this divERGe convention, usually sign=-1:
#   chi0 = -L_saved, Gamma = -K_saved.
#
# Choose sign by which chi0 is closest to PSD.
# ============================================================

def hermitian_part(M):
    M = np.asarray(M, dtype=np.complex128)
    return 0.5 * (M + M.conj().T)


def psd_project_and_sqrt(M, neg_weight_tol=5e-3):
    H = hermitian_part(M)
    evals, evecs = np.linalg.eigh(H)

    neg = evals[evals < 0.0]
    neg_weight = float(np.sum(np.abs(neg)))
    total_weight = float(np.sum(np.abs(evals)) + 1e-300)
    neg_weight_ratio = neg_weight / total_weight

    max_abs = max(float(np.max(np.abs(evals))), 1.0)
    min_ratio = float(np.min(evals) / max_abs)

    if neg_weight_ratio > neg_weight_tol:
        raise ValueError(
            f"chi0 not safely PSD: "
            f"neg_weight_ratio={neg_weight_ratio:.3e}, "
            f"min_ratio={min_ratio:.3e}"
        )

    evals_pos = np.clip(evals, 0.0, None)
    M_pos = hermitian_part((evecs * evals_pos) @ evecs.conj().T)
    M_sqrt = hermitian_part((evecs * np.sqrt(evals_pos)) @ evecs.conj().T)

    diag = {
        "chi0_min_eval_raw": float(np.min(evals)),
        "chi0_max_eval_raw": float(np.max(evals)),
        "chi0_neg_weight_ratio": float(neg_weight_ratio),
        "chi0_min_ratio": float(min_ratio),
        "chi0_rank_pos": int(np.sum(evals_pos > 1e-14 * max(float(np.max(evals_pos)), 1.0))),
    }

    return M_pos, M_sqrt, diag


def choose_chi0_and_gamma(K_saved, L_saved, neg_weight_tol=5e-3):
    K_h = hermitian_part(K_saved)
    L_h = hermitian_part(L_saved)

    candidates = []
    for sign in [+1, -1]:
        chi0_try = sign * L_h
        evals = np.linalg.eigvalsh(chi0_try)

        neg_weight = float(np.sum(np.abs(evals[evals < 0.0])))
        total_weight = float(np.sum(np.abs(evals)) + 1e-300)
        neg_weight_ratio = neg_weight / total_weight

        max_abs = max(float(np.max(np.abs(evals))), 1.0)
        min_ratio = float(np.min(evals) / max_abs)

        candidates.append({
            "sign": sign,
            "chi0_try": chi0_try,
            "Gamma_try": sign * K_h,
            "neg_weight_ratio": float(neg_weight_ratio),
            "min_ratio": float(min_ratio),
            "min_eval": float(np.min(evals)),
            "max_eval": float(np.max(evals)),
        })

    candidates.sort(key=lambda x: x["neg_weight_ratio"])
    best = candidates[0]

    chi0, chi0_sqrt, psd_diag = psd_project_and_sqrt(
        best["chi0_try"],
        neg_weight_tol=neg_weight_tol,
    )

    diag = {
        "loop_orientation_sign": int(best["sign"]),
        "chosen_neg_weight_ratio": float(best["neg_weight_ratio"]),
        "chosen_min_ratio": float(best["min_ratio"]),
        "chosen_min_eval": float(best["min_eval"]),
        "chosen_max_eval": float(best["max_eval"]),
        **psd_diag,
    }

    return chi0, chi0_sqrt, best["Gamma_try"], diag


def strict_chi_for_eta(eta, K_saved, L_saved, neg_weight_tol=5e-3):
    eta = np.asarray(eta, dtype=np.complex128)
    eta = eta / np.linalg.norm(eta)

    K_saved = np.asarray(K_saved, dtype=np.complex128)
    L_saved = np.asarray(L_saved, dtype=np.complex128)

    chi0, chi0_sqrt, Gamma, diag0 = choose_chi0_and_gamma(
        K_saved,
        L_saved,
        neg_weight_tol=neg_weight_tol,
    )

    B = hermitian_part(chi0_sqrt @ Gamma @ chi0_sqrt)
    Bevals, Bevecs = np.linalg.eigh(B)

    I = np.eye(B.shape[0], dtype=np.complex128)
    M = I - B

    Rmat = hermitian_part(chi0_sqrt @ np.linalg.solve(M, chi0_sqrt))
    chi_direct = np.vdot(eta, Rmat @ eta)

    b = chi0_sqrt @ eta
    coeffs = Bevecs.conj().T @ b
    chi_spectral = np.sum(np.abs(coeffs) ** 2 / (1.0 - Bevals))

    old_proxy = np.vdot(eta, (K_saved @ L_saved) @ eta)

    diag = {
        **diag0,
        "K_saved_hermiticity_rel": float(np.linalg.norm(K_saved - K_saved.conj().T) / max(np.linalg.norm(K_saved), 1e-300)),
        "L_saved_hermiticity_rel": float(np.linalg.norm(L_saved - L_saved.conj().T) / max(np.linalg.norm(L_saved), 1e-300)),
        "B_min_eval": float(np.min(Bevals)),
        "B_max_eval": float(np.max(Bevals)),
        "min_1_minus_B": float(np.min(1.0 - Bevals)),
        "condition_I_minus_B": float(np.linalg.cond(M)),
        "R_hermiticity_rel": float(np.linalg.norm(Rmat - Rmat.conj().T) / max(np.linalg.norm(Rmat), 1e-300)),
        "chi_strict_real": float(np.real(chi_direct)),
        "chi_strict_imag": float(np.imag(chi_direct)),
        "chi_spectral_real": float(np.real(chi_spectral)),
        "chi_spectral_imag": float(np.imag(chi_spectral)),
        "direct_spectral_diff": float(abs(chi_direct - chi_spectral)),
        "old_proxy_real": float(np.real(old_proxy)),
        "old_proxy_imag": float(np.imag(old_proxy)),
        "normal_state_valid": bool(np.max(Bevals) < 1.0),
        "precritical_0p95": bool(np.max(Bevals) < 0.95),
        "precritical_0p90": bool(np.max(Bevals) < 0.90),
    }

    return diag



# ============================================================
# Case and snapshot discovery
# ============================================================

def pnum(s):
    return float(s.replace("p", "."))


def parse_params(name):
    out = {
        "lambda_nn": np.nan,
        "lambda_nnn": np.nan,
        "U": np.nan,
        "V": np.nan,
    }

    m = re.search(r"lnn([0-9]+p[0-9]+|[0-9]+)", name)
    if m:
        out["lambda_nn"] = pnum(m.group(1))

    m = re.search(r"lnnn([0-9]+p[0-9]+|[0-9]+)", name)
    if m:
        out["lambda_nnn"] = pnum(m.group(1))

    m = re.search(r"_U([0-9]+p[0-9]+|[0-9]+)", name)
    if m:
        out["U"] = pnum(m.group(1))

    m = re.search(r"_V([0-9]+p[0-9]+|[0-9]+)", name)
    if m:
        out["V"] = pnum(m.group(1))

    return out


def has_output_dir(p):
    p = Path(p)
    return (p / "O.dvg").exists() or (p / "snapshots" / "final" / "O.dvg").exists()


def normalize_case_run_dir(p):
    """
    Return the directory that actually contains O.dvg/snapshots.
    Handles both:
        case/O.dvg
        case/snapshots/final/O.dvg
    and:
        case/run/O.dvg
        case/run/snapshots/final/O.dvg
    """
    p = Path(p)

    if has_output_dir(p):
        return p

    if has_output_dir(p / "run"):
        return p / "run"

    return None


def case_display_name(run_dir):
    run_dir = Path(run_dir)
    if run_dir.name == "run":
        return run_dir.parent.name
    return run_dir.name


def find_case_roots(root):
    root = Path(root)
    if not root.exists():
        return []

    out = []
    for p in sorted(root.iterdir()):
        if not p.is_dir():
            continue
        rd = normalize_case_run_dir(p)
        if rd is not None:
            out.append(rd)

    return out


def snapshot_sort_key(p):
    p = Path(p)
    if p.name == "final":
        return 10**12
    m = re.search(r"iter[_-]?([0-9]+)", p.name)
    if m:
        return int(m.group(1))
    return -1


def find_snapshots(run_dir):
    """
    Return available snapshots under run_dir.
    Include root-level O.dvg as root_final, then iter_*, then final.
    """
    run_dir = Path(run_dir)
    snaps = []

    if (run_dir / "O.dvg").exists():
        snaps.append(run_dir)

    sdir = run_dir / "snapshots"
    if sdir.exists():
        for p in sorted(sdir.iterdir(), key=snapshot_sort_key):
            if p.is_dir() and (p / "O.dvg").exists():
                snaps.append(p)

    # Deduplicate.
    uniq = []
    seen = set()
    for s in snaps:
        key = str(s.resolve())
        if key not in seen:
            seen.add(key)
            uniq.append(s)

    return sorted(uniq, key=snapshot_sort_key)


def snapshot_label(snapshot_dir, run_dir):
    snapshot_dir = Path(snapshot_dir)
    run_dir = Path(run_dir)

    if snapshot_dir.resolve() == run_dir.resolve():
        return "root_final"
    return snapshot_dir.name


def snapshot_iter(snapshot_dir, run_dir):
    lab = snapshot_label(snapshot_dir, run_dir)
    if lab in ["root_final", "final"]:
        return 10**12
    m = re.search(r"iter[_-]?([0-9]+)", lab)
    if m:
        return int(m.group(1))
    return -1


def select_light_snapshots(snaps):
    """
    To keep the job light, evaluate all available snapshots if there are not many.
    If there are many, keep every other/early-to-late coarse subset plus final.
    Since we only compute charge sector, this is still much lighter than v3.
    """
    snaps = list(snaps)
    if len(snaps) <= 12:
        return snaps

    # Keep roughly 10 snapshots: first, then every few, then final.
    idxs = sorted(set(
        [0, len(snaps)-1] +
        list(np.linspace(1, len(snaps)-2, 10, dtype=int))
    ))
    return [snaps[i] for i in idxs]


# ============================================================
# Main calculation
# ============================================================

def compute_snapshot(root_label, root_path, run_dir, snapshot_dir):
    case_name = case_display_name(run_dir)
    snap_lab = snapshot_label(snapshot_dir, run_dir)
    snap_it = snapshot_iter(snapshot_dir, run_dir)
    params = parse_params(case_name)

    print(f"[snapshot] {root_label} :: {case_name} :: {snap_lab}", flush=True)

    data = inst.load_case(snapshot_dir)
    inst.assert_raw_channel_compatibility(data)

    qmap = inst.q_labels_to_indices(data, ["Gamma"])
    iq = qmap["Gamma"]

    sources = make_fig5_sources(data.O)
    kernels = inst.build_explicit_kernels(data, iq)

    rows = []

    for spec in kernels:
        if spec["family"] != "ph":
            continue
        if spec["sector"] not in ["ph-sx", "ph-sy", "ph-sz"]:
            continue

        K_saved = np.asarray(spec["K"], dtype=np.complex128)
        L_saved = np.asarray(spec["L"], dtype=np.complex128)

        for source_name, (eta, source_diag) in sources.items():
            diag = strict_chi_for_eta(eta, K_saved, L_saved)

            rows.append({
                "root_label": root_label,
                "root_path": str(root_path),
                "case_name": case_name,
                "run_dir": str(run_dir),
                "snapshot_label": snap_lab,
                "snapshot_iter": int(snap_it),
                "snapshot_dir": str(snapshot_dir),
                "q_label": "Gamma",
                "iq": int(iq),
                "sector": "ph-rho",
                "source": source_name,
                **source_diag,
                **diag,
                **params,
            })

    return rows


def select_last_precritical(df, threshold):
    if len(df) == 0:
        return df

    d = df[df["B_max_eval"] < threshold].copy()
    if len(d) == 0:
        return d

    keys = ["root_label", "case_name", "U", "V", "sector", "source"]
    idx = d.groupby(keys)["snapshot_iter"].idxmax()
    return d.loc[idx].sort_values(keys).reset_index(drop=True)


def write_outputs(all_rows, errors):
    df = pd.DataFrame(all_rows)
    df.to_csv(OUTDIR / "fig5_corrected_strict_spin_precritical_all.csv", index=False)
    (OUTDIR / "errors.json").write_text(json.dumps(errors, indent=2))

    if len(df) == 0:
        return

    for thr, tag in [(0.90, "0p90"), (0.95, "0p95"), (1.00, "1p00")]:
        sel = select_last_precritical(df, thr)
        sel.to_csv(OUTDIR / f"selected_Bmax_lt_{tag}.csv", index=False)

        if len(sel):
            compact = sel.pivot_table(
                index=[
                    "root_label", "case_name", "U", "V",
                    "snapshot_label", "snapshot_iter",
                    "B_max_eval", "loop_orientation_sign",
                    "chi0_neg_weight_ratio",
                ],
                columns="source",
                values="chi_strict_real",
                aggfunc="first",
            ).reset_index()

            compact.to_csv(OUTDIR / f"selected_Bmax_lt_{tag}_compact.csv", index=False)


def main():
    all_rows = []
    errors = []

    inventory = []
    for root_label, root_str in ROOTS:
        case_roots = find_case_roots(root_str)

        cases = []
        for rd in case_roots:
            snaps_all = find_snapshots(rd)
            snaps_light = select_light_snapshots(snaps_all)
            cases.append({
                "case_name": case_display_name(rd),
                "run_dir": str(rd),
                "n_snapshots_all": len(snaps_all),
                "n_snapshots_used": len(snaps_light),
                "snapshots_used": [str(s) for s in snaps_light],
            })

        inventory.append({
            "root_label": root_label,
            "root_path": root_str,
            "n_cases": len(case_roots),
            "cases": cases,
        })

    (OUTDIR / "inventory.json").write_text(json.dumps(inventory, indent=2))

    print("=== Inventory ===", flush=True)
    for item in inventory:
        nused = sum(c["n_snapshots_used"] for c in item["cases"])
        print(
            f"{item['root_label']:20s} "
            f"{item['n_cases']:3d} cases, {nused:4d} selected snapshots  {item['root_path']}",
            flush=True,
        )

    for root_label, root_str in ROOTS:
        for rd in find_case_roots(root_str):
            snaps_all = find_snapshots(rd)
            snaps = select_light_snapshots(snaps_all)

            if len(snaps) == 0:
                errors.append({
                    "root_label": root_label,
                    "run_dir": str(rd),
                    "error": "no snapshots found",
                })
                write_outputs(all_rows, errors)
                continue

            for s in snaps:
                try:
                    rows = compute_snapshot(root_label, root_str, rd, s)
                    all_rows.extend(rows)
                    write_outputs(all_rows, errors)
                except Exception as e:
                    print(f"[ERROR] {root_label} {rd} {s}: {e}", flush=True)
                    errors.append({
                        "root_label": root_label,
                        "run_dir": str(rd),
                        "snapshot_dir": str(s),
                        "error": repr(e),
                        "traceback": traceback.format_exc(),
                    })
                    write_outputs(all_rows, errors)

    write_outputs(all_rows, errors)

    df = pd.DataFrame(all_rows)
    print("\n=== Done ===", flush=True)
    print(f"rows: {len(df)}", flush=True)
    print(f"errors: {len(errors)}", flush=True)
    print(f"outdir: {OUTDIR.resolve()}", flush=True)

    if len(df):
        print("\n=== Loop orientation counts ===", flush=True)
        print(df["loop_orientation_sign"].value_counts().to_string(), flush=True)

        print("\n=== Normal-state validity counts ===", flush=True)
        print(df["normal_state_valid"].value_counts().to_string(), flush=True)

        p = OUTDIR / "selected_Bmax_lt_0p95_compact.csv"
        print("\n=== Selected Bmax < 0.95 compact preview ===", flush=True)
        if p.exists():
            print(pd.read_csv(p).to_string(index=False), flush=True)
        else:
            print("No Bmax<0.95 selected data.", flush=True)

if __name__ == "__main__":
    main()
