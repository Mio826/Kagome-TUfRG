#!/usr/bin/env python3
from pathlib import Path
import json
import traceback
import numpy as np
import pandas as pd

import TUfRG_instability as inst

# Reuse the already-tested D6a source and path utilities.
from calc_fig6a_d6a_corrected_strict_precritical import (
    ROOTS,
    parse_params,
    find_case_roots,
    case_display_name,
    make_d6a_sources_on_O,
)

OUTDIR = Path("fig6a_d6a_final_kappa_tau_kappa")
OUTDIR.mkdir(parents=True, exist_ok=True)

Q_LABELS = ["M1", "M2", "M3"]
SECTORS = ["ph-rho", "ph-sx", "ph-sy", "ph-sz"]


def hermitize(M):
    M = np.asarray(M, dtype=np.complex128)
    return 0.5 * (M + M.conj().T)


def psd_project(M, rel_tol=1e-10):
    """
    Project a Hermitian matrix to its positive semidefinite part.
    This is only for small numerical negative noise.
    """
    M = hermitize(M)
    evals, evecs = np.linalg.eigh(M)

    scale = max(float(np.max(np.abs(evals))), 1.0)
    evals2 = evals.copy()
    evals2[evals2 < rel_tol * scale] = np.maximum(evals2[evals2 < rel_tol * scale], 0.0)

    # If there are serious negative eigenvalues, keep diagnostic but still clip for stability.
    evals_clip = np.clip(evals, 0.0, None)
    Mpsd = (evecs * evals_clip) @ evecs.conj().T

    neg_weight = float(np.sum(np.abs(evals[evals < 0.0])) / (np.sum(np.abs(evals)) + 1e-300))

    return hermitize(Mpsd), {
        "raw_min_eval": float(np.min(evals)),
        "raw_max_eval": float(np.max(evals)),
        "raw_neg_weight_ratio": neg_weight,
    }


def choose_kappa_tau(K_saved, L_saved):
    """
    Convert saved kernel/loop convention into:
        kappa = chi0, positive semidefinite
        tau   = Gamma

    We choose sign s = +/-1 such that:
        kappa = s * L_saved
    is as positive-semidefinite as possible.

    Then choose:
        tau = K_saved / s
    so that:
        tau @ kappa = K_saved @ L_saved.
    """
    K_saved = hermitize(K_saved)
    L_saved = hermitize(L_saved)

    candidates = []

    for s in [+1.0, -1.0]:
        raw = hermitize(s * L_saved)
        evals = np.linalg.eigvalsh(raw)
        neg_weight = float(np.sum(np.abs(evals[evals < 0.0])) / (np.sum(np.abs(evals)) + 1e-300))
        candidates.append((neg_weight, s, raw, evals))

    candidates.sort(key=lambda x: x[0])
    neg_weight, s, kappa_raw, evals = candidates[0]

    kappa, psd_diag = psd_project(kappa_raw)

    tau = hermitize(K_saved / s)

    diag = {
        "loop_orientation_sign": int(s),
        "kappa_raw_min_eval": float(np.min(evals)),
        "kappa_raw_max_eval": float(np.max(evals)),
        "kappa_raw_neg_weight_ratio": float(neg_weight),
        "kappa_psd_raw_min_eval": psd_diag["raw_min_eval"],
        "kappa_psd_raw_max_eval": psd_diag["raw_max_eval"],
        "kappa_psd_raw_neg_weight_ratio": psd_diag["raw_neg_weight_ratio"],
    }

    return kappa, tau, diag


def sqrt_psd(M):
    M = hermitize(M)
    evals, evecs = np.linalg.eigh(M)
    evals_clip = np.clip(evals, 0.0, None)
    S = (evecs * np.sqrt(evals_clip)) @ evecs.conj().T
    return hermitize(S), evals


def kappa_tau_kappa_for_eta(eta, K_saved, L_saved):
    """
    Compute:
        chi0 = eta^dag kappa eta
        chi2 = eta^dag kappa tau kappa eta
        chi02 = chi0 + chi2

    Also compute A = sqrt(kappa) tau sqrt(kappa) diagnostics.
    """
    eta = np.asarray(eta, dtype=np.complex128).reshape(-1)

    kappa, tau, diag0 = choose_kappa_tau(K_saved, L_saved)

    ket0 = kappa @ eta

    chi0 = np.vdot(eta, ket0)
    chi2 = np.vdot(eta, kappa @ (tau @ (kappa @ eta)))
    chi02 = chi0 + chi2

    # Equivalent symmetric expression for chi2:
    # b = sqrt(kappa) eta, A = sqrt(kappa) tau sqrt(kappa), chi2 = b^dag A b.
    sqrt_kappa, kap_evals = sqrt_psd(kappa)
    A = hermitize(sqrt_kappa @ tau @ sqrt_kappa)
    A_evals = np.linalg.eigvalsh(A)

    b = sqrt_kappa @ eta
    chi0_sym = np.vdot(b, b)
    chi2_sym = np.vdot(b, A @ b)

    diag = {
        **diag0,
        "kappa_min_eval": float(np.min(kap_evals)),
        "kappa_max_eval": float(np.max(kap_evals)),
        "A_min_eval": float(np.min(A_evals)),
        "A_max_eval": float(np.max(A_evals)),
        "min_1_minus_A": float(1.0 - np.max(A_evals)),
        "postcritical_flag_Amax_gt_1": bool(np.max(A_evals) > 1.0),

        "chi0_real": float(np.real(chi0)),
        "chi0_imag": float(np.imag(chi0)),
        "chi2_real": float(np.real(chi2)),
        "chi2_imag": float(np.imag(chi2)),
        "chi0plus2_real": float(np.real(chi02)),
        "chi0plus2_imag": float(np.imag(chi02)),

        "chi0_sym_real": float(np.real(chi0_sym)),
        "chi2_sym_real": float(np.real(chi2_sym)),
        "chi0_direct_sym_diff": float(abs(chi0 - chi0_sym)),
        "chi2_direct_sym_diff": float(abs(chi2 - chi2_sym)),
    }

    return diag


def qmap_get(qmap, label):
    if label in qmap:
        return qmap[label]
    alt = label + "_nearest"
    if alt in qmap:
        return qmap[alt]
    raise KeyError(f"Cannot find q label {label}; qmap={qmap}")


def preferred_final_snapshot(run_dir: Path):
    """
    Light final-only mode.

    Priority:
      1. snapshots/final
      2. root O.dvg
      3. latest iter_XXXX snapshot
    """
    final_dir = run_dir / "snapshots" / "final"
    if (final_dir / "O.dvg").exists():
        return final_dir, "final"

    if (run_dir / "O.dvg").exists():
        return run_dir, "root_final"

    snap_root = run_dir / "snapshots"
    if snap_root.exists():
        iters = []
        for p in snap_root.iterdir():
            if p.is_dir() and p.name.startswith("iter_") and (p / "O.dvg").exists():
                try:
                    it = int(p.name.split("_")[1])
                except Exception:
                    it = -1
                iters.append((it, p))
        if iters:
            iters.sort()
            return iters[-1][1], iters[-1][1].name

    return None, None


def compute_one_case(root_label, root_path, run_dir, snapshot_dir, snap_label):
    case_name = case_display_name(run_dir)
    params = parse_params(case_name)

    print(f"[case] {root_label} :: {case_name} :: {snap_label}", flush=True)

    data = inst.load_case(snapshot_dir)
    inst.assert_raw_channel_compatibility(data)

    qmap = inst.q_labels_to_indices(data, Q_LABELS)
    etas_by_q, source_diag = make_d6a_sources_on_O(data.O)

    rows = []

    for qlab in Q_LABELS:
        iq = qmap_get(qmap, qlab)
        eta = etas_by_q[qlab]

        kernels = inst.build_explicit_kernels(data, iq)

        for spec in kernels:
            if spec["family"] != "ph":
                continue
            if spec["sector"] not in SECTORS:
                continue

            K_saved = np.asarray(spec["K"], dtype=np.complex128)
            L_saved = np.asarray(spec["L"], dtype=np.complex128)

            diag = kappa_tau_kappa_for_eta(eta, K_saved, L_saved)

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
                "source": "D6a_fig6a_corrected_final_kappa_tau_kappa",
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
        "sector", "snapshot_label"
    ]

    out = df.groupby(keys).agg(
        chi0_D6a_sum=("chi0_real", "sum"),
        chi2_D6a_sum=("chi2_real", "sum"),
        chi0plus2_D6a_sum=("chi0plus2_real", "sum"),
        chi0_imag_abs_max=("chi0_imag", lambda x: float(np.max(np.abs(x)))),
        chi2_imag_abs_max=("chi2_imag", lambda x: float(np.max(np.abs(x)))),
        chi0plus2_imag_abs_max=("chi0plus2_imag", lambda x: float(np.max(np.abs(x)))),
        A_max_eval_max=("A_max_eval", "max"),
        min_1_minus_A_min=("min_1_minus_A", "min"),
        postcritical_any=("postcritical_flag_Amax_gt_1", "max"),
        loop_orientation_sign=("loop_orientation_sign", "first"),
        kappa_neg_weight_ratio_max=("kappa_raw_neg_weight_ratio", "max"),
        direct_sym_diff_max=("chi2_direct_sym_diff", "max"),
        n_q=("q_label", "nunique"),
    ).reset_index()

    return out.sort_values(keys).reset_index(drop=True)


def write_outputs(rows, errors):
    df = pd.DataFrame(rows)
    df.to_csv(OUTDIR / "fig6a_d6a_final_kappa_tau_kappa_all.csv", index=False)

    summed = make_summed(df)
    summed.to_csv(OUTDIR / "fig6a_d6a_final_kappa_tau_kappa_summed.csv", index=False)

    if len(summed):
        compact = summed.pivot_table(
            index=[
                "root_label", "case_name", "U", "V",
                "snapshot_label", "A_max_eval_max",
                "postcritical_any",
                "loop_orientation_sign",
                "kappa_neg_weight_ratio_max",
            ],
            columns="sector",
            values=["chi0_D6a_sum", "chi2_D6a_sum", "chi0plus2_D6a_sum"],
            aggfunc="first",
        )

        compact.columns = [
            f"{a}_{b}" for a, b in compact.columns.to_flat_index()
        ]
        compact = compact.reset_index()
        compact.to_csv(OUTDIR / "fig6a_d6a_final_kappa_tau_kappa_compact.csv", index=False)

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
            snap, lab = preferred_final_snapshot(rd)
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
            snap, lab = preferred_final_snapshot(rd)

            if snap is None:
                errors.append({
                    "root_label": root_label,
                    "run_dir": str(rd),
                    "error": "no usable O.dvg found",
                })
                write_outputs(rows, errors)
                continue

            try:
                new_rows = compute_one_case(root_label, root_str, rd, snap, lab)
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
                "source_norm2_M1",
                "source_norm2_M2",
                "source_norm2_M3",
            ]].drop_duplicates().to_string(index=False),
            flush=True,
        )

        print("\n=== q counts ===", flush=True)
        print(df["q_label"].value_counts().to_string(), flush=True)

        print("\n=== sector counts ===", flush=True)
        print(df["sector"].value_counts().to_string(), flush=True)

        print("\n=== compact preview ===", flush=True)
        p = OUTDIR / "fig6a_d6a_final_kappa_tau_kappa_compact.csv"
        if p.exists():
            print(pd.read_csv(p).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
