#!/usr/bin/env python3
from pathlib import Path
import json
import traceback
import numpy as np
import pandas as pd

import TUfRG_instability as inst

# Reuse existing audited D6b source and case discovery helpers.
from calc_fig6a_d6a_corrected_strict_precritical import (
    ROOTS,
    parse_params,
    find_case_roots,
    case_display_name,
)

from calc_fig6b_d6b_corrected_strict_final_light import (
    make_d6b_sources_on_O,
    Q_LABELS,
    qmap_get,
    preferred_snapshot,
)

OUTDIR = Path("fig6b_d6b_second_ladder_final")
OUTDIR.mkdir(parents=True, exist_ok=True)


def hermitize(M):
    M = np.asarray(M, dtype=np.complex128)
    return 0.5 * (M + M.conj().T)


def psd_project_and_sqrt(M, eps=1e-12):
    """
    Return PSD-projected M and sqrt(M).

    Small negative eigenvalues are clipped to zero. Large negative spectral
    weight is recorded by diagnostics upstream.
    """
    M = hermitize(M)
    evals, evecs = np.linalg.eigh(M)

    total_abs = float(np.sum(np.abs(evals))) + 1e-300
    neg_weight = float(np.sum(np.abs(evals[evals < 0.0])) / total_abs)

    evals_clip = np.clip(evals, 0.0, None)
    sqrtM = (evecs * np.sqrt(evals_clip)) @ evecs.conj().T
    Mpsd = (evecs * evals_clip) @ evecs.conj().T

    diag = {
        "chi0_min_eval_raw": float(evals.min()),
        "chi0_max_eval_raw": float(evals.max()),
        "chi0_neg_weight_ratio": float(neg_weight),
        "chi0_rank_psd": int(np.count_nonzero(evals_clip > eps * max(float(evals_clip.max()), 1.0))),
    }

    return Mpsd, sqrtM, diag


def choose_chi0_and_tau(K_saved, L_saved):
    """
    divERGe saved phloop sign can be opposite to the physical positive bubble.
    We choose sign so chi0 = sign * L_saved is PSD.

    To preserve the loop-dressed instability convention:
        tau @ chi0 = K_saved @ L_saved
    choose:
        tau = sign * K_saved
        chi0 = sign * L_saved

    because:
        tau chi0 = (sign K)(sign L) = K L.
    """
    K_saved = hermitize(K_saved)
    L_saved = hermitize(L_saved)

    candidates = []
    for sign in [+1, -1]:
        chi0_raw = hermitize(sign * L_saved)
        evals = np.linalg.eigvalsh(chi0_raw)
        total_abs = float(np.sum(np.abs(evals))) + 1e-300
        neg_weight = float(np.sum(np.abs(evals[evals < 0.0])) / total_abs)
        candidates.append((neg_weight, sign, evals.min(), evals.max()))

    candidates.sort(key=lambda x: x[0])
    neg_weight, sign, min_eval, max_eval = candidates[0]

    chi0 = hermitize(sign * L_saved)
    tau = hermitize(sign * K_saved)

    return chi0, tau, {
        "loop_orientation_sign": int(sign),
        "chosen_neg_weight_ratio": float(neg_weight),
        "chosen_chi0_min_eval": float(min_eval),
        "chosen_chi0_max_eval": float(max_eval),
    }


def second_ladder_for_eta(eta, K_saved, L_saved):
    """
    Compute truncated ladder response:

        chi0 = eta^dag chi0 eta
        chi2 = eta^dag chi0 tau chi0 eta
        chi_0_plus_2 = chi0 + chi2

    Symmetric stable implementation:

        B = sqrt(chi0) tau sqrt(chi0)
        g = sqrt(chi0) eta

        chi0 = g^dag g
        chi2 = g^dag B g

    No inverse is used, so post-critical B_max > 1 is allowed.
    """
    eta = np.asarray(eta, dtype=np.complex128).reshape(-1)

    chi0_raw, tau, diag0 = choose_chi0_and_tau(K_saved, L_saved)
    chi0_psd, sqrt_chi0, diag_psd = psd_project_and_sqrt(chi0_raw)

    B = hermitize(sqrt_chi0 @ tau @ sqrt_chi0)
    Bevals = np.linalg.eigvalsh(B)

    g = sqrt_chi0 @ eta

    chi0_val = np.vdot(g, g)
    chi2_val = np.vdot(g, B @ g)
    chi02_val = chi0_val + chi2_val

    # Direct version as a cross-check.
    direct_chi0 = np.vdot(eta, chi0_psd @ eta)
    direct_chi2 = np.vdot(eta, chi0_psd @ tau @ chi0_psd @ eta)

    diag = {}
    diag.update(diag0)
    diag.update(diag_psd)

    diag.update({
        "B_min_eval": float(Bevals.min()),
        "B_max_eval": float(Bevals.max()),
        "B_abs_max_eval": float(np.max(np.abs(Bevals))),
        "normal_state_valid_Bmax_lt_1": bool(Bevals.max() < 1.0),

        "chi0_real": float(np.real(chi0_val)),
        "chi0_imag": float(np.imag(chi0_val)),
        "chi2_real": float(np.real(chi2_val)),
        "chi2_imag": float(np.imag(chi2_val)),
        "chi0_plus_chi2_real": float(np.real(chi02_val)),
        "chi0_plus_chi2_imag": float(np.imag(chi02_val)),

        "direct_chi0_diff": float(abs(direct_chi0 - chi0_val)),
        "direct_chi2_diff": float(abs(direct_chi2 - chi2_val)),
    })

    return diag


def compute_one(root_label, root_path, run_dir, snapshot_dir, snap_label):
    case_name = case_display_name(run_dir)
    params = parse_params(case_name)

    print(f"[case] {root_label} :: {case_name} :: {snap_label}", flush=True)

    data = inst.load_case(snapshot_dir)
    inst.assert_raw_channel_compatibility(data)

    qmap = inst.q_labels_to_indices(data, Q_LABELS)
    etas_by_q, source_diag = make_d6b_sources_on_O(data.O)

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
                "source": "D6b_fig6b_corrected_second_ladder_final",
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
        chi0_sum=("chi0_real", "sum"),
        chi2_sum=("chi2_real", "sum"),
        chi0_plus_chi2_sum=("chi0_plus_chi2_real", "sum"),
        chi0_imag_abs_max=("chi0_imag", lambda x: float(np.max(np.abs(x)))),
        chi2_imag_abs_max=("chi2_imag", lambda x: float(np.max(np.abs(x)))),
        chi0_plus_chi2_imag_abs_max=("chi0_plus_chi2_imag", lambda x: float(np.max(np.abs(x)))),
        B_max_eval_max=("B_max_eval", "max"),
        B_abs_max_eval_max=("B_abs_max_eval", "max"),
        n_q=("q_label", "nunique"),
        loop_orientation_sign=("loop_orientation_sign", "first"),
        chi0_neg_weight_ratio_max=("chi0_neg_weight_ratio", "max"),
        direct_chi0_diff_max=("direct_chi0_diff", "max"),
        direct_chi2_diff_max=("direct_chi2_diff", "max"),
    ).reset_index()

    return out.sort_values(keys).reset_index(drop=True)


def write_outputs(rows, errors):
    df = pd.DataFrame(rows)
    df.to_csv(OUTDIR / "fig6b_d6b_second_ladder_final_all.csv", index=False)

    summed = make_summed(df)
    summed.to_csv(OUTDIR / "fig6b_d6b_second_ladder_final_summed.csv", index=False)

    if len(summed):
        compact = summed.pivot_table(
            index=[
                "root_label", "case_name", "U", "V",
                "snapshot_label", "B_max_eval_max",
                "B_abs_max_eval_max",
                "loop_orientation_sign",
                "chi0_neg_weight_ratio_max",
            ],
            columns="sector",
            values="chi0_plus_chi2_sum",
            aggfunc="first",
        ).reset_index()

        compact.to_csv(OUTDIR / "fig6b_d6b_second_ladder_final_compact_chi0_plus_chi2.csv", index=False)

        compact_chi2 = summed.pivot_table(
            index=[
                "root_label", "case_name", "U", "V",
                "snapshot_label", "B_max_eval_max",
                "B_abs_max_eval_max",
                "loop_orientation_sign",
                "chi0_neg_weight_ratio_max",
            ],
            columns="sector",
            values="chi2_sum",
            aggfunc="first",
        ).reset_index()

        compact_chi2.to_csv(OUTDIR / "fig6b_d6b_second_ladder_final_compact_chi2_only.csv", index=False)

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
            snap, lab = preferred_snapshot(rd)
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
            snap, lab = preferred_snapshot(rd)

            if snap is None:
                errors.append({
                    "root_label": root_label,
                    "run_dir": str(rd),
                    "error": "no usable O.dvg found",
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

        print("\n=== Bmax summary ===", flush=True)
        print(df.groupby(["root_label"])["B_max_eval"].agg(["min", "median", "max"]).to_string(), flush=True)

        print("\n=== Compact chi0+chi2 preview ===", flush=True)
        p = OUTDIR / "fig6b_d6b_second_ladder_final_compact_chi0_plus_chi2.csv"
        if p.exists():
            print(pd.read_csv(p).to_string(index=False), flush=True)

        print("\n=== Compact chi2-only preview ===", flush=True)
        p2 = OUTDIR / "fig6b_d6b_second_ladder_final_compact_chi2_only.csv"
        if p2.exists():
            print(pd.read_csv(p2).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
