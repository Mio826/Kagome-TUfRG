#!/usr/bin/env python3
from pathlib import Path
import json
import traceback
import numpy as np
import pandas as pd

import TUfRG_instability as inst

# Reuse D6a precritical utilities.
from calc_fig6a_d6a_corrected_strict_precritical import (
    ROOTS,
    strict_chi_for_eta,
    parse_params,
    find_case_roots,
    find_snapshots,
    select_light_snapshots,
    case_display_name,
    snapshot_label,
    snapshot_iter,
)

# Reuse the already audited D6b source.
from calc_fig6b_d6b_corrected_strict_final_light import (
    make_d6b_sources_on_O,
    Q_LABELS,
    qmap_get,
)

OUTDIR = Path("fig6b_d6b_corrected_strict_precritical")
OUTDIR.mkdir(parents=True, exist_ok=True)


def compute_snapshot(root_label, root_path, run_dir, snapshot_dir):
    case_name = case_display_name(run_dir)
    snap_lab = snapshot_label(snapshot_dir, run_dir)
    snap_it = snapshot_iter(snapshot_dir, run_dir)
    params = parse_params(case_name)

    print(f"[snapshot] {root_label} :: {case_name} :: {snap_lab}", flush=True)

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

            diag = strict_chi_for_eta(eta, K_saved, L_saved)

            rows.append({
                "root_label": root_label,
                "root_path": str(root_path),
                "case_name": case_name,
                "run_dir": str(run_dir),
                "snapshot_label": snap_lab,
                "snapshot_iter": int(snap_it),
                "snapshot_dir": str(snapshot_dir),
                "q_label": qlab,
                "iq": int(iq),
                "sector": spec["sector"],
                "source": "D6b_fig6b_corrected",
                **source_diag,
                **diag,
                **params,
            })

    return rows


def select_last_precritical_sum(df, threshold):
    """
    Sum chi over Gamma,M1,M2,M3 only if all four q rows are available
    and all have B_max_eval < threshold.
    Then select the latest snapshot per case/sector.
    """
    if len(df) == 0:
        return df

    keys0 = [
        "root_label", "case_name", "U", "V",
        "sector", "snapshot_label", "snapshot_iter"
    ]

    agg = df.groupby(keys0).agg(
        chi_D6b_sum=("chi_strict_real", "sum"),
        chi_D6b_imag_sum=("chi_strict_imag", "sum"),
        B_max_eval_max=("B_max_eval", "max"),
        min_1_minus_B_min=("min_1_minus_B", "min"),
        direct_spectral_diff_max=("direct_spectral_diff", "max"),
        loop_orientation_sign=("loop_orientation_sign", "first"),
        chi0_neg_weight_ratio_max=("chi0_neg_weight_ratio", "max"),
        n_q=("q_label", "nunique"),
    ).reset_index()

    agg = agg[(agg["n_q"] == 4) & (agg["B_max_eval_max"] < threshold)].copy()

    if len(agg) == 0:
        return agg

    keys1 = ["root_label", "case_name", "U", "V", "sector"]
    idx = agg.groupby(keys1)["snapshot_iter"].idxmax()

    return agg.loc[idx].sort_values(keys1).reset_index(drop=True)


def write_outputs(all_rows, errors):
    df = pd.DataFrame(all_rows)
    df.to_csv(OUTDIR / "fig6b_d6b_corrected_strict_precritical_all.csv", index=False)
    (OUTDIR / "errors.json").write_text(json.dumps(errors, indent=2))

    if len(df) == 0:
        return

    for thr, tag in [(0.90, "0p90"), (0.95, "0p95"), (1.00, "1p00")]:
        sel = select_last_precritical_sum(df, thr)
        sel.to_csv(OUTDIR / f"selected_sum_Bmax_lt_{tag}.csv", index=False)

        if len(sel):
            compact = sel.pivot_table(
                index=[
                    "root_label", "case_name", "U", "V",
                    "snapshot_label", "snapshot_iter",
                    "B_max_eval_max",
                    "loop_orientation_sign",
                    "chi0_neg_weight_ratio_max",
                ],
                columns="sector",
                values="chi_D6b_sum",
                aggfunc="first",
            ).reset_index()

            compact.to_csv(
                OUTDIR / f"selected_sum_Bmax_lt_{tag}_compact.csv",
                index=False,
            )


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
        print("\n=== Source q weights ===", flush=True)
        cols = [
            "source_norm2_Gamma",
            "source_norm2_M1",
            "source_norm2_M2",
            "source_norm2_M3",
        ]
        print(df[cols].drop_duplicates().to_string(index=False), flush=True)

        print("\n=== Sector counts ===", flush=True)
        print(df["sector"].value_counts().to_string(), flush=True)

        print("\n=== q counts ===", flush=True)
        print(df["q_label"].value_counts().to_string(), flush=True)

        print("\n=== Loop orientation counts ===", flush=True)
        print(df["loop_orientation_sign"].value_counts().to_string(), flush=True)

        print("\n=== Normal-state validity counts ===", flush=True)
        print(df["normal_state_valid"].value_counts().to_string(), flush=True)

        p = OUTDIR / "selected_sum_Bmax_lt_0p95_compact.csv"
        print("\n=== Selected summed D6b, Bmax < 0.95 compact preview ===", flush=True)
        if p.exists():
            print(pd.read_csv(p).to_string(index=False), flush=True)
        else:
            print("No Bmax<0.95 selected data.", flush=True)


if __name__ == "__main__":
    main()
