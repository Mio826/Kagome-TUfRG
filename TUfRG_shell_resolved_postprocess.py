#!/usr/bin/env python3
import argparse
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd


SQRT3 = math.sqrt(3.0)

# Kagome convention with NN distance = 1.
# Bravais vectors:
#   a1 = (2, 0)
#   a2 = (1, sqrt(3))
# Sublattice positions:
#   tau0 = (0, 0)
#   tau1 = (1, 0)
#   tau2 = (1/2, sqrt(3)/2)
#
# This convention is consistent with:
#   same sublattice, R=(1,0),(0,1),(1,-1) -> distance 2
#   nearest inter-sublattice bonds -> distance 1
A1 = np.array([2.0, 0.0])
A2 = np.array([1.0, SQRT3])
A3 = np.array([0.0, 0.0])

TAU = {
    0: np.array([0.0, 0.0]),
    1: np.array([1.0, 0.0]),
    2: np.array([0.5, SQRT3 / 2.0]),
}


def real_displacement(row):
    r0 = int(row["R0"]) if "R0" in row and pd.notna(row["R0"]) else 0
    r1 = int(row["R1"]) if "R1" in row and pd.notna(row["R1"]) else 0
    r2 = int(row["R2"]) if "R2" in row and pd.notna(row["R2"]) else 0

    ofrom = int(row["ofrom"])
    oto = int(row["oto"])

    # The sign convention for tau shift may be reversed depending on how
    # ofrom/oto is defined, but the distance shell is invariant under reversal
    # for the purpose of classifying shell length.
    disp = r0 * A1 + r1 * A2 + r2 * A3 + TAU[ofrom] - TAU[oto]
    return disp


def classify_shell(dist, tol=1e-6):
    if abs(dist - 0.0) < tol:
        return "onsite_0"
    if abs(dist - 1.0) < tol:
        return "nn_1"
    if abs(dist - SQRT3) < tol:
        return "second_neighbor_sqrt3"
    if abs(dist - 2.0) < tol:
        return "cross_hexagon_2"
    return "other"


def process_one_top_components(path: Path, tol: float = 1e-6):
    df = pd.read_csv(path)

    required = {"mode_key", "q_label", "family", "sector", "mode", "R0", "R1", "R2", "ofrom", "oto", "value_abs"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing required columns {sorted(missing)}")

    dxs, dys, dists, shells, subrels, amp2s = [], [], [], [], [], []

    for _, row in df.iterrows():
        disp = real_displacement(row)
        dist = float(np.linalg.norm(disp))
        shell = classify_shell(dist, tol=tol)

        ofrom = int(row["ofrom"])
        oto = int(row["oto"])
        subrel = "same_sub" if ofrom == oto else "inter_sub"

        amp2 = float(row["value_abs"]) ** 2

        dxs.append(disp[0])
        dys.append(disp[1])
        dists.append(dist)
        shells.append(shell)
        subrels.append(subrel)
        amp2s.append(amp2)

    out = df.copy()
    out["dx"] = dxs
    out["dy"] = dys
    out["distance"] = dists
    out["shell"] = shells
    out["sub_relation"] = subrels
    out["top_amp2"] = amp2s

    # Summary over the saved top components.
    # This is NOT full eigenvector weight unless the top component list is complete.
    group_cols = ["mode_key", "q_label", "family", "sector", "mode"]

    rows = []
    for key, g in out.groupby(group_cols, dropna=False):
        base = dict(zip(group_cols, key))
        total_top_amp2 = float(g["top_amp2"].sum())

        row = dict(base)
        row["n_top_components"] = int(len(g))
        row["top_amp2_sum"] = total_top_amp2

        for shell in ["onsite_0", "nn_1", "second_neighbor_sqrt3", "cross_hexagon_2", "other"]:
            gg = g[g["shell"] == shell]
            s = float(gg["top_amp2"].sum())
            n = int(len(gg))
            row[f"{shell}_top_amp2"] = s
            row[f"{shell}_top_frac"] = s / total_top_amp2 if total_top_amp2 > 0 else np.nan
            row[f"{shell}_ncomp"] = n
            row[f"{shell}_avg_amp2_per_component"] = s / n if n > 0 else 0.0

        for rel in ["same_sub", "inter_sub"]:
            gg = g[g["sub_relation"] == rel]
            s = float(gg["top_amp2"].sum())
            n = int(len(gg))
            row[f"{rel}_top_amp2"] = s
            row[f"{rel}_top_frac"] = s / total_top_amp2 if total_top_amp2 > 0 else np.nan
            row[f"{rel}_ncomp"] = n
            row[f"{rel}_avg_amp2_per_component"] = s / n if n > 0 else 0.0

        # A compact label for dominant top-component shell.
        shell_fracs = {
            shell: row[f"{shell}_top_frac"]
            for shell in ["onsite_0", "nn_1", "second_neighbor_sqrt3", "cross_hexagon_2", "other"]
        }
        rel_fracs = {
            rel: row[f"{rel}_top_frac"]
            for rel in ["same_sub", "inter_sub"]
        }

        row["dominant_top_shell"] = max(shell_fracs, key=lambda k: -1 if pd.isna(shell_fracs[k]) else shell_fracs[k])
        row["dominant_top_sub_relation"] = max(rel_fracs, key=lambda k: -1 if pd.isna(rel_fracs[k]) else rel_fracs[k])

        rows.append(row)

    summary = pd.DataFrame(rows)

    out_path = path.with_name("shell_resolved_top_components.csv")
    sum_path = path.with_name("shell_resolved_top_summary.csv")

    out.to_csv(out_path, index=False)
    summary.to_csv(sum_path, index=False)

    return out_path, sum_path, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case_or_outdir", help="A case directory or an OUTDIR containing case directories.")
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--sc-only", action="store_true", help="Also write a compact SC-only aggregate.")
    args = ap.parse_args()

    root = Path(args.case_or_outdir).resolve()
    if not root.exists():
        raise SystemExit(f"Path does not exist: {root}")

    files = sorted(root.glob("**/formfactor_top_components.csv"))
    if not files:
        raise SystemExit(f"No formfactor_top_components.csv found under {root}")

    print(f"Found {len(files)} formfactor_top_components.csv files")

    all_summaries = []
    for i, path in enumerate(files, 1):
        try:
            out_path, sum_path, summary = process_one_top_components(path, tol=args.tol)

            # Add context columns from path.
            # Expected path:
            #   CASE/snapshots/ITER/formfactor/formfactor_top_components.csv
            parts = path.parts
            case = ""
            snapshot = ""
            if "snapshots" in parts:
                idx = parts.index("snapshots")
                if idx >= 1:
                    case = parts[idx - 1]
                if idx + 1 < len(parts):
                    snapshot = parts[idx + 1]

            summary.insert(0, "case", case)
            summary.insert(1, "snapshot", snapshot)
            summary.insert(2, "source_file", str(path))

            all_summaries.append(summary)

            print(f"[{i}/{len(files)}] OK {path}")
        except Exception as e:
            print(f"[{i}/{len(files)}] BAD {path}: {e}")

    if all_summaries:
        all_df = pd.concat(all_summaries, ignore_index=True)
        aggregate_path = root / "shell_resolved_all_modes_top_summary.csv"
        all_df.to_csv(aggregate_path, index=False)
        print(f"\nWrote aggregate: {aggregate_path}")

        if args.sc_only:
            sc = all_df[
                (all_df["family"].astype(str) == "pp")
                & (all_df["sector"].astype(str).str.contains("singlet", case=False, na=False))
            ].copy()
            sc_path = root / "shell_resolved_SC_top_summary.csv"
            sc.to_csv(sc_path, index=False)
            print(f"Wrote SC aggregate: {sc_path}")

            if not sc.empty:
                cols = [
                    "case", "snapshot", "q_label", "sector", "mode",
                    "dominant_top_shell", "dominant_top_sub_relation",
                    "onsite_0_top_frac", "nn_1_top_frac",
                    "second_neighbor_sqrt3_top_frac", "cross_hexagon_2_top_frac",
                    "same_sub_top_frac", "inter_sub_top_frac",
                    "n_top_components",
                ]
                print("\nSC compact preview:")
                print(sc[cols].tail(30).to_string(index=False))


if __name__ == "__main__":
    main()
