#!/usr/bin/env python3
# coding: utf-8
"""
Inspect q=0 and Q=M TUFRG instabilities from projected divERGe O.dvg/T.dvg output.

This file intentionally does NOT rely only on O.S_charge/O.S_mag/O.S_sc,
because those divERGe fields are q=0-only linearized-gap outputs. For M-point
CBO/SBO, this script uses the q-resolved P/C/D channel matrices and loops:
The input O.dvg must be generated with cfg.tu_channel_calc_project=-1.
Then the saved channels are interpreted as:
  mag(q)    = - C_saved(q)
  charge(q) = D_saved(q) = 2 D(q) - C(q)
  sc(q)     = P_saved(q)
The script reports raw and loop-dressed strengths at Gamma and the three M points.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List

import numpy as np
import diverge.output as ro

from kagome_prr_mu0_projected_common import (
    high_symmetry_q_indices,
    physical_kernel,
    qmesh_from_output,
    summarize_q_channel,
    top_svd,
    uniform_projected_svd,
    vhs_sanity_report,
    write_csv,
)


def inspect_case(case_dir: Path, *, n_top: int = 8) -> List[dict]:
    Ofile = case_dir / "O.dvg"
    Tfile = case_dir / "T.dvg"
    if not Ofile.exists() or not Tfile.exists():
        raise FileNotFoundError(f"Missing O.dvg/T.dvg in {case_dir}")

    O = ro.read(str(Ofile))
    T = ro.read(str(Tfile))
    qmesh = qmesh_from_output(O, T)
    qidx = high_symmetry_q_indices(qmesh)

    print("\n" + "#" * 100)
    print(f"CASE {case_dir.name}")
    print("#" * 100)
    print(f"SU2={getattr(O, 'SU2', None)}, n_orb={getattr(O, 'n_orb', None)}, n_orbff={getattr(O, 'n_orbff', None)}")
    print(f"nk={getattr(O, 'nk', None)}, qmesh size={qmesh.shape[0]}")
    print("High-symmetry q indices, reduced reciprocal coordinates:")
    for name in ["Gamma", "M1", "M2", "M3"]:
        iq = qidx[name]
        print(f"  {name:6s}: iq={iq:4d}, q={qmesh[iq]}")

    print("\nBuilt-in divERGe q=0 lingap S values, if present:")
    for ch in ["mag", "charge", "sc"]:
        S = getattr(O, f"S_{ch}", None)
        print(f"  S_{ch:6s} = {None if S is None else np.asarray(S)[:n_top]}")

    rows: List[dict] = []
    for qlabel in ["Gamma", "M1", "M2", "M3"]:
        iq = qidx[qlabel]
        qcoord = qmesh[iq]
        print("\n" + "=" * 90)
        print(f"q = {qlabel}, iq={iq}, coord={qcoord}")
        print("=" * 90)
        for ch in ["mag", "charge", "sc"]:
            row = summarize_q_channel(O, iq, qcoord, ch, n=n_top)
            row["case"] = case_dir.name
            row["q_label"] = qlabel
            rows.append(row)
            print(f"\n[{ch}] {row['kernel_label']}")
            print(f"  raw hermitian eig min/max: {row['raw_herm_min']:+.8e}, {row['raw_herm_max']:+.8e}")
            print(f"  raw SVD top             : {row['raw_svd_0']:.8e}")
            print(f"  loop SVD top, K@L       : {row['loop_svd_KL_0']:.8e}")
            print(f"  loop SVD top, L@K       : {row['loop_svd_LK_0']:.8e}")
            if row.get("charge_q0_uniform_projected_svd_top"):
                print(f"  charge q=0 projected top: {row['charge_q0_uniform_projected_svd_top']}")

    # Calibration: compare q=0 K@L SVD against divERGe S_*.
    print("\n" + "-" * 90)
    print("q=0 rough calibration against divERGe built-in S_* (K@L convention; projected saved channels):")
    iq0 = qidx["Gamma"]
    for ch in ["mag", "charge", "sc"]:
        S_pkg = getattr(O, f"S_{ch}", None)
        if S_pkg is None:
            continue
        K, L, _ = physical_kernel(O, ch, iq0)
        S_ours = top_svd(K @ L, n=min(n_top, len(np.asarray(S_pkg))))
        S_pkg = np.asarray(S_pkg)[:len(S_ours)]
        denom = np.maximum(np.abs(S_pkg), 1e-30)
        rel = np.max(np.abs(S_ours - S_pkg) / denom) if S_pkg.size else np.nan
        print(f"  {ch:6s}: package={S_pkg}")
        print(f"          ours   ={S_ours}")
        print(f"          max relative mismatch={rel:.3e}")
    return rows


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("root", nargs="?", default="outputs/kagome_prr_mu0_projected_small", help="root output folder or one case folder")
    p.add_argument("--csv", default=None, help="optional CSV output path")
    p.add_argument("--n-top", type=int, default=8)
    p.add_argument("--vhs-check", action="store_true", help="print independent NN kagome mu=0 VHS/filling check")
    args = p.parse_args()

    if args.vhs_check:
        print("=== independent NN kagome mu=0 check, hop_amp=-1 ===")
        rep = vhs_sanity_report(hop_amp=-1.0, mu=0.0, ngrid=120)
        print(rep)

    root = Path(args.root)
    if (root / "O.dvg").exists():
        case_dirs = [root]
    else:
        case_dirs = sorted([p for p in root.iterdir() if p.is_dir() and (p / "O.dvg").exists()])
    if not case_dirs:
        raise FileNotFoundError(f"No case directories containing O.dvg found under {root}")

    all_rows: List[dict] = []
    for d in case_dirs:
        all_rows.extend(inspect_case(d, n_top=args.n_top))

    csv_path = args.csv or str(root / "diagnosis_qresolved_summary.csv")
    write_csv(csv_path, all_rows)
    print(f"\nWrote CSV summary: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
