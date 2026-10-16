#!/usr/bin/env python3
# coding: utf-8
"""
TUfRG_plot_snapshot_evolution.py
Version: snapshot-plotter-2026-05-20

Plot snapshot diagnosis evolution from
CASE_DIR/snapshot_diagnosis/snapshot_diagnosis_summary.csv.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Sequence

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def cols_existing(df: pd.DataFrame, cols: Sequence[str]) -> List[str]:
    return [c for c in cols if c in df.columns]


def plot_lines(df: pd.DataFrame, xcol: str, cols: Sequence[str], outpath: Path, *, logy: bool, title: str) -> None:
    cols = cols_existing(df, cols)
    if not cols:
        return
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    x = pd.to_numeric(df[xcol], errors="coerce")
    for c in cols:
        y = pd.to_numeric(df[c], errors="coerce")
        if logy:
            plt.semilogy(x, y.abs() + 1e-300, marker="o", ms=3, lw=1.4, label=c)
        else:
            plt.plot(x, y, marker="o", ms=3, lw=1.4, label=c)
    plt.xlabel(xcol)
    plt.ylabel("value")
    plt.title(title)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()


def plot_top_sector(df: pd.DataFrame, xcol: str, outpath: Path) -> None:
    if "top_sector" not in df.columns or "top_strength" not in df.columns:
        return
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    x = pd.to_numeric(df[xcol], errors="coerce")
    for sec, g in df.groupby("top_sector", dropna=False):
        gx = pd.to_numeric(g[xcol], errors="coerce")
        gy = pd.to_numeric(g["top_strength"], errors="coerce")
        plt.semilogy(gx, gy.abs() + 1e-300, marker="o", ms=7, lw=0, label=str(sec))
    plt.xlabel(xcol)
    plt.ylabel("top strength")
    plt.title("Top instability sector by snapshot")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot TUFRG snapshot diagnosis evolution.")
    p.add_argument("case_dir")
    p.add_argument("--summary", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--x", choices=["iter", "completed_steps", "Lambda", "vertex_max"], default="iter")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    case_dir = Path(args.case_dir).resolve()
    summary = Path(args.summary) if args.summary else case_dir / "snapshot_diagnosis" / "snapshot_diagnosis_summary.csv"
    outdir = Path(args.out) if args.out else case_dir / "snapshot_diagnosis" / "plots"
    df = pd.read_csv(summary)
    xcol = args.x if args.x in df.columns and not df[args.x].isna().all() else "completed_steps"

    plot_lines(df, xcol, ["vertex_max", "chan_P", "chan_C", "chan_D"], outdir / "flow_scalars_log.png", logy=True, title="Flow scalars")
    plot_lines(df, xcol, ["top_strength", "Gamma_pp_singlet", "Gamma_ph-rho", "Gamma_ph-sx", "Gamma_ph-sy", "Gamma_ph-sz"], outdir / "gamma_strengths_log.png", logy=True, title="Gamma strengths")
    plot_lines(df, xcol, ["Mbest_ph-rho", "Mbest_ph-sx", "Mbest_ph-sy", "Mbest_ph-sz", "Mbest_pp-singlet"], outdir / "Mbest_strengths_log.png", logy=True, title="Best M strengths")
    plot_lines(df, xcol, ["Gamma_ph-rho_LCO_score", "Gamma_ph-sz_LCO_score", "Mbest_ph-rho_LCO_score", "Mbest_ph-sz_LCO_score"], outdir / "LCO_scores.png", logy=False, title="LCO scores")
    plot_lines(df, xcol, ["Gamma_ph-rho_CBO_SBO_score", "Gamma_ph-sz_CBO_SBO_score", "Mbest_ph-rho_CBO_SBO_score", "Mbest_ph-sz_CBO_SBO_score"], outdir / "CBO_SBO_scores.png", logy=False, title="CBO/SBO scores")
    plot_lines(df, xcol, ["Gamma_ph-rho_subspace_LCO_score", "Gamma_ph-sz_subspace_LCO_score", "Mbest_ph-rho_subspace_LCO_score", "Mbest_ph-sz_subspace_LCO_score"], outdir / "subspace_LCO_scores.png", logy=False, title="Subspace LCO scores")
    plot_top_sector(df, xcol, outdir / "top_sector.png")
    print("read", summary)
    print("wrote plots to", outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
