#!/usr/bin/env python3
# coding: utf-8
"""
TUfRG_diagnose_snapshots.py

Run/collect diagnosis for snapshot directories.

Important correction:
    For q=Gamma, sector=ph-rho, the physical ordering strength uses
    strength_after_remove_onsite_A1 when available, instead of raw
    strength_svd_KL. This avoids treating uniform q=0 charge forward scattering
    as a symmetry-breaking instability.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_csv_safe(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def run_cmd(cmd: Sequence[str]) -> None:
    print("[run]", " ".join(str(x) for x in cmd), flush=True)
    subprocess.run(list(cmd), check=True)


def parse_iter_from_name(path: Path) -> Optional[int]:
    m = re.search(r"iter[_-](\d+)", path.name)
    if m:
        return int(m.group(1))
    if path.name == "final":
        return None
    return None


def find_snapshots(case_dir: Path, snapshot_root: str) -> List[Path]:
    root = case_dir / snapshot_root
    if not root.exists():
        raise FileNotFoundError(root)
    snaps = []
    for p in root.iterdir():
        if p.is_dir() and (p / "O.dvg").exists():
            snaps.append(p)

    def key(p: Path):
        it = parse_iter_from_name(p)
        return (10**9 if it is None else it, p.name)

    return sorted(snaps, key=key)


def diagnosis_exists(snap: Path, instability_dir: str, formfactor_dir: str, order_dir: str) -> bool:
    required = [
        snap / instability_dir / "instability_summary.csv",
        snap / instability_dir / "instability_modes.npz",
        snap / formfactor_dir / "formfactor_summary.csv",
        snap / order_dir / "order_summary.csv",
    ]
    return all(p.exists() and p.stat().st_size > 0 for p in required)


def diagnose_snapshot(
    snap: Path,
    n_modes: int,
    instability_dir: str,
    formfactor_dir: str,
    order_dir: str,
    order_top_components: int,
    skip_existing: bool,
) -> None:
    if skip_existing and diagnosis_exists(snap, instability_dir, formfactor_dir, order_dir):
        print("[skip existing]", snap, flush=True)
        return

    run_cmd([
        "python", "-u", "TUfRG_instability.py", str(snap),
        "--out", instability_dir,
        "--n-modes", str(n_modes),
    ])

    run_cmd([
        "python", "-u", "TUfRG_formfactor_from_modes.py", str(snap),
        "--instability-dir", instability_dir,
        "--out", formfactor_dir,
    ])

    run_cmd([
        "python", "-u", "TUfRG_order_classifier.py", str(snap),
        "--instability-dir", instability_dir,
        "--out", order_dir,
        "--max-rows", "120",
        "--top-components", str(order_top_components),
    ])


def corrected_ordering_strength(row: pd.Series) -> float:
    raw = float(row.get("strength_svd_KL", 0.0))
    q = str(row.get("q_label", ""))
    fam = str(row.get("family", ""))
    sec = str(row.get("sector", ""))

    # New default: projected Gamma charge/rho modes are real rows with their own vectors.
    # Their strength is already the projected SVD strength.
    if q == "Gamma" and fam == "ph" and sec in {"ph-rho-A1proj", "charge-A1proj"}:
        return raw

    # Keep raw Gamma ph-rho as a diagnostic row only. Do not let it win the
    # global top competition when projected A1 rows exist, otherwise strength
    # and eigenvector are not mode-aligned.
    if q == "Gamma" and fam == "ph" and sec in {"ph-rho", "charge"}:
        val = row.get("strength_after_remove_onsite_A1", np.nan)
        try:
            val = float(val)
            if np.isfinite(val):
                return val
        except Exception:
            pass

    return raw

def get_strength(inst: pd.DataFrame, q: str, family: str, sector: str, corrected: bool = True) -> float:
    if inst.empty:
        return np.nan
    df = inst[
        (inst["q_label"].astype(str) == q)
        & (inst["family"].astype(str) == family)
        & (inst["sector"].astype(str) == sector)
    ].copy()
    if df.empty:
        return np.nan
    if corrected:
        vals = df.apply(corrected_ordering_strength, axis=1).astype(float)
    else:
        vals = df["strength_svd_KL"].astype(float)
    return float(vals.max())


def get_mbest_strength(inst: pd.DataFrame, family: str, sector: str, corrected: bool = True) -> float:
    if inst.empty:
        return np.nan
    df = inst[
        inst["q_label"].astype(str).str.startswith("M")
        & (inst["family"].astype(str) == family)
        & (inst["sector"].astype(str) == sector)
    ].copy()
    if df.empty:
        return np.nan
    if corrected:
        vals = df.apply(corrected_ordering_strength, axis=1).astype(float)
    else:
        vals = df["strength_svd_KL"].astype(float)
    return float(vals.max())


def get_score(order: pd.DataFrame, q_kind: str, sector: str, score_col: str) -> float:
    if order.empty or score_col not in order.columns:
        return np.nan
    df = order[(order["family"].astype(str) == "ph") & (order["sector"].astype(str) == sector)].copy()
    if q_kind == "Gamma":
        df = df[df["q_label"].astype(str) == "Gamma"]
    elif q_kind == "Mbest":
        df = df[df["q_label"].astype(str).str.startswith("M")]
    else:
        df = df[df["q_label"].astype(str) == q_kind]
    if df.empty:
        return np.nan
    return float(pd.to_numeric(df[score_col], errors="coerce").max())


def load_flow_row(case_dir: Path, iteration: Optional[int]) -> Dict[str, float]:
    if iteration is None:
        return {}
    fdat = case_dir / "F.dat"
    if not fdat.exists():
        return {}
    try:
        flow = pd.read_csv(fdat)
    except Exception:
        return {}
    if flow.empty:
        return {}

    if "iter" in flow.columns:
        row = flow.iloc[(flow["iter"].astype(int) - int(iteration)).abs().argsort()[:1]]
        if row.empty:
            return {}
        r = row.iloc[0]
    else:
        idx = min(max(int(iteration), 0), len(flow) - 1)
        r = flow.iloc[idx]

    out = {}
    for c in ["Lambda", "dLambda", "vertex_max", "chan_P", "chan_C", "chan_D", "loop_pp", "loop_ph"]:
        if c in flow.columns:
            try:
                out[c] = float(r[c])
            except Exception:
                pass
    return out



def select_best_inst_row(inst: pd.DataFrame, mask: pd.Series, strength_col: str = "ordering_strength"):
    if inst.empty:
        return None
    df = inst[mask].copy()
    if df.empty:
        return None
    if strength_col not in df.columns:
        strength_col = "strength_svd_KL"
    vals = pd.to_numeric(df[strength_col], errors="coerce")
    if vals.isna().all():
        return None
    return df.iloc[vals.fillna(-np.inf).argmax()]


def order_row_by_mode_key(order: pd.DataFrame, mode_key: object):
    if order.empty:
        return None
    df = order[order["mode_key"].astype(str) == str(mode_key)]
    if df.empty:
        return None
    return df.iloc[0]


def add_mode_aligned_fields(row: Dict[str, object], prefix: str, inst_row, order: pd.DataFrame) -> None:
    """Append mode-aligned strength and structural classification fields.

    This is the important summary fix: the LCO/CBO scores written here belong
    to exactly the same mode_key as the strength.
    """
    if inst_row is None:
        return

    row[f"{prefix}_mode_key"] = inst_row.get("mode_key", "")
    row[f"{prefix}_q"] = inst_row.get("q_label", "")
    row[f"{prefix}_family"] = inst_row.get("family", "")
    row[f"{prefix}_sector"] = inst_row.get("sector", "")
    row[f"{prefix}_mode"] = inst_row.get("mode", "")
    row[f"{prefix}_strength"] = float(inst_row.get("ordering_strength", inst_row.get("strength_svd_KL", np.nan)))
    row[f"{prefix}_strength_raw"] = float(inst_row.get("strength_svd_KL", np.nan))

    o = order_row_by_mode_key(order, inst_row.get("mode_key", ""))
    if o is None:
        return

    for c in [
        "order_label",
        "onsite_weight",
        "nn_weight",
        "long_weight",
        "diagonal_weight",
        "offdiag_weight",
        "bond_symmetric_nn_weight",
        "bond_antisymmetric_nn_weight",
        "LCO_score",
        "CBO_SBO_score",
    ]:
        if c in o.index:
            row[f"{prefix}_{c}"] = o.get(c, np.nan)


def get_strength_any_sector(inst: pd.DataFrame, q: str, family: str, sectors: Sequence[str], corrected: bool = True) -> float:
    if inst.empty:
        return np.nan
    df = inst[
        (inst["q_label"].astype(str) == q)
        & (inst["family"].astype(str) == family)
        & (inst["sector"].astype(str).isin(list(sectors)))
    ].copy()
    if df.empty:
        return np.nan
    if corrected and "ordering_strength" in df.columns:
        vals = pd.to_numeric(df["ordering_strength"], errors="coerce")
    elif corrected:
        vals = df.apply(corrected_ordering_strength, axis=1).astype(float)
    else:
        vals = pd.to_numeric(df["strength_svd_KL"], errors="coerce")
    return float(vals.max())


def get_score_for_top_mode(order: pd.DataFrame, inst_row, score_col: str) -> float:
    if inst_row is None or order.empty or score_col not in order.columns:
        return np.nan
    o = order_row_by_mode_key(order, inst_row.get("mode_key", ""))
    if o is None:
        return np.nan
    return float(pd.to_numeric(pd.Series([o.get(score_col, np.nan)]), errors="coerce").iloc[0])

def collect_one_snapshot(
    case_dir: Path,
    snap: Path,
    instability_dir: str,
    formfactor_dir: str,
    order_dir: str,
) -> Dict[str, object]:
    iteration = parse_iter_from_name(snap)

    inst = read_csv_safe(snap / instability_dir / "instability_summary.csv")
    form = read_csv_safe(snap / formfactor_dir / "formfactor_summary.csv")
    order = read_csv_safe(snap / order_dir / "order_summary.csv")
    sub = read_csv_safe(snap / order_dir / "order_subspace_summary.csv")

    row: Dict[str, object] = {
        "snapshot": snap.name,
        "iter": iteration if iteration is not None else np.nan,
    }
    row.update(load_flow_row(case_dir, iteration))

    if not inst.empty:
        inst = inst.copy()
        inst["ordering_strength"] = inst.apply(corrected_ordering_strength, axis=1)

        # If A1-projected Gamma charge/rho rows exist, raw Gamma ph-rho/charge rows
        # must not be used as physical top modes, because their vector is raw.
        has_A1proj = (
            (inst["q_label"].astype(str) == "Gamma")
            & (inst["family"].astype(str) == "ph")
            & (inst["sector"].astype(str).isin(["ph-rho-A1proj", "charge-A1proj"]))
        ).any()
        if has_A1proj:
            raw_gamma_charge = (
                (inst["q_label"].astype(str) == "Gamma")
                & (inst["family"].astype(str) == "ph")
                & (inst["sector"].astype(str).isin(["ph-rho", "charge"]))
            )
            inst.loc[raw_gamma_charge, "ordering_strength"] = -np.inf

        top = inst.sort_values("ordering_strength", ascending=False).iloc[0]
        row.update({
            "top_q": top.get("q_label", ""),
            "top_family": top.get("family", ""),
            "top_sector": top.get("sector", ""),
            "top_mode": int(top.get("mode", 0)),
            "top_strength": float(top.get("ordering_strength", np.nan)),
            "top_strength_raw": float(top.get("strength_svd_KL", np.nan)),
            "top_strength_note": "Gamma ph-rho uses explicit ph-rho-A1proj row when available",
        })
        add_mode_aligned_fields(row, "top", top, order)

        gamma_A1 = select_best_inst_row(
            inst,
            (inst["q_label"].astype(str) == "Gamma")
            & (inst["family"].astype(str) == "ph")
            & (inst["sector"].astype(str).isin(["ph-rho-A1proj", "charge-A1proj"])),
            "ordering_strength",
        )
        add_mode_aligned_fields(row, "Gamma_ph_rho_A1proj_top", gamma_A1, order)

        m_rho_top = select_best_inst_row(
            inst,
            (inst["q_label"].astype(str).str.startswith("M"))
            & (inst["family"].astype(str) == "ph")
            & (inst["sector"].astype(str).isin(["ph-rho", "charge"])),
            "ordering_strength",
        )
        add_mode_aligned_fields(row, "Mbest_ph_rho_top", m_rho_top, order)

        row["Gamma_pp_singlet"] = get_strength(inst, "Gamma", "pp", "pp-singlet")
        row["Gamma_ph_rho_raw"] = get_strength(inst, "Gamma", "ph", "ph-rho", corrected=False)
        row["Gamma_ph_rho_A1proj"] = get_strength_any_sector(inst, "Gamma", "ph", ["ph-rho-A1proj", "charge-A1proj"], corrected=True)
        row["Gamma_ph_rho"] = row["Gamma_ph_rho_A1proj"] if np.isfinite(row["Gamma_ph_rho_A1proj"]) else get_strength(inst, "Gamma", "ph", "ph-rho", corrected=True)
        row["Gamma_ph_sx"] = get_strength(inst, "Gamma", "ph", "ph-sx")
        row["Gamma_ph_sy"] = get_strength(inst, "Gamma", "ph", "ph-sy")
        row["Gamma_ph_sz"] = get_strength(inst, "Gamma", "ph", "ph-sz")

        row["Mbest_pp_singlet"] = get_mbest_strength(inst, "pp", "pp-singlet")
        row["Mbest_ph_rho"] = get_mbest_strength(inst, "ph", "ph-rho")
        row["Mbest_ph_sx"] = get_mbest_strength(inst, "ph", "ph-sx")
        row["Mbest_ph_sy"] = get_mbest_strength(inst, "ph", "ph-sy")
        row["Mbest_ph_sz"] = get_mbest_strength(inst, "ph", "ph-sz")

    for sec in ["ph-rho", "ph-sx", "ph-sy", "ph-sz"]:
        clean = sec.replace("-", "_")

        if sec == "ph-rho" and "Gamma_ph_rho_A1proj_top_mode_key" in row:
            row[f"Gamma_{clean}_LCO_score"] = get_score_for_top_mode(order, gamma_A1, "LCO_score")
            row[f"Gamma_{clean}_CBO_SBO_score"] = get_score_for_top_mode(order, gamma_A1, "CBO_SBO_score")
            row[f"Gamma_{clean}_order_label"] = row.get("Gamma_ph_rho_A1proj_top_order_label", "")
            row[f"Gamma_{clean}_score_note"] = "mode-aligned score from Gamma ph-rho-A1proj top mode"
        else:
            row[f"Gamma_{clean}_LCO_score"] = get_score(order, "Gamma", sec, "LCO_score")
            row[f"Gamma_{clean}_CBO_SBO_score"] = get_score(order, "Gamma", sec, "CBO_SBO_score")

        if sec == "ph-rho" and "Mbest_ph_rho_top_mode_key" in row:
            row[f"Mbest_{clean}_LCO_score"] = get_score_for_top_mode(order, m_rho_top, "LCO_score")
            row[f"Mbest_{clean}_CBO_SBO_score"] = get_score_for_top_mode(order, m_rho_top, "CBO_SBO_score")
            row[f"Mbest_{clean}_order_label"] = row.get("Mbest_ph_rho_top_order_label", "")
            row[f"Mbest_{clean}_score_note"] = "mode-aligned score from strongest M ph-rho top mode"
        else:
            row[f"Mbest_{clean}_LCO_score"] = get_score(order, "Mbest", sec, "LCO_score")
            row[f"Mbest_{clean}_CBO_SBO_score"] = get_score(order, "Mbest", sec, "CBO_SBO_score")

    # Basic subspace scores, if present.
    if not sub.empty:
        for sec in ["ph-rho", "ph-sz"]:
            sdf = sub[sub["sector"].astype(str) == sec].copy()
            gdf = sdf[sdf["q_label"].astype(str) == "Gamma"]
            mdf = sdf[sdf["q_label"].astype(str).str.startswith("M")]
            for prefix, df in [(f"Gamma_{sec.replace('-', '_')}", gdf), (f"Mbest_{sec.replace('-', '_')}", mdf)]:
                if not df.empty:
                    for col in ["subspace_LCO_score", "subspace_CBO_SBO_score"]:
                        if col in df.columns:
                            row[prefix + "_" + col] = float(pd.to_numeric(df[col], errors="coerce").max())

    return row


def plot_summary(summary: pd.DataFrame, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    if summary.empty:
        return
    df = summary.sort_values("iter").copy()
    x = df["iter"].astype(float)

    # Flow scalars.
    plt.figure(figsize=(10, 6))
    for c in ["vertex_max", "chan_P", "chan_C", "chan_D"]:
        if c in df.columns:
            plt.semilogy(x, pd.to_numeric(df[c], errors="coerce") + 1e-300, marker="o", label=c)
    plt.xlabel("iter")
    plt.ylabel("value")
    plt.title("Flow scalars")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "flow_scalars_log.png", dpi=180)
    plt.close()

    # Top sector using corrected strength.
    plt.figure(figsize=(10, 6))
    for sec, g in df.groupby("top_sector"):
        plt.semilogy(g["iter"], g["top_strength"], "o", label=sec)
    plt.xlabel("iter")
    plt.ylabel("top physical ordering strength")
    plt.title("Top instability sector by snapshot, Gamma-rho A1-projected")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "top_sector_corrected.png", dpi=180)
    plt.close()

    # Gamma strengths.
    plt.figure(figsize=(10, 6))
    cols = ["top_strength", "Gamma_pp_singlet", "Gamma_ph_rho", "Gamma_ph_rho_raw", "Gamma_ph_sx", "Gamma_ph_sy", "Gamma_ph_sz"]
    for c in cols:
        if c in df.columns:
            plt.semilogy(x, pd.to_numeric(df[c], errors="coerce") + 1e-300, marker="o", label=c)
    plt.xlabel("iter")
    plt.ylabel("value")
    plt.title("Gamma strengths; ph-rho corrected plus raw shown")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "gamma_strengths_corrected_log.png", dpi=180)
    plt.close()

    # M strengths.
    plt.figure(figsize=(10, 6))
    for c in ["Mbest_ph_rho", "Mbest_ph_sx", "Mbest_ph_sy", "Mbest_ph_sz", "Mbest_pp_singlet"]:
        if c in df.columns:
            plt.semilogy(x, pd.to_numeric(df[c], errors="coerce") + 1e-300, marker="o", label=c)
    plt.xlabel("iter")
    plt.ylabel("value")
    plt.title("Best M strengths")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "Mbest_strengths_log.png", dpi=180)
    plt.close()

    # LCO scores.
    plt.figure(figsize=(10, 6))
    for c in ["Gamma_ph_rho_LCO_score", "Gamma_ph_sz_LCO_score", "Mbest_ph_rho_LCO_score", "Mbest_ph_sz_LCO_score"]:
        if c in df.columns:
            plt.plot(x, pd.to_numeric(df[c], errors="coerce"), marker="o", label=c)
    plt.xlabel("iter")
    plt.ylabel("value")
    plt.title("LCO scores")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "LCO_scores.png", dpi=180)
    plt.close()

    # CBO/SBO scores.
    plt.figure(figsize=(10, 6))
    for c in ["Gamma_ph_rho_CBO_SBO_score", "Gamma_ph_sz_CBO_SBO_score", "Mbest_ph_rho_CBO_SBO_score", "Mbest_ph_sz_CBO_SBO_score"]:
        if c in df.columns:
            plt.plot(x, pd.to_numeric(df[c], errors="coerce"), marker="o", label=c)
    plt.xlabel("iter")
    plt.ylabel("value")
    plt.title("CBO/SBO scores")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "CBO_SBO_scores.png", dpi=180)
    plt.close()


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Diagnose TUFRG snapshots and collect corrected snapshot summary.")
    p.add_argument("case_dir", nargs="?", default=None)
    p.add_argument("--snapshot-root", default="snapshots")
    p.add_argument("--instability-dir", default="instability")
    p.add_argument("--formfactor-dir", default="formfactor")
    p.add_argument("--order-dir", default="order_classification")
    p.add_argument("--out", default="snapshot_diagnosis")
    p.add_argument("--n-modes", type=int, default=8)
    p.add_argument("--order-top-components", type=int, default=30)
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--collect-only", action="store_true", help="Do not run diagnosis scripts; only collect existing outputs.")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    case_arg = args.case_dir or os.environ.get("CASE_DIR")
    if not case_arg:
        raise SystemExit("Need case_dir argument or CASE_DIR env var.")
    case_dir = Path(case_arg).resolve()
    snaps = find_snapshots(case_dir, args.snapshot_root)
    print(f"case_dir = {case_dir}")
    print(f"found snapshots = {len(snaps)}")

    if not args.collect_only:
        for snap in snaps:
            diagnose_snapshot(
                snap=snap,
                n_modes=args.n_modes,
                instability_dir=args.instability_dir,
                formfactor_dir=args.formfactor_dir,
                order_dir=args.order_dir,
                order_top_components=args.order_top_components,
                skip_existing=args.skip_existing,
            )

    rows = [
        collect_one_snapshot(
            case_dir=case_dir,
            snap=snap,
            instability_dir=args.instability_dir,
            formfactor_dir=args.formfactor_dir,
            order_dir=args.order_dir,
        )
        for snap in snaps
    ]

    outdir = case_dir / args.out
    outdir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(rows).sort_values("iter")
    summary_path = outdir / "snapshot_diagnosis_summary.csv"
    summary.to_csv(summary_path, index=False)
    print("wrote", summary_path)

    plot_dir = outdir / "plots"
    plot_summary(summary, plot_dir)
    print("wrote plots to", plot_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
