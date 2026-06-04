#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_num_token(x):
    if x is None:
        return np.nan
    if isinstance(x, (int, float, np.number)):
        return float(x)
    s = str(x).strip()
    if not s:
        return np.nan
    s = s.replace("m", "-").replace("p", ".")
    try:
        return float(s)
    except Exception:
        return np.nan


def find_metadata(start: Path):
    for p in [start] + list(start.parents):
        f = p / "metadata.json"
        if f.exists():
            try:
                return json.loads(f.read_text())
            except Exception:
                return {}
    return {}


def parse_case_from_path(path: Path):
    text = str(path)

    meta = find_metadata(path.parent)
    out = {}

    # Metadata first, if available.
    for key in ["U", "V", "V1", "lambda_nn", "lambda_nnn", "mu", "target_filling"]:
        if key in meta:
            out[key] = parse_num_token(meta[key])
    if "params" in meta and isinstance(meta["params"], dict):
        for key in ["U", "V", "V1", "lambda_nn", "lambda_nnn", "mu", "target_filling"]:
            if key in meta["params"]:
                out[key] = parse_num_token(meta["params"][key])

    # Directory-name fallback.
    # Important: lnn but not lnnn.
    m = re.search(r"lnn(?!n)([0-9mp\.\-]+)", text)
    if "lambda_nn" not in out and m:
        out["lambda_nn"] = parse_num_token(m.group(1))

    m = re.search(r"U([0-9mp\.\-]+)", text)
    if "U" not in out and m:
        out["U"] = parse_num_token(m.group(1))

    m = re.search(r"V([0-9mp\.\-]+)", text)
    if "V" not in out and m:
        out["V"] = parse_num_token(m.group(1))

    if "V" not in out and "V1" in out:
        out["V"] = out["V1"]

    # Ignore lambda_nnn folders by default unless lambda_nn is explicitly present.
    if "lambda_nn" not in out:
        out["lambda_nn"] = np.nan
    if "lambda_nnn" not in out:
        out["lambda_nnn"] = 0.0

    # Case directory: first parent containing metadata, otherwise nearest folder.
    case_dir = path.parent
    for p in [path.parent] + list(path.parents):
        if (p / "metadata.json").exists():
            case_dir = p
            break

    out["case_dir"] = str(case_dir)
    out["case"] = case_dir.name
    return out


def canonical_q(q):
    s = str(q)
    if s.lower().startswith("gamma") or s in ["0", "G", "Γ"]:
        return "Gamma"
    if s.lower().startswith("m") or "Mbest" in s:
        return "M"
    return s


def canonical_sector(sec):
    s = str(sec).strip()
    s = s.replace("_", "-")
    mapping = {
        "ph-rho": "ph-rho",
        "rho": "ph-rho",
        "charge": "ph-rho",
        "ph-sx": "ph-sx",
        "sx": "ph-sx",
        "ph-sy": "ph-sy",
        "sy": "ph-sy",
        "ph-sz": "ph-sz",
        "sz": "ph-sz",
        "spin": "ph-spin",
        "mag": "ph-spin",
        "pp-singlet": "pp-singlet",
        "singlet": "pp-singlet",
    }
    return mapping.get(s, s)


def first_existing_col(df, candidates):
    cols = list(df.columns)
    lower = {c.lower(): c for c in cols}
    for c in candidates:
        if c in df.columns:
            return c
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def row_common(df, i):
    def get(cands, default=np.nan):
        c = first_existing_col(df, cands)
        if c is None:
            return default
        return df.iloc[i][c]

    return {
        "snapshot": get(["snapshot", "snapshot_id"], ""),
        "iter": parse_num_token(get(["iter", "flow_iter", "iteration"], np.nan)),
        "Lambda": parse_num_token(get(["Lambda", "lambda"], np.nan)),
        "vertex_max": parse_num_token(get(["vertex_max", "max_vertex"], np.nan)),
        "top_strength": parse_num_token(get(["top_strength", "top_strength_raw"], np.nan)),
        "top_q": get(["top_q", "q_label", "q"], ""),
        "top_sector": get(["top_sector", "sector"], ""),
    }


def extract_wide(df, csv_path: Path):
    out = []
    meta = parse_case_from_path(csv_path)

    lco_cols = [c for c in df.columns if str(c).endswith("_LCO_score")]
    for i in range(len(df)):
        common = row_common(df, i)

        for lco_col in lco_cols:
            prefix = lco_col[:-len("_LCO_score")]

            # Common wide prefixes:
            # Gamma_ph_rho_LCO_score paired with Gamma_ph_rho
            # Mbest_ph_sz_LCO_score paired with Mbest_ph_sz
            strength_col = prefix
            if strength_col not in df.columns:
                continue

            lco = parse_num_token(df.iloc[i][lco_col])
            strength = parse_num_token(df.iloc[i][strength_col])
            if not np.isfinite(lco) or not np.isfinite(strength):
                continue

            cbo_col = prefix + "_CBO_SBO_score"
            cbo = parse_num_token(df.iloc[i][cbo_col]) if cbo_col in df.columns else np.nan

            q = "Gamma" if prefix.startswith("Gamma") else ("M" if prefix.startswith("Mbest") else "")
            sector = prefix
            sector = sector.replace("Gamma_", "").replace("Mbest_", "")
            sector = canonical_sector(sector)

            row = {
                **meta,
                **common,
                "source_csv": str(csv_path),
                "format": "wide",
                "q": q,
                "sector": sector,
                "mode_index": np.nan,
                "strength": strength,
                "LCO_score": lco,
                "CBO_SBO_score": cbo,
                "LCO_weighted_strength": strength * lco,
            }
            out.append(row)

    return out


def extract_long(df, csv_path: Path):
    out = []
    meta = parse_case_from_path(csv_path)

    lco_col = first_existing_col(df, [
        "LCO_score", "lco_score",
        "NN_minus_weight", "W_NN_minus",
        "nn_minus_weight",
        "charge_LCO_score", "spin_LCO_score",
    ])
    if lco_col is None:
        return out

    strength_col = first_existing_col(df, [
        "strength", "top_strength", "singular_value", "S", "sv", "value",
    ])
    if strength_col is None:
        return out

    q_col = first_existing_col(df, ["q", "q_label", "top_q", "momentum"])
    sec_col = first_existing_col(df, ["sector", "top_sector", "channel", "family"])
    mode_col = first_existing_col(df, ["mode", "mode_index", "top_mode"])

    cbo_col = first_existing_col(df, [
        "CBO_SBO_score", "cbo_sbo_score", "NN_plus_weight", "W_NN_plus", "nn_plus_weight",
    ])

    onsite_col = first_existing_col(df, ["onsite_weight", "W_onsite"])
    nn_col = first_existing_col(df, ["nn_weight", "W_NN"])
    long_col = first_existing_col(df, ["long_weight", "W_long"])

    for i in range(len(df)):
        common = row_common(df, i)
        lco = parse_num_token(df.iloc[i][lco_col])
        strength = parse_num_token(df.iloc[i][strength_col])
        if not np.isfinite(lco) or not np.isfinite(strength):
            continue

        q_raw = df.iloc[i][q_col] if q_col else common["top_q"]
        sec_raw = df.iloc[i][sec_col] if sec_col else common["top_sector"]

        row = {
            **meta,
            **common,
            "source_csv": str(csv_path),
            "format": "long",
            "q": canonical_q(q_raw),
            "sector": canonical_sector(sec_raw),
            "mode_index": parse_num_token(df.iloc[i][mode_col]) if mode_col else np.nan,
            "strength": strength,
            "LCO_score": lco,
            "CBO_SBO_score": parse_num_token(df.iloc[i][cbo_col]) if cbo_col else np.nan,
            "onsite_weight": parse_num_token(df.iloc[i][onsite_col]) if onsite_col else np.nan,
            "nn_weight": parse_num_token(df.iloc[i][nn_col]) if nn_col else np.nan,
            "long_weight": parse_num_token(df.iloc[i][long_col]) if long_col else np.nan,
            "LCO_weighted_strength": strength * lco,
        }
        out.append(row)

    return out


def load_all_rows(roots):
    rows = []
    for root in roots:
        root = Path(root).expanduser().resolve()
        if not root.exists():
            print(f"[WARN] root does not exist: {root}")
            continue

        csvs = sorted(root.rglob("*.csv"))
        print(f"[INFO] scanning {root}, found {len(csvs)} csv files")

        for csv_path in csvs:
            name = csv_path.name.lower()

            # Skip huge/irrelevant raw component dumps unless they contain useful columns.
            if any(x in name for x in ["top_components", "components"]) and "summary" not in name:
                continue

            try:
                df = pd.read_csv(csv_path)
            except Exception:
                continue

            cols = set(map(str, df.columns))
            if not any(c.endswith("_LCO_score") for c in cols) and not any(
                c.lower() in ["lco_score", "nn_minus_weight", "w_nn_minus"] for c in cols
            ):
                continue

            wide = extract_wide(df, csv_path)
            long = extract_long(df, csv_path)

            # Use both if available; they answer slightly different things.
            rows.extend(wide)
            rows.extend(long)

    return pd.DataFrame(rows)


def make_summaries(all_rows, outdir: Path, vertex_min, vertex_max):
    outdir.mkdir(parents=True, exist_ok=True)

    df = all_rows.copy()

    # Keep only lambda_nn scans, not lambda_nnn scans.
    df = df[df["lambda_nn"].notna()]
    if "lambda_nnn" in df.columns:
        df = df[(df["lambda_nnn"].isna()) | (np.isclose(df["lambda_nnn"], 0.0))]

    # Only PH sectors are meaningful for LCO tendency.
    df = df[df["sector"].astype(str).str.startswith("ph-")]

    # Pre-runaway window: if vertex_max absent, keep but mark as unknown.
    v = pd.to_numeric(df["vertex_max"], errors="coerce")
    has_v = v.notna()
    in_window = (~has_v) | ((v >= vertex_min) & (v <= vertex_max))
    df["in_window"] = in_window

    df.to_csv(outdir / "lco_tendency_all_rows.csv", index=False)

    win = df[df["in_window"]].copy()
    if win.empty:
        raise RuntimeError("No rows survived the pre-runaway window. Try wider --vertex-min/--vertex-max.")

    group_cols = ["U", "V", "lambda_nn", "q", "sector"]

    # Best single-mode tendency: max S * W_LCO.
    idx = win.groupby(group_cols)["LCO_weighted_strength"].idxmax()
    best = win.loc[idx].copy().sort_values(group_cols)
    best.rename(columns={
        "LCO_weighted_strength": "T_LCO_max_mode",
        "LCO_score": "LCO_score_at_Tmax",
        "strength": "strength_at_Tmax",
    }, inplace=True)

    # Spectral proxy: sum over saved modes per snapshot, then max over snapshot.
    snapshot_cols = ["U", "V", "lambda_nn", "q", "sector", "case", "snapshot", "iter"]
    spec_by_snap = (
        win.groupby(snapshot_cols, dropna=False)["LCO_weighted_strength"]
        .sum()
        .reset_index()
        .rename(columns={"LCO_weighted_strength": "T_LCO_spectral_sum"})
    )
    idx2 = spec_by_snap.groupby(group_cols)["T_LCO_spectral_sum"].idxmax()
    spec = spec_by_snap.loc[idx2].copy().sort_values(group_cols)

    best.to_csv(outdir / "lco_tendency_best_mode_summary.csv", index=False)
    spec.to_csv(outdir / "lco_tendency_spectral_summary.csv", index=False)

    # Ratio to lambda_nn = 0 baseline for each (U,V,q,sector).
    base = best[np.isclose(best["lambda_nn"], 0.0)].copy()
    base = base[group_cols[:2] + ["q", "sector", "T_LCO_max_mode"]]
    base = base.rename(columns={"T_LCO_max_mode": "T_LCO_lambda0"})

    ratio = best.merge(base, on=["U", "V", "q", "sector"], how="left")
    ratio["R_vs_lambda0"] = ratio["T_LCO_max_mode"] / ratio["T_LCO_lambda0"]
    ratio.to_csv(outdir / "lco_tendency_ratio_to_lambda0.csv", index=False)

    # Compact table.
    compact_cols = [
        "U", "V", "lambda_nn", "q", "sector",
        "T_LCO_max_mode", "T_LCO_lambda0", "R_vs_lambda0",
        "strength_at_Tmax", "LCO_score_at_Tmax",
        "case", "snapshot", "iter", "vertex_max", "source_csv",
    ]
    ratio[[c for c in compact_cols if c in ratio.columns]].to_csv(
        outdir / "lco_tendency_compact.csv", index=False
    )

    # Plot each U,V,q,sector.
    plotdir = outdir / "plots"
    plotdir.mkdir(exist_ok=True)

    for (U, V, q, sector), sub in ratio.groupby(["U", "V", "q", "sector"], dropna=False):
        sub = sub.sort_values("lambda_nn")
        if len(sub) < 2:
            continue

        plt.figure(figsize=(6, 4))
        plt.plot(sub["lambda_nn"], sub["T_LCO_max_mode"], marker="o")
        plt.xlabel(r"$\lambda_{\rm NN}$")
        plt.ylabel(r"$T_{\rm LCO}^{\max} = S_n W_n^{\rm LCO}$")
        plt.title(f"U={U}, V={V}, {q}, {sector}")
        plt.tight_layout()
        fname = f"U{U}_V{V}_{q}_{sector}.png".replace(".", "p").replace("/", "_")
        plt.savefig(plotdir / fname, dpi=180)
        plt.close()

    return ratio, best, spec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--roots",
        nargs="+",
        default=[
            "runs/km_spinful_runs",
            "runs/km_lco_finetune_runs",
            "km_spinful_runs",
            "km_lco_finetune_runs",
        ],
        help="Root folders to scan recursively.",
    )
    ap.add_argument("--outdir", default="lco_tendency_lambda_nn_analysis")
    ap.add_argument("--vertex-min", type=float, default=2.0)
    ap.add_argument("--vertex-max", type=float, default=8.0)
    args = ap.parse_args()

    roots = [Path(r) for r in args.roots]
    df = load_all_rows(roots)

    if df.empty:
        raise RuntimeError(
            "No usable rows found. I looked for CSV columns like *_LCO_score, "
            "LCO_score, NN_minus_weight, and strength/top_strength."
        )

    outdir = Path(args.outdir)
    ratio, best, spec = make_summaries(df, outdir, args.vertex_min, args.vertex_max)

    print("\n[DONE] Wrote:")
    for f in [
        "lco_tendency_all_rows.csv",
        "lco_tendency_best_mode_summary.csv",
        "lco_tendency_spectral_summary.csv",
        "lco_tendency_ratio_to_lambda0.csv",
        "lco_tendency_compact.csv",
    ]:
        print("  ", outdir / f)
    print("  ", outdir / "plots")

    print("\n[QUICK VIEW: ratio to lambda_nn=0]")
    show = ratio[
        ["U", "V", "lambda_nn", "q", "sector", "T_LCO_max_mode", "T_LCO_lambda0", "R_vs_lambda0"]
    ].sort_values(["U", "V", "q", "sector", "lambda_nn"])
    with pd.option_context("display.max_rows", 200, "display.width", 160):
        print(show.to_string(index=False))


if __name__ == "__main__":
    main()
