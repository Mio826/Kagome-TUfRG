#!/usr/bin/env python3
from pathlib import Path
import sys
import re
import json
import argparse
import traceback
import numpy as np
import pandas as pd

RUNS = Path.cwd()
PYSCRIPT = RUNS / "current_pyscript"
OUTDIR_BASE = RUNS / "current_diagnosis"

for c in [RUNS, PYSCRIPT, RUNS / "current_pyscript_appendixmodel"]:
    if c.exists() and str(c) not in sys.path:
        sys.path.insert(0, str(c))

import TUfRG_instability as inst

from calc_fig6a_d6a_corrected_strict_precritical import make_d6a_sources_on_O
from calc_fig5_corrected_strict_charge_precritical import strict_chi_for_eta

ROOT_LABEL = "nosoc"
ROOT_PATH = "no soc"

SECTORS = ["ph-rho", "ph-sx", "ph-sy", "ph-sz"]
QLABS = ["M1", "M2", "M3"]


def parse_num(tok):
    return float(tok.replace("p", "."))


def parse_params(case_name):
    out = {
        "U": np.nan,
        "V": np.nan,
        "lambda_nn": 0.0,
        "lambda_nnn": 0.0,
    }

    m = re.search(r"_U([0-9]+(?:p[0-9]+)?)", case_name)
    if m:
        out["U"] = parse_num(m.group(1))

    m = re.search(r"_V([0-9]+(?:p[0-9]+)?)", case_name)
    if m:
        out["V"] = parse_num(m.group(1))

    return out


def safe_json_load(p):
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def recursive_find_lambda(obj):
    hits = []

    def walk(x, path=""):
        if isinstance(x, dict):
            for k, v in x.items():
                pp = f"{path}.{k}" if path else str(k)
                kl = str(k).lower()

                if kl in ["lambda", "cutoff", "scale"]:
                    try:
                        fv = float(v)
                        if np.isfinite(fv) and fv > 0:
                            hits.append((pp, fv))
                    except Exception:
                        pass

                walk(v, pp)

        elif isinstance(x, list):
            for i, v in enumerate(x[:200]):
                walk(v, f"{path}[{i}]")

    walk(obj)

    for k, v in hits:
        if k.lower().endswith("flow_report.last_row.lambda"):
            return v, k

    for k, v in hits:
        if k.lower().endswith(".lambda"):
            return v, k

    if hits:
        return hits[0][1], hits[0][0]

    return np.nan, ""


def snapshot_iter(label):
    if label == "final":
        return 10**9
    m = re.match(r"iter_(\d+)$", label)
    if m:
        return int(m.group(1))
    return -1


def case_root_from_snapshot(snap_dir):
    if snap_dir.parent.name != "snapshots":
        return snap_dir.parent

    case_or_run = snap_dir.parent.parent
    if case_or_run.name == "run":
        return case_or_run.parent
    return case_or_run


def find_snapshots(root):
    snaps = []
    for odvg in sorted(root.glob("**/snapshots/*/O.dvg")):
        snap_dir = odvg.parent
        if snap_dir.name.startswith("iter_") or snap_dir.name == "final":
            snaps.append(snap_dir)
    return snaps


def extract_lambda(snap_dir):
    case_root = case_root_from_snapshot(snap_dir)

    candidates = [
        snap_dir / "metadata.json",
        snap_dir.parent / "metadata.json",
        case_root / "metadata.json",
        case_root / "run" / "metadata.json",
    ]

    for p in candidates:
        if p.exists():
            j = safe_json_load(p)
            if j is None:
                continue

            lam, key = recursive_find_lambda(j)
            if np.isfinite(lam) and lam > 0:
                return lam, str(p), key

    return np.nan, "", ""


def build_inventory():
    root = RUNS / ROOT_PATH
    if not root.exists():
        raise FileNotFoundError(f"Cannot find root folder: {root}")

    rows = []

    for snap_dir in find_snapshots(root):
        case_root = case_root_from_snapshot(snap_dir)
        case_name = case_root.name
        label = snap_dir.name
        it = snapshot_iter(label)
        odvg = snap_dir / "O.dvg"

        lam, src_file, src_key = extract_lambda(snap_dir)

        try:
            size_gb = odvg.stat().st_size / 1024**3
        except Exception:
            size_gb = np.nan

        rows.append({
            "root_label": ROOT_LABEL,
            "root_path": ROOT_PATH,
            "case_name": case_name,
            "case_root": str(case_root),
            "snapshot_label": label,
            "snapshot_iter": it,
            "snapshot_dir": str(snap_dir),
            "Lambda": lam,
            "Lambda_source_file": src_file,
            "Lambda_source_key": src_key,
            "O_dvg_size_GB": size_gb,
            **parse_params(case_name),
        })

    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values(["U", "V", "case_name", "snapshot_iter"])
    return df


def choose_snapshots_for_target(inv, target_lambda, max_logdiff):
    rows = []
    skipped = []

    good = inv[np.isfinite(inv["Lambda"]) & (inv["Lambda"] > 0)].copy()

    for case_name, g in good.groupby("case_name", dropna=False):
        gg = g.copy()
        gg["logdiff"] = np.abs(np.log(gg["Lambda"] / target_lambda))
        gg["is_final"] = (gg["snapshot_label"] == "final").astype(int)

        best = gg.sort_values(["logdiff", "is_final", "snapshot_iter"]).iloc[0].to_dict()

        final = gg[gg["snapshot_label"] == "final"]
        if len(final):
            final_lambda = float(final["Lambda"].iloc[-1])
        else:
            final_lambda = np.nan

        best["target_lambda"] = target_lambda
        best["final_lambda"] = final_lambda
        best["target_above_final_cutoff"] = bool(np.isfinite(final_lambda) and target_lambda >= final_lambda)

        if best["logdiff"] <= max_logdiff and best["target_above_final_cutoff"]:
            rows.append(best)
        else:
            skipped.append({
                "case_name": case_name,
                "U": best.get("U", np.nan),
                "V": best.get("V", np.nan),
                "target_lambda": target_lambda,
                "best_snapshot_label": best["snapshot_label"],
                "best_lambda": best["Lambda"],
                "final_lambda": final_lambda,
                "best_logdiff": best["logdiff"],
                "max_logdiff": max_logdiff,
                "target_above_final_cutoff": best["target_above_final_cutoff"],
            })

    selected = pd.DataFrame(rows)
    skipped = pd.DataFrame(skipped)

    if len(selected):
        selected = selected.sort_values(["U", "V", "case_name"])

    return selected, skipped


def compute_snapshot(row):
    snap_dir = Path(row["snapshot_dir"])

    print(
        f"[compute] U={row['U']} V={row['V']} "
        f"{row['case_name']} {row['snapshot_label']} "
        f"Lambda={row['Lambda']:.8g}",
        flush=True,
    )

    data = inst.load_case(snap_dir)
    inst.assert_raw_channel_compatibility(data)

    qmap = inst.q_labels_to_indices(data, QLABS)
    etas_by_q, source_diag = make_d6a_sources_on_O(data.O)

    out_rows = []

    for qlab in QLABS:
        iq = int(qmap[qlab])
        eta = etas_by_q[qlab]
        eta_norm2 = float(np.vdot(eta, eta).real)

        kernels = inst.build_explicit_kernels(data, iq)

        for spec in kernels:
            if spec.get("family") != "ph":
                continue
            if spec.get("sector") not in SECTORS:
                continue

            K = np.asarray(spec["K"], dtype=np.complex128)
            L = np.asarray(spec["L"], dtype=np.complex128)

            d = strict_chi_for_eta(eta, K, L)

            out_rows.append({
                "root_label": ROOT_LABEL,
                "root_path": ROOT_PATH,
                "case_name": row["case_name"],
                "case_root": row["case_root"],
                "snapshot_label": row["snapshot_label"],
                "snapshot_iter": int(row["snapshot_iter"]),
                "snapshot_dir": row["snapshot_dir"],
                "Lambda": float(row["Lambda"]),
                "target_lambda": float(row["target_lambda"]),
                "logdiff_to_target": float(row["logdiff"]),
                "final_lambda": float(row["final_lambda"]),
                "pattern": "fig6a_d6a_corrected",
                "q_label": qlab,
                "iq": iq,
                "sector": spec["sector"],
                "eta_norm2_q": eta_norm2,
                "U": row["U"],
                "V": row["V"],
                "lambda_nn": 0.0,
                "lambda_nnn": 0.0,
                **source_diag,
                **d,
            })

    return out_rows


def write_outputs(outdir, all_rows, errors):
    all_df = pd.DataFrame(all_rows)

    if len(all_df) and "B_abs_max_eval" not in all_df.columns:
        if "B_min_eval" in all_df.columns and "B_max_eval" in all_df.columns:
            all_df["B_abs_max_eval"] = all_df[["B_min_eval", "B_max_eval"]].abs().max(axis=1)
        elif "B_max_eval" in all_df.columns:
            all_df["B_abs_max_eval"] = all_df["B_max_eval"].abs()
        else:
            all_df["B_abs_max_eval"] = np.nan

    all_df.to_csv(outdir / "nosoc_fixed_cutoff_d6a_all_q.csv", index=False)
    (outdir / "errors.json").write_text(json.dumps(errors, indent=2))

    if len(all_df) == 0:
        return

    group_keys = [
        "root_label", "case_name", "U", "V",
        "snapshot_label", "snapshot_iter", "Lambda",
        "target_lambda", "logdiff_to_target", "final_lambda",
        "pattern", "sector"
    ]

    summed = all_df.groupby(group_keys, dropna=False).agg(
        chi_sum=("chi_strict_real", "sum"),
        chi_imag_abs_max=("chi_strict_imag", lambda x: float(np.max(np.abs(x)))),
        Bmax=("B_max_eval", "max"),
        Babsmax=("B_abs_max_eval", "max"),
        old_proxy_sum=("old_proxy_real", "sum"),
        n_q=("q_label", "nunique"),
        eta_norm2_total=("eta_norm2_q", "sum"),
        loop_orientation_sign=("loop_orientation_sign", "first"),
        chi0_neg_weight_ratio_max=("chi0_neg_weight_ratio", "max"),
    ).reset_index()

    summed.to_csv(outdir / "nosoc_fixed_cutoff_d6a_summed.csv", index=False)

    pv = summed.pivot_table(
        index=[
            "case_name", "U", "V",
            "snapshot_label", "snapshot_iter", "Lambda",
            "target_lambda", "logdiff_to_target", "final_lambda", "pattern"
        ],
        columns="sector",
        values=["chi_sum", "Bmax", "old_proxy_sum"],
        aggfunc="first",
    ).reset_index()

    pv.columns = ["__".join([str(x) for x in c if str(x) != ""]) for c in pv.columns]

    for prefix in ["chi_sum", "Bmax", "old_proxy_sum"]:
        spin_cols = [f"{prefix}__ph-sx", f"{prefix}__ph-sy", f"{prefix}__ph-sz"]
        have = [c for c in spin_cols if c in pv.columns]
        if have:
            pv[f"{prefix}__spin_avg"] = pv[have].mean(axis=1)
            pv[f"{prefix}__spin_min"] = pv[have].min(axis=1)
            pv[f"{prefix}__spin_max"] = pv[have].max(axis=1)

    pv = pv.sort_values(["U", "V", "case_name"])
    pv.to_csv(outdir / "nosoc_fixed_cutoff_d6a_compact.csv", index=False)


def run_one_target(target_lambda, max_logdiff):
    tag = f"Lambda_{target_lambda:.6g}".replace(".", "p")
    outdir = OUTDIR_BASE / "nosoc_fixed_cutoff_d6a_strict" / tag
    outdir.mkdir(parents=True, exist_ok=True)

    inv = build_inventory()
    inv.to_csv(outdir / "nosoc_snapshot_inventory.csv", index=False)

    selected, skipped = choose_snapshots_for_target(inv, target_lambda, max_logdiff)

    selected.to_csv(outdir / "selected_snapshots.csv", index=False)
    skipped.to_csv(outdir / "skipped_cases.csv", index=False)

    print("\n" + "=" * 100, flush=True)
    print(f"Target Lambda = {target_lambda}, max_logdiff = {max_logdiff}", flush=True)
    print("=" * 100, flush=True)
    print("n selected:", len(selected), flush=True)
    print("n skipped:", len(skipped), flush=True)

    if len(selected):
        cols = ["U", "V", "case_name", "snapshot_label", "Lambda", "logdiff", "final_lambda"]
        print("\nSelected snapshots:", flush=True)
        print(selected[cols].to_string(index=False), flush=True)

    if len(skipped):
        print("\nSkipped cases:", flush=True)
        print(skipped.to_string(index=False), flush=True)

    all_rows = []
    errors = []

    for _, row in selected.iterrows():
        try:
            all_rows.extend(compute_snapshot(row))
            write_outputs(outdir, all_rows, errors)
        except Exception as e:
            print(f"[ERROR] {row['case_name']} {row['snapshot_label']}: {e}", flush=True)
            errors.append({
                "case_name": row["case_name"],
                "snapshot_label": row["snapshot_label"],
                "snapshot_dir": row["snapshot_dir"],
                "error": repr(e),
                "traceback": traceback.format_exc(),
            })
            write_outputs(outdir, all_rows, errors)

    write_outputs(outdir, all_rows, errors)

    print("\n=== Done target ===", flush=True)
    print("outdir:", outdir, flush=True)
    print("rows all_q:", len(all_rows), flush=True)
    print("errors:", len(errors), flush=True)

    compact = outdir / "nosoc_fixed_cutoff_d6a_compact.csv"
    if compact.exists:
        df = pd.read_csv(compact)
        print("\nCompact preview:", flush=True)
        cols = [
            "U", "V", "snapshot_label", "Lambda", "logdiff_to_target",
            "chi_sum__ph-rho", "chi_sum__spin_avg",
            "Bmax__ph-rho", "Bmax__spin_max",
        ]
        cols = [c for c in cols if c in df.columns]
        print(df[cols].to_string(index=False), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="+", type=float, default=[0.5, 0.3])
    ap.add_argument("--max-logdiff-0p5", type=float, default=0.10)
    ap.add_argument("--max-logdiff-0p3", type=float, default=0.15)
    ap.add_argument("--max-logdiff-default", type=float, default=0.15)
    args = ap.parse_args()

    for target in args.targets:
        if abs(target - 0.5) < 1e-12:
            tol = args.max_logdiff_0p5
        elif abs(target - 0.3) < 1e-12:
            tol = args.max_logdiff_0p3
        else:
            tol = args.max_logdiff_default

        run_one_target(target, tol)


if __name__ == "__main__":
    main()
