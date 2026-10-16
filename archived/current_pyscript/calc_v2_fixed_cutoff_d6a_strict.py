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

if str(PYSCRIPT) not in sys.path:
    sys.path.insert(0, str(PYSCRIPT))



ROOT_LABEL = "v2_nosoc"
ROOT_PATH = "no soc -v2"

SECTORS = ["ph-rho", "ph-sx", "ph-sy", "ph-sz"]
QLABS = ["M1", "M2", "M3"]

def load_susceptibility_helpers():
    """
    Heavy imports are delayed until actual susceptibility calculation.
    This lets --inventory-only run without TUfRG_instability.
    """
    import sys
    from pathlib import Path

    candidates = [
        RUNS,
        RUNS / "current_pyscript",
        RUNS / "current_pyscript_appendixmodel",
        RUNS / "scripts",
        RUNS.parent,
    ]

    for c in candidates:
        if c.exists() and str(c) not in sys.path:
            sys.path.insert(0, str(c))

    import TUfRG_instability as inst
    from calc_fig6a_d6a_corrected_strict_precritical import make_d6a_sources_on_O
    from calc_fig5_corrected_strict_charge_precritical import strict_chi_for_eta

    return inst, make_d6a_sources_on_O, strict_chi_for_eta




def parse_num(tok):
    return float(tok.replace("p", "."))


def parse_params(case_name):
    """
    Expected examples:
      v2_nosoc_nk24_nkf12_U0p0_V10p0_V20p5_pureV2
      v2_nosoc_nk24_nkf12_U0p0_V10p5_V21p0_PI_to_current
      v2_nosoc_nk24_nkf12_U3p0_V10p5_V21p5_CBO_internal
    """
    out = {
        "U": np.nan,
        "V1": np.nan,
        "V2": np.nan,
        "trajectory": "",
    }

    m = re.search(r"_U([0-9]+(?:p[0-9]+)?)", case_name)
    if m:
        out["U"] = parse_num(m.group(1))

    m = re.search(r"_V1([0-9]+(?:p[0-9]+)?)", case_name)
    if m:
        out["V1"] = parse_num(m.group(1))

    m = re.search(r"_V2([0-9]+(?:p[0-9]+)?)", case_name)
    if m:
        out["V2"] = parse_num(m.group(1))

    # Everything after V2xxx_ is a trajectory label.
    m = re.search(r"_V2[0-9]+(?:p[0-9]+)?_(.+)$", case_name)
    if m:
        out["trajectory"] = m.group(1)

    return out


def safe_json_load(p):
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def recursive_find_lambda(obj):
    """
    Find likely cutoff/Lambda fields in metadata.json.
    Highest priority is flow_report.last_row.Lambda.
    """
    hits = []

    def walk(x, path=""):
        if isinstance(x, dict):
            for k, v in x.items():
                pp = f"{path}.{k}" if path else str(k)
                if str(k).lower() in ["lambda", "cutoff", "scale"]:
                    try:
                        hits.append((pp, float(v)))
                    except Exception:
                        pass
                walk(v, pp)
        elif isinstance(x, list):
            for i, v in enumerate(x[:200]):
                walk(v, f"{path}[{i}]")

    walk(obj)

    # Best known key from previous inventory.
    for k, v in hits:
        if k.lower().endswith("flow_report.last_row.lambda"):
            return v, k

    # Otherwise prefer any key ending in .Lambda
    for k, v in hits:
        if k.lower().endswith(".lambda"):
            return v, k

    # Fallback
    if hits:
        return hits[0][1], hits[0][0]

    return np.nan, ""


def snapshot_label(snap_dir):
    return snap_dir.name


def snapshot_iter(label):
    m = re.match(r"iter_(\d+)$", label)
    if m:
        return int(m.group(1))
    if label == "final":
        return 10**9
    return -1


def case_root_from_snapshot(snap_dir):
    """
    snap_dir = case/snapshots/iter_xxxx or case/run/snapshots/iter_xxxx
    """
    if snap_dir.parent.name != "snapshots":
        return snap_dir.parent

    case_or_run = snap_dir.parent.parent
    if case_or_run.name == "run":
        return case_or_run.parent
    return case_or_run


def find_snapshots(root):
    """
    Find all snapshots containing O.dvg.
    """
    snaps = []
    for odvg in sorted(root.glob("**/snapshots/*/O.dvg")):
        snap_dir = odvg.parent
        if snap_dir.name.startswith("iter_") or snap_dir.name == "final":
            snaps.append(snap_dir)
    return snaps


def extract_cutoff_for_snapshot(snap_dir):
    """
    Prefer snapshot/metadata.json.
    Fallback to nearby metadata.json.
    """
    candidates = [
        snap_dir / "metadata.json",
        snap_dir.parent / "metadata.json",
        case_root_from_snapshot(snap_dir) / "metadata.json",
        case_root_from_snapshot(snap_dir) / "run" / "metadata.json",
    ]

    for p in candidates:
        if p.exists():
            j = safe_json_load(p)
            if j is not None:
                lam, key = recursive_find_lambda(j)
                if np.isfinite(lam) and lam > 0:
                    return lam, str(p), key

    return np.nan, "", ""


def build_inventory(root):
    rows = []
    for snap_dir in find_snapshots(root):
        case_root = case_root_from_snapshot(snap_dir)
        case_name = case_root.name
        label = snapshot_label(snap_dir)
        it = snapshot_iter(label)
        lam, meta_file, meta_key = extract_cutoff_for_snapshot(snap_dir)
        odvg = snap_dir / "O.dvg"
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
            "O_dvg_size_GB": size_gb,
            "Lambda": lam,
            "Lambda_source_file": meta_file,
            "Lambda_source_key": meta_key,
            **parse_params(case_name),
        })

    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values(["trajectory", "U", "V1", "V2", "case_name", "snapshot_iter"])
    return df


def choose_fixed_cutoff_snapshots(inv, target_lambda, max_logdiff):
    """
    For each case, choose snapshot closest to target_lambda in log Lambda.
    Only keep cases within max_logdiff.
    """
    rows = []
    skipped = []

    good = inv[np.isfinite(inv["Lambda"]) & (inv["Lambda"] > 0)].copy()

    for case_name, g in good.groupby("case_name", dropna=False):
        g = g.copy()
        g["logdiff"] = np.abs(np.log(g["Lambda"] / target_lambda))
        best = g.sort_values(["logdiff", "snapshot_iter"]).iloc[0].to_dict()

        if best["logdiff"] <= max_logdiff:
            rows.append(best)
        else:
            skipped.append({
                "case_name": case_name,
                "reason": "no snapshot close enough to target_lambda",
                "target_lambda": target_lambda,
                "best_lambda": best["Lambda"],
                "best_snapshot_label": best["snapshot_label"],
                "best_logdiff": best["logdiff"],
                "max_logdiff": max_logdiff,
            })

    selected = pd.DataFrame(rows)
    skipped = pd.DataFrame(skipped)

    if len(selected):
        selected = selected.sort_values(["trajectory", "U", "V1", "V2", "case_name"])

    return selected, skipped


def compute_snapshot(row):
    inst, make_d6a_sources_on_O, strict_chi_for_eta = load_susceptibility_helpers()

    snap_dir = Path(row["snapshot_dir"])
    print(
        f"[compute] {row['case_name']} {row['snapshot_label']} "
        f"Lambda={row['Lambda']:.8g}",
        flush=True
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
                "pattern": "fig6a_d6a_corrected",
                "q_label": qlab,
                "iq": iq,
                "sector": spec["sector"],
                "eta_norm2_q": eta_norm2,
                "U": row["U"],
                "V1": row["V1"],
                "V2": row["V2"],
                "trajectory": row["trajectory"],
                **source_diag,
                **d,
            })

    return out_rows


def write_outputs(outdir, all_rows, errors):
    all_df = pd.DataFrame(all_rows)

    # Compatibility patch:
    # some strict_chi_for_eta versions output B_min_eval/B_max_eval
    # but not B_abs_max_eval.
    if len(all_df) and "B_abs_max_eval" not in all_df.columns:
        if "B_min_eval" in all_df.columns and "B_max_eval" in all_df.columns:
            all_df["B_abs_max_eval"] = all_df[["B_min_eval", "B_max_eval"]].abs().max(axis=1)
        elif "B_max_eval" in all_df.columns:
            all_df["B_abs_max_eval"] = all_df["B_max_eval"].abs()
        else:
            all_df["B_abs_max_eval"] = float("nan")

    all_df.to_csv(outdir / "v2_fixed_cutoff_d6a_all_q.csv", index=False)
    (outdir / "errors.json").write_text(json.dumps(errors, indent=2))

    if len(all_df) == 0:
        return

    group_keys = [
        "root_label", "case_name", "U", "V1", "V2", "trajectory",
        "snapshot_label", "snapshot_iter", "Lambda", "target_lambda",
        "logdiff_to_target", "pattern", "sector"
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

    summed.to_csv(outdir / "v2_fixed_cutoff_d6a_summed.csv", index=False)

    pv = summed.pivot_table(
        index=[
            "case_name", "U", "V1", "V2", "trajectory",
            "snapshot_label", "snapshot_iter", "Lambda", "target_lambda",
            "logdiff_to_target", "pattern"
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

    pv = pv.sort_values(["trajectory", "U", "V1", "V2", "case_name"])
    pv.to_csv(outdir / "v2_fixed_cutoff_d6a_compact.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-lambda", type=float, default=None,
                    help="Target cutoff Lambda for fixed-cutoff calculation.")
    ap.add_argument("--max-logdiff", type=float, default=0.20,
                    help="Maximum |log(Lambda_snapshot / target_lambda)| allowed. 0.20 means about 22 percent.")
    ap.add_argument("--inventory-only", action="store_true",
                    help="Only scan snapshots and do not compute susceptibility.")
    args = ap.parse_args()

    root = RUNS / ROOT_PATH
    if not root.exists():
        raise FileNotFoundError(f"Cannot find root: {root}")

    tag = "inventory_only" if args.target_lambda is None else f"Lambda_{args.target_lambda:.6g}".replace(".", "p")
    outdir = OUTDIR_BASE / "v2_fixed_cutoff_d6a_strict" / tag
    outdir.mkdir(parents=True, exist_ok=True)

    inv = build_inventory(root)
    inv.to_csv(outdir / "v2_snapshot_inventory.csv", index=False)

    print("=== V2 snapshot inventory ===", flush=True)
    print("root:", root, flush=True)
    print("n snapshots:", len(inv), flush=True)
    if len(inv):
        print("n cases:", inv["case_name"].nunique(), flush=True)
        print("Lambda range:", inv["Lambda"].min(), inv["Lambda"].max(), flush=True)
        print("\nPreview:", flush=True)
        cols = ["case_name", "U", "V1", "V2", "trajectory", "snapshot_label", "Lambda", "O_dvg_size_GB"]
        print(inv[cols].head(80).to_string(index=False), flush=True)

        # Suggest candidate target cutoffs by quantiles.
        good = inv[np.isfinite(inv["Lambda"]) & (inv["Lambda"] > 0)].copy()
        if len(good):
            qs = good["Lambda"].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).to_dict()
            (outdir / "lambda_quantiles.json").write_text(json.dumps({str(k): float(v) for k, v in qs.items()}, indent=2))
            print("\nLambda quantiles:", qs, flush=True)

    if args.inventory_only or args.target_lambda is None:
        print("\nInventory-only mode. Choose a target Lambda and rerun with --target-lambda.", flush=True)
        return

    selected, skipped = choose_fixed_cutoff_snapshots(inv, args.target_lambda, args.max_logdiff)
    selected["target_lambda"] = args.target_lambda

    selected.to_csv(outdir / "selected_snapshots.csv", index=False)
    skipped.to_csv(outdir / "skipped_cases.csv", index=False)

    print("\n=== Selected snapshots ===", flush=True)
    if len(selected):
        cols = ["case_name", "U", "V1", "V2", "trajectory", "snapshot_label", "Lambda", "logdiff"]
        print(selected[cols].to_string(index=False), flush=True)
    print("\nSkipped cases:", len(skipped), flush=True)
    if len(skipped):
        print(skipped.head(60).to_string(index=False), flush=True)

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

    print("\n=== Done ===", flush=True)
    print("outdir:", outdir, flush=True)
    print("rows all_q:", len(all_rows), flush=True)
    print("errors:", len(errors), flush=True)

    compact = outdir / "v2_fixed_cutoff_d6a_compact.csv"
    if compact.exists():
        df = pd.read_csv(compact)
        print("\n=== Compact preview ===", flush=True)
        cols = [
            "U", "V1", "V2", "trajectory", "snapshot_label", "Lambda",
            "chi_sum__ph-rho", "chi_sum__spin_avg",
            "Bmax__ph-rho", "Bmax__spin_max",
        ]
        cols = [c for c in cols if c in df.columns]
        print(df[cols].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
