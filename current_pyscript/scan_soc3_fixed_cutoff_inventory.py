#!/usr/bin/env python3
from pathlib import Path
import json
import re
import math
import numpy as np
import pandas as pd

RUNS = Path.cwd()
OUTDIR = RUNS / "current_diagnosis" / "soc3_fixed_cutoff_inventory"
OUTDIR.mkdir(parents=True, exist_ok=True)

ROOTS = [
    ("nosoc", "no soc"),
    ("lambda_nn_0p1", "lnn0p1"),
    ("lambda_nnn_0p1", "lnnn0p1"),
]

TARGETS = [5.0, 3.0, 2.0, 1.5, 1.0, 0.7, 0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
TOLS = [0.10, 0.15, 0.20, 0.30, 0.40, 0.50]


def safe_json_load(p):
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def recursive_find_lambda(obj):
    """
    Find cutoff Lambda in metadata.json.
    Prefer flow_report.last_row.Lambda.
    """
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

    # Best known key from previous metadata.
    for k, v in hits:
        if k.lower().endswith("flow_report.last_row.lambda"):
            return v, k

    # Then any key ending in .Lambda
    for k, v in hits:
        if k.lower().endswith(".lambda"):
            return v, k

    # Fallback to first positive candidate.
    if hits:
        return hits[0][1], hits[0][0]

    return np.nan, ""


def parse_num(tok):
    return float(tok.replace("p", "."))


def parse_params(case_name, root_label):
    out = {
        "U": np.nan,
        "V": np.nan,
        "lambda_nn": 0.0,
        "lambda_nnn": 0.0,
    }

    m = re.search(r"_U([0-9]+(?:p[0-9]+)?)", case_name)
    if m:
        out["U"] = parse_num(m.group(1))

    # For PRR V1 scan, folder normally contains _V0p5, _V1p5, etc.
    # Avoid matching V2 if future folders appear.
    m = re.search(r"_V(?![0-9])([0-9]+(?:p[0-9]+)?)", case_name)
    if m:
        out["V"] = parse_num(m.group(1))
    else:
        # Fallback: sometimes simply _V0p5 still works; keep another permissive search.
        m = re.search(r"_V([0-9]+(?:p[0-9]+)?)", case_name)
        if m:
            out["V"] = parse_num(m.group(1))

    if root_label == "lambda_nn_0p1":
        out["lambda_nn"] = 0.1
        out["lambda_nnn"] = 0.0
    elif root_label == "lambda_nnn_0p1":
        out["lambda_nn"] = 0.0
        out["lambda_nnn"] = 0.1
    else:
        out["lambda_nn"] = 0.0
        out["lambda_nnn"] = 0.0

    return out


def snapshot_iter(label):
    if label == "final":
        return 10**9
    m = re.match(r"iter_(\d+)$", label)
    if m:
        return int(m.group(1))
    return -1


def case_root_from_snapshot(snap_dir):
    """
    snap_dir can be:
      case/snapshots/iter_xxxx
      case/snapshots/final
      case/run/snapshots/iter_xxxx
      case/run/snapshots/final
    """
    if snap_dir.parent.name != "snapshots":
        return snap_dir.parent

    case_or_run = snap_dir.parent.parent
    if case_or_run.name == "run":
        return case_or_run.parent
    return case_or_run


def find_snapshots(root_path):
    root = RUNS / root_path
    snaps = []

    if not root.exists():
        return snaps

    for odvg in sorted(root.glob("**/snapshots/*/O.dvg")):
        snap_dir = odvg.parent
        if snap_dir.name.startswith("iter_") or snap_dir.name == "final":
            snaps.append(snap_dir)

    return snaps


def extract_lambda(snap_dir):
    """
    Try snapshot metadata first, then nearby metadata.
    """
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


def inventory():
    rows = []

    for root_label, root_path in ROOTS:
        snaps = find_snapshots(root_path)

        for snap_dir in snaps:
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
                "root_label": root_label,
                "root_path": root_path,
                "case_name": case_name,
                "case_root": str(case_root),
                "snapshot_label": label,
                "snapshot_iter": it,
                "snapshot_dir": str(snap_dir),
                "Lambda": lam,
                "Lambda_source_file": src_file,
                "Lambda_source_key": src_key,
                "O_dvg_size_GB": size_gb,
                **parse_params(case_name, root_label),
            })

    df = pd.DataFrame(rows)

    if len(df):
        df = df.sort_values(["root_label", "U", "V", "case_name", "snapshot_iter"])

    return df


def build_case_summary(df):
    rows = []

    for keys, g in df.groupby(["root_label", "case_name", "U", "V"], dropna=False):
        root_label, case_name, U, V = keys
        gg = g.sort_values("snapshot_iter")

        final = gg[gg["snapshot_label"] == "final"]
        if len(final):
            final_lambda = float(final["Lambda"].iloc[-1])
        else:
            final_lambda = np.nan

        rows.append({
            "root_label": root_label,
            "case_name": case_name,
            "U": U,
            "V": V,
            "n_snapshots": len(gg),
            "first_snapshot": gg["snapshot_label"].iloc[0],
            "last_snapshot": gg["snapshot_label"].iloc[-1],
            "first_Lambda": float(gg["Lambda"].iloc[0]) if np.isfinite(gg["Lambda"].iloc[0]) else np.nan,
            "last_Lambda": float(gg["Lambda"].iloc[-1]) if np.isfinite(gg["Lambda"].iloc[-1]) else np.nan,
            "final_Lambda": final_lambda,
            "min_Lambda": float(np.nanmin(gg["Lambda"])) if np.any(np.isfinite(gg["Lambda"])) else np.nan,
            "max_Lambda": float(np.nanmax(gg["Lambda"])) if np.any(np.isfinite(gg["Lambda"])) else np.nan,
            "total_O_size_GB": float(np.nansum(gg["O_dvg_size_GB"])),
            "max_O_size_GB": float(np.nanmax(gg["O_dvg_size_GB"])) if np.any(np.isfinite(gg["O_dvg_size_GB"])) else np.nan,
            "snapshot_labels": ";".join(map(str, gg["snapshot_label"].tolist())),
            "Lambda_list": ";".join("NA" if not np.isfinite(x) else f"{x:.8g}" for x in gg["Lambda"].tolist()),
        })

    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(["root_label", "U", "V", "case_name"])
    return out


def choose_for_target(df, target):
    """
    Choose the closest snapshot in log Lambda for each case.
    """
    rows = []

    good = df[np.isfinite(df["Lambda"]) & (df["Lambda"] > 0)].copy()

    for keys, g in good.groupby(["root_label", "case_name", "U", "V"], dropna=False):
        root_label, case_name, U, V = keys
        gg = g.copy()
        gg["logdiff"] = np.abs(np.log(gg["Lambda"] / target))

        # Avoid final if an equally close iter snapshot exists.
        gg["is_final"] = (gg["snapshot_label"] == "final").astype(int)

        best = gg.sort_values(["logdiff", "is_final", "snapshot_iter"]).iloc[0].to_dict()
        best["target_Lambda"] = target

        # Is this target below final cutoff? If yes, the case cannot truly reach target.
        final = gg[gg["snapshot_label"] == "final"]
        if len(final):
            final_Lambda = float(final["Lambda"].iloc[-1])
        else:
            final_Lambda = np.nan

        best["final_Lambda"] = final_Lambda
        best["target_above_final_cutoff"] = bool(np.isfinite(final_Lambda) and target >= final_Lambda)
        rows.append(best)

    sel = pd.DataFrame(rows)
    if len(sel):
        sel = sel.sort_values(["root_label", "U", "V", "case_name"])
    return sel


def coverage_analysis(df):
    all_case_count = df.groupby(["root_label", "case_name", "U", "V"], dropna=False).ngroups

    cov_rows = []
    selection_rows = []

    for target in TARGETS:
        sel = choose_for_target(df, target)
        if len(sel) == 0:
            continue

        for _, r in sel.iterrows():
            selection_rows.append({
                "target_Lambda": target,
                "root_label": r["root_label"],
                "case_name": r["case_name"],
                "U": r["U"],
                "V": r["V"],
                "snapshot_label": r["snapshot_label"],
                "snapshot_iter": r["snapshot_iter"],
                "snapshot_Lambda": r["Lambda"],
                "final_Lambda": r["final_Lambda"],
                "logdiff": r["logdiff"],
                "target_above_final_cutoff": r["target_above_final_cutoff"],
                "snapshot_dir": r["snapshot_dir"],
            })

        for tol in TOLS:
            good_tol = sel[sel["logdiff"] <= tol]
            good_reachable = sel[(sel["logdiff"] <= tol) & (sel["target_above_final_cutoff"] == True)]

            cov_rows.append({
                "target_Lambda": target,
                "max_logdiff": tol,
                "n_total_cases": all_case_count,
                "n_selected_within_tol": int(len(good_tol)),
                "n_selected_within_tol_and_target_reachable": int(len(good_reachable)),
                "n_skipped_by_tol": int(all_case_count - len(good_tol)),
                "n_skipped_by_tol_or_unreachable": int(all_case_count - len(good_reachable)),
                "max_logdiff_selected": float(good_tol["logdiff"].max()) if len(good_tol) else np.nan,
                "median_logdiff_selected": float(good_tol["logdiff"].median()) if len(good_tol) else np.nan,
            })

            # Per-root counts
            for root_label in sorted(sel["root_label"].unique()):
                ss = sel[sel["root_label"] == root_label]
                rr_tol = ss[ss["logdiff"] <= tol]
                rr_reach = ss[(ss["logdiff"] <= tol) & (ss["target_above_final_cutoff"] == True)]
                cov_rows.append({
                    "target_Lambda": target,
                    "max_logdiff": tol,
                    "root_label": root_label,
                    "n_total_cases": int(len(ss)),
                    "n_selected_within_tol": int(len(rr_tol)),
                    "n_selected_within_tol_and_target_reachable": int(len(rr_reach)),
                    "n_skipped_by_tol": int(len(ss) - len(rr_tol)),
                    "n_skipped_by_tol_or_unreachable": int(len(ss) - len(rr_reach)),
                    "max_logdiff_selected": float(rr_tol["logdiff"].max()) if len(rr_tol) else np.nan,
                    "median_logdiff_selected": float(rr_tol["logdiff"].median()) if len(rr_tol) else np.nan,
                })

    cov = pd.DataFrame(cov_rows)
    selections = pd.DataFrame(selection_rows)

    return cov, selections


def print_human_summary(df, case_summary, cov, selections):
    print("=== SOC3 fixed-cutoff inventory ===")
    print("Roots:")
    for root_label, root_path in ROOTS:
        print(f"  {root_label:16s} -> {root_path}")
    print()

    print("Total O.dvg snapshots:", len(df))
    print("Total cases:", case_summary["case_name"].nunique() if len(case_summary) else 0)
    print()

    if len(case_summary):
        print("Cases by root:")
        print(case_summary.groupby("root_label")["case_name"].nunique().to_string())
        print()

        print("Final Lambda summary by root:")
        print(case_summary.groupby("root_label")["final_Lambda"].describe().to_string())
        print()

        print("Case summary preview:")
        cols = ["root_label", "U", "V", "n_snapshots", "first_snapshot", "last_snapshot", "first_Lambda", "final_Lambda", "min_Lambda", "max_Lambda", "total_O_size_GB"]
        print(case_summary[cols].to_string(index=False))
        print()

    print("Coverage summary for all roots together:")
    cc = cov[pd.isna(cov.get("root_label", np.nan))].copy() if "root_label" in cov.columns else cov.copy()
    if len(cc):
        cols = [
            "target_Lambda", "max_logdiff",
            "n_selected_within_tol", "n_selected_within_tol_and_target_reachable",
            "n_skipped_by_tol", "n_skipped_by_tol_or_unreachable",
            "max_logdiff_selected",
        ]
        print(cc[cols].to_string(index=False))
        print()

    print("Suggested targets to inspect first:")
    print("  1) lowest target with 33/33 reachable at max_logdiff <= 0.30 or 0.40")
    print("  2) lower target with partial but balanced coverage across three roots")
    print()

    # Print best candidates by all-case reachable coverage.
    if len(cc):
        best = cc.sort_values(
            ["n_selected_within_tol_and_target_reachable", "target_Lambda", "max_logdiff"],
            ascending=[False, True, True],
        ).head(20)
        print("Top coverage candidates:")
        print(best[[
            "target_Lambda", "max_logdiff",
            "n_selected_within_tol",
            "n_selected_within_tol_and_target_reachable",
            "n_skipped_by_tol_or_unreachable",
            "max_logdiff_selected",
        ]].to_string(index=False))


def main():
    df = inventory()
    df.to_csv(OUTDIR / "soc3_snapshot_inventory.csv", index=False)

    case_summary = build_case_summary(df)
    case_summary.to_csv(OUTDIR / "soc3_case_summary.csv", index=False)

    cov, selections = coverage_analysis(df)
    cov.to_csv(OUTDIR / "soc3_target_coverage_summary.csv", index=False)
    selections.to_csv(OUTDIR / "soc3_target_selections_all.csv", index=False)

    report = {
        "roots": ROOTS,
        "n_snapshots": int(len(df)),
        "n_cases": int(case_summary["case_name"].nunique()) if len(case_summary) else 0,
        "outputs": {
            "inventory": str(OUTDIR / "soc3_snapshot_inventory.csv"),
            "case_summary": str(OUTDIR / "soc3_case_summary.csv"),
            "coverage": str(OUTDIR / "soc3_target_coverage_summary.csv"),
            "selections": str(OUTDIR / "soc3_target_selections_all.csv"),
        },
        "note": "Choose cutoff from coverage table. target_above_final_cutoff=True means the target is reachable before the flow stops."
    }
    (OUTDIR / "inventory_report.json").write_text(json.dumps(report, indent=2))

    print_human_summary(df, case_summary, cov, selections)
    print()
    print("Wrote:", OUTDIR)


if __name__ == "__main__":
    main()
