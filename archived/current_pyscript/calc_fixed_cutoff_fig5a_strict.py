#!/usr/bin/env python3
from pathlib import Path
import sys, re, json, argparse, traceback
import numpy as np
import pandas as pd

RUNS = Path.cwd()
PYSCRIPT = RUNS / "current_pyscript"
OUTDIR_BASE = RUNS / "current_diagnosis"

for c in [RUNS, PYSCRIPT, RUNS / "current_pyscript_appendixmodel"]:
    if c.exists() and str(c) not in sys.path:
        sys.path.insert(0, str(c))

import TUfRG_instability as inst
from calc_fig5_corrected_strict_charge_precritical import strict_chi_for_eta

SECTORS = ["ph-rho", "ph-sx", "ph-sy", "ph-sz"]
SUB = {"A": 0, "B": 1, "C": 2}

# Correct Fig.5(a): up/down triangles have same flux.
UP_KEYS = [
    ((0, 0), SUB["A"], SUB["B"]),
    ((0, 0), SUB["B"], SUB["C"]),
    ((0, 0), SUB["C"], SUB["A"]),
]
UP_SIGNS = [1, 1, 1]

DOWN_KEYS = [
    ((-1, 0), SUB["B"], SUB["A"]),
    ((0, -1), SUB["C"], SUB["A"]),
    ((1, -1), SUB["C"], SUB["B"]),
]
DOWN_SIGNS = [-1, 1, -1]


def parse_num(tok):
    return float(tok.replace("p", "."))


def parse_params(case_name):
    out = {"U": np.nan, "V": np.nan}
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
                if str(k).lower() in ["lambda", "cutoff", "scale"]:
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
    return int(m.group(1)) if m else -1


def case_root_from_snapshot(snap_dir):
    if snap_dir.parent.name != "snapshots":
        return snap_dir.parent
    case_or_run = snap_dir.parent.parent
    return case_or_run.parent if case_or_run.name == "run" else case_or_run


def find_snapshots(root):
    snaps = []
    for odvg in sorted(root.glob("**/snapshots/*/O.dvg")):
        d = odvg.parent
        if d.name.startswith("iter_") or d.name == "final":
            snaps.append(d)
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
            if j is not None:
                lam, key = recursive_find_lambda(j)
                if np.isfinite(lam) and lam > 0:
                    return lam, str(p), key
    return np.nan, "", ""


def comp_key(R, a, b):
    return ((int(R[0]), int(R[1])), int(a), int(b))


def reverse_key(key):
    R, a, b = key
    return ((-R[0], -R[1]), b, a)


def make_fig5a_source_on_O(O):
    R = np.asarray(O.mi_to_R, dtype=int)
    ofrom = np.asarray(O.mi_to_ofrom, dtype=int)
    oto = np.asarray(O.mi_to_oto, dtype=int)
    n = len(ofrom)

    key_to_idx = {}
    for i in range(n):
        key_to_idx[comp_key(R[i], ofrom[i], oto[i])] = i

    eta = np.zeros(n, dtype=np.complex128)
    amp = 1j / np.sqrt(12.0)

    for key, sgn in list(zip(UP_KEYS, UP_SIGNS)) + list(zip(DOWN_KEYS, DOWN_SIGNS)):
        k = comp_key(*key)
        rk = reverse_key(k)
        if k not in key_to_idx:
            raise KeyError(f"Missing Fig5a key {k}")
        if rk not in key_to_idx:
            raise KeyError(f"Missing Fig5a reverse key {rk}")

        i = key_to_idx[k]
        j = key_to_idx[rk]
        eta[i] += sgn * amp
        eta[j] += -sgn * amp

    # Hermiticity check: eta(reverse)=eta^*
    maxerr = 0.0
    for key, idx in key_to_idx.items():
        rk = reverse_key(key)
        if rk in key_to_idx:
            maxerr = max(maxerr, abs(eta[key_to_idx[rk]] - np.conj(eta[idx])))

    norm = float(np.vdot(eta, eta).real)
    if abs(norm - 1.0) > 1e-10:
        raise RuntimeError(f"Fig5a source norm not 1: {norm}")
    if maxerr > 1e-12:
        raise RuntimeError(f"Fig5a Hermiticity error: {maxerr}")

    return eta, {
        "source_norm2": norm,
        "source_nonzero": int(np.count_nonzero(np.abs(eta) > 1e-14)),
        "source_hermiticity_error": float(maxerr),
        "source_up_signs": str(UP_SIGNS),
        "source_down_signs": str(DOWN_SIGNS),
    }


def get_gamma_iq(data):
    for label in ["G", "Gamma", "Γ"]:
        try:
            return int(inst.q_labels_to_indices(data, [label])[label])
        except Exception:
            pass
    # fallback: q=0 should usually be Gamma
    return 0


def build_inventory(root_path, root_label, lambda_nn, lambda_nnn):
    root = RUNS / root_path
    if not root.exists():
        raise FileNotFoundError(f"Cannot find root: {root}")

    rows = []
    for snap_dir in find_snapshots(root):
        case_root = case_root_from_snapshot(snap_dir)
        case_name = case_root.name
        label = snap_dir.name
        lam, src_file, src_key = extract_lambda(snap_dir)
        odvg = snap_dir / "O.dvg"
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
            "snapshot_iter": snapshot_iter(label),
            "snapshot_dir": str(snap_dir),
            "Lambda": lam,
            "Lambda_source_file": src_file,
            "Lambda_source_key": src_key,
            "O_dvg_size_GB": size_gb,
            "lambda_nn": lambda_nn,
            "lambda_nnn": lambda_nnn,
            **parse_params(case_name),
        })

    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values(["U", "V", "case_name", "snapshot_iter"])
    return df


def choose_snapshots(inv, target_lambda, max_logdiff):
    rows, skipped = [], []
    good = inv[np.isfinite(inv["Lambda"]) & (inv["Lambda"] > 0)].copy()

    for case_name, g in good.groupby("case_name", dropna=False):
        gg = g.copy()
        gg["logdiff"] = np.abs(np.log(gg["Lambda"] / target_lambda))
        gg["is_final"] = (gg["snapshot_label"] == "final").astype(int)
        best = gg.sort_values(["logdiff", "is_final", "snapshot_iter"]).iloc[0].to_dict()

        final = gg[gg["snapshot_label"] == "final"]
        final_lambda = float(final["Lambda"].iloc[-1]) if len(final) else np.nan

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

    return pd.DataFrame(rows), pd.DataFrame(skipped)


def compute_snapshot(row):
    snap_dir = Path(row["snapshot_dir"])
    print(f"[compute] {row['root_label']} U={row['U']} V={row['V']} {row['snapshot_label']} Lambda={row['Lambda']:.8g}", flush=True)

    data = inst.load_case(snap_dir)
    inst.assert_raw_channel_compatibility(data)

    iq = get_gamma_iq(data)
    eta, source_diag = make_fig5a_source_on_O(data.O)

    kernels = inst.build_explicit_kernels(data, iq)
    out = []

    for spec in kernels:
        if spec.get("family") != "ph":
            continue
        if spec.get("sector") not in SECTORS:
            continue

        K = np.asarray(spec["K"], dtype=np.complex128)
        L = np.asarray(spec["L"], dtype=np.complex128)
        d = strict_chi_for_eta(eta, K, L)

        if "B_abs_max_eval" not in d:
            bmin = d.get("B_min_eval", np.nan)
            bmax = d.get("B_max_eval", np.nan)
            d["B_abs_max_eval"] = float(np.nanmax(np.abs([bmin, bmax])))

        out.append({
            "root_label": row["root_label"],
            "root_path": row["root_path"],
            "case_name": row["case_name"],
            "case_root": row["case_root"],
            "snapshot_label": row["snapshot_label"],
            "snapshot_iter": int(row["snapshot_iter"]),
            "snapshot_dir": row["snapshot_dir"],
            "Lambda": float(row["Lambda"]),
            "target_lambda": float(row["target_lambda"]),
            "logdiff_to_target": float(row["logdiff"]),
            "final_lambda": float(row["final_lambda"]),
            "pattern": "fig5a_same_flux_gamma",
            "q_label": "Gamma",
            "iq": iq,
            "sector": spec["sector"],
            "U": row["U"],
            "V": row["V"],
            "lambda_nn": row["lambda_nn"],
            "lambda_nnn": row["lambda_nnn"],
            **source_diag,
            **d,
        })
    return out


def write_outputs(outdir, rows, errors, prefix):
    df = pd.DataFrame(rows)
    df.to_csv(outdir / f"{prefix}_fig5a_all.csv", index=False)
    (outdir / "errors.json").write_text(json.dumps(errors, indent=2))

    if len(df) == 0:
        return

    group_keys = [
        "root_label", "case_name", "U", "V", "lambda_nn", "lambda_nnn",
        "snapshot_label", "snapshot_iter", "Lambda", "target_lambda",
        "logdiff_to_target", "final_lambda", "pattern", "sector"
    ]

    summed = df.groupby(group_keys, dropna=False).agg(
        chi=("chi_strict_real", "first"),
        chi_imag=("chi_strict_imag", "first"),
        Bmax=("B_max_eval", "first"),
        Babsmax=("B_abs_max_eval", "first"),
        old_proxy=("old_proxy_real", "first"),
        source_norm2=("source_norm2", "first"),
        source_nonzero=("source_nonzero", "first"),
        source_hermiticity_error=("source_hermiticity_error", "first"),
        chi0_neg_weight_ratio=("chi0_neg_weight_ratio", "first"),
    ).reset_index()

    summed.to_csv(outdir / f"{prefix}_fig5a_summed.csv", index=False)

    pv = summed.pivot_table(
        index=[
            "root_label", "case_name", "U", "V", "lambda_nn", "lambda_nnn",
            "snapshot_label", "snapshot_iter", "Lambda", "target_lambda",
            "logdiff_to_target", "final_lambda", "pattern"
        ],
        columns="sector",
        values=["chi", "Bmax", "old_proxy"],
        aggfunc="first",
    ).reset_index()

    pv.columns = ["__".join([str(x) for x in c if str(x) != ""]) for c in pv.columns]

    for pref in ["chi", "Bmax", "old_proxy"]:
        spin_cols = [f"{pref}__ph-sx", f"{pref}__ph-sy", f"{pref}__ph-sz"]
        have = [c for c in spin_cols if c in pv.columns]
        if have:
            pv[f"{pref}__spin_avg"] = pv[have].mean(axis=1)
            pv[f"{pref}__spin_min"] = pv[have].min(axis=1)
            pv[f"{pref}__spin_max"] = pv[have].max(axis=1)

    pv = pv.sort_values(["U", "V", "case_name"])
    pv.to_csv(outdir / f"{prefix}_fig5a_compact.csv", index=False)


def run_target(args, target, tol):
    tag = f"Lambda_{target:.6g}".replace(".", "p")
    outdir = OUTDIR_BASE / "fixed_cutoff_fig5a_strict" / args.root_label / tag
    outdir.mkdir(parents=True, exist_ok=True)

    inv = build_inventory(args.root_path, args.root_label, args.lambda_nn, args.lambda_nnn)
    inv.to_csv(outdir / f"{args.root_label}_snapshot_inventory.csv", index=False)

    selected, skipped = choose_snapshots(inv, target, tol)
    selected.to_csv(outdir / "selected_snapshots.csv", index=False)
    skipped.to_csv(outdir / "skipped_cases.csv", index=False)

    print("\n" + "="*100)
    print(f"{args.root_label}: target={target}, tol={tol}")
    print("selected:", len(selected), "skipped:", len(skipped))
    if len(selected):
        print(selected[["U", "V", "case_name", "snapshot_label", "Lambda", "logdiff", "final_lambda"]].to_string(index=False))

    rows, errors = [], []
    for _, row in selected.iterrows():
        try:
            rows.extend(compute_snapshot(row))
            write_outputs(outdir, rows, errors, args.root_label)
        except Exception as e:
            print(f"[ERROR] {row['case_name']} {row['snapshot_label']}: {e}", flush=True)
            errors.append({
                "case_name": row["case_name"],
                "snapshot_label": row["snapshot_label"],
                "snapshot_dir": row["snapshot_dir"],
                "error": repr(e),
                "traceback": traceback.format_exc(),
            })
            write_outputs(outdir, rows, errors, args.root_label)

    write_outputs(outdir, rows, errors, args.root_label)
    print("rows:", len(rows), "errors:", len(errors), "outdir:", outdir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root-label", required=True)
    ap.add_argument("--root-path", required=True)
    ap.add_argument("--lambda-nn", type=float, required=True)
    ap.add_argument("--lambda-nnn", type=float, required=True)
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
        run_target(args, target, tol)


if __name__ == "__main__":
    main()
