#!/usr/bin/env python3
import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from TUfRG_formfactor_from_modes import load_output
import TUfRG_model as km


def parse_UV_from_case(name: str):
    mU = re.search(r"_U([0-9p]+)_", name)
    mV = re.search(r"_V([0-9p]+)_", name)
    U = float(mU.group(1).replace("p", ".")) if mU else np.nan
    V = float(mV.group(1).replace("p", ".")) if mV else np.nan
    return U, V


def read_ff_map(odvg_path: Path):
    """
    Read TU form-factor mapping from O.dvg using the same divERGe loader
    used by the existing diagnosis scripts.
    """
    O = load_output(odvg_path)

    R = np.asarray(getattr(O, "mi_to_R"), dtype=int)
    ofrom = np.asarray(getattr(O, "mi_to_ofrom"), dtype=int).reshape(-1)
    oto = np.asarray(getattr(O, "mi_to_oto"), dtype=int).reshape(-1)

    # Convert possible 1-based orbital labels to 0-based.
    # If already 0-based, this does nothing.
    if ofrom.min() >= 1 and oto.min() >= 1:
        ofrom = ofrom - 1
        oto = oto - 1

    if R.ndim == 1:
        R = R.reshape(-1, 3)

    if len(R) != len(ofrom) or len(R) != len(oto):
        raise RuntimeError(
            f"ff map length mismatch: R={len(R)}, ofrom={len(ofrom)}, oto={len(oto)}"
        )

    return R, ofrom, oto


def classify_components(R, ofrom, oto, tol=1e-5):
    """
    Classify each TU form-factor component by physical distance shell
    using the official TUfRG_model geometry.

    This must match the model construction:
        lattice, pos = km.kagome_lattice_and_positions()
        dr = km.displacement_vector(lattice, pos, Rx, Ry, ofrom, oto)
    """
    lattice, pos = km.kagome_lattice_and_positions()

    nff = len(ofrom)
    deltas = np.zeros((nff, 3), dtype=float)
    dist = np.zeros(nff, dtype=float)

    for i in range(nff):
        aa = int(ofrom[i])
        bb = int(oto[i])
        dr = km.displacement_vector(
            lattice, pos,
            int(R[i, 0]), int(R[i, 1]),
            aa, bb
        )
        deltas[i] = dr
        dist[i] = np.linalg.norm(dr[:2])

    nonzero = dist[dist > tol]
    if len(nonzero) == 0:
        d_nn = 1.0
    else:
        d_nn = nonzero.min()

    ratio = dist / d_nn

    shell = []
    for d, rr, aa, bb in zip(dist, ratio, ofrom, oto):
        if d <= tol and aa == bb:
            shell.append("onsite")
        elif abs(rr - 1.0) <= 1e-4:
            shell.append("nn")
        elif abs(rr - np.sqrt(3.0)) <= 1e-4:
            shell.append("sqrt3")
        elif abs(rr - 2.0) <= 1e-4:
            shell.append("dist2")
        else:
            shell.append("other")

    sub_type = np.where(ofrom == oto, "same_sub", "inter_sub")

    comp = pd.DataFrame({
        "component_index": np.arange(nff),
        "R0": R[:, 0],
        "R1": R[:, 1],
        "R2": R[:, 2] if R.shape[1] > 2 else 0,
        "ofrom": ofrom,
        "oto": oto,
        "dx": deltas[:, 0],
        "dy": deltas[:, 1],
        "distance": dist,
        "distance_over_nn": ratio,
        "shell": shell,
        "sublattice_type": sub_type,
    })

    return comp, d_nn


def shell_stats_for_vector(v, comp):
    """
    Compute total and per-component average weights by shell and sublattice type.
    """
    v = np.asarray(v).reshape(-1)
    abs2 = np.abs(v) ** 2
    total = float(abs2.sum())
    if total <= 0:
        total = np.nan

    result = {}

    shell_order = ["onsite", "nn", "sqrt3", "dist2", "other"]
    for sh in shell_order:
        idx = (comp["shell"].values == sh)
        count = int(idx.sum())
        s = float(abs2[idx].sum()) if count else 0.0
        result[f"{sh}_count"] = count
        result[f"{sh}_total_weight"] = s / total if total == total else np.nan
        result[f"{sh}_avg_amp2"] = s / count if count else 0.0
        result[f"{sh}_avg_amp2_norm"] = (s / count) / total if count and total == total else np.nan

    for st in ["same_sub", "inter_sub"]:
        idx = (comp["sublattice_type"].values == st)
        count = int(idx.sum())
        s = float(abs2[idx].sum()) if count else 0.0
        result[f"{st}_count"] = count
        result[f"{st}_total_weight"] = s / total if total == total else np.nan
        result[f"{st}_avg_amp2"] = s / count if count else 0.0
        result[f"{st}_avg_amp2_norm"] = (s / count) / total if count and total == total else np.nan

    # More refined combined categories
    for sh in shell_order:
        for st in ["same_sub", "inter_sub"]:
            idx = (comp["shell"].values == sh) & (comp["sublattice_type"].values == st)
            count = int(idx.sum())
            s = float(abs2[idx].sum()) if count else 0.0
            key = f"{sh}_{st}"
            result[f"{key}_count"] = count
            result[f"{key}_total_weight"] = s / total if total == total else np.nan
            result[f"{key}_avg_amp2_norm"] = (s / count) / total if count and total == total else np.nan

    # Dominant categories
    shell_weights = {sh: result[f"{sh}_total_weight"] for sh in shell_order}
    shell_avg = {sh: result[f"{sh}_avg_amp2_norm"] for sh in shell_order}

    result["dominant_shell_total"] = max(shell_weights, key=lambda k: -1 if np.isnan(shell_weights[k]) else shell_weights[k])
    result["dominant_shell_avg"] = max(shell_avg, key=lambda k: -1 if np.isnan(shell_avg[k]) else shell_avg[k])

    sub_weights = {st: result[f"{st}_total_weight"] for st in ["same_sub", "inter_sub"]}
    result["dominant_sublattice_total"] = max(sub_weights, key=sub_weights.get)

    return result


def top_components_for_vector(v, comp, top_n=30):
    v = np.asarray(v).reshape(-1)
    abs2 = np.abs(v) ** 2
    total = float(abs2.sum())
    if total <= 0:
        total = np.nan

    order = np.argsort(abs2)[::-1][:top_n]
    rows = []
    for rank, i in enumerate(order, start=1):
        c = comp.iloc[int(i)]
        rows.append({
            "rank": rank,
            "component_index": int(i),
            "abs2": float(abs2[i]),
            "weight": float(abs2[i] / total) if total == total else np.nan,
            "Re": float(np.real(v[i])),
            "Im": float(np.imag(v[i])),
            "R0": int(c["R0"]),
            "R1": int(c["R1"]),
            "R2": int(c["R2"]),
            "ofrom": int(c["ofrom"]),
            "oto": int(c["oto"]),
            "dx": float(c["dx"]),
            "dy": float(c["dy"]),
            "distance": float(c["distance"]),
            "distance_over_nn": float(c["distance_over_nn"]),
            "shell": c["shell"],
            "sublattice_type": c["sublattice_type"],
        })
    return rows


def mode_priority_sector(row):
    """
    Mark modes important for phase diagram.
    """
    q = row.get("q_label", "")
    sec = row.get("sector", "")
    if q == "Gamma" and sec in ["pp-singlet", "ph-sz", "ph-rho-A1proj", "ph-rho"]:
        return True
    if q in ["M1", "M2", "M3"] and sec in ["ph-rho", "ph-sx", "ph-sy", "ph-sz"]:
        return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="km_scan_lnn0p1_nk24_nkf12_reduced_REAL")
    ap.add_argument("--out", default="lnn0p1_nk24_REAL_shell_diagnosis")
    ap.add_argument("--top-n", type=int, default=40)
    ap.add_argument("--all-modes", action="store_true",
                    help="Process all modes in formfactor_summary.csv instead of phase-relevant modes only.")
    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.out)
    out.mkdir(exist_ok=True, parents=True)

    all_rows = []
    top_rows = []
    component_maps_written = 0

    cases = sorted([p for p in root.iterdir() if p.is_dir()])
    if not cases:
        raise SystemExit(f"No case directories found under {root}")

    for case in cases:
        U, V1 = parse_UV_from_case(case.name)

        snap = case / "snapshots" / "final"
        summary_f = snap / "formfactor" / "formfactor_summary.csv"
        modes_f = snap / "instability" / "instability_modes.npz"

        odvg_candidates = [
            snap / "O.dvg",
            case / "O.dvg",
        ]
        odvg = next((p for p in odvg_candidates if p.exists()), None)

        if not summary_f.exists() or not modes_f.exists() or odvg is None:
            print(f"[SKIP] missing files for {case.name}")
            print("  summary:", summary_f.exists(), summary_f)
            print("  modes:", modes_f.exists(), modes_f)
            print("  odvg:", odvg)
            continue

        print(f"[case] {case.name}")
        print(f"  O.dvg = {odvg}")

        R, ofrom, oto = read_ff_map(odvg)
        comp, d_nn = classify_components(R, ofrom, oto)

        # write one component map per first case, useful for sanity check
        comp_out = out / f"{case.name}_component_shell_map.csv"
        comp.to_csv(comp_out, index=False)
        component_maps_written += 1

        summ = pd.read_csv(summary_f)
        modes = np.load(modes_f, allow_pickle=True)

        for _, row in summ.iterrows():
            if not args.all_modes and not mode_priority_sector(row):
                continue

            mode_key = row.get("mode_key", None)
            if mode_key is None or mode_key not in modes:
                continue

            v = np.asarray(modes[mode_key]).reshape(-1)
            if len(v) != len(comp):
                print(f"  [WARN] mode length mismatch {mode_key}: mode={len(v)}, nff={len(comp)}")
                continue

            stats = shell_stats_for_vector(v, comp)

            base = {
                "case": case.name,
                "U": U,
                "V1": V1,
                "mode_key": mode_key,
                "q_label": row.get("q_label", ""),
                "sector": row.get("sector", ""),
                "mode_index": row.get("mode_index", np.nan),
                "strength_svd_KL": row.get("strength_svd_KL", np.nan),
                "old_onsite_weight": row.get("onsite_weight", np.nan),
                "old_nn_weight": row.get("nn_weight", np.nan),
                "old_long_weight": row.get("long_weight", np.nan),
            }

            # keep other useful old columns if present
            for col in [
                "classification", "irrep_label", "lco_score", "cbo_score",
                "sbo_score", "site_A1_overlap", "site_E_overlap"
            ]:
                if col in row:
                    base[col] = row[col]

            base.update(stats)
            all_rows.append(base)

            # top components
            for tr in top_components_for_vector(v, comp, top_n=args.top_n):
                tr.update(base)
                top_rows.append(tr)

    df = pd.DataFrame(all_rows)
    top = pd.DataFrame(top_rows)

    df.to_csv(out / "shell_resolved_formfactor_summary.csv", index=False)
    top.to_csv(out / "shell_resolved_top_components.csv", index=False)

    if len(df):
        # compact phase-relevant summary: choose strongest within common targets per case
        targets = []
        for case_name, g in df.groupby("case"):
            for label, mask in [
                ("Gamma_SC", (g.q_label == "Gamma") & (g.sector == "pp-singlet")),
                ("Gamma_Sz_MAG", (g.q_label == "Gamma") & (g.sector == "ph-sz")),
                ("Gamma_charge_PI", (g.q_label == "Gamma") & (g.sector == "ph-rho-A1proj")),
                ("M_charge", (g.q_label.isin(["M1", "M2", "M3"])) & (g.sector == "ph-rho")),
                ("M_spin", (g.q_label.isin(["M1", "M2", "M3"])) & (g.sector.isin(["ph-sx", "ph-sy", "ph-sz"]))),
            ]:
                gg = g[mask].copy()
                if len(gg):
                    r = gg.sort_values("strength_svd_KL", ascending=False).iloc[0].to_dict()
                    r["target_label"] = label
                    targets.append(r)

        phase = pd.DataFrame(targets)
        phase.to_csv(out / "phase_relevant_shell_summary.csv", index=False)

        print("\n==== phase relevant shell summary ====")
        cols = [
            "U", "V1", "target_label", "q_label", "sector", "strength_svd_KL",
            "onsite_total_weight", "nn_total_weight", "sqrt3_total_weight",
            "dist2_total_weight", "other_total_weight",
            "same_sub_total_weight", "inter_sub_total_weight",
            "onsite_avg_amp2_norm", "nn_avg_amp2_norm",
            "sqrt3_avg_amp2_norm", "dist2_avg_amp2_norm",
            "dominant_shell_total", "dominant_shell_avg",
            "dominant_sublattice_total",
        ]
        cols = [c for c in cols if c in phase.columns]
        print(phase[cols].sort_values(["U", "V1", "target_label"]).to_string(index=False))

    print("\n[DONE]")
    print("Wrote:")
    print(" ", out / "shell_resolved_formfactor_summary.csv")
    print(" ", out / "shell_resolved_top_components.csv")
    print(" ", out / "phase_relevant_shell_summary.csv")
    print(f"Component maps written: {component_maps_written}")


if __name__ == "__main__":
    main()
