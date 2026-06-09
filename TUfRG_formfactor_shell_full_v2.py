#!/usr/bin/env python3
# coding: utf-8
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import TUfRG_model as km
except Exception as e:
    raise RuntimeError("Cannot import TUfRG_model. Run this from the runs directory.") from e


# ---------- basic helpers ----------

def load_output(path: Path):
    import diverge.output as ro
    return ro.read(str(path))


def norm2(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=complex).reshape(-1)
    return float(np.vdot(x, x).real)


def get_mi_arrays(O: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.asarray(O.mi_to_R, dtype=int),
        np.asarray(O.mi_to_ofrom, dtype=int),
        np.asarray(O.mi_to_oto, dtype=int),
    )


def safe_label(x: str) -> str:
    return (
        str(x)
        .replace(" ", "_")
        .replace("-", "m")
        .replace("+", "p")
        .replace(".", "p")
        .replace("/", "_")
        .replace("(", "")
        .replace(")", "")
    )


def ratio_label(ratio: float, tol: float = 1e-5) -> str:
    targets = [
        (0.0, "0"),
        (1.0, "1"),
        (math.sqrt(3.0), "sqrt3"),
        (2.0, "2"),
        (math.sqrt(7.0), "sqrt7"),
        (3.0, "3"),
        (2.0 * math.sqrt(3.0), "2sqrt3"),
        (math.sqrt(13.0), "sqrt13"),
        (4.0, "4"),
        (math.sqrt(19.0), "sqrt19"),
        (math.sqrt(21.0), "sqrt21"),
        (5.0, "5"),
        (3.0 * math.sqrt(3.0), "3sqrt3"),
        (math.sqrt(31.0), "sqrt31"),
    ]
    for val, name in targets:
        if abs(float(ratio) - val) < tol:
            return name
    return "ratio_" + safe_label(f"{ratio:.6f}")


def r_shell_label(rnorm: float, tol: float = 1e-5) -> str:
    targets = [
        (0.0, "0"),
        (1.0, "1"),
        (math.sqrt(3.0), "sqrt3"),
        (2.0, "2"),
        (math.sqrt(7.0), "sqrt7"),
        (3.0, "3"),
    ]
    for val, name in targets:
        if abs(float(rnorm) - val) < tol:
            return name
    return "R_" + safe_label(f"{rnorm:.6f}")


# ---------- geometry reconstruction ----------

def component_geometry(O: Any, tol: float = 1e-5) -> pd.DataFrame:
    R, ofrom, oto = get_mi_arrays(O)
    lattice, pos = km.kagome_lattice_and_positions()

    n = len(ofrom)
    dx = np.zeros(n, dtype=float)
    dy = np.zeros(n, dtype=float)
    dist = np.zeros(n, dtype=float)
    Rdx = np.zeros(n, dtype=float)
    Rdy = np.zeros(n, dtype=float)
    Rnorm = np.zeros(n, dtype=float)

    for i in range(n):
        dr = km.tu_ff_displacement_vector(
            lattice, pos,
            int(R[i, 0]), int(R[i, 1]),
            int(ofrom[i]), int(oto[i]),
        )
        dx[i] = float(dr[0])
        dy[i] = float(dr[1])
        dist[i] = float(np.linalg.norm(dr[:2]))

        rv = int(R[i, 0]) * lattice[0, :2] + int(R[i, 1]) * lattice[1, :2]
        if R.shape[1] > 2:
            rv = rv + int(R[i, 2]) * lattice[2, :2]
        Rdx[i] = float(rv[0])
        Rdy[i] = float(rv[1])
        Rnorm[i] = float(np.linalg.norm(rv[:2]))

    positive = dist[dist > 1e-8]
    if len(positive) == 0:
        raise RuntimeError("Cannot infer nearest-neighbor distance from component map.")
    nn_distance = float(np.min(positive))

    ratio = dist / nn_distance

    rows = []
    for i in range(n):
        phys_lab = ratio_label(float(ratio[i]), tol=tol)
        R_lab = r_shell_label(float(Rnorm[i]), tol=tol)
        subrel = "same_sub" if int(ofrom[i]) == int(oto[i]) else "inter_sub"

        rows.append({
            "component_index": int(i),
            "R0": int(R[i, 0]),
            "R1": int(R[i, 1]),
            "R2": int(R[i, 2]) if R.shape[1] > 2 else 0,
            "ofrom": int(ofrom[i]),
            "oto": int(oto[i]),
            "dx": float(dx[i]),
            "dy": float(dy[i]),
            "distance": float(dist[i]),
            "nn_distance": float(nn_distance),
            "distance_over_nn": float(ratio[i]),
            "phys_shell": "phys_" + phys_lab,
            "Rdx": float(Rdx[i]),
            "Rdy": float(Rdy[i]),
            "Rnorm": float(Rnorm[i]),
            "R_shell": "R_" + R_lab,
            "sub_relation": subrel,
        })

    return pd.DataFrame(rows)


# ---------- weights ----------

def add_weight_columns(
    out: Dict[str, Any],
    prefix: str,
    labels: List[str],
    label_array: np.ndarray,
    amp2: np.ndarray,
    den: float,
) -> None:
    for lab in labels:
        mask = label_array == lab
        s = float(np.sum(amp2[mask]))
        n = int(np.sum(mask))
        out[f"{prefix}_{lab}_weight"] = s / den if den > 1e-14 else 0.0
        out[f"{prefix}_{lab}_amp2_sum"] = s
        out[f"{prefix}_{lab}_ncomp"] = n
        out[f"{prefix}_{lab}_avg_amp2_per_component"] = s / n if n > 0 else 0.0
        out[f"{prefix}_{lab}_avg_weight_per_component"] = (s / den / n) if (den > 1e-14 and n > 0) else 0.0


def summarize_mode(f: np.ndarray, comp: pd.DataFrame) -> Dict[str, Any]:
    f = np.asarray(f, dtype=complex).reshape(-1)
    amp2 = np.abs(f) ** 2
    den = float(np.sum(amp2))

    if len(f) != len(comp):
        raise ValueError(f"mode length {len(f)} != component map length {len(comp)}")

    out: Dict[str, Any] = {"norm2": den}

    phys_labels = sorted(comp["phys_shell"].unique().tolist())
    R_labels = sorted(comp["R_shell"].unique().tolist())
    sub_labels = ["same_sub", "inter_sub"]

    add_weight_columns(out, "phys", phys_labels, comp["phys_shell"].values, amp2, den)
    add_weight_columns(out, "R", R_labels, comp["R_shell"].values, amp2, den)
    add_weight_columns(out, "sub", sub_labels, comp["sub_relation"].values, amp2, den)

    # Standard columns that downstream analysis expects.
    standard_phys = [
        "phys_0", "phys_1", "phys_sqrt3", "phys_2",
        "phys_sqrt7", "phys_3", "phys_2sqrt3", "phys_sqrt13", "phys_4",
        "phys_sqrt19", "phys_sqrt21", "phys_5", "phys_3sqrt3", "phys_sqrt31",
    ]
    for lab in standard_phys:
        for suffix in ["weight", "amp2_sum", "ncomp", "avg_amp2_per_component", "avg_weight_per_component"]:
            out.setdefault(f"phys_{lab}_{suffix}", 0.0)

    standard_R = ["R_0", "R_1", "R_sqrt3", "R_2", "R_sqrt7", "R_3"]
    for lab in standard_R:
        for suffix in ["weight", "amp2_sum", "ncomp", "avg_amp2_per_component", "avg_weight_per_component"]:
            out.setdefault(f"R_{lab}_{suffix}", 0.0)

    # Important combined shell/subrelation weights.
    for phys in phys_labels:
        for sub in sub_labels:
            mask = (comp["phys_shell"].values == phys) & (comp["sub_relation"].values == sub)
            s = float(np.sum(amp2[mask]))
            n = int(np.sum(mask))
            col = f"{phys}_{sub}"
            out[f"{col}_weight"] = s / den if den > 1e-14 else 0.0
            out[f"{col}_ncomp"] = n
            out[f"{col}_avg_weight_per_component"] = (s / den / n) if (den > 1e-14 and n > 0) else 0.0

    # Dominant physical shell.
    phys_w = {lab: out.get(f"phys_{lab}_weight", 0.0) for lab in phys_labels}
    R_w = {lab: out.get(f"R_{lab}_weight", 0.0) for lab in R_labels}
    sub_w = {lab: out.get(f"sub_{lab}_weight", 0.0) for lab in sub_labels}

    if phys_w:
        best = max(phys_w, key=phys_w.get)
        out["dominant_phys_shell"] = best
        out["dominant_phys_shell_weight"] = phys_w[best]
    if R_w:
        best = max(R_w, key=R_w.get)
        out["dominant_R_shell"] = best
        out["dominant_R_shell_weight"] = R_w[best]
    if sub_w:
        best = max(sub_w, key=sub_w.get)
        out["dominant_sub_relation"] = best
        out["dominant_sub_relation_weight"] = sub_w[best]

    # Old-style comparison, but now from actual shells.
    out["onsite_weight_new"] = out.get("phys_phys_0_weight", 0.0)
    out["nn_weight_new"] = out.get("phys_phys_1_weight", 0.0)
    out["second_neighbor_sqrt3_weight"] = out.get("phys_phys_sqrt3_weight", 0.0)
    out["distance2_weight"] = out.get("phys_phys_2_weight", 0.0)
    out["non_onsite_non_nn_weight_new"] = 1.0 - out["onsite_weight_new"] - out["nn_weight_new"]

    return out


def top_components(f: np.ndarray, comp: pd.DataFrame, top: int, base: Dict[str, Any]) -> pd.DataFrame:
    f = np.asarray(f, dtype=complex).reshape(-1)
    order = np.argsort(np.abs(f))[::-1][:int(top)]
    rows = []

    for rank, idx in enumerate(order):
        c = comp.iloc[int(idx)].to_dict()
        row = dict(base)
        row.update({
            "rank": int(rank),
            "component_index": int(idx),
            "R0": int(c["R0"]),
            "R1": int(c["R1"]),
            "R2": int(c["R2"]),
            "ofrom": int(c["ofrom"]),
            "oto": int(c["oto"]),
            "dx": float(c["dx"]),
            "dy": float(c["dy"]),
            "distance": float(c["distance"]),
            "nn_distance": float(c["nn_distance"]),
            "distance_over_nn": float(c["distance_over_nn"]),
            "phys_shell": str(c["phys_shell"]),
            "Rnorm": float(c["Rnorm"]),
            "R_shell": str(c["R_shell"]),
            "sub_relation": str(c["sub_relation"]),
            "value_abs": float(abs(f[idx])),
            "value_real": float(np.real(f[idx])),
            "value_imag": float(np.imag(f[idx])),
            "value_amp2": float(abs(f[idx]) ** 2),
        })
        rows.append(row)

    return pd.DataFrame(rows)


# ---------- per snapshot/case ----------

def diagnose_snapshot(snapshot_dir: Path, top_components_n: int, max_rows: int) -> None:
    O_path = snapshot_dir / "O.dvg"
    inst_dir = snapshot_dir / "instability"

    if not O_path.exists():
        raise FileNotFoundError(f"Missing {O_path}")
    if not (inst_dir / "instability_summary.csv").exists():
        raise FileNotFoundError(f"Missing {inst_dir / 'instability_summary.csv'}")
    if not (inst_dir / "instability_modes.npz").exists():
        raise FileNotFoundError(f"Missing {inst_dir / 'instability_modes.npz'}")

    O = load_output(O_path)
    comp = component_geometry(O)

    outdir = snapshot_dir / "formfactor_shell_v2"
    outdir.mkdir(parents=True, exist_ok=True)

    comp.to_csv(outdir / "component_shell_map.csv", index=False)

    summ = pd.read_csv(inst_dir / "instability_summary.csv")
    modes = np.load(inst_dir / "instability_modes.npz")

    if "strength_svd_KL" in summ.columns:
        summ = summ.sort_values("strength_svd_KL", ascending=False).head(int(max_rows))
    else:
        summ = summ.head(int(max_rows))

    summary_rows = []
    top_tables = []

    for _, r in summ.iterrows():
        key = str(r["mode_key"])
        if key not in modes:
            continue

        f = np.asarray(modes[key], dtype=complex).reshape(-1)
        base = r.to_dict()
        base["mode_key"] = key

        sw = summarize_mode(f, comp)
        row = dict(base)
        row.update(sw)
        summary_rows.append(row)

        top_base = {
            k: row[k]
            for k in ["mode_key", "q_label", "family", "sector", "convention", "mode"]
            if k in row
        }
        top_tables.append(top_components(f, comp, top_components_n, top_base))

    pd.DataFrame(summary_rows).to_csv(outdir / "formfactor_shell_summary.csv", index=False)
    if top_tables:
        pd.concat(top_tables, ignore_index=True).to_csv(outdir / "formfactor_shell_top_components.csv", index=False)
    else:
        pd.DataFrame().to_csv(outdir / "formfactor_shell_top_components.csv", index=False)

    print(f"wrote {outdir / 'formfactor_shell_summary.csv'}")
    print(f"wrote {outdir / 'formfactor_shell_top_components.csv'}")


def find_snapshots(case_or_snapshot: Path) -> List[Path]:
    snaps: List[Path] = []

    if (case_or_snapshot / "O.dvg").exists() and (case_or_snapshot / "instability").exists():
        snaps.append(case_or_snapshot)

    root = case_or_snapshot / "snapshots"
    if root.exists():
        for p in sorted(root.iterdir()):
            if p.is_dir() and (p.name.startswith("iter_") or p.name == "final"):
                snaps.append(p)

    # unique
    seen = set()
    out = []
    for s in snaps:
        r = str(s.resolve())
        if r not in seen:
            seen.add(r)
            out.append(s)
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("case_dir", help="case directory containing snapshots/, or a single snapshot directory")
    ap.add_argument("--top-components", type=int, default=40)
    ap.add_argument("--max-rows", type=int, default=120)
    args = ap.parse_args(argv)

    case_dir = Path(args.case_dir).resolve()
    snaps = find_snapshots(case_dir)

    if not snaps:
        raise SystemExit(f"No snapshot directories found under {case_dir}")

    print(f"Found {len(snaps)} snapshot dirs under {case_dir}")

    nbad = 0
    for i, snap in enumerate(snaps, 1):
        print(f"[{i}/{len(snaps)}] {snap}")
        try:
            diagnose_snapshot(snap, args.top_components, args.max_rows)
        except Exception as e:
            nbad += 1
            print(f"BAD {snap}: {type(e).__name__}: {e}")

    if nbad:
        raise SystemExit(f"Finished with {nbad} bad snapshots.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
