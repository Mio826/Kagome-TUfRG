#!/usr/bin/env python3
# coding: utf-8
"""
TUfRG_formfactor_from_modes_v1.py
Version: v1-from-loop-dressed-instability-modes-2026-05-13

Level-2 form-factor classifier for TUFRG instability modes.

Input must come from TUfRG_instability_v1.py, especially instability_modes.npz.
This file does NOT read raw Pvec/Cvec/Dvec.  It classifies the loop-dressed
right singular vectors produced by Level 1.

Outputs
-------
case_dir/formfactor_v1/formfactor_summary.csv
case_dir/formfactor_v1/formfactor_top_components.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import TUfRG_model as km
except Exception:  # pragma: no cover
    km = None


def load_output(path: Path):
    import diverge.output as ro
    return ro.read(str(path))


def norm2(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=complex).reshape(-1)
    return float(np.vdot(x, x).real)


def get_mi_arrays(O: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (np.asarray(O.mi_to_R, dtype=int), np.asarray(O.mi_to_ofrom, dtype=int), np.asarray(O.mi_to_oto, dtype=int))


def component_distances(O: Any) -> np.ndarray:
    R, ofrom, oto = get_mi_arrays(O)
    if km is None:
        out = np.full(len(ofrom), np.inf)
        out[(R[:, 0] == 0) & (R[:, 1] == 0) & (ofrom == oto)] = 0.0
        return out
    lattice, pos = km.kagome_lattice_and_positions()
    d = np.zeros(len(ofrom), dtype=float)
    for i in range(len(ofrom)):
        dr = km.displacement_vector(lattice, pos, int(R[i, 0]), int(R[i, 1]), int(ofrom[i]), int(oto[i]))
        d[i] = float(np.linalg.norm(dr[:2]))
    return d


def component_masks(O: Any) -> Dict[str, np.ndarray]:
    R, ofrom, oto = get_mi_arrays(O)
    d = component_distances(O)
    onsite = (np.abs(d) < 1e-6) & (ofrom == oto)
    nn = np.abs(d - 0.5) < 1e-5
    diagonal = ofrom == oto
    offdiag = ~diagonal
    long = ~(onsite | nn)
    return {"onsite": onsite, "nn": nn, "long": long, "diagonal": diagonal, "offdiag": offdiag}


def weight(f: np.ndarray, mask: np.ndarray) -> float:
    den = norm2(f)
    if den <= 0:
        return 0.0
    return float(norm2(np.asarray(f)[np.asarray(mask, dtype=bool)]) / den)


def onsite_diagonal_components(O: Any, f: np.ndarray) -> np.ndarray:
    R, ofrom, oto = get_mi_arrays(O)
    vals = []
    for a in range(3):
        mask = (R[:, 0] == 0) & (R[:, 1] == 0) & (ofrom == a) & (oto == a)
        if R.shape[1] > 2:
            mask &= (R[:, 2] == 0)
        vals.append(np.sum(f[mask]) if np.any(mask) else 0.0 + 0.0j)
    return np.asarray(vals, dtype=complex)


def onsite_irrep_summary(O: Any, f: np.ndarray) -> Dict[str, float]:
    total = norm2(f)
    od = onsite_diagonal_components(O, f)
    od_norm = float(np.sum(np.abs(od) ** 2))
    A1 = np.array([1, 1, 1], dtype=complex) / np.sqrt(3)
    E1 = np.array([2, -1, -1], dtype=complex) / np.sqrt(6)
    E2 = np.array([0, 1, -1], dtype=complex) / np.sqrt(2)
    def total_w(e):
        return float(abs(np.vdot(e, od)) ** 2 / total) if total > 1e-14 else 0.0
    def onsite_w(e):
        return float(abs(np.vdot(e, od)) ** 2 / od_norm) if od_norm > 1e-14 else 0.0
    return {
        "onsite_diag_A_real": float(np.real(od[0])), "onsite_diag_A_imag": float(np.imag(od[0])),
        "onsite_diag_B_real": float(np.real(od[1])), "onsite_diag_B_imag": float(np.imag(od[1])),
        "onsite_diag_C_real": float(np.real(od[2])), "onsite_diag_C_imag": float(np.imag(od[2])),
        "onsite_diag_norm_fraction": float(od_norm / total) if total > 1e-14 else 0.0,
        "onsite_A1_weight_total": total_w(A1),
        "onsite_E1_weight_total": total_w(E1),
        "onsite_E2_weight_total": total_w(E2),
        "onsite_E_weight_total": total_w(E1) + total_w(E2),
        "onsite_A1_weight_within_onsite": onsite_w(A1),
        "onsite_E_weight_within_onsite": onsite_w(E1) + onsite_w(E2),
    }


def summarize_vector(O: Any, f: np.ndarray) -> Dict[str, float]:
    masks = component_masks(O)
    nnmask = masks["nn"]
    nn_den = float(np.sum(np.abs(f[nnmask]) ** 2))
    real_proxy = float(np.sum(np.real(f[nnmask]) ** 2) / nn_den) if nn_den > 1e-14 else 0.0
    imag_proxy = float(np.sum(np.imag(f[nnmask]) ** 2) / nn_den) if nn_den > 1e-14 else 0.0
    out = {
        "norm2": norm2(f),
        "onsite_weight": weight(f, masks["onsite"]),
        "nn_weight": weight(f, masks["nn"]),
        "long_weight": weight(f, masks["long"]),
        "diagonal_weight": weight(f, masks["diagonal"]),
        "offdiag_weight": weight(f, masks["offdiag"]),
        "real_bond_proxy_weight": real_proxy,
        "imag_bond_proxy_weight": imag_proxy,
    }
    out.update(onsite_irrep_summary(O, f))
    return out


def classify(row: Mapping[str, Any]) -> str:
    q = str(row.get("q_label", ""))
    sector = str(row.get("sector", ""))
    onsite = float(row.get("onsite_weight", 0.0))
    nn = float(row.get("nn_weight", 0.0))
    imag = float(row.get("imag_bond_proxy_weight", 0.0))
    a1 = float(row.get("onsite_A1_weight_within_onsite", 0.0))
    e = float(row.get("onsite_E_weight_within_onsite", 0.0))
    onsite_frac = float(row.get("onsite_diag_norm_fraction", 0.0))
    if sector in {"mag", "ph-sz", "ph-z"} or sector.startswith("ph-s"):
        if q == "Gamma" and onsite > 0.5 and onsite_frac > 0.5 and a1 > 0.8:
            return "q0-FM-onsite-A1-like"
        if q.startswith("M") and onsite > 0.5:
            return "finiteM-SDW-onsite-like"
        if q.startswith("M") and nn > 0.25 and imag > 0.25:
            return "finiteM-spin-current/SBO-imag-bond-candidate"
        if q.startswith("M") and nn > 0.25:
            return "finiteM-SBO-real-bond-like"
    if sector in {"charge", "ph-rho"} or "rho" in sector:
        if q == "Gamma" and onsite > 0.5 and onsite_frac > 0.5 and a1 > 0.95:
            return "q0-uniform-charge-A1-forward-scattering-like"
        if q == "Gamma" and onsite > 0.5 and onsite_frac > 0.5 and e > 0.8:
            return "q0-PI-nematic-onsite-E-like"
        if q.startswith("M") and onsite > 0.5:
            return "finiteM-CDW-onsite-like"
        if q.startswith("M") and nn > 0.25 and imag > 0.25:
            return "finiteM-charge-current/cBO-imag-bond-candidate"
        if q.startswith("M") and nn > 0.25:
            return "finiteM-CBO-real-bond-like"
    if sector.startswith("pp") or sector == "sc":
        if q == "Gamma" and onsite > 0.5 and a1 > 0.5:
            return "q0-A1-onsite-SC-like"
        if q == "Gamma" and nn > 0.25:
            return "q0-NN-pairing-form-factor"
        if q.startswith("M"):
            return "finiteQ-pairing/PDW-like"
    if nn > 0.25 and imag > 0.25:
        return "imaginary-bond-rich-current-like-formfactor"
    if nn > 0.25:
        return "NN-bond-rich-formfactor"
    if onsite > 0.5:
        return "onsite-dominated-formfactor"
    return "mixed-or-longer-range-formfactor"


def top_components(O: Any, f: np.ndarray, top: int, base: Mapping[str, Any]) -> List[Dict[str, Any]]:
    R, ofrom, oto = get_mi_arrays(O)
    masks = component_masks(O)
    order = np.argsort(np.abs(f))[::-1][: int(top)]
    rows = []
    for rank, i in enumerate(order):
        if masks["onsite"][i]: typ = "onsite"
        elif masks["nn"][i]: typ = "NN"
        else: typ = "long"
        rows.append({
            **{k: base[k] for k in ["mode_key", "q_label", "family", "sector", "convention", "mode"] if k in base},
            "rank": int(rank), "component_index": int(i),
            "R0": int(R[i,0]), "R1": int(R[i,1]), "R2": int(R[i,2]) if R.shape[1] > 2 else 0,
            "ofrom": int(ofrom[i]), "oto": int(oto[i]), "component_type": typ,
            "value_abs": float(abs(f[i])), "value_real": float(np.real(f[i])), "value_imag": float(np.imag(f[i])),
        })
    return rows


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text(""); return
    keys: List[str] = []
    for r in rows:
        for k in r:
            if k not in keys: keys.append(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Classify loop-dressed TUFRG instability mode form factors.")
    p.add_argument("case_dir", help="case directory containing O.dvg and instability outputs")
    p.add_argument("--instability-dir", default="instability")
    p.add_argument("--out", default="formfactor")
    p.add_argument("--top-components", type=int, default=12)
    p.add_argument("--max-rows", type=int, default=80, help="classify top N rows by strength_svd_KL")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    case_dir = Path(args.case_dir).resolve()
    O = load_output(case_dir / "O.dvg")
    inst_dir = case_dir / str(args.instability_dir)
    summ = pd.read_csv(inst_dir / "instability_summary.csv")
    modes = np.load(inst_dir / "instability_modes.npz")
    summ = summ.sort_values("strength_svd_KL", ascending=False).head(int(args.max_rows))
    rows: List[Dict[str, Any]] = []
    top_rows: List[Dict[str, Any]] = []
    for _, r in summ.iterrows():
        key = str(r["mode_key"])
        if key not in modes:
            continue
        f = np.asarray(modes[key], dtype=complex).reshape(-1)
        fs = summarize_vector(O, f)
        row: Dict[str, Any] = {k: r[k] for k in r.index}
        row.update(fs)
        row["candidate_phase_label"] = classify(row)
        rows.append(row)
        top_rows.extend(top_components(O, f, int(args.top_components), row))
    outdir = case_dir / str(args.out)
    write_csv(outdir / "formfactor_summary.csv", rows)
    write_csv(outdir / "formfactor_top_components.csv", top_rows)
    print("wrote", outdir / "formfactor_summary.csv")
    print("wrote", outdir / "formfactor_top_components.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
