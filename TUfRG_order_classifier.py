#!/usr/bin/env python3
# coding: utf-8
"""
TUfRG_order_classifier.py
Version: clean-structural-order-classifier-2026-05-19

Structural order classifier for loop-dressed TUFRG instability modes.

Purpose
-------
This file is the final order-labeling layer after Level-1 instability extraction.
It reads

    case_dir/<instability-dir>/instability_summary.csv
    case_dir/<instability-dir>/instability_modes.npz
    case_dir/O.dvg

and classifies particle-hole modes using orbital/bond form-factor structure.

Main idea
---------
The robust distinction between real bond order and loop-current order should not
be made from raw Re(F) vs Im(F), because a singular vector has an arbitrary
complex global phase.  Instead, this file uses a bond-reversal structural test.

For a TU component alpha=(R, a, b), define reverse(alpha)=(-R, b, a).  For a
finite-q mode, first remove the convention phase exp(i q.R):

    F_tilde(R,a,b) = exp(-i q.R) F(R,a,b).

Then decompose under linear bond reversal:

    F_plus (R,a,b) = [F_tilde(R,a,b) + F_tilde(-R,b,a)] / 2
    F_minus(R,a,b) = [F_tilde(R,a,b) - F_tilde(-R,b,a)] / 2

F_plus is real-bond-order-like.  F_minus is loop-current-like.  The corresponding
weights are invariant under F -> exp(i theta) F.

Outputs
-------
case_dir/order_classification/order_summary.csv
case_dir/order_classification/order_components.csv
case_dir/order_classification/order_subspace_summary.csv
case_dir/order_classification/order_report.txt

Recommended role in the pipeline
--------------------------------
TUfRG_instability.py              -> Level-1: channel/q/mode strengths
TUfRG_formfactor_from_modes.py    -> Level-2: onsite/NN/long coarse weights
TUfRG_order_classifier.py         -> final structural CDW/CBO/LCO labels
TUfRG_realspace_from_modes.py     -> visualization/current fingerprints
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import TUfRG_model as km
except Exception:  # pragma: no cover
    km = None


# -----------------------------------------------------------------------------
# Basic IO
# -----------------------------------------------------------------------------

def load_output(path: Path):
    import diverge.output as ro
    return ro.read(str(path))


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    keys: List[str] = []
    for r in rows:
        for k in r.keys():
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def norm2(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=complex).reshape(-1)
    return float(np.vdot(x, x).real)


# -----------------------------------------------------------------------------
# TU basis helpers
# -----------------------------------------------------------------------------

def get_mi_arrays(O: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.asarray(getattr(O, "mi_to_R"), dtype=int),
        np.asarray(getattr(O, "mi_to_ofrom"), dtype=int),
        np.asarray(getattr(O, "mi_to_oto"), dtype=int),
    )


def component_distances(O: Any) -> np.ndarray:
    R, ofrom, oto = get_mi_arrays(O)
    if km is None:
        # Fallback: only onsite is reliable without the model geometry.
        out = np.full(len(ofrom), np.inf)
        onsite = (ofrom == oto) & np.all(R == 0, axis=1)
        out[onsite] = 0.0
        return out
    lattice, pos = km.kagome_lattice_and_positions()
    d = np.zeros(len(ofrom), dtype=float)
    for i in range(len(ofrom)):
        dr = km.tu_ff_displacement_vector(lattice, pos, int(R[i, 0]), int(R[i, 1]), int(ofrom[i]), int(oto[i]))
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
    return {"onsite": onsite, "nn": nn, "diagonal": diagonal, "offdiag": offdiag, "long": long}


def reverse_index_map(O: Any) -> np.ndarray:
    """Map alpha=(R,ofrom,oto) to reverse alpha_bar=(-R,oto,ofrom).

    This is a linear bond-reversal map.  It is not complex conjugation.
    """
    R, ofrom, oto = get_mi_arrays(O)
    keys: Dict[Tuple[Tuple[int, ...], int, int], int] = {}
    for i in range(len(ofrom)):
        keys[(tuple(int(x) for x in R[i]), int(ofrom[i]), int(oto[i]))] = i
    rev = np.full(len(ofrom), -1, dtype=int)
    for i in range(len(ofrom)):
        rrev = tuple(int(-x) for x in R[i])
        rev[i] = int(keys.get((rrev, int(oto[i]), int(ofrom[i])), -1))
    return rev


def infer_nk(O: Any, metadata: Mapping[str, Any]) -> int:
    try:
        return int(metadata.get("mesh", {}).get("nk"))
    except Exception:
        pass
    nq = int(getattr(O, "nk", 1))
    return int(round(math.sqrt(nq)))


def parse_q_reduced(x: Any, iq: int, nk: int) -> Tuple[float, float, float]:
    if isinstance(x, str) and x.strip():
        try:
            vals = ast.literal_eval(x)
            if isinstance(vals, (tuple, list)) and len(vals) >= 2:
                return (float(vals[0]), float(vals[1]), float(vals[2]) if len(vals) > 2 else 0.0)
        except Exception:
            pass
    iq = int(iq)
    nk = int(nk)
    return (float(iq % nk) / nk, float(iq // nk) / nk, 0.0)


def q_unwind_formfactor(O: Any, f: np.ndarray, qred: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    """Return F_tilde=exp(-i q.R)F and the phase array.

    qred is in reciprocal reduced coordinates, so q.R uses 2*pi*qred.R.
    """
    R, _, _ = get_mi_arrays(O)
    q = np.asarray(list(qred) + [0.0, 0.0, 0.0], dtype=float)[: R.shape[1]]
    phase = np.exp(-2j * np.pi * (R @ q))
    return np.asarray(f, dtype=complex).reshape(-1) * phase, phase


# -----------------------------------------------------------------------------
# Structural diagnostics
# -----------------------------------------------------------------------------

def structural_decomposition(O: Any, f: np.ndarray, qred: Sequence[float]) -> Dict[str, Any]:
    f = np.asarray(f, dtype=complex).reshape(-1)
    ftilde, unwind_phase = q_unwind_formfactor(O, f, qred)
    rev = reverse_index_map(O)
    masks = component_masks(O)

    plus = np.zeros_like(ftilde)
    minus = np.zeros_like(ftilde)
    matched = rev >= 0
    for i in range(len(ftilde)):
        j = int(rev[i])
        if j >= 0:
            plus[i] = 0.5 * (ftilde[i] + ftilde[j])
            minus[i] = 0.5 * (ftilde[i] - ftilde[j])

    total = norm2(ftilde)
    if total <= 1e-300:
        total = 1.0

    def w(mask: np.ndarray, vec: np.ndarray) -> float:
        m = np.asarray(mask, dtype=bool) & matched
        if not np.any(m):
            return 0.0
        return float(norm2(vec[m]) / total)

    out: Dict[str, Any] = {
        "ftilde": ftilde,
        "phase": unwind_phase,
        "plus": plus,
        "minus": minus,
        "rev": rev,
        "matched": matched,
        "total_norm2": float(total),
        "onsite_weight": w(masks["onsite"], ftilde),
        "nn_weight": w(masks["nn"], ftilde),
        "long_weight": w(masks["long"], ftilde),
        "diagonal_weight": w(masks["diagonal"], ftilde),
        "offdiag_weight": w(masks["offdiag"], ftilde),
        "bond_symmetric_nn_weight": w(masks["nn"], plus),
        "bond_antisymmetric_nn_weight": w(masks["nn"], minus),
        "bond_symmetric_all_weight": w(np.ones_like(matched, dtype=bool), plus),
        "bond_antisymmetric_all_weight": w(np.ones_like(matched, dtype=bool), minus),
        "unmatched_reverse_weight": float(norm2(ftilde[~matched]) / total) if np.any(~matched) else 0.0,
    }
    return out


def is_gamma_label(q_label: str) -> bool:
    return str(q_label) in {"Gamma", "G", "q0", "0", "iq0"}


def structural_label(sector: str, q_label: str, onsite: float, sym_nn: float, antisym_nn: float) -> str:
    """Conservative structural label.

    The channel, transfer momentum, and spin/charge/pp sector are reliable.
    The bond-symmetric / bond-antisymmetric classifier is only used at Gamma.
    At finite q, the correct Hermitian current/bond source embedding requires
    additional reciprocal-gauge phases, so we avoid CBO/SBO/LCO labels here.
    """
    sector = str(sector)
    q_label = str(q_label)
    is_charge = sector in {"ph-rho", "charge"} or "rho" in sector
    is_spin = sector in {"ph-sz", "mag"} or sector.startswith("ph-s")
    is_gamma = is_gamma_label(q_label)

    if is_charge:
        if not is_gamma:
            return "finiteQ-charge-PH-order-structural-subtype-unresolved"
        if antisym_nn > 0.35 and onsite < 0.25:
            return "q0-charge-LCO-like"
        if sym_nn > 0.35:
            return "q0-CBO/nematic-like"
        if onsite > 0.40:
            return "q0-charge-onsite/forward-like"
        return "q0-charge-mixed"

    if is_spin:
        if not is_gamma:
            return "finiteQ-spin-PH-order-structural-subtype-unresolved"
        if antisym_nn > 0.35 and onsite < 0.25:
            return "q0-spin-LCO-like"
        if sym_nn > 0.35:
            return "q0-SBO-like"
        if onsite > 0.40:
            return "FM-like"
        return "q0-spin-mixed"

    if sector.startswith("pp") or sector == "sc":
        return "pairing-sector-not-structurally-classified-here"
    return "not-classified"


def order_scores_for_mode(O: Any, f: np.ndarray, row: Mapping[str, Any], metadata: Mapping[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    nk = infer_nk(O, metadata)
    qred = parse_q_reduced(row.get("q_reduced", ""), int(row.get("iq", 0)), nk)
    decomp = structural_decomposition(O, f, qred)
    R, ofrom, oto = get_mi_arrays(O)
    masks = component_masks(O)
    rev = np.asarray(decomp["rev"], dtype=int)
    ftilde = np.asarray(decomp["ftilde"], dtype=complex)
    plus = np.asarray(decomp["plus"], dtype=complex)
    minus = np.asarray(decomp["minus"], dtype=complex)
    phase = np.asarray(decomp["phase"], dtype=complex)

    sector = str(row.get("sector", ""))
    q_label = str(row.get("q_label", ""))
    label = structural_label(
        sector,
        q_label,
        float(decomp["onsite_weight"]),
        float(decomp["bond_symmetric_nn_weight"]),
        float(decomp["bond_antisymmetric_nn_weight"]),
    )

    is_gamma = is_gamma_label(q_label)
    finite_q_warning = "" if is_gamma else (
        "finite-q bond-symmetric/antisymmetric and LCO/CBO/SBO scores are not used for phase labels; "
        "TU form-factor Hermiticity needs reciprocal-gauge calibration"
    )

    summary = {
        "mode_key": str(row.get("mode_key", "")),
        "case": row.get("case", ""),
        "family": row.get("family", ""),
        "sector": sector,
        "q_label": q_label,
        "iq": row.get("iq", ""),
        "q_reduced": row.get("q_reduced", ""),
        "mode": row.get("mode", ""),
        "strength_svd_KL": row.get("strength_svd_KL", ""),
        "structural_norm2": decomp["total_norm2"],
        "onsite_weight": decomp["onsite_weight"],
        "nn_weight": decomp["nn_weight"],
        "long_weight": decomp["long_weight"],
        "diagonal_weight": decomp["diagonal_weight"],
        "offdiag_weight": decomp["offdiag_weight"],
        "bond_symmetric_nn_weight": decomp["bond_symmetric_nn_weight"],
        "bond_antisymmetric_nn_weight": decomp["bond_antisymmetric_nn_weight"],
        "bond_symmetric_all_weight": decomp["bond_symmetric_all_weight"],
        "bond_antisymmetric_all_weight": decomp["bond_antisymmetric_all_weight"],
        "unmatched_reverse_weight": decomp["unmatched_reverse_weight"],
        "LCO_score": decomp["bond_antisymmetric_nn_weight"] if is_gamma else float("nan"),
        "CBO_SBO_score": decomp["bond_symmetric_nn_weight"] if is_gamma else float("nan"),
        "finite_q_structural_warning": finite_q_warning,
        "order_label": label,
        "classification_note": (
            "Gamma structural classifier uses corrected divERGe TU displacement convention. "
            "Finite-q CBO/SBO/LCO sublabels are disabled pending explicit source/gauge calibration."
        ),
    }

    comp_rows: List[Dict[str, Any]] = []
    for i in range(len(ftilde)):
        j = int(rev[i])
        comp = {
            "mode_key": str(row.get("mode_key", "")),
            "family": row.get("family", ""),
            "sector": sector,
            "q_label": q_label,
            "mode": row.get("mode", ""),
            "alpha": int(i),
            "reverse_alpha": int(j),
            "R0": int(R[i, 0]),
            "R1": int(R[i, 1]),
            "R2": int(R[i, 2]) if R.shape[1] > 2 else 0,
            "ofrom": int(ofrom[i]),
            "oto": int(oto[i]),
            "is_onsite": bool(masks["onsite"][i]),
            "is_NN": bool(masks["nn"][i]),
            "is_matched_reverse": bool(j >= 0),
            "q_unwind_phase_real": float(np.real(phase[i])),
            "q_unwind_phase_imag": float(np.imag(phase[i])),
            "F_tilde_real": float(np.real(ftilde[i])),
            "F_tilde_imag": float(np.imag(ftilde[i])),
            "F_plus_real": float(np.real(plus[i])),
            "F_plus_imag": float(np.imag(plus[i])),
            "F_minus_real": float(np.real(minus[i])),
            "F_minus_imag": float(np.imag(minus[i])),
            "abs_F_tilde2": float(abs(ftilde[i]) ** 2),
            "abs_F_plus2": float(abs(plus[i]) ** 2),
            "abs_F_minus2": float(abs(minus[i]) ** 2),
        }
        if j >= 0:
            comp.update({
                "F_reverse_tilde_real": float(np.real(ftilde[j])),
                "F_reverse_tilde_imag": float(np.imag(ftilde[j])),
            })
        comp_rows.append(comp)
    return summary, comp_rows


# -----------------------------------------------------------------------------
# Subspace diagnostics
# -----------------------------------------------------------------------------

def orthonormalize_rows(vecs: List[np.ndarray]) -> List[np.ndarray]:
    if not vecs:
        return []
    M = np.vstack([np.asarray(v, dtype=complex).reshape(1, -1) for v in vecs])
    # QR on columns of M.T gives orthonormal basis vectors as columns.
    Q, R = np.linalg.qr(M.T)
    rank = int(np.sum(np.abs(np.diag(R)) > 1e-10)) if R.size else 0
    return [Q[:, i] for i in range(rank)]


def subspace_summary(O: Any, group_rows: pd.DataFrame, modes: Mapping[str, np.ndarray], metadata: Mapping[str, Any]) -> Dict[str, Any]:
    vecs = []
    strengths = []
    keys = []
    for _, r in group_rows.iterrows():
        key = str(r["mode_key"])
        if key in modes:
            vecs.append(np.asarray(modes[key], dtype=complex).reshape(-1))
            strengths.append(float(r.get("strength_svd_KL", 0.0)))
            keys.append(key)
    basis = orthonormalize_rows(vecs)
    if not basis:
        return {}
    # Average trace weight over an orthonormal basis. This is invariant under unitary rotations inside the subspace.
    accum = {
        "onsite_weight": 0.0,
        "nn_weight": 0.0,
        "bond_symmetric_nn_weight": 0.0,
        "bond_antisymmetric_nn_weight": 0.0,
        "bond_symmetric_all_weight": 0.0,
        "bond_antisymmetric_all_weight": 0.0,
    }
    first = group_rows.iloc[0]
    for v in basis:
        one, _ = order_scores_for_mode(O, v, first, metadata)
        for k in accum:
            accum[k] += float(one.get(k, 0.0))
    for k in accum:
        accum[k] /= len(basis)
    q_label_first = str(first.get("q_label", ""))
    is_gamma_subspace = is_gamma_label(q_label_first)
    label = structural_label(
        str(first.get("sector", "")),
        q_label_first,
        accum["onsite_weight"],
        accum["bond_symmetric_nn_weight"],
        accum["bond_antisymmetric_nn_weight"],
    )
    return {
        "family": first.get("family", ""),
        "sector": first.get("sector", ""),
        "q_label": first.get("q_label", ""),
        "convention": first.get("convention", ""),
        "n_input_modes": len(vecs),
        "subspace_rank": len(basis),
        "max_strength_svd_KL": float(np.max(strengths)) if strengths else 0.0,
        "min_strength_svd_KL": float(np.min(strengths)) if strengths else 0.0,
        "included_mode_keys": ";".join(keys),
        **{f"subspace_{k}": v for k, v in accum.items()},
        "subspace_LCO_score": accum["bond_antisymmetric_nn_weight"] if is_gamma_subspace else float("nan"),
        "subspace_CBO_SBO_score": accum["bond_symmetric_nn_weight"] if is_gamma_subspace else float("nan"),
        "subspace_finite_q_structural_warning": "" if is_gamma_subspace else "finite-q subspace CBO/SBO/LCO scores disabled pending TU source/gauge calibration",
        "subspace_order_label": label,
        "subspace_note": "trace-average over orthonormalized near-degenerate modes; finite-q CBO/SBO/LCO sublabels disabled",
    }


def make_subspace_groups(df: pd.DataFrame, rel_tol: float, abs_tol: float, max_modes: int) -> List[pd.DataFrame]:
    groups: List[pd.DataFrame] = []
    if df.empty:
        return groups
    keys = ["family", "sector", "q_label", "convention"]
    for _, g in df.groupby(keys, dropna=False):
        g = g.sort_values("strength_svd_KL", ascending=False).copy()
        if g.empty:
            continue
        top = float(g.iloc[0]["strength_svd_KL"])
        if top <= 0:
            continue
        keep = g[(top - g["strength_svd_KL"].astype(float) <= max(abs_tol, rel_tol * abs(top)))]
        keep = keep.head(int(max_modes))
        if len(keep) >= 2:
            groups.append(keep)
    return groups


# -----------------------------------------------------------------------------
# Main analysis
# -----------------------------------------------------------------------------

def load_instability(case_dir: Path, instability_dir: str) -> Tuple[pd.DataFrame, Mapping[str, np.ndarray]]:
    inst_dir = case_dir / instability_dir
    summary_path = inst_dir / "instability_summary.csv"
    modes_path = inst_dir / "instability_modes.npz"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    if not modes_path.exists():
        raise FileNotFoundError(modes_path)
    return pd.read_csv(summary_path), np.load(modes_path)


def filter_rows(summary: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    df = summary.copy()
    if args.family:
        df = df[df["family"].astype(str).isin(args.family)]
    if args.sector:
        df = df[df["sector"].astype(str).isin(args.sector)]
    if args.q:
        df = df[df["q_label"].astype(str).isin(args.q)]
    if args.mode is not None:
        df = df[df["mode"].astype(int).isin([int(x) for x in args.mode])]
    if args.mode_key:
        df = df[df["mode_key"].astype(str).isin(set(args.mode_key))]
    if args.max_rows is not None:
        df = df.sort_values("strength_svd_KL", ascending=False).head(int(args.max_rows))
    return df.copy()


def analyze(case_dir: Path, args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], str]:
    O = load_output(case_dir / "O.dvg")
    metadata = read_json(case_dir / "metadata.json")
    summary, modes = load_instability(case_dir, args.instability_dir)
    rows = filter_rows(summary, args)

    order_rows: List[Dict[str, Any]] = []
    comp_rows: List[Dict[str, Any]] = []
    for _, r in rows.iterrows():
        key = str(r["mode_key"])
        if key not in modes:
            continue
        f = np.asarray(modes[key], dtype=complex).reshape(-1)
        one, comps = order_scores_for_mode(O, f, r, metadata)
        order_rows.append(one)
        if args.top_components is None or args.top_components <= 0:
            comp_rows.extend(comps)
        else:
            comps_sorted = sorted(comps, key=lambda x: max(float(x["abs_F_plus2"]), float(x["abs_F_minus2"]), float(x["abs_F_tilde2"])), reverse=True)
            comp_rows.extend(comps_sorted[: int(args.top_components)])

    sub_rows: List[Dict[str, Any]] = []
    groups = make_subspace_groups(rows, float(args.subspace_rel_tol), float(args.subspace_abs_tol), int(args.subspace_max_modes))
    for g in groups:
        sr = subspace_summary(O, g, modes, metadata)
        if sr:
            sub_rows.append(sr)

    report_lines = [
        f"case_dir: {case_dir}",
        f"instability_dir: {args.instability_dir}",
        f"selected_rows: {len(rows)}",
        f"classified_modes: {len(order_rows)}",
        f"subspace_groups: {len(sub_rows)}",
        "classifier: q-unwound bond-reversal structural weights",
        "raw Re/Im parts are not used for final CBO/LCO classification.",
        "LCO-like means NN bond-antisymmetric weight is large and onsite weight is small.",
        "For final figures, use TUfRG_realspace_from_modes.py to reconstruct current arrows/patterns.",
    ]
    return order_rows, comp_rows, sub_rows, "\n".join(report_lines)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Structural order classifier for loop-dressed TUFRG instability modes.")
    p.add_argument("case_dir", help="case directory containing O.dvg and instability outputs")
    p.add_argument("--instability-dir", default="instability", help="directory inside case_dir containing instability_summary.csv and instability_modes.npz")
    p.add_argument("--out", default="order_classification", help="output directory inside case_dir")
    p.add_argument("--family", nargs="*", default=["ph"], help="families to classify, default: ph")
    p.add_argument("--sector", nargs="*", default=None, help="sectors to classify, e.g. ph-rho ph-sz")
    p.add_argument("--q", nargs="*", default=None, help="q labels to classify, e.g. Gamma M1 M2 M3")
    p.add_argument("--mode", nargs="*", type=int, default=None, help="mode indices to classify")
    p.add_argument("--mode-key", nargs="*", default=None, help="explicit mode_key(s) to classify")
    p.add_argument("--max-rows", type=int, default=120, help="classify top N rows by strength_svd_KL after filters")
    p.add_argument("--top-components", type=int, default=40, help="write top N components per mode; <=0 writes all components")
    p.add_argument("--subspace-rel-tol", type=float, default=0.02, help="relative strength tolerance for near-degenerate subspace groups")
    p.add_argument("--subspace-abs-tol", type=float, default=0.0, help="absolute strength tolerance for near-degenerate subspace groups")
    p.add_argument("--subspace-max-modes", type=int, default=8, help="max modes per near-degenerate subspace group")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    case_dir = Path(args.case_dir).resolve()
    order_rows, comp_rows, sub_rows, report = analyze(case_dir, args)
    outdir = case_dir / str(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    write_csv(outdir / "order_summary.csv", order_rows)
    write_csv(outdir / "order_components.csv", comp_rows)
    write_csv(outdir / "order_subspace_summary.csv", sub_rows)
    (outdir / "order_report.txt").write_text(report)
    print(report)
    print("wrote", outdir / "order_summary.csv")
    print("wrote", outdir / "order_components.csv")
    print("wrote", outdir / "order_subspace_summary.csv")
    print("wrote", outdir / "order_report.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
