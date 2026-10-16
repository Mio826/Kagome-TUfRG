#!/usr/bin/env python3
# coding: utf-8
"""
TUfRG_realspace_from_modes.py
Version: structuraldiag-bond-pair-lco-2026-05-19

Real-space / orbital-space reconstruction for calibrated TUFRG instability modes.

This file is the Level-3 companion to TUfRG_instability_v2.py.  It reads
loop-dressed right singular vectors from

    case_dir/<instability-dir>/instability_modes.npz
    case_dir/<instability-dir>/instability_summary.csv

and reconstructs selected site, bond, current, and pairing fingerprints in the
TU form-factor basis.  It deliberately does NOT read raw Pvec/Cvec/Dvec.

Scope and interpretation
------------------------
* The input vector is a Level-1 loop-dressed instability mode, not a raw saved
  channel eigenvector.
* The output is a normalized fingerprint of the possible ordered pattern.  It
  is not a self-consistent mean-field expectation value.
* For PH current fingerprints, rho and sz sectors use the spin-diagonal hopping
  matrix t_up/t_down to report charge-current and spin-z-current proxies,
  including real-hop and SOC-hop split columns.
* This version also writes structural bond-pair diagnostics:

      bond_pair_components.csv
      structural_order_summary.csv

  The structural LCO score uses the q-unwound form factor and its
  symmetric/antisymmetric structure under bond reversal,
      (R,a,b) <-> (-R,b,a),
  not the raw imaginary part of an arbitrary SVD eigenvector.
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

try:
    import TUfRG_model as km
except Exception:  # pragma: no cover
    km = None

try:
    import TUfRG_instability as inst
except Exception:  # pragma: no cover
    inst = None


PH_SECTORS = {"ph-rho", "ph-sx", "ph-sy", "ph-sz", "charge", "mag"}
PP_SECTORS = {"pp-singlet", "pp-triplet0", "pp-triplet-plus", "pp-triplet-minus", "sc"}
ALL_OBSERVABLES = {"density", "bond", "current", "pairing"}


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


def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.+-]+", "_", str(s)).strip("_")


def parse_csv_list(values: Sequence[str]) -> List[str]:
    out: List[str] = []
    for v in values:
        out.extend([x.strip() for x in str(v).split(",") if x.strip()])
    return out


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


def infer_nk(O: Any, metadata: Mapping[str, Any]) -> int:
    try:
        return int(metadata.get("mesh", {}).get("nk"))
    except Exception:
        pass
    nq = int(getattr(O, "nk", 1))
    return int(round(math.sqrt(nq)))


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
        out = np.full(len(ofrom), np.inf)
        out[(R[:, 0] == 0) & (R[:, 1] == 0) & (ofrom == oto)] = 0.0
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


# -----------------------------------------------------------------------------
# structural diagnostics for CBO/LCO
# -----------------------------------------------------------------------------

def norm2_vec(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=complex).reshape(-1)
    return float(np.vdot(x, x).real)


def reverse_index_map(O: Any) -> np.ndarray:
    """Map alpha=(R,ofrom,oto) to reverse alpha_bar=(-R,oto,ofrom).

    This is the structural operation appearing in the structural distinction
    between bond order and loop-current order.  It is a linear bond-reversal
    operation, not complex conjugation.
    """
    R, ofrom, oto = get_mi_arrays(O)
    keys = {}
    for i in range(len(ofrom)):
        keys[(tuple(int(x) for x in R[i]), int(ofrom[i]), int(oto[i]))] = i
    rev = np.full(len(ofrom), -1, dtype=int)
    for i in range(len(ofrom)):
        rrev = tuple(int(-x) for x in R[i])
        rev[i] = int(keys.get((rrev, int(oto[i]), int(ofrom[i])), -1))
    return rev


def q_unwind_formfactor(O: Any, f: np.ndarray, qred: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    """Remove the structural convention factor exp(i q.R) from a TU form factor.

    The finite-q convention writes finite-M orders schematically as Delta(R)=exp(i M.R)*tildeDelta(R)
    (plus an extra i for LCO).  The structural symmetric/antisymmetric test is
    performed on tildeDelta, not on the raw Delta.
    """
    R, _, _ = get_mi_arrays(O)
    q = np.asarray(list(qred) + [0.0, 0.0, 0.0], dtype=float)[: R.shape[1]]
    phase = np.exp(-2j * np.pi * (R @ q))
    return np.asarray(f, dtype=complex).reshape(-1) * phase, phase


def is_gamma_label(q_label: str) -> bool:
    return str(q_label) in {"Gamma", "G", "q0", "0", "iq0"}


def structural_label_from_scores(sector: str, q_label: str, onsite: float, sym_nn: float, antisym_nn: float) -> str:
    """Conservative structural label from structural weights.

    Use bond-symmetric / bond-antisymmetric labels only at Gamma.  Finite-q
    LCO/CBO/SBO labels require an explicitly calibrated TU source convention
    and are intentionally disabled here.
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

    return "not-PH-charge/spin-classified"


def structural_bond_pair_diagnostics(
    O: Any,
    f: np.ndarray,
    row: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Compute structural bond-reversal structural weights for one mode.

    The phase-invariant structural distinction is:

        tildeF_sym  = (tildeF(R,a,b) + tildeF(-R,b,a)) / 2
        tildeF_asym = (tildeF(R,a,b) - tildeF(-R,b,a)) / 2

    where tildeF = exp(-i q.R) F.  This mimics the structural convention
    Delta_LCO(R)=i exp(i q.R) tildeDelta(R), with tildeDelta antisymmetric
    under bond reversal.  Because the operation is linear, the weights are
    invariant under the arbitrary global phase of the SVD vector.
    """
    nk = infer_nk(O, metadata)
    qred = parse_q_reduced(row.get("q_reduced", ""), int(row.get("iq", 0)), nk)
    f = np.asarray(f, dtype=complex).reshape(-1)
    ftilde, unwind_phase = q_unwind_formfactor(O, f, qred)
    R, ofrom, oto = get_mi_arrays(O)
    masks = component_masks(O)
    rev = reverse_index_map(O)

    sym = np.zeros_like(ftilde)
    asym = np.zeros_like(ftilde)
    matched = rev >= 0
    for i in range(len(ftilde)):
        j = rev[i]
        if j >= 0:
            sym[i] = 0.5 * (ftilde[i] + ftilde[j])
            asym[i] = 0.5 * (ftilde[i] - ftilde[j])

    total = norm2_vec(ftilde)
    if total <= 1e-300:
        total = 1.0

    def w(mask: np.ndarray, vec: np.ndarray = ftilde) -> float:
        mask = np.asarray(mask, dtype=bool) & matched
        return float(norm2_vec(vec[mask]) / total) if np.any(mask) else 0.0

    onsite_w = w(masks["onsite"], ftilde)
    nn_w = w(masks["nn"], ftilde)
    sym_nn_w = w(masks["nn"], sym)
    antisym_nn_w = w(masks["nn"], asym)
    sym_all_w = w(np.ones(len(ftilde), dtype=bool), sym)
    antisym_all_w = w(np.ones(len(ftilde), dtype=bool), asym)
    unmatched_w = float(norm2_vec(ftilde[~matched]) / total) if np.any(~matched) else 0.0

    mode_key = str(row.get("mode_key", ""))
    sector = str(row.get("sector", ""))
    q_label = str(row.get("q_label", ""))
    is_gamma = is_gamma_label(q_label)
    summary = {
        "mode_key": mode_key,
        "family": row.get("family", ""),
        "sector": sector,
        "q_label": q_label,
        "mode": row.get("mode", ""),
        "strength_svd_KL": row.get("strength_svd_KL", ""),
        "structural_norm2": float(total),
        "structural_onsite_weight": onsite_w,
        "structural_nn_weight": nn_w,
        "structural_symmetric_nn_weight": sym_nn_w,
        "structural_antisymmetric_nn_weight": antisym_nn_w,
        "structural_symmetric_all_weight": sym_all_w,
        "structural_antisymmetric_all_weight": antisym_all_w,
        "structural_unmatched_reverse_weight": unmatched_w,
        "structural_lco_score": antisym_nn_w if is_gamma else float("nan"),
        "structural_cbo_sbo_score": sym_nn_w if is_gamma else float("nan"),
        "structural_finite_q_warning": "" if is_gamma else "finite-q current/CBO/SBO subtype disabled pending TU source/gauge calibration",
        "structural_label": structural_label_from_scores(sector, q_label, onsite_w, sym_nn_w, antisym_nn_w),
        "structural_note": "Gamma structural weights use corrected divERGe TU displacement convention; finite-q CBO/SBO/LCO sublabels disabled",
    }

    comp_rows: List[Dict[str, Any]] = []
    for i in range(len(ftilde)):
        j = int(rev[i])
        base = {
            "mode_key": mode_key,
            "family": row.get("family", ""),
            "sector": sector,
            "q_label": q_label,
            "mode": row.get("mode", ""),
            "alpha": int(i),
            "reverse_alpha": j,
            "R0": int(R[i, 0]),
            "R1": int(R[i, 1]),
            "R2": int(R[i, 2]) if R.shape[1] > 2 else 0,
            "ofrom": int(ofrom[i]),
            "oto": int(oto[i]),
            "is_onsite": bool(masks["onsite"][i]),
            "is_NN": bool(masks["nn"][i]),
            "is_matched_reverse": bool(j >= 0),
            "q_unwind_phase_real": float(np.real(unwind_phase[i])),
            "q_unwind_phase_imag": float(np.imag(unwind_phase[i])),
            "F_raw_real": float(np.real(f[i])),
            "F_raw_imag": float(np.imag(f[i])),
            "F_tilde_real": float(np.real(ftilde[i])),
            "F_tilde_imag": float(np.imag(ftilde[i])),
            "F_sym_real": float(np.real(sym[i])),
            "F_sym_imag": float(np.imag(sym[i])),
            "F_antisym_real": float(np.real(asym[i])),
            "F_antisym_imag": float(np.imag(asym[i])),
            "abs_F_tilde2": float(abs(ftilde[i]) ** 2),
            "abs_F_sym2": float(abs(sym[i]) ** 2),
            "abs_F_antisym2": float(abs(asym[i]) ** 2),
        }
        if j >= 0:
            base.update({
                "F_reverse_tilde_real": float(np.real(ftilde[j])),
                "F_reverse_tilde_imag": float(np.imag(ftilde[j])),
            })
        comp_rows.append(base)
    return summary, comp_rows


def realspace_coordinates(cell: Tuple[int, int], orbital: int) -> np.ndarray:
    if km is None:
        # fallback kagome positions in reduced 2D plotting coordinates
        pos = np.asarray([[0.0, 0.0], [0.5, 0.0], [0.25, 0.4330127018922193]])
        return np.asarray([cell[0], cell[1]], dtype=float) + pos[int(orbital)]
    lattice, pos = km.kagome_lattice_and_positions()
    return cell[0] * lattice[0, :2] + cell[1] * lattice[1, :2] + pos[int(orbital), :2]


def endpoint_coordinates(cell: Tuple[int, int], R: Sequence[int], ofrom: int, oto: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return plotting endpoints for a divERGe TU form-factor component.

    divERGe stores a TU component as (R, ofrom, oto), with ``ofrom`` located
    in cell R and ``oto`` located in the home cell.  Therefore the physical
    displacement represented by the component is

        tau_oto - tau_ofrom - R.

    This is different from the model-construction hopping convention, where a
    hopping entry usually means from ``ofrom`` in the home cell to ``oto`` in
    cell R.  This helper is used only for visualizing/reconstructing TU output
    components read from O.dvg.
    """
    from_cell = (cell[0] + int(R[0]), cell[1] + int(R[1]))
    to_cell = cell

    x0 = realspace_coordinates(from_cell, ofrom)
    x1 = realspace_coordinates(to_cell, oto)
    return x0, x1


# -----------------------------------------------------------------------------
# Hopping/current helpers
# -----------------------------------------------------------------------------

def extract_nested(d: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, Mapping) or k not in cur:
            return default
        cur = cur[k]
    return cur


def resolved_params(metadata: Mapping[str, Any]) -> Dict[str, float]:
    # support several metadata layouts used during development
    p = extract_nested(metadata, "model_summary", "metadata", "resolved_params", default=None)
    if p is None:
        p = extract_nested(metadata, "model_summary", "metadata", "input_params", default=None)
    if p is None:
        p = metadata.get("input_params", {})
    return {
        "t": float(p.get("t", 1.0)),
        "mu": float(p.get("mu", 0.0)),
        "lambda_nn": float(p.get("lambda_nn", 0.0)),
        "lambda_nnn": float(p.get("lambda_nnn", 0.0)),
    }


def hopping_dict(metadata: Mapping[str, Any]) -> Dict[Tuple[Tuple[int, int, int], int, int, int], complex]:
    if km is None:
        return {}
    p = resolved_params(metadata)
    hops = km.spinful_hoppings(
        t=p["t"], mu=p["mu"], lambda_nn=p["lambda_nn"], lambda_nnn=p["lambda_nnn"], include_mu=True
    )
    d: Dict[Tuple[Tuple[int, int, int], int, int, int], complex] = {}
    for R, ofrom, oto, s1, s2, amp in hops:
        if int(s1) != int(s2):
            continue
        key = (tuple(int(x) for x in R), int(ofrom), int(oto), int(s1))
        if isinstance(amp, (tuple, list, np.ndarray)):
            z = complex(float(amp[0]), float(amp[1]))
        else:
            z = complex(amp)
        d[key] = d.get(key, 0.0 + 0.0j) + z
    return d


def current_from_sector(sector: str, tup: complex, tdn: complex, F: complex) -> Dict[str, float]:
    """Return charge/spin-z current proxies for a PH sector form factor.

    The overall normalization depends on the convention for normalized spin
    matrices and mode normalization.  Relative patterns and real/SOC splits are
    the intended output.
    """
    t_real = 0.5 * (tup + tdn)
    t_soc = 0.5 * (tup - tdn)
    sector = str(sector)

    if sector in {"ph-rho", "charge"}:
        Jc_real = 2.0 * np.imag(t_real * F)
        Jc_soc = 0.0
        Js_real = 0.0
        Js_soc = 2.0 * np.imag(t_soc * F)
    elif sector in {"ph-sz", "mag"}:
        Jc_real = 0.0
        Jc_soc = 2.0 * np.imag(t_soc * F)
        Js_real = 2.0 * np.imag(t_real * F)
        Js_soc = 0.0
    else:
        # transverse spin sectors: charge current is not naturally defined for a
        # spin-offdiagonal density with spin-diagonal hopping.  We still report a
        # same-sector proxy using the spin-independent hopping part.
        Jc_real = 0.0
        Jc_soc = 0.0
        Js_real = 2.0 * np.imag(t_real * F)
        Js_soc = 0.0

    return {
        "J_charge_realhop": float(Jc_real),
        "J_charge_sochop": float(Jc_soc),
        "J_charge": float(Jc_real + Jc_soc),
        "J_spin_z_realhop": float(Js_real),
        "J_spin_z_sochop": float(Js_soc),
        "J_spin_z": float(Js_real + Js_soc),
        "J_charge_split_error": float((Jc_real + Jc_soc) - (Jc_real + Jc_soc)),
        "J_spin_z_split_error": float((Js_real + Js_soc) - (Js_real + Js_soc)),
    }


# -----------------------------------------------------------------------------
# Mode selection
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


def select_rows(summary: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    df = summary.copy()
    if args.mode_key:
        keys = set(args.mode_key)
        return df[df["mode_key"].astype(str).isin(keys)].copy()
    if args.family:
        df = df[df["family"].astype(str).eq(args.family)]
    if args.sector:
        df = df[df["sector"].astype(str).isin(args.sector)]
    if args.q:
        df = df[df["q_label"].astype(str).isin(args.q)]
    if args.mode is not None:
        df = df[df["mode"].astype(int).isin([int(x) for x in args.mode])]
    if args.top:
        df = df.sort_values("strength_svd_KL", ascending=False).head(int(args.top))
    return df.copy()


def phase_factor(qred: Sequence[float], cell: Tuple[int, int], theta: float = 0.0) -> complex:
    return np.exp(1j * (float(theta) + 2.0 * np.pi * (float(qred[0]) * cell[0] + float(qred[1]) * cell[1])))


def theta_for_row(args: argparse.Namespace, row_index: int, n_rows: int) -> float:
    if args.theta is None or len(args.theta) == 0:
        return 0.0
    if len(args.theta) == 1:
        return float(args.theta[0])
    if len(args.theta) != n_rows:
        raise ValueError(f"Need one theta or one theta per selected row. Got {len(args.theta)} for {n_rows} rows.")
    return float(args.theta[row_index])


# -----------------------------------------------------------------------------
# Reconstruction
# -----------------------------------------------------------------------------

def build_combined_mode(rows: pd.DataFrame, modes: Mapping[str, np.ndarray], O: Any, metadata: Mapping[str, Any], args: argparse.Namespace, cell: Tuple[int, int]) -> np.ndarray:
    nk = infer_nk(O, metadata)
    total: Optional[np.ndarray] = None
    for j, (_, r) in enumerate(rows.iterrows()):
        key = str(r["mode_key"])
        if key not in modes:
            raise KeyError(f"mode_key {key} not found in instability_modes.npz")
        f = np.asarray(modes[key], dtype=complex).reshape(-1)
        qred = parse_q_reduced(r.get("q_reduced", ""), int(r.get("iq", 0)), nk)
        th = theta_for_row(args, j, len(rows))
        amp = phase_factor(qred, cell, th)
        if total is None:
            total = np.zeros_like(f, dtype=complex)
        total += amp * f
    if total is None:
        raise ValueError("No selected modes.")
    return total


def reconstruct(case_dir: Path, rows: pd.DataFrame, modes: Mapping[str, np.ndarray], args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    O = load_output(case_dir / "O.dvg")
    metadata = read_json(case_dir / "metadata.json")
    R, ofrom, oto = get_mi_arrays(O)
    masks = component_masks(O)
    hops = hopping_dict(metadata)
    observables = parse_csv_list(args.observable)
    bad = set(observables) - ALL_OBSERVABLES
    if bad:
        raise ValueError(f"Unknown observables: {sorted(bad)}")

    Nx, Ny = int(args.supercell[0]), int(args.supercell[1])
    first_sector = str(rows.iloc[0]["sector"])
    first_family = str(rows.iloc[0]["family"])

    site_rows: List[Dict[str, Any]] = []
    bond_rows: List[Dict[str, Any]] = []
    current_rows: List[Dict[str, Any]] = []
    pairing_rows: List[Dict[str, Any]] = []
    structural_summary_rows: List[Dict[str, Any]] = []
    bond_pair_rows: List[Dict[str, Any]] = []

    # structural diagnostics are done per selected Level-1 mode,
    # before optional real-space supercell/phase combination used for plotting.
    for _, r in rows.iterrows():
        key = str(r["mode_key"])
        if key in modes:
            one_summary, one_components = structural_bond_pair_diagnostics(O, np.asarray(modes[key], dtype=complex).reshape(-1), r, metadata)
            structural_summary_rows.append(one_summary)
            bond_pair_rows.extend(one_components)

    for cx in range(Nx):
        for cy in range(Ny):
            cell = (cx, cy)
            f = build_combined_mode(rows, modes, O, metadata, args, cell)
            for a in range(len(ofrom)):
                rtuple = tuple(int(x) for x in R[a])
                o0, o1 = int(ofrom[a]), int(oto[a])
                x0, x1 = endpoint_coordinates(cell, rtuple, o0, o1)
                common = {
                    "cell_x": cx,
                    "cell_y": cy,
                    "R0": rtuple[0],
                    "R1": rtuple[1],
                    "R2": rtuple[2] if len(rtuple) > 2 else 0,
                    "ofrom": o0,
                    "oto": o1,
                    "alpha": int(a),
                    "x_from": float(x0[0]),
                    "y_from": float(x0[1]),
                    "x_to": float(x1[0]),
                    "y_to": float(x1[1]),
                    "family": first_family,
                    "sector": first_sector,
                }

                if "density" in observables and bool(masks["onsite"][a]):
                    site_rows.append({
                        "cell_x": cx,
                        "cell_y": cy,
                        "orbital": o0,
                        "x": float(x0[0]),
                        "y": float(x0[1]),
                        "family": first_family,
                        "sector": first_sector,
                        "density_value": float(2.0 * np.real(f[a])),
                        "abs_complex_amplitude": float(abs(f[a])),
                        "real_complex_amplitude": float(np.real(f[a])),
                        "imag_complex_amplitude": float(np.imag(f[a])),
                    })

                if "bond" in observables and o0 != o1:
                    bond_rows.append({
                        **common,
                        "bond_real_value": float(2.0 * np.real(f[a])),
                        "bond_imag_value": float(2.0 * np.imag(f[a])),
                        "bond_abs_complex": float(abs(f[a])),
                        "bond_phase": float(np.angle(f[a])) if abs(f[a]) > 0 else 0.0,
                        "is_NN": bool(masks["nn"][a]),
                    })

                if "current" in observables and first_family == "ph" and o0 != o1:
                    spin_up = getattr(km, "SPIN_UP", 0) if km is not None else 0
                    spin_dn = getattr(km, "SPIN_DN", 1) if km is not None else 1
                    model_R = tuple(-int(x) for x in rtuple)
                    tup = hops.get((model_R, o0, o1, spin_up), 0.0 + 0.0j)
                    tdn = hops.get((model_R, o0, o1, spin_dn), 0.0 + 0.0j)
                    cur = current_from_sector(first_sector, tup, tdn, f[a])
                    current_rows.append({
                        **common,
                        **cur,
                        "t_up_real": float(np.real(tup)),
                        "t_up_imag": float(np.imag(tup)),
                        "t_down_real": float(np.real(tdn)),
                        "t_down_imag": float(np.imag(tdn)),
                        "is_NN": bool(masks["nn"][a]),
                    })

                if "pairing" in observables and first_family == "pp" and o0 != o1:
                    pairing_rows.append({
                        **common,
                        "pair_real_value": float(2.0 * np.real(f[a])),
                        "pair_imag_value": float(2.0 * np.imag(f[a])),
                        "pair_abs_complex": float(abs(f[a])),
                        "pair_phase": float(np.angle(f[a])) if abs(f[a]) > 0 else 0.0,
                        "is_NN": bool(masks["nn"][a]),
                    })

    summary_rows = []
    for _, r in rows.iterrows():
        summary_rows.append({k: r[k] for k in r.index})
    return site_rows, bond_rows, current_rows, pairing_rows, summary_rows, bond_pair_rows, structural_summary_rows


# -----------------------------------------------------------------------------
# Optional plotting
# -----------------------------------------------------------------------------

def plot_pattern_png(path: Path, site_rows: List[Dict[str, Any]], bond_rows: List[Dict[str, Any]], current_rows: List[Dict[str, Any]], pairing_rows: List[Dict[str, Any]]) -> None:
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(7, 6))

    if site_rows:
        xs = np.asarray([r["x"] for r in site_rows], dtype=float)
        ys = np.asarray([r["y"] for r in site_rows], dtype=float)
        vals = np.asarray([r["density_value"] for r in site_rows], dtype=float)
        sizes = 40 + 220 * np.abs(vals) / (np.max(np.abs(vals)) + 1e-12)
        ax.scatter(xs, ys, s=sizes, alpha=0.85, label="site")

    draw_rows = bond_rows if bond_rows else pairing_rows
    value_key = "bond_real_value" if bond_rows else "pair_real_value"
    if draw_rows:
        maxv = max(abs(float(r.get(value_key, 0.0))) for r in draw_rows) + 1e-12
        for r in draw_rows:
            val = abs(float(r.get(value_key, 0.0)))
            lw = 0.4 + 4.0 * val / maxv
            ax.plot([r["x_from"], r["x_to"]], [r["y_from"], r["y_to"]], lw=lw, alpha=0.65)

    if current_rows:
        key = "J_charge" if any(abs(float(r.get("J_charge", 0.0))) > 0 for r in current_rows) else "J_spin_z"
        maxJ = max(abs(float(r.get(key, 0.0))) for r in current_rows) + 1e-12
        for r in current_rows:
            J = float(r.get(key, 0.0))
            x0, y0, x1, y1 = float(r["x_from"]), float(r["y_from"]), float(r["x_to"]), float(r["y_to"])
            mx, my = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
            dx, dy = x1 - x0, y1 - y0
            scale = 0.28 * J / maxJ
            ax.arrow(mx - 0.5 * scale * dx, my - 0.5 * scale * dy, scale * dx, scale * dy,
                     head_width=0.035, length_includes_head=True, alpha=0.75)

    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(path.parent.name)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def analysis_name(rows: pd.DataFrame, args: argparse.Namespace) -> str:
    if args.name:
        return safe_name(args.name)
    if args.mode_key:
        return safe_name("keys_" + "_".join(args.mode_key[:2]) + ("_more" if len(args.mode_key) > 2 else ""))
    fam = str(rows.iloc[0]["family"])
    sec = str(rows.iloc[0]["sector"])
    qs = "_".join(str(x) for x in rows["q_label"].tolist())
    modes = "_".join(str(int(x)) for x in rows["mode"].tolist())
    obs = "_".join(parse_csv_list(args.observable))
    return safe_name(f"{fam}_{sec}_{qs}_m{modes}_{obs}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reconstruct calibrated loop-dressed TUFRG modes as real-space patterns.")
    p.add_argument("case_dir", help="case directory containing O.dvg and instability outputs")
    p.add_argument("--instability-dir", default="instability", help="directory inside case_dir containing instability_summary.csv and instability_modes.npz")
    p.add_argument("--mode-key", nargs="*", default=None, help="explicit mode_key(s) to reconstruct")
    p.add_argument("--family", choices=["ph", "pp"], default=None, help="select family if not using --mode-key")
    p.add_argument("--sector", nargs="*", default=None, help="select sector(s), e.g. ph-sz, ph-rho, pp-singlet")
    p.add_argument("--q", nargs="*", default=None, help="select q label(s), e.g. Gamma M1 M2 M3")
    p.add_argument("--mode", nargs="*", type=int, default=None, help="select mode index/indices from instability_summary")
    p.add_argument("--top", type=int, default=None, help="instead of exact mode selection, take top N rows after filters")
    p.add_argument("--observable", nargs="+", default=["density", "bond", "current"], help="density,bond,current,pairing; comma-separated accepted")
    p.add_argument("--theta", nargs="*", type=float, default=None, help="global phase(s), one or one per selected row")
    p.add_argument("--supercell", nargs=2, type=int, default=[2, 2], help="Nx Ny real-space cells")
    p.add_argument("--out", default="realspace_from_modes", help="output directory inside case_dir")
    p.add_argument("--name", default=None, help="optional analysis subdirectory name")
    p.add_argument("--no-plot", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    case_dir = Path(args.case_dir).resolve()
    summary, modes = load_instability(case_dir, args.instability_dir)
    rows = select_rows(summary, args)
    if rows.empty:
        raise SystemExit("No instability rows selected. Use --mode-key or filters such as --family ph --sector ph-sz --q Gamma --mode 0")
    if len(set(rows["family"].astype(str))) != 1 or len(set(rows["sector"].astype(str))) != 1:
        raise SystemExit("Selected rows must have a single family and single sector for one real-space reconstruction. Run separate commands otherwise.")

    site_rows, bond_rows, current_rows, pairing_rows, selected_rows, bond_pair_rows, structural_summary_rows = reconstruct(case_dir, rows, modes, args)
    outdir = case_dir / str(args.out) / analysis_name(rows, args)
    outdir.mkdir(parents=True, exist_ok=True)

    write_csv(outdir / "selected_modes.csv", selected_rows)
    write_csv(outdir / "site_pattern.csv", site_rows)
    write_csv(outdir / "bond_pattern.csv", bond_rows)
    write_csv(outdir / "current_pattern.csv", current_rows)
    write_csv(outdir / "pairing_pattern.csv", pairing_rows)
    write_csv(outdir / "bond_pair_components.csv", bond_pair_rows)
    write_csv(outdir / "structural_order_summary.csv", structural_summary_rows)

    if not args.no_plot:
        plot_pattern_png(outdir / "pattern.png", site_rows, bond_rows, current_rows, pairing_rows)

    print("Wrote real-space reconstruction to", outdir)
    print("Selected mode keys:")
    for k in rows["mode_key"].astype(str).tolist():
        print(" ", k)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
