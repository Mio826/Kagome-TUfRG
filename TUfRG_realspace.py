#!/usr/bin/env python3
# coding: utf-8
"""
TUfRG_realspace_v2.py
Version: v2-current-split-2026-05-12

Chosen-mode real-space reconstruction for spinful kagome Kane-Mele divERGe
TUFRG outputs.

Purpose
-------
This script is intentionally a *selected-candidate* tool.  First run
TUfRG_diagnosis.py to identify a promising (channel, q, mode).  Then use this
script to reconstruct site/bond/current/pairing patterns in a finite real-space
supercell.

It supports:
  * single-Q reconstruction;
  * manual 3Q reconstruction from M1/M2/M3 modes;
  * PH sectors: ph-charge, ph-spin-z;
  * PP sectors: pp-singlet, pp-triplet0, pp-triplet-plus, pp-triplet-minus;
  * density, bond, current, pairing observables;
  * CSV outputs and simple PNG visualization;
  * limited first-pass template overlaps for Q=0 current-like patterns.

Interpretation warning
----------------------
The reconstructed pattern is a form-factor direction inferred from the saved
channel decomposition.  It is not a self-consistent mean-field expectation value
unless further analysis is done.  Template overlaps are evidence, not automatic
phase labels.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

import TUfRG_model as km
import TUfRG_diagnosis as diag


PH_SECTORS = {"ph-charge", "ph-spin-z"}
PP_SECTORS = {"pp-singlet", "pp-triplet0", "pp-triplet-plus", "pp-triplet-minus"}
ALL_SECTORS = PH_SECTORS | PP_SECTORS


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def parse_csv_list(values: Sequence[str]) -> List[str]:
    out: List[str] = []
    for v in values:
        out.extend([x.strip() for x in str(v).split(",") if x.strip()])
    return out


def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.+-]+", "_", s).strip("_")


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
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def resolve_qs(data: diag.CaseData, q_labels: Sequence[str]) -> List[Tuple[str, int, Tuple[float, float, float]]]:
    qmap = diag.q_labels_to_indices(data, q_labels)
    return [(lab, iq, diag.q_reduced_from_iq(iq, data.nk)) for lab, iq in qmap.items()]


def mode_list_for_qs(modes: Sequence[int], n_q: int) -> List[int]:
    if len(modes) == 1:
        return [int(modes[0])] * n_q
    if len(modes) != n_q:
        raise ValueError(f"Need either one --mode or one mode per q. Got {len(modes)} modes for {n_q} q points.")
    return [int(x) for x in modes]


def theta_list_for_qs(theta: Optional[Sequence[float]], n_q: int) -> List[float]:
    if theta is None or len(theta) == 0:
        return [0.0] * n_q
    if len(theta) == 1:
        return [float(theta[0])] * n_q
    if len(theta) != n_q:
        raise ValueError(f"Need either one --theta or one theta per q. Got {len(theta)} theta values for {n_q} q points.")
    return [float(x) for x in theta]


# -----------------------------------------------------------------------------
# Spin-sector extraction
# -----------------------------------------------------------------------------

def sector_vector(v: np.ndarray, sector: str) -> np.ndarray:
    comps = diag.spin_combinations(v)
    key = sector.replace("-", "_")
    if key not in comps:
        raise ValueError(f"sector {sector!r} not available. Choices: {sorted(ALL_SECTORS)}")
    return comps[key]


def validate_channel_sector(channel: str, sector: str, observables: Sequence[str]) -> None:
    channel = channel.upper()
    if channel == "P":
        if sector not in PP_SECTORS:
            raise ValueError("P channel is pairing-like; use pp-singlet/pp-triplet0/pp-triplet-plus/pp-triplet-minus")
        bad = set(observables) - {"pairing", "bond"}
        if bad:
            raise ValueError(f"P channel does not support observables {sorted(bad)}. Use --observable pairing.")
    else:
        if sector not in PH_SECTORS:
            raise ValueError("C/D channel is PH-like; use ph-charge or ph-spin-z")
        bad = set(observables) - {"density", "bond", "current"}
        if bad:
            raise ValueError(f"C/D channel does not support observables {sorted(bad)}")


# -----------------------------------------------------------------------------
# Hopping dictionary for physical current proxy
# -----------------------------------------------------------------------------

def resolved_params_from_metadata(metadata: Mapping[str, Any]) -> Dict[str, float]:
    rp = diag.extract_nested(metadata, "model_summary", "metadata", "resolved_params", default=None)
    if rp is None:
        rp = diag.extract_nested(metadata, "model_summary", "metadata", "input_params", default={})
    return {
        "t": float(rp.get("t", 1.0)),
        "mu": float(rp.get("mu", 0.0)),
        "lambda_nn": float(rp.get("lambda_nn", 0.0)),
        "lambda_nnn": float(rp.get("lambda_nnn", 0.0)),
    }


def _hopping_dict_from_hops(hops: Sequence[Tuple[Any, int, int, int, int, Any]]) -> Dict[Tuple[Tuple[int, int, int], int, int, int], complex]:
    """Convert a spinful_hoppings list into a lookup dictionary.

    Key is (R_tuple, ofrom, oto, spin).  Only spin-diagonal hoppings are kept,
    because the current formulas implemented here assume Sz-conserving hopping.
    """
    d: Dict[Tuple[Tuple[int, int, int], int, int, int], complex] = {}
    for R, ofrom, oto, s1, s2, amp in hops:
        if int(s1) != int(s2):
            continue
        key = (tuple(int(x) for x in R), int(ofrom), int(oto), int(s1))
        d[key] = d.get(key, 0.0) + complex(amp[0], amp[1])
    return d


def hopping_dict_parts_from_metadata(
    metadata: Mapping[str, Any],
) -> Tuple[
    Dict[Tuple[Tuple[int, int, int], int, int, int], complex],
    Dict[Tuple[Tuple[int, int, int], int, int, int], complex],
    Dict[Tuple[Tuple[int, int, int], int, int, int], complex],
]:
    """Return total, real-hopping-only, and SOC-hopping-only dictionaries.

    The total hopping is decomposed as

        t^sigma_ij = t^real_ij + t^SOC,sigma_ij.

    This split is important for current fingerprints.  The real-hopping part
    mainly measures the imaginary-bond content of the mode itself, while the
    SOC-hopping part can convert a mostly-real bond mode into a current-like
    metric because the bare Kane-Mele hopping is imaginary.
    """
    p = resolved_params_from_metadata(metadata)

    # Exclude chemical potential from current dictionaries.  Current rows are
    # only generated for off-diagonal bonds, but excluding mu makes the split
    # explicit and safer.
    hops_total = km.spinful_hoppings(
        t=p["t"], mu=0.0, lambda_nn=p["lambda_nn"], lambda_nnn=p["lambda_nnn"], include_mu=False
    )
    hops_real = km.spinful_hoppings(
        t=p["t"], mu=0.0, lambda_nn=0.0, lambda_nnn=0.0, include_mu=False
    )
    hops_soc = km.spinful_hoppings(
        t=0.0, mu=0.0, lambda_nn=p["lambda_nn"], lambda_nnn=p["lambda_nnn"], include_mu=False
    )

    return (
        _hopping_dict_from_hops(hops_total),
        _hopping_dict_from_hops(hops_real),
        _hopping_dict_from_hops(hops_soc),
    )


def hopping_dict_from_metadata(metadata: Mapping[str, Any]) -> Dict[Tuple[Tuple[int, int, int], int, int, int], complex]:
    """Backward-compatible total-hopping dictionary."""
    total, _, _ = hopping_dict_parts_from_metadata(metadata)
    return total


# -----------------------------------------------------------------------------
# Reconstruction core
# -----------------------------------------------------------------------------

def load_selected_modes(
    data: diag.CaseData,
    channel: str,
    q_labels: Sequence[str],
    modes: Sequence[int],
    theta: Optional[Sequence[float]],
) -> List[Dict[str, Any]]:
    q_info = resolve_qs(data, q_labels)
    mode_list = mode_list_for_qs(modes, len(q_info))
    theta_list = theta_list_for_qs(theta, len(q_info))
    out: List[Dict[str, Any]] = []
    for (q_label, iq, qred), mode, th in zip(q_info, mode_list, theta_list):
        val, vec, loc = diag.get_mode(data.O, channel, iq, mode)
        out.append({
            "q_label": q_label,
            "iq": iq,
            "q_reduced": qred,
            "mode": int(mode),
            "theta": float(th),
            "raw_value": float(val),
            "global_loc": int(loc),
            "vec": vec,
        })
    return out


def phase_factor(qred: Sequence[float], cell: Tuple[int, int], theta: float = 0.0) -> complex:
    return np.exp(1j * (float(theta) + 2.0 * np.pi * (float(qred[0]) * cell[0] + float(qred[1]) * cell[1])))


def combined_alpha_vector(selected: List[Dict[str, Any]], sector: str, cell: Tuple[int, int]) -> np.ndarray:
    total: Optional[np.ndarray] = None
    for item in selected:
        f = sector_vector(item["vec"], sector)
        amp = phase_factor(item["q_reduced"], cell, item["theta"])
        if total is None:
            total = np.zeros_like(f, dtype=complex)
        total += amp * f
    assert total is not None
    return total


def combined_spin_diag_vectors(selected: List[Dict[str, Any]], cell: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    up: Optional[np.ndarray] = None
    dn: Optional[np.ndarray] = None
    for item in selected:
        v = np.asarray(item["vec"], dtype=complex)
        amp = phase_factor(item["q_reduced"], cell, item["theta"])
        if up is None:
            up = np.zeros_like(v[0, 0, :], dtype=complex)
            dn = np.zeros_like(v[1, 1, :], dtype=complex)
        up += amp * v[0, 0, :]
        dn += amp * v[1, 1, :]
    assert up is not None and dn is not None
    return up, dn


def realspace_coordinates(cell: Tuple[int, int], orbital: int) -> np.ndarray:
    lattice, pos = km.kagome_lattice_and_positions()
    return cell[0] * lattice[0, :2] + cell[1] * lattice[1, :2] + pos[int(orbital), :2]


def endpoint_coordinates(cell: Tuple[int, int], R: Sequence[int], ofrom: int, oto: int) -> Tuple[np.ndarray, np.ndarray]:
    x0 = realspace_coordinates(cell, ofrom)
    to_cell = (cell[0] + int(R[0]), cell[1] + int(R[1]))
    x1 = realspace_coordinates(to_cell, oto)
    return x0, x1


def reconstruct_patterns(
    data: diag.CaseData,
    selected: List[Dict[str, Any]],
    *,
    channel: str,
    sector: str,
    observables: Sequence[str],
    supercell: Tuple[int, int],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    R, ofrom, oto = diag.get_mi_arrays(data.O)
    masks = diag.component_masks(data.O)
    hop_total_dict, hop_real_dict, hop_soc_dict = hopping_dict_parts_from_metadata(data.metadata)
    Nx, Ny = int(supercell[0]), int(supercell[1])

    site_rows: List[Dict[str, Any]] = []
    bond_rows: List[Dict[str, Any]] = []
    current_rows: List[Dict[str, Any]] = []
    pairing_rows: List[Dict[str, Any]] = []

    for cx in range(Nx):
        for cy in range(Ny):
            cell = (cx, cy)
            f = combined_alpha_vector(selected, sector, cell)
            fup, fdn = combined_spin_diag_vectors(selected, cell) if channel.upper() != "P" else (None, None)

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
                    "x_from": float(x0[0]),
                    "y_from": float(x0[1]),
                    "x_to": float(x1[0]),
                    "y_to": float(x1[1]),
                    "alpha": a,
                }

                if "density" in observables and bool(masks["onsite"][a]):
                    site_rows.append({
                        "cell_x": cx,
                        "cell_y": cy,
                        "orbital": o0,
                        "x": float(x0[0]),
                        "y": float(x0[1]),
                        "sector": sector,
                        "density_value": float(2.0 * np.real(f[a])),
                        "abs_complex_amplitude": float(abs(f[a])),
                        "real_complex_amplitude": float(np.real(f[a])),
                        "imag_complex_amplitude": float(np.imag(f[a])),
                    })

                if "bond" in observables and o0 != o1:
                    bond_rows.append({
                        **common,
                        "sector": sector,
                        "bond_real_value": float(2.0 * np.real(f[a])),
                        "bond_imag_value": float(2.0 * np.imag(f[a])),
                        "bond_abs_complex": float(abs(f[a])),
                        "bond_phase": float(np.angle(f[a])) if abs(f[a]) > 0 else 0.0,
                        "is_NN": bool(masks["nn"][a]),
                    })

                if "current" in observables and channel.upper() != "P" and o0 != o1:
                    key_up = (rtuple, o0, o1, km.SPIN_UP)
                    key_dn = (rtuple, o0, o1, km.SPIN_DN)

                    tup_total = hop_total_dict.get(key_up, 0.0)
                    tdn_total = hop_total_dict.get(key_dn, 0.0)
                    tup_real = hop_real_dict.get(key_up, 0.0)
                    tdn_real = hop_real_dict.get(key_dn, 0.0)
                    tup_soc = hop_soc_dict.get(key_up, 0.0)
                    tdn_soc = hop_soc_dict.get(key_dn, 0.0)

                    Jup_real = 2.0 * np.imag(tup_real * fup[a]) if fup is not None else 0.0
                    Jdn_real = 2.0 * np.imag(tdn_real * fdn[a]) if fdn is not None else 0.0
                    Jup_soc = 2.0 * np.imag(tup_soc * fup[a]) if fup is not None else 0.0
                    Jdn_soc = 2.0 * np.imag(tdn_soc * fdn[a]) if fdn is not None else 0.0
                    Jup = 2.0 * np.imag(tup_total * fup[a]) if fup is not None else 0.0
                    Jdn = 2.0 * np.imag(tdn_total * fdn[a]) if fdn is not None else 0.0

                    current_rows.append({
                        **common,
                        "sector": sector,
                        # Total current fingerprint using full hopping.
                        "J_up": float(Jup),
                        "J_down": float(Jdn),
                        "J_charge": float(Jup + Jdn),
                        "J_spin_z": float(Jup - Jdn),
                        # Contribution from spin-independent real hopping only.
                        "J_up_realhop": float(Jup_real),
                        "J_down_realhop": float(Jdn_real),
                        "J_charge_realhop": float(Jup_real + Jdn_real),
                        "J_spin_z_realhop": float(Jup_real - Jdn_real),
                        # Contribution from Kane-Mele SOC hopping only.
                        "J_up_sochop": float(Jup_soc),
                        "J_down_sochop": float(Jdn_soc),
                        "J_charge_sochop": float(Jup_soc + Jdn_soc),
                        "J_spin_z_sochop": float(Jup_soc - Jdn_soc),
                        # Hopping amplitudes used in the decomposition.
                        "t_up_real": float(np.real(tup_total)),
                        "t_up_imag": float(np.imag(tup_total)),
                        "t_down_real": float(np.real(tdn_total)),
                        "t_down_imag": float(np.imag(tdn_total)),
                        "t_up_realhop_real": float(np.real(tup_real)),
                        "t_up_realhop_imag": float(np.imag(tup_real)),
                        "t_down_realhop_real": float(np.real(tdn_real)),
                        "t_down_realhop_imag": float(np.imag(tdn_real)),
                        "t_up_sochop_real": float(np.real(tup_soc)),
                        "t_up_sochop_imag": float(np.imag(tup_soc)),
                        "t_down_sochop_real": float(np.real(tdn_soc)),
                        "t_down_sochop_imag": float(np.imag(tdn_soc)),
                        # Numerical consistency checks.  These should be near zero.
                        "J_up_split_error": float(Jup - (Jup_real + Jup_soc)),
                        "J_down_split_error": float(Jdn - (Jdn_real + Jdn_soc)),
                        "J_charge_split_error": float((Jup + Jdn) - ((Jup_real + Jdn_real) + (Jup_soc + Jdn_soc))),
                        "J_spin_z_split_error": float((Jup - Jdn) - ((Jup_real - Jdn_real) + (Jup_soc - Jdn_soc))),
                        "is_NN": bool(masks["nn"][a]),
                    })

                if "pairing" in observables and channel.upper() == "P" and o0 != o1:
                    pairing_rows.append({
                        **common,
                        "sector": sector,
                        "pair_real_value": float(2.0 * np.real(f[a])),
                        "pair_imag_value": float(2.0 * np.imag(f[a])),
                        "pair_abs_complex": float(abs(f[a])),
                        "pair_phase": float(np.angle(f[a])) if abs(f[a]) > 0 else 0.0,
                        "is_NN": bool(masks["nn"][a]),
                    })

    return site_rows, bond_rows, current_rows, pairing_rows


# -----------------------------------------------------------------------------
# Template overlap: limited first-pass prototypes
# -----------------------------------------------------------------------------

def vector_from_rows(rows: List[Dict[str, Any]], key: str, *, only_nn: bool = True) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    filtered = [r for r in rows if (not only_nn or bool(r.get("is_NN", False)))]
    return np.asarray([float(r.get(key, 0.0)) for r in filtered], dtype=float), filtered


def q0_current_templates(rows: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    """Prototype 1x1 current templates on directed NN bonds.

    These are first-pass basis patterns intended to distinguish gross Q=0
    current structures. They should not be treated as a complete symmetry
    classification of the flux paper.
    """
    _, filtered = vector_from_rows(rows, "J_charge", only_nn=True)
    out: Dict[str, List[float]] = {
        "Q0_Nagaosa_like_KM_eta": [],
        "Q0_flow_all_canonical": [],
        "Q0_flow_AB_vs_rest": [],
        "Q0_flow_AC_vs_rest": [],
        "Q0_flow_BC_vs_rest": [],
    }
    eta = km.PAIR_ETA_DEFAULT
    for r in filtered:
        o0, o1 = int(r["ofrom"]), int(r["oto"])
        try:
            pair = km.PAIR_LABEL[(o0, o1)]
            orient = km.canonical_direction_sign(o0, o1)
            out["Q0_Nagaosa_like_KM_eta"].append(float(eta[pair] * orient))
            out["Q0_flow_all_canonical"].append(float(orient))
            out["Q0_flow_AB_vs_rest"].append(float(orient * (+1 if pair == "AB" else -1)))
            out["Q0_flow_AC_vs_rest"].append(float(orient * (+1 if pair == "AC" else -1)))
            out["Q0_flow_BC_vs_rest"].append(float(orient * (+1 if pair == "BC" else -1)))
        except Exception:
            for k in out:
                out[k].append(0.0)
    return {k: np.asarray(v, dtype=float) for k, v in out.items()}


def normalized_overlap(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    if len(x) != len(y) or len(x) == 0:
        return 0.0, 0.0
    nx = float(np.linalg.norm(x))
    ny = float(np.linalg.norm(y))
    if nx <= 0 or ny <= 0:
        return 0.0, 0.0
    signed = float(np.dot(x, y) / (nx * ny))
    return float(signed * signed), signed


def template_overlap_rows(current_rows: List[Dict[str, Any]], key: str = "J_charge") -> List[Dict[str, Any]]:
    x, filtered = vector_from_rows(current_rows, key, only_nn=True)
    out: List[Dict[str, Any]] = []
    for name, t in q0_current_templates(current_rows).items():
        ov, sov = normalized_overlap(x, t)
        out.append({
            "template_name": name,
            "template_family": "Q0_current_prototype",
            "observable_key": key,
            "overlap_squared": ov,
            "signed_overlap": sov,
            "norm_pattern": float(np.linalg.norm(x)),
            "norm_template": float(np.linalg.norm(t)),
            "comment": "prototype template; use real-space plot/CSV for final interpretation",
        })
    return out


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def plot_pattern_png(path: Path, site_rows: List[Dict[str, Any]], bond_rows: List[Dict[str, Any]], current_rows: List[Dict[str, Any]], pairing_rows: List[Dict[str, Any]]) -> None:
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(7, 6))

    # Plot sites.
    if site_rows:
        xs = np.asarray([r["x"] for r in site_rows], dtype=float)
        ys = np.asarray([r["y"] for r in site_rows], dtype=float)
        vals = np.asarray([r["density_value"] for r in site_rows], dtype=float)
        sizes = 40 + 200 * (np.abs(vals) / (np.max(np.abs(vals)) + 1e-12))
        ax.scatter(xs, ys, s=sizes, alpha=0.8, label="site density")

    # Plot bonds or pairing as line width.
    draw_rows = bond_rows if bond_rows else pairing_rows
    value_key = "bond_real_value" if bond_rows else "pair_real_value"
    for r in draw_rows:
        val = abs(float(r.get(value_key, 0.0)))
        lw = 0.5 + 4.0 * val / (max(abs(float(rr.get(value_key, 0.0))) for rr in draw_rows) + 1e-12)
        ax.plot([r["x_from"], r["x_to"]], [r["y_from"], r["y_to"]], lw=lw, alpha=0.65)

    # Plot currents as quiver arrows on bond centers.
    if current_rows:
        maxJ = max(abs(float(r.get("J_charge", 0.0))) for r in current_rows) + 1e-12
        for r in current_rows:
            J = float(r.get("J_charge", 0.0))
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
    if site_rows:
        ax.legend(loc="best")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main analysis
# -----------------------------------------------------------------------------

def analysis_name(args: argparse.Namespace) -> str:
    qs = "".join(args.q)
    modes = "_".join(str(x) for x in args.mode)
    combine = "3Q" if args.combine == "3Q" else "singleQ"
    obs = "_".join(parse_csv_list(args.observable))
    return safe_name(f"{args.channel}_{args.sector}_{qs}_m{modes}_{combine}_{obs}")


def run_realspace(args: argparse.Namespace) -> Path:
    data = diag.load_case(Path(args.case_dir))
    observables = parse_csv_list(args.observable)
    validate_channel_sector(args.channel, args.sector, observables)

    selected = load_selected_modes(data, args.channel, args.q, args.mode, args.theta)
    if args.combine == "singleQ" and len(selected) != 1:
        raise ValueError("--combine singleQ requires exactly one --q. Use --combine 3Q for multiple q labels.")
    if args.combine == "3Q" and len(selected) < 2:
        raise ValueError("--combine 3Q requires multiple --q labels, usually M1 M2 M3.")

    site_rows, bond_rows, current_rows, pairing_rows = reconstruct_patterns(
        data,
        selected,
        channel=args.channel,
        sector=args.sector,
        observables=observables,
        supercell=(int(args.supercell[0]), int(args.supercell[1])),
    )

    outroot = Path(args.case_dir).resolve() / str(args.out)
    outdir = outroot / analysis_name(args)
    outdir.mkdir(parents=True, exist_ok=True)

    write_csv(outdir / "site_pattern.csv", site_rows)
    write_csv(outdir / "bond_pattern.csv", bond_rows)
    write_csv(outdir / "current_pattern.csv", current_rows)
    write_csv(outdir / "pairing_pattern.csv", pairing_rows)

    summary_rows = []
    for item in selected:
        summary_rows.append({
            "channel": args.channel,
            "sector": args.sector,
            "q_label": item["q_label"],
            "iq": item["iq"],
            "q_reduced": str(item["q_reduced"]),
            "mode": item["mode"],
            "global_loc": item["global_loc"],
            "raw_value": item["raw_value"],
            "theta": item["theta"],
        })
    write_csv(outdir / "pattern_summary.csv", summary_rows)

    if args.template_overlap:
        if current_rows:
            rows = template_overlap_rows(current_rows, key="J_charge")
            write_csv(outdir / "template_overlap.csv", rows)
        else:
            write_csv(outdir / "template_overlap.csv", [{"comment": "template overlap currently implemented for current rows only"}])

    if not args.no_plot:
        plot_pattern_png(outdir / "pattern.png", site_rows, bond_rows, current_rows, pairing_rows)

    print("Wrote real-space analysis to", outdir)
    print("Selected modes:")
    for r in summary_rows:
        print(" ", r)
    return outdir


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reconstruct selected TUFRG mode as a real-space kagome pattern.")
    p.add_argument("case_dir", help="runner output case directory containing O.dvg and metadata.json")
    p.add_argument("--channel", required=True, choices=["P", "C", "D"], help="saved divERGe channel to read")
    p.add_argument("--sector", required=True, choices=sorted(ALL_SECTORS), help="spin/physical sector to reconstruct")
    p.add_argument("--q", nargs="+", required=True, help="q label(s), e.g. Gamma or M1 M2 M3")
    p.add_argument("--mode", nargs="+", type=int, required=True, help="mode index/indices; one value or one per q")
    p.add_argument("--observable", nargs="+", required=True, help="density,bond,current for PH; pairing for P")
    p.add_argument("--combine", choices=["singleQ", "3Q"], default="singleQ")
    p.add_argument("--theta", nargs="+", type=float, default=None, help="global phase(s) in radians; one or one per q")
    p.add_argument("--supercell", nargs=2, type=int, default=[2, 2], help="Nx Ny real-space cells to reconstruct")
    p.add_argument("--template-overlap", action="store_true", help="compute limited prototype template overlaps")
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--out", default="realspace", help="output subdirectory name inside case_dir")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    run_realspace(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
