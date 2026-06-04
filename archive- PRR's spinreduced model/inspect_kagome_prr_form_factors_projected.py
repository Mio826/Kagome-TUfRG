#!/usr/bin/env python3
"""
Inspect projected divERGe TU form factors for the kagome PRR-style mu=0 runs.

This script assumes the O.dvg files were produced with

    cfg.tu_channel_calc_project = -1

so that, in the SU(2) case,

    saved D channel = physical charge channel = 2D - C,
    saved C channel = sign-flipped magnetic channel.

It does not recompute the fRG flow. It reads the q-resolved decomposed vectors
Pvec/Cvec/Dvec and analyzes their support in the TU real-space form-factor basis
(R, ofrom, oto).

Main purpose:
    - distinguish onsite density-like vs bond-order-like form factors;
    - identify whether M-charge is CDW-like or cBO-like;
    - identify whether Gamma-magnetic is onsite FM-like;
    - identify whether Gamma-SC is onsite/bond A1-like.

Outputs:
    form_factor_summary.csv
    form_factor_top_components.csv

Usage:
    python inspect_kagome_prr_form_factors_projected.py ../outputs/kagome_prr_mu0_projected_small
    python inspect_kagome_prr_form_factors_projected.py ../outputs/kagome_prr_mu0_projected_small --case U1p0_V0p0
"""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import diverge.output as ro


CHANNEL_INFO = {
    # label: (saved channel letter, physical interpretation)
    "Gamma_mag": ("C", "q=0 magnetic / FM candidate; use -C_saved convention"),
    "Gamma_charge": ("D", "q=0 physical charge = 2D-C; beware uniform charge"),
    "Gamma_sc": ("P", "q=0 superconducting pairing"),
    "M_charge": ("D", "M-point physical charge = cBO/CDW sector"),
    "M_mag": ("C", "M-point magnetic = sBO/SDW sector; use -C_saved convention"),
    "M_sc": ("P", "M-point finite-Q pairing / PDW sector"),
}

PAIR_NAMES = {
    (0, 1): "AB", (1, 0): "AB",
    (1, 2): "BC", (2, 1): "BC",
    (0, 2): "AC", (2, 0): "AC",
}


@dataclass
class ModeData:
    value: float
    vector: np.ndarray
    mode_index: int
    abs_rank_value: float


def _arr(x):
    return np.asarray(x)


def normalize_phase(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=complex).reshape(-1).copy()
    if v.size == 0:
        return v
    imax = int(np.argmax(np.abs(v)))
    if abs(v[imax]) > 0:
        v *= np.exp(-1j * np.angle(v[imax]))
        if np.real(v[imax]) < 0:
            v *= -1
    return v


def get_qmesh(O, T) -> np.ndarray:
    idx = np.asarray(O.idx_ibz_in_bz, dtype=int)
    nk = np.asarray(T.nk, dtype=int).reshape(-1)
    N0, N1 = int(nk[0]), int(nk[1])

    def decode(ind: int) -> np.ndarray:
        ix = ind % N0
        iy = ind // N0
        return np.array([ix / N0, iy / N1, 0.0], dtype=float)

    return np.array([decode(int(i)) for i in idx], dtype=float)


def periodic_dist(q: np.ndarray, target: np.ndarray) -> float:
    d = np.asarray(q[:2], dtype=float) - np.asarray(target[:2], dtype=float)
    d = d - np.round(d)
    return float(np.linalg.norm(d))


def high_symmetry_indices(O, T) -> Dict[str, int]:
    qmesh = get_qmesh(O, T)
    targets = {
        "Gamma": np.array([0.0, 0.0, 0.0]),
        "M1": np.array([0.5, 0.0, 0.0]),
        "M2": np.array([0.0, 0.5, 0.0]),
        "M3": np.array([0.5, 0.5, 0.0]),
    }
    out = {}
    for name, target in targets.items():
        dists = np.array([periodic_dist(q, target) for q in qmesh])
        iq = int(np.argmin(dists))
        out[name] = iq
    return out


def decomp_mode(O, X: str, iq: int, mode_index: int = 0) -> ModeData:
    vals = np.asarray(getattr(O, f"{X}val"), dtype=float)
    vecs = np.asarray(getattr(O, f"{X}vec"))
    off = np.asarray(getattr(O, f"{X}off"), dtype=int)
    length = np.asarray(getattr(O, f"{X}len"), dtype=int)
    if int(length[iq]) <= mode_index:
        raise IndexError(f"channel {X}, iq={iq} has only len={length[iq]} modes")
    loc = int(off[iq] + mode_index)
    v = np.asarray(vecs[loc]).reshape(-1).astype(complex)
    v = normalize_phase(v)
    return ModeData(value=float(vals[loc]), vector=v, mode_index=mode_index, abs_rank_value=float(abs(vals[loc])))


def distances_from_tuff(O, T) -> np.ndarray:
    R = np.asarray(O.mi_to_R, dtype=int)
    ofrom = np.asarray(O.mi_to_ofrom, dtype=int)
    oto = np.asarray(O.mi_to_oto, dtype=int)
    n = len(ofrom)

    # Preferred: use T.tu_ff distance if available.
    try:
        tuff = np.asarray(T.tu_ff)
        if tuff.size > 0 and {"R", "ofrom", "oto", "d"}.issubset(set(tuff.dtype.names or [])):
            dmap = {}
            for rec in tuff:
                key = (tuple(np.asarray(rec["R"], dtype=int).tolist()), int(rec["ofrom"]), int(rec["oto"]))
                dmap[key] = float(rec["d"])
            out = np.full(n, np.nan, dtype=float)
            for i in range(n):
                key = (tuple(R[i].tolist()), int(ofrom[i]), int(oto[i]))
                if key in dmap:
                    out[i] = dmap[key]
            if np.all(np.isfinite(out)):
                return out
    except Exception:
        pass

    # Fallback: compute real-space distance in lattice coordinates.
    positions = np.asarray(T.positions, dtype=float)
    lattice = np.asarray(T.lattice, dtype=float)
    out = np.zeros(n, dtype=float)
    for i in range(n):
        # Unit-cell displacement plus orbital position difference.
        # The lattice convention is row-vector-like in the official examples.
        dr = np.asarray(R[i], dtype=float) @ lattice + positions[int(oto[i])] - positions[int(ofrom[i])]
        out[i] = float(np.linalg.norm(dr))
    return out


def classify_components(O, T, *, nn_dist: float = 0.5, nn_tol: float = 1e-6):
    R = np.asarray(O.mi_to_R, dtype=int)
    ofrom = np.asarray(O.mi_to_ofrom, dtype=int)
    oto = np.asarray(O.mi_to_oto, dtype=int)
    dist = distances_from_tuff(O, T)

    onsite = (ofrom == oto) & np.all(R == 0, axis=1)
    diagonal = (ofrom == oto)
    offdiag = ~diagonal
    nn = np.abs(dist - nn_dist) <= nn_tol
    nn_offdiag = nn & offdiag
    longer = ~(onsite | nn_offdiag)
    return {
        "R": R,
        "ofrom": ofrom,
        "oto": oto,
        "dist": dist,
        "onsite": onsite,
        "diagonal": diagonal,
        "offdiag": offdiag,
        "nn": nn,
        "nn_offdiag": nn_offdiag,
        "longer": longer,
    }


def norm_weight(v: np.ndarray, mask: np.ndarray) -> float:
    denom = float(np.sum(np.abs(v) ** 2))
    if denom <= 0:
        return 0.0
    return float(np.sum(np.abs(v[mask]) ** 2) / denom)


def overlap_template(v: np.ndarray, mask: np.ndarray, signs: Optional[np.ndarray] = None) -> float:
    v = np.asarray(v, dtype=complex).reshape(-1)
    t = np.zeros_like(v)
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return 0.0
    if signs is None:
        t[idx] = 1.0
    else:
        t[idx] = np.asarray(signs, dtype=complex)[idx]
    nv = np.linalg.norm(v)
    nt = np.linalg.norm(t)
    if nv <= 0 or nt <= 0:
        return 0.0
    return float(abs(np.vdot(t / nt, v / nv)) ** 2)


def bond_pair_summary(v: np.ndarray, comp: Dict[str, np.ndarray]) -> Dict[str, float | str]:
    ofrom = comp["ofrom"]
    oto = comp["oto"]
    nn = comp["nn_offdiag"]
    out: Dict[str, float | str] = {}
    best_pair = "none"
    best_abs = -1.0
    for pair in ["AB", "BC", "AC"]:
        mask = np.zeros_like(nn, dtype=bool)
        for i in range(len(mask)):
            if nn[i] and PAIR_NAMES.get((int(ofrom[i]), int(oto[i])), "") == pair:
                mask[i] = True
        amp = complex(np.sum(v[mask]))
        weight = norm_weight(v, mask)
        out[f"nn_{pair}_sum_real"] = float(np.real(amp))
        out[f"nn_{pair}_sum_imag"] = float(np.imag(amp))
        out[f"nn_{pair}_sum_abs"] = float(abs(amp))
        out[f"nn_{pair}_weight"] = float(weight)
        if abs(amp) > best_abs:
            best_abs = abs(amp)
            best_pair = pair
    out["dominant_nn_pair"] = best_pair
    out["dominant_nn_pair_sum_abs"] = float(best_abs)
    return out


def onsite_pattern(v: np.ndarray, comp: Dict[str, np.ndarray], norb: int) -> Dict[str, float]:
    out = {}
    R = comp["R"]
    ofrom = comp["ofrom"]
    oto = comp["oto"]
    for o in range(norb):
        mask = (ofrom == o) & (oto == o) & np.all(R == 0, axis=1)
        amp = complex(np.sum(v[mask]))
        out[f"onsite_o{o}_real"] = float(np.real(amp))
        out[f"onsite_o{o}_imag"] = float(np.imag(amp))
        out[f"onsite_o{o}_abs"] = float(abs(amp))
    return out


def infer_label(target: str, qname: str, X: str, summary: Dict[str, float | str]) -> str:
    onsite_w = float(summary.get("onsite_weight", 0.0))
    nn_w = float(summary.get("nn_bond_weight", 0.0))
    imag_frac = float(summary.get("imag_fraction", 0.0))
    onsite_a1 = float(summary.get("onsite_A1_overlap", 0.0))
    nn_a1 = float(summary.get("nn_A1_overlap", 0.0))

    if target == "Gamma_mag":
        if onsite_w > 0.5 and onsite_a1 > 0.5:
            return "FM_like_onsite_A1"
        if onsite_w > 0.5:
            return "q0_magnetic_onsite_nonuniform"
        return "q0_magnetic_bond_like"
    if target == "Gamma_sc":
        if onsite_w > 0.5 and onsite_a1 > 0.5:
            return "A1_SC_onsite_like"
        if nn_w > 0.5 and nn_a1 > 0.3:
            return "A1_SC_NN_bond_like"
        if nn_w > 0.5:
            return "NN_bond_SC_nontrivial_irrep_candidate"
        return "SC_mixed_form_factor"
    if target == "M_charge":
        if nn_w > onsite_w and nn_w > 0.3:
            return "cBO_like_NN_bond_form_factor_L1_like"
        if onsite_w > nn_w and onsite_w > 0.3:
            return "CDW_like_onsite_form_factor"
        return "M_charge_mixed_CDW_cBO"
    if target == "M_mag":
        if nn_w > onsite_w and nn_w > 0.3:
            return "sBO_like_NN_bond_form_factor_L1_like"
        if onsite_w > nn_w and onsite_w > 0.3:
            return "SDW_like_onsite_form_factor"
        return "M_magnetic_mixed_SDW_sBO"
    if "charge" in target and qname == "Gamma":
        return "q0_charge_PI_or_uniform_needs_projection"
    return "unclassified"


def analyze_mode(case: str, target: str, qname: str, iq: int, X: str, O, T, mode_index: int, top_n: int):
    mode = decomp_mode(O, X, iq, mode_index=mode_index)
    v = mode.vector
    comp = classify_components(O, T)
    n = v.size
    norm = float(np.sum(np.abs(v) ** 2))
    norb = int(np.asarray(T.n_orb).item()) if hasattr(T, "n_orb") else int(O.n_orb)

    summary: Dict[str, float | str | int] = {
        "case": case,
        "target": target,
        "qname": qname,
        "iq": int(iq),
        "channel_saved": X,
        "mode_index": int(mode_index),
        "channel_decomp_value": float(mode.value),
        "abs_channel_decomp_value": float(abs(mode.value)),
        "norm": norm,
        "onsite_weight": norm_weight(v, comp["onsite"]),
        "diagonal_weight": norm_weight(v, comp["diagonal"]),
        "offdiag_weight": norm_weight(v, comp["offdiag"]),
        "nn_bond_weight": norm_weight(v, comp["nn_offdiag"]),
        "longer_weight": norm_weight(v, comp["longer"]),
        "real_fraction": float(np.sum(np.real(v) ** 2) / max(norm, 1e-300)),
        "imag_fraction": float(np.sum(np.imag(v) ** 2) / max(norm, 1e-300)),
        "onsite_A1_overlap": overlap_template(v, comp["onsite"]),
        "nn_A1_overlap": overlap_template(v, comp["nn_offdiag"]),
    }
    summary.update(bond_pair_summary(v, comp))
    summary.update(onsite_pattern(v, comp, norb=norb))
    summary["inferred_form_factor"] = infer_label(target, qname, X, summary)

    order = np.argsort(-np.abs(v))[:top_n]
    top_rows = []
    R = comp["R"]
    ofrom = comp["ofrom"]
    oto = comp["oto"]
    dist = comp["dist"]
    for rank, idx in enumerate(order):
        if comp["onsite"][idx]:
            ctype = "onsite"
        elif comp["nn_offdiag"][idx]:
            ctype = "NN_bond"
        elif comp["diagonal"][idx]:
            ctype = "diagonal_longer"
        else:
            ctype = "longer_orbital"
        top_rows.append({
            "case": case,
            "target": target,
            "qname": qname,
            "iq": int(iq),
            "channel_saved": X,
            "mode_index": int(mode_index),
            "rank": int(rank),
            "component_index": int(idx),
            "R0": int(R[idx, 0]),
            "R1": int(R[idx, 1]),
            "R2": int(R[idx, 2]),
            "ofrom": int(ofrom[idx]),
            "oto": int(oto[idx]),
            "pair": PAIR_NAMES.get((int(ofrom[idx]), int(oto[idx])), "diag" if ofrom[idx] == oto[idx] else "other"),
            "distance": float(dist[idx]),
            "component_type": ctype,
            "value_real": float(np.real(v[idx])),
            "value_imag": float(np.imag(v[idx])),
            "value_abs": float(abs(v[idx])),
        })

    return summary, top_rows


def iter_cases(root: str, selected: Optional[str]) -> List[str]:
    if selected:
        return [selected]
    cases = []
    for name in sorted(os.listdir(root)):
        if os.path.exists(os.path.join(root, name, "O.dvg")):
            cases.append(name)
    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="output root, e.g. ../outputs/kagome_prr_mu0_projected_small")
    ap.add_argument("--case", default=None, help="only analyze one case")
    ap.add_argument("--top", type=int, default=16, help="number of top components to print/save")
    ap.add_argument("--mode", type=int, default=0, help="decomposition mode index at each q")
    ap.add_argument("--include-extra", action="store_true", help="also include Gamma_charge, M_mag, M_sc")
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    cases = iter_cases(root, args.case)
    print("root =", root)
    print("cases =", cases)

    targets_base = ["Gamma_mag", "Gamma_sc", "M_charge"]
    targets_extra = ["Gamma_charge", "M_mag", "M_sc"]
    targets = targets_base + (targets_extra if args.include_extra else [])

    summary_rows = []
    top_rows_all = []

    for case in cases:
        Ofile = os.path.join(root, case, "O.dvg")
        Tfile = os.path.join(root, case, "T.dvg")
        if not os.path.exists(Ofile):
            print("skip missing", case)
            continue
        O = ro.read(Ofile)
        T = ro.read(Tfile)
        qidx = high_symmetry_indices(O, T)
        qmesh = get_qmesh(O, T)

        print("\n" + "#" * 110)
        print("CASE", case)
        print("#" * 110)
        print("q indices:", {k: int(v) for k, v in qidx.items()})

        for target in targets:
            X, interp = CHANNEL_INFO[target]
            qnames: List[str]
            if target.startswith("Gamma"):
                qnames = ["Gamma"]
            else:
                qnames = ["M1", "M2", "M3"]

            for qname in qnames:
                iq = qidx[qname]
                try:
                    summary, top_rows = analyze_mode(
                        case, target, qname, iq, X, O, T, mode_index=args.mode, top_n=args.top
                    )
                except Exception as exc:
                    print(f"  {target:13s} {qname:5s}: FAILED {exc}")
                    continue
                summary["q0"] = float(qmesh[iq, 0])
                summary["q1"] = float(qmesh[iq, 1])
                summary["q2"] = float(qmesh[iq, 2])
                summary["interpretation_note"] = interp
                summary_rows.append(summary)
                top_rows_all.extend(top_rows)

                print(
                    f"  {target:13s} {qname:5s} X={X} val={summary['channel_decomp_value']:+.6e} "
                    f"onsite={summary['onsite_weight']:.3f} nn={summary['nn_bond_weight']:.3f} "
                    f"A1site={summary['onsite_A1_overlap']:.3f} A1nn={summary['nn_A1_overlap']:.3f} "
                    f"label={summary['inferred_form_factor']}"
                )
                print(
                    f"      onsite pattern abs=({summary.get('onsite_o0_abs',0):.3f}, "
                    f"{summary.get('onsite_o1_abs',0):.3f}, {summary.get('onsite_o2_abs',0):.3f}); "
                    f"dominant NN pair={summary.get('dominant_nn_pair','none')}"
                )

    out_summary = os.path.join(root, "form_factor_summary.csv")
    out_top = os.path.join(root, "form_factor_top_components.csv")

    if summary_rows:
        keys = sorted({k for row in summary_rows for k in row.keys()})
        with open(out_summary, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for row in summary_rows:
                w.writerow(row)
        print("\nwrote", out_summary)

    if top_rows_all:
        keys = sorted({k for row in top_rows_all for k in row.keys()})
        with open(out_top, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for row in top_rows_all:
                w.writerow(row)
        print("wrote", out_top)


if __name__ == "__main__":
    main()
