#!/usr/bin/env python3
"""
Inspect subleading M-charge form factors and onsite-projected M-charge residual modes
for projected divERGe TUFRG kagome PRR-style runs.

Assumes O.dvg was produced with:
    cfg.tu_channel_calc_project = -1
so saved D channel is the physical charge channel, D_saved = 2D - C.

Main checks:
  1. For M1/M2/M3 charge channel, inspect modes 0..N-1 from Dval/Dvec.
  2. For each M point, project out onsite diagonal form-factor subspace
     (R=0, ofrom=oto) from the full Dchannel matrix and decompose the residual.
     If the leading residual has large NN-bond weight, it is a cBO-like candidate.

Outputs:
  Mcharge_subleading_modes.csv
  Mcharge_subleading_top_components.csv
  Mcharge_onsite_projected_residual.csv
  Mcharge_onsite_projected_top_components.csv

Usage:
  python inspect_kagome_prr_Mcharge_subleading_projected.py ../outputs/kagome_prr_mu0_projected_small
  python inspect_kagome_prr_Mcharge_subleading_projected.py ../outputs/kagome_prr_mu0_projected_small --case U0p5_V1p0 --nmodes 8
"""
from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import diverge.output as ro

PAIR_NAMES = {
    (0, 1): "AB", (1, 0): "AB",
    (1, 2): "BC", (2, 1): "BC",
    (0, 2): "AC", (2, 0): "AC",
}


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
    out = []
    for ind in idx:
        ix = int(ind) % N0
        iy = int(ind) // N0
        out.append([ix / N0, iy / N1, 0.0])
    return np.asarray(out, dtype=float)


def periodic_dist(q: np.ndarray, target: np.ndarray) -> float:
    d = np.asarray(q[:2], dtype=float) - np.asarray(target[:2], dtype=float)
    d -= np.round(d)
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
        out[name] = int(np.argmin(dists))
    return out


def distances_from_tuff(O, T) -> np.ndarray:
    R = np.asarray(O.mi_to_R, dtype=int)
    ofrom = np.asarray(O.mi_to_ofrom, dtype=int)
    oto = np.asarray(O.mi_to_oto, dtype=int)
    n = len(ofrom)
    try:
        tuff = np.asarray(T.tu_ff)
        names = set(tuff.dtype.names or [])
        if tuff.size and {"R", "ofrom", "oto", "d"}.issubset(names):
            dmap = {}
            for rec in tuff:
                key = (tuple(np.asarray(rec["R"], dtype=int).tolist()), int(rec["ofrom"]), int(rec["oto"]))
                dmap[key] = float(rec["d"])
            out = np.full(n, np.nan)
            for i in range(n):
                key = (tuple(R[i].tolist()), int(ofrom[i]), int(oto[i]))
                if key in dmap:
                    out[i] = dmap[key]
            if np.all(np.isfinite(out)):
                return out
    except Exception:
        pass
    positions = np.asarray(T.positions, dtype=float)
    lattice = np.asarray(T.lattice, dtype=float)
    out = np.zeros(n, dtype=float)
    for i in range(n):
        dr = np.asarray(R[i], dtype=float) @ lattice + positions[int(oto[i])] - positions[int(ofrom[i])]
        out[i] = float(np.linalg.norm(dr))
    return out


def component_masks(O, T, nn_dist: float = 0.5, nn_tol: float = 1e-6) -> Dict[str, np.ndarray]:
    R = np.asarray(O.mi_to_R, dtype=int)
    ofrom = np.asarray(O.mi_to_ofrom, dtype=int)
    oto = np.asarray(O.mi_to_oto, dtype=int)
    dist = distances_from_tuff(O, T)
    onsite = (ofrom == oto) & np.all(R == 0, axis=1)
    diagonal = ofrom == oto
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
        "nn_offdiag": nn_offdiag,
        "longer": longer,
    }


def norm_weight(v: np.ndarray, mask: np.ndarray) -> float:
    v = np.asarray(v, dtype=complex).reshape(-1)
    denom = float(np.sum(np.abs(v) ** 2))
    if denom <= 0:
        return 0.0
    return float(np.sum(np.abs(v[mask]) ** 2) / denom)


def overlap_template(v: np.ndarray, mask: np.ndarray) -> float:
    v = np.asarray(v, dtype=complex).reshape(-1)
    t = np.zeros_like(v)
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return 0.0
    t[idx] = 1.0
    nv = np.linalg.norm(v)
    nt = np.linalg.norm(t)
    if nv <= 0 or nt <= 0:
        return 0.0
    return float(abs(np.vdot(t / nt, v / nv)) ** 2)


def onsite_pattern(v: np.ndarray, comp: Dict[str, np.ndarray], norb: int) -> Dict[str, float]:
    out = {}
    R, ofrom, oto = comp["R"], comp["ofrom"], comp["oto"]
    for o in range(norb):
        mask = (ofrom == o) & (oto == o) & np.all(R == 0, axis=1)
        amp = complex(np.sum(v[mask]))
        out[f"onsite_o{o}_real"] = float(np.real(amp))
        out[f"onsite_o{o}_imag"] = float(np.imag(amp))
        out[f"onsite_o{o}_abs"] = float(abs(amp))
    return out


def bond_pair_summary(v: np.ndarray, comp: Dict[str, np.ndarray]) -> Dict[str, float | str]:
    ofrom, oto, nn = comp["ofrom"], comp["oto"], comp["nn_offdiag"]
    out: Dict[str, float | str] = {}
    best_pair = "none"
    best_weight = -1.0
    for pair in ["AB", "BC", "AC"]:
        mask = np.zeros_like(nn, dtype=bool)
        for i in range(len(mask)):
            if nn[i] and PAIR_NAMES.get((int(ofrom[i]), int(oto[i])), "") == pair:
                mask[i] = True
        amp = complex(np.sum(v[mask]))
        w = norm_weight(v, mask)
        out[f"nn_{pair}_weight"] = float(w)
        out[f"nn_{pair}_sum_real"] = float(np.real(amp))
        out[f"nn_{pair}_sum_imag"] = float(np.imag(amp))
        out[f"nn_{pair}_sum_abs"] = float(abs(amp))
        if w > best_weight:
            best_weight = w
            best_pair = pair
    out["dominant_nn_pair_by_weight"] = best_pair
    out["dominant_nn_pair_weight"] = float(best_weight)
    return out


def classify_label(summary: Dict[str, float]) -> str:
    onsite = float(summary.get("onsite_weight", 0.0))
    nn = float(summary.get("nn_bond_weight", 0.0))
    if nn > 0.5 and nn > onsite:
        return "cBO_like_NN_bond_dominated"
    if nn > 0.25 and nn > 0.5 * onsite:
        return "mixed_with_substantial_cBO_bond_weight"
    if onsite > 0.5 and onsite > nn:
        return "CDW_like_onsite_dominated"
    return "mixed_or_longer_range"


def summarize_vector(case: str, qname: str, iq: int, mode_index: int, value: float, v: np.ndarray, O, T, source: str) -> Dict[str, float | int | str]:
    v = normalize_phase(v)
    comp = component_masks(O, T)
    norm = float(np.sum(np.abs(v) ** 2))
    norb = int(np.asarray(T.n_orb).item()) if hasattr(T, "n_orb") else int(O.n_orb)
    row: Dict[str, float | int | str] = {
        "case": case,
        "source": source,
        "qname": qname,
        "iq": int(iq),
        "mode_index": int(mode_index),
        "value": float(value),
        "abs_value": float(abs(value)),
        "norm": norm,
        "onsite_weight": norm_weight(v, comp["onsite"]),
        "diagonal_weight": norm_weight(v, comp["diagonal"]),
        "offdiag_weight": norm_weight(v, comp["offdiag"]),
        "nn_bond_weight": norm_weight(v, comp["nn_offdiag"]),
        "longer_weight": norm_weight(v, comp["longer"]),
        "onsite_A1_overlap": overlap_template(v, comp["onsite"]),
        "nn_A1_overlap": overlap_template(v, comp["nn_offdiag"]),
        "real_fraction": float(np.sum(np.real(v) ** 2) / max(norm, 1e-300)),
        "imag_fraction": float(np.sum(np.imag(v) ** 2) / max(norm, 1e-300)),
    }
    row.update(onsite_pattern(v, comp, norb))
    row.update(bond_pair_summary(v, comp))
    row["inferred_form_factor"] = classify_label(row)  # type: ignore[arg-type]
    return row


def top_components(case: str, qname: str, iq: int, mode_index: int, v: np.ndarray, O, T, source: str, top_n: int) -> List[Dict[str, float | int | str]]:
    v = normalize_phase(v)
    comp = component_masks(O, T)
    R, ofrom, oto, dist = comp["R"], comp["ofrom"], comp["oto"], comp["dist"]
    order = np.argsort(-np.abs(v))[:top_n]
    rows = []
    for rank, idx in enumerate(order):
        if comp["onsite"][idx]:
            ctype = "onsite"
        elif comp["nn_offdiag"][idx]:
            ctype = "NN_bond"
        elif comp["diagonal"][idx]:
            ctype = "diagonal_longer"
        else:
            ctype = "longer_orbital"
        rows.append({
            "case": case,
            "source": source,
            "qname": qname,
            "iq": int(iq),
            "mode_index": int(mode_index),
            "rank": int(rank),
            "component_index": int(idx),
            "R0": int(R[idx, 0]), "R1": int(R[idx, 1]), "R2": int(R[idx, 2]),
            "ofrom": int(ofrom[idx]), "oto": int(oto[idx]),
            "pair": PAIR_NAMES.get((int(ofrom[idx]), int(oto[idx])), "diag" if ofrom[idx] == oto[idx] else "other"),
            "distance": float(dist[idx]),
            "component_type": ctype,
            "value_real": float(np.real(v[idx])),
            "value_imag": float(np.imag(v[idx])),
            "value_abs": float(abs(v[idx])),
        })
    return rows


def decomp_vector(O, X: str, iq: int, mode_index: int) -> Tuple[float, np.ndarray]:
    vals = np.asarray(getattr(O, f"{X}val"), dtype=float)
    vecs = np.asarray(getattr(O, f"{X}vec"))
    off = np.asarray(getattr(O, f"{X}off"), dtype=int)
    length = np.asarray(getattr(O, f"{X}len"), dtype=int)
    if mode_index >= int(length[iq]):
        raise IndexError(f"channel {X}, iq={iq} has len={length[iq]}, requested mode {mode_index}")
    loc = int(off[iq] + mode_index)
    return float(vals[loc]), np.asarray(vecs[loc], dtype=complex).reshape(-1)


def channel_matrix(O, X: str, iq: int) -> np.ndarray:
    A = np.asarray(getattr(O, f"{X}channel"))
    return np.asarray(A[0, :, 0, :, iq], dtype=complex)


def complement_basis(mask: np.ndarray, n: int) -> np.ndarray:
    keep = np.flatnonzero(~mask)
    C = np.eye(n, dtype=complex)[:, keep]
    return C


def projected_residual_modes(O, T, X: str, iq: int, n_modes: int) -> Tuple[str, np.ndarray, np.ndarray]:
    """Project out onsite diagonal components and decompose residual channel matrix.

    Returns (method, values, vectors_full), where vectors_full has shape (modes, n).
    If residual matrix is Hermitian, use eigh sorted by descending |eigenvalue|.
    Otherwise use SVD sorted by singular values.
    """
    K = channel_matrix(O, X, iq)
    n = K.shape[0]
    comp = component_masks(O, T)
    onsite = np.asarray(comp["onsite"], dtype=bool)
    C = complement_basis(onsite, n)
    Kred = C.conj().T @ K @ C
    herm_rel = np.linalg.norm(Kred - Kred.conj().T) / max(np.linalg.norm(Kred), 1e-300)
    m = min(n_modes, Kred.shape[0])
    if herm_rel < 1e-9:
        vals, vecs = np.linalg.eigh(0.5 * (Kred + Kred.conj().T))
        order = np.argsort(-np.abs(vals))[:m]
        vals = vals[order]
        vecs = vecs[:, order]
        full = (C @ vecs).T
        return f"onsite_projected_eigh_abs_hermrel={herm_rel:.3e}", np.asarray(vals, dtype=float), np.asarray(full, dtype=complex)
    else:
        U, s, Vh = np.linalg.svd(Kred, full_matrices=False)
        full = (C @ Vh[:m].conj().T).T
        return f"onsite_projected_svd_hermrel={herm_rel:.3e}", np.asarray(s[:m], dtype=float), np.asarray(full, dtype=complex)


def write_csv(path: str, rows: List[Dict]) -> None:
    if not rows:
        return
    keys = sorted({k for row in rows for k in row.keys()})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def iter_cases(root: str, selected: Optional[str]) -> List[str]:
    if selected:
        return [selected]
    return [name for name in sorted(os.listdir(root)) if os.path.exists(os.path.join(root, name, "O.dvg"))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="projected output root, e.g. ../outputs/kagome_prr_mu0_projected_small")
    ap.add_argument("--case", default=None)
    ap.add_argument("--nmodes", type=int, default=8)
    ap.add_argument("--top", type=int, default=12)
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    cases = iter_cases(root, args.case)
    print("root =", root)
    print("cases =", cases)

    sub_rows: List[Dict] = []
    sub_top_rows: List[Dict] = []
    proj_rows: List[Dict] = []
    proj_top_rows: List[Dict] = []

    for case in cases:
        Ofile = os.path.join(root, case, "O.dvg")
        Tfile = os.path.join(root, case, "T.dvg")
        O = ro.read(Ofile)
        T = ro.read(Tfile)
        qidx = high_symmetry_indices(O, T)
        print("\n" + "#" * 110)
        print("CASE", case)
        print("#" * 110)

        for qname in ["M1", "M2", "M3"]:
            iq = qidx[qname]
            print(f"\n{case} {qname}: M-charge saved D channel")
            # 1. Existing subleading decomposition vectors.
            for mode in range(args.nmodes):
                try:
                    val, vec = decomp_vector(O, "D", iq, mode)
                except Exception as exc:
                    print(f"  mode {mode}: failed {exc}")
                    continue
                row = summarize_vector(case, qname, iq, mode, val, vec, O, T, source="saved_D_decomp")
                sub_rows.append(row)
                sub_top_rows.extend(top_components(case, qname, iq, mode, vec, O, T, source="saved_D_decomp", top_n=args.top))
                print(
                    f"  saved mode {mode:2d}: val={val:+.6e} "
                    f"onsite={row['onsite_weight']:.3f} nn={row['nn_bond_weight']:.3f} "
                    f"label={row['inferred_form_factor']}"
                )

            # 2. Project out onsite diagonal subspace from full Dchannel and decompose residual.
            method, vals, vecs = projected_residual_modes(O, T, "D", iq, args.nmodes)
            print(f"  onsite-projected residual method: {method}")
            for mode in range(len(vals)):
                vec = vecs[mode]
                val = float(vals[mode])
                row = summarize_vector(case, qname, iq, mode, val, vec, O, T, source=method)
                proj_rows.append(row)
                proj_top_rows.extend(top_components(case, qname, iq, mode, vec, O, T, source=method, top_n=args.top))
                print(
                    f"  residual mode {mode:2d}: val={val:+.6e} "
                    f"onsite={row['onsite_weight']:.3f} nn={row['nn_bond_weight']:.3f} "
                    f"longer={row['longer_weight']:.3f} label={row['inferred_form_factor']}"
                )

    paths = {
        "sub": os.path.join(root, "Mcharge_subleading_modes.csv"),
        "sub_top": os.path.join(root, "Mcharge_subleading_top_components.csv"),
        "proj": os.path.join(root, "Mcharge_onsite_projected_residual.csv"),
        "proj_top": os.path.join(root, "Mcharge_onsite_projected_top_components.csv"),
    }
    write_csv(paths["sub"], sub_rows)
    write_csv(paths["sub_top"], sub_top_rows)
    write_csv(paths["proj"], proj_rows)
    write_csv(paths["proj_top"], proj_top_rows)
    print("\nwrote", paths["sub"])
    print("wrote", paths["sub_top"])
    print("wrote", paths["proj"])
    print("wrote", paths["proj_top"])


if __name__ == "__main__":
    main()
