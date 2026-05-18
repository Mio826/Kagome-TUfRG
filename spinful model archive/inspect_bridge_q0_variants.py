#!/usr/bin/env python3
# coding: utf-8
"""
Lightweight q=0 bridge inspector for explicit-spin divERGe TUFRG output.

Purpose
-------
Diagnose whether the corrected explicit-spin no-SOC run reproduces the
SU2-reduced PRR benchmark.  This script does NOT rerun the flow.
It only reads existing T.dvg/O.dvg and inspects Gamma-sector kernels.

Main checks
-----------
1. PH spin/charge projection variant scan:
   - K choices: D-C, D, C, 2D-C
   - spin-pair grouping: direct vs crossed, independently for K and L
   - reports rho/sx/sy/sz strengths and triplet anisotropy.

2. q=0 charge uniform projection:
   - For selected convention K=D-C, K crossed, L crossed by default,
     computes raw rho strength, strength after removing onsite A1,
     and strength after removing all onsite diagonal components.

3. Form-factor top components:
   - extracts the leading right singular vector for rho, sx, sy, sz, and pp-singlet
   - reports onsite/NN/longer weights and top TU components.

No pandas required.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import diverge.output as ro

PAIR_ORDER = ("uu", "ud", "du", "dd")
SPINS = (0, 1)
PAIR_TO_INDEX = {p: i for i, p in enumerate(PAIR_ORDER)}


def spin_matrix_vec(label: str, *, normalized: bool = True) -> np.ndarray:
    """Vectorize a PH spin matrix in pair basis (out,in) = uu,ud,du,dd."""
    M = np.zeros((2, 2), dtype=complex)
    if label == "rho":
        M[0, 0] = 1.0
        M[1, 1] = 1.0
    elif label == "sx":
        M[0, 1] = 1.0
        M[1, 0] = 1.0
    elif label == "sy":
        M[0, 1] = -1j
        M[1, 0] = 1j
    elif label == "sz":
        M[0, 0] = 1.0
        M[1, 1] = -1.0
    else:
        raise ValueError(label)
    if normalized:
        M = M / np.sqrt(2.0)
    return np.array([M[0,0], M[0,1], M[1,0], M[1,1]], dtype=complex)


def pp_spin_vec(label: str) -> np.ndarray:
    v = np.zeros(4, dtype=complex)
    if label == "singlet":
        v[PAIR_TO_INDEX["ud"]] = 1/np.sqrt(2)
        v[PAIR_TO_INDEX["du"]] = -1/np.sqrt(2)
    elif label == "triplet0":
        v[PAIR_TO_INDEX["ud"]] = 1/np.sqrt(2)
        v[PAIR_TO_INDEX["du"]] = 1/np.sqrt(2)
    elif label == "triplet_plus":
        v[PAIR_TO_INDEX["uu"]] = 1.0
    elif label == "triplet_minus":
        v[PAIR_TO_INDEX["dd"]] = 1.0
    else:
        raise ValueError(label)
    return v


def as5(arr: np.ndarray, iq: int) -> np.ndarray:
    """Return channel/loop at q as shape (4,nff,4,nff)."""
    A = np.asarray(arr, dtype=complex)
    if A.ndim != 5:
        raise ValueError(f"expected 5d array, got shape {A.shape}")
    return A[:, :, :, :, int(iq)]


def project_direct(A4: np.ndarray, left: np.ndarray, right: np.ndarray | None = None) -> np.ndarray:
    """Project A[spinpairL,m,spinpairR,n] with spin-pair vectors."""
    if right is None:
        right = left
    return np.einsum("a,ambn,b->mn", left.conj(), A4, right, optimize=True)


def project_crossed(A4: np.ndarray, left: np.ndarray, right: np.ndarray | None = None) -> np.ndarray:
    r"""Project raw A[s0s1,m,s2s3,n] into PH bilinear grouping.

    crossed grouping means left bilinear = (s0,s3), right bilinear = (s1,s2).
    left/right are vectors in matrix basis (out,in) = uu,ud,du,dd.
    """
    if right is None:
        right = left
    nff = A4.shape[1]
    out = np.zeros((nff, nff), dtype=complex)
    for s0 in SPINS:
        for s1 in SPINS:
            a_raw = 2*s0 + s1
            for s2 in SPINS:
                for s3 in SPINS:
                    b_raw = 2*s2 + s3
                    a_ph = 2*s0 + s3  # (out,in) of left bilinear
                    b_ph = 2*s1 + s2  # (out,in) of right bilinear
                    out += left[a_ph].conj() * A4[a_raw, :, b_raw, :] * right[b_ph]
    return out


def top_svd(A: np.ndarray, k: int = 1) -> np.ndarray:
    return np.linalg.svd(np.asarray(A, dtype=complex), compute_uv=False)[:k]


def leading_right_sv(A: np.ndarray) -> Tuple[float, np.ndarray]:
    U, s, Vh = np.linalg.svd(np.asarray(A, dtype=complex), full_matrices=False)
    v = Vh[0].conj()
    return float(s[0]), normalize_phase(v)


def normalize_phase(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=complex).reshape(-1).copy()
    if v.size == 0:
        return v
    i = int(np.argmax(np.abs(v)))
    if abs(v[i]) > 0:
        v *= np.exp(-1j*np.angle(v[i]))
        if np.real(v[i]) < 0:
            v *= -1
    return v


def projector_remove(mask_vec: np.ndarray) -> np.ndarray:
    """Return P = I - |u><u| for a template vector; if zero, identity."""
    u = np.asarray(mask_vec, dtype=complex).reshape(-1)
    n = u.size
    norm = np.linalg.norm(u)
    if norm <= 0:
        return np.eye(n, dtype=complex)
    u = u / norm
    return np.eye(n, dtype=complex) - np.outer(u, u.conj())


def projector_keep_complement(mask: np.ndarray) -> np.ndarray:
    """Project out all basis components where mask is True."""
    mask = np.asarray(mask, dtype=bool)
    P = np.eye(mask.size, dtype=complex)
    P[mask, :] = 0.0
    P[:, mask] = 0.0
    return P


def component_info(O, T) -> Dict[str, np.ndarray]:
    R = np.asarray(O.mi_to_R, dtype=int)
    ofrom = np.asarray(O.mi_to_ofrom, dtype=int)
    oto = np.asarray(O.mi_to_oto, dtype=int)
    n = len(ofrom)
    positions = np.asarray(T.positions, dtype=float)
    lattice = np.asarray(T.lattice, dtype=float)
    dist = np.zeros(n, dtype=float)
    for i in range(n):
        dr = np.asarray(R[i], dtype=float) @ lattice + positions[int(oto[i])] - positions[int(ofrom[i])]
        dist[i] = float(np.linalg.norm(dr[:2]))
    onsite = (ofrom == oto) & np.all(R == 0, axis=1)
    diagonal = (ofrom == oto)
    offdiag = ~diagonal
    nn = (np.abs(dist - 0.5) < 1e-6) & offdiag
    longer = ~(onsite | nn)
    return dict(R=R, ofrom=ofrom, oto=oto, dist=dist, onsite=onsite, diagonal=diagonal, offdiag=offdiag, nn=nn, longer=longer)


def norm_weight(v: np.ndarray, mask: np.ndarray) -> float:
    den = float(np.sum(np.abs(v)**2))
    if den <= 0:
        return 0.0
    return float(np.sum(np.abs(v[np.asarray(mask, dtype=bool)])**2)/den)


def onsite_A1_template(comp: Dict[str, np.ndarray]) -> np.ndarray:
    t = np.zeros(len(comp["ofrom"]), dtype=complex)
    t[comp["onsite"]] = 1.0
    return t


def summarize_vector(v: np.ndarray, comp: Dict[str, np.ndarray]) -> Dict[str, float]:
    tA1 = onsite_A1_template(comp)
    nv = np.linalg.norm(v)
    nt = np.linalg.norm(tA1)
    ov = 0.0 if nv <= 0 or nt <= 0 else float(abs(np.vdot(tA1/nt, v/nv))**2)
    return {
        "onsite_weight": norm_weight(v, comp["onsite"]),
        "nn_weight": norm_weight(v, comp["nn"]),
        "longer_weight": norm_weight(v, comp["longer"]),
        "diagonal_weight": norm_weight(v, comp["diagonal"]),
        "offdiag_weight": norm_weight(v, comp["offdiag"]),
        "onsite_A1_overlap": ov,
        "real_fraction": float(np.sum(np.real(v)**2)/max(np.sum(np.abs(v)**2), 1e-300)),
        "imag_fraction": float(np.sum(np.imag(v)**2)/max(np.sum(np.abs(v)**2), 1e-300)),
    }


def top_components(v: np.ndarray, comp: Dict[str, np.ndarray], top: int = 30) -> List[Dict[str, object]]:
    rows = []
    R, ofrom, oto, dist = comp["R"], comp["ofrom"], comp["oto"], comp["dist"]
    order = np.argsort(-np.abs(v))[:top]
    for rank, idx in enumerate(order):
        if comp["onsite"][idx]: ctype = "onsite"
        elif comp["nn"][idx]: ctype = "NN_bond"
        elif comp["diagonal"][idx]: ctype = "diagonal_longer"
        else: ctype = "longer_orbital"
        rows.append({
            "rank": rank, "component_index": int(idx),
            "R0": int(R[idx,0]), "R1": int(R[idx,1]), "R2": int(R[idx,2]),
            "ofrom": int(ofrom[idx]), "oto": int(oto[idx]),
            "distance": float(dist[idx]), "component_type": ctype,
            "value_real": float(np.real(v[idx])), "value_imag": float(np.imag(v[idx])), "value_abs": float(abs(v[idx])),
        })
    return rows


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def analyze_case(case_dir: Path, outdir: Path, top: int = 30) -> None:
    O = ro.read(str(case_dir / "O.dvg"))
    T = ro.read(str(case_dir / "T.dvg"))
    comp = component_info(O, T)
    iq = 0  # Gamma
    P = as5(O.Pchannel, iq)
    C = as5(O.Cchannel, iq)
    D = as5(O.Dchannel, iq)
    Lph = as5(O.phloop, iq)
    Lpp = as5(O.pploop, iq)

    Kchoices = {
        "D_minus_C": D - C,
        "D": D,
        "C": C,
        "twoD_minus_C": 2*D - C,
    }
    proj = {"direct": project_direct, "crossed": project_crossed}
    sectors = ["rho", "sx", "sy", "sz"]

    scan_rows = []
    for kname, K4 in Kchoices.items():
        for pk_name, pk in proj.items():
            for pl_name, pl in proj.items():
                vals = {}
                for sec in sectors:
                    e = spin_matrix_vec(sec)
                    Ksec = pk(K4, e)
                    Lsec = pl(Lph, e)
                    vals[sec] = float(top_svd(Ksec @ Lsec, 1)[0])
                trip = np.array([vals["sx"], vals["sy"], vals["sz"]], dtype=float)
                scan_rows.append({
                    "case": case_dir.name,
                    "K_choice": kname,
                    "K_projection": pk_name,
                    "L_projection": pl_name,
                    "rho": vals["rho"],
                    "sx": vals["sx"],
                    "sy": vals["sy"],
                    "sz": vals["sz"],
                    "triplet_mean": float(np.mean(trip)),
                    "triplet_cv": float(np.std(trip)/max(np.mean(trip), 1e-300)),
                    "triplet_spread": float((np.max(trip)-np.min(trip))/max(np.max(trip), 1e-300)),
                })

    # PP sectors, direct pair basis
    pp_rows = []
    for sec in ["singlet", "triplet0", "triplet_plus", "triplet_minus"]:
        e = pp_spin_vec(sec)
        Ksec = project_direct(P, e)
        Lsec = project_direct(Lpp, e)
        val, vec = leading_right_sv(Ksec @ Lsec)
        row = {"case": case_dir.name, "sector": f"pp-{sec}", "strength": val}
        row.update(summarize_vector(vec, comp))
        pp_rows.append(row)

    # Default PH convention from current rawguard: K=D-C, K crossed, L crossed.
    default_rows = []
    default_top_rows = []
    Kdef4 = Kchoices["D_minus_C"]
    for sec in sectors:
        e = spin_matrix_vec(sec)
        Ksec = project_crossed(Kdef4, e)
        Lsec = project_crossed(Lph, e)
        A = Ksec @ Lsec
        val, vec = leading_right_sv(A)
        row = {"case": case_dir.name, "sector": f"ph-{sec}", "strength_raw": val}
        row.update(summarize_vector(vec, comp))
        # For rho only: remove onsite A1 and all onsite components.
        if sec == "rho":
            P_no_A1 = projector_remove(onsite_A1_template(comp))
            P_no_onsite = projector_keep_complement(comp["onsite"])
            row["strength_after_remove_onsite_A1"] = float(top_svd(P_no_A1 @ A @ P_no_A1, 1)[0])
            row["strength_after_remove_all_onsite"] = float(top_svd(P_no_onsite @ A @ P_no_onsite, 1)[0])
        default_rows.append(row)
        for tr in top_components(vec, comp, top=top):
            tr.update({"case": case_dir.name, "sector": f"ph-{sec}"})
            default_top_rows.append(tr)

    # top components for PP singlet too
    pp_top_rows = []
    e = pp_spin_vec("singlet")
    val, vec = leading_right_sv(project_direct(P, e) @ project_direct(Lpp, e))
    for tr in top_components(vec, comp, top=top):
        tr.update({"case": case_dir.name, "sector": "pp-singlet"})
        pp_top_rows.append(tr)

    out = outdir / case_dir.name
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "q0_ph_projection_variant_scan.csv", scan_rows)
    write_csv(out / "q0_pp_summary.csv", pp_rows)
    write_csv(out / "q0_default_ph_mode_summary.csv", default_rows)
    write_csv(out / "q0_default_ph_mode_top_components.csv", default_top_rows)
    write_csv(out / "q0_pp_singlet_top_components.csv", pp_top_rows)

    print("="*100)
    print("CASE", case_dir.name)
    print("Wrote", out)
    print("Default PH rows:")
    for r in default_rows:
        print(r)
    print("PP rows:")
    for r in pp_rows:
        print(r)
    print("Best PH variants by triplet_cv then rho/spin summary:")
    for r in sorted(scan_rows, key=lambda x: (x["triplet_cv"], -x["triplet_mean"]))[:8]:
        print(r)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="bridge output root containing U1p5_V0p0 etc")
    ap.add_argument("--case", action="append", default=None, help="case name; can repeat")
    ap.add_argument("--out", default="bridge_q0_variant_inspection")
    ap.add_argument("--top", type=int, default=30)
    args = ap.parse_args()
    root = Path(args.root).resolve()
    outdir = root / args.out
    cases = args.case or [p.name for p in sorted(root.iterdir()) if (p / "O.dvg").exists()]
    for c in cases:
        analyze_case(root / c, outdir, top=args.top)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
