#!/usr/bin/env python3
"""
analyze_nn_current_source_susceptibility_v1.py

First working version of a Bethe-Salpeter source-response analysis for
NN imaginary bond-current sources in TU/fRG divERGe output.

IMPORTANT SCOPE OF V1
---------------------
This script computes the susceptibility of the *general NN imaginary
bond-current source subspace*, not yet the strictly divergence-free LCO
subspace. It is designed to answer:

    Among all NN imaginary hopping/current sources, how strong is the
    response at Gamma and at the M momenta?

This is the correct broad first step. Later we can add a divergence-free
projection and named-pattern overlaps, using the leading eigenvector saved
by this script.

Physics convention used in V1
-----------------------------
From the calibration test:
    K_ph = Dchannel - Cchannel
    L_ph = - phloop

For each q and spin/charge sector mu, we build
    A_mu(q) = L_mu^{1/2} K_mu(q) L_mu^{1/2}
    R_mu(q) = L_mu^{1/2} [I - A_mu(q)]^{-1} L_mu^{1/2}

Then we project
    chi_current(q) = Q_current(q)^dagger R_mu(q) Q_current(q)

and report its largest eigenvalue and leading source vector.

Run on compute node, not login node. One O.dvg can require ~50 GB.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from TUfRG_formfactor_from_modes import load_output

# -----------------------------
# Small utilities
# -----------------------------

def parse_num_token(tok: str):
    if tok is None:
        return None
    try:
        return float(str(tok).replace("p", "."))
    except Exception:
        return None


def parse_case_from_name(name: str) -> Dict[str, Optional[float]]:
    out = {"U": None, "V1": None, "lambda_nn": None, "lambda_nnn": None}

    m = re.search(r"_U([0-9p.+-]+)_", name)
    if m:
        out["U"] = parse_num_token(m.group(1))

    m = re.search(r"_V([0-9p.+-]+)_", name)
    if m:
        out["V1"] = parse_num_token(m.group(1))

    m = re.search(r"lnn(?!n)([0-9p.+-]+)", name)
    if m:
        out["lambda_nn"] = parse_num_token(m.group(1))

    m = re.search(r"lnnn([0-9p.+-]+)", name)
    if m:
        out["lambda_nnn"] = parse_num_token(m.group(1))

    return out


def read_metadata(case_dir: Path) -> Dict:
    md = {}
    for p in [case_dir / "metadata.json", case_dir / "snapshots" / "final" / "metadata.json"]:
        if p.exists():
            try:
                md.update(json.loads(p.read_text()))
            except Exception:
                pass
    fallback = parse_case_from_name(case_dir.name)
    for k, v in fallback.items():
        if k not in md or md.get(k) is None:
            md[k] = v
    if "V1" not in md or md.get("V1") is None:
        md["V1"] = md.get("V", None)
    return md


def find_case_dirs(root: Path) -> List[Path]:
    # A case dir is any dir containing snapshots/final/O.dvg or final/O.dvg or O.dvg.
    dirs = []
    for odvg in root.glob("**/O.dvg"):
        p = odvg.parent
        if p.name == "final" and p.parent.name == "snapshots":
            dirs.append(p.parent.parent)
        else:
            dirs.append(p)
    # Unique, stable order.
    unique = []
    seen = set()
    for d in dirs:
        dr = d.resolve()
        if dr not in seen:
            seen.add(dr)
            unique.append(d)
    return sorted(unique)


def odvg_path_for_case(case_dir: Path, snapshot: str = "final") -> Optional[Path]:
    candidates = [
        case_dir / "snapshots" / snapshot / "O.dvg",
        case_dir / snapshot / "O.dvg",
        case_dir / "O.dvg",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None

# -----------------------------
# q labels / indices
# -----------------------------

def q_index_map(nk: int, nq: int, labels: List[str]) -> Dict[str, int]:
    """Flattening convention confirmed by smoke test: iq = k1 + nk*k2."""
    half = nk // 2
    full = {
        "Gamma": 0,
        "G": 0,
        "M1": half,
        "M2": nk * half,
        "M3": nk * half + half,
    }
    out = {}
    for lab in labels:
        if lab not in full:
            raise ValueError(f"Unknown q label {lab}; use Gamma M1 M2 M3")
        iq = full[lab]
        if iq >= nq:
            raise ValueError(f"q label {lab} gives iq={iq}, but nq={nq}")
        out["Gamma" if lab == "G" else lab] = iq
    return out


def q_grid_coords(label: str, nk: int) -> Tuple[int, int]:
    half = nk // 2
    if label in ["Gamma", "G"]:
        return (0, 0)
    if label == "M1":
        return (half, 0)
    if label == "M2":
        return (0, half)
    if label == "M3":
        return (half, half)
    raise ValueError(label)

# -----------------------------
# Spin projection
# -----------------------------

I2 = np.eye(2, dtype=complex)
sx = np.array([[0, 1], [1, 0]], dtype=complex)
sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
sz = np.array([[1, 0], [0, -1]], dtype=complex)
MATS = {
    "rho": I2 / np.sqrt(2),
    "sx": sx / np.sqrt(2),
    "sy": sy / np.sqrt(2),
    "sz": sz / np.sqrt(2),
}


def project_spin_crossed(Xraw_q: np.ndarray, mu: str) -> np.ndarray:
    """
    Project explicit spin-pair object to charge/spin bilinear sector.

    Xraw_q shape: (4, nff, 4, nff), with spin-pair p = s_left*2+s_right.

    Crossed bilinear grouping:
        M^*_{s0,s3} X_{s0s1; s2s3} M_{s1,s2}
    """
    M = MATS[mu]
    nff = Xraw_q.shape[1]
    out = np.zeros((nff, nff), dtype=complex)
    for s0 in range(2):
        for s1 in range(2):
            pL = s0 * 2 + s1
            for s2 in range(2):
                for s3 in range(2):
                    pR = s2 * 2 + s3
                    coeff = np.conj(M[s0, s3]) * M[s1, s2]
                    out += coeff * Xraw_q[pL, :, pR, :]
    return out

# -----------------------------
# Linear algebra
# -----------------------------

def herm(A: np.ndarray) -> np.ndarray:
    return 0.5 * (A + A.conj().T)


def psd_sqrt(L: np.ndarray, tol: float = 1e-10) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Hermitize L and keep positive eigenvalues for L^{1/2}."""
    Lh = herm(L)
    evals, evecs = np.linalg.eigh(Lh)
    pos = evals > tol
    evals_clip = np.where(pos, evals, 0.0)
    Lhalf = (evecs * np.sqrt(evals_clip)) @ evecs.conj().T
    info = {
        "L_eval_min": float(evals.min()),
        "L_eval_max": float(evals.max()),
        "L_n_pos": int(np.sum(evals > tol)),
        "L_n_neg": int(np.sum(evals < -tol)),
        "L_kept_weight": float(np.sum(evals_clip)),
        "L_abs_weight": float(np.sum(np.abs(evals))),
        "L_kept_over_abs": float(np.sum(evals_clip) / max(np.sum(np.abs(evals)), 1e-30)),
    }
    return Lhalf, evals, info


def build_response(Kmu: np.ndarray, Lmu: np.ndarray, pole_tol: float = 1e-8) -> Tuple[np.ndarray, Dict]:
    """Build A and R=Lhalf(I-A)^-1Lhalf using spectral decomposition of Hermitian A."""
    Kmu = herm(Kmu)
    Lmu = herm(Lmu)
    Lhalf, Levals, linfo = psd_sqrt(Lmu)
    A = herm(Lhalf @ Kmu @ Lhalf)
    ae, U = np.linalg.eigh(A)

    denom = 1.0 - ae
    near = np.abs(denom) < pole_tol
    # Do not silently regularize poles too much; keep sign but avoid division by zero.
    denom_safe = denom.copy()
    denom_safe[near] = np.sign(denom_safe[near] + 0.0) * pole_tol
    denom_safe[denom_safe == 0] = pole_tol

    # R = Lhalf U diag(1/(1-ae)) U^dag Lhalf
    B = Lhalf @ U
    R = (B * (1.0 / denom_safe)) @ B.conj().T
    R = herm(R)

    info = dict(linfo)
    info.update({
        "A_eval_min": float(ae.min()),
        "A_eval_max": float(ae.max()),
        "A_gap_to_1": float(1.0 - ae.max()),
        "A_n_over_1": int(np.sum(ae > 1.0)),
        "A_n_near_pole": int(np.sum(near)),
        "A_antiherm_rel": float(np.linalg.norm(A - A.conj().T) / max(np.linalg.norm(A), 1e-30)),
    })
    return R, info

# -----------------------------
# Current source basis
# -----------------------------

def import_model_geometry():
    try:
        import TUfRG_model as km
        lattice, pos = km.kagome_lattice_and_positions()
        return km, np.asarray(lattice), np.asarray(pos)
    except Exception:
        # fallback geometry used in earlier scripts; prefer TUfRG_model if available.
        km = None
        lattice = np.array([[1.0, 0.0, 0.0], [0.5, np.sqrt(3)/2, 0.0], [0.0, 0.0, 1.0]])
        pos = np.array([
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.25, np.sqrt(3)/4, 0.0],
        ])
        return km, lattice, pos


def displacement_vec(km, lattice: np.ndarray, pos: np.ndarray, R: np.ndarray, ofrom: int, oto: int) -> np.ndarray:
    # In existing code, displacement_vector(lattice, pos, Rx, Ry, ofrom, oto)
    if km is not None and hasattr(km, "displacement_vector"):
        return np.asarray(km.displacement_vector(lattice, pos, int(R[0]), int(R[1]), int(ofrom), int(oto)))
    return int(R[0]) * lattice[0] + int(R[1]) * lattice[1] + pos[int(ofrom)] - pos[int(oto)]


def q_phase_from_R(R: np.ndarray, q_m: Tuple[int, int], nk: int) -> complex:
    # q·R = 2π (m1 R0 + m2 R1) / nk in reciprocal lattice coordinates.
    return np.exp(1j * 2.0 * np.pi * (q_m[0] * int(R[0]) + q_m[1] * int(R[1])) / float(nk))


def build_nn_current_basis(mi_to_R: np.ndarray,
                           ofrom: np.ndarray,
                           oto: np.ndarray,
                           q_label: str,
                           nk: int,
                           nn_tol: float = 1e-5,
                           finite_q_unwind: bool = True) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Build an orthonormal basis Q for all NN imaginary bond-current sources.

    Each independent bond-reversal pair (R,a,b) <-> (-R,b,a) becomes one
    source vector. For finite q, we impose antisymmetry in the q-unwound source:
        X_ab(R) = - X_ba(-R), eta_ab(R;q)=exp(i q.R) X_ab(R)
    and choose X = i on one oriented component.

    Returns Q with shape (nff, nbasis), Euclidean-orthonormal columns.
    """
    R = np.asarray(mi_to_R, dtype=int)
    ofrom = np.asarray(ofrom, dtype=int).reshape(-1)
    oto = np.asarray(oto, dtype=int).reshape(-1)
    nff = len(ofrom)

    km, lattice, pos = import_model_geometry()
    dist = np.zeros(nff, dtype=float)
    for i in range(nff):
        d = displacement_vec(km, lattice, pos, R[i], int(ofrom[i]), int(oto[i]))
        dist[i] = np.linalg.norm(np.asarray(d)[:2])

    nonzero = dist[dist > nn_tol]
    if len(nonzero) == 0:
        raise RuntimeError("Could not determine NN distance from form-factor map.")
    d_nn = float(nonzero.min())
    is_nn = np.abs(dist - d_nn) <= 1e-4

    key_to_idx = {}
    for i in range(nff):
        key = (int(R[i, 0]), int(R[i, 1]), int(R[i, 2]) if R.shape[1] > 2 else 0,
               int(ofrom[i]), int(oto[i]))
        key_to_idx[key] = i

    q_m = q_grid_coords(q_label, nk)

    vectors = []
    rows = []
    used = set()

    for i in range(nff):
        if not is_nn[i]:
            continue
        Ri = R[i]
        a = int(ofrom[i])
        b = int(oto[i])
        rev_key = (-int(Ri[0]), -int(Ri[1]), -int(Ri[2]) if R.shape[1] > 2 else 0, b, a)
        j = key_to_idx.get(rev_key, None)
        if j is None:
            # Some form-factor maps may not explicitly contain both directions.
            continue
        if not is_nn[j]:
            continue
        if i == j:
            continue
        pair = tuple(sorted([i, j]))
        if pair in used:
            continue
        used.add(pair)

        v = np.zeros(nff, dtype=complex)
        # choose canonical orientation i -> j based on smaller index i0
        i0, j0 = pair
        Ri0 = R[i0]
        Rj0 = R[j0]
        if finite_q_unwind:
            pi = q_phase_from_R(Ri0, q_m, nk)
            pj = q_phase_from_R(Rj0, q_m, nk)
        else:
            pi = 1.0 + 0j
            pj = 1.0 + 0j
        v[i0] = 1j * pi / np.sqrt(2)
        v[j0] = -1j * pj / np.sqrt(2)
        vectors.append(v)
        rows.append({
            "basis_id": len(vectors) - 1,
            "i": i0,
            "j_reverse": j0,
            "R_i": tuple(int(x) for x in R[i0]),
            "ofrom_i": int(ofrom[i0]),
            "oto_i": int(oto[i0]),
            "R_j": tuple(int(x) for x in R[j0]),
            "ofrom_j": int(ofrom[j0]),
            "oto_j": int(oto[j0]),
            "distance_i": float(dist[i0]),
            "distance_j": float(dist[j0]),
            "d_nn": d_nn,
            "q_label": q_label,
        })

    if not vectors:
        raise RuntimeError(f"No NN current basis vectors found for q={q_label}")

    Qraw = np.column_stack(vectors)
    # QR for numerical orthonormality. The pair basis should already be orthonormal,
    # but QR is cheap and safer.
    Q, Rmat = np.linalg.qr(Qraw)
    # Remove possible near-zero columns.
    diag = np.abs(np.diag(Rmat)) if Rmat.ndim == 2 else np.array([])
    if len(diag) == Q.shape[1]:
        keep = diag > 1e-10
        Q = Q[:, keep]
        rows = [r for r, k in zip(rows, keep) if k]
    meta = pd.DataFrame(rows)
    return Q, meta


def projected_susceptibility(Rmat: np.ndarray, Q: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """Return max eigenvalue, eigenvalues, leading full nff source vector."""
    X = herm(Q.conj().T @ Rmat @ Q)
    vals, vecs = np.linalg.eigh(X)
    imax = int(np.argmax(vals.real))
    vmax = vecs[:, imax]
    eta_lead = Q @ vmax
    # Normalize full vector.
    nrm = np.linalg.norm(eta_lead)
    if nrm > 0:
        eta_lead = eta_lead / nrm
    return float(vals[imax].real), vals.real, eta_lead

# -----------------------------
# Main per-case analysis
# -----------------------------

def analyze_case(case_dir: Path,
                 outdir: Path,
                 q_labels: List[str],
                 mu: str,
                 snapshot: str,
                 pole_tol: float,
                 save_vectors: bool) -> Dict:
    odvg = odvg_path_for_case(case_dir, snapshot=snapshot)
    if odvg is None:
        raise FileNotFoundError(f"No O.dvg found for {case_dir}")

    md = read_metadata(case_dir)
    O = load_output(odvg)

    C = np.asarray(O.Cchannel)
    D = np.asarray(O.Dchannel)
    Lraw = np.asarray(O.phloop)
    mi_to_R = np.asarray(O.mi_to_R)
    ofrom = np.asarray(O.mi_to_ofrom)
    oto = np.asarray(O.mi_to_oto)

    if C.ndim != 5:
        raise RuntimeError(f"Cchannel ndim={C.ndim}; expected 5")
    nspin, nff, _, _, nq = C.shape
    nk = int(getattr(O, "nk", md.get("nk", int(round(math.sqrt(nq))))))
    qmap = q_index_map(nk, nq, q_labels)

    case_out = outdir / "vectors" / case_dir.name
    if save_vectors:
        case_out.mkdir(parents=True, exist_ok=True)

    row = {
        "case": case_dir.name,
        "odvg": str(odvg),
        "U": md.get("U", None),
        "V1": md.get("V1", md.get("V", None)),
        "V2": md.get("V2", None),
        "lambda_nn": md.get("lambda_nn", None),
        "lambda_nnn": md.get("lambda_nnn", None),
        "nk": nk,
        "nq": nq,
        "nff": nff,
        "mu": mu,
        "snapshot": snapshot,
        "basis_kind": "NN_imaginary_current_not_divfree_v1",
        "K_convention": "D-C",
        "L_convention": "-phloop",
    }

    q_chis = {}
    q_dims = {}

    for qlab, iq in qmap.items():
        Kraw_q = D[..., iq] - C[..., iq]
        Lraw_q = -Lraw[..., iq]
        Kmu = project_spin_crossed(Kraw_q, mu)
        Lmu = project_spin_crossed(Lraw_q, mu)

        Rresp, info = build_response(Kmu, Lmu, pole_tol=pole_tol)
        Q, basis_meta = build_nn_current_basis(mi_to_R, ofrom, oto, qlab, nk)
        chi_max, chi_evals, eta_lead = projected_susceptibility(Rresp, Q)

        q_chis[qlab] = chi_max
        q_dims[qlab] = Q.shape[1]

        prefix = f"{qlab}"
        row[f"chi_{prefix}_NNcurrent"] = chi_max
        row[f"basis_dim_{prefix}"] = Q.shape[1]
        row[f"Amax_{prefix}"] = info["A_eval_max"]
        row[f"Agap_{prefix}"] = info["A_gap_to_1"]
        row[f"A_n_over_1_{prefix}"] = info["A_n_over_1"]
        row[f"L_kept_over_abs_{prefix}"] = info["L_kept_over_abs"]
        row[f"L_n_pos_{prefix}"] = info["L_n_pos"]
        row[f"L_n_neg_{prefix}"] = info["L_n_neg"]

        if save_vectors:
            np.savez_compressed(
                case_out / f"leading_eta_{qlab}_{mu}.npz",
                eta=eta_lead,
                chi_evals=chi_evals,
                mi_to_R=mi_to_R,
                mi_to_ofrom=ofrom,
                mi_to_oto=oto,
                q_label=qlab,
                mu=mu,
                nk=nk,
                iq=iq,
                basis_kind="NN_imaginary_current_not_divfree_v1",
            )
            basis_meta.to_csv(case_out / f"basis_meta_{qlab}_{mu}.csv", index=False)

        # release per-q large matrices
        del Kraw_q, Lraw_q, Kmu, Lmu, Rresp, Q
        gc.collect()

    # 1x1 = Gamma if present
    if "Gamma" in q_chis:
        row["chi_1x1_NNcurrent"] = q_chis["Gamma"]
    else:
        row["chi_1x1_NNcurrent"] = np.nan

    # 2x2 = max over M labels present
    m_items = {k: v for k, v in q_chis.items() if k.startswith("M")}
    if m_items:
        qwin = max(m_items, key=m_items.get)
        row["chi_2x2_NNcurrent"] = m_items[qwin]
        row["q_winner_2x2"] = qwin
    else:
        row["chi_2x2_NNcurrent"] = np.nan
        row["q_winner_2x2"] = ""

    # overall winner among requested q's
    qwin_all = max(q_chis, key=q_chis.get)
    row["chi_max_requested_q"] = q_chis[qwin_all]
    row["q_winner_all"] = qwin_all

    # explicitly delete O and big arrays
    del O, C, D, Lraw
    gc.collect()
    return row

# -----------------------------
# CLI
# -----------------------------

def main():
    ap = argparse.ArgumentParser(description="NN imaginary current source susceptibility from divERGe O.dvg, V1.")
    ap.add_argument("--root", required=True, help="Root containing case folders or one case folder")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--snapshot", default="final", help="Snapshot name, default final")
    ap.add_argument("--mu", default="rho", choices=["rho", "sx", "sy", "sz"], help="charge/spin bilinear sector")
    ap.add_argument("--q", nargs="+", default=["Gamma", "M1", "M2", "M3"], help="q labels")
    ap.add_argument("--max-cases", type=int, default=None, help="Debug: only process first N cases")
    ap.add_argument("--case-glob", default=None, help="Optional substring/glob-ish filter on case name")
    ap.add_argument("--pole-tol", type=float, default=1e-8)
    ap.add_argument("--no-save-vectors", action="store_true", help="Do not save leading eta vectors")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    outdir = Path(args.out).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "vectors").mkdir(exist_ok=True)

    # Determine cases.
    if odvg_path_for_case(root, snapshot=args.snapshot) is not None:
        cases = [root]
    else:
        cases = find_case_dirs(root)

    if args.case_glob:
        pat = args.case_glob
        cases = [c for c in cases if pat in c.name]

    if args.max_cases is not None:
        cases = cases[: args.max_cases]

    print(f"[INFO] root={root}")
    print(f"[INFO] outdir={outdir}")
    print(f"[INFO] cases={len(cases)}")
    print(f"[INFO] q={args.q}, mu={args.mu}, snapshot={args.snapshot}")

    rows = []
    errors = []

    for idx, case in enumerate(cases, 1):
        print(f"\n[CASE {idx}/{len(cases)}] {case.name}", flush=True)
        try:
            row = analyze_case(
                case_dir=case,
                outdir=outdir,
                q_labels=args.q,
                mu=args.mu,
                snapshot=args.snapshot,
                pole_tol=args.pole_tol,
                save_vectors=(not args.no_save_vectors),
            )
            rows.append(row)
            print(f"  chi_1x1={row.get('chi_1x1_NNcurrent')} chi_2x2={row.get('chi_2x2_NNcurrent')} q2={row.get('q_winner_2x2')}", flush=True)
        except Exception as e:
            print(f"  [ERROR] {repr(e)}", flush=True)
            errors.append({"case": case.name, "error": repr(e)})
            gc.collect()

        # Write incrementally in case job dies.
        if rows:
            pd.DataFrame(rows).to_csv(outdir / "nn_current_source_susceptibility_rows.csv", index=False)
        if errors:
            pd.DataFrame(errors).to_csv(outdir / "errors.csv", index=False)

    if rows:
        df = pd.DataFrame(rows)
        # sort if possible
        sort_cols = [c for c in ["lambda_nn", "lambda_nnn", "U", "V1"] if c in df.columns]
        if sort_cols:
            df = df.sort_values(sort_cols)
        df.to_csv(outdir / "nn_current_source_susceptibility_summary.csv", index=False)
        print("\n[DONE] wrote", outdir / "nn_current_source_susceptibility_summary.csv")
    else:
        print("\n[WARN] no successful rows")

    if errors:
        print("[WARN] errors written to", outdir / "errors.csv")


if __name__ == "__main__":
    main()
