#!/usr/bin/env python3
# coding: utf-8
"""
TUfRG_instability.py
Version: structuraldiag-top8-2026-05-19

Level-1 instability inspector for divERGe TUFRG output.

This version adds a hard guard against projected explicit-spin outputs and keeps the calibrated non-SU2 PH channel construction.
It uses the no-SOC SU(2) calibration found from the explicit-spin output:

    K_ph^phys(q) = D(q) - C(q)

with crossed PH leg projection

    raw slots       = [s0, s1, s2, s3]
    PH bilinear 1  = (out1, in1) = (s0, s3)
    PH bilinear 2  = (out2, in2) = (s1, s2)

or leg_perm = (0, 3, 1, 2), assuming raw spin-pair order

    pair_order = (uu, ud, du, dd).

The physical PH sectors are then projected with normalized Pauli matrices:

    rho = I/sqrt(2), sx = sigma_x/sqrt(2),
    sy = sigma_y/sqrt(2), sz = sigma_z/sqrt(2).

For no-SOC explicit-spin runs, this convention should give Sx/Sy/Sz nearly
degenerate.  The script writes a calibration table into the report.

Important scope
---------------
1. This is a Level-1 file. It decides instability strength from loop-dressed
   singular values of K(q) @ L(q), not from raw Pvec/Cvec/Dvec.
2. It does not classify form factors. Use TUfRG_formfactor_from_modes_v1.py on
   the output modes.
3. Default --n-modes is 8 so downstream structural order classifiers can inspect near-degenerate subspaces.
4. For SU2-projected structural output, the established convention is retained:

       K_sc     = P_saved
       K_charge = D_saved             if tu_channel_calc_project != 0
       K_mag    = -C_saved            if tu_channel_calc_project < 0

   For unprojected SU2 fallback:

       K_charge = 2D - C,   K_mag = C.

Outputs
-------
case_dir/instability_v2/instability_summary.csv
case_dir/instability_v2/instability_modes.npz
case_dir/instability_v2/instability_report.txt
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    import TUfRG_model as km
except Exception:  # pragma: no cover
    km = None

DEFAULT_Q = ("Gamma", "M1", "M2", "M3")
PAIR_ORDER = ("uu", "ud", "du", "dd")
PH_LEG_PERM = (0, 3, 1, 2)
SPIN_VALUE = {"u": 0, "d": 1}
PH_SECTORS = ("rho", "sx", "sy", "sz")
PP_SECTORS = ("singlet", "triplet0", "triplet_plus", "triplet_minus")


@dataclass
class CaseData:
    case_dir: Path
    metadata: Dict[str, Any]
    O: Any
    T: Any | None
    nk: int
    q_indices: Dict[str, int]
    SU2: int
    n_spin: int
    nff: int
    tu_project: Optional[int]


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def load_output(path: Path):
    import diverge.output as ro
    return ro.read(str(path))


def extract_nested(d: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, Mapping) or k not in cur:
            return default
        cur = cur[k]
    return cur


def infer_nk_from_O(O: Any, metadata: Mapping[str, Any]) -> int:
    nk_meta = extract_nested(metadata, "mesh", "nk", default=None)
    if nk_meta is not None:
        return int(nk_meta)
    nq = int(getattr(O, "nk", 1))
    return int(round(math.sqrt(nq)))


def q_indices_from_nk(nk: int) -> Dict[str, int]:
    if nk % 2 != 0:
        half = nk // 2
        return {
            "Gamma": 0,
            "M1_nearest": half,
            "M2_nearest": half * nk,
            "M3_nearest": half * nk + half,
        }
    return {
        "Gamma": 0,
        "M1": nk // 2,
        "M2": (nk // 2) * nk,
        "M3": (nk // 2) * nk + nk // 2,
    }


def load_case(case_dir: Path) -> CaseData:
    case_dir = Path(case_dir).resolve()
    metadata = read_json(case_dir / "metadata.json")
    O = load_output(case_dir / "O.dvg")
    T = load_output(case_dir / "T.dvg") if (case_dir / "T.dvg").exists() else None
    nk = infer_nk_from_O(O, metadata)
    q_indices = extract_nested(metadata, "model_summary", "metadata", "q_indices", default=None)
    if q_indices is None:
        q_indices = extract_nested(metadata, "q_indices", default=None)
    if q_indices is None:
        q_indices = q_indices_from_nk(nk)
    q_indices = {str(k): int(v) for k, v in q_indices.items()}
    pp_report = metadata.get("postprocess_report", {})
    tu_project = pp_report.get("tu_channel_calc_project", None)
    if tu_project is not None:
        try:
            tu_project = int(tu_project)
        except Exception:
            tu_project = None
    nff = int(getattr(O, "n_orbff", 0) or np.asarray(getattr(O, "mi_to_ofrom")).size)
    return CaseData(
        case_dir=case_dir,
        metadata=metadata,
        O=O,
        T=T,
        nk=nk,
        q_indices=q_indices,
        SU2=int(getattr(O, "SU2", 0)),
        n_spin=int(getattr(O, "n_spin", 2 if int(getattr(O, "SU2", 0)) == 0 else 1)),
        nff=nff,
        tu_project=tu_project,
    )


def assert_raw_channel_compatibility(data: CaseData) -> None:
    """Fail fast for the convention that broke the explicit-spin bridge test.

    Original divERGe code shows that tu_channel_calc_project != 0 stores
    projected channel objects rather than raw Pch/Cch/Dch.  The explicit-spin
    diagnosis below assumes raw unprojected P/C/D tensors and constructs
    K_ph = D - C before Pauli projection.
    """
    if data.SU2 == 0 and data.n_spin == 2 and data.tu_project not in (None, 0):
        raise RuntimeError(
            "Explicit-spin diagnosis requires raw unprojected P/C/D channels. "
            f"This output has tu_channel_calc_project={data.tu_project}. "
            "Rerun TUfRG_runner.py with --tu-channel-calc-project 0."
        )


def q_reduced(iq: int, nk: int) -> Tuple[float, float, float]:
    iq = int(iq)
    nk = int(nk)
    return (float(iq % nk) / nk, float(iq // nk) / nk, 0.0)


def q_labels_to_indices(data: CaseData, labels: Sequence[str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    nq = int(getattr(data.O, "nk", data.nk * data.nk))
    for lab in labels:
        if lab in data.q_indices:
            out[lab] = int(data.q_indices[lab])
            continue
        if lab == "M1" and "M1_nearest" in data.q_indices:
            out["M1_nearest"] = int(data.q_indices["M1_nearest"])
            continue
        if lab == "M2" and "M2_nearest" in data.q_indices:
            out["M2_nearest"] = int(data.q_indices["M2_nearest"])
            continue
        if lab == "M3" and "M3_nearest" in data.q_indices:
            out["M3_nearest"] = int(data.q_indices["M3_nearest"])
            continue
        try:
            iq = int(lab)
        except ValueError as exc:
            raise ValueError(f"Unknown q label {lab!r}; known labels={sorted(data.q_indices)}") from exc
        if iq < 0 or iq >= nq:
            raise ValueError(f"q index {iq} out of range [0,{nq})")
        out[f"iq{iq}"] = iq
    return out


def channel_tensor(O: Any, X: str, iq: int) -> np.ndarray:
    arr = np.asarray(getattr(O, f"{X.upper()}channel"), dtype=complex)
    if arr.ndim != 5:
        raise ValueError(f"{X}channel should have shape (spinpair,nff,spinpair,nff,nq); got {arr.shape}")
    return np.asarray(arr[:, :, :, :, int(iq)], dtype=complex)


def loop_tensor(O: Any, kind: str, iq: int) -> np.ndarray:
    attr = "pploop" if kind == "pp" else "phloop"
    arr = np.asarray(getattr(O, attr), dtype=complex)
    if arr.ndim != 5:
        raise ValueError(f"{attr} should have shape (spinpair,nff,spinpair,nff,nq); got {arr.shape}")
    return np.asarray(arr[:, :, :, :, int(iq)], dtype=complex)


def flatten_1spin(A: np.ndarray) -> np.ndarray:
    return np.asarray(A[0, :, 0, :], dtype=complex)


def spin_vector_pp(label: str) -> np.ndarray:
    label = label.lower()
    if label == "singlet":
        return np.array([0, 1, -1, 0], dtype=complex) / np.sqrt(2)
    if label == "triplet0":
        return np.array([0, 1, 1, 0], dtype=complex) / np.sqrt(2)
    if label == "triplet_plus":
        return np.array([1, 0, 0, 0], dtype=complex)
    if label == "triplet_minus":
        return np.array([0, 0, 0, 1], dtype=complex)
    raise ValueError(label)


def project_spin_tensor_direct(A: np.ndarray, left: np.ndarray, right: Optional[np.ndarray] = None) -> np.ndarray:
    """Direct spin-pair projection for PP: (spinpair,nff,spinpair,nff) -> (nff,nff)."""
    if right is None:
        right = left
    A = np.asarray(A, dtype=complex)
    left = np.asarray(left, dtype=complex).reshape(A.shape[0])
    right = np.asarray(right, dtype=complex).reshape(A.shape[2])
    return np.einsum("a,aibj,b->ij", left.conj(), A, right, optimize=True)


def spin_matrix_ph(label: str) -> np.ndarray:
    label = label.lower()
    if label == "rho":
        return np.array([[1, 0], [0, 1]], dtype=complex) / np.sqrt(2)
    if label == "sx":
        return np.array([[0, 1], [1, 0]], dtype=complex) / np.sqrt(2)
    if label == "sy":
        return np.array([[0, -1j], [1j, 0]], dtype=complex) / np.sqrt(2)
    if label == "sz":
        return np.array([[1, 0], [0, -1]], dtype=complex) / np.sqrt(2)
    raise ValueError(label)


def decode_pair(raw_pair_index: int) -> Tuple[int, int]:
    lab = PAIR_ORDER[int(raw_pair_index)]
    return SPIN_VALUE[lab[0]], SPIN_VALUE[lab[1]]


def project_ph_crossed(A: np.ndarray, left_sector: str, right_sector: Optional[str] = None) -> np.ndarray:
    """
    Calibrated explicit-spin PH projection.

    Raw A shape: (4,nff,4,nff), with raw spin slots [s0,s1,s2,s3]
    from row-pair and column-pair.  The calibrated PH bilinear grouping is

        (out1,in1; out2,in2) = (s0,s3; s1,s2)

    i.e. PH_LEG_PERM=(0,3,1,2).
    """
    if right_sector is None:
        right_sector = left_sector
    A = np.asarray(A, dtype=complex)
    nff = A.shape[1]
    Ml = spin_matrix_ph(left_sector)
    Mr = spin_matrix_ph(right_sector)
    out = np.zeros((nff, nff), dtype=complex)
    for rp in range(4):
        r0, r1 = decode_pair(rp)
        for cp in range(4):
            c0, c1 = decode_pair(cp)
            raw_slots = [r0, r1, c0, c1]
            out1 = raw_slots[PH_LEG_PERM[0]]
            in1 = raw_slots[PH_LEG_PERM[1]]
            out2 = raw_slots[PH_LEG_PERM[2]]
            in2 = raw_slots[PH_LEG_PERM[3]]
            w = np.conjugate(Ml[out1, in1]) * Mr[out2, in2]
            if abs(w) > 0:
                out += w * A[rp, :, cp, :]
    return out


def svd_strength_and_vectors(A: np.ndarray, n_modes: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    U, s, Vh = np.linalg.svd(np.asarray(A, dtype=complex), full_matrices=False)
    m = min(int(n_modes), len(s))
    right = Vh[:m, :].conj()
    left = U[:, :m].T
    return s[:m], right, left


def hermitian_eigs(A: np.ndarray, n_modes: int) -> Tuple[np.ndarray, np.ndarray]:
    H = 0.5 * (A + A.conj().T)
    vals, vecs = np.linalg.eigh(H)
    order = np.argsort(-np.abs(vals))[: int(n_modes)]
    return vals[order], vecs[:, order].T


def matrix_norms(K: np.ndarray, L: np.ndarray) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for name, M in [("K", K), ("L", L)]:
        out[f"{name}_fro_norm"] = float(np.linalg.norm(M))
        out[f"{name}_hermiticity_rel"] = float(np.linalg.norm(M - M.conj().T) / max(np.linalg.norm(M), 1e-300))
    return out


def built_in_S(data: CaseData, sector: str) -> Optional[np.ndarray]:
    if sector == "mag":
        x = getattr(data.O, "S_mag", None)
    elif sector == "charge":
        x = getattr(data.O, "S_charge", None)
    elif sector == "sc":
        x = getattr(data.O, "S_sc", None)
    else:
        return None
    return None if x is None else np.asarray(x, dtype=float)


def tu_component_arrays(O: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return TU form-factor maps (R, ofrom, oto)."""
    return (
        np.asarray(getattr(O, "mi_to_R"), dtype=int),
        np.asarray(getattr(O, "mi_to_ofrom"), dtype=int),
        np.asarray(getattr(O, "mi_to_oto"), dtype=int),
    )


def onsite_A1_template(O: Any) -> np.ndarray:
    """Uniform onsite A1 vector in the TU form-factor basis."""
    R, ofrom, oto = tu_component_arrays(O)
    onsite = (ofrom == oto) & np.all(R == 0, axis=1)
    t = np.zeros(len(ofrom), dtype=complex)
    t[onsite] = 1.0
    return t


def projector_remove_template(t: np.ndarray) -> np.ndarray:
    """Return I-|t><t| for a nonzero template vector, otherwise identity."""
    t = np.asarray(t, dtype=complex).reshape(-1)
    n = t.size
    norm = np.linalg.norm(t)
    if norm <= 0:
        return np.eye(n, dtype=complex)
    u = t / norm
    return np.eye(n, dtype=complex) - np.outer(u, u.conj())


def projector_remove_onsite_components(O: Any) -> np.ndarray:
    """Project out all onsite diagonal form-factor basis components."""
    R, ofrom, oto = tu_component_arrays(O)
    onsite = (ofrom == oto) & np.all(R == 0, axis=1)
    P = np.eye(len(ofrom), dtype=complex)
    P[onsite, :] = 0.0
    P[:, onsite] = 0.0
    return P


def q0_charge_projection_diagnostics(O: Any, A: np.ndarray) -> Dict[str, float]:
    """Return q=0 charge strengths after removing forward-scattering-like directions.

    This is mainly for PH rho/charge at Gamma.  The raw Gamma charge sector often
    contains a uniform compressibility / forward-scattering-like A1 component.  We
    report both the raw strength and the strengths after removing the onsite A1
    direction or all onsite components, instead of silently treating raw Gamma
    charge as a physical ordered phase.
    """
    A = np.asarray(A, dtype=complex)
    P_no_A1 = projector_remove_template(onsite_A1_template(O))
    P_no_onsite = projector_remove_onsite_components(O)
    return {
        "strength_after_remove_onsite_A1": float(np.linalg.svd(P_no_A1 @ A @ P_no_A1, compute_uv=False)[0]),
        "strength_after_remove_all_onsite": float(np.linalg.svd(P_no_onsite @ A @ P_no_onsite, compute_uv=False)[0]),
    }


def build_su2_kernels(data: CaseData, iq: int) -> List[Dict[str, Any]]:
    P = channel_tensor(data.O, "P", iq)
    C = channel_tensor(data.O, "C", iq)
    D = channel_tensor(data.O, "D", iq)
    Lpp = loop_tensor(data.O, "pp", iq)
    Lph = loop_tensor(data.O, "ph", iq)
    p = 0 if data.tu_project is None else int(data.tu_project)
    if p != 0:
        K_charge = flatten_1spin(D)
        if p < 0:
            K_mag = -flatten_1spin(C)
            mag_note = "projected SU2: K_mag=-C_saved"
        else:
            K_mag = flatten_1spin(C)
            mag_note = "projected SU2: K_mag=C_saved (project sign not negative)"
        charge_note = "projected SU2: K_charge=D_saved=(2D-C)_projected"
    else:
        K_charge = 2 * flatten_1spin(D) - flatten_1spin(C)
        K_mag = flatten_1spin(C)
        charge_note = "unprojected SU2: K_charge=2D-C"
        mag_note = "unprojected SU2: K_mag=C"
    return [
        {"family": "pp", "sector": "sc", "convention": "su2", "K": flatten_1spin(P), "L": flatten_1spin(Lpp), "note": "SU2: K_sc=P"},
        {"family": "ph", "sector": "charge", "convention": "su2", "K": K_charge, "L": flatten_1spin(Lph), "note": charge_note},
        {"family": "ph", "sector": "mag", "convention": "su2", "K": K_mag, "L": flatten_1spin(Lph), "note": mag_note},
    ]


def build_explicit_kernels(data: CaseData, iq: int) -> List[Dict[str, Any]]:
    P = channel_tensor(data.O, "P", iq)
    C = channel_tensor(data.O, "C", iq)
    D = channel_tensor(data.O, "D", iq)
    Lpp = loop_tensor(data.O, "pp", iq)
    Lph = loop_tensor(data.O, "ph", iq)
    rows: List[Dict[str, Any]] = []

    # PP sectors use direct spin-pair singlet/triplet projection.
    for sec in PP_SECTORS:
        e = spin_vector_pp(sec)
        rows.append({
            "family": "pp",
            "sector": f"pp-{sec.replace('_', '-')}",
            "convention": "explicit-P-spin-projected-direct",
            "K": project_spin_tensor_direct(P, e),
            "L": project_spin_tensor_direct(Lpp, e),
            "note": f"explicit spin PP direct singlet/triplet projection {sec}",
        })

    # Calibrated PH convention.
    Kph = D - C
    for sec in PH_SECTORS:
        rows.append({
            "family": "ph",
            "sector": f"ph-{sec}",
            "convention": "explicit-calibrated-DminusC-crossed0312",
            "K": project_ph_crossed(Kph, sec),
            "L": project_ph_crossed(Lph, sec),
            "note": "explicit spin calibrated PH: K=D-C, crossed legs (s0,s3;s1,s2), pair order uu ud du dd"
        })
    return rows


def mode_key(row_id: int, q_label: str, family: str, sector: str, convention: str, mode: int) -> str:
    safe = f"r{row_id:05d}_{q_label}_{family}_{sector}_{convention}_m{mode}"
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in safe)


def calibration_rows_explicit(data: CaseData, qmap: Mapping[str, int]) -> List[Dict[str, Any]]:
    """No-SOC SU(2) calibration table for explicit-spin PH convention."""
    rows: List[Dict[str, Any]] = []
    if data.SU2 != 0 or data.n_spin != 2:
        return rows
    for q_label, iq in qmap.items():
        C = channel_tensor(data.O, "C", iq)
        D = channel_tensor(data.O, "D", iq)
        Lph = loop_tensor(data.O, "ph", iq)
        Kph = D - C
        strengths: Dict[str, float] = {}
        for sec in PH_SECTORS:
            Ksec = project_ph_crossed(Kph, sec)
            Lsec = project_ph_crossed(Lph, sec)
            strengths[sec] = float(np.linalg.svd(Ksec @ Lsec, compute_uv=False)[0])
        trip = np.array([strengths["sx"], strengths["sy"], strengths["sz"]], dtype=float)
        rows.append({
            "q_label": q_label,
            "iq": int(iq),
            "rho": strengths["rho"],
            "sx": strengths["sx"],
            "sy": strengths["sy"],
            "sz": strengths["sz"],
            "triplet_mean": float(np.mean(trip)),
            "triplet_cv": float(np.std(trip) / max(np.mean(trip), 1e-300)),
            "triplet_spread": float((np.max(trip) - np.min(trip)) / max(np.max(trip), 1e-300)),
        })
    return rows


def analyze_case(data: CaseData, q_labels: Sequence[str], *, n_modes: int, method: str) -> Tuple[List[Dict[str, Any]], Dict[str, np.ndarray], str, List[Dict[str, Any]]]:
    qmap = q_labels_to_indices(data, q_labels)
    rows: List[Dict[str, Any]] = []
    modes: Dict[str, np.ndarray] = {}
    report: List[str] = []
    report.append(f"case_dir: {data.case_dir}")
    report.append(f"SU2={data.SU2}, n_spin={data.n_spin}, nff={data.nff}, nk={data.nk}, tu_project={data.tu_project}")
    if data.nk % 2 != 0:
        report.append("WARNING: nk is odd; exact M points are not on the q mesh. M labels are nearest-grid only.")
    report.append("Level-1 object: singular values of K(q) @ L(q), not raw P/C/D decompositions.")
    if data.SU2 == 0 and data.n_spin == 2:
        report.append("Explicit-spin PH convention: K_ph=D-C, crossed leg projection (s0,s3;s1,s2), pair order uu ud du dd.")
    report.append("")

    cal_rows = calibration_rows_explicit(data, qmap)
    if cal_rows:
        report.append("Explicit-spin no-SOC/SU2 calibration table for calibrated PH convention:")
        for r in cal_rows:
            report.append(
                f"  q={r['q_label']:8s} rho={r['rho']:.8e} sx={r['sx']:.8e} sy={r['sy']:.8e} sz={r['sz']:.8e} "
                f"triplet_cv={r['triplet_cv']:.3e} triplet_spread={r['triplet_spread']:.3e}"
            )
        report.append("")

    row_id = 0
    for q_label, iq in qmap.items():
        report.append(f"q={q_label}, iq={iq}, reduced={q_reduced(iq, data.nk)}")
        if data.SU2 == 1 or data.n_spin == 1:
            kernels = build_su2_kernels(data, iq)
        else:
            kernels = build_explicit_kernels(data, iq)
        for spec in kernels:
            K = np.asarray(spec["K"], dtype=complex)
            L = np.asarray(spec["L"], dtype=complex)
            A = K @ L
            s, right, left = svd_strength_and_vectors(A, n_modes)
            eig_vals = np.array([], dtype=complex)
            if method in {"both", "hermitian"}:
                try:
                    eig_vals, _ = hermitian_eigs(A, n_modes)
                except Exception:
                    pass
            norms = matrix_norms(K, L)
            Sref = built_in_S(data, spec["sector"] if spec["sector"] in {"mag", "charge", "sc"} else "")
            for m in range(len(s)):
                key = mode_key(row_id, q_label, str(spec["family"]), str(spec["sector"]), str(spec["convention"]), m)
                modes[key] = right[m]
                row = {
                    "row_id": row_id,
                    "mode_key": key,
                    "case": data.case_dir.name,
                    "q_label": q_label,
                    "iq": int(iq),
                    "q_reduced": str(q_reduced(iq, data.nk)),
                    "family": spec["family"],
                    "sector": spec["sector"],
                    "convention": spec["convention"],
                    "mode": m,
                    "strength_svd_KL": float(s[m]),
                    "strength_source": "svd(K@L)",
                    "kernel_note": spec["note"],
                    "vector_kind": "right_singular_vector_of_KL_in_TU_formfactor_basis",
                    **norms,
                }
                if q_label == "Gamma" and str(spec["family"]) == "ph" and str(spec["sector"]) in {"ph-rho", "charge"}:
                    row.update(q0_charge_projection_diagnostics(data.O, A))
                if Sref is not None and q_label == "Gamma" and m < len(Sref):
                    row["builtin_S_q0_reference"] = float(Sref[m])
                    row["builtin_S_relative_mismatch"] = float(abs(s[m] - Sref[m]) / max(abs(Sref[m]), 1e-300))
                if m < len(eig_vals):
                    ev = eig_vals[m]
                    row["eig_herm_KL_abs_sorted_value"] = float(np.real(ev)) if abs(np.imag(ev)) < 1e-12 else complex(ev)
                rows.append(row)
                row_id += 1
            report.append(f"  {spec['family']:2s} {spec['sector']:18s} {spec['convention']:45s} top={s[0]:.8e} note={spec['note']}")
        report.append("")
    return rows, modes, "\n".join(report), cal_rows


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


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Calibrated loop-dressed Level-1 TUFRG instability diagnosis.")
    p.add_argument("case_dir", help="case directory containing O.dvg")
    p.add_argument("--q", nargs="+", default=list(DEFAULT_Q), help="q labels or numeric q indices")
    p.add_argument("--n-modes", type=int, default=8, help="number of singular modes saved per sector/q; default increased for near-degenerate M-point/subspace diagnosis")
    p.add_argument("--method", choices=["svd", "hermitian", "both"], default="svd")
    p.add_argument("--out", default="instability")
    return p.parse_args(argv)



def append_gamma_A1proj_modes(data: CaseData, rows: List[Dict[str, Any]], modes: Dict[str, np.ndarray], *, n_modes: int) -> List[Dict[str, Any]]:
    """Append mode-aligned Gamma ph-rho/charge modes after removing onsite uniform A1.

    Old diagnosis only reported strength_after_remove_onsite_A1 for raw Gamma
    ph-rho rows.  That made strength and structure not mode-aligned.

    This function fixes it by constructing

        Aproj = P_noA1 @ (K_Gamma_rho @ L_Gamma_rho) @ P_noA1

    then re-SVDing Aproj and saving the projected right singular vectors as
    normal instability modes with sector ph-rho-A1proj or charge-A1proj.
    """
    qmap = q_labels_to_indices(data, ["Gamma"])
    iq = int(qmap["Gamma"])

    if data.SU2 == 1 or data.n_spin == 1:
        kernels = build_su2_kernels(data, iq)
        target_sectors = {"charge"}
    else:
        kernels = build_explicit_kernels(data, iq)
        target_sectors = {"ph-rho"}

    P_no_A1 = projector_remove_template(onsite_A1_template(data.O))

    start_row_id = 0
    if rows:
        start_row_id = max(int(r.get("row_id", 0)) for r in rows) + 1

    new_rows: List[Dict[str, Any]] = []

    for spec in kernels:
        family = str(spec["family"])
        sector = str(spec["sector"])
        if family != "ph" or sector not in target_sectors:
            continue

        K = np.asarray(spec["K"], dtype=complex)
        L = np.asarray(spec["L"], dtype=complex)
        A = K @ L
        Aproj = P_no_A1 @ A @ P_no_A1

        svals, right, left = svd_strength_and_vectors(Aproj, n_modes)
        norms = matrix_norms(K, L)

        for m in range(len(svals)):
            vec = P_no_A1 @ right[m]
            nrm = np.linalg.norm(vec)
            if nrm > 1e-14:
                vec = vec / nrm

            new_sector = f"{sector}-A1proj"
            key = mode_key(
                start_row_id + len(new_rows),
                "Gamma",
                family,
                new_sector,
                "onsiteA1_projected_KL_resvd",
                m,
            )
            modes[key] = vec

            row = {
                "row_id": start_row_id + len(new_rows),
                "mode_key": key,
                "case": data.case_dir.name,
                "q_label": "Gamma",
                "iq": iq,
                "q_reduced": str(q_reduced(iq, data.nk)),
                "family": family,
                "sector": new_sector,
                "convention": "onsiteA1_projected_KL_resvd",
                "mode": m,
                "strength_svd_KL": float(svals[m]),
                "strength_source": "svd(P_no_onsite_A1 @ K@L @ P_no_onsite_A1)",
                "kernel_note": (
                    "Gamma ph-rho/charge re-SVD after removing onsite uniform A1; "
                    "saved vector is projected and renormalized"
                ),
                "vector_kind": "right_singular_vector_of_projected_KL_in_TU_formfactor_basis",
                **norms,
            }
            new_rows.append(row)

    rows.extend(new_rows)
    return new_rows

def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    data = load_case(Path(args.case_dir))
    assert_raw_channel_compatibility(data)
    rows, modes, report, cal_rows = analyze_case(data, args.q, n_modes=args.n_modes, method=args.method)

    # Fix Gamma ph-rho mode-alignment:
    # add projected Gamma charge/rho modes after removing onsite uniform A1.
    new_A1_rows = append_gamma_A1proj_modes(data, rows, modes, n_modes=args.n_modes)
    if new_A1_rows:
        report += "\n\nGamma onsite-A1-projected re-SVD modes added:\n"
        for r in new_A1_rows:
            report += (
                f"  {r['mode_key']}  sector={r['sector']}  "
                f"strength={float(r['strength_svd_KL']):.8e}\n"
            )

    outdir = data.case_dir / str(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    write_csv(outdir / "instability_summary.csv", rows)
    write_csv(outdir / "explicit_spin_ph_calibration.csv", cal_rows)
    np.savez_compressed(outdir / "instability_modes.npz", **modes)
    (outdir / "instability_report.txt").write_text(report)
    print(report)
    print("wrote", outdir / "instability_summary.csv")
    print("wrote", outdir / "explicit_spin_ph_calibration.csv")
    print("wrote", outdir / "instability_modes.npz")
    print("wrote", outdir / "instability_report.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
