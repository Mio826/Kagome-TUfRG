#!/usr/bin/env python3
# coding: utf-8
"""
TUfRG_diagnosis.py

Version: v3-no-spinflip-recommend-2026-05-12

General q-resolved diagnosis for spinful kagome Kane-Mele divERGe TUFRG outputs.

Purpose
-------
This script reads a completed runner output directory containing
    T.dvg, F.dat, O.dvg, metadata.json
and performs a conservative, general diagnosis:

  * q-resolved mode extraction using Xoff[iq] + mode_index;
  * spin-tensor decomposition of P/C/D eigenvectors;
  * onsite / NN / longer-range TU-form-factor weights;
  * leading TU components;
  * near-degeneracy checks within each fixed (channel, q) sector;
  * optional loop-dressed singular-value probes of K(q) L(q) and L(q) K(q).

It deliberately does NOT reconstruct real-space patterns.  Use
TUfRG_realspace.py after this file identifies an interesting candidate mode.

Important interpretation notes
------------------------------
* Pval/Cval/Dval and Pvec/Cvec/Dvec are saved decompositions of channel
  matrices, not automatically the same as bubble-dressed susceptibility modes.
* divERGe built-in S_* outputs are q=0 references only.
* For explicit-spin non-SU2 runs, this script treats P/C/D as saved channels
  and diagnoses their spin tensor structure; it does not blindly relabel C/D as
  magnetic/charge channels.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    import TUfRG_model as km
except Exception:  # pragma: no cover
    km = None


CHANNELS = ("P", "C", "D")
DEFAULT_Q_LABELS = ("Gamma", "M1", "M2", "M3")


@dataclass
class CaseData:
    case_dir: Path
    metadata: Dict[str, Any]
    O: Any
    q_indices: Dict[str, int]
    nk: int


# -----------------------------------------------------------------------------
# Basic IO
# -----------------------------------------------------------------------------

def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def load_output(path: Path):
    import diverge.output as ro
    return ro.read(str(path))


def is_case_dir(path: Path) -> bool:
    return path.is_dir() and (path / "O.dvg").exists()


def find_case_dirs(path: Path, recursive: bool = False) -> List[Path]:
    path = path.resolve()
    if is_case_dir(path):
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(path)
    if recursive:
        out = sorted({p.parent for p in path.rglob("O.dvg")})
    else:
        out = sorted([p for p in path.iterdir() if is_case_dir(p)])
    if not out:
        raise FileNotFoundError(f"No case directories with O.dvg found under {path}")
    return out


def extract_nested(d: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, Mapping) or k not in cur:
            return default
        cur = cur[k]
    return cur


def load_case(case_dir: Path) -> CaseData:
    metadata = read_json(case_dir / "metadata.json")
    O = load_output(case_dir / "O.dvg")

    q_indices = extract_nested(metadata, "model_summary", "metadata", "q_indices", default=None)
    if q_indices is None:
        q_indices = extract_nested(metadata, "q_indices", default=None)
    if q_indices is None:
        mesh = metadata.get("mesh", {})
        nk = int(mesh.get("nk", round(math.sqrt(int(getattr(O, "nk", 1))))))
        if km is None:
            raise RuntimeError("Cannot infer q indices without metadata q_indices or TUfRG_model.py")
        q_indices = km.high_symmetry_q_indices(nk)

    q_indices = {str(k): int(v) for k, v in q_indices.items()}
    nk = int(extract_nested(metadata, "mesh", "nk", default=0) or round(math.sqrt(int(getattr(O, "nk", 1)))))
    if nk <= 0:
        # O.nk is often nq, not nk; fall back from q index maximum.
        max_iq = max(q_indices.values())
        nk = int(round(math.sqrt(max(int(getattr(O, "nk", max_iq + 1)), max_iq + 1))))
    return CaseData(case_dir=case_dir, metadata=metadata, O=O, q_indices=q_indices, nk=nk)


# -----------------------------------------------------------------------------
# q-resolved extraction
# -----------------------------------------------------------------------------

def channel_arrays(O: Any, X: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X = X.upper()
    vals = np.asarray(getattr(O, f"{X}val"))
    vecs = np.asarray(getattr(O, f"{X}vec"))
    off = np.asarray(getattr(O, f"{X}off"), dtype=int)
    length = np.asarray(getattr(O, f"{X}len"), dtype=int)
    return vals, vecs, off, length


def mode_global_loc(O: Any, X: str, iq: int, mode: int) -> int:
    vals, vecs, off, length = channel_arrays(O, X)
    iq = int(iq)
    mode = int(mode)
    if iq < 0 or iq >= len(off):
        raise IndexError(f"iq={iq} out of range for {X}off length {len(off)}")
    if mode < 0 or mode >= int(length[iq]):
        raise IndexError(f"mode={mode} out of range for channel {X}, iq={iq}, available={int(length[iq])}")
    return int(off[iq]) + mode


def get_mode(O: Any, X: str, iq: int, mode: int) -> Tuple[float, np.ndarray, int]:
    vals, vecs, off, length = channel_arrays(O, X)
    loc = mode_global_loc(O, X, iq, mode)
    return float(vals[loc]), np.asarray(vecs[loc], dtype=complex), loc


def q_reduced_from_iq(iq: int, nk: int) -> Tuple[float, float, float]:
    iq = int(iq)
    nk = int(nk)
    ix = iq % nk
    iy = iq // nk
    return (ix / nk, iy / nk, 0.0)


# -----------------------------------------------------------------------------
# Spin decomposition
# -----------------------------------------------------------------------------

def ensure_spin_vec(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=complex)
    if v.ndim != 3 or v.shape[0] != 2 or v.shape[1] != 2:
        raise ValueError(f"Expected vector shape (2,2,nff), got {v.shape}")
    return v


def norm2(x: np.ndarray) -> float:
    return float(np.vdot(x, x).real)


def norm(x: np.ndarray) -> float:
    return float(np.sqrt(max(norm2(x), 0.0)))


def spin_combinations(v: np.ndarray) -> Dict[str, np.ndarray]:
    v = ensure_spin_vec(v)
    uu, ud, du, dd = v[0, 0, :], v[0, 1, :], v[1, 0, :], v[1, 1, :]
    return {
        "uu": uu,
        "ud": ud,
        "du": du,
        "dd": dd,
        "ph_charge": uu + dd,
        "ph_spin_z": uu - dd,
        "pp_singlet": ud - du,
        "pp_triplet0": ud + du,
        "pp_triplet_plus": uu,
        "pp_triplet_minus": dd,
    }


def display_sector_name(name: str) -> str:
    """Convert internal underscore sector names to CLI/output names.

    Internal arrays use keys like ``ph_charge`` because they are Python dict
    keys.  User-facing CSV/CLI names use hyphens, e.g. ``ph-charge``.
    """
    return str(name).replace("_", "-")


def internal_sector_name(name: str) -> str:
    """Convert user-facing hyphen sector names back to internal keys."""
    return str(name).replace("-", "_")


def spin_norm_summary(v: np.ndarray) -> Dict[str, float]:
    c = spin_combinations(v)
    return {
        "norm_total": norm(v.reshape(-1)),
        "norm_uu": norm(c["uu"]),
        "norm_ud": norm(c["ud"]),
        "norm_du": norm(c["du"]),
        "norm_dd": norm(c["dd"]),
        "PH_charge_norm": norm(c["ph_charge"]),
        "PH_Sz_norm": norm(c["ph_spin_z"]),
        "PP_singlet_norm": norm(c["pp_singlet"]),
        "PP_triplet0_norm": norm(c["pp_triplet0"]),
        "PP_triplet_plus_norm": norm(c["pp_triplet_plus"]),
        "PP_triplet_minus_norm": norm(c["pp_triplet_minus"]),
    }


def dominant_spin_character(X: str, v: np.ndarray) -> str:
    s = spin_norm_summary(v)
    X = X.upper()
    if X == "P":
        keys = ["PP_singlet_norm", "PP_triplet0_norm", "PP_triplet_plus_norm", "PP_triplet_minus_norm"]
    else:
        keys = ["PH_charge_norm", "PH_Sz_norm", "norm_ud", "norm_du"]
    best = max(keys, key=lambda k: s[k])
    return best.replace("_norm", "")


# -----------------------------------------------------------------------------
# TU form factor masks and weights
# -----------------------------------------------------------------------------

def get_mi_arrays(O: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    R = np.asarray(getattr(O, "mi_to_R"), dtype=int)
    ofrom = np.asarray(getattr(O, "mi_to_ofrom"), dtype=int)
    oto = np.asarray(getattr(O, "mi_to_oto"), dtype=int)
    return R, ofrom, oto


def component_distances(O: Any) -> np.ndarray:
    R, ofrom, oto = get_mi_arrays(O)
    if km is None:
        # Crude fallback: onsite if exactly same, otherwise unknown long.
        out = np.full(len(ofrom), np.inf)
        out[(R[:, 0] == 0) & (R[:, 1] == 0) & (ofrom == oto)] = 0.0
        return out
    lattice, pos = km.kagome_lattice_and_positions()
    d = np.zeros(len(ofrom), dtype=float)
    for i in range(len(ofrom)):
        dr = km.displacement_vector(lattice, pos, int(R[i, 0]), int(R[i, 1]), int(ofrom[i]), int(oto[i]))
        d[i] = float(np.linalg.norm(dr[:2]))
    return d


def component_masks(O: Any, nn_tol: float = 1e-6) -> Dict[str, np.ndarray]:
    R, ofrom, oto = get_mi_arrays(O)
    d = component_distances(O)
    onsite = (d < nn_tol) & (ofrom == oto)
    nn = np.abs(d - 0.5) < max(nn_tol, 1e-5)
    diagonal = ofrom == oto
    offdiag = ~diagonal
    long = ~(onsite | nn)
    return {
        "onsite": onsite,
        "nn": nn,
        "long": long,
        "diagonal": diagonal,
        "offdiag": offdiag,
    }


def weight_on_mask(f: np.ndarray, mask: np.ndarray) -> float:
    denom = norm2(f)
    if denom <= 0:
        return 0.0
    return float(norm2(f[np.asarray(mask, dtype=bool)]) / denom)


def real_imag_proxy_weights(f: np.ndarray, mask: np.ndarray) -> Tuple[float, float]:
    """Phase-dependent proxy for real-bond vs imaginary-bond content.

    This is NOT a physical current classification.  TUfRG_realspace.py should be
    used for current patterns with real hopping matrix elements.
    """
    mask = np.asarray(mask, dtype=bool)
    denom = float(np.sum(np.abs(f[mask]) ** 2))
    if denom <= 0:
        return 0.0, 0.0
    return float(np.sum(np.real(f[mask]) ** 2) / denom), float(np.sum(np.imag(f[mask]) ** 2) / denom)


def summarize_form_factor(f: np.ndarray, masks: Mapping[str, np.ndarray]) -> Dict[str, float]:
    realw, imagw = real_imag_proxy_weights(f, masks["nn"])
    return {
        "onsite_weight": weight_on_mask(f, masks["onsite"]),
        "nn_weight": weight_on_mask(f, masks["nn"]),
        "long_weight": weight_on_mask(f, masks["long"]),
        "diagonal_weight": weight_on_mask(f, masks["diagonal"]),
        "offdiag_weight": weight_on_mask(f, masks["offdiag"]),
        "real_bond_proxy_weight": realw,
        "imag_bond_proxy_weight": imagw,
    }


def recommended_sector_and_observable(X: str, v: np.ndarray, masks: Mapping[str, np.ndarray]) -> Tuple[str, str, bool, str]:
    """Recommend a user-facing sector name and observable.

    Returned sector names use hyphens so that they can be copied directly into
    TUfRG_realspace.py, e.g. ``ph-charge`` or ``pp-singlet``.  Internally, use
    ``internal_sector_name(sector)`` to index ``spin_combinations``.
    """
    X = X.upper()
    c = spin_combinations(v)
    char = dominant_spin_character(X, v)

    if X == "P":
        if char == "PP_singlet":
            internal = "pp_singlet"
        elif char == "PP_triplet0":
            internal = "pp_triplet0"
        elif char == "PP_triplet_plus":
            internal = "pp_triplet_plus"
        elif char == "PP_triplet_minus":
            internal = "pp_triplet_minus"
        else:
            internal = "pp_singlet"
        sector = display_sector_name(internal)
        obs = "pairing"
        rec = norm(c.get(internal, c["pp_singlet"])) > 0
        reason = "pairing-channel mode"
        return sector, obs, rec, reason

    if char == "PH_charge":
        internal = "ph_charge"
    elif char == "PH_Sz":
        internal = "ph_spin_z"
    else:
        internal = "ph_spin_flip"

    sector = display_sector_name(internal)

    # Conservative interface rule: TUfRG_realspace.py currently supports only
    # longitudinal PH sectors (ph-charge and ph-spin-z).  If the saved-channel
    # eigenvector is dominated by transverse/spin-flip components, report it but
    # do NOT recommend a real-space reconstruction command that the current
    # realspace tool cannot understand.  A later version can add ph-spin-x/y or
    # ph-spin-plus/minus reconstruction explicitly.
    if internal == "ph_spin_flip":
        return (
            sector,
            "unsupported",
            False,
            "spin-flip PH sector detected; TUfRG_realspace.py currently supports only ph-charge and ph-spin-z",
        )

    f = c[internal] if internal in c else c["ph_charge"]
    ff = summarize_form_factor(f, masks)
    if ff["onsite_weight"] > 0.5:
        obs = "density"
        rec = True
        reason = "onsite-dominated PH-like mode"
    elif ff["nn_weight"] > 0.25 and ff["imag_bond_proxy_weight"] > 0.25:
        obs = "current"
        rec = True
        reason = "NN imaginary-bond proxy is sizable; inspect real-space current pattern"
    elif ff["nn_weight"] > 0.25:
        obs = "bond"
        rec = True
        reason = "NN bond-dominated PH-like mode"
    else:
        obs = "density,bond"
        rec = False
        reason = "no simple onsite/NN dominance"
    return sector, obs, rec, reason

# -----------------------------------------------------------------------------
# Optional channel/loop SVD probes
# -----------------------------------------------------------------------------

def flatten_channel_at_q(O: Any, X: str, iq: int) -> np.ndarray:
    K = np.asarray(getattr(O, f"{X.upper()}channel"))
    if K.ndim != 5:
        raise ValueError(f"{X}channel expected ndim=5, got {K.shape}")
    M = K[:, :, :, :, int(iq)]
    return M.reshape(M.shape[0] * M.shape[1], M.shape[2] * M.shape[3])


def flatten_loop_at_q(O: Any, X: str, iq: int) -> np.ndarray:
    if X.upper() == "P":
        L = np.asarray(getattr(O, "pploop"))
    else:
        L = np.asarray(getattr(O, "phloop"))
    M = L[:, :, :, :, int(iq)]
    return M.reshape(M.shape[0] * M.shape[1], M.shape[2] * M.shape[3])


def top_singular_value(A: np.ndarray) -> float:
    try:
        return float(np.linalg.svd(A, compute_uv=False)[0])
    except Exception:
        return float("nan")


# -----------------------------------------------------------------------------
# CSV helpers and analysis
# -----------------------------------------------------------------------------

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


def q_labels_to_indices(data: CaseData, labels: Sequence[str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    nq = int(getattr(data.O, "nk", data.nk * data.nk))
    for lab in labels:
        if lab in data.q_indices:
            out[lab] = int(data.q_indices[lab])
        else:
            try:
                iq = int(lab)
            except ValueError as exc:
                raise ValueError(f"Unknown q label {lab!r}; known labels: {sorted(data.q_indices)}") from exc
            if iq < 0 or iq >= nq:
                raise ValueError(f"q index {iq} out of range [0,{nq})")
            out[f"iq{iq}"] = iq
    return out


def top_components_rows(
    *,
    data: CaseData,
    case_name: str,
    X: str,
    q_label: str,
    iq: int,
    mode: int,
    loc: int,
    vec: np.ndarray,
    top: int,
) -> List[Dict[str, Any]]:
    R, ofrom, oto = get_mi_arrays(data.O)
    masks = component_masks(data.O)
    combos = spin_combinations(vec)
    rows: List[Dict[str, Any]] = []
    for spin_name, f in combos.items():
        if spin_name in {"uu", "ud", "du", "dd"}:
            continue
        order = np.argsort(np.abs(f))[::-1][:int(top)]
        for rank, a in enumerate(order):
            comp_type = "onsite" if masks["onsite"][a] else "NN" if masks["nn"][a] else "long"
            rows.append({
                "case": case_name,
                "channel": X,
                "q_label": q_label,
                "iq": iq,
                "mode": mode,
                "global_loc": loc,
                "component_rank": rank,
                "spin_component": display_sector_name(spin_name),
                "R0": int(R[a, 0]),
                "R1": int(R[a, 1]),
                "R2": int(R[a, 2]) if R.shape[1] > 2 else 0,
                "ofrom": int(ofrom[a]),
                "oto": int(oto[a]),
                "abs_value": float(abs(f[a])),
                "real_value": float(np.real(f[a])),
                "imag_value": float(np.imag(f[a])),
                "component_type": comp_type,
            })
    return rows


def analyze_case(data: CaseData, args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], str]:
    case_name = data.case_dir.name
    qmap = q_labels_to_indices(data, args.q)
    masks = component_masks(data.O)

    q_rows: List[Dict[str, Any]] = []
    mode_rows: List[Dict[str, Any]] = []
    top_rows: List[Dict[str, Any]] = []
    deg_rows: List[Dict[str, Any]] = []
    report_lines: List[str] = []
    report_lines.append(f"Case: {case_name}")
    report_lines.append(f"Directory: {data.case_dir}")
    report_lines.append("Interpretation warning: P/C/D vecs are saved channel decompositions, not automatically loop-dressed instability eigenvectors.")
    report_lines.append("")

    for X in args.channels:
        X = X.upper()
        vals, vecs, off, length = channel_arrays(data.O, X)
        for q_label, iq in qmap.items():
            n_avail = int(length[iq])
            n_modes = min(int(args.n_modes), n_avail)
            loc0 = int(off[iq])
            block_vals = np.asarray(vals[loc0: loc0 + n_avail], dtype=float)
            top_abs = float(np.max(np.abs(block_vals[:n_modes]))) if n_modes else float("nan")
            top_raw = float(block_vals[0]) if n_modes else float("nan")

            qrow: Dict[str, Any] = {
                "case": case_name,
                "channel": X,
                "q_label": q_label,
                "iq": int(iq),
                "q_reduced": str(q_reduced_from_iq(iq, data.nk)),
                "n_modes_available": n_avail,
                "top_raw_value": top_raw,
                "top_abs_raw_value_in_shown_modes": top_abs,
            }
            if args.include_loop_svd:
                try:
                    K = flatten_channel_at_q(data.O, X, iq)
                    L = flatten_loop_at_q(data.O, X, iq)
                    qrow["raw_channel_svd_top"] = top_singular_value(K)
                    qrow["loop_svd_KL_top"] = top_singular_value(K @ L)
                    qrow["loop_svd_LK_top"] = top_singular_value(L @ K)
                except Exception as exc:
                    qrow["loop_svd_error"] = f"{type(exc).__name__}: {exc}"
            q_rows.append(qrow)

            shown_vals = block_vals[:n_modes]
            # Degeneracy groups: consecutive modes close in absolute raw value.
            group_id = 0
            start = 0
            while start < n_modes:
                base = abs(shown_vals[start])
                end = start + 1
                while end < n_modes:
                    denom = max(base, abs(shown_vals[end]), 1e-12)
                    if abs(abs(shown_vals[end]) - base) / denom <= float(args.degeneracy_tol):
                        end += 1
                    else:
                        break
                if end - start >= 2:
                    subvals = shown_vals[start:end]
                    spread = (float(np.max(np.abs(subvals))) - float(np.min(np.abs(subvals)))) / max(float(np.max(np.abs(subvals))), 1e-12)
                    deg_rows.append({
                        "case": case_name,
                        "channel": X,
                        "q_label": q_label,
                        "iq": int(iq),
                        "group_id": f"{X}_{q_label}_{group_id}",
                        "mode_start": start,
                        "mode_end_inclusive": end - 1,
                        "n_modes": end - start,
                        "values": " ".join(f"{x:.8e}" for x in subvals),
                        "relative_spread_abs_values": spread,
                        "comment": "near-degenerate within same saved channel and same q sector",
                    })
                    group_id += 1
                start = end

            report_lines.append(f"[{X} / {q_label} / iq={iq}] available={n_avail}, showing={n_modes}")
            for mode in range(n_modes):
                raw, v, loc = get_mode(data.O, X, iq, mode)
                ssum = spin_norm_summary(v)
                char = dominant_spin_character(X, v)
                combos = spin_combinations(v)
                sector, obs, rec, reason = recommended_sector_and_observable(X, v, masks)
                f_for_weights = combos.get(internal_sector_name(sector), combos["ph_charge"])
                fsum = summarize_form_factor(f_for_weights, masks)

                row: Dict[str, Any] = {
                    "case": case_name,
                    "channel": X,
                    "q_label": q_label,
                    "iq": int(iq),
                    "q_reduced": str(q_reduced_from_iq(iq, data.nk)),
                    "mode": int(mode),
                    "global_loc": int(loc),
                    "raw_value": float(raw),
                    "abs_raw_value": float(abs(raw)),
                    "dominant_spin_character": char,
                    "recommended_sector": sector,
                    "recommended_observable": obs,
                    "recommend_realspace": bool(rec),
                    "recommendation_reason": reason,
                }
                row.update(ssum)
                row.update(fsum)
                mode_rows.append(row)
                top_rows.extend(top_components_rows(
                    data=data, case_name=case_name, X=X, q_label=q_label, iq=iq,
                    mode=mode, loc=loc, vec=v, top=int(args.top_components)
                ))
                report_lines.append(
                    f"  mode {mode:2d}: val={raw:+.6e}, char={char}, sector={sector}, "
                    f"onsite={fsum['onsite_weight']:.3f}, NN={fsum['nn_weight']:.3f}, "
                    f"imag_proxy={fsum['imag_bond_proxy_weight']:.3f}, realspace={rec}"
                )
            report_lines.append("")
    return q_rows, mode_rows, top_rows, deg_rows, "\n".join(report_lines)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="General q-resolved diagnosis for spinful kagome TUFRG outputs.")
    p.add_argument("path", help="case directory or output root containing case directories")
    p.add_argument("--recursive", action="store_true", help="recursively find all directories containing O.dvg")
    p.add_argument("--q", nargs="+", default=list(DEFAULT_Q_LABELS), help="q labels or numeric q indices")
    p.add_argument("--channels", nargs="+", default=list(CHANNELS), choices=list(CHANNELS))
    p.add_argument("--n-modes", type=int, default=8, help="number of saved modes per fixed (channel,q) to inspect")
    p.add_argument("--top-components", type=int, default=12)
    p.add_argument("--degeneracy-tol", type=float, default=1e-3)
    p.add_argument("--include-loop-svd", action="store_true", help="compute costly SVD probes of K, KL, LK")
    p.add_argument("--out", default="diagnosis", help="output subdirectory name inside each case")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    case_dirs = find_case_dirs(Path(args.path), recursive=bool(args.recursive))
    all_mode_rows: List[Dict[str, Any]] = []
    all_q_rows: List[Dict[str, Any]] = []
    all_deg_rows: List[Dict[str, Any]] = []

    for case_dir in case_dirs:
        print(f"Analyzing case: {case_dir}")
        data = load_case(case_dir)
        q_rows, mode_rows, top_rows, deg_rows, report = analyze_case(data, args)
        outdir = case_dir / str(args.out)
        outdir.mkdir(parents=True, exist_ok=True)
        write_csv(outdir / "q_summary.csv", q_rows)
        write_csv(outdir / "mode_summary.csv", mode_rows)
        write_csv(outdir / "top_components.csv", top_rows)
        write_csv(outdir / "degeneracy_summary.csv", deg_rows)
        (outdir / "diagnosis_report.txt").write_text(report)
        print(report)
        all_mode_rows.extend(mode_rows)
        all_q_rows.extend(q_rows)
        all_deg_rows.extend(deg_rows)

    if len(case_dirs) > 1:
        root = Path(args.path).resolve()
        outdir = root / str(args.out)
        outdir.mkdir(parents=True, exist_ok=True)
        write_csv(outdir / "all_q_summary.csv", all_q_rows)
        write_csv(outdir / "all_mode_summary.csv", all_mode_rows)
        write_csv(outdir / "all_degeneracy_summary.csv", all_deg_rows)
        print("Wrote aggregate diagnosis to", outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
