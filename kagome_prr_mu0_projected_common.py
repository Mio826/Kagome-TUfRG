#!/usr/bin/env python3
# coding: utf-8
"""
Shared helpers for small-scale PRR-style kagome TUFRG tests with projected q-resolved channels.

Conventions used here:
  - Kagome has 3 sublattices/orbitals per unit cell.
  - divERGe TU model is SU2-reduced: SU2=True, n_spin=1.
  - PRR upper/p-type VHS target for the NN kagome model is enforced by
    mu=0 and hopping amplitude -t, with t>0. With this convention the
    occupied fraction below mu=0 is approximately 5/12.
  - Momentum coordinates are reduced reciprocal-lattice coordinates.
  - For projected output with cfg.tu_channel_calc_project=-1, Dchannel is already
    physical charge (2D-C), and Cchannel is sign-flipped magnetic.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


# -----------------------------------------------------------------------------
# Kagome geometry and band checks
# -----------------------------------------------------------------------------


def norm(x: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(x, dtype=float)))


def kagome_lattice_and_positions() -> Tuple[np.ndarray, np.ndarray]:
    """Return lattice vectors and orbital positions used by divERGe kagome test."""
    lattice = np.eye(3, dtype=float)
    lattice[0, 0] = 0.5 * np.sqrt(3.0)
    lattice[0, 1] = -0.5
    lattice[1, 0] = 0.5 * np.sqrt(3.0)
    lattice[1, 1] = 0.5
    lattice[2, 2] = 1.0

    pos = np.zeros((3, 3), dtype=float)
    pos[0, 0] = 0.5 * np.sqrt(3.0)
    pos[0, 1] = 0.0
    pos[1, 0] = 0.5 * np.sqrt(3.0) - 0.5 * lattice[0, 0]
    pos[1, 1] = -0.5 * lattice[0, 1]
    pos[2, 0] = 0.5 * np.sqrt(3.0) + 0.5 * lattice[1, 0]
    pos[2, 1] = 0.5 * lattice[1, 1]
    return lattice, pos


def displacement_vector(
    lattice: np.ndarray,
    pos: np.ndarray,
    Rx: int,
    Ry: int,
    o1: int,
    o2: int,
) -> np.ndarray:
    """Vector from orbital o1 in home cell to orbital o2 in cell R."""
    return pos[o2] - pos[o1] + Rx * lattice[0] + Ry * lattice[1]


def kagome_nn_hopping_entries(*, hop_amp: float = -1.0) -> List[Tuple[List[int], int, int, int, int, Tuple[float, float]]]:
    """Nearest-neighbor hopping entries for divERGe rs_hopping_t."""
    lattice, pos = kagome_lattice_and_positions()
    out = []
    for Rx in range(-3, 4):
        for Ry in range(-3, 4):
            for o1 in range(3):
                for o2 in range(3):
                    bond = displacement_vector(lattice, pos, Rx, Ry, o1, o2)
                    if abs(norm(bond) - 0.5) < 1e-7:
                        out.append(([Rx, Ry, 0], o1, o2, 0, 0, (float(hop_amp), 0.0)))
    return out


def kagome_uv_vertex_entries(
    *,
    U: float,
    V: float,
) -> List[Tuple[str, List[int], int, int, int, int, int, int, Tuple[float, float]]]:
    """Onsite U and NN V density-density vertices for divERGe rs_vertex_t.

    Uses chan='D' and s1=-1, matching the official SU2 density convention.
    """
    lattice, pos = kagome_lattice_and_positions()
    out = []
    for Rx in range(-2, 3):
        for Ry in range(-2, 3):
            for o1 in range(3):
                for o2 in range(3):
                    bond = displacement_vector(lattice, pos, Rx, Ry, o1, o2)
                    dist = norm(bond)
                    if dist < 1e-7 and abs(U) > 0:
                        out.append(("D", [Rx, Ry, 0], o1, o2, -1, 0, 0, 0, (float(U), 0.0)))
                    elif abs(dist - 0.5) < 1e-7 and abs(V) > 0:
                        out.append(("D", [Rx, Ry, 0], o1, o2, -1, 0, 0, 0, (float(V), 0.0)))
    return out


def bloch_hamiltonian_from_entries(
    k_red: Sequence[float],
    hopping_entries: Sequence[Tuple[List[int], int, int, int, int, Tuple[float, float]]],
    *,
    mu: float = 0.0,
    norb: int = 3,
) -> np.ndarray:
    """Build H(k) in reduced reciprocal coordinates using divERGe R phases.

    This is intended as a sanity check for the NN kagome VHS only. It uses
    phase exp(2*pi*i*(k1*R1+k2*R2)); that matches the standard tight-binding
    convention for the stored cell displacement R.
    """
    k = np.asarray(k_red, dtype=float)
    H = np.zeros((norb, norb), dtype=complex)
    for R, o1, o2, _s1, _s2, amp in hopping_entries:
        val = complex(amp[0], amp[1])
        phase = np.exp(2j * np.pi * (k[0] * R[0] + k[1] * R[1] + k[2] * R[2]))
        H[int(o1), int(o2)] += val * phase
    if abs(mu) > 0:
        H -= float(mu) * np.eye(norb, dtype=complex)
    return H


def band_energies(
    k_red: Sequence[float],
    *,
    hop_amp: float = -1.0,
    mu: float = 0.0,
) -> np.ndarray:
    entries = kagome_nn_hopping_entries(hop_amp=hop_amp)
    return np.linalg.eigvalsh(bloch_hamiltonian_from_entries(k_red, entries, mu=mu))


def estimate_filling_fraction(
    *,
    hop_amp: float = -1.0,
    mu: float = 0.0,
    ngrid: int = 240,
    half_weight_equal: bool = True,
) -> float:
    """Estimate occupied single-particle fraction over all 3 bands.

    The result is a fraction in [0,1] over the spinless/SU2-reduced band count.
    For a spin-degenerate system, the same fraction applies per spin.
    """
    entries = kagome_nn_hopping_entries(hop_amp=hop_amp)
    n_less = 0.0
    ntot = float(ngrid * ngrid * 3)
    tol = 1e-10
    for ix in range(ngrid):
        kx = ix / ngrid
        for iy in range(ngrid):
            ky = iy / ngrid
            ev = np.linalg.eigvalsh(bloch_hamiltonian_from_entries([kx, ky, 0.0], entries, mu=mu))
            n_less += float(np.count_nonzero(ev < -tol))
            if half_weight_equal:
                n_less += 0.5 * float(np.count_nonzero(np.abs(ev) <= tol))
    return n_less / ntot


def vhs_sanity_report(*, hop_amp: float = -1.0, mu: float = 0.0, ngrid: int = 120) -> Dict[str, object]:
    """Return a small noninteracting check for mu=0 upper/p-type VHS."""
    hs = {
        "Gamma": np.array([0.0, 0.0, 0.0]),
        "M1": np.array([0.5, 0.0, 0.0]),
        "M2": np.array([0.0, 0.5, 0.0]),
        "M3": np.array([0.5, 0.5, 0.0]),
        "K": np.array([1.0 / 3.0, 1.0 / 3.0, 0.0]),
    }
    energies = {name: band_energies(k, hop_amp=hop_amp, mu=mu).tolist() for name, k in hs.items()}
    filling = estimate_filling_fraction(hop_amp=hop_amp, mu=mu, ngrid=ngrid)
    return {
        "hop_amp": float(hop_amp),
        "mu": float(mu),
        "target_filling": 5.0 / 12.0,
        "estimated_filling": float(filling),
        "energies": energies,
        "max_abs_E_M_middle": float(max(abs(energies[m][1]) for m in ["M1", "M2", "M3"])),
    }


# -----------------------------------------------------------------------------
# divERGe output diagnosis helpers
# -----------------------------------------------------------------------------


def qmesh_from_output(O, T) -> np.ndarray:
    """Reconstruct q mesh from O.idx_ibz_in_bz and T.nk.

    For the present no-symmetry TU run, idx_ibz_in_bz is simply the coarse
    bosonic mesh index. Coordinates are reduced reciprocal coordinates.
    """
    idx = np.asarray(O.idx_ibz_in_bz, dtype=int)
    nk = np.asarray(T.nk, dtype=int)
    if idx.size == 0:
        # Fallback for output without idx map.
        idx = np.arange(int(nk[0] * nk[1]), dtype=int)
    N0 = int(nk[0])
    N1 = int(nk[1])
    qmesh = np.zeros((idx.size, 3), dtype=float)
    for a, ind in enumerate(idx):
        ix = int(ind) % N0
        iy = int(ind) // N0
        qmesh[a, :] = [ix / N0, iy / N1, 0.0]
    return qmesh


def high_symmetry_q_indices(qmesh: np.ndarray) -> Dict[str, int]:
    targets = {
        "Gamma": np.array([0.0, 0.0, 0.0]),
        "M1": np.array([0.5, 0.0, 0.0]),
        "M2": np.array([0.0, 0.5, 0.0]),
        "M3": np.array([0.5, 0.5, 0.0]),
    }
    out = {}
    for name, target in targets.items():
        d = qmesh[:, :2] - target[:2]
        d = d - np.round(d)
        dist = np.linalg.norm(d, axis=1)
        out[name] = int(np.argmin(dist))
    return out


def channel_matrix(O, X: str, iq: int) -> np.ndarray:
    arr = np.asarray(getattr(O, f"{X}channel"))
    return np.asarray(arr[0, :, 0, :, int(iq)], dtype=complex)


def loop_matrix(O, kind: str, iq: int) -> np.ndarray:
    if kind not in {"ph", "pp"}:
        raise ValueError("kind must be 'ph' or 'pp'")
    arr = np.asarray(getattr(O, f"{kind}loop"))
    return np.asarray(arr[0, :, 0, :, int(iq)], dtype=complex)


def physical_kernel(O, channel: str, iq: int) -> Tuple[np.ndarray, np.ndarray, str]:
    """Return K, L, label for projected physical channel at q index iq.

    This helper assumes the O.dvg file was generated with
        cfg.tu_channel_calc_project = -1
    in divERGe postprocessing.  With SU(2), the saved q-resolved channels obey:
        Pchannel: projected pairing channel P
        Cchannel: sign-flipped projected magnetic channel, i.e. roughly -C
        Dchannel: projected physical charge channel, i.e. 2D - C

    Therefore we do NOT construct 2*D-C here again.
    """
    channel = channel.lower()
    if channel == "mag":
        return -channel_matrix(O, "C", iq), loop_matrix(O, "ph", iq), "mag=-C_saved(projected)"
    if channel == "charge":
        return channel_matrix(O, "D", iq), loop_matrix(O, "ph", iq), "charge=D_saved=(2D-C)_projected"
    if channel == "sc":
        return channel_matrix(O, "P", iq), loop_matrix(O, "pp", iq), "sc=P_saved(projected)"
    raise ValueError("channel must be mag, charge, or sc")

def hermitian_eigs(M: np.ndarray) -> np.ndarray:
    H = 0.5 * (M + M.conjugate().T)
    return np.linalg.eigvalsh(H)


def top_svd(M: np.ndarray, n: int = 8) -> np.ndarray:
    return np.linalg.svd(np.asarray(M, dtype=complex), compute_uv=False)[:n]


def uniform_projected_svd(K: np.ndarray, L: np.ndarray, *, n: int = 8) -> np.ndarray:
    """Project uniform form-factor direction from q=0 charge before SVD.

    This is not divERGe's built-in convention. It is the optional analogue of
    the old band-basis diagnosis that removes the forward-scattering direction.
    """
    dim = K.shape[0]
    u = np.ones((dim, 1), dtype=complex) / np.sqrt(float(dim))
    P = np.eye(dim, dtype=complex) - u @ u.conjugate().T
    A = P @ K @ L @ P
    return top_svd(A, n=n)


def summarize_q_channel(O, iq: int, qcoord: Sequence[float], channel: str, *, n: int = 8) -> Dict[str, object]:
    K, L, label = physical_kernel(O, channel, iq)
    raw_e = hermitian_eigs(K)
    A_kl = K @ L
    A_lk = L @ K
    row = {
        "q_label": None,
        "iq": int(iq),
        "qx": float(qcoord[0]),
        "qy": float(qcoord[1]),
        "channel": channel,
        "kernel_label": label,
        "raw_herm_max": float(raw_e[-1]),
        "raw_herm_min": float(raw_e[0]),
        "raw_svd_0": float(top_svd(K, n=1)[0]),
        "loop_svd_KL_0": float(top_svd(A_kl, n=1)[0]),
        "loop_svd_LK_0": float(top_svd(A_lk, n=1)[0]),
    }
    # Store a few top values as compact strings.
    row["loop_svd_KL_top"] = " ".join(f"{x:.8e}" for x in top_svd(A_kl, n=n))
    row["loop_svd_LK_top"] = " ".join(f"{x:.8e}" for x in top_svd(A_lk, n=n))
    if channel == "charge" and np.allclose(qcoord[:2], 0.0, atol=1e-12):
        row["charge_q0_uniform_projected_svd_top"] = " ".join(
            f"{x:.8e}" for x in uniform_projected_svd(K, L, n=n)
        )
    else:
        row["charge_q0_uniform_projected_svd_top"] = ""
    return row


def write_csv(path: os.PathLike, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        return
    keys: List[str] = []
    for r in rows:
        for k in r.keys():
            if k not in keys:
                keys.append(k)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow(dict(r))
