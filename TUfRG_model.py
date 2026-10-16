#!/usr/bin/env python3
# coding: utf-8
"""
TUfRG_model.py

Official model-construction helpers for spinful kagome extended Hubbard models
with NN/NNN Kane-Mele and NN Rashba spin-orbit coupling and U/V1/V2/V3 density interactions, intended for divERGe TUFRG runs.

Scope
-----
This module builds and validates the MODEL only:
    * kagome lattice geometry and orbital positions;
    * spinful real-space hopping list for NN hopping + NN/NNN Kane-Mele SOC + NN Rashba SOC;
    * official divERGe D-channel density-density interaction vertices for onsite U, NN V1, second-neighbor V2, and same-sublattice cross-hexagon V3;
    * fixed-chemical-potential OR fixed-filling chemical-potential resolution;
    * pure-numpy one-body sanity checks;
    * divERGe model initialization and TU form-factor basis generation.

It does NOT run the RG flow and does NOT diagnose instabilities. Those belong in
separate runner/inspector scripts.

Gauge note
----------
No Bloch-eigenvector gauge fixing is required in this model builder. The TUFRG
model is specified in a real-space/orbital basis through hopping and interaction
entries. The gauge-sensitive active-band projection used in scalar N-patch code
is not part of this model construction. What must be fixed here is the real-space
bond orientation/sign convention for Kane-Mele SOC. Eigenvector/global phase
issues appear later in diagnosis of form-factor/current patterns, not here.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, asdict, replace
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

A, B, C = 0, 1, 2
SPIN_UP, SPIN_DN = 0, 1
N_ORB = 3
N_SPIN = 2

HopEntry = Tuple[List[int], int, int, int, int, Tuple[float, float]]
# (chan, R, o1, o2, s1, s2, s3, s4, V). For official divERGe
# density vertices we use chan="D" and s1=-1; s2/s3/s4 are ignored by
# divERGe's auto-expansion but kept as zeros for dtype compatibility.
VertexEntry = Tuple[str, List[int], int, int, int, int, int, int, Tuple[float, float]]

PAIR_LABEL = {
    (A, B): "AB", (B, A): "AB",
    (A, C): "AC", (C, A): "AC",
    (B, C): "BC", (C, B): "BC",
}
PAIR_ETA_DEFAULT: Dict[str, float] = {"AB": +1.0, "AC": -1.0, "BC": +1.0}
CANONICAL_DIR = {(A, B), (A, C), (B, C)}


@dataclass(frozen=True)
class KMParameters:
    """Physical parameters for the one-body and density interaction model.

    filling_mode:
        "fixed_mu"      -> use `mu` directly.
        "fixed_filling" -> compute chemical potential so the zero-temperature
                            occupied fraction equals `target_filling`, then use
                            that computed mu in the divERGe hopping list.

    target_filling is a fraction of the full spinful six-band Hilbert space.
    For the PRR upper/higher-VH target, use target_filling=5/12.
    """

    t: float = 1.0
    mu: float = 0.0
    target_filling: float = 5.0 / 12.0
    filling_mode: str = "fixed_mu"  # "fixed_mu" or "fixed_filling"
    lambda_nn: float = 0.0
    lambda_nnn: float = 0.0
    lambda_rashba: float = 0.0
    U: float = 0.0
    V1: float = 0.0
    V2: float = 0.0
    V3: float = 0.0
    filling_ngrid: int = 120

    def validate(self) -> None:
        if self.filling_mode not in {"fixed_mu", "fixed_filling"}:
            raise ValueError("filling_mode must be 'fixed_mu' or 'fixed_filling'")
        if not (0.0 <= float(self.target_filling) <= 1.0):
            raise ValueError("target_filling must be a fraction in [0,1]")
        if int(self.filling_ngrid) <= 0:
            raise ValueError("filling_ngrid must be positive")


@dataclass(frozen=True)
class MeshParameters:
    """divERGe momentum/form-factor discretization parameters."""

    nk: int = 12
    nkf: int = 3
    tu_distance: float = 1.01

    def validate(self) -> None:
        if int(self.nk) <= 0:
            raise ValueError("nk must be positive")
        if int(self.nkf) <= 0:
            raise ValueError("nkf must be positive")
        if float(self.tu_distance) <= 0:
            raise ValueError("tu_distance must be positive")


# -----------------------------------------------------------------------------
# Geometry
# -----------------------------------------------------------------------------

def kagome_lattice_and_positions() -> Tuple[np.ndarray, np.ndarray]:
    """Return real-space lattice vectors and kagome orbital positions.

    The convention matches the PRR/divERGe benchmark geometry used earlier.
    Lattice vectors are stored as rows: lattice[0] = a1, lattice[1] = a2.
    """
    lattice = np.eye(3, dtype=float)
    lattice[0, 0] = 0.5 * np.sqrt(3.0)
    lattice[0, 1] = -0.5
    lattice[1, 0] = 0.5 * np.sqrt(3.0)
    lattice[1, 1] = 0.5
    lattice[2, 2] = 1.0

    pos = np.zeros((3, 3), dtype=float)
    pos[A, 0] = 0.5 * np.sqrt(3.0)
    pos[A, 1] = 0.0
    pos[B, 0] = 0.5 * np.sqrt(3.0) - 0.5 * lattice[0, 0]
    pos[B, 1] = -0.5 * lattice[0, 1]
    pos[C, 0] = 0.5 * np.sqrt(3.0) + 0.5 * lattice[1, 0]
    pos[C, 1] = 0.5 * lattice[1, 1]
    return lattice, pos


def displacement_vector(lattice: np.ndarray, pos: np.ndarray, Rx: int, Ry: int, o_from: int, o_to: int) -> np.ndarray:
    """Vector from orbital `o_from` in home cell to `o_to` in cell (Rx,Ry).

    This is a generic model-construction geometry helper.  Do not use it for
    interpreting divERGe TU form-factor components read from O.dvg; for those
    use :func:`tu_ff_displacement_vector` below.
    """
    return pos[o_to] - pos[o_from] + Rx * lattice[0] + Ry * lattice[1]


def tu_ff_displacement_vector(lattice: np.ndarray, pos: np.ndarray, Rx: int, Ry: int, o_from: int, o_to: int) -> np.ndarray:
    """Physical displacement of a divERGe TU form-factor component.

    divERGe stores each TU component as (R, ofrom, oto), but the C source
    constructs the real-space bond with ``ofrom`` in cell R and ``oto`` in the
    home cell.  The displacement from the annihilation/from end to the
    creation/to end is therefore

        tau_oto - (tau_ofrom + R) = tau_oto - tau_ofrom - R.

    This convention was verified from ``src/tu/diverge_interface.c`` where
    ``from = position[ofrom] + R`` and ``to = position[oto]`` are used to build
    ``real_ff``, and from reduced-loop maps where form-factor orbital pairs
    enter as (oto, ofrom).

    Use this helper only for post-processing arrays ``mi_to_R``,
    ``mi_to_ofrom``, and ``mi_to_oto`` loaded from divERGe output.
    """
    return pos[o_to] - pos[o_from] - Rx * lattice[0] - Ry * lattice[1]


def canonical_direction_sign(o_from: int, o_to: int) -> float:
    """Return +1 for A->B, A->C, B->C and -1 for reversed directions."""
    if (o_from, o_to) in CANONICAL_DIR:
        return +1.0
    if (o_to, o_from) in CANONICAL_DIR:
        return -1.0
    raise ValueError(f"not an off-diagonal kagome pair: {(o_from, o_to)}")


def enumerate_offdiag_bonds_at_distance(distance: float, *, search_range: int = 3, tol: float = 1e-7) -> List[Tuple[List[int], int, int, np.ndarray]]:
    """Enumerate off-diagonal kagome bonds with a target real-space distance."""
    lattice, pos = kagome_lattice_and_positions()
    bonds: List[Tuple[List[int], int, int, np.ndarray]] = []
    for Rx in range(-search_range, search_range + 1):
        for Ry in range(-search_range, search_range + 1):
            for o_from in range(N_ORB):
                for o_to in range(N_ORB):
                    if o_from == o_to:
                        continue
                    dr = displacement_vector(lattice, pos, Rx, Ry, o_from, o_to)
                    if abs(float(np.linalg.norm(dr[:2])) - float(distance)) <= tol:
                        bonds.append(([Rx, Ry, 0], o_from, o_to, dr.copy()))
    bonds.sort(key=lambda x: (x[0][0], x[0][1], x[1], x[2]))
    return bonds


def high_symmetry_q_indices(nk: int) -> Dict[str, int]:
    """Return Gamma/M indices for divERGe q order q_index=ix*nk+iy.

    M points require even nk. Indices are based on the coarse bosonic q mesh,
    not the fine nk*nkf loop mesh.
    """
    nk = int(nk)
    if nk <= 0:
        raise ValueError("nk must be positive")
    if nk % 2 != 0:
        raise ValueError("M-point q indices require even nk")
    return {
        "Gamma": 0,
        "M1": (nk // 2) * nk,
        "M2": nk // 2,
        "M3": (nk // 2) * nk + nk // 2,
    }


def qmesh_reduced(nk: int) -> np.ndarray:
    """Return reduced q coordinates in divERGe order ix*nk + iy."""
    nk = int(nk)
    out = np.zeros((nk * nk, 3), dtype=float)
    for ix in range(nk):
        for iy in range(nk):
            out[ix * nk + iy] = [ix / nk, iy / nk, 0.0]
    return out


# -----------------------------------------------------------------------------
# Hopping and interaction entries
# -----------------------------------------------------------------------------

def spinful_hoppings(
    *,
    t: float = 1.0,
    mu: float = 0.0,
    lambda_nn: float = 0.0,
    lambda_nnn: float = 0.0,
    lambda_rashba: float = 0.0,
    include_mu: bool = True,
    pair_eta: Optional[Mapping[str, float]] = None,
    search_range: int = 3,
) -> List[HopEntry]:
    """Build divERGe-compatible spinful hopping entries.

    If include_mu is True, includes the onsite term -mu n. For solving the
    chemical potential, use include_mu=False and treat mu as the Fermi level.
    """
    if pair_eta is None:
        pair_eta = PAIR_ETA_DEFAULT
    missing = set(PAIR_ETA_DEFAULT) - set(pair_eta)
    if missing:
        raise ValueError(f"pair_eta missing keys: {sorted(missing)}")

    out: List[HopEntry] = []

    # Real NN hopping, spin independent.
    for R, o_from, o_to, _dr in enumerate_offdiag_bonds_at_distance(0.5, search_range=search_range):
        for spin in (SPIN_UP, SPIN_DN):
            out.append((R, o_from, o_to, spin, spin, (-float(t), 0.0)))

    # Onsite chemical potential, -mu n. In divERGe the Fermi level is then zero.
    if include_mu and abs(float(mu)) > 0:
        for o in range(N_ORB):
            for spin in (SPIN_UP, SPIN_DN):
                out.append(([0, 0, 0], o, o, spin, spin, (-float(mu), 0.0)))

    def add_km_terms(distance: float, lam: float) -> None:
        if abs(float(lam)) == 0:
            return
        for R, o_from, o_to, _dr in enumerate_offdiag_bonds_at_distance(distance, search_range=search_range):
            pair = PAIR_LABEL[(o_from, o_to)]
            eta = float(pair_eta[pair])
            orient = canonical_direction_sign(o_from, o_to)
            for spin in (SPIN_UP, SPIN_DN):
                sz = +1.0 if spin == SPIN_UP else -1.0
                amp = 1j * sz * float(lam) * eta * orient
                out.append((R, o_from, o_to, spin, spin, (float(np.real(amp)), float(np.imag(amp)))))

    add_km_terms(0.5, lambda_nn)
    add_km_terms(np.sqrt(3.0) / 2.0, lambda_nnn)

    # Nearest-neighbor Rashba: H_ij^R = i*lambda_rashba *
    # [sigma_x*dhat_y - sigma_y*dhat_x], with dhat pointing FROM
    # (home cell, o_from) TO (R, o_to), and |dhat|=1.
    # Only spin-flip entries occur. The reversed bond automatically has the
    # Hermitian conjugate amplitude; time reversal is preserved while Sz
    # conservation is broken. This sign convention is explicitly chosen here;
    # the old TUfRG_model_rashba.py was NOT among the supplied source files.
    if abs(float(lambda_rashba)) > 0:
        for R, o_from, o_to, dr in enumerate_offdiag_bonds_at_distance(0.5, search_range=search_range):
            d = np.asarray(dr[:2], dtype=float)
            d /= np.linalg.norm(d)
            dx, dy = float(d[0]), float(d[1])
            # sigma_x*dy - sigma_y*dx: (up,down)=dy+i*dx,
            #                              (down,up)=dy-i*dx.
            amp_ud = 1j * float(lambda_rashba) * (dy + 1j*dx)
            amp_du = 1j * float(lambda_rashba) * (dy - 1j*dx)
            out.append((R, o_from, o_to, SPIN_UP, SPIN_DN,
                        (float(amp_ud.real), float(amp_ud.imag))))
            out.append((R, o_from, o_to, SPIN_DN, SPIN_UP,
                        (float(amp_du.real), float(amp_du.imag))))
    return out


def density_vertices_manual(*, U: float = 0.0, V1: float = 0.0, V2: float = 0.0, V3: float = 0.0, search_range: int = 2) -> List[VertexEntry]:
    r"""Legacy explicit-spin density-density interaction entries.

    This is the old/manual convention:
        (s1,s2,s3,s4) = (s_left,s_right,s_left,s_right).

    It is kept only for bridge tests against the new official-style builder.
    divERGe's original code/documentation treats ``rs_vertex_t`` spin/orbital
    slots as channel-specific.  For density interactions in spinful non-SU2
    models, the official examples use D-channel vertices with ``s1=-1``.
    """
    lattice, pos = kagome_lattice_and_positions()
    out: List[VertexEntry] = []

    for Rx in range(-search_range, search_range + 1):
        for Ry in range(-search_range, search_range + 1):
            for o_left in range(N_ORB):
                for o_right in range(N_ORB):
                    dr = displacement_vector(lattice, pos, Rx, Ry, o_left, o_right)
                    dist = float(np.linalg.norm(dr[:2]))

                    if dist < 1e-7 and abs(float(U)) > 0:
                        for s_left, s_right in ((SPIN_UP, SPIN_DN), (SPIN_DN, SPIN_UP)):
                            out.append(("D", [Rx, Ry, 0], o_left, o_right,
                                        s_left, s_right, s_left, s_right, (float(U), 0.0)))
                    elif abs(dist - 0.5) < 1e-7 and abs(float(V1)) > 0:
                        for s_left in (SPIN_UP, SPIN_DN):
                            for s_right in (SPIN_UP, SPIN_DN):
                                out.append(("D", [Rx, Ry, 0], o_left, o_right,
                                            s_left, s_right, s_left, s_right, (float(V1), 0.0)))
                    elif abs(dist - np.sqrt(3.0) / 2.0) < 1e-7 and abs(float(V2)) > 0:
                        for s_left in (SPIN_UP, SPIN_DN):
                            for s_right in (SPIN_UP, SPIN_DN):
                                out.append(("D", [Rx, Ry, 0], o_left, o_right,
                                            s_left, s_right, s_left, s_right, (float(V2), 0.0)))
                    elif o_left == o_right and abs(dist - 1.0) < 1e-7 and abs(float(V3)) > 0:
                        # V3: same-sublattice cross-hexagon / third-neighbor density interaction.
                        # In this geometry NN distance is 0.5, so the cross-hexagon same-sublattice
                        # separation is 1.0.  The o_left == o_right filter is essential: distance 1.0
                        # alone would also include unwanted inter-sublattice bonds in other geometries.
                        for s_left in (SPIN_UP, SPIN_DN):
                            for s_right in (SPIN_UP, SPIN_DN):
                                out.append(("D", [Rx, Ry, 0], o_left, o_right,
                                            s_left, s_right, s_left, s_right, (float(V3), 0.0)))
    return out


def density_vertices_official_s1minus1(*, U: float = 0.0, V1: float = 0.0, V2: float = 0.0, V3: float = 0.0, search_range: int = 2) -> List[VertexEntry]:
    r"""Official-style divERGe density-density interaction entries.

    Original-code convention:
      * use channel ``'D'`` for density-density interactions;
      * set ``s1=-1`` so divERGe auto-generates the SU(2)-related spin
        components using its channel-specific crossing relations.

    This is the style used in divERGe's own spinful Kane-Mele/Rashba/Hubbard
    examples.  For D-channel non-SU2 vertices, the C code expands ``s1=-1`` as

        (s1,s2,s3,s4) = (si1, si1, si2, si2),

    which is not the same as the old manual tuple

        (s_left,s_right,s_left,s_right).

    The remaining spin slots are set to zero only for dtype compatibility.
    """
    lattice, pos = kagome_lattice_and_positions()
    out: List[VertexEntry] = []

    for Rx in range(-search_range, search_range + 1):
        for Ry in range(-search_range, search_range + 1):
            for o_left in range(N_ORB):
                for o_right in range(N_ORB):
                    dr = displacement_vector(lattice, pos, Rx, Ry, o_left, o_right)
                    dist = float(np.linalg.norm(dr[:2]))

                    if dist < 1e-7 and abs(float(U)) > 0:
                        out.append(("D", [Rx, Ry, 0], o_left, o_right,
                                    -1, 0, 0, 0, (float(U), 0.0)))
                    elif abs(dist - 0.5) < 1e-7 and abs(float(V1)) > 0:
                        out.append(("D", [Rx, Ry, 0], o_left, o_right,
                                    -1, 0, 0, 0, (float(V1), 0.0)))
                    elif abs(dist - np.sqrt(3.0) / 2.0) < 1e-7 and abs(float(V2)) > 0:
                        out.append(("D", [Rx, Ry, 0], o_left, o_right,
                                    -1, 0, 0, 0, (float(V2), 0.0)))
                    elif o_left == o_right and abs(dist - 1.0) < 1e-7 and abs(float(V3)) > 0:
                        # V3: same-sublattice cross-hexagon / third-neighbor density interaction.
                        out.append(("D", [Rx, Ry, 0], o_left, o_right,
                                    -1, 0, 0, 0, (float(V3), 0.0)))
    return out


# Default production builder: official divERGe D-channel s1=-1 convention.
density_vertices = density_vertices_official_s1minus1

kagome_spinful_hopping_entries = spinful_hoppings
spinful_vertex_entries_density_density = density_vertices


def summarize_hoppings(hops: Sequence[HopEntry]) -> Dict[str, int]:
    """Count entries by real/complex amplitude (not a unique physical term type)."""
    counts: Counter[str] = Counter()
    for R, o1, o2, _s1, _s2, amp in hops:
        if abs(float(amp[1])) > 0:
            counts["imag/SOC"] += 1
        elif o1 == o2 and list(R) == [0, 0, 0]:
            counts["onsite"] += 1
        else:
            counts["real"] += 1
    return dict(counts)


def summarize_vertices(vertices: Sequence[VertexEntry]) -> Dict[str, int]:
    """Count onsite-U, NN-V1, second-neighbor-V2, and same-sublattice cross-hexagon-V3 vertex entries."""
    lattice, pos = kagome_lattice_and_positions()
    counts: Counter[str] = Counter()
    for _chan, R, o1, o2, *_rest in vertices:
        if o1 == o2 and list(R) == [0, 0, 0]:
            counts["onsite_U_like"] += 1
            continue
        dr = displacement_vector(lattice, pos, int(R[0]), int(R[1]), int(o1), int(o2))
        dist = float(np.linalg.norm(dr[:2]))
        if abs(dist - 0.5) < 1e-7:
            counts["NN_V1_like"] += 1
        elif abs(dist - np.sqrt(3.0) / 2.0) < 1e-7:
            counts["NNN_V2_like"] += 1
        elif o1 == o2 and abs(dist - 1.0) < 1e-7:
            counts["cross_hexagon_V3_like"] += 1
        else:
            counts["other_vertex"] += 1
    return dict(counts)


# -----------------------------------------------------------------------------
# Pure-numpy Hamiltonian, filling, and sanity checks
# -----------------------------------------------------------------------------

def Hk_reduced(k_red: Sequence[float], hops: Sequence[HopEntry], *, norb: int = N_ORB, nspin: int = N_SPIN) -> np.ndarray:
    """Construct H(k) from hopping entries in reduced reciprocal coordinates."""
    k = np.asarray(k_red, dtype=float).reshape(3)
    H = np.zeros((nspin * norb, nspin * norb), dtype=complex)
    for R, o1, o2, s1, s2, amp in hops:
        phase = np.exp(2j * np.pi * float(np.dot(k, np.asarray(R, dtype=float))))
        row = int(s1) * norb + int(o1)
        col = int(s2) * norb + int(o2)
        H[row, col] += complex(float(amp[0]), float(amp[1])) * phase
    return H


def minus_k(k_red: Sequence[float]) -> np.ndarray:
    """Return -k folded into [0,1) reduced coordinates."""
    k = np.asarray(k_red, dtype=float).reshape(3)
    out = -k
    out = out - np.floor(out)
    out[np.isclose(out, 1.0, atol=1e-12)] = 0.0
    out[np.isclose(out, 0.0, atol=1e-12)] = 0.0
    return out


def time_reversal_unitary(*, norb: int = N_ORB) -> np.ndarray:
    """Unitary part of T = U_T K in basis (up orbitals, down orbitals)."""
    I = np.eye(norb, dtype=complex)
    return np.block([[np.zeros((norb, norb), dtype=complex), I],
                     [-I, np.zeros((norb, norb), dtype=complex)]])


def Sz_matrix(*, norb: int = N_ORB) -> np.ndarray:
    """S_z matrix up to an irrelevant factor of 1/2."""
    return np.diag([+1.0] * norb + [-1.0] * norb).astype(complex)


def band_energies(k_red: Sequence[float], hops: Sequence[HopEntry]) -> np.ndarray:
    """Sorted spinful band energies at reduced momentum k."""
    return np.linalg.eigvalsh(Hk_reduced(k_red, hops))


def estimate_filling(hops: Sequence[HopEntry], *, fermi_level: float = 0.0, ngrid: int = 80) -> float:
    """Estimate occupied fraction over all six spinful bands at a Fermi level.

    If `hops` already include the onsite term -mu, use fermi_level=0.
    If `hops` do not include -mu, use fermi_level=mu.
    """
    if int(ngrid) <= 0:
        raise ValueError("ngrid must be positive")
    count = 0.0
    tol = 1e-10
    for ix in range(int(ngrid)):
        for iy in range(int(ngrid)):
            k = np.array([(ix + 0.5) / ngrid, (iy + 0.5) / ngrid, 0.0])
            ev = band_energies(k, hops)
            count += float(np.count_nonzero(ev < fermi_level - tol))
            count += 0.5 * float(np.count_nonzero(np.abs(ev - fermi_level) <= tol))
    return count / float(ngrid * ngrid * N_ORB * N_SPIN)


def solve_mu_for_filling(
    *,
    target_filling: float = 5.0 / 12.0,
    t: float = 1.0,
    lambda_nn: float = 0.0,
    lambda_nnn: float = 0.0,
    lambda_rashba: float = 0.0,
    ngrid: int = 120,
    bracket: Optional[Tuple[float, float]] = None,
    maxiter: int = 80,
    tol: float = 1e-8,
) -> Tuple[float, Dict[str, float]]:
    """Resolve chemical potential for a target zero-temperature filling fraction.

    This routine uses a finite uniform k-grid and all six spinful bands.  It
    solves the *noninteracting* filling problem for H0 without an onsite -mu term.
    The returned mu should then be inserted into divERGe as an onsite hopping
    -mu, so divERGe's Fermi level remains zero.

    Because the finite-grid filling is a step function of mu, the primary solver
    is a sorted-spectrum quantile rather than pure bisection.  `bracket`,
    `maxiter`, and `tol` are accepted for API stability but are not normally
    needed.
    """
    del bracket, maxiter, tol  # kept for backward-compatible signature

    target = float(target_filling)
    if not (0.0 <= target <= 1.0):
        raise ValueError("target_filling must be in [0,1]")
    if int(ngrid) <= 0:
        raise ValueError("ngrid must be positive")

    base_hops = spinful_hoppings(t=t, mu=0.0, lambda_nn=lambda_nn, lambda_nnn=lambda_nnn, lambda_rashba=lambda_rashba, include_mu=False)
    energies: List[float] = []
    for ix in range(int(ngrid)):
        for iy in range(int(ngrid)):
            k = np.array([(ix + 0.5) / ngrid, (iy + 0.5) / ngrid, 0.0])
            energies.extend(float(x) for x in band_energies(k, base_hops))
    e = np.sort(np.asarray(energies, dtype=float))
    ntot = int(e.size)

    if target <= 0.0:
        mu = float(e[0] - 1.0)
    elif target >= 1.0:
        mu = float(e[-1] + 1.0)
    else:
        x = target * ntot
        n_occ = int(round(x))
        # If target*ntot is not integer, choose the nearest quantile.  This is
        # unavoidable on a finite grid; the report records the achieved filling.
        n_occ = max(0, min(ntot, n_occ))
        if n_occ == 0:
            mu = float(e[0] - 1e-9)
        elif n_occ == ntot:
            mu = float(e[-1] + 1e-9)
        else:
            mu = float(0.5 * (e[n_occ - 1] + e[n_occ]))

    achieved = estimate_filling(base_hops, fermi_level=mu, ngrid=ngrid)
    return float(mu), {
        "target_filling": target,
        "resolved_mu": float(mu),
        "estimated_filling": float(achieved),
        "filling_error": float(achieved - target),
        "ngrid": int(ngrid),
        "ntotal_states_on_grid": int(ntot),
        "target_states_on_grid": float(target * ntot),
        "method": "sorted_spectrum_quantile",
    }


def resolve_parameters(params: KMParameters) -> Tuple[KMParameters, Dict[str, object]]:
    """Return parameters with the actual mu used in the model plus a report."""
    params.validate()
    if params.filling_mode == "fixed_mu":
        hops_no_mu = spinful_hoppings(
            t=params.t,
            mu=0.0,
            lambda_nn=params.lambda_nn,
            lambda_nnn=params.lambda_nnn,
            lambda_rashba=params.lambda_rashba,
            include_mu=False,
        )
        filling = estimate_filling(hops_no_mu, fermi_level=params.mu, ngrid=params.filling_ngrid)
        return params, {
            "filling_mode": "fixed_mu",
            "input_mu": float(params.mu),
            "resolved_mu": float(params.mu),
            "estimated_filling_at_mu": float(filling),
            "target_filling_reference": float(params.target_filling),
            "ngrid": int(params.filling_ngrid),
        }

    mu, rep = solve_mu_for_filling(
        target_filling=params.target_filling,
        t=params.t,
        lambda_nn=params.lambda_nn,
        lambda_nnn=params.lambda_nnn,
        lambda_rashba=params.lambda_rashba,
        ngrid=params.filling_ngrid,
    )
    return replace(params, mu=mu), {"filling_mode": "fixed_filling", **rep}


def noninteracting_sanity_report(
    *,
    t: float = 1.0,
    mu: float = 0.0,
    lambda_nn: float = 0.0,
    lambda_nnn: float = 0.0,
    lambda_rashba: float = 0.0,
    ngrid: int = 80,
    n_random: int = 32,
    seed: int = 0,
) -> Dict[str, object]:
    """Return one-body sanity checks for the final H(k)-mu Hamiltonian.

    The hopping list includes the onsite term -mu. Occupancy is measured at
    Fermi level zero. For fixed-filling mode, use `resolve_parameters` first.
    """
    hops = spinful_hoppings(t=t, mu=mu, lambda_nn=lambda_nn, lambda_nnn=lambda_nnn, lambda_rashba=lambda_rashba, include_mu=True)
    rng = np.random.default_rng(seed)
    UT = time_reversal_unitary()
    Sz = Sz_matrix()

    max_herm = max_tr = max_sz = max_spin_offdiag = max_ud_relation = 0.0

    for _ in range(int(n_random)):
        k = rng.random(3)
        k[2] = 0.0
        H = Hk_reduced(k, hops)
        Hm = Hk_reduced(minus_k(k), hops)

        Hup = H[:N_ORB, :N_ORB]
        Hud = H[:N_ORB, N_ORB:]
        Hdu = H[N_ORB:, :N_ORB]
        Hdn = H[N_ORB:, N_ORB:]
        Hup_m = Hm[:N_ORB, :N_ORB]

        max_herm = max(max_herm, float(np.linalg.norm(H - H.conj().T, ord=np.inf)))
        max_tr = max(max_tr, float(np.linalg.norm(UT @ H.conj() @ UT.conj().T - Hm, ord=np.inf)))
        max_sz = max(max_sz, float(np.linalg.norm(H @ Sz - Sz @ H, ord=np.inf)))
        max_spin_offdiag = max(max_spin_offdiag, float(np.linalg.norm(Hud, ord=np.inf)), float(np.linalg.norm(Hdu, ord=np.inf)))
        max_ud_relation = max(max_ud_relation, float(np.linalg.norm(Hdn - Hup_m.conj(), ord=np.inf)))

    hs = {
        "Gamma": np.array([0.0, 0.0, 0.0]),
        "M1": np.array([0.5, 0.0, 0.0]),
        "M2": np.array([0.0, 0.5, 0.0]),
        "M3": np.array([0.5, 0.5, 0.0]),
        "K": np.array([1.0 / 3.0, 1.0 / 3.0, 0.0]),
    }

    return {
        "parameters": {"t": float(t), "mu": float(mu), "lambda_nn": float(lambda_nn), "lambda_nnn": float(lambda_nnn), "lambda_rashba": float(lambda_rashba)},
        "hop_summary": summarize_hoppings(hops),
        "max_hermiticity_error": max_herm,
        "max_time_reversal_error": max_tr,
        "max_Sz_commutator_error": max_sz,
        "max_spin_offdiag_block_norm": max_spin_offdiag,
        "max_Hdn_minus_Hup_star_minus_k": max_ud_relation,
        "expected_Sz_conserved": abs(float(lambda_rashba)) == 0.0,
        "high_symmetry_energies_shifted": {name: band_energies(k, hops).tolist() for name, k in hs.items()},
        "estimated_filling_at_fermi_zero": estimate_filling(hops, fermi_level=0.0, ngrid=ngrid),
        "target_filling_5_over_12": 5.0 / 12.0,
    }


def assert_sanity_passes(report: Mapping[str, object], *, tol: float = 1e-8) -> None:
    """Raise if core one-body symmetries fail in a sanity report."""
    keys = ["max_hermiticity_error", "max_time_reversal_error"]
    # The following are symmetry tests only when lambda_rashba == 0.
    # Rashba deliberately mixes up/down and breaks the Sz U(1).
    if bool(report.get("expected_Sz_conserved", True)):
        keys.extend(["max_Sz_commutator_error",
                     "max_spin_offdiag_block_norm",
                     "max_Hdn_minus_Hup_star_minus_k"])
    bad = {k: float(report[k]) for k in keys if float(report[k]) > tol}
    if not bool(report.get("expected_Sz_conserved", True)) and \
            float(report.get("max_spin_offdiag_block_norm", 0.0)) <= tol:
        bad["Rashba_spin_flip_missing"] = float(report.get("max_spin_offdiag_block_norm", 0.0))
    if bad:
        raise RuntimeError(f"one-body sanity checks failed: {bad}")


# -----------------------------------------------------------------------------
# divERGe model construction
# -----------------------------------------------------------------------------

def _set_model_arrays(model, lattice_np: np.ndarray, pos_np: np.ndarray) -> None:
    """Fill divERGe model lattice/position arrays."""
    import diverge
    lattice = diverge.view_array(model.contents.lattice, dtype=np.float64, shape=(3, 3))
    lattice[:, :] = lattice_np
    positions = diverge.view_array(model.contents.positions, dtype=np.float64, shape=(N_ORB, 3))
    positions[:, :] = pos_np


def build_model(
    *,
    params: KMParameters = KMParameters(),
    mesh: MeshParameters = MeshParameters(),
    name: bytes = b"kagome_km",
    model_outfile: Optional[str] = None,
    call_tu: bool = True,
    validate_sanity: bool = True,
    vertex_convention: str = "official_s1minus1",
):
    """Build and initialize a spinful divERGe kagome KM model.

    Returns a divERGe model pointer. The allocated numpy arrays are attached to
    `model._py_keepalive` so the C-side pointers remain valid while the model is
    alive. The actual chemical potential used is stored in model._py_metadata.
    """
    import diverge

    mesh.validate()
    if int(mesh.nk) % 2 != 0:
        raise ValueError("Use even nk so Gamma/M1/M2/M3 are exactly on the q mesh")

    resolved_params, filling_report = resolve_parameters(params)

    if validate_sanity:
        rep = noninteracting_sanity_report(
            t=resolved_params.t,
            mu=resolved_params.mu,
            lambda_nn=resolved_params.lambda_nn,
            lambda_nnn=resolved_params.lambda_nnn,
            lambda_rashba=resolved_params.lambda_rashba,
            ngrid=max(20, min(80, int(resolved_params.filling_ngrid))),
            n_random=16,
        )
        assert_sanity_passes(rep, tol=1e-8)
    else:
        rep = {}

    model = diverge.model_init()
    model.contents.name = name
    model.contents.n_orb = N_ORB
    model.contents.SU2 = False
    model.contents.n_spin = N_SPIN

    model.contents.nk[0] = int(mesh.nk)
    model.contents.nk[1] = int(mesh.nk)
    model.contents.nk[2] = 1
    model.contents.nkf[0] = int(mesh.nkf)
    model.contents.nkf[1] = int(mesh.nkf)
    model.contents.nkf[2] = 1

    lattice_np, pos_np = kagome_lattice_and_positions()
    _set_model_arrays(model, lattice_np, pos_np)

    hops = spinful_hoppings(
        t=resolved_params.t,
        mu=resolved_params.mu,
        lambda_nn=resolved_params.lambda_nn,
        lambda_nnn=resolved_params.lambda_nnn,
        lambda_rashba=resolved_params.lambda_rashba,
        include_mu=True,
    )
    hoppings = diverge.alloc_array((len(hops),), dtype="rs_hopping_t")
    hoppings[:] = hops
    model.contents.hop = hoppings.ctypes.data
    model.contents.n_hop = hoppings.size

    if vertex_convention in {"official", "official_s1minus1", "s1minus1"}:
        vertices_list = density_vertices_official_s1minus1(U=resolved_params.U, V1=resolved_params.V1, V2=resolved_params.V2, V3=resolved_params.V3)
    elif vertex_convention in {"manual", "legacy_manual"}:
        vertices_list = density_vertices_manual(U=resolved_params.U, V1=resolved_params.V1, V2=resolved_params.V2, V3=resolved_params.V3)
    else:
        raise ValueError(f"unknown vertex_convention={vertex_convention!r}; use official_s1minus1 or manual")
    vertices = diverge.alloc_array((len(vertices_list),), dtype="rs_vertex_t")
    vertices[:] = vertices_list
    model.contents.vert = vertices.ctypes.data
    model.contents.n_vert = vertices.size

    # Gamma-M-K-Gamma path. No point-group symmetries are supplied for now; this
    # avoids subtle spinful SOC symmetry mistakes.
    ibz_path = diverge.view_array(model.contents.ibz_path, dtype=np.float64, shape=(4, 3))
    ibz_path[:, :] = 0.0
    ibz_path[1, 0] = 0.5
    ibz_path[2, 0] = 0.5
    ibz_path[2, 1] = 0.5
    model.contents.n_ibz_path = 4
    model.contents.n_sym = 0

    err = diverge.model_validate(model)
    if err:
        raise RuntimeError(f"diverge.model_validate failed with code {err}")
    diverge.model_internals_common(model)
    if call_tu:
        diverge.model_internals_tu(model, float(mesh.tu_distance))

    checksum = None
    if model_outfile is not None:
        checksum = diverge.model_to_file(model, str(model_outfile))

    model._py_keepalive = (hoppings, vertices, hops, vertices_list)
    model._py_metadata = {
        "input_params": asdict(params),
        "resolved_params": asdict(resolved_params),
        **asdict(mesh),
        "filling_report": filling_report,
        "sanity_report": rep,
        "hop_summary": summarize_hoppings(hops),
        "vertex_summary": summarize_vertices(vertices_list),
        "vertex_convention": str(vertex_convention),
        "q_indices": high_symmetry_q_indices(mesh.nk),
        "model_outfile": model_outfile,
        "model_checksum": checksum,
        "gauge_note": "real-space/orbital TU model; no Bloch gauge fixing used in model construction",
    }
    return model


build_diverge_spinful_kagome_model = build_model


def model_summary(model) -> Dict[str, object]:
    """Return a compact Python summary for a built divERGe model."""
    return {
        "n_orb": int(model.contents.n_orb),
        "n_spin": int(model.contents.n_spin),
        "SU2": int(model.contents.SU2),
        "n_hop": int(model.contents.n_hop),
        "n_vert": int(model.contents.n_vert),
        "n_tu_ff": int(getattr(model.contents, "n_tu_ff", 0)),
        "metadata": getattr(model, "_py_metadata", {}),
    }


__all__ = [
    "A", "B", "C", "SPIN_UP", "SPIN_DN", "N_ORB", "N_SPIN",
    "KMParameters", "MeshParameters", "HopEntry", "VertexEntry",
    "kagome_lattice_and_positions", "displacement_vector", "tu_ff_displacement_vector",
    "enumerate_offdiag_bonds_at_distance", "high_symmetry_q_indices", "qmesh_reduced",
    "spinful_hoppings", "density_vertices", "density_vertices_official_s1minus1", "density_vertices_manual", "kagome_spinful_hopping_entries", "spinful_vertex_entries_density_density",
    "summarize_hoppings", "summarize_vertices",
    "Hk_reduced", "minus_k", "time_reversal_unitary", "Sz_matrix",
    "band_energies", "estimate_filling", "solve_mu_for_filling", "resolve_parameters",
    "noninteracting_sanity_report", "assert_sanity_passes",
    "build_model", "build_diverge_spinful_kagome_model", "model_summary",
]
