#!/usr/bin/env python3
# coding: utf-8
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import TUfRG_model as km


def load_json(p: Path):
    with open(p, "r") as f:
        return json.load(f)


def recursive_find_key(obj, key):
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            ans = recursive_find_key(v, key)
            if ans is not None:
                return ans
    elif isinstance(obj, list):
        for v in obj:
            ans = recursive_find_key(v, key)
            if ans is not None:
                return ans
    return None


def normalize(v):
    v = np.asarray(v, dtype=complex).reshape(-1)
    n = np.sqrt(np.vdot(v, v).real)
    if n < 1e-14:
        raise ValueError("zero vector")
    return v / n


def orthonormalize_pair(v1, v2):
    q1 = normalize(v1)
    q2 = np.asarray(v2, dtype=complex).reshape(-1) - q1 * np.vdot(q1, v2)
    q2 = normalize(q2)
    return q1, q2


def reduced_kmesh(nk):
    for i in range(nk):
        for j in range(nk):
            yield np.array([i / nk, j / nk, 0.0], dtype=float)


def minus_k(k):
    return np.mod(-np.asarray(k, dtype=float), 1.0)


def bdg_matrix(hk, hmk, Delta):
    n = hk.shape[0]
    H = np.zeros((2 * n, 2 * n), dtype=complex)
    H[:n, :n] = hk
    H[:n, n:] = Delta
    H[n:, :n] = Delta.conj().T
    H[n:, n:] = -hmk.T
    return 0.5 * (H + H.conj().T)


def spin_singlet_delta6_spinmajor(Dorb):
    Delta = np.zeros((6, 6), dtype=complex)

    def idx(a, s):
        # spin-major basis: Aup, Bup, Cup, Adn, Bdn, Cdn
        return a + 3 * s

    for a in range(3):
        for b in range(3):
            Delta[idx(a, 0), idx(b, 1)] += Dorb[a, b]
            Delta[idx(a, 1), idx(b, 0)] += -Dorb[a, b]

    return Delta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case-dir", required=True)
    ap.add_argument("--nk", type=int, default=24)
    ap.add_argument("--delta0", type=float, default=0.01)
    ap.add_argument("--ntheta", type=int, default=41)
    ap.add_argument("--nphi", type=int, default=41)
    ap.add_argument("--phase-sign", type=int, default=1)
    ap.add_argument("--basis-rotation", type=float, default=0.0)
    ap.add_argument("--out-subdir", default="bdg_e2_theta_phi_scan_trb")
    args = ap.parse_args()

    case_dir = Path(args.case_dir)
    snap = case_dir / "snapshots" / "final"
    outdir = case_dir / args.out_subdir
    outdir.mkdir(parents=True, exist_ok=True)

    print("============================================================")
    print("E2-SC theta-phi BdG scan with TRB measure")
    print("============================================================")
    print("case_dir =", case_dir)
    print("nk =", args.nk)
    print("delta0 =", args.delta0)
    print("ntheta,nphi =", args.ntheta, args.nphi)
    print("basis_rotation =", args.basis_rotation)

    # ---------------- metadata ----------------
    metas = []
    for p in [case_dir / "metadata.json", snap / "metadata.json"]:
        if p.exists():
            metas.append(load_json(p))

    resolved_mu = None
    for m in metas:
        fr = recursive_find_key(m, "filling_report")
        if isinstance(fr, dict) and "resolved_mu" in fr:
            resolved_mu = float(fr["resolved_mu"])
            break
    if resolved_mu is None:
        resolved_mu = 0.011859505773669515

    t = 1.0
    lambda_nn = 0.0
    lambda_nnn = 0.1
    U = None
    V1 = None

    for m in metas:
        ip = recursive_find_key(m, "input_params")
        if isinstance(ip, dict):
            t = float(ip.get("t", t))
            lambda_nn = float(ip.get("lambda_nn", lambda_nn))
            lambda_nnn = float(ip.get("lambda_nnn", lambda_nnn))
            U = ip.get("U", U)
            V1 = ip.get("V1", V1)
            break

    print("Model:")
    print("  U,V1 =", U, V1)
    print("  mu =", resolved_mu)
    print("  lambda_nn =", lambda_nn)
    print("  lambda_nnn =", lambda_nnn)

    hops = km.kagome_spinful_hopping_entries(
        t=t,
        mu=resolved_mu,
        lambda_nn=lambda_nn,
        lambda_nnn=lambda_nnn,
        include_mu=True,
    )

    # ---------------- modes ----------------
    inst = pd.read_csv(snap / "instability" / "instability_summary.csv")
    modes = np.load(snap / "instability" / "instability_modes.npz", allow_pickle=True)
    comp = pd.read_csv(snap / "formfactor_shell_v2" / "component_shell_map.csv")

    sel = inst[
        (inst["q_label"].astype(str) == "Gamma")
        & (inst["family"].astype(str) == "pp")
        & (inst["sector"].astype(str).str.contains("singlet", case=False, na=False))
    ].copy()

    sel = sel.sort_values("strength_svd_KL", ascending=False)

    print("\nTop Gamma pp-singlet modes:")
    print(sel.head(6)[["mode_key", "mode", "strength_svd_KL"]].to_string(index=False))

    row1 = sel.iloc[0]
    row2 = sel.iloc[1]

    key1 = str(row1["mode_key"])
    key2 = str(row2["mode_key"])

    psi1 = np.asarray(modes[key1], dtype=complex).reshape(-1)
    psi2 = np.asarray(modes[key2], dtype=complex).reshape(-1)
    psi1, psi2 = orthonormalize_pair(psi1, psi2)

    alpha = args.basis_rotation
    p1 = math.cos(alpha) * psi1 + math.sin(alpha) * psi2
    p2 = -math.sin(alpha) * psi1 + math.cos(alpha) * psi2
    psi1, psi2 = orthonormalize_pair(p1, p2)

    rel_split = abs(row1["strength_svd_KL"] - row2["strength_svd_KL"]) / max(abs(row1["strength_svd_KL"]), 1e-30)

    print("\nSelected two modes:")
    print("  key1 =", key1)
    print("  key2 =", key2)
    print("  strength1 =", row1["strength_svd_KL"])
    print("  strength2 =", row2["strength_svd_KL"])
    print("  relative splitting =", rel_split)
    print("  <psi1|psi2> =", np.vdot(psi1, psi2))

    # ---------------- form factor map ----------------
    R0 = comp["R0"].to_numpy(dtype=float)
    R1 = comp["R1"].to_numpy(dtype=float)
    R2 = comp["R2"].to_numpy(dtype=float)
    ofrom = comp["ofrom"].to_numpy(dtype=int)
    oto = comp["oto"].to_numpy(dtype=int)

    def delta_orb_raw(kred, psi):
        D = np.zeros((3, 3), dtype=complex)
        k = np.asarray(kred, dtype=float)
        phase_arg = R0 * k[0] + R1 * k[1] + R2 * k[2]
        phases = np.exp(args.phase_sign * 2j * np.pi * phase_arg)
        for m in range(len(psi)):
            D[int(ofrom[m]), int(oto[m])] += psi[m] * phases[m]
        return D

    def delta_orb_singlet(kred, psi):
        Dk = delta_orb_raw(kred, psi)
        Dmk = delta_orb_raw(minus_k(kred), psi)
        return 0.5 * (Dk + Dmk.T)

    def delta6(kred, psi):
        return spin_singlet_delta6_spinmajor(delta_orb_singlet(kred, psi))

    # ---------------- cache ----------------
    klist = list(reduced_kmesh(args.nk))
    HK, HMK, D1, D2, E0_terms = [], [], [], [], []

    zero_delta = np.zeros((6, 6), dtype=complex)

    print("\nCaching matrices...")
    for ik, k in enumerate(klist):
        if ik % max(1, len(klist)//10) == 0:
            print(f"  cache {ik}/{len(klist)}")

        hk = km.Hk_reduced(k, hops)
        hmk = km.Hk_reduced(minus_k(k), hops)
        d1 = delta6(k, psi1)
        d2 = delta6(k, psi2)

        HK.append(hk)
        HMK.append(hmk)
        D1.append(d1)
        D2.append(d2)

        H0 = bdg_matrix(hk, hmk, zero_delta)
        ev0 = np.linalg.eigvalsh(H0)
        E0_terms.append(-0.5 * float(np.sum(np.abs(ev0))))

    E0_NORMAL = float(np.mean(E0_terms))

    def combo_coeffs(theta, phi):
        return math.cos(theta), np.exp(1j * phi) * math.sin(theta)

    def combo_rms(c1, c2):
        total = 0.0
        count = 0
        for d1, d2 in zip(D1, D2):
            D = c1 * d1 + c2 * d2
            total += float(np.trace(D.conj().T @ D).real)
            count += D.shape[0]
        return math.sqrt(total / max(count, 1))

    def trb_measure(c1, c2):
        # orbital/spin gap matrix TR proxy:
        # compare Delta(k) and Delta^*(-k), allowing a global phase.
        g = []
        gt = []
        for d1, d2 in zip(D1, D2):
            D = c1 * d1 + c2 * d2
            g.append(D.reshape(-1))

        # D(-k) needs same ordering; rebuild dictionary by direct evaluation is safer.
        # For speed, use actual klist.
        for k in klist:
            d1m = delta6(minus_k(k), psi1)
            d2m = delta6(minus_k(k), psi2)
            Dtr = (c1 * d1m + c2 * d2m).conj()
            gt.append(Dtr.reshape(-1))

        g = np.concatenate(g)
        gt = np.concatenate(gt)

        ng = math.sqrt(np.vdot(g, g).real)
        nt = math.sqrt(np.vdot(gt, gt).real)
        if ng < 1e-14 or nt < 1e-14:
            return np.nan

        overlap = abs(np.vdot(g, gt)) / (ng * nt)
        return float(max(0.0, 1.0 - overlap))

    def free_energy(theta, phi):
        c1, c2 = combo_coeffs(theta, phi)

        rms = combo_rms(c1, c2)
        c1 = c1 / rms
        c2 = c2 / rms

        vals = []
        min_gap = 1e100

        for hk, hmk, d1, d2 in zip(HK, HMK, D1, D2):
            Delta = args.delta0 * (c1 * d1 + c2 * d2)
            H = bdg_matrix(hk, hmk, Delta)
            ev = np.linalg.eigvalsh(H)

            pos = ev[ev > 1e-10]
            if len(pos):
                min_gap = min(min_gap, float(np.min(pos)))

            vals.append(-0.5 * float(np.sum(np.abs(ev))))

        F = float(np.mean(vals))
        trb = trb_measure(c1, c2)

        return F - E0_NORMAL, min_gap, rms, trb

    # ---------------- scan ----------------
    theta_vals = np.linspace(0.0, math.pi, args.ntheta)
    phi_vals = np.linspace(0.0, math.pi, args.nphi)

    rows = []
    total_jobs = len(theta_vals) * len(phi_vals)
    counter = 0

    print("\nRunning theta-phi scan...")
    for theta in theta_vals:
        for phi in phi_vals:
            counter += 1
            if counter % max(1, total_jobs // 20) == 0:
                print(f"  scan {counter}/{total_jobs}")

            dF, gap, rms, trb = free_energy(theta, phi)

            rows.append({
                "theta": theta,
                "phi": phi,
                "deltaF": dF,
                "min_positive_BdG_gap": gap,
                "gap_rms_before_norm": rms,
                "trb_measure": trb,
                "delta0": args.delta0,
                "nk": args.nk,
                "basis_rotation": args.basis_rotation,
                "U": U,
                "V1": V1,
                "relative_splitting": rel_split,
            })

    grid = pd.DataFrame(rows)
    Emin = grid["deltaF"].min()
    grid["deltaF_minus_min"] = grid["deltaF"] - Emin

    candidates = [
        ("psi1", "real", 0.0, 0.0),
        ("psi2", "real", math.pi / 2, 0.0),
        ("real_plus", "real", math.pi / 4, 0.0),
        ("real_minus", "real", math.pi / 4, math.pi),
        ("chiral_plus", "chiral", math.pi / 4, math.pi / 2),
        ("chiral_minus", "chiral", math.pi / 4, -math.pi / 2),
    ]

    cand_rows = []
    for name, kind, theta, phi in candidates:
        dF, gap, rms, trb = free_energy(theta, phi)
        cand_rows.append({
            "name": name,
            "kind": kind,
            "theta": theta,
            "phi": phi,
            "deltaF": dF,
            "deltaF_minus_grid_min": dF - Emin,
            "min_positive_BdG_gap": gap,
            "gap_rms_before_norm": rms,
            "trb_measure": trb,
            "delta0": args.delta0,
            "nk": args.nk,
            "basis_rotation": args.basis_rotation,
            "U": U,
            "V1": V1,
        })

    cand = pd.DataFrame(cand_rows)

    best_grid = grid.sort_values("deltaF").iloc[0]
    best_real = cand[cand["kind"] == "real"].sort_values("deltaF").iloc[0]
    best_chiral = cand[cand["kind"] == "chiral"].sort_values("deltaF").iloc[0]

    summary = pd.DataFrame([{
        "U": U,
        "V1": V1,
        "nk": args.nk,
        "delta0": args.delta0,
        "ntheta": args.ntheta,
        "nphi": args.nphi,
        "basis_rotation": args.basis_rotation,
        "relative_splitting": rel_split,
        "grid_min_theta": best_grid["theta"],
        "grid_min_phi": best_grid["phi"],
        "grid_min_deltaF": best_grid["deltaF"],
        "grid_min_trb": best_grid["trb_measure"],
        "best_real_name": best_real["name"],
        "best_real_deltaF": best_real["deltaF"],
        "best_real_trb": best_real["trb_measure"],
        "best_chiral_name": best_chiral["name"],
        "best_chiral_deltaF": best_chiral["deltaF"],
        "best_chiral_trb": best_chiral["trb_measure"],
        "chiral_minus_real": best_chiral["deltaF"] - best_real["deltaF"],
        "candidate_favored": "real/nematic" if best_chiral["deltaF"] > best_real["deltaF"] else "chiral",
    }])

    tag = (
        f"nk{args.nk}_d{args.delta0:g}_nth{args.ntheta}_nph{args.nphi}_"
        f"rot{args.basis_rotation:.6f}"
    ).replace(".", "p").replace("-", "m")

    grid_csv = outdir / f"theta_phi_grid_trb_{tag}.csv"
    cand_csv = outdir / f"candidate_points_trb_{tag}.csv"
    summary_csv = outdir / f"theta_phi_summary_trb_{tag}.csv"

    grid.to_csv(grid_csv, index=False)
    cand.to_csv(cand_csv, index=False)
    summary.to_csv(summary_csv, index=False)

    print("\nSummary:")
    print(summary.to_string(index=False))
    print("\nCandidates:")
    print(cand.to_string(index=False))
    print("\nBest grid points:")
    print(grid.sort_values("deltaF").head(10).to_string(index=False))

    print("\nWrote:")
    print(" ", grid_csv)
    print(" ", cand_csv)
    print(" ", summary_csv)


if __name__ == "__main__":
    main()
