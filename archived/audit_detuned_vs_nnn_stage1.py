#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd


DET_ROOT = Path(
    "/storage/scratch1/4/mli826/divERGe_test_run/runs/"
    "nosoc_mu0p011_prrflow_nk24_nkf12_11pts"
)

NNN_ROOT = Path(
    "/storage/scratch1/4/mli826/divERGe_test_run/runs/lnnn0p1"
)

OUTDIR = Path("residual_audit_detuned_vs_lnnn0p1")
OUTDIR.mkdir(parents=True, exist_ok=True)


POINTS = [
    (0.0, 0.5),
    (0.0, 1.5),
    (1.0, 0.0),
    (1.0, 0.5),
    (1.0, 1.5),
    (2.0, 0.0),
    (2.0, 0.5),
    (2.0, 1.5),
    (3.0, 0.0),
    (3.0, 0.5),
    (3.0, 1.5),
]


def tag(x):
    return f"{x:.1f}".replace(".", "p")


def case_paths(U, V):
    ut = tag(U)
    vt = tag(V)

    det = DET_ROOT / f"nosoc_mu0p011_U{ut}_V{vt}"

    nnn = NNN_ROOT / (
        f"lnnn0p1_prrflow_nk24_nkf12_U{ut}_V{vt}_scan"
    )

    return det, nnn


def diagnosis_files(case):
    base = case / "snapshots" / "final" / "instability"
    return (
        base / "instability_summary.csv",
        base / "instability_modes.npz",
    )


def normalize(v):
    v = np.asarray(v, dtype=complex).reshape(-1)
    n = np.linalg.norm(v)
    if n <= 0:
        raise RuntimeError("zero-norm mode")
    return v / n


def get_sector(df, q, sector):
    x = df[
        (df["q_label"].astype(str) == q)
        & (df["sector"].astype(str) == sector)
    ].copy()

    if "family" in x.columns:
        fam = "pp" if sector.startswith("pp-") else "ph"
        x = x[x["family"].astype(str) == fam]

    x["mode"] = pd.to_numeric(x["mode"], errors="coerce")
    x["strength_svd_KL"] = pd.to_numeric(
        x["strength_svd_KL"], errors="coerce"
    )

    return x.sort_values("mode")


def load_sc(case):
    summary_f, modes_f = diagnosis_files(case)

    if not summary_f.exists():
        raise FileNotFoundError(summary_f)
    if not modes_f.exists():
        raise FileNotFoundError(modes_f)

    df = pd.read_csv(summary_f)
    z = np.load(modes_f, allow_pickle=True)

    sc = get_sector(df, "Gamma", "pp-singlet")

    if len(sc) < 2:
        raise RuntimeError(
            f"{case}: fewer than two Gamma pp-singlet modes"
        )

    vecs = []
    strengths = []
    keys = []

    for _, row in sc.iterrows():
        key = str(row["mode_key"])
        if key not in z:
            raise KeyError(f"{case}: missing NPZ key {key}")

        vecs.append(normalize(z[key]))
        strengths.append(float(row["strength_svd_KL"]))
        keys.append(key)

    return vecs, np.asarray(strengths), keys


def top_spin_strength(df, q, sector):
    x = get_sector(df, q, sector)

    if len(x) == 0:
        return np.nan

    # mode 0 is the leading singular mode for that q/sector.
    return float(x.iloc[0]["strength_svd_KL"])


def spin_info(case):
    summary_f, _ = diagnosis_files(case)
    df = pd.read_csv(summary_f)

    out = {}

    for q in ["Gamma", "M1", "M2", "M3"]:
        sx = top_spin_strength(df, q, "ph-sx")
        sy = top_spin_strength(df, q, "ph-sy")
        sz = top_spin_strength(df, q, "ph-sz")

        xy = np.nanmean([sx, sy])

        rz = sz / xy if np.isfinite(xy) and xy != 0 else np.nan

        denom = np.nanmean([sx, sy, sz])
        anis = (
            (sz - xy) / denom
            if np.isfinite(denom) and denom != 0
            else np.nan
        )

        xy_split = (
            abs(sx - sy) / xy
            if np.isfinite(xy) and xy != 0
            else np.nan
        )

        out[f"{q}_Sx"] = sx
        out[f"{q}_Sy"] = sy
        out[f"{q}_Sz"] = sz
        out[f"{q}_Rz"] = rz
        out[f"{q}_anis"] = anis
        out[f"{q}_xy_split"] = xy_split

    return out


def principal_singular_values(V1, V2, dim=2):
    """
    Compare two mode subspaces.

    Columns are orthonormalized before comparison, so arbitrary
    rotations/phases inside a degenerate subspace do not matter.
    """
    A = np.column_stack(V1[:dim])
    B = np.column_stack(V2[:dim])

    QA, _ = np.linalg.qr(A)
    QB, _ = np.linalg.qr(B)

    sv = np.linalg.svd(QA.conj().T @ QB, compute_uv=False)

    return np.sort(np.real_if_close(sv))[::-1]


rows = []

for U, V in POINTS:
    det, nnn = case_paths(U, V)

    print()
    print("=" * 78)
    print(f"U={U:g}, V={V:g}")
    print("DET:", det)
    print("NNN:", nnn)

    if not det.exists():
        print("[MISSING DET]")
        continue

    if not nnn.exists():
        print("[MISSING NNN]")
        continue

    dvec, ds, dkeys = load_sc(det)
    nvec, ns, nkeys = load_sc(nnn)

    if len(dvec[0]) != len(nvec[0]):
        raise RuntimeError(
            f"TU dimension mismatch at U={U},V={V}: "
            f"{len(dvec[0])} vs {len(nvec[0])}"
        )

    # --------------------------------------------------------
    # Nondegenerate top-mode overlap.
    # Global complex phase drops out of |<F|G>|^2.
    # --------------------------------------------------------
    inner = np.vdot(dvec[0], nvec[0])

    overlap0 = float(abs(inner) ** 2)

    phi = float(np.angle(inner))

    # Align NNN mode to detuned mode for an intuitive L2 mismatch.
    n_aligned = nvec[0] * np.exp(-1j * phi)

    aligned_l2 = float(np.linalg.norm(dvec[0] - n_aligned))

    # --------------------------------------------------------
    # Two-dimensional subspace comparison.
    # Essential for E-type doublets.
    # --------------------------------------------------------
    ps = principal_singular_values(dvec, nvec, dim=2)

    sigma1 = float(ps[0])
    sigma2 = float(ps[1])

    theta1 = float(np.degrees(np.arccos(np.clip(sigma1, 0, 1))))
    theta2 = float(np.degrees(np.arccos(np.clip(sigma2, 0, 1))))

    subspace_fidelity = float((sigma1**2 + sigma2**2) / 2)

    # SC spectral separation.
    det_s1_s0 = float(ds[1] / ds[0])
    nnn_s1_s0 = float(ns[1] / ns[0])

    # --------------------------------------------------------
    # Spin anisotropy.
    # --------------------------------------------------------
    dspin = spin_info(det)
    nspin = spin_info(nnn)

    row = {
        "U": U,
        "V1": V,

        "det_SC_S0": ds[0],
        "det_SC_S1": ds[1],
        "det_SC_S1_over_S0": det_s1_s0,

        "nnn_SC_S0": ns[0],
        "nnn_SC_S1": ns[1],
        "nnn_SC_S1_over_S0": nnn_s1_s0,

        "SC_mode0_overlap_sq": overlap0,
        "SC_mode0_abs_overlap": float(abs(inner)),
        "SC_mode0_opt_phase_rad": phi,
        "SC_mode0_aligned_L2": aligned_l2,

        "SC_2D_sigma1": sigma1,
        "SC_2D_sigma2": sigma2,
        "SC_2D_theta1_deg": theta1,
        "SC_2D_theta2_deg": theta2,
        "SC_2D_subspace_fidelity": subspace_fidelity,

        "det_mode0_key": dkeys[0],
        "det_mode1_key": dkeys[1],
        "nnn_mode0_key": nkeys[0],
        "nnn_mode1_key": nkeys[1],
    }

    for k, x in dspin.items():
        row[f"det_{k}"] = x

    for k, x in nspin.items():
        row[f"nnn_{k}"] = x

    rows.append(row)

    print(
        f"  S0 det/NNN       = {ds[0]:.6g} / {ns[0]:.6g}"
    )
    print(
        f"  S1/S0 det/NNN    = {det_s1_s0:.6f} / {nnn_s1_s0:.6f}"
    )
    print(
        f"  mode-0 overlap^2 = {overlap0:.8f}"
    )
    print(
        f"  2D principal sv  = ({sigma1:.8f}, {sigma2:.8f})"
    )
    print(
        f"  principal angles = ({theta1:.4f}, {theta2:.4f}) deg"
    )
    print(
        f"  Gamma Rz det/NNN = "
        f"{dspin['Gamma_Rz']:.8f} / {nspin['Gamma_Rz']:.8f}"
    )


out = pd.DataFrame(rows)

csv = OUTDIR / "stage1_SC_overlap_and_spin_anisotropy.csv"
out.to_csv(csv, index=False)


# ------------------------------------------------------------
# Compact human-readable table
# ------------------------------------------------------------
cols = [
    "U", "V1",
    "det_SC_S1_over_S0",
    "nnn_SC_S1_over_S0",
    "SC_mode0_overlap_sq",
    "SC_2D_sigma1",
    "SC_2D_sigma2",
    "SC_2D_theta1_deg",
    "SC_2D_theta2_deg",
    "det_Gamma_Rz",
    "nnn_Gamma_Rz",
]

compact = out[cols].copy()

txt = OUTDIR / "stage1_compact_table.txt"
txt.write_text(compact.to_string(index=False))

print()
print("=" * 78)
print("COMPACT RESULT")
print("=" * 78)
print(compact.to_string(index=False))

print()
print("WROTE")
print(csv)
print(txt)
