#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import re
import traceback
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

# Use existing production loader/kernel convention.
# This should be available in your runs/project path.
import TUfRG_instability as inst


ROOTS = {
    "nosoc": "nosoc",
    "lambda_nn_0p1": "lnn0p1_soc",
    "lambda_nnn_0p1": "lnnn0p1_soc",
}

OUTDIR = Path("fig5_final_kappa_tau_kappa")
OUTDIR.mkdir(parents=True, exist_ok=True)

SUB = {"A": 0, "B": 1, "C": 2}
A = SUB["A"]
B = SUB["B"]
C = SUB["C"]

# Corrected PACE/TU convention for Fig.5(a,b) 1x1 LCO sources.
# Component key ((R1,R2), ofrom, oto) means source on TU component (R, ofrom, oto).
UP_KEYS = [
    ((0, 0), A, B),  # A -> B
    ((0, 0), B, C),  # B -> C
    ((0, 0), C, A),  # C -> A
]

DOWN_KEYS = [
    ((-1, 0), B, A),  # B(-a1) -> A
    ((0, -1), C, A),  # C(-a2) -> A
    ((1, -1), C, B),  # equivalent translated down-triangle bond
]

SOURCES = {
    # same-flux pattern
    "fig5a": {
        "up":   [+1, +1, +1],
        "down": [-1, +1, -1],
    },
    # opposite-flux pattern
    "fig5b": {
        "up":   [+1, +1, +1],
        "down": [+1, -1, +1],
    },
}

SECTORS = ["ph-rho", "ph-sx", "ph-sy", "ph-sz"]

# This sign is just a diagnostic label following our corrected Fig.5 source convention.
LOOP_ORIENTATION_SIGN = -1


def parse_float_token(x):
    if x is None:
        return np.nan
    if isinstance(x, (int, float, np.number)):
        return float(x)
    s = str(x).strip()
    s = s.replace("m", "-").replace("p", ".")
    try:
        return float(s)
    except Exception:
        return np.nan


def read_metadata_nearest(p: Path):
    for x in [p] + list(p.parents):
        f = x / "metadata.json"
        if f.exists():
            try:
                return json.loads(f.read_text()), x
            except Exception:
                return {}, x
    return {}, p


def parse_case_params(snapshot_dir: Path):
    meta, meta_dir = read_metadata_nearest(snapshot_dir)
    out = {}

    for key in ["U", "V", "V1", "lambda_nn", "lambda_nnn", "mu", "target_filling"]:
        if key in meta:
            out[key] = parse_float_token(meta[key])

    if isinstance(meta.get("params"), dict):
        for key in ["U", "V", "V1", "lambda_nn", "lambda_nnn", "mu", "target_filling"]:
            if key in meta["params"]:
                out[key] = parse_float_token(meta["params"][key])

    txt = str(snapshot_dir)

    if "lambda_nn" not in out:
        m = re.search(r"lnn(?!n)([0-9mp\.\-]+)", txt)
        if m:
            out["lambda_nn"] = parse_float_token(m.group(1))

    if "lambda_nnn" not in out:
        m = re.search(r"lnnn([0-9mp\.\-]+)", txt)
        if m:
            out["lambda_nnn"] = parse_float_token(m.group(1))

    if "U" not in out:
        m = re.search(r"U([0-9mp\.\-]+)", txt)
        if m:
            out["U"] = parse_float_token(m.group(1))

    if "V" not in out and "V1" not in out:
        m = re.search(r"V([0-9mp\.\-]+)", txt)
        if m:
            out["V"] = parse_float_token(m.group(1))

    if "V" not in out and "V1" in out:
        out["V"] = out["V1"]

    out.setdefault("U", np.nan)
    out.setdefault("V", np.nan)
    out.setdefault("lambda_nn", 0.0)
    out.setdefault("lambda_nnn", 0.0)
    out["metadata_dir"] = str(meta_dir)
    return out


def canonical_case_root_from_snapshot(snapshot_dir: Path):
    # Typical final snapshot:
    #   case/snapshots/final
    if snapshot_dir.name == "final" and snapshot_dir.parent.name == "snapshots":
        return snapshot_dir.parent.parent
    return snapshot_dir


def find_final_snapshot_dirs(root: Path):
    """Find only final snapshot O.dvg. Do not scan iter_XXXX snapshots."""
    root = Path(root)
    found = []

    # Preferred: case/snapshots/final/O.dvg
    for odvg in sorted(root.rglob("snapshots/final/O.dvg")):
        found.append(odvg.parent)

    # Fallback: root/case/O.dvg, but skip all snapshots/iter_XXXX.
    for odvg in sorted(root.rglob("O.dvg")):
        p = odvg.parent
        parts = set(p.parts)
        if p.name == "final" and p.parent.name == "snapshots":
            continue
        if "snapshots" in parts:
            # skip iter_XXXX or other non-final snapshots
            continue
        # Only use fallback if this case has no preferred final snapshot already.
        case_root = p
        has_preferred = (case_root / "snapshots" / "final" / "O.dvg").exists()
        if not has_preferred:
            found.append(p)

    # Deduplicate
    out = []
    seen = set()
    for d in found:
        key = str(d.resolve())
        if key not in seen:
            seen.add(key)
            out.append(d)
    return out


def tu_arrays(O):
    if hasattr(inst, "tu_component_arrays"):
        R, ofrom, oto = inst.tu_component_arrays(O)
    else:
        R = np.asarray(O.mi_to_R, dtype=int)
        ofrom = np.asarray(O.mi_to_ofrom, dtype=int)
        oto = np.asarray(O.mi_to_oto, dtype=int)
    R = np.asarray(R, dtype=int)
    ofrom = np.asarray(ofrom, dtype=int)
    oto = np.asarray(oto, dtype=int)
    if R.ndim == 1:
        R = R.reshape((-1, 3))
    if R.shape[1] == 2:
        R = np.column_stack([R, np.zeros(len(R), dtype=int)])
    return R, ofrom, oto


def build_key_index(O):
    R, ofrom, oto = tu_arrays(O)
    idx = {}
    for i in range(len(ofrom)):
        idx[(int(R[i, 0]), int(R[i, 1]), int(ofrom[i]), int(oto[i]))] = i
    return idx, R, ofrom, oto


def reverse_key(key):
    (R12, a, b) = key
    return ((-int(R12[0]), -int(R12[1])), int(b), int(a))


def flat_key(key):
    (R12, a, b) = key
    return (int(R12[0]), int(R12[1]), int(a), int(b))


def build_fig5_eta(O, source_name):
    idx, R, ofrom, oto = build_key_index(O)
    nff = len(ofrom)
    eta = np.zeros(nff, dtype=np.complex128)

    spec = SOURCES[source_name]
    amp = 1j / math.sqrt(12.0)

    missing = []
    duplicate_check = defaultdict(int)

    for key, sgn in zip(UP_KEYS, spec["up"]):
        k = flat_key(key)
        rk = flat_key(reverse_key(key))
        if k not in idx:
            missing.append(("key", key))
            continue
        if rk not in idx:
            missing.append(("reverse", reverse_key(key)))
            continue
        eta[idx[k]] += amp * float(sgn)
        eta[idx[rk]] += np.conjugate(amp * float(sgn))
        duplicate_check[idx[k]] += 1
        duplicate_check[idx[rk]] += 1

    for key, sgn in zip(DOWN_KEYS, spec["down"]):
        k = flat_key(key)
        rk = flat_key(reverse_key(key))
        if k not in idx:
            missing.append(("key", key))
            continue
        if rk not in idx:
            missing.append(("reverse", reverse_key(key)))
            continue
        eta[idx[k]] += amp * float(sgn)
        eta[idx[rk]] += np.conjugate(amp * float(sgn))
        duplicate_check[idx[k]] += 1
        duplicate_check[idx[rk]] += 1

    if missing:
        raise RuntimeError(f"Missing Fig.5 source keys for {source_name}: {missing[:20]}")

    # Source norm check.
    norm = float(np.vdot(eta, eta).real)

    # Hermiticity / bond-reversal check: eta(reverse)=conj(eta).
    herm_err = 0.0
    for i in np.flatnonzero(np.abs(eta) > 1e-14):
        key = ((int(R[i, 0]), int(R[i, 1])), int(ofrom[i]), int(oto[i]))
        rk = flat_key(reverse_key(key))
        if rk not in idx:
            herm_err = max(herm_err, 1.0)
        else:
            j = idx[rk]
            herm_err = max(herm_err, float(abs(eta[j] - np.conjugate(eta[i]))))

    nnz = int(np.count_nonzero(np.abs(eta) > 1e-14))

    if abs(norm - 1.0) > 1e-12:
        raise RuntimeError(f"{source_name}: source norm is not 1: {norm}")
    if nnz != 12:
        raise RuntimeError(f"{source_name}: expected 12 nonzero components, got {nnz}")
    if herm_err > 1e-12:
        raise RuntimeError(f"{source_name}: source Hermiticity error too large: {herm_err}")

    info = {
        "source_norm": norm,
        "source_nnz": nnz,
        "source_hermiticity_err": herm_err,
    }
    return eta, info


def get_gamma_iq(data):
    qmap = inst.q_labels_to_indices(data, ["Gamma"])
    return int(qmap["Gamma"])


def build_sector_specs(data, iq):
    if getattr(data, "SU2", 0) == 0 and getattr(data, "n_spin", 2) == 2:
        specs = inst.build_explicit_kernels(data, iq)
    else:
        specs = inst.build_su2_kernels(data, iq)

    out = {}
    for sp in specs:
        if str(sp.get("family", "")) != "ph":
            continue
        sec = str(sp.get("sector", ""))
        if sec in SECTORS:
            out[sec] = sp
    return out


def hermitian_part(M):
    M = np.asarray(M, dtype=np.complex128)
    return 0.5 * (M + M.conj().T)


def choose_loop_sign(L_saved):
    """Choose s=+1 or -1 so chi0=s*L_saved is closest to PSD."""
    Lh = hermitian_part(L_saved)

    best = None
    for s in [+1.0, -1.0]:
        vals = np.linalg.eigvalsh(s * Lh)
        pos_weight = float(np.sum(vals[vals > 0.0]))
        neg_weight = float(np.sum(-vals[vals < 0.0]))
        total_abs = pos_weight + neg_weight
        neg_ratio = neg_weight / max(total_abs, 1e-300)
        # Prefer smaller negative ratio; tie-breaker larger positive weight.
        score = (neg_ratio, -pos_weight)
        if best is None or score < best[0]:
            best = (score, s, vals, neg_ratio)
    _, s, vals, neg_ratio = best
    return float(s), vals, float(neg_ratio)


def psd_sqrt_from_chi0(chi0, rel_tol=1e-12, abs_tol=1e-14):
    chi0h = hermitian_part(chi0)
    vals, vecs = np.linalg.eigh(chi0h)
    vmax = float(np.max(np.abs(vals))) if len(vals) else 0.0
    cutoff = max(abs_tol, rel_tol * vmax)
    keep = vals > cutoff
    if not np.any(keep):
        return np.zeros_like(chi0h), vals, keep
    Lsqrt = (vecs[:, keep] * np.sqrt(vals[keep])) @ vecs[:, keep].conj().T
    return Lsqrt, vals, keep


def compute_terms_for_source(K_saved, L_saved, eta):
    # Auto sign convention:
    # chi0 = s L_saved, Gamma = s K_saved
    # Then Gamma*chi0 = K_saved*L_saved is unchanged.
    s, chi0_evals_raw, neg_ratio = choose_loop_sign(L_saved)

    chi0 = s * hermitian_part(L_saved)
    Gamma = s * np.asarray(K_saved, dtype=np.complex128)

    v0 = chi0 @ eta
    chi0_val = np.vdot(eta, v0)

    chi2_val = np.vdot(eta, chi0 @ (Gamma @ (chi0 @ eta)))
    total_val = chi0_val + chi2_val

    ratio = np.nan
    if abs(chi0_val) > 1e-300:
        ratio = float(np.real(chi2_val / chi0_val))

    # Diagnostic A = chi0^{1/2} Gamma chi0^{1/2}
    Lsqrt, chi0_evals, keep = psd_sqrt_from_chi0(chi0)
    A = Lsqrt @ Gamma @ Lsqrt

    # Hermitian diagnostic max eigenvalue.
    Ah = hermitian_part(A)
    try:
        evals_h = np.linalg.eigvalsh(Ah)
        A_max_eval = float(np.max(evals_h)) if len(evals_h) else np.nan
    except Exception:
        A_max_eval = np.nan

    # Non-Hermitian spectral radius diagnostic.
    try:
        evals = np.linalg.eigvals(A)
        A_abs_max_eval = float(np.max(np.abs(evals))) if len(evals) else np.nan
    except Exception:
        A_abs_max_eval = np.nan

    return {
        "loop_sign_s": s,
        "chi0_real": float(np.real(chi0_val)),
        "chi0_imag": float(np.imag(chi0_val)),
        "chi2_kappa_tau_kappa_real": float(np.real(chi2_val)),
        "chi2_kappa_tau_kappa_imag": float(np.imag(chi2_val)),
        "chi0_plus_chi2_real": float(np.real(total_val)),
        "chi0_plus_chi2_imag": float(np.imag(total_val)),
        "enhancement_ratio_chi2_over_chi0": ratio,
        "A_max_eval": A_max_eval,
        "A_abs_max_eval": A_abs_max_eval,
        "normal_state_by_Amax_lt_1": bool(A_max_eval < 1.0) if np.isfinite(A_max_eval) else False,
        "chi0_neg_weight_ratio": float(neg_ratio),
        "chi0_rank_psd": int(np.count_nonzero(keep)),
        "chi0_min_eval_signed": float(np.min(chi0_evals)) if len(chi0_evals) else np.nan,
        "chi0_max_eval_signed": float(np.max(chi0_evals)) if len(chi0_evals) else np.nan,
    }


def analyze_snapshot(root_label, snapshot_dir: Path):
    case_root = canonical_case_root_from_snapshot(snapshot_dir)
    params = parse_case_params(snapshot_dir)
    case_name = case_root.name

    data = inst.load_case(snapshot_dir)
    iq = get_gamma_iq(data)
    specs = build_sector_specs(data, iq)

    missing_sectors = [s for s in SECTORS if s not in specs]
    if missing_sectors:
        raise RuntimeError(f"Missing PH sectors {missing_sectors}; found {sorted(specs)}")

    etas = {}
    source_infos = {}
    for src in SOURCES:
        eta, info = build_fig5_eta(data.O, src)
        etas[src] = eta
        source_infos[src] = info

    rows = []
    for src, eta in etas.items():
        for sector in SECTORS:
            sp = specs[sector]
            K_saved = np.asarray(sp["K"], dtype=np.complex128)
            L_saved = np.asarray(sp["L"], dtype=np.complex128)

            vals = compute_terms_for_source(K_saved, L_saved, eta)
            row = {
                "root_label": root_label,
                "case_name": case_name,
                "U": params.get("U", np.nan),
                "V": params.get("V", np.nan),
                "lambda_nn": params.get("lambda_nn", np.nan),
                "lambda_nnn": params.get("lambda_nnn", np.nan),
                "snapshot_dir": str(snapshot_dir),
                "source": src,
                "sector": sector,
                "iq": iq,
                "q_label": "Gamma",
                "loop_orientation_sign": LOOP_ORIENTATION_SIGN,
                **source_infos[src],
                **vals,
            }
            rows.append(row)
    return rows, source_infos


def make_compact(df):
    idx_cols = [
        "root_label", "case_name", "U", "V",
        "lambda_nn", "lambda_nnn", "snapshot_dir",
    ]
    metrics = [
        ("chi0_real", "chi0"),
        ("chi2_kappa_tau_kappa_real", "chi2"),
        ("chi0_plus_chi2_real", "chi0_plus_chi2"),
        ("enhancement_ratio_chi2_over_chi0", "ratio_chi2_over_chi0"),
        ("A_max_eval", "Amax"),
        ("A_abs_max_eval", "Aabsmax"),
        ("chi0_neg_weight_ratio", "chi0_neg_ratio"),
    ]

    base = df[idx_cols].drop_duplicates().copy()
    out = base.copy()

    for col, label in metrics:
        tmp = df[idx_cols + ["source", "sector", col]].copy()
        tmp["metric_name"] = tmp.apply(
            lambda r: f"{label}__{r['source']}__{r['sector']}", axis=1
        )
        piv = tmp.pivot_table(
            index=idx_cols,
            columns="metric_name",
            values=col,
            aggfunc="first",
        ).reset_index()
        out = out.merge(piv, on=idx_cols, how="left")

    return out


def near(a, b, rtol=1e-5, atol=1e-8):
    return abs(a - b) <= atol + rtol * max(abs(a), abs(b), 1.0)


def run_sanity_checks(df, inventory):
    checks = []

    # Each case/source should have all 4 sectors.
    for keys, sub in df.groupby(["root_label", "case_name", "source"], dropna=False):
        got = sorted(sub["sector"].unique())
        ok = got == sorted(SECTORS)
        checks.append({
            "check": "all_four_sectors_present",
            "keys": list(keys),
            "ok": bool(ok),
            "got": got,
        })

    # no-SOC spin degeneracy: sx, sy, sz approximately equal.
    # Use chi0/chi2/total real values.
    for (root_label, case_name, source), sub in df.groupby(["root_label", "case_name", "source"], dropna=False):
        if root_label != "nosoc":
            continue
        d = {r["sector"]: r for _, r in sub.iterrows()}
        if all(s in d for s in ["ph-sx", "ph-sy", "ph-sz"]):
            for metric in ["chi0_real", "chi2_kappa_tau_kappa_real", "chi0_plus_chi2_real"]:
                x = float(d["ph-sx"][metric])
                y = float(d["ph-sy"][metric])
                z = float(d["ph-sz"][metric])
                ok = near(x, y, rtol=2e-4, atol=1e-7) and near(x, z, rtol=2e-4, atol=1e-7)
                checks.append({
                    "check": "nosoc_sx_sy_sz_degeneracy",
                    "case_name": case_name,
                    "source": source,
                    "metric": metric,
                    "ok": bool(ok),
                    "sx": x,
                    "sy": y,
                    "sz": z,
                })

    # lambda_nn: sx should equal sy; sz may differ.
    for (root_label, case_name, source), sub in df.groupby(["root_label", "case_name", "source"], dropna=False):
        if root_label != "lambda_nn_0p1":
            continue
        d = {r["sector"]: r for _, r in sub.iterrows()}
        if all(s in d for s in ["ph-sx", "ph-sy"]):
            for metric in ["chi0_real", "chi2_kappa_tau_kappa_real", "chi0_plus_chi2_real"]:
                x = float(d["ph-sx"][metric])
                y = float(d["ph-sy"][metric])
                ok = near(x, y, rtol=2e-4, atol=1e-7)
                checks.append({
                    "check": "lambda_nn_sx_sy_degeneracy",
                    "case_name": case_name,
                    "source": source,
                    "metric": metric,
                    "ok": bool(ok),
                    "sx": x,
                    "sy": y,
                })

    inventory["sanity_checks"] = checks
    inventory["sanity_all_ok"] = bool(all(c.get("ok", False) for c in checks))


def main():
    all_rows = []
    errors = []
    inventory = {
        "roots": ROOTS,
        "outdir": str(OUTDIR),
        "root_inventory": {},
        "notes": [
            "Final snapshot only.",
            "Fig.5(a,b) are Gamma 1x1 corrected LCO sources.",
            "No resolvent is computed.",
            "chi0 = s L_saved and Gamma = s K_saved with s chosen to make chi0 closest to PSD.",
            "This preserves Gamma*chi0 = K_saved*L_saved.",
        ],
    }

    for root_label, root_str in ROOTS.items():
        root = Path(root_str)
        if not root.exists():
            errors.append({
                "root_label": root_label,
                "root": root_str,
                "snapshot_dir": None,
                "error": "missing root",
            })
            inventory["root_inventory"][root_label] = {
                "root": root_str,
                "exists": False,
                "n_snapshot_dirs": 0,
            }
            continue

        snap_dirs = find_final_snapshot_dirs(root)
        inventory["root_inventory"][root_label] = {
            "root": root_str,
            "exists": True,
            "n_snapshot_dirs": len(snap_dirs),
            "snapshot_dirs": [str(x) for x in snap_dirs],
        }

        print(f"[ROOT] {root_label}: found {len(snap_dirs)} final/fallback snapshot dirs")

        for sd in snap_dirs:
            try:
                rows, source_infos = analyze_snapshot(root_label, sd)
                all_rows.extend(rows)
                print(f"[OK] {root_label} {sd} -> {len(rows)} rows")
            except Exception as e:
                err = {
                    "root_label": root_label,
                    "snapshot_dir": str(sd),
                    "error": repr(e),
                    "traceback": traceback.format_exc(),
                }
                errors.append(err)
                print(f"[FAIL] {root_label} {sd}: {e}")

    if all_rows:
        df = pd.DataFrame(all_rows)
        df = df.sort_values(["root_label", "U", "V", "case_name", "source", "sector"])
        df.to_csv(OUTDIR / "fig5_final_kappa_tau_kappa_all.csv", index=False)

        compact = make_compact(df)
        compact = compact.sort_values(["root_label", "U", "V", "case_name"])
        compact.to_csv(OUTDIR / "fig5_final_kappa_tau_kappa_compact.csv", index=False)

        run_sanity_checks(df, inventory)
    else:
        inventory["sanity_checks"] = []
        inventory["sanity_all_ok"] = False

    with open(OUTDIR / "inventory.json", "w") as f:
        json.dump(inventory, f, indent=2)

    with open(OUTDIR / "errors.json", "w") as f:
        json.dump(errors, f, indent=2)

    print("\n[DONE]")
    print("rows =", len(all_rows))
    print("errors =", len(errors))
    print("outdir =", OUTDIR)
    if errors:
        print("First error:")
        print(json.dumps(errors[0], indent=2)[:3000])


if __name__ == "__main__":
    main()
