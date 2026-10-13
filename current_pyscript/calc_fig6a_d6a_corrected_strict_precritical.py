#!/usr/bin/env python3
from pathlib import Path
import json
import traceback
import numpy as np
import pandas as pd

import TUfRG_instability as inst

# Reuse the already-tested strict-BSE utilities from Fig.5.
from calc_fig5_corrected_strict_charge_precritical import (
    ROOTS,
    comp_key,
    reverse_key,
    strict_chi_for_eta,
    parse_params,
    find_case_roots,
    find_snapshots,
    select_light_snapshots,
    case_display_name,
    snapshot_label,
    snapshot_iter,
)

OUTDIR = Path("fig6a_d6a_corrected_strict_precritical")
OUTDIR.mkdir(parents=True, exist_ok=True)

SUB = {"A": 0, "B": 1, "C": 2}
SUB_INV = {v: k for k, v in SUB.items()}

# ============================================================
# Corrected D6a / Fig.6(a) source
#
# Important correction:
#   B13 is flipped from +1 to -1.
#
# With this correction:
#   ||eta_G||^2 = 0
#   ||eta_M1||^2 = ||eta_M2||^2 = ||eta_M3||^2 = 1/3
# ============================================================

sqrt3 = np.sqrt(3.0)

A1_VIS = np.array([2.0, 0.0])
A2_VIS = np.array([1.0, sqrt3])

TAU_VIS = {
    "A": np.array([0.0, 0.0]),
    "B": np.array([1.0, 0.0]),
    "C": np.array([-0.5, sqrt3 / 2.0]),
}

SITE_XY = {
    "r1_1": np.array([1.5, 0.86602540]),

    "r2_0": np.array([-1.0, 1.73205081]),
    "r2_1": np.array([ 0.0, 1.73205081]),
    "r2_2": np.array([ 1.0, 1.73205081]),
    "r2_3": np.array([ 2.0, 1.73205081]),
    "r2_4": np.array([ 3.0, 1.73205081]),
    "r2_5": np.array([ 4.0, 1.73205081]),

    "r3_1": np.array([0.5, 2.59807621]),
    "r3_2": np.array([2.5, 2.59807621]),

    "r4_0": np.array([-1.0, 3.46410162]),
    "r4_1": np.array([ 0.0, 3.46410162]),
    "r4_2": np.array([ 1.0, 3.46410162]),
    "r4_3": np.array([ 2.0, 3.46410162]),
    "r4_4": np.array([ 3.0, 3.46410162]),
    "r4_5": np.array([ 4.0, 3.46410162]),

    "r5_0": np.array([-0.5, 4.33012702]),
    "r5_1": np.array([ 1.5, 4.33012702]),
    "r5_2": np.array([ 3.5, 4.33012702]),

    "r6_1": np.array([1.0, 5.19615242]),
    "r6_2": np.array([2.0, 5.19615242]),
}

B_CANONICAL = {
    "B00": ("b32", "r4_3", "r4_2", (( 1,  0), "A", "B")),
    "B01": ("b33", "r4_3", "r4_4", (( 0,  0), "A", "B")),
    "B02": ("b38", "r4_3", "r5_1", (( 0,  0), "A", "C")),
    "B03": ("b27", "r4_3", "r3_2", ((-1,  1), "A", "C")),
    "B04": ("b28", "r4_4", "r3_2", ((-1,  1), "B", "C")),
    "B05": ("b30", "r4_0", "r4_1", ((-1,  0), "B", "A")),
    "B06": ("b39", "r4_4", "r5_2", ((-1,  0), "B", "C")),
    "B07": ("b37", "r5_1", "r4_2", (( 1,  0), "C", "B")),
    "B08": ("b42", "r5_1", "r6_1", (( 1, -1), "C", "A")),
    "B09": ("b43", "r5_1", "r6_2", (( 1, -1), "C", "B")),

    "B10": ("b14", "r2_2", "r2_1", (( 1,  0), "A", "B")),
    "B11": ("b15", "r2_2", "r2_3", (( 0,  0), "A", "B")),
    "B12": ("b20", "r2_2", "r3_1", (( 0,  0), "A", "C")),
    "B13": ("b09", "r2_2", "r1_1", ((-1,  1), "A", "C")),
    "B14": ("b10", "r2_3", "r1_1", ((-1,  1), "B", "C")),
    "B15": ("b16", "r2_3", "r2_4", ((-1,  0), "B", "A")),
    "B16": ("b21", "r2_3", "r3_2", ((-1,  0), "B", "C")),
    "B17": ("b19", "r3_1", "r2_1", (( 1,  0), "C", "B")),
    "B18": ("b25", "r3_1", "r4_1", (( 1, -1), "C", "A")),
    "B19": ("b26", "r3_1", "r4_2", (( 1, -1), "C", "B")),

    "B20": ("b31", "r4_1", "r4_2", (( 0,  0), "A", "B")),
    "B21": ("b36", "r4_1", "r5_0", (( 0,  0), "A", "C")),
    "B22": ("b13", "r2_0", "r2_1", (( 0,  0), "A", "B")),
    "B23": ("b22", "r2_4", "r3_2", (( 0,  0), "A", "C")),
}

EXPECTED_B_CLASSES = [f"B{i:02d}" for i in range(24)]

D6A_B_SIGN = {
    "B00": -1,
    "B01": +1,
    "B02": -1,
    "B03": +1,
    "B04": +1,
    "B05": +1,
    "B06": -1,
    "B07": -1,
    "B08": -1,
    "B09": +1,
    "B10": +1,
    "B11": -1,
    "B12": +1,

    # Corrected typo:
    "B13": -1,

    "B14": +1,
    "B15": -1,
    "B16": -1,
    "B17": -1,
    "B18": +1,
    "B19": +1,
    "B20": +1,
    "B21": +1,
    "B22": -1,
    "B23": -1,
}

QRED = {
    "M1": (0.5, 0.0),
    "M2": (0.0, 0.5),
    "M3": (0.5, 0.5),
}


def cell_pos(n1, n2, sub):
    return n1 * A1_VIS + n2 * A2_VIS + TAU_VIS[sub]


def infer_cell_sub_from_xy(xy, search_range=range(-4, 6), tol=1e-6):
    xy = np.asarray(xy, dtype=float)
    matches = []
    for n1 in search_range:
        for n2 in search_range:
            for sub in ["A", "B", "C"]:
                err = np.linalg.norm(cell_pos(n1, n2, sub) - xy)
                if err < tol:
                    matches.append((n1, n2, sub))
    if len(matches) != 1:
        raise RuntimeError(f"Could not uniquely infer cell/sub for xy={xy}, matches={matches}")
    return matches[0]


def q_phase(qred, cell):
    q1, q2 = qred
    n1, n2 = cell
    return np.exp(-2j * np.pi * (q1 * n1 + q2 * n2))


def build_d6a_source_terms():
    assert sorted(B_CANONICAL.keys()) == EXPECTED_B_CLASSES
    assert sorted(D6A_B_SIGN.keys()) == EXPECTED_B_CLASSES

    terms = []

    for B in EXPECTED_B_CLASSES:
        bid, can0, can1, tu_key_raw = B_CANONICAL[B]
        sign = int(D6A_B_SIGN[B])

        R_raw, ofrom, oto = tu_key_raw
        R = (int(R_raw[0]), int(R_raw[1]))

        s_n1, s_n2, s_sub = infer_cell_sub_from_xy(SITE_XY[can0])
        e_n1, e_n2, e_sub = infer_cell_sub_from_xy(SITE_XY[can1])

        if s_sub != ofrom or e_sub != oto:
            raise RuntimeError(f"{B}: sublattice mismatch")

        home = (e_n1, e_n2)
        expected_start = (home[0] + R[0], home[1] + R[1])
        actual_start = (s_n1, s_n2)

        if actual_start != expected_start:
            raise RuntimeError(
                f"{B}: TU key geometry mismatch: "
                f"actual_start={actual_start}, home+R={expected_start}, "
                f"R={R}, ofrom={ofrom}, oto={oto}"
            )

        key = ((R[0], R[1]), ofrom, oto)
        rkey = ((-R[0], -R[1]), oto, ofrom)
        rhome = (home[0] + R[0], home[1] + R[1])

        terms.append({
            "B": B,
            "sign": sign,
            "key": key,
            "home": home,
            "reverse_key": rkey,
            "reverse_home": rhome,
        })

    return terms


D6A_TERMS = build_d6a_source_terms()


def make_d6a_sources_on_O(O):
    """
    Return q_label -> normalized eta_q on the actual O.dvg TU basis.

    Global normalization:
        sum_{M1,M2,M3} ||eta_q||^2 = 1.

    For corrected D6a:
        each ||eta_Mi||^2 should be 1/3.
    """
    Rarr = np.asarray(O.mi_to_R, dtype=int)
    ofrom = np.asarray(O.mi_to_ofrom, dtype=int)
    oto = np.asarray(O.mi_to_oto, dtype=int)
    nff = len(ofrom)

    key_to_idx = {
        comp_key(Rarr[i], ofrom[i], oto[i]): i
        for i in range(nff)
    }

    etas = {}

    for qlab, qred in QRED.items():
        eta = np.zeros(nff, dtype=np.complex128)

        for t in D6A_TERMS:
            s = t["sign"]

            key = comp_key((t["key"][0][0], t["key"][0][1]), SUB[t["key"][1]], SUB[t["key"][2]])
            rkey = reverse_key(key)

            if key not in key_to_idx:
                raise RuntimeError(f"Missing D6a key {key} from {t}")
            if rkey not in key_to_idx:
                raise RuntimeError(f"Missing D6a reverse key {rkey} from {t}")

            eta[key_to_idx[key]] += 1j * s * q_phase(qred, t["home"])
            eta[key_to_idx[rkey]] += -1j * s * q_phase(qred, t["reverse_home"])

        etas[qlab] = eta

    total_norm2 = sum(float(np.vdot(e, e).real) for e in etas.values())
    if total_norm2 < 1e-14:
        raise RuntimeError("zero D6a source")

    global_norm = np.sqrt(total_norm2)
    etas = {q: e / global_norm for q, e in etas.items()}

    norm2 = {q: float(np.vdot(e, e).real) for q, e in etas.items()}
    total = sum(norm2.values())

    if abs(total - 1.0) > 1e-12:
        raise RuntimeError(f"bad total source norm: {total}")

    for q in ["M1", "M2", "M3"]:
        if abs(norm2[q] - 1/3) > 1e-12:
            raise RuntimeError(f"D6a q weight not 1/3 for {q}: {norm2}")

    # Hermiticity check for each q.
    max_herm_err = 0.0
    for q, eta in etas.items():
        for i in range(nff):
            key = comp_key(Rarr[i], ofrom[i], oto[i])
            rkey = reverse_key(key)
            j = key_to_idx.get(rkey, None)
            if j is not None:
                max_herm_err = max(max_herm_err, abs(eta[j] - np.conjugate(eta[i])))

    # For finite-q Fourier components eta_q need not be Hermitian by themselves
    # in the same q=0 sense, so we record this but do not assert it.
    diag = {
        "source_norm2_total": float(total),
        "source_norm2_M1": norm2["M1"],
        "source_norm2_M2": norm2["M2"],
        "source_norm2_M3": norm2["M3"],
        "source_nonzero_M1": int(np.count_nonzero(np.abs(etas["M1"]) > 1e-12)),
        "source_nonzero_M2": int(np.count_nonzero(np.abs(etas["M2"]) > 1e-12)),
        "source_nonzero_M3": int(np.count_nonzero(np.abs(etas["M3"]) > 1e-12)),
        "finite_q_reverse_conj_check_recorded": float(max_herm_err),
    }

    return etas, diag


def compute_snapshot(root_label, root_path, run_dir, snapshot_dir):
    case_name = case_display_name(run_dir)
    snap_lab = snapshot_label(snapshot_dir, run_dir)
    snap_it = snapshot_iter(snapshot_dir, run_dir)
    params = parse_params(case_name)

    print(f"[snapshot] {root_label} :: {case_name} :: {snap_lab}", flush=True)

    data = inst.load_case(snapshot_dir)
    inst.assert_raw_channel_compatibility(data)

    qmap = inst.q_labels_to_indices(data, ["M1", "M2", "M3"])
    etas_by_q, source_diag = make_d6a_sources_on_O(data.O)

    rows = []

    for qlab in ["M1", "M2", "M3"]:
        iq = qmap[qlab]
        eta = etas_by_q[qlab]

        kernels = inst.build_explicit_kernels(data, iq)

        for spec in kernels:
            if spec["family"] != "ph":
                continue
            if spec["sector"] not in ["ph-rho", "ph-sx", "ph-sy", "ph-sz"]:
                continue

            K_saved = np.asarray(spec["K"], dtype=np.complex128)
            L_saved = np.asarray(spec["L"], dtype=np.complex128)

            diag = strict_chi_for_eta(eta, K_saved, L_saved)

            rows.append({
                "root_label": root_label,
                "root_path": str(root_path),
                "case_name": case_name,
                "run_dir": str(run_dir),
                "snapshot_label": snap_lab,
                "snapshot_iter": int(snap_it),
                "snapshot_dir": str(snapshot_dir),
                "q_label": qlab,
                "iq": int(iq),
                "sector": spec["sector"],
                "source": "D6a_fig6a_corrected",
                **source_diag,
                **diag,
                **params,
            })

    return rows


def select_last_precritical_sum(df, threshold):
    """
    For a given snapshot/sector, sum chi over M1,M2,M3 only if all three q rows
    are available and all have B_max_eval < threshold.
    Then select the latest snapshot per case/sector.
    """
    if len(df) == 0:
        return df

    keys0 = [
        "root_label", "case_name", "U", "V",
        "sector", "snapshot_label", "snapshot_iter"
    ]

    agg = df.groupby(keys0).agg(
        chi_D6a_sum=("chi_strict_real", "sum"),
        chi_D6a_imag_sum=("chi_strict_imag", "sum"),
        B_max_eval_max=("B_max_eval", "max"),
        min_1_minus_B_min=("min_1_minus_B", "min"),
        direct_spectral_diff_max=("direct_spectral_diff", "max"),
        loop_orientation_sign=("loop_orientation_sign", "first"),
        chi0_neg_weight_ratio_max=("chi0_neg_weight_ratio", "max"),
        n_q=("q_label", "nunique"),
    ).reset_index()

    agg = agg[(agg["n_q"] == 3) & (agg["B_max_eval_max"] < threshold)].copy()
    if len(agg) == 0:
        return agg

    keys1 = ["root_label", "case_name", "U", "V", "sector"]
    idx = agg.groupby(keys1)["snapshot_iter"].idxmax()
    return agg.loc[idx].sort_values(keys1).reset_index(drop=True)


def write_outputs(all_rows, errors):
    df = pd.DataFrame(all_rows)
    df.to_csv(OUTDIR / "fig6a_d6a_corrected_strict_precritical_all.csv", index=False)
    (OUTDIR / "errors.json").write_text(json.dumps(errors, indent=2))

    if len(df) == 0:
        return

    for thr, tag in [(0.90, "0p90"), (0.95, "0p95"), (1.00, "1p00")]:
        sel = select_last_precritical_sum(df, thr)
        sel.to_csv(OUTDIR / f"selected_sum_Bmax_lt_{tag}.csv", index=False)

        if len(sel):
            compact = sel.pivot_table(
                index=[
                    "root_label", "case_name", "U", "V",
                    "snapshot_label", "snapshot_iter",
                    "B_max_eval_max",
                    "loop_orientation_sign",
                    "chi0_neg_weight_ratio_max",
                ],
                columns="sector",
                values="chi_D6a_sum",
                aggfunc="first",
            ).reset_index()

            compact.to_csv(OUTDIR / f"selected_sum_Bmax_lt_{tag}_compact.csv", index=False)


def main():
    all_rows = []
    errors = []

    inventory = []
    for root_label, root_str in ROOTS:
        case_roots = find_case_roots(root_str)

        cases = []
        for rd in case_roots:
            snaps_all = find_snapshots(rd)
            snaps_light = select_light_snapshots(snaps_all)
            cases.append({
                "case_name": case_display_name(rd),
                "run_dir": str(rd),
                "n_snapshots_all": len(snaps_all),
                "n_snapshots_used": len(snaps_light),
                "snapshots_used": [str(s) for s in snaps_light],
            })

        inventory.append({
            "root_label": root_label,
            "root_path": root_str,
            "n_cases": len(case_roots),
            "cases": cases,
        })

    (OUTDIR / "inventory.json").write_text(json.dumps(inventory, indent=2))

    print("=== Inventory ===", flush=True)
    for item in inventory:
        nused = sum(c["n_snapshots_used"] for c in item["cases"])
        print(
            f"{item['root_label']:20s} "
            f"{item['n_cases']:3d} cases, {nused:4d} selected snapshots  {item['root_path']}",
            flush=True,
        )

    for root_label, root_str in ROOTS:
        for rd in find_case_roots(root_str):
            snaps_all = find_snapshots(rd)
            snaps = select_light_snapshots(snaps_all)

            if len(snaps) == 0:
                errors.append({
                    "root_label": root_label,
                    "run_dir": str(rd),
                    "error": "no snapshots found",
                })
                write_outputs(all_rows, errors)
                continue

            for s in snaps:
                try:
                    rows = compute_snapshot(root_label, root_str, rd, s)
                    all_rows.extend(rows)
                    write_outputs(all_rows, errors)
                except Exception as e:
                    print(f"[ERROR] {root_label} {rd} {s}: {e}", flush=True)
                    errors.append({
                        "root_label": root_label,
                        "run_dir": str(rd),
                        "snapshot_dir": str(s),
                        "error": repr(e),
                        "traceback": traceback.format_exc(),
                    })
                    write_outputs(all_rows, errors)

    write_outputs(all_rows, errors)

    df = pd.DataFrame(all_rows)
    print("\n=== Done ===", flush=True)
    print(f"rows: {len(df)}", flush=True)
    print(f"errors: {len(errors)}", flush=True)
    print(f"outdir: {OUTDIR.resolve()}", flush=True)

    if len(df):
        print("\n=== Source q weights ===", flush=True)
        print(df[["source_norm2_M1", "source_norm2_M2", "source_norm2_M3"]].drop_duplicates().to_string(index=False), flush=True)

        print("\n=== Sector counts ===", flush=True)
        print(df["sector"].value_counts().to_string(), flush=True)

        print("\n=== q counts ===", flush=True)
        print(df["q_label"].value_counts().to_string(), flush=True)

        print("\n=== Loop orientation counts ===", flush=True)
        print(df["loop_orientation_sign"].value_counts().to_string(), flush=True)

        print("\n=== Normal-state validity counts ===", flush=True)
        print(df["normal_state_valid"].value_counts().to_string(), flush=True)

        p = OUTDIR / "selected_sum_Bmax_lt_0p95_compact.csv"
        print("\n=== Selected summed D6a, Bmax < 0.95 compact preview ===", flush=True)
        if p.exists():
            print(pd.read_csv(p).to_string(index=False), flush=True)
        else:
            print("No Bmax<0.95 selected data.", flush=True)


if __name__ == "__main__":
    main()
