#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Read-only preflight for archived divERGe TUfRG kagome KM calculations.

Stage 3, module 1/3: archive I/O and compatibility, NOT an instability solver.
This file reads O.dvg headers, FF geometry and T.dvg metadata, but it never
materializes the large P/C/D arrays. It cannot certify that restored vertices
are correct: that is the job of the source-backed projection and physical
kernels in later modules.

Supported production scope: divERGe 0.10 archived 2D, n_orb=3,
explicit n_spin=2 / SU2=0, intrinsic NN/NNN KM SOC (including zero SOC),
no Rashba, uncompressed square q mesh, and T/O in the same snapshot directory.

Usage:
  python TUfRG_archive.py --case /path/to/snapshots/final
  python TUfRG_archive.py --case /path/to/snapshots/final \
      --binary /path/to/libdivERGe.so --json archive_manifest.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

# The *production runtime* audited in Stage 2. Other builds require re-audit.
VERIFIED_BINARY_SHA256 = (
    "cd300dd6ccc9b1f972ff8d93a3478db0695ba1465e3423951bdb9674d3e4e69b"
)
O_BLOCKS = {"P": 84, "C": 86, "D": 88, "PP_loop": 90, "PH_loop": 92}


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def digest(path: Path) -> str:
    """SHA256 of a model or shared library (NOT of a multi-GB O.dvg)."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_header(path: Path, count: int) -> np.ndarray:
    require(path.is_file(), f"Missing file: {path}")
    with path.open("rb") as f:
        header = np.fromfile(f, dtype="<i8", count=count)
    require(header.size == count, f"Truncated header: {path}")
    return header


def check_range(path: Path, offset: int, size: int, description: str) -> None:
    require(offset >= 0 and size >= 0, f"Negative {description} offset/count")
    require(offset + size <= path.stat().st_size,
            f"Truncated {description} in {path}: end={offset+size}, filesize={path.stat().st_size}")


def read_t(path: Path) -> dict[str, Any]:
    """Validate the archived divERGe model structure and its raw-data extents."""
    h = read_header(path, 128)
    require(int(h[0]) == 145 and int(h[1]) == 2048 and int(h[2]) == 1024,
            "Unrecognized T.dvg v0.10 model header; do not guess ABI")
    norb, nspin, su2 = int(h[25]), int(h[46]), int(h[45])
    nk = tuple(map(int, h[4:7]))
    nkf = tuple(map(int, h[7:10]))
    require(norb == 3 and nspin == 2 and su2 == 0,
            f"Unsupported model: norb={norb}, nspin={nspin}, SU2={su2}")
    require(nk[0] == nk[1] and nk[0] > 0 and nk[0] % 2 == 0 and nk[2] == 1,
            f"Unsupported model momentum mesh: {nk}")
    require(nkf[0] == nkf[1] and nkf[0] > 0 and nkf[2] == 1,
            f"Unsupported fine mesh: {nkf}")
    nff = int(h[50])
    require(nff >= norb, f"Invalid model form-factor count: {nff}")
    check_range(path, int(h[35]), 3 * norb * 8, "model orbital positions")
    positions = np.fromfile(path, dtype="<f8", count=3 * norb,
                            offset=int(h[35])).reshape(norb, 3)
    require(np.isfinite(positions).all(), "Invalid orbital positions")
    nhop, nvert = int(h[42]), int(h[47])
    require(nhop >= 0 and nvert >= 0, "Negative number of hopping/vertex records")
    require(int(h[44]) == nhop * int(h[108]), "Invalid hopping record stride/bytes")
    require(int(h[49]) == nvert * int(h[118]), "Invalid vertex record stride/bytes")
    check_range(path, int(h[43]), int(h[44]), "hopping entries")
    check_range(path, int(h[48]), int(h[49]), "interaction entries")
    return {"n_orb": norb, "n_spin": nspin, "SU2": su2, "nk": list(nk),
            "nkf": list(nkf), "nff": nff, "n_hopping": nhop,
            "n_interaction": nvert, "T_sha256": digest(path)}


def read_o(path: Path) -> dict[str, Any]:
    """Check all five serialized blocks and the reversal-closed FF geometry."""
    h = read_header(path, 256)
    norb, nspin, nq, nff = map(int, (h[1], h[2], h[3], h[7]))
    require(int(h[0]) == 84, "Unrecognized O.dvg v0.10 header")
    require(norb == 3 and nspin == 2 and int(h[6]) == 0,
            "O.dvg must be explicit-spin n_orb=3, n_spin=2, SU2=0")
    nk = int(round(nq ** 0.5))
    # Actual v0.10 C serialization in src/tu/postproduction.c (lines 647-650):
    #   h[3] = vertex->nk             (coarse q-grid)
    #   h[4] = vertex->nk*vertex->nkf (full fine-grid; NOT the coarse grid)
    #   h[5] = tu_data->nkibz         (irreducible coarse q-grid)
    # In production nk=24, nkf=12: h[3]=576, h[4]=82944, h[5]=576.
    nqfine, nqibz = int(h[4]), int(h[5])
    require(nk >= 2 and nk % 2 == 0 and nk * nk == nq and
            nqibz == nq and int(h[9]) == 0,
            f"O.dvg requires an even square, unreduced coarse q mesh; "
            f"observed nq={nq}, nqibz={nqibz}, nsym={int(h[9])}")
    require(nqfine >= nq and nqfine % nq == 0,
            f"O.dvg invalid fine momentum count: nqfine={nqfine}, nq={nq}")
    require(nff >= 3, "Invalid form factor count")
    blocks = {}
    expected = 16 * nff * nff * nq  # 2^4 spin components
    for name, j in O_BLOCKS.items():
        off, count = int(h[j]), int(h[j + 1])
        require(count == expected,
                f"{name} length mismatch: {count} != {expected}")
        check_range(path, off, 16 * count, name)
        blocks[name] = {"offset": off, "entries": count}
    # Form factors: ofrom[nff], oto[nff], R[nff,3]; read only ~5 kB.
    for offset_index, length_index, size, label in (
        (10, 11, nff, "FF from-orbital"),
        (12, 13, nff, "FF to-orbital"),
        (14, 15, 3 * nff, "FF translation"),
    ):
        require(int(h[length_index]) == size, f"Unexpected {label} length")
        check_range(path, int(h[offset_index]), 8 * size, label)
    ff_from = np.fromfile(path, dtype="<i8", count=nff, offset=int(h[10]))
    ff_to = np.fromfile(path, dtype="<i8", count=nff, offset=int(h[12]))
    ff_R = np.fromfile(path, dtype="<i8", count=3 * nff,
                       offset=int(h[14])).reshape(nff, 3)
    require(np.all((ff_from >= 0) & (ff_from < norb)) and
            np.all((ff_to >= 0) & (ff_to < norb)), "Out-of-range FF orbital")
    require(np.all(ff_R[:, 2] == 0), "Expected 2D FF translation vectors")
    ff_keys = [(tuple(map(int, ff_R[i])), int(ff_from[i]), int(ff_to[i]))
               for i in range(nff)]
    lookup = {key: i for i, key in enumerate(ff_keys)}
    require(len(lookup) == nff, "Duplicate FF tuples in O.dvg")
    reverse = []
    for R, src, dst in ff_keys:
        partner = (tuple(-v for v in R), dst, src)
        require(partner in lookup, f"Missing reverse FF for {(R, src, dst)}")
        reverse.append(lookup[partner])
    reverse = np.asarray(reverse, dtype=int)
    require(np.array_equal(reverse[reverse], np.arange(nff)),
            "FF reverse mapping is not an involution")
    half = nk // 2
    return {"n_orb": norb, "n_spin": nspin, "SU2": 0,
            "nk": nk, "nq": nq, "nqfine": nqfine, "nq_ibz": nqibz,
            "nff": nff,
            "O_size_bytes": path.stat().st_size,
            "blocks": blocks, "ff_reverse_closed": True,
            # Avoid M1/M2 naming disagreements between old runner and Stage 2.
            "q_indices": {"Gamma": 0, "M_a": half,
                          "M_b": half * nk, "M_c": half * nk + half}}


def inspect_case(case: Path, binary: Path | None = None,
                 require_soc: bool = False) -> dict[str, Any]:
    """Read-only case compatibility gate; no FRG iterations and no full O read."""
    case = case.resolve(strict=True)
    require(case.is_dir(), f"Not a case directory: {case}")
    T, O, meta = (case / name for name in ("T.dvg", "O.dvg", "metadata.json"))
    require(meta.is_file(), f"Missing historical metadata: {meta}")
    metadata = json.loads(meta.read_text(encoding="utf-8"))
    params = metadata.get("input_params")
    mesh = metadata.get("mesh")
    require(isinstance(params, dict) and isinstance(mesh, dict),
            "Historical metadata must specify input_params and mesh")
    require(float(params.get("lambda_rashba", 0.0)) == 0.0,
            "Rashba SOC is outside the Stage 3 intrinsic-KM diagnosis")
    if require_soc:
        require(abs(float(params.get("lambda_nn", 0.0))) +
                abs(float(params.get("lambda_nnn", 0.0))) > 0.0,
                "SOC was required, but selected case has neither NN nor NNN KM SOC")
    vt, vo = read_t(T), read_o(O)
    require(vt["nff"] == vo["nff"], "Mismatch: T and O form-factor counts")
    require(vt["nk"] == [vo["nk"], vo["nk"], 1],
            "Mismatch: T and O coarse momentum meshes")
    expected_nqfine = vo["nq"] * vt["nkf"][0] * vt["nkf"][1] * vt["nkf"][2]
    require(vo["nqfine"] == expected_nqfine,
            f"Mismatch: O.dvg fine momentum count {vo['nqfine']} != "
            f"T.dvg nk*nkf product {expected_nqfine}")
    require(int(mesh["nk"]) == vo["nk"] and int(mesh["nkf"]) == vt["nkf"][0],
            "Mismatch: archived mesh metadata and T/O headers")
    require("tu_distance" in mesh and float(mesh["tu_distance"]) > 0,
            "Missing/invalid TU distance; cannot safely rebuild FF basis")
    if "vertex_convention" in metadata:
        require(metadata["vertex_convention"] in ("official_s1minus1", "official", "s1minus1"),
                "This pipeline expects official D-channel s1=-1 density interactions")
    post = metadata.get("postprocess_report", {})
    if isinstance(post, dict) and "tu_channel_calc_project" in post:
        require(int(post["tu_channel_calc_project"]) == 0,
                "Expect archived raw P/C/D; tu_channel_calc_project must be 0")
    binary_hash = None
    if binary is not None:
        binary = binary.resolve(strict=True)
        binary_hash = digest(binary)
        require(binary_hash == VERIFIED_BINARY_SHA256,
                "libdivERGe.so checksum differs from audited production binary")
    return {"status": "ARCHIVE_PREFLIGHT_PASSED", "scope": "structural_only",
            "case_path": str(case), "parameters": params,
            "resolved_params": metadata.get("model_summary", {}).get("metadata", {}).get("resolved_params"),
            "mesh": mesh, "T": vt, "O": vo,
            "binary_sha256": binary_hash,
            "note": ("No physical-order diagnosis is performed by this module. "
                     "Full C-backed projection and Hermitian-source gap kernels still required.")}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True,
                        help="Directory containing T.dvg, O.dvg, metadata.json")
    parser.add_argument("--binary", type=Path, default=None,
                        help="Optional audited libdivERGe.so checksum gate")
    parser.add_argument("--require-soc", action="store_true")
    parser.add_argument("--json", type=Path, default=None,
                        help="Write small preflight report; never overwrites input files")
    args = parser.parse_args(argv)
    try:
        result = inspect_case(args.case, args.binary, args.require_soc)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        parser.exit(2, f"ARCHIVE_PREFLIGHT_FAILED: {exc}\n")
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "case": result["case_path"],
                      "nff": result["O"]["nff"], "nk": result["O"]["nk"],
                      "nkf": result["T"]["nkf"][0],
                      "q_indices": result["O"]["q_indices"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
