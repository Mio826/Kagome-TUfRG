#!/usr/bin/env python3
# coding: utf-8
"""
TUfRG_runner.py
Version: v2-jsonsafe-2026-05-12

Official small-run / benchmark runner for the spinful kagome Kane-Mele extended
Hubbard model built by TUfRG_model.py.

Scope
-----
This script runs divERGe TUFRG for one or more parameter points:
    1. create a spinful kagome model through TUfRG_model.build_model;
    2. run a TU flow in P/C/D channels;
    3. save flow history to F.dat;
    4. optionally postprocess and save O.dvg;
    5. save metadata.json for reproducibility.

It deliberately does NOT perform physical diagnosis of the output.  Diagnosis is
kept in a separate inspector script.  This separation avoids mixing model/flow
choices with q-resolved channel and form-factor interpretation.

Important convention notes
--------------------------
* The model is explicit spin: SU2=False, n_spin=2.
* Fixed-mu and fixed-filling modes are both delegated to TUfRG_model.py.
* The bosonic q mesh has nk*nk points.  M-point q indices come from nk, not nk*nkf.
* divERGe built-in S_* postprocess outputs are q=0 only.  M-point analysis must
  use q-resolved P/C/D output later in the inspector.
* For spinful non-SU2 runs used by our explicit-spin diagnosis, always save
  RAW unprojected P/C/D channels: --tu-channel-calc-project 0.  Projected
  saved channels are incompatible with the explicit-spin Pauli projection in
  TUfRG_instability_rawguard.py.
* Density interactions are built by the official divERGe D-channel s1=-1
  convention by default.  The legacy manual spin enumeration can be selected
  only for bridge tests.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

import TUfRG_model as km

# Loaded in main(); kept global so helper functions can use the same module.
diverge = None


def _json_safe(obj: Any) -> Any:
    """Convert numpy/bytes/ctypes/divERGe objects into JSON-serializable objects."""
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, bytes):
        try:
            return obj.decode(errors="replace")
        except Exception:
            return repr(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return _json_safe(obj.tolist())
    if isinstance(obj, dict):
        return {str(_json_safe(k)): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    if hasattr(obj, "tolist"):
        try:
            return _json_safe(obj.tolist())
        except Exception:
            pass
    if hasattr(obj, "item"):
        try:
            return _json_safe(obj.item())
        except Exception:
            pass
    # Last resort for ctypes/custom divERGe objects, e.g. class name ``String``.
    return str(obj)


def parse_case(case: str) -> Tuple[str, float, float]:
    """Parse a case string NAME:U:V."""
    parts = case.split(":")
    if len(parts) != 3:
        raise ValueError(f"Bad --case {case!r}; expected NAME:U:V")
    name, U, V = parts
    if not name:
        raise ValueError("case name must be non-empty")
    return name, float(U), float(V)


def make_params(args: argparse.Namespace, *, U: float, V: float) -> km.KMParameters:
    return km.KMParameters(
        t=float(args.t),
        mu=float(args.mu),
        target_filling=float(args.target_filling),
        filling_mode=str(args.filling_mode),
        lambda_nn=float(args.lambda_nn),
        lambda_nnn=float(args.lambda_nnn),
        U=float(U),
        V=float(V),
        filling_ngrid=int(args.filling_ngrid),
    )


def make_mesh(args: argparse.Namespace) -> km.MeshParameters:
    return km.MeshParameters(
        nk=int(args.nk),
        nkf=int(args.nkf),
        tu_distance=float(args.tu_distance),
    )


def configure_euler(args: argparse.Namespace):
    """Create and configure divERGe Euler parameters."""
    eu = diverge.euler_defaults_CPP()

    if args.Lambda is not None:
        eu.Lambda = float(args.Lambda)
    if args.dLambda is not None:
        eu.dLambda = float(args.dLambda)

    if hasattr(eu, "dLambda_fac_scale"):
        eu.dLambda_fac_scale = float(args.dLambda_fac_scale)
    if hasattr(eu, "dLambda_fac"):
        eu.dLambda_fac = float(args.dLambda_fac)
    if hasattr(eu, "maxvert"):
        eu.maxvert = float(args.maxvert)
    if hasattr(eu, "vertmax"):
        eu.vertmax = float(args.maxvert)
    if hasattr(eu, "maxiter"):
        # We keep our own loop counter, but setting this helps if the backend uses it.
        eu.maxiter = int(args.niter)

    return eu


def run_flow(step, args: argparse.Namespace, flow_file: Path) -> Dict[str, Any]:
    """Run the TUFRG Euler flow and write a plain CSV flow history."""
    max_vert = np.zeros(1, dtype=np.float64)
    max_loop = np.zeros(2, dtype=np.float64)
    max_chan = np.zeros(3, dtype=np.float64)
    eu = configure_euler(args)

    rows: List[Dict[str, float | int | bool]] = []
    flow_file.parent.mkdir(parents=True, exist_ok=True)

    with flow_file.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "iter", "Lambda", "dLambda",
                "loop_pp", "loop_ph",
                "chan_P", "chan_C", "chan_D",
                "vertex_max", "euler_next",
            ],
        )
        writer.writeheader()

        print("\nFlow history:")
        print("# iter Lambda dLambda loop_pp loop_ph chan_P chan_C chan_D vertex_max")

        for it in range(int(args.niter)):
            diverge.flow_step_euler(step, eu.Lambda, eu.dLambda)
            diverge.flow_step_vertmax(step, max_vert.ctypes.data)
            diverge.flow_step_loopmax(step, max_loop.ctypes.data)
            diverge.flow_step_chanmax(step, max_chan.ctypes.data)

            row: Dict[str, float | int | bool] = {
                "iter": int(it),
                "Lambda": float(eu.Lambda),
                "dLambda": float(eu.dLambda),
                "loop_pp": float(max_loop[0]),
                "loop_ph": float(max_loop[1]),
                "chan_P": float(max_chan[0]),
                "chan_C": float(max_chan[1]),
                "chan_D": float(max_chan[2]),
                "vertex_max": float(max_vert[0]),
                "euler_next": True,
            }
            print(
                f"{it:d} {row['Lambda']:.16e} {row['dLambda']:.16e} "
                f"{row['loop_pp']:.16e} {row['loop_ph']:.16e} "
                f"{row['chan_P']:.16e} {row['chan_C']:.16e} {row['chan_D']:.16e} "
                f"{row['vertex_max']:.16e}"
            )

            ok = bool(diverge.euler_next(diverge.byref(eu), max_vert[0]))
            row["euler_next"] = ok
            writer.writerow(row)
            rows.append(row)
            f.flush()

            if not ok:
                print("euler_next returned False; stopping flow.")
                break

    return {
        "flow_file": str(flow_file),
        "n_steps_completed": len(rows),
        "last_row": rows[-1] if rows else None,
    }


def postprocess(step, args: argparse.Namespace, output_file: Path) -> Dict[str, Any]:
    """Run divERGe finegrained postprocess and save O.dvg."""
    cfg = diverge.postprocess_conf_defaults_CPP()

    if hasattr(cfg, "tu_channel_calc_project"):
        cfg.tu_channel_calc_project = int(args.tu_channel_calc_project)
    if hasattr(cfg, "tu_channels"):
        cfg.tu_channels = True
    if hasattr(cfg, "tu_lingap"):
        cfg.tu_lingap = True
    if hasattr(cfg, "tu_n_decomp_values"):
        cfg.tu_n_decomp_values = int(args.tu_n_decomp_values)
    if hasattr(cfg, "tu_n_singular_values"):
        cfg.tu_n_singular_values = int(args.tu_n_singular_values)
    if hasattr(cfg, "tu_lingap_solver_mode"):
        cfg.tu_lingap_solver_mode = str(args.tu_lingap_solver_mode).encode()

    output_file.parent.mkdir(parents=True, exist_ok=True)
    diverge.postprocess_and_write_finegrained(step, str(output_file), cfg)
    return {
        "postprocess_file": str(output_file),
        "tu_channel_calc_project": int(args.tu_channel_calc_project),
        "tu_n_decomp_values": int(args.tu_n_decomp_values),
        "tu_n_singular_values": int(args.tu_n_singular_values),
        "tu_lingap_solver_mode": str(args.tu_lingap_solver_mode),
    }


def run_one_case(case_name: str, params: km.KMParameters, mesh: km.MeshParameters, args: argparse.Namespace) -> Dict[str, Any]:
    """Build, run, postprocess, and clean up one parameter point."""
    case_dir = Path(args.outdir).resolve() / case_name
    case_dir.mkdir(parents=True, exist_ok=True)

    Tfile = case_dir / "T.dvg"
    Ffile = case_dir / "F.dat"
    Ofile = case_dir / "O.dvg"
    metadata_file = case_dir / "metadata.json"

    print("\n" + "=" * 100)
    print(f"CASE {case_name}")
    print("=" * 100)
    print("Output dir:", case_dir)
    print("input params:", asdict(params))
    print("mesh:", asdict(mesh))
    print("vertex convention:", args.vertex_convention)
    print("q indices:", km.high_symmetry_q_indices(mesh.nk))

    model = None
    step = None
    metadata: Dict[str, Any] = {
        "case": case_name,
        "input_params": asdict(params),
        "vertex_convention": str(args.vertex_convention),
        "mesh": asdict(mesh),
        "output_dir": str(case_dir),
        "files": {"T": str(Tfile), "F": str(Ffile), "O": str(Ofile), "metadata": str(metadata_file)},
    }

    try:
        model = km.build_model(
            params=params,
            mesh=mesh,
            name=case_name.encode(),
            model_outfile=str(Tfile),
            call_tu=True,
            validate_sanity=not bool(args.skip_sanity),
            vertex_convention=str(args.vertex_convention),
        )
        summary = km.model_summary(model)
        metadata["model_summary"] = summary

        print("\nModel summary:")
        for k, v in summary.items():
            if k == "metadata":
                continue
            print(f"  {k}: {v}")
        md = summary.get("metadata", {})
        print("  hop_summary:", md.get("hop_summary"))
        print("  vertex_summary:", md.get("vertex_summary"))
        print("  filling_report:", md.get("filling_report"))
        print("  q_indices:", md.get("q_indices"))

        print("\nInitializing flow step...")
        step = diverge.flow_step_init(model, str(args.flow_mode), str(args.channels))
        flow_report = run_flow(step, args, Ffile)
        metadata["flow_report"] = flow_report

        if not args.no_postprocess:
            print("\nPostprocessing...")
            post_report = postprocess(step, args, Ofile)
            metadata["postprocess_report"] = post_report
            print("wrote", Ofile)
        else:
            metadata["postprocess_report"] = {"skipped": True}
            print("Postprocess skipped by --no-postprocess")

        metadata["status"] = "success"
        return metadata

    except Exception as exc:
        metadata["status"] = "failed"
        metadata["error"] = f"{type(exc).__name__}: {exc}"
        raise

    finally:
        # Always write metadata, even on failure.
        metadata_file.write_text(json.dumps(_json_safe(metadata), indent=2, sort_keys=True, default=str))
        print("wrote", metadata_file)

        if step is not None:
            try:
                diverge.flow_step_free(step)
            except Exception as exc:
                print("WARNING: flow_step_free failed:", repr(exc))
        if model is not None:
            try:
                diverge.model_free(model)
            except Exception as exc:
                print("WARNING: model_free failed:", repr(exc))


def build_case_list(args: argparse.Namespace) -> List[Tuple[str, float, float]]:
    if args.case:
        return [parse_case(x) for x in args.case]
    return [(str(args.tag), float(args.U), float(args.V))]


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run spinful kagome Kane-Mele TUFRG small/benchmark cases.")

    # Physical parameters.
    p.add_argument("--U", type=float, default=0.2, help="onsite Hubbard U for single-case mode")
    p.add_argument("--V", type=float, default=0.0, help="NN density interaction V for single-case mode")
    p.add_argument("--t", type=float, default=1.0, help="positive t; real NN hopping is -t")
    p.add_argument("--mu", type=float, default=0.0, help="chemical potential for fixed_mu mode")
    p.add_argument("--target-filling", type=float, default=5.0 / 12.0, help="target filling fraction for fixed_filling mode")
    p.add_argument("--filling-mode", choices=["fixed_mu", "fixed_filling"], default="fixed_mu")
    p.add_argument("--filling-ngrid", type=int, default=120, help="k grid for resolving/estimating filling")
    p.add_argument("--lambda-nn", type=float, default=0.0, help="NN Kane-Mele SOC strength")
    p.add_argument("--lambda-nnn", type=float, default=0.0, help="NNN Kane-Mele SOC strength")

    # Mesh / TU basis.
    p.add_argument("--nk", type=int, default=4, help="coarse nk x nk q/channel momentum mesh; use even nk")
    p.add_argument("--nkf", type=int, default=1, help="fine loop mesh factor")
    p.add_argument("--tu-distance", type=float, default=1.01, help="TU form-factor real-space cutoff distance")

    # Flow controls.
    p.add_argument("--niter", type=int, default=3, help="maximum number of Euler flow iterations")
    p.add_argument("--maxvert", type=float, default=50.0, help="vertex stopping scale")
    p.add_argument("--Lambda", type=float, default=None, help="optional initial Lambda override")
    p.add_argument("--dLambda", type=float, default=None, help="optional initial dLambda override")
    p.add_argument("--dLambda-fac", type=float, default=0.1)
    p.add_argument("--dLambda-fac-scale", type=float, default=1.0)
    p.add_argument("--flow-mode", default="tu", help="divERGe flow mode, normally 'tu'")
    p.add_argument("--channels", default="PCD", help="channels passed to flow_step_init, normally PCD")

    # Postprocess controls.
    p.add_argument("--no-postprocess", action="store_true")
    p.add_argument("--tu-channel-calc-project", type=int, default=0,
                   help="postprocess tu_channel_calc_project; must be 0 for explicit-spin raw-channel diagnosis")
    p.add_argument("--allow-projected-explicit", action="store_true",
                   help="allow nonzero tu_channel_calc_project despite explicit-spin diagnosis incompatibility")
    p.add_argument("--vertex-convention", choices=["official_s1minus1", "manual"], default="official_s1minus1",
                   help="interaction vertex convention; official_s1minus1 uses divERGe D-channel s1=-1")
    p.add_argument("--tu-n-decomp-values", type=int, default=16)
    p.add_argument("--tu-n-singular-values", type=int, default=16)
    p.add_argument("--tu-lingap-solver-mode", default="sss")

    # Output/cases.
    p.add_argument("--outdir", default="km_spinful_runs")
    p.add_argument("--tag", default="run", help="case name for single-case mode")
    p.add_argument("--case", action="append", default=None,
                   help="batch case NAME:U:V; can be repeated. Uses common t/mu/SOC/mesh/flow settings.")

    # Runtime.
    p.add_argument("--skip-sanity", action="store_true", help="skip pure-numpy model sanity check before building")
    p.add_argument("--loglevel", type=int, default=1, help="divERGe mpi_loglevel_set value")

    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    global diverge
    args = parse_args(argv)
    if int(args.tu_channel_calc_project) != 0 and not bool(args.allow_projected_explicit):
        raise RuntimeError(
            "This explicit-spin runner is meant for raw-channel diagnosis. "
            "Use --tu-channel-calc-project 0, or pass --allow-projected-explicit for debugging only."
        )
    mesh = make_mesh(args)
    mesh.validate()

    cases = build_case_list(args)

    import diverge as _diverge
    diverge = _diverge

    diverge.init(None, None)
    try:
        try:
            diverge.mpi_loglevel_set(int(args.loglevel))
        except Exception as exc:
            print("WARNING: mpi_loglevel_set failed:", repr(exc))

        all_metadata: List[Dict[str, Any]] = []
        for name, U, V in cases:
            params = make_params(args, U=U, V=V)
            params.validate()
            md = run_one_case(name, params, mesh, args)
            all_metadata.append(md)

        outroot = Path(args.outdir).resolve()
        outroot.mkdir(parents=True, exist_ok=True)
        summary_file = outroot / "run_summary.json"
        summary_file.write_text(json.dumps(_json_safe(all_metadata), indent=2, sort_keys=True, default=str))
        print("\nWrote run summary:", summary_file)
        return 0

    finally:
        try:
            diverge.finalize()
        except Exception as exc:
            print("WARNING: diverge.finalize failed:", repr(exc))


if __name__ == "__main__":
    raise SystemExit(main())
