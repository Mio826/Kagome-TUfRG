#!/usr/bin/env python3
# coding: utf-8
"""
TUfRG_runner_snapshots.py
Version: snapshot-runner-v2plusV2-2026-05-21

A conservative wrapper/fork around TUfRG_runner.py that saves intermediate
postprocess snapshots during the flow.

Each snapshot is written as a mini case directory:

    CASE_DIR/snapshots/iter_XXXX/
        O.dvg
        T.dvg
        F.dat
        metadata.json

The existing diagnosis scripts can then run on any snapshot directory without
modification.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

import TUfRG_model_rashba as km
import TUfRG_runner_rashba as base


def parse_int_set(s: str | None) -> set[int]:
    if s is None or str(s).strip() == "":
        return set()
    out: set[int] = set()
    for x in str(s).split(','):
        x = x.strip()
        if x:
            out.add(int(x))
    return out


def copy_or_link(src: Path, dst: Path, *, prefer_link: bool = False) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if prefer_link:
        try:
            rel = os.path.relpath(src, dst.parent)
            os.symlink(rel, dst)
            return
        except Exception:
            pass
    shutil.copy2(src, dst)


def write_flow_rows(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    fieldnames = [
        "iter", "Lambda", "dLambda", "loop_pp", "loop_ph",
        "chan_P", "chan_C", "chan_D", "vertex_max", "euler_next",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def should_snapshot(iter_index: int, args: argparse.Namespace, explicit: set[int]) -> bool:
    # iter_index is the same as the F.dat iter column.  Periodic snapshots at
    # --snapshot-every 10 are taken at iter 9, 19, 29, ... after completed steps.
    if iter_index in explicit:
        return True
    every = int(args.snapshot_every)
    if every <= 0:
        return False
    completed = iter_index + 1
    if completed < int(args.snapshot_after):
        return False
    return completed % every == 0


def postprocess_to_snapshot(step, args: argparse.Namespace, case_dir: Path, Tfile: Path,
                            rows: Sequence[Dict[str, Any]], row: Dict[str, Any],
                            metadata_base: Dict[str, Any]) -> Dict[str, Any]:
    it = int(row["iter"])
    snap_dir = case_dir / str(args.snapshot_root) / f"iter_{it:04d}"
    snap_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[SNAPSHOT] iter={it} completed={it+1} -> {snap_dir}")

    Ofile = snap_dir / "O.dvg"
    post_report = base.postprocess(step, args, Ofile)
    copy_or_link(Tfile, snap_dir / "T.dvg", prefer_link=bool(args.snapshot_symlink_T))
    write_flow_rows(snap_dir / "F.dat", rows)

    md = dict(metadata_base)
    md["output_dir"] = str(snap_dir)
    md["parent_case_dir"] = str(case_dir)
    md["is_snapshot"] = True
    md["snapshot"] = {
        "iter": it,
        "completed_steps": it + 1,
        "Lambda": row.get("Lambda"),
        "dLambda": row.get("dLambda"),
        "vertex_max": row.get("vertex_max"),
        "chan_P": row.get("chan_P"),
        "chan_C": row.get("chan_C"),
        "chan_D": row.get("chan_D"),
        "loop_pp": row.get("loop_pp"),
        "loop_ph": row.get("loop_ph"),
    }
    md["files"] = {
        "T": str(snap_dir / "T.dvg"),
        "F": str(snap_dir / "F.dat"),
        "O": str(Ofile),
        "metadata": str(snap_dir / "metadata.json"),
    }
    md["postprocess_report"] = post_report
    (snap_dir / "metadata.json").write_text(json.dumps(base._json_safe(md), indent=2, sort_keys=True, default=str))
    return {"iter": it, "completed_steps": it + 1, "snapshot_dir": str(snap_dir), "O": str(Ofile), "vertex_max": row.get("vertex_max")}


def run_flow_with_snapshots(step, args: argparse.Namespace, case_dir: Path, Tfile: Path,
                            Ffile: Path, metadata_base: Dict[str, Any]) -> Dict[str, Any]:
    max_vert = np.zeros(1, dtype=np.float64)
    max_loop = np.zeros(2, dtype=np.float64)
    max_chan = np.zeros(3, dtype=np.float64)
    eu = base.configure_euler(args)
    explicit = parse_int_set(args.snapshot_at)

    rows: List[Dict[str, Any]] = []
    snaps: List[Dict[str, Any]] = []
    Ffile.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["iter", "Lambda", "dLambda", "loop_pp", "loop_ph", "chan_P", "chan_C", "chan_D", "vertex_max", "euler_next"]

    with Ffile.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        print("\nFlow history:")
        print("# iter Lambda dLambda loop_pp loop_ph chan_P chan_C chan_D vertex_max")
        for it in range(int(args.niter)):
            base.diverge.flow_step_euler(step, eu.Lambda, eu.dLambda)
            base.diverge.flow_step_vertmax(step, max_vert.ctypes.data)
            base.diverge.flow_step_loopmax(step, max_loop.ctypes.data)
            base.diverge.flow_step_chanmax(step, max_chan.ctypes.data)
            row: Dict[str, Any] = {
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
            print(f"{it:d} {row['Lambda']:.16e} {row['dLambda']:.16e} {row['loop_pp']:.16e} {row['loop_ph']:.16e} {row['chan_P']:.16e} {row['chan_C']:.16e} {row['chan_D']:.16e} {row['vertex_max']:.16e}")
            ok = bool(base.diverge.euler_next(base.diverge.byref(eu), max_vert[0]))
            row["euler_next"] = ok
            w.writerow(row)
            rows.append(row)
            f.flush()

            if should_snapshot(it, args, explicit):
                snaps.append(postprocess_to_snapshot(step, args, case_dir, Tfile, rows, row, metadata_base))
            if not ok:
                print("euler_next returned False; stopping flow.")
                break

    return {"flow_file": str(Ffile), "n_steps_completed": len(rows), "last_row": rows[-1] if rows else None, "snapshots": snaps}


def run_one_case(case_name: str, params: km.KMParameters, mesh: km.MeshParameters, args: argparse.Namespace) -> Dict[str, Any]:
    case_dir = Path(args.outdir).resolve() / case_name
    case_dir.mkdir(parents=True, exist_ok=True)
    Tfile, Ffile, Ofile, metadata_file = case_dir / "T.dvg", case_dir / "F.dat", case_dir / "O.dvg", case_dir / "metadata.json"

    print("\n" + "=" * 100)
    print(f"CASE {case_name}")
    print("=" * 100)
    print("Output dir:", case_dir)
    print("input params:", asdict(params))
    print("mesh:", asdict(mesh))
    print("vertex convention:", args.vertex_convention)
    print("q indices:", km.high_symmetry_q_indices(mesh.nk))
    print("snapshot config:", {"every": args.snapshot_every, "after": args.snapshot_after, "at": args.snapshot_at, "root": args.snapshot_root})

    model = None
    step = None
    metadata: Dict[str, Any] = {
        "case": case_name,
        "input_params": asdict(params),
        "vertex_convention": str(args.vertex_convention),
        "mesh": asdict(mesh),
        "output_dir": str(case_dir),
        "snapshot_config": {"snapshot_every": int(args.snapshot_every), "snapshot_after": int(args.snapshot_after), "snapshot_at": str(args.snapshot_at), "snapshot_root": str(args.snapshot_root)},
        "files": {"T": str(Tfile), "F": str(Ffile), "O": str(Ofile), "metadata": str(metadata_file)},
    }
    try:
        model = km.build_model(params=params, mesh=mesh, name=case_name.encode(), model_outfile=str(Tfile), call_tu=True,
                               validate_sanity=not bool(args.skip_sanity), vertex_convention=str(args.vertex_convention))
        summary = km.model_summary(model)
        metadata["model_summary"] = summary
        print("\nModel summary:")
        for k, v in summary.items():
            if k != "metadata":
                print(f"  {k}: {v}")
        md = summary.get("metadata", {})
        print("  hop_summary:", md.get("hop_summary"))
        print("  vertex_summary:", md.get("vertex_summary"))
        print("  filling_report:", md.get("filling_report"))
        print("  q_indices:", md.get("q_indices"))

        print("\nInitializing flow step...")
        step = base.diverge.flow_step_init(model, str(args.flow_mode), str(args.channels))
        metadata["flow_report"] = run_flow_with_snapshots(step, args, case_dir, Tfile, Ffile, metadata)

        if not args.no_postprocess:
            print("\nFinal postprocessing...")
            metadata["postprocess_report"] = base.postprocess(step, args, Ofile)
            print("wrote", Ofile)
            if args.snapshot_final:
                final_dir = case_dir / str(args.snapshot_root) / "final"
                final_dir.mkdir(parents=True, exist_ok=True)
                copy_or_link(Tfile, final_dir / "T.dvg", prefer_link=bool(args.snapshot_symlink_T))
                shutil.copy2(Ffile, final_dir / "F.dat")
                shutil.copy2(Ofile, final_dir / "O.dvg")
                final_md = dict(metadata)
                final_md["output_dir"] = str(final_dir)
                final_md["parent_case_dir"] = str(case_dir)
                final_md["is_snapshot"] = True
                final_md["snapshot"] = {"is_final_copy": True, "completed_steps": metadata["flow_report"].get("n_steps_completed")}
                final_md["files"] = {"T": str(final_dir / "T.dvg"), "F": str(final_dir / "F.dat"), "O": str(final_dir / "O.dvg"), "metadata": str(final_dir / "metadata.json")}
                (final_dir / "metadata.json").write_text(json.dumps(base._json_safe(final_md), indent=2, sort_keys=True, default=str))
                metadata["flow_report"].setdefault("snapshots", []).append({"snapshot_dir": str(final_dir), "is_final_copy": True})
        else:
            metadata["postprocess_report"] = {"skipped": True}
        metadata["status"] = "success"
        return metadata
    except Exception as exc:
        metadata["status"] = "failed"
        metadata["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        metadata_file.write_text(json.dumps(base._json_safe(metadata), indent=2, sort_keys=True, default=str))
        print("wrote", metadata_file)
        if step is not None:
            try: base.diverge.flow_step_free(step)
            except Exception as exc: print("WARNING: flow_step_free failed:", repr(exc))
        if model is not None:
            try: base.diverge.model_free(model)
            except Exception as exc: print("WARNING: model_free failed:", repr(exc))


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run spinful kagome Kane-Mele/Rashba TUFRG with O.dvg snapshots.")
    # Same common interface as TUfRG_runner.py.
    p.add_argument("--U", type=float, default=0.2)
    p.add_argument("--V", type=float, default=None, help="alias for --V1")
    p.add_argument("--V1", type=float, default=None)
    p.add_argument("--V2", type=float, default=0.0)
    p.add_argument("--t", type=float, default=1.0)
    p.add_argument("--mu", type=float, default=0.0)
    p.add_argument("--target-filling", type=float, default=5.0 / 12.0)
    p.add_argument("--filling-mode", choices=["fixed_mu", "fixed_filling"], default="fixed_mu")
    p.add_argument("--filling-ngrid", type=int, default=120)
    p.add_argument("--lambda-nn", type=float, default=0.0)
    p.add_argument("--lambda-nnn", type=float, default=0.0)
    p.add_argument("--lambda-rashba", type=float, default=0.0, help="NN Rashba SOC strength (spin-mixing, TR-preserving)")
    p.add_argument("--nk", type=int, default=4)
    p.add_argument("--nkf", type=int, default=1)
    p.add_argument("--tu-distance", type=float, default=1.01)
    p.add_argument("--niter", type=int, default=3)
    p.add_argument("--maxvert", type=float, default=50.0)
    p.add_argument("--Lambda", type=float, default=None)
    p.add_argument("--dLambda", type=float, default=None)
    p.add_argument("--dLambda-fac", type=float, default=0.1)
    p.add_argument("--dLambda-fac-scale", type=float, default=1.0)
    p.add_argument("--flow-mode", default="tu")
    p.add_argument("--channels", default="PCD")
    p.add_argument("--no-postprocess", action="store_true")
    p.add_argument("--tu-channel-calc-project", type=int, default=0)
    p.add_argument("--allow-projected-explicit", action="store_true")
    p.add_argument("--vertex-convention", choices=["official_s1minus1", "manual"], default="official_s1minus1")
    p.add_argument("--tu-n-decomp-values", type=int, default=16)
    p.add_argument("--tu-n-singular-values", type=int, default=16)
    p.add_argument("--tu-lingap-solver-mode", default="sss")
    p.add_argument("--outdir", default="km_spinful_runs")
    p.add_argument("--tag", default="run")
    p.add_argument("--case", action="append", default=None)
    p.add_argument("--skip-sanity", action="store_true")
    p.add_argument("--loglevel", type=int, default=1)
    # New snapshot options.
    p.add_argument("--snapshot-every", type=int, default=0, help="save O.dvg every N completed steps; 0 disables periodic snapshots")
    p.add_argument("--snapshot-after", type=int, default=1)
    p.add_argument("--snapshot-at", default="", help="comma-separated F.dat iter indices, e.g. 59,69,79")
    p.add_argument("--snapshot-root", default="snapshots")
    p.add_argument("--snapshot-final", action="store_true")
    p.add_argument("--snapshot-symlink-T", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if int(args.tu_channel_calc_project) != 0 and not bool(args.allow_projected_explicit):
        raise RuntimeError("Use --tu-channel-calc-project 0 for explicit-spin raw-channel diagnosis.")
    mesh = base.make_mesh(args)
    mesh.validate()
    cases = base.build_case_list(args)

    import diverge as _diverge
    base.diverge = _diverge
    base.diverge.init(None, None)
    try:
        try:
            base.diverge.mpi_loglevel_set(int(args.loglevel))
        except Exception as exc:
            print("WARNING: mpi_loglevel_set failed:", repr(exc))
        all_md: List[Dict[str, Any]] = []
        for name, U, V1, V2 in cases:
            params = base.make_params(args, U=U, V1=V1, V2=V2)
            params.validate()
            all_md.append(run_one_case(name, params, mesh, args))
        outroot = Path(args.outdir).resolve()
        outroot.mkdir(parents=True, exist_ok=True)
        summary_file = outroot / "run_summary_snapshots.json"
        summary_file.write_text(json.dumps(base._json_safe(all_md), indent=2, sort_keys=True, default=str))
        print("\nWrote snapshot run summary:", summary_file)
        return 0
    finally:
        try: base.diverge.finalize()
        except Exception as exc: print("WARNING: diverge.finalize failed:", repr(exc))


if __name__ == "__main__":
    raise SystemExit(main())
