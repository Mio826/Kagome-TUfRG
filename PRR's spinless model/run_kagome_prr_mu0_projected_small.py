#!/usr/bin/env python3
# coding: utf-8
"""
Small-scale PRR-style kagome Hubbard TUFRG run at strict mu=0, with projected physical q-resolved TU channels.

Purpose:
  - Not a new fRG solver: divERGe runs the TUFRG flow.
  - Controlled noninteracting setup for the NN kagome upper/p-type VHS:
        filling = 5/12, mu = 0, hopping amplitude = -t with t>0.
  - Save divERGe T.dvg/F.dat/O.dvg for each (U,V) point.
  - Optionally run q-resolved Gamma/M diagnosis after the flow.

Important:
  divERGe's built-in O.S_charge/O.S_mag/O.S_sc are q=0-only lingap outputs.
  M-point CBO/SBO are inspected from projected q-resolved P/C/D channels. With tu_channel_calc_project=-1, Dchannel is physical charge (2D-C), and Cchannel is sign-flipped magnetic.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import diverge

from kagome_prr_mu0_projected_common import (
    kagome_lattice_and_positions,
    kagome_nn_hopping_entries,
    kagome_uv_vertex_entries,
    vhs_sanity_report,
)


def build_kagome_model(
    *,
    U: float,
    V: float,
    t: float = 1.0,
    mu: float = 0.0,
    nk: int = 12,
    nkf: int = 3,
    tu_distance: float = 2.01,
    model_outfile: str = "T.dvg",
):
    """Build SU2-reduced kagome U-V model at fixed mu.

    The hopping amplitude is -t. The optional mu is included as onsite hopping
    -mu; for the target run it is exactly zero.
    """
    model = diverge.model_init()
    model.contents.name = b"kagome_prr_mu0_projected_small"
    model.contents.n_orb = 3
    model.contents.SU2 = True
    model.contents.n_spin = 1
    model.contents.nk[0] = int(nk)
    model.contents.nk[1] = int(nk)
    model.contents.nk[2] = 1
    model.contents.nkf[0] = int(nkf)
    model.contents.nkf[1] = int(nkf)
    model.contents.nkf[2] = 1

    lattice_np, pos_np = kagome_lattice_and_positions()
    lattice = diverge.view_array(model.contents.lattice, dtype=np.float64, shape=(3, 3))
    lattice[:, :] = lattice_np
    positions = diverge.view_array(model.contents.positions, dtype=np.float64, shape=(3, 3))
    positions[:, :] = pos_np

    hop_list = kagome_nn_hopping_entries(hop_amp=-float(t))
    if abs(mu) > 0:
        for o in range(3):
            hop_list.append(([0, 0, 0], o, o, 0, 0, (-float(mu), 0.0)))

    hoppings = diverge.alloc_array((len(hop_list),), dtype="rs_hopping_t")
    hoppings[:] = hop_list
    model.contents.hop = hoppings.ctypes.data
    model.contents.n_hop = hoppings.size
    diverge.mpi_py_eprint(f"using {hoppings.size} hopping elements; NN hop_amp={-float(t)}, mu={mu}")

    vert_list = kagome_uv_vertex_entries(U=U, V=V)
    vertices = diverge.alloc_array((len(vert_list),), dtype="rs_vertex_t")
    vertices[:] = vert_list
    model.contents.vert = vertices.ctypes.data
    model.contents.n_vert = vertices.size
    diverge.mpi_py_eprint(f"using {vertices.size} vertex elements; U={U}, V={V}")

    # High-symmetry path Gamma-M-K-Gamma in reduced coordinates, matching the
    # official kagome C test convention for M points.
    ibz_path = diverge.view_array(model.contents.ibz_path, dtype=np.float64, shape=(4, 3))
    ibz_path[:, :] = 0.0
    ibz_path[1, 0] = 0.5
    ibz_path[2, 0] = 0.5
    ibz_path[2, 1] = 0.5
    model.contents.n_ibz_path = 4
    model.contents.n_sym = 0

    if diverge.model_validate(model):
        raise RuntimeError("diverge.model_validate failed for kagome model")

    diverge.model_internals_common(model)

    # Intentionally do NOT call model_set_filling. For this PRR sanity run,
    # the noninteracting target is fixed by the Hamiltonian: mu=0.
    report = vhs_sanity_report(hop_amp=-float(t), mu=float(mu), ngrid=120)
    diverge.mpi_py_eprint("mu=0 VHS sanity report:")
    diverge.mpi_py_eprint(str(report))
    if abs(report["estimated_filling"] - 5.0 / 12.0) > 2e-2:
        diverge.mpi_py_eprint("WARNING: estimated filling is not close to 5/12. Check hopping sign/convention.")
    if report["max_abs_E_M_middle"] > 1e-8:
        diverge.mpi_py_eprint("WARNING: middle-band M energy is not numerically zero.")

    diverge.model_internals_tu(model, float(tu_distance))
    diverge.mpi_py_eprint(f"found {model.contents.n_tu_ff} unique TU form factors")

    checksum = diverge.model_to_file(model, model_outfile)
    diverge.mpi_py_eprint(f"wrote model to file {model_outfile} ({checksum})")
    return model


def run_flow(
    model,
    *,
    flow_outfile: str = "F.dat",
    mode: str = "tu",
    channels: str = "PCD",
    maxvert: float = 80.0,
    maxiter: int = 80,
):
    step = diverge.flow_step_init(model, mode, channels)
    maxs = dict(vert=np.zeros(1), loop=np.zeros(2), chan=np.zeros(3))

    eu = diverge.euler_defaults_CPP()
    eu.dLambda_fac_scale = 1.0
    eu.dLambda_fac = 0.1
    if hasattr(eu, "maxvert"):
        eu.maxvert = float(maxvert)
    if hasattr(eu, "vertmax"):
        eu.vertmax = float(maxvert)
    if hasattr(eu, "maxiter"):
        eu.maxiter = int(maxiter)

    stdout = sys.stdout
    sys.stdout = diverge.mpi_stdout_logger(flow_outfile)
    diverge.mpi_py_eprint(f"flow output to {flow_outfile}...")

    diverge.print("# Lambda dLambda Lp Lm P C D V")
    it = 0
    while True:
        diverge.flow_step_euler(step, eu.Lambda, eu.dLambda)
        diverge.flow_step_vertmax(step, maxs["vert"].ctypes.data)
        diverge.flow_step_loopmax(step, maxs["loop"].ctypes.data)
        diverge.flow_step_chanmax(step, maxs["chan"].ctypes.data)
        diverge.print("%.8e %.8e %.8e %.8e %.8e %.8e %.8e %.8e" % (
            eu.Lambda, eu.dLambda, *maxs["loop"], *maxs["chan"], *maxs["vert"]
        ))
        it += 1
        eu_next = diverge.euler_next(diverge.byref(eu), maxs["vert"][0])
        if not eu_next:
            diverge.mpi_py_eprint("divERGe euler_next requested stop")
            break
        if int(maxiter) > 0 and it >= int(maxiter):
            diverge.mpi_py_eprint(f"stopping after maxiter={maxiter}")
            break

    sys.stdout = stdout
    return step


def postprocess(step, *, post_outfile: str = "O.dvg"):
    cfg = diverge.postprocess_conf_defaults_CPP()
    # Critical for q-resolved PRR-style diagnosis:
    #   tu_channel_calc_project = -1 makes divERGe save projected physical channels.
    #   In SU(2): Dchannel becomes physical charge (2D-C), and Cchannel is sign-flipped.
    if hasattr(cfg, "tu_channel_calc_project"):
        cfg.tu_channel_calc_project = -1
    if hasattr(cfg, "tu_channels"):
        cfg.tu_channels = True
    if hasattr(cfg, "tu_lingap"):
        cfg.tu_lingap = True
    if hasattr(cfg, "tu_n_decomp_values"):
        cfg.tu_n_decomp_values = 8
    if hasattr(cfg, "tu_n_singular_values"):
        cfg.tu_n_singular_values = 8
    if hasattr(cfg, "tu_lingap_solver_mode"):
        cfg.tu_lingap_solver_mode = b"sss"
    diverge.postprocess_and_write_finegrained(step, post_outfile, cfg)
    diverge.mpi_py_eprint(f"wrote postprocess file {post_outfile}")


def parse_case(s: str) -> Tuple[str, float, float]:
    parts = s.split(":")
    if len(parts) != 3:
        raise ValueError(f"Bad case {s!r}; expected NAME:U:V")
    return parts[0], float(parts[1]), float(parts[2])


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", default="outputs/kagome_prr_mu0_projected_small")
    p.add_argument("--nk", type=int, default=12)
    p.add_argument("--nkf", type=int, default=3)
    p.add_argument("--tu-distance", type=float, default=2.01)
    p.add_argument("--maxvert", type=float, default=80.0)
    p.add_argument("--maxiter", type=int, default=80)
    p.add_argument("--t", type=float, default=1.0, help="positive t; hopping amplitude is -t")
    p.add_argument("--mu", type=float, default=0.0, help="chemical potential; PRR target uses mu=0")
    p.add_argument("--case", action="append", default=None, help="case as NAME:U:V; can be repeated")
    p.add_argument("--diagnose", action="store_true", help="run q-resolved diagnosis after all cases finish")
    args = p.parse_args(argv)

    default_cases = [
        "U0p5_V0p0:0.5:0.0",
        "U1p0_V0p0:1.0:0.0",
        "U1p5_V0p0:1.5:0.0",
        "U0p5_V0p5:0.5:0.5",
        "U0p5_V1p0:0.5:1.0",
        "U1p0_V1p0:1.0:1.0",
        "U0p0_V1p0:0.0:1.0",
        "U0p0_V1p5:0.0:1.5",
    ]
    cases = [parse_case(x) for x in (args.case if args.case else default_cases)]

    diverge.init(None, None)
    diverge.mpi_loglevel_set(1)
    diverge.compilation_status()

    outroot = Path(args.outdir).resolve()
    outroot.mkdir(parents=True, exist_ok=True)

    try:
        for name, U, V in cases:
            case_dir = outroot / name
            case_dir.mkdir(parents=True, exist_ok=True)
            print(f"\n========== CASE {name}: U={U}, V={V}, mu={args.mu} ==========")
            cwd = os.getcwd()
            os.chdir(case_dir)
            model = None
            step = None
            try:
                model = build_kagome_model(
                    U=U,
                    V=V,
                    t=args.t,
                    mu=args.mu,
                    nk=args.nk,
                    nkf=args.nkf,
                    tu_distance=args.tu_distance,
                    model_outfile="T.dvg",
                )
                step = run_flow(
                    model,
                    flow_outfile="F.dat",
                    mode="tu",
                    channels="PCD",
                    maxvert=args.maxvert,
                    maxiter=args.maxiter,
                )
                postprocess(step, post_outfile="O.dvg")
            finally:
                if step is not None:
                    diverge.flow_step_free(step)
                if model is not None:
                    diverge.model_free(model)
                os.chdir(cwd)
    finally:
        diverge.finalize()

    if args.diagnose:
        inspector = Path(__file__).resolve().with_name("inspect_kagome_prr_qresolved_projected.py")
        cmd = [sys.executable, str(inspector), str(outroot), "--vhs-check"]
        print("\nRunning diagnosis:", " ".join(cmd))
        subprocess.run(cmd, check=False)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
