# Kagome spinful LCO — minimal FRG-to-GL code

This directory contains the **minimal verified code** needed to reproduce the restricted
three-M fermionic Ginzburg–Landau analysis from a finished Stage-3/Stage-4B FRG diagnosis.
It does **not** rerun TUfRG, solve a self-consistent Hartree–Fock problem, or decide the
full interacting ground state.

## What to commit

| File | Purpose |
|---|---|
| `gl_gateB_reusable.py` | Main CLI: validate model/source labels and calculate quartic GL coefficients |
| `gl_gateA_build_W.py` | Full finite-M spinful gap vector → Hermitian 24×24 operators `W_a,b,c(K)`; symmetry checks |
| `gl_quartic.py` | Matsubara traces for `Z1,Z2` |
| `gl_thermo.py` | Solve chemical potential at the target normal-state filling |
| `gl_gateA_preflight.py` | Exact folded 2×2 normal-state Hamiltonian |
| `TUfRG_model.py`, `TUfRG_physics.py` | Source-backed kagome hoppings and original Hermitian TR-odd basis |
| `run_gl.sh` | Thin command-line wrapper; can auto-extract `ff_map.json` from a new diagnosis ZIP |
| `tests/test_core.py` | Smoke/regression checks |

Dependencies: Python >= 3.10; `numpy` (`pip install -r requirements.txt`).

## Physics convention

Using **full spinful FRG gap eigenvectors**, not physical current/response CSVs,

`H_MF(K) = H_0(K) + eta_a W_a(K) + eta_b W_b(K) + eta_c W_c(K)`.

For the restricted fermionic determinant, **per 2×2 supercell**,

`F4 = (Z1/2) sum_a eta_a^4 + Z2 sum_{a<b} eta_a^2 eta_b^2`.

`Z1 = T/(2 N_K) sum_{K,omega} tr[(G_0 W_a)^4]`.

`Z2 = T/N_K sum_{K,omega} tr[(G_0 W_a)^2(G_0 W_b)^2 + (G_0 W_a G_0 W_b)^2/2]`.

The analysis checks self/mixed coefficient consistency across the M star. A positive
`Z1-Z2` favors equal-amplitude triple M **only within the restricted quartic GL subspace**.
No HS quadratic cost or secondary-order relaxation is included. The GL integration mesh
`MESH` is unrelated to the original FRG `Nk`.

## Run on Nk=30 diagnosed data (FF map embedded in ZIP)

```bash
bash run_gl.sh /path/to/nk30_U3_V1p5_M_gap_vs_nk24.zip iter_final
```

Defaults: `T=0.12`, `MESH=12`, `NMAT=256` positive Matsubara frequencies,
output `results/iter_final_T0.12_N12.json`. Change them without editing Python:

```bash
T=0.06 MESH=24 NMAT=256 OUT=results/nk30_T006_N24.json \
  bash run_gl.sh /path/to/nk30_U3_V1p5_M_gap_vs_nk24.zip iter_final
```

## Run on old Nk=24 benchmark (FF map separate)

```bash
T=0.12 MESH=12 NMAT=128 OUT=results/nk24_baseline.json \
  bash run_gl.sh /path/to/stage4b_lco_U3p0_V1p5.zip iter_0074 /path/to/ff_map.json
```

Reference baseline: `Z1=0.933744875920`, `Z2=0.815713794933`, `Z1-Z2=0.118031080987`.
The Nk=30 `iter_final` archive also passes this solver's symmetry checks and at the
same T/GL integration settings yields `Z1-Z2=0.118037826211`. This is one benchmark
setting, not a full GL convergence study of the Nk=30 modes.

## Tests and validation

```bash
python -m unittest discover -s tests -v
```

Optionally rerun the full real-input baseline regression by setting environment variables:

```bash
LCO_GL_REFERENCE_ARCHIVE=/path/to/stage4b_lco_U3p0_V1p5.zip \
LCO_GL_REFERENCE_MAP=/path/to/ff_map.json \
python -m unittest discover -s tests -v
```

**Input contract:** The diagnosis ZIP must provide
`snapshots/<snapshot>/diagnosis.json` and `leading_modes.npz` for all M points;
`ff_map.json` must be extracted from the *matching* TU basis and should never be
substituted by another run's FF table without validation. Metadata are read from
`diagnosis.json` (SOC, hopping, filling). Nonzero Rashba is rejected.

Large `O.dvg`, raw FRG snapshots and diagnosis ZIPs should be kept outside Git;
these code files are sufficient once compact gap diagnosis archives are available.
For an exhaustive archival record, retain the original `LCO_GL_Handoff_2026-10-19.zip`.
