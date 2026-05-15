# Spinful Kagome Kane-Mele TUFRG Pipeline

This repository contains a small, explicit-spin TUFRG workflow for kagome extended Hubbard models with optional NN and NNN Kane-Mele spin-orbit coupling, using the `divERGe` package.

The current code is organized around a strict separation of responsibilities:

```text
model construction  ->  TUFRG flow  ->  Level-1 instability diagnosis
                     ->  Level-2 form-factor classification
                     ->  Level-3 real-space / current fingerprint reconstruction
```

The most important design rule is:

> Do not identify physical phases directly from raw `Pvec/Cvec/Dvec` or `Pval/Cval/Dval`.  The physical instability modes used downstream must come from the loop-dressed Level-1 diagnosis.

---

## 1. File overview

### `TUfRG_model.py`

Builds the explicit-spin kagome Kane-Mele extended Hubbard model.

Responsibilities:

- Defines kagome lattice vectors and orbital positions.
- Builds spinful real-space hoppings:
  - real NN hopping,
  - NN Kane-Mele SOC,
  - NNN Kane-Mele SOC,
  - optional onsite chemical potential term.
- Builds explicit-spin density-density interaction vertices for onsite `U` and NN `V`.
- Supports either fixed chemical potential or fixed filling.
- Initializes the `divERGe` model and TU form-factor basis.
- Provides one-body sanity checks: Hermiticity, time reversal, `S_z` conservation, spin block structure.

This file does **not** run the RG flow and does **not** diagnose instabilities.

---

### `TUfRG_runner.py`

Runs one or more TUFRG cases.

Responsibilities:

- Calls `TUfRG_model.build_model(...)`.
- Initializes the `divERGe` TU flow.
- Runs the Euler flow.
- Writes:

```text
T.dvg          # model output
F.dat          # flow history
O.dvg          # postprocessed TU output
metadata.json  # parameters and reproducibility metadata
```

This file deliberately does **not** perform physical diagnosis.  The runner only generates data.

---

### `TUfRG_instability.py`

Level-1 instability inspector.

This is the first file that decides the strength of physical instability sectors.

It reads `O.dvg` and constructs loop-dressed instability operators of the form

\[
K(q)L(q),
\]

where `K(q)` is a physical channel kernel and `L(q)` is the corresponding pp/ph loop.

For explicit-spin non-SU2 output, the calibrated PH convention is

\[
K_{\rm ph}^{\rm phys}(q)=D(q)-C(q),
\]

with crossed PH leg grouping

\[
(s_0,s_1;s_2,s_3)\rightarrow(s_0,s_3;s_1,s_2).
\]

The physical PH sectors are then obtained by Pauli projections:

\[
\rho=I/\sqrt{2},\quad
S^x=\sigma_x/\sqrt{2},\quad
S^y=\sigma_y/\sqrt{2},\quad
S^z=\sigma_z/\sqrt{2}.
\]

The code writes a calibration table `explicit_spin_ph_calibration.csv`.  In no-SOC explicit-spin runs, `Sx`, `Sy`, and `Sz` should be nearly degenerate.  This is the key regression test for the explicit-spin charge/spin construction.

Outputs:

```text
case_dir/instability_v2/instability_summary.csv
case_dir/instability_v2/instability_modes.npz
case_dir/instability_v2/explicit_spin_ph_calibration.csv
case_dir/instability_v2/instability_report.txt
```

The saved modes are loop-dressed right singular vectors in the TU form-factor basis.

---

### `TUfRG_formfactor_from_modes.py`

Level-2 form-factor classifier.

This file reads the loop-dressed modes from

```text
case_dir/instability_v2/instability_modes.npz
case_dir/instability_v2/instability_summary.csv
```

and classifies their TU form-factor content.

It reports quantities such as:

- onsite weight,
- NN-bond weight,
- longer-range weight,
- onsite `A1` versus `E` weight,
- real-bond proxy weight,
- imaginary-bond / current-like proxy weight,
- candidate phase labels such as FM-like, SDW-like, CDW-like, CBO-like, current-like, SC-like.

This file does **not** read raw `Pvec/Cvec/Dvec` or raw `Pval/Cval/Dval` in executable code.  It is downstream of Level 1.

Outputs:

```text
case_dir/formfactor_from_v2/formfactor_summary.csv
case_dir/formfactor_from_v2/formfactor_top_components.csv
```

---

### `TUfRG_realspace_from_modes.py`

Level-3 real-space / orbital-space fingerprint reconstruction.

This file reconstructs selected loop-dressed instability modes into site, bond, current, and pairing fingerprints.  It reads the Level-1 outputs:

```text
case_dir/instability_v2/instability_modes.npz
case_dir/instability_v2/instability_summary.csv
```

It does **not** read raw `Pvec/Cvec/Dvec`.

Typical outputs:

```text
selected_modes.csv
site_pattern.csv
bond_pattern.csv
current_pattern.csv
pairing_pattern.csv
pattern.png
```

Interpretation warning: Level-3 output is a normalized fingerprint of a possible ordered pattern.  It is not a self-consistent mean-field expectation value and not an independent susceptibility calculation.

---

### `run_smoke.sbatch`

PACE smoke-test script.

Use this first after editing files or changing environment settings.  Defaults are intentionally small:

```text
nk=4
nkf=1
niter=3
short walltime
small memory
```

It runs:

1. `TUfRG_runner.py`,
2. `TUfRG_instability.py`,
3. `TUfRG_formfactor_from_modes.py`,
4. optionally `TUfRG_realspace_from_modes.py`.

---

### `run.sbatch`

PACE production-style script.

Defaults are larger than the smoke test and intended for real parameter scans.  Parameters can be overridden from the terminal using `sbatch --export`.

---

## 2. Required environment

The scripts assume a working `divERGe` Python installation.  On Georgia Tech PACE, the current default environment used during development was:

```bash
module load anaconda3/2023.03
source activate /storage/scratch1/4/mli826/conda_envs/diverge-test
```

The sbatch files expose this through:

```bash
CONDA_ENV=/storage/scratch1/4/mli826/conda_envs/diverge-test
```

Override it if your environment path changes.

---

## 3. Recommended quick start

### 3.1 Run a smoke test on PACE

```bash
sbatch run_smoke.sbatch
```

Override parameters without editing the file:

```bash
sbatch --export=ALL,U=0.2,V=0.1,LAMBDA_NN=0.05,TAG=smoke_U0p2_V0p1 run_smoke.sbatch
```

After the job finishes, inspect:

```text
km_spinful_runs/<TAG>/F.dat
km_spinful_runs/<TAG>/instability_v2/instability_summary.csv
km_spinful_runs/<TAG>/instability_v2/explicit_spin_ph_calibration.csv
km_spinful_runs/<TAG>/formfactor_from_v2/formfactor_summary.csv
```

For a no-SOC explicit-spin test, check that `Sx`, `Sy`, and `Sz` in `explicit_spin_ph_calibration.csv` are nearly degenerate.

---

### 3.2 Run a production-style case

```bash
sbatch --export=ALL,U=1.0,V=0.5,LAMBDA_NN=0.05,NK=12,NKF=2,TAG=U1_V0p5_lnn005 run.sbatch
```

Common overrides:

```bash
U=1.0
V=0.5
LAMBDA_NN=0.05
LAMBDA_NNN=0.0
NK=12
NKF=2
TU_DISTANCE=1.01
NITER=80
MAXVERT=50.0
TAG=my_case_name
OUTDIR=km_spinful_runs
```

---

### 3.3 Scan a parameter line

Example: scan `V` at fixed `U` and SOC.

```bash
for V in 0.0 0.2 0.4 0.6 0.8 1.0
 do
   Vtag=${V/./p}
   sbatch --export=ALL,U=1.0,V=$V,LAMBDA_NN=0.05,TAG=scan_U1_V${Vtag}_l1_0p05 run.sbatch
 done
```

---

## 4. Manual command-line pipeline

The sbatch scripts automate the pipeline, but each step can also be run manually.

### 4.1 Run one TUFRG case

```bash
python -u TUfRG_runner.py \
  --U 1.0 \
  --V 0.5 \
  --lambda-nn 0.05 \
  --lambda-nnn 0.0 \
  --filling-mode fixed_mu \
  --mu 0.0 \
  --nk 12 \
  --nkf 2 \
  --tu-distance 1.01 \
  --niter 80 \
  --Lambda 50.0 \
  --dLambda -5.0 \
  --tag U1_V0p5_lnn005 \
  --outdir km_spinful_runs
```

This creates:

```text
km_spinful_runs/U1_V0p5_lnn005/
  T.dvg
  F.dat
  O.dvg
  metadata.json
```

---

### 4.2 Run Level-1 instability diagnosis

```bash
python -u TUfRG_instability.py km_spinful_runs/U1_V0p5_lnn005 \
  --q Gamma M1 M2 M3 \
  --n-modes 8 \
  --out instability_v2
```

Important output:

```text
instability_summary.csv
instability_modes.npz
explicit_spin_ph_calibration.csv
instability_report.txt
```

Use `instability_summary.csv` to rank loop-dressed physical sectors by `strength_svd_KL`.

---

### 4.3 Run Level-2 form-factor classification

```bash
python -u TUfRG_formfactor_from_modes.py km_spinful_runs/U1_V0p5_lnn005 \
  --instability-dir instability_v2 \
  --out formfactor_from_v2 \
  --top-components 12 \
  --max-rows 120
```

Important output:

```text
formfactor_summary.csv
formfactor_top_components.csv
```

Use `candidate_phase_label`, `onsite_weight`, `nn_weight`, `onsite_A1_weight_within_onsite`, and `imag_bond_proxy_weight` to classify the Level-1 modes.

---

### 4.4 Optional Level-3 real-space reconstruction

Example: reconstruct the leading `Gamma` spin-z mode.

```bash
python -u TUfRG_realspace_from_modes.py km_spinful_runs/U1_V0p5_lnn005 \
  --instability-dir instability_v2 \
  --family ph \
  --sector ph-sz \
  --q Gamma \
  --mode 0 \
  --observable density,bond,current \
  --supercell 2 2 \
  --out realspace_from_v2
```

Example: reconstruct a charge-current-like mode.

```bash
python -u TUfRG_realspace_from_modes.py km_spinful_runs/U1_V0p5_lnn005 \
  --instability-dir instability_v2 \
  --family ph \
  --sector ph-rho \
  --q Gamma \
  --mode 0 \
  --observable density,bond,current \
  --supercell 2 2 \
  --out realspace_from_v2
```

Example: reconstruct a 3Q finite-M combination.  The phases are manual and should be scanned or chosen according to the desired real-space pattern.

```bash
python -u TUfRG_realspace_from_modes.py km_spinful_runs/U1_V0p5_lnn005 \
  --instability-dir instability_v2 \
  --family ph \
  --sector ph-rho \
  --q M1 M2 M3 \
  --mode 0 \
  --theta 0 0 0 \
  --observable density,bond,current \
  --supercell 2 2 \
  --out realspace_from_v2
```

---

## 5. Output directory structure

A completed run looks like:

```text
km_spinful_runs/<TAG>/
  T.dvg
  F.dat
  O.dvg
  metadata.json

  instability_v2/
    instability_summary.csv
    instability_modes.npz
    explicit_spin_ph_calibration.csv
    instability_report.txt

  formfactor_from_v2/
    formfactor_summary.csv
    formfactor_top_components.csv

  realspace_from_v2/                 # optional
    <analysis_name>/
      selected_modes.csv
      site_pattern.csv
      bond_pattern.csv
      current_pattern.csv
      pairing_pattern.csv
      pattern.png
```

---

## 6. Physics conventions

### 6.1 PRR SU2-reduced convention

The PRR-style SU2-reduced basis is not physically spinless.  It uses SU(2) symmetry to analytically reduce spin indices.

For unprojected SU2 TU channels:

\[
K_{\rm SC}=P,
\]

\[
K_{\rm mag}=C,
\]

\[
K_{\rm charge}=2D-C.
\]

For projected SU2 output with `tu_channel_calc_project=-1`:

\[
K_{\rm charge}=D_{\rm saved},
\]

\[
K_{\rm mag}=-C_{\rm saved},
\]

\[
K_{\rm SC}=P_{\rm saved}.
\]

### 6.2 Explicit-spin convention used here

The Kane-Mele SOC model is explicit spin:

```text
SU2 = False
n_spin = 2
```

For PH sectors, the calibrated convention is:

\[
K_{\rm ph}^{\rm phys}=D-C.
\]

The spin slots are grouped as:

\[
(s_0,s_1;s_2,s_3)\rightarrow(s_0,s_3;s_1,s_2).
\]

Then project with normalized Pauli matrices:

\[
\rho=I/\sqrt{2},\quad
S^x=\sigma_x/\sqrt{2},\quad
S^y=\sigma_y/\sqrt{2},\quad
S^z=\sigma_z/\sqrt{2}.
\]

This convention was chosen because, in no-SOC explicit-spin runs, it restores near-degeneracy of `Sx`, `Sy`, and `Sz` at `Gamma`, `M1`, `M2`, and `M3`.  The old guess `2D-C` fails this calibration and should not be used for explicit-spin PH diagnosis.

For PP sectors, direct spin-pair projections are used:

\[
|S\rangle=(|ud\rangle-|du\rangle)/\sqrt{2},
\]

\[
|T_0\rangle=(|ud\rangle+|du\rangle)/\sqrt{2},
\]

\[
|T_+\rangle=|uu\rangle,
\qquad
|T_-\rangle=|dd\rangle.
\]

---

## 7. How to interpret the CSV files

### `instability_summary.csv`

Main columns:

```text
q_label
family                  # ph or pp
sector                  # ph-rho, ph-sx, ph-sy, ph-sz, pp-singlet, ...
convention
mode
strength_svd_KL         # Level-1 loop-dressed strength
mode_key                # key into instability_modes.npz
kernel_note
```

Sort by `strength_svd_KL` to identify leading loop-dressed sectors.

### `explicit_spin_ph_calibration.csv`

For no-SOC explicit-spin runs, check:

```text
sx ≈ sy ≈ sz
triplet_cv small
triplet_spread small
```

If this fails badly, do not trust explicit-spin PH interpretation for that run.

### `formfactor_summary.csv`

Main columns:

```text
candidate_phase_label
onsite_weight
nn_weight
long_weight
onsite_A1_weight_within_onsite
onsite_E_weight_within_onsite
real_bond_proxy_weight
imag_bond_proxy_weight
```

This is Level-2 classification.  It labels the form-factor content of Level-1 modes; it does not independently decide the instability strength.

### `current_pattern.csv`

For PH real-space reconstructions, current columns include:

```text
J_charge
J_charge_realhop
J_charge_sochop
J_spin_z
J_spin_z_realhop
J_spin_z_sochop
```

These are fingerprints of the selected Level-1 mode, not self-consistent expectation values.

---

## 8. Phase-label guide

| Physical tendency | Level-1 sector | Momentum | Level-2 signature |
|---|---|---|---|
| FM | `ph-sx/sy/sz` | `Gamma` | onsite, `A1` |
| SDW | `ph-sx/sy/sz` | `M1/M2/M3` | onsite spin density |
| SBO | `ph-sx/sy/sz` | `M1/M2/M3` | NN real bond weight |
| spin loop current | `ph-sz` or transverse spin sector | `Gamma` or `M` | NN imaginary/current-like weight |
| uniform charge / forward scattering | `ph-rho` | `Gamma` | onsite `A1` |
| PI / nematic | `ph-rho` | `Gamma` | nontrivial onsite/bond form factor after removing uniform `A1` |
| CDW | `ph-rho` | `M1/M2/M3` | onsite charge density |
| CBO | `ph-rho` | `M1/M2/M3` | NN real bond weight |
| charge loop current | `ph-rho` | `Gamma` or `M` | NN imaginary/current-like weight |
| SC | `pp-singlet/triplet...` | `Gamma` | pairing form factor |
| PDW | `pp-singlet/triplet...` | finite q | finite-momentum pairing |

---

## 9. Important caveats

1. `S_charge`, `S_mag`, and `S_sc` from divERGe are q=0 references only.  They cannot diagnose M-point CBO/SBO.

2. `Pvec/Cvec/Dvec` and `Pval/Cval/Dval` are saved decompositions of channel matrices.  They are not automatically bubble-dressed instability modes.

3. Level-1 uses `K(q) @ L(q)` singular values.  These are diagnostic strengths, not final thermodynamic order parameters.

4. Level-2 form-factor labels are classification aids.  Near-degenerate subspaces must be treated as subspaces, not as isolated single vectors.

5. Level-3 real-space reconstruction is a fingerprint.  It assumes the selected eigenmode direction is the linearized profile of an incipient order, but it does not solve a self-consistent ordered state.

6. Use even `NK` if you want exact `M1`, `M2`, and `M3` on the bosonic q mesh.

---

## 10. Filename consistency note

This README uses the cleaned filenames:

```text
TUfRG_instability.py
TUfRG_formfactor_from_modes.py
TUfRG_realspace_from_modes.py
run.sbatch
run_smoke.sbatch
```

If your local sbatch files still reference older names such as `TUfRG_instability_v2.py` or `TUfRG_formfactor_from_modes_v1.py`, either update those references or create compatibility symlinks, e.g.

```bash
ln -s TUfRG_instability.py TUfRG_instability_v2.py
ln -s TUfRG_formfactor_from_modes.py TUfRG_formfactor_from_modes_v1.py
ln -s TUfRG_realspace_from_modes.py TUfRG_realspace_from_modes_v1.py
```

For a clean repository, updating the sbatch references is preferred.

---

## 11. Minimal validation checklist before real scans

Before trusting a new environment or branch:

1. Run `sbatch run_smoke.sbatch`.
2. Confirm `O.dvg` exists.
3. Confirm `instability_summary.csv` exists.
4. For no-SOC explicit-spin runs, confirm `explicit_spin_ph_calibration.csv` has nearly degenerate `sx`, `sy`, `sz`.
5. Confirm `formfactor_summary.csv` is generated from `instability_modes.npz`.
6. Optionally run one Level-3 reconstruction for a known onsite FM-like mode.

Only after this should you launch a large parameter scan.
