# Spinful Kagome Kane-Mele TUfRG Pipeline

This folder contains a four-file workflow for running and diagnosing a spinful kagome Kane-Mele extended Hubbard model with the divERGe TUfRG backend.

The intended workflow is:

```text
TUfRG_model.py      -> construct and sanity-check the model
TUfRG_runner.py     -> run divERGe TUfRG flow and postprocess
TUfRG_diagnosis.py  -> general q-resolved mode diagnosis
TUfRG_realspace.py  -> chosen-mode real-space pattern reconstruction
```

If you are using the versioned files generated during testing, rename them before use:

```bash
cp TUfRG_runner_v2.py TUfRG_runner.py
cp TUfRG_diagnosis_v2.py TUfRG_diagnosis.py
```

The recommended corrected versions have these markers inside the source:

```text
TUfRG_runner.py     Version: v2-jsonsafe-2026-05-12
TUfRG_diagnosis.py  Version: v2-sector-names-2026-05-12
```

---

## 1. Requirements

You need a Python environment where `diverge` is importable and usable. In Colab or PACE, check:

```python
import diverge
print(hasattr(diverge, "init"))
print(hasattr(diverge, "model_init"))
```

The scripts also use:

```text
numpy
pandas          # only for notebook inspection, not strictly required by scripts
matplotlib      # for TUfRG_realspace.py plotting
```

---

## 2. File responsibilities

### `TUfRG_model.py`

Purpose: model construction only.

It defines:

- kagome lattice geometry;
- spinful NN hopping and optional NN / NNN Kane-Mele SOC;
- explicit-spin onsite `U` and NN density `V` vertices;
- fixed chemical potential and fixed filling modes;
- pure-numpy sanity checks:
  \[
  H(k)=H^\dagger(k),\quad \Theta H(k)\Theta^{-1}=H(-k),\quad [H,S_z]=0.
  \]
- `build_model(...)`, which creates a divERGe model and writes `T.dvg`.

Important point: this TUfRG model is real-space/orbital based. It does **not** require Bloch eigenvector gauge fixing during model construction.

### `TUfRG_runner.py`

Purpose: run flow and postprocess.

It does:

```text
build_model -> flow_step_init -> Euler flow -> postprocess -> write metadata
```

Outputs per run:

```text
km_spinful_runs/<tag>/
    T.dvg
    F.dat
    O.dvg
    metadata.json
```

`metadata.json` records the resolved chemical potential, filling report, model summary, q indices, and flow settings.

### `TUfRG_diagnosis.py`

Purpose: general q-resolved diagnosis.

It reads a completed case directory and inspects the saved divERGe channel decompositions:

```text
Pval/Pvec/Poff/Plen
Cval/Cvec/Coff/Clen
Dval/Dvec/Doff/Dlen
mi_to_R, mi_to_ofrom, mi_to_oto
```

For each selected `(channel, q)` sector, it uses the correct q-resolved index:

\[
\text{global_loc}=Xoff[iq]+\text{mode}.
\]

It outputs:

```text
case_dir/diagnosis/
    q_summary.csv
    mode_summary.csv
    top_components.csv
    degeneracy_summary.csv
    diagnosis_report.txt
```

It keeps the first few modes in each fixed `(channel, q)` sector so that near-degenerate subspaces are visible. This is important for cases such as E-type superconducting channels or other multi-component irreps.

Caveat: `Pvec/Cvec/Dvec` are saved channel decompositions, not automatically bubble-dressed susceptibility eigenvectors.

### `TUfRG_realspace.py`

Purpose: selected-mode real-space reconstruction.

It should be used after `TUfRG_diagnosis.py` identifies a candidate mode.

It reconstructs real-space patterns for a specified:

```text
channel, sector, q-label(s), mode index/indices, observable
```

Supported sectors:

```text
ph-charge
ph-spin-z
pp-singlet
pp-triplet0
pp-triplet-plus
pp-triplet-minus
```

Supported observables:

```text
density,bond,current   # PH sectors
pairing                # PP sectors
```

Outputs:

```text
case_dir/realspace/<analysis_name>/
    site_pattern.csv
    bond_pattern.csv
    current_pattern.csv
    pairing_pattern.csv
    pattern_summary.csv
    template_overlap.csv
    pattern.png
```

The current reconstruction uses spin-dependent hopping from the model convention, so Kane-Mele SOC enters the current fingerprint.

---

## 3. Basic commands

### 3.1 Smoke-test runner: NN KM, fixed chemical potential

```bash
python TUfRG_runner.py \
  --U 0.2 \
  --V 0.0 \
  --lambda-nn 0.05 \
  --lambda-nnn 0.0 \
  --filling-mode fixed_mu \
  --mu 0.0 \
  --nk 4 \
  --nkf 1 \
  --tu-distance 1.01 \
  --niter 3 \
  --maxvert 50.0 \
  --tag sanity_runner_nnkm_fixed_mu
```

### 3.2 Smoke-test runner: NNN KM, fixed filling

```bash
python TUfRG_runner.py \
  --U 0.2 \
  --V 0.0 \
  --lambda-nn 0.0 \
  --lambda-nnn 0.05 \
  --filling-mode fixed_filling \
  --target-filling 0.4166666666666667 \
  --filling-ngrid 40 \
  --nk 4 \
  --nkf 1 \
  --tu-distance 1.01 \
  --niter 3 \
  --maxvert 50.0 \
  --tag sanity_runner_nnnkm_fixed_filling
```

### 3.3 General diagnosis

```bash
python TUfRG_diagnosis.py km_spinful_runs/sanity_runner_nnkm_fixed_mu \
  --q Gamma M1 M2 M3 \
  --channels P C D \
  --n-modes 4 \
  --top-components 8 \
  --degeneracy-tol 1e-3
```

### 3.4 Real-space reconstruction: single-Q PH mode

```bash
python TUfRG_realspace.py km_spinful_runs/sanity_runner_nnkm_fixed_mu \
  --channel D \
  --sector ph-charge \
  --q Gamma \
  --mode 0 \
  --observable density,bond,current \
  --combine singleQ \
  --supercell 2 2 \
  --template-overlap
```

### 3.5 Real-space reconstruction: 3Q PH pattern

```bash
python TUfRG_realspace.py km_spinful_runs/sanity_runner_nnkm_fixed_mu \
  --channel D \
  --sector ph-charge \
  --q M1 M2 M3 \
  --mode 0 0 0 \
  --observable density,bond,current \
  --combine 3Q \
  --theta 0 0 0 \
  --supercell 2 2 \
  --template-overlap
```

### 3.6 Real-space reconstruction: PP singlet pairing

```bash
python TUfRG_realspace.py km_spinful_runs/sanity_runner_nnkm_fixed_mu \
  --channel P \
  --sector pp-singlet \
  --q Gamma \
  --mode 0 \
  --observable pairing \
  --combine singleQ \
  --supercell 1 1
```

---

## 4. Meaning of key runner parameters

### `nk`

The coarse `nk x nk` momentum mesh for saved bosonic transfer momenta `q`. The number of q points is:

\[
N_q=nk^2.
\]

For kagome M points, use even `nk`, so that `(1/2,0)`, `(0,1/2)`, and `(1/2,1/2)` are on the mesh.

### `nkf`

Fine mesh factor for loop/bubble integration. Roughly, `nk` controls the saved q-channel mesh, while `nkf` improves internal loop resolution.

### `tu-distance`

Real-space cutoff for TU form factors. It controls how many form-factor components `n_tu_ff` are retained. Larger values include longer-range bond/pair structures but are more expensive.

### `niter`

Maximum number of Euler flow iterations.

### `maxvert`

Vertex threshold passed to the divERGe Euler controller. The flow can stop when the vertex scale becomes large or when `euler_next(...)` returns false.

### `tag`

Name of the output case folder:

```text
km_spinful_runs/<tag>/
```

---

## 5. Interpretation warnings

1. The scripts are designed to distinguish three layers:

   ```text
   flow/postprocess data -> general q-resolved diagnosis -> real-space pattern reconstruction
   ```

2. `Pval/Cval/Dval` and `Pvec/Cvec/Dvec` are saved channel decompositions. Do not interpret them automatically as fully bubble-dressed instability eigenmodes.

3. `S_sc`, `S_mag`, and `S_charge` from divERGe are q=0 references only. They should not be used to diagnose M-point CBO/SBO/CDW directly.

4. In explicit-spin non-SU2 runs, the scripts treat `P/C/D` as saved divERGe channels. They do not blindly relabel `C` as magnetic or `D` as charge. The physical sector is diagnosed from the spin tensor.

5. Real-space pattern reconstruction is a fingerprint of a form-factor direction, not a self-consistent mean-field expectation value.

6. Template overlap in `TUfRG_realspace.py` is currently limited and prototype-level. It should be trusted only together with the raw CSV and the plotted pattern.

---

## 6. Practical workflow

1. Run one or more cases with `TUfRG_runner.py`.
2. Run `TUfRG_diagnosis.py` to inspect the first few modes in each `(channel, q)` sector.
3. Open `diagnosis/mode_summary.csv` and look for modes with large `nn_weight`, `onsite_weight`, or relevant sector character.
4. Use `TUfRG_realspace.py` on selected modes only.
5. Do not over-interpret tiny norms. If the real-space current/bond norm is around `1e-14`, template overlaps can be numerically misleading.

---

## 7. Recommended next physical runs after smoke tests

For a slightly more meaningful debug scan:

```bash
python TUfRG_runner.py \
  --lambda-nn 0.05 \
  --filling-mode fixed_mu \
  --mu 0.0 \
  --nk 4 \
  --nkf 1 \
  --tu-distance 1.01 \
  --niter 20 \
  --maxvert 20.0 \
  --case Uonly:1.0:0.0 \
  --case Vonly:0.0:1.0 \
  --case mixed:0.5:1.0
```

Then run recursive diagnosis:

```bash
python TUfRG_diagnosis.py km_spinful_runs --recursive \
  --q Gamma M1 M2 M3 \
  --channels P C D \
  --n-modes 8 \
  --top-components 12
```
