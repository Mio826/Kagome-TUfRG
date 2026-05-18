# Spinful Kagome TUFRG project files

This is the clean active set for the explicit-spin kagome extended Hubbard model with optional NN/NNN Kane-Mele SOC.

## Core pipeline

1. `TUfRG_model.py`
   - Builds the explicit-spin kagome model.
   - Uses divERGe's official-style density vertices by default: D-channel with `s1=-1`.
   - Keeps `density_vertices_manual` only for bridge/debug comparisons.

2. `TUfRG_runner.py`
   - Runs the divERGe TUFRG flow and postprocess.
   - Default postprocess setting is `--tu-channel-calc-project 0` so explicit-spin output stores raw unprojected `P/C/D` channels.
   - Default vertex convention is `--vertex-convention official_s1minus1`.

3. `TUfRG_instability.py`
   - Level-1 diagnosis.
   - Reads `O.dvg`, constructs loop-dressed kernels `K(q) @ L(q)`, and saves right singular vectors.
   - Explicit-spin PH convention:
     `K_ph = D - C` plus crossed Pauli projection `(s0,s1;s2,s3) -> (s0,s3;s1,s2)`.
   - Hard-fails if explicit-spin data were postprocessed with nonzero `tu_channel_calc_project`.
   - Reports q=0 charge strength after removing onsite A1 and all onsite components.

4. `TUfRG_formfactor_from_modes.py`
   - Level-2 diagnosis.
   - Classifies loop-dressed modes from `instability_modes.npz` in TU form-factor space.
   - Allows A1-core modes with extended diagonal dressing instead of requiring purely onsite modes.

5. `TUfRG_realspace_from_modes.py`
   - Level-3 fingerprints.
   - Reconstructs site, bond, current, and pairing patterns from selected loop-dressed modes.
   - Use as a fingerprint / mean-field-like reconstruction, not as an independent susceptibility.

## Debug / calibration

`tests/inspect_bridge_q0_variants.py` is kept as a bridge-test/debug tool. It is not part of the daily production pipeline.

## Minimal usage

Run one case:

```bash
python -u TUfRG_runner.py \
  --U 1.5 --V 0.0 \
  --lambda-nn 0.0 --lambda-nnn 0.0 \
  --nk 12 --nkf 3 --tu-distance 1.01 \
  --niter 80 --maxvert 50 \
  --outdir runs --tag U1p5_V0p0 \
  --tu-channel-calc-project 0 \
  --vertex-convention official_s1minus1
```

Diagnose:

```bash
python -u TUfRG_instability.py runs/U1p5_V0p0 --out instability
python -u TUfRG_formfactor_from_modes.py runs/U1p5_V0p0 --instability-dir instability --out formfactor
```

Reconstruct a real-space fingerprint:

```bash
python -u TUfRG_realspace_from_modes.py runs/U1p5_V0p0 \
  --instability-dir instability \
  --family ph --sector ph-sz --q Gamma --mode 0 \
  --observable density bond current
```
