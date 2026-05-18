#!/bin/bash
#SBATCH --job-name=bridge-diag-s1m1
#SBATCH --account=gts-ikimchi3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=08:00:00
#SBATCH --qos=inferno
#SBATCH --output=logs/bridge-diag-s1m1-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=miaoli@gatech.edu

set -euo pipefail

module load anaconda3
conda activate /storage/scratch1/4/mli826/conda_envs/diverge-test

cd /storage/scratch1/4/mli826/divERGe_test_run/runs
mkdir -p logs

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

OUT=${OUT:-bridge_spinful_prr_raw_s1minus1}
NMODES=${NMODES:-4}
TOP_COMPONENTS=${TOP_COMPONENTS:-12}
MAX_ROWS=${MAX_ROWS:-160}

python - <<'PY'
import sys, importlib.util
print('python =', sys.executable)
for m in ['numpy','pandas','diverge']:
    print(m, 'found =', importlib.util.find_spec(m) is not None)
if importlib.util.find_spec('pandas') is None:
    raise SystemExit('pandas is missing. Run: python -m pip install pandas   in the diverge-test env, then resubmit.')
PY

for CASE in U1p5_V0p0 U0p0_V1p5
 do
  CASEDIR="$OUT/$CASE"
  echo "============================================================"
  echo "CASE: $CASE"
  echo "CASEDIR: $CASEDIR"
  echo "============================================================"

  test -f "$CASEDIR/O.dvg"
  test -f "$CASEDIR/T.dvg"
  test -f "$CASEDIR/metadata.json"

  echo "------------------------------------------------------------"
  echo "Level-1 instability: $CASE"
  echo "------------------------------------------------------------"
  srun -n 1 python -u TUfRG_instability_rawguard.py "$CASEDIR" \
    --q Gamma M1 M2 M3 \
    --n-modes "$NMODES" \
    --out instability

  echo "------------------------------------------------------------"
  echo "Level-2 form factor: $CASE"
  echo "------------------------------------------------------------"
  srun -n 1 python -u TUfRG_formfactor_from_modes.py "$CASEDIR" \
    --instability-dir instability \
    --out formfactor \
    --top-components "$TOP_COMPONENTS" \
    --max-rows "$MAX_ROWS"

  echo "------------------------------------------------------------"
  echo "Quick summary files for $CASE"
  echo "------------------------------------------------------------"
  ls -lh "$CASEDIR/instability" "$CASEDIR/formfactor" || true
  echo
  echo "instability_summary.csv head:"
  head -20 "$CASEDIR/instability/instability_summary.csv" || true
  echo
  echo "formfactor_summary.csv head:"
  head -20 "$CASEDIR/formfactor/formfactor_summary.csv" || true
 done

python - <<'PY'
from pathlib import Path
import pandas as pd
import numpy as np

OUT = Path('bridge_spinful_prr_raw_s1minus1')
cases = ['U1p5_V0p0', 'U0p0_V1p5']
rows = []

def top_strength(inst, q_label, family=None, sector=None):
    df = inst[inst['q_label'].astype(str).eq(q_label)].copy()
    if family is not None:
        df = df[df['family'].astype(str).eq(family)]
    if sector is not None:
        df = df[df['sector'].astype(str).eq(sector)]
    if len(df) == 0:
        return np.nan
    col = 'strength_svd_KL' if 'strength_svd_KL' in df.columns else 'strength'
    return float(df.sort_values(col, ascending=False).iloc[0][col])

def best_ff(ff, q_label, family=None, sector=None):
    df = ff[ff['q_label'].astype(str).eq(q_label)].copy()
    if family is not None:
        df = df[df['family'].astype(str).eq(family)]
    if sector is not None:
        df = df[df['sector'].astype(str).eq(sector)]
    if len(df) == 0:
        return None
    col = 'strength_svd_KL' if 'strength_svd_KL' in df.columns else ('strength' if 'strength' in df.columns else df.columns[0])
    return df.sort_values(col, ascending=False).iloc[0]

for case in cases:
    inst_path = OUT/case/'instability'/'instability_summary.csv'
    ff_path = OUT/case/'formfactor'/'formfactor_summary.csv'
    if not inst_path.exists() or not ff_path.exists():
        continue
    inst = pd.read_csv(inst_path)
    ff = pd.read_csv(ff_path)
    spin_vals = {sec: top_strength(inst, 'Gamma', 'ph', sec) for sec in ['ph-sx','ph-sy','ph-sz']}
    gamma_spin = np.nanmax(list(spin_vals.values()))
    gamma_charge = top_strength(inst, 'Gamma', 'ph', 'ph-rho')
    gamma_sc = top_strength(inst, 'Gamma', 'pp', 'pp-singlet')
    ff_charge = best_ff(ff, 'Gamma', 'ph', 'ph-rho')
    ff_spin = None
    for sec in ['ph-sx','ph-sy','ph-sz']:
        cand = best_ff(ff, 'Gamma', 'ph', sec)
        if cand is not None:
            ff_spin = cand
            break
    ff_sc = best_ff(ff, 'Gamma', 'pp', 'pp-singlet')
    rows.append({
        'case': case,
        'Gamma_charge_raw': gamma_charge,
        'Gamma_spin_max': gamma_spin,
        'Gamma_sx': spin_vals.get('ph-sx'),
        'Gamma_sy': spin_vals.get('ph-sy'),
        'Gamma_sz': spin_vals.get('ph-sz'),
        'Gamma_pp_singlet': gamma_sc,
        'Gamma_charge_label': None if ff_charge is None else ff_charge.get('candidate_phase_label', ff_charge.get('inferred_form_factor', None)),
        'Gamma_charge_onsite_weight': np.nan if ff_charge is None else float(ff_charge.get('onsite_weight', np.nan)),
        'Gamma_charge_A1': np.nan if ff_charge is None else float(ff_charge.get('onsite_A1_weight_within_onsite', ff_charge.get('onsite_A1_overlap', np.nan))),
        'Gamma_spin_label': None if ff_spin is None else ff_spin.get('candidate_phase_label', ff_spin.get('inferred_form_factor', None)),
        'Gamma_SC_label': None if ff_sc is None else ff_sc.get('candidate_phase_label', ff_sc.get('inferred_form_factor', None)),
    })

summary = pd.DataFrame(rows)
summary.to_csv(OUT/'bridge_diagnosis_summary.csv', index=False)
print('\n============================================================')
print('Bridge diagnosis compact summary')
print('============================================================')
print(summary.to_string(index=False))
print('\nwrote', OUT/'bridge_diagnosis_summary.csv')
PY
