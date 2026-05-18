#!/bin/bash
#SBATCH --job-name=spinful_bridge_s1m1
#SBATCH --account=gts-ikimchi3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=40G
#SBATCH --time=12:00:00
#SBATCH --qos=inferno
#SBATCH --output=logs/bridge-s1m1-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=miaoli@gatech.edu

set -euo pipefail

module load anaconda3
conda activate /storage/scratch1/4/mli826/conda_envs/diverge-test

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

echo "Job ID: ${SLURM_JOB_ID}"
echo "Submit dir: ${SLURM_SUBMIT_DIR}"
echo "Python: $(which python)"
python - <<'PY'
import diverge
print('diverge:', diverge.__file__)
PY

OUTDIR="${OUTDIR:-bridge_spinful_prr_raw_s1minus1}"
NK="${NK:-12}"
NKF="${NKF:-3}"
TU_DISTANCE="${TU_DISTANCE:-2.01}"
NITER="${NITER:-80}"
MAXVERT="${MAXVERT:-80.0}"
FILLING_NGRID="${FILLING_NGRID:-120}"

rm -rf "$OUTDIR"
mkdir -p "$OUTDIR"

echo "============================================================"
echo "Running corrected explicit-spin no-SOC bridge test"
echo "OUTDIR=${OUTDIR}"
echo "NK=${NK}, NKF=${NKF}, TU_DISTANCE=${TU_DISTANCE}, NITER=${NITER}, MAXVERT=${MAXVERT}"
echo "============================================================"

python -u TUfRG_runner_raw_s1minus1.py \
  --outdir "$OUTDIR" \
  --vertex-convention official_s1minus1 \
  --filling-mode fixed_mu \
  --mu 0.0 \
  --target-filling 0.4166666666666667 \
  --filling-ngrid "$FILLING_NGRID" \
  --nk "$NK" \
  --nkf "$NKF" \
  --tu-distance "$TU_DISTANCE" \
  --niter "$NITER" \
  --maxvert "$MAXVERT" \
  --lambda-nn 0.0 \
  --lambda-nnn 0.0 \
  --tu-channel-calc-project 0 \
  --tu-n-decomp-values 16 \
  --tu-n-singular-values 16 \
  --case U1p5_V0p0:1.5:0.0 \
  --case U0p0_V1p5:0.0:1.5

for CASE in U1p5_V0p0 U0p0_V1p5; do
  CASEDIR="$OUTDIR/$CASE"
  echo "============================================================"
  echo "Level-1 diagnosis: $CASE"
  echo "============================================================"
  python -u TUfRG_instability_rawguard.py "$CASEDIR" \
    --q Gamma M1 M2 M3 \
    --n-modes 8 \
    --out instability

  echo "============================================================"
  echo "Level-2 form factor: $CASE"
  echo "============================================================"
  python -u TUfRG_formfactor_from_modes.py "$CASEDIR" \
    --instability-dir instability \
    --out formfactor \
    --top-components 12 \
    --max-rows 120
done

python - <<'PY'
from pathlib import Path
import json
import numpy as np
import pandas as pd

OUT = Path(__import__('os').environ.get('OUTDIR', 'bridge_spinful_prr_raw_s1minus1'))
cases = ['U1p5_V0p0', 'U0p0_V1p5']

def top_strength(inst, q_label, family=None, sector=None):
    df = inst[inst['q_label'].astype(str).eq(q_label)].copy()
    if family is not None:
        df = df[df['family'].astype(str).eq(family)]
    if sector is not None:
        df = df[df['sector'].astype(str).eq(sector)]
    if len(df) == 0:
        return np.nan
    return float(df.sort_values('strength_svd_KL', ascending=False).iloc[0]['strength_svd_KL'])

def max_M(inst, family=None, sector=None):
    vals = [top_strength(inst, q, family, sector) for q in ['M1','M2','M3']]
    return float(np.nanmax(vals))

def best_ff(ff, q_label, family=None, sector=None, sector_contains=None):
    df = ff[ff['q_label'].astype(str).eq(q_label)].copy()
    if family is not None:
        df = df[df['family'].astype(str).eq(family)]
    if sector is not None:
        df = df[df['sector'].astype(str).eq(sector)]
    if sector_contains is not None:
        df = df[df['sector'].astype(str).str.contains(sector_contains, regex=False)]
    if len(df) == 0:
        return None
    return df.sort_values('strength_svd_KL', ascending=False).iloc[0]

rows = []
for case in cases:
    cdir = OUT / case
    inst = pd.read_csv(cdir / 'instability' / 'instability_summary.csv')
    ff = pd.read_csv(cdir / 'formfactor' / 'formfactor_summary.csv')
    cal_path = cdir / 'instability' / 'explicit_spin_ph_calibration.csv'
    cal = pd.read_csv(cal_path) if cal_path.exists() and cal_path.stat().st_size > 0 else pd.DataFrame()
    md = json.loads((cdir / 'metadata.json').read_text())

    gamma_spin_vals = {sec: top_strength(inst, 'Gamma', 'ph', sec) for sec in ['ph-sx','ph-sy','ph-sz']}
    gamma_spin = max(gamma_spin_vals.values())
    gamma_sc = top_strength(inst, 'Gamma', 'pp', 'pp-singlet')
    gamma_charge = top_strength(inst, 'Gamma', 'ph', 'ph-rho')
    M_charge = max_M(inst, 'ph', 'ph-rho')
    M_spin = max(max_M(inst, 'ph', sec) for sec in ['ph-sx','ph-sy','ph-sz'])
    M_sc = max_M(inst, 'pp', 'pp-singlet')

    compare = {
        'Gamma_spin': gamma_spin,
        'Gamma_pp_singlet': gamma_sc,
        'M_charge': M_charge,
        'M_spin': M_spin,
        'M_pp_singlet': M_sc,
    }
    leading = max(compare, key=lambda k: compare[k] if np.isfinite(compare[k]) else -np.inf)
    ff_spin = best_ff(ff, 'Gamma', family='ph', sector_contains='ph-s')
    ff_sc = best_ff(ff, 'Gamma', family='pp', sector='pp-singlet')

    rows.append({
        'case': case,
        'tu_channel_calc_project': md.get('postprocess_report', {}).get('tu_channel_calc_project'),
        'vertex_convention': md.get('model_summary', {}).get('metadata', {}).get('vertex_convention'),
        'Gamma_spin_max': gamma_spin,
        'Gamma_spin_sx': gamma_spin_vals['ph-sx'],
        'Gamma_spin_sy': gamma_spin_vals['ph-sy'],
        'Gamma_spin_sz': gamma_spin_vals['ph-sz'],
        'Gamma_pp_singlet': gamma_sc,
        'Gamma_charge_ignored': gamma_charge,
        'M_charge_max': M_charge,
        'M_spin_max': M_spin,
        'M_pp_singlet_max': M_sc,
        'leading_excluding_Gamma_charge': leading,
        'Gamma_spin_label': None if ff_spin is None else ff_spin.get('candidate_phase_label'),
        'Gamma_SC_label': None if ff_sc is None else ff_sc.get('candidate_phase_label'),
        'Gamma_SC_onsite_weight': np.nan if ff_sc is None else float(ff_sc.get('onsite_weight', np.nan)),
        'Gamma_SC_A1_within_onsite': np.nan if ff_sc is None else float(ff_sc.get('onsite_A1_weight_within_onsite', np.nan)),
        'max_triplet_cv': np.nan if cal.empty else float(cal['triplet_cv'].max()),
        'max_triplet_spread': np.nan if cal.empty else float(cal['triplet_spread'].max()),
    })

summary = pd.DataFrame(rows)
summary.to_csv(OUT / 'bridge_summary.csv', index=False)
print('\n' + '='*100)
print('BRIDGE SUMMARY')
print('='*100)
print(summary.to_string(index=False))
print('\nWrote', OUT / 'bridge_summary.csv')
PY

echo "Done. Outputs are in $OUTDIR"
