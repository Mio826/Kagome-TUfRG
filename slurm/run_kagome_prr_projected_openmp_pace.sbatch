#!/bin/bash
#SBATCH --job-name=prr_su2_omp
#SBATCH --account=gts-ikimchi3
#SBATCH --qos=inferno
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=miaoli@gatech.edu

set -euo pipefail
mkdir -p logs

module load anaconda3
conda activate /storage/scratch1/4/mli826/conda_envs/diverge-test

cd "$SLURM_SUBMIT_DIR"

# OpenMP tuning for divERGe. This is one Python process with many C/OpenMP threads.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export DIVERGE_OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export DIVERGE_FFTW_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export DIVERGE_OMP_IMPLICIT_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

OUTDIR="${OUTDIR:-outputs/kagome_prr_su2_openmp_baseline}"
NK="${NK:-18}"
NKF="${NKF:-3}"
TU_DISTANCE="${TU_DISTANCE:-2.01}"
MAXITER="${MAXITER:-80}"
MAXVERT="${MAXVERT:-80}"
T_HOP="${T_HOP:-1.0}"
MU="${MU:-0.0}"

# Default four benchmark cases. Override CASES from sbatch if needed.
CASES="${CASES:-U1p0_V0p0:1.0:0.0 U0p0_V1p5:0.0:1.5 U3p0_V1p0:3.0:1.0 U1p5_V1p5:1.5:1.5}"

mkdir -p "$OUTDIR"

echo "============================================================"
echo "SU2-reduced PRR projected OpenMP run"
echo "============================================================"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK"
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "DIVERGE_OMP_NUM_THREADS=$DIVERGE_OMP_NUM_THREADS"
echo "DIVERGE_FFTW_NUM_THREADS=$DIVERGE_FFTW_NUM_THREADS"
echo "OPENBLAS_NUM_THREADS=$OPENBLAS_NUM_THREADS"
echo "OUTDIR=$OUTDIR"
echo "NK=$NK NKF=$NKF TU_DISTANCE=$TU_DISTANCE"
echo "MAXITER=$MAXITER MAXVERT=$MAXVERT"
echo "CASES=$CASES"
echo "============================================================"

CMD=(python -u run_kagome_prr_mu0_projected_small.py \
  --outdir "$OUTDIR" \
  --nk "$NK" \
  --nkf "$NKF" \
  --tu-distance "$TU_DISTANCE" \
  --maxiter "$MAXITER" \
  --maxvert "$MAXVERT" \
  --t "$T_HOP" \
  --mu "$MU")

for C in $CASES; do
  CMD+=(--case "$C")
done

"${CMD[@]}"

echo "============================================================"
echo "q-resolved loop-dressed diagnosis for all cases"
echo "============================================================"

for C in $CASES; do
  NAME="${C%%:*}"
  echo "---------------- $NAME ----------------"
  python -u inspect_kagome_prr_qresolved_projected.py \
    "$OUTDIR" \
    --case "$NAME" \
    --csv "$OUTDIR/${NAME}_qresolved_summary.csv" || \
  python -u inspect_kagome_prr_qresolved_projected.py \
    "$OUTDIR/$NAME" \
    --csv "$OUTDIR/$NAME/qresolved_summary.csv"
done

echo "============================================================"
echo "Form-factor/channel-decomposition inspector"
echo "============================================================"
python -u inspect_kagome_prr_form_factors_projected.py "$OUTDIR" || true

echo "============================================================"
echo "M-charge subleading/residual inspector"
echo "============================================================"
python -u inspect_kagome_prr_Mcharge_subleading_projected.py "$OUTDIR" || true

cat > "$OUTDIR/summarize_qresolved.py" <<'PY'
import pandas as pd
from pathlib import Path

out = Path('.')
rows = []
for p in sorted(out.glob('*_qresolved_summary.csv')):
    case = p.name.replace('_qresolved_summary.csv', '')
    df = pd.read_csv(p)
    def get(q, ch):
        sel = df[(df['q_label'] == q) & (df['channel'] == ch)]
        return float(sel['loop_svd_KL_0'].iloc[0]) if len(sel) else float('nan')
    row = {
        'case': case,
        'Gamma_charge': get('Gamma', 'charge'),
        'Gamma_mag': get('Gamma', 'mag'),
        'Gamma_sc': get('Gamma', 'sc'),
        'M_charge': max(get('M1', 'charge'), get('M2', 'charge'), get('M3', 'charge')),
        'M_mag': max(get('M1', 'mag'), get('M2', 'mag'), get('M3', 'mag')),
        'M_sc': max(get('M1', 'sc'), get('M2', 'sc'), get('M3', 'sc')),
    }
    vals = {k:v for k,v in row.items() if k != 'case'}
    row['leading_including_Gamma_charge'] = max(vals, key=vals.get)
    vals_no_gcharge = {k:v for k,v in vals.items() if k != 'Gamma_charge'}
    row['leading_excluding_Gamma_charge'] = max(vals_no_gcharge, key=vals_no_gcharge.get)
    rows.append(row)

summary = pd.DataFrame(rows)
summary.to_csv('qresolved_loop_summary_all_cases.csv', index=False)
print(summary.to_string(index=False))
PY

( cd "$OUTDIR" && python summarize_qresolved.py )

echo "Done. Summary: $OUTDIR/qresolved_loop_summary_all_cases.csv"
