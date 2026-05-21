#!/bin/bash
#SBATCH --job-name=tufrg_snapdiag
#SBATCH --account=gts-ikimchi3
#SBATCH --qos=inferno
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=miaoli@gatech.edu

set -euo pipefail
mkdir -p logs

module load anaconda3
conda activate /storage/scratch1/4/mli826/conda_envs/diverge-test
cd "$SLURM_SUBMIT_DIR"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export DIVERGE_OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export DIVERGE_FFTW_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export DIVERGE_OMP_IMPLICIT_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

CASE_DIR="${CASE_DIR:?Need CASE_DIR, e.g. km_spinful_runs/my_case}"
N_MODES="${N_MODES:-8}"
ORDER_TOP_COMPONENTS="${ORDER_TOP_COMPONENTS:-30}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

SKIP_FLAG=""
if [ "$SKIP_EXISTING" = "1" ] || [ "$SKIP_EXISTING" = "true" ]; then
  SKIP_FLAG="--skip-existing"
fi

echo "============================================================"
echo "TUFRG snapshot diagnosis"
echo "CASE_DIR=$CASE_DIR"
echo "N_MODES=$N_MODES ORDER_TOP_COMPONENTS=$ORDER_TOP_COMPONENTS SKIP_EXISTING=$SKIP_EXISTING"
echo "============================================================"

python -u TUfRG_diagnose_snapshots.py "$CASE_DIR" \
  --n-modes "$N_MODES" \
  --order-top-components "$ORDER_TOP_COMPONENTS" \
  $SKIP_FLAG

python -u TUfRG_plot_snapshot_evolution.py "$CASE_DIR"
