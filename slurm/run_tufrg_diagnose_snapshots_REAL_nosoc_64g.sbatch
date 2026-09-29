#!/bin/bash
#SBATCH --job-name=diag_nosoc
#SBATCH --account=gts-ikimchi3
#SBATCH --qos=inferno
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/diag_nosoc_%j.out
#SBATCH --error=logs/diag_nosoc_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=miaoli@gatech.edu

set -euo pipefail

mkdir -p logs

module load anaconda3
conda activate /storage/scratch1/4/mli826/conda_envs/diverge-test

unset PYTHONHOME
unset PYTHONPATH

cd "$SLURM_SUBMIT_DIR"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

CASE_DIR="${CASE_DIR:?CASE_DIR is required}"
N_MODES="${N_MODES:-8}"
ORDER_TOP_COMPONENTS="${ORDER_TOP_COMPONENTS:-40}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"

echo "============================================================"
echo "TUFRG TRUSTED SNAPSHOT DIAGNOSIS"
echo "============================================================"
echo "CASE_DIR              = $CASE_DIR"
echo "N_MODES               = $N_MODES"
echo "ORDER_TOP_COMPONENTS  = $ORDER_TOP_COMPONENTS"
echo "SKIP_EXISTING         = $SKIP_EXISTING"
echo "python                = $(which python)"
echo "============================================================"

test -f "$CASE_DIR/snapshots/final/O.dvg" || {
    echo "[ERROR] missing final O.dvg:"
    echo "$CASE_DIR/snapshots/final/O.dvg"
    exit 1
}

ARGS=(
    python -u TUfRG_diagnose_snapshots.py
    "$CASE_DIR"
    --n-modes "$N_MODES"
    --order-top-components "$ORDER_TOP_COMPONENTS"
)

if [[ "$SKIP_EXISTING" == "1" || "$SKIP_EXISTING" == "true" ]]; then
    ARGS+=(--skip-existing)
fi

echo
echo "Running:"
printf ' %q' "${ARGS[@]}"
echo
echo

"${ARGS[@]}"

SUMMARY="$CASE_DIR/snapshot_diagnosis/snapshot_diagnosis_summary.csv"

echo
echo "============================================================"
echo "FINAL CHECK"
echo "============================================================"

if [[ ! -s "$SUMMARY" ]]; then
    echo "[ERROR] missing diagnosis summary:"
    echo "$SUMMARY"
    exit 1
fi

echo "[SUCCESS]"
echo "$SUMMARY"
