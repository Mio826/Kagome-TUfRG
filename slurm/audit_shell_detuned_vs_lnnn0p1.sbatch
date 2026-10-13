#!/bin/bash
#SBATCH -J shellAudit
#SBATCH -A gts-ikimchi3
#SBATCH --qos=inferno
#SBATCH --partition=cpu-large
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --array=1-2
#SBATCH --output=logs/shellAudit_%A_%a.out
#SBATCH --error=logs/shellAudit_%A_%a.err

set -eo pipefail

cd /storage/scratch1/4/mli826/divERGe_test_run/runs

module load anaconda3
conda activate /storage/scratch1/4/mli826/conda_envs/diverge-test

unset PYTHONHOME
unset PYTHONPATH
export PYTHONNOUSERSITE=1

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

OUTBASE="residual_audit_detuned_vs_lnnn0p1"
mkdir -p "$OUTBASE"

if [ "${SLURM_ARRAY_TASK_ID}" -eq 1 ]; then

    ROOT="nosoc_mu0p011_prrflow_nk24_nkf12_11pts"
    OUT="$OUTBASE/shell_detuned"

elif [ "${SLURM_ARRAY_TASK_ID}" -eq 2 ]; then

    ROOT="lnnn0p1"
    OUT="$OUTBASE/shell_lnnn0p1"

else
    echo "Unknown array task ${SLURM_ARRAY_TASK_ID}"
    return 1 2>/dev/null || true
fi

echo "============================================================"
echo "HOST=$(hostname)"
echo "TASK=${SLURM_ARRAY_TASK_ID}"
echo "ROOT=$ROOT"
echo "OUT=$OUT"
echo "============================================================"

which python
python --version

python TUfRG_shell_resolved_formfactor.py \
    --root "$ROOT" \
    --out "$OUT" \
    --top-n 40

echo
echo "============================================================"
echo "OUTPUT FILES"
echo "============================================================"

find "$OUT" -maxdepth 1 -type f -printf '%f %k KB\n' | sort

echo
echo "DONE"
