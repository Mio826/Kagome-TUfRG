#!/bin/bash
#SBATCH --job-name=mu0011PRR
#SBATCH --account=gts-ikimchi3
#SBATCH --qos=inferno
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=24:00:00
#SBATCH --array=1-11
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=miaoli@gatech.edu

set -euo pipefail

mkdir -p logs

module load anaconda3
conda activate /storage/scratch1/4/mli826/conda_envs/diverge-test

# Avoid PACE/XALT Python-path contamination
unset PYTHONHOME
unset PYTHONPATH

cd "$SLURM_SUBMIT_DIR"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export DIVERGE_OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export DIVERGE_FFTW_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export DIVERGE_OMP_IMPLICIT_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1


# ============================================================
# 11-point U,V grid
# ============================================================

CASES=(
    "0.0 0.5"
    "0.0 1.5"
    "1.0 0.0"
    "1.0 0.5"
    "1.0 1.5"
    "2.0 0.0"
    "2.0 0.5"
    "2.0 1.5"
    "3.0 0.0"
    "3.0 0.5"
    "3.0 1.5"
)

IDX=$((SLURM_ARRAY_TASK_ID - 1))
read -r U V <<< "${CASES[$IDX]}"


# ============================================================
# Physics parameters: controlled detuning
# ============================================================

LAMBDA_NN=0.0
LAMBDA_NNN=0.0

FILLING_MODE=fixed_mu
MU=0.011
FILLING_NGRID=120


# ============================================================
# Numerical parameters
# ============================================================

NK=24
NKF=12
TU_DISTANCE=2.01

NITER=100
MAXVERT=80.0

SNAPSHOT_EVERY=5
SNAPSHOT_AFTER=1
SNAPSHOT_FINAL=1


# ============================================================
# Output
# ============================================================

U_TAG=${U/./p}
V_TAG=${V/./p}

OUTDIR="nosoc_mu0p011_prrflow_nk24_nkf12_11pts"
TAG="nosoc_mu0p011_U${U_TAG}_V${V_TAG}"


echo "============================================================"
echo "CONTROLLED DETUNING TUFRG"
echo "============================================================"
echo "job             = ${SLURM_JOB_ID}"
echo "array task      = ${SLURM_ARRAY_TASK_ID}"
echo "hostname        = $(hostname)"
echo "python          = $(which python)"
echo
echo "U               = $U"
echo "V               = $V"
echo "lambda_nn       = $LAMBDA_NN"
echo "lambda_nnn      = $LAMBDA_NNN"
echo
echo "filling_mode    = $FILLING_MODE"
echo "mu              = $MU"
echo
echo "NK              = $NK"
echo "NKF             = $NKF"
echo "TU_DISTANCE     = $TU_DISTANCE"
echo "NITER           = $NITER"
echo "MAXVERT         = $MAXVERT"
echo
echo "SNAPSHOT_EVERY  = $SNAPSHOT_EVERY"
echo "SNAPSHOT_AFTER  = $SNAPSHOT_AFTER"
echo
echo "OUTDIR          = $OUTDIR"
echo "TAG             = $TAG"
echo "============================================================"


ARGS=(
    python -u TUfRG_runner_snapshots.py
    --U "$U"
    --V "$V"
    --lambda-nn "$LAMBDA_NN"
    --lambda-nnn "$LAMBDA_NNN"
    --filling-mode "$FILLING_MODE"
    --mu "$MU"
    --filling-ngrid "$FILLING_NGRID"
    --nk "$NK"
    --nkf "$NKF"
    --tu-distance "$TU_DISTANCE"
    --niter "$NITER"
    --maxvert "$MAXVERT"
    --tu-channel-calc-project 0
    --vertex-convention official_s1minus1
    --outdir "$OUTDIR"
    --tag "$TAG"
    --snapshot-every "$SNAPSHOT_EVERY"
    --snapshot-after "$SNAPSHOT_AFTER"
    --snapshot-final
)

echo
echo "Running:"
printf ' %q' "${ARGS[@]}"
echo
echo

"${ARGS[@]}"


FINAL="${OUTDIR}/${TAG}/snapshots/final/O.dvg"

echo
echo "============================================================"
echo "FINAL CHECK"
echo "============================================================"
echo "$FINAL"

if [[ ! -f "$FINAL" ]]; then
    echo "[ERROR] final O.dvg missing"
    exit 1
fi

echo "[SUCCESS]"
