#!/bin/bash
#SBATCH --job-name=tufrg_fixmu
#SBATCH --account=gts-ikimchi3
#SBATCH --qos=inferno
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
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

# ============================================================
# Physics parameters
# ============================================================
U="${U:-1.0}"
V="${V:-0.0}"

LAMBDA_NN="${LAMBDA_NN:-0.0}"
LAMBDA_NNN="${LAMBDA_NNN:-0.0}"

FILLING_MODE="${FILLING_MODE:-fixed_mu}"
MU="${MU:-0.0}"
FILLING_NGRID="${FILLING_NGRID:-120}"

# ============================================================
# Numerical parameters
# ============================================================
NK="${NK:-18}"
NKF="${NKF:-12}"
TU_DISTANCE="${TU_DISTANCE:-2.01}"

NITER="${NITER:-100}"
MAXVERT="${MAXVERT:-80.0}"

# ============================================================
# Snapshot parameters
# ============================================================
SNAPSHOT_EVERY="${SNAPSHOT_EVERY:-5}"
SNAPSHOT_AFTER="${SNAPSHOT_AFTER:-1}"
SNAPSHOT_AT="${SNAPSHOT_AT:-}"
SNAPSHOT_FINAL="${SNAPSHOT_FINAL:-1}"

# ============================================================
# Output
# ============================================================
OUTDIR="${OUTDIR:-nosoc_detuned}"
TAG="${TAG:-snapshot_U${U}_V${V}_mu${MU}}"

echo "============================================================"
echo "TUFRG FIXED-MU SNAPSHOT RUN"
echo "============================================================"
echo "hostname        = $(hostname)"
echo "pwd             = $(pwd)"
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
echo "SNAPSHOT_FINAL  = $SNAPSHOT_FINAL"
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
)

if [[ -n "$SNAPSHOT_AT" ]]; then
    ARGS+=(--snapshot-at "$SNAPSHOT_AT")
fi

if [[ "$SNAPSHOT_FINAL" == "1" || "$SNAPSHOT_FINAL" == "true" ]]; then
    ARGS+=(--snapshot-final)
fi

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
