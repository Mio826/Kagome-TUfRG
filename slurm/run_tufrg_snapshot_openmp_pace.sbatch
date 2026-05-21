#!/bin/bash
#SBATCH --job-name=tufrg_snap
#SBATCH --account=gts-ikimchi3
#SBATCH --qos=inferno
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=24:00:00
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

U="${U:-1.0}"
V="${V:-0.0}"
LAMBDA_NN="${LAMBDA_NN:-0.0}"
LAMBDA_NNN="${LAMBDA_NNN:-0.0}"
NK="${NK:-18}"
NKF="${NKF:-12}"
TU_DISTANCE="${TU_DISTANCE:-2.01}"
NITER="${NITER:-100}"
MAXVERT="${MAXVERT:-80.0}"
TAG="${TAG:-snapshot_U${U}_V${V}_lnn${LAMBDA_NN}}"
SNAPSHOT_EVERY="${SNAPSHOT_EVERY:-10}"
SNAPSHOT_AFTER="${SNAPSHOT_AFTER:-1}"
SNAPSHOT_AT="${SNAPSHOT_AT:-}"
SNAPSHOT_FINAL="${SNAPSHOT_FINAL:-1}"

FINAL_FLAG=""
if [ "$SNAPSHOT_FINAL" = "1" ] || [ "$SNAPSHOT_FINAL" = "true" ]; then
  FINAL_FLAG="--snapshot-final"
fi

echo "============================================================"
echo "TUFRG snapshot run"
echo "TAG=$TAG"
echo "U=$U V=$V lambda_nn=$LAMBDA_NN lambda_nnn=$LAMBDA_NNN"
echo "NK=$NK NKF=$NKF TU_DISTANCE=$TU_DISTANCE NITER=$NITER MAXVERT=$MAXVERT"
echo "SNAPSHOT_EVERY=$SNAPSHOT_EVERY SNAPSHOT_AFTER=$SNAPSHOT_AFTER SNAPSHOT_AT=$SNAPSHOT_AT SNAPSHOT_FINAL=$SNAPSHOT_FINAL"
echo "============================================================"

python -u TUfRG_runner_snapshots.py \
  --filling-mode fixed_mu \
  --mu 0.0 \
  --filling-ngrid 120 \
  --nk "$NK" \
  --nkf "$NKF" \
  --tu-distance "$TU_DISTANCE" \
  --niter "$NITER" \
  --maxvert "$MAXVERT" \
  --U "$U" \
  --V "$V" \
  --lambda-nn "$LAMBDA_NN" \
  --lambda-nnn "$LAMBDA_NNN" \
  --tu-channel-calc-project 0 \
  --vertex-convention official_s1minus1 \
  --tag "$TAG" \
  --snapshot-every "$SNAPSHOT_EVERY" \
  --snapshot-after "$SNAPSHOT_AFTER" \
  --snapshot-at "$SNAPSHOT_AT" \
  $FINAL_FLAG
