#!/bin/bash
#SBATCH --job-name=tufrg_omp
#SBATCH --account=gts-ikimchi3
#SBATCH --qos=inferno
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
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

# Important: set thread controls BEFORE Python imports diverge/numpy.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export DIVERGE_OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export DIVERGE_FFTW_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

# Avoid nested BLAS/OpenMP oversubscription.
export DIVERGE_OMP_IMPLICIT_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

U="${U:-1.5}"
V="${V:-1.5}"
LAMBDA_NN="${LAMBDA_NN:-0.0}"
LAMBDA_NNN="${LAMBDA_NNN:-0.0}"
NK="${NK:-18}"
NKF="${NKF:-3}"
TU_DISTANCE="${TU_DISTANCE:-2.01}"
NITER="${NITER:-80}"
MAXVERT="${MAXVERT:-80.0}"
TAG="${TAG:-omp_U${U}_V${V}_nk${NK}_nkf${NKF}_tu${TU_DISTANCE}}"

echo "============================================================"
echo "OpenMP TUfRG run"
echo "============================================================"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK"
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "DIVERGE_OMP_NUM_THREADS=$DIVERGE_OMP_NUM_THREADS"
echo "DIVERGE_FFTW_NUM_THREADS=$DIVERGE_FFTW_NUM_THREADS"
echo "OPENBLAS_NUM_THREADS=$OPENBLAS_NUM_THREADS"
echo "U=$U V=$V lambda_nn=$LAMBDA_NN lambda_nnn=$LAMBDA_NNN"
echo "NK=$NK NKF=$NKF TU_DISTANCE=$TU_DISTANCE"
echo "NITER=$NITER MAXVERT=$MAXVERT"
echo "TAG=$TAG"
echo "============================================================"

python -u TUfRG_runner.py \
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
  --tag "$TAG"

CASE_DIR="km_spinful_runs/${TAG}"

echo "============================================================"
echo "Diagnosis"
echo "============================================================"

python -u TUfRG_instability.py "$CASE_DIR" --out instability
python -u TUfRG_formfactor_from_modes.py "$CASE_DIR" --instability-dir instability --out formfactor

echo "Done: $CASE_DIR"
