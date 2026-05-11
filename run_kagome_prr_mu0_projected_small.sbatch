#!/bin/bash
#SBATCH --job-name=diverge-kagome-prr-mu0-proj
#SBATCH --account=gts-ikimchi3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --qos=inferno
#SBATCH --output=logs/diverge-kagome-prr-mu0-proj-%j.out
#SBATCH --error=logs/diverge-kagome-prr-mu0-proj-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=miaoli@gatech.edu

set -euo pipefail

cd /storage/home/hcoda1/4/mli826/scratch/divERGe_test_run
mkdir -p logs outputs/kagome_prr_mu0_projected_small runs

echo "Job started on $(hostname)"
echo "Working directory: $(pwd)"
echo "SLURM job ID: ${SLURM_JOB_ID}"
echo "Start time: $(date)"

module purge
module load anaconda3/2023.03
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /storage/scratch1/4/mli826/conda_envs/diverge-test

echo "Python path: $(which python)"
python -c "import diverge; print('diverge imported from:', diverge.__file__)"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# Small controlled run. Override at submission time, e.g.
#   NK=16 NKF=3 MAXITER=120 sbatch runs/run_kagome_prr_mu0_small.sbatch
NK=${NK:-12}
NKF=${NKF:-3}
TU_DISTANCE=${TU_DISTANCE:-2.01}
MAXVERT=${MAXVERT:-80.0}
MAXITER=${MAXITER:-80}
TVAL=${TVAL:-1.0}
MU=${MU:-0.0}
OUTDIR=${OUTDIR:-outputs/kagome_prr_mu0_projected_small}
DIAGNOSE=${DIAGNOSE:-1}

DIAG_FLAG=""
if [[ "${DIAGNOSE}" == "1" ]]; then
  DIAG_FLAG="--diagnose"
fi

python runs/run_kagome_prr_mu0_projected_small.py \
  --outdir "${OUTDIR}" \
  --nk "${NK}" \
  --nkf "${NKF}" \
  --tu-distance "${TU_DISTANCE}" \
  --maxvert "${MAXVERT}" \
  --maxiter "${MAXITER}" \
  --t "${TVAL}" \
  --mu "${MU}" \
  ${DIAG_FLAG}

echo "End time: $(date)"
echo "Job finished"
