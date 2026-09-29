#!/bin/bash
#SBATCH --job-name=lnnn01PRR
#SBATCH --account=gts-ikimchi3
#SBATCH --partition=cpu-large
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/lnnn01PRR_%A_%a.out
#SBATCH --error=logs/lnnn01PRR_%A_%a.err

set -eo pipefail

cd /storage/home/hcoda1/4/mli826/scratch/divERGe_test_run/runs

source ~/.bashrc
conda activate /storage/scratch1/4/mli826/conda_envs/diverge-test

GRID="scan_lnnn0p1_prrflow_nk24_nkf12_11pts_grid.csv"

LINE=$(awk -F, -v id="${SLURM_ARRAY_TASK_ID}" 'NR>1 && $1==id {print $0}' "$GRID")
if [ -z "$LINE" ]; then
  echo "[ERROR] no grid row for array id ${SLURM_ARRAY_TASK_ID}"
  exit 1
fi

U=$(echo "$LINE" | awk -F, '{print $2}')
V=$(echo "$LINE" | awk -F, '{print $3}')
NOTE=$(echo "$LINE" | awk -F, '{print $4}')

Utag=${U/./p}
Vtag=${V/./p}

OUTROOT="/storage/home/hcoda1/4/mli826/scratch/divERGe_test_run/runs/lnnn0p1-prrflow-nk24-nkf12-11pts"
TAG="lnnn0p1_prrflow_nk24_nkf12_U${Utag}_V${Vtag}_${NOTE}"

echo "============================================================"
echo "[lambda_NNN=0.1 PRR-flow nk24/nkf12 11-point scan]"
echo "U=${U}"
echo "V=${V}"
echo "NOTE=${NOTE}"
echo "lambda_nn=0.0"
echo "lambda_nnn=0.1"
echo "nk=24"
echo "nkf=12"
echo "tu_distance=2.01"
echo "niter=100"
echo "maxvert=80"
echo "snapshot_every=5"
echo "snapshot_after=1"
echo "OUTROOT=${OUTROOT}"
echo "TAG=${TAG}"
echo "============================================================"

python TUfRG_runner_snapshots.py \
  --U "${U}" \
  --V "${V}" \
  --lambda-nn 0.0 \
  --lambda-nnn 0.1 \
  --filling-mode fixed_filling \
  --target-filling 0.4166666666666667 \
  --nk 24 \
  --nkf 12 \
  --tu-distance 2.01 \
  --niter 100 \
  --maxvert 80 \
  --tu-channel-calc-project 0 \
  --vertex-convention official_s1minus1 \
  --tag "${TAG}" \
  --outdir "${OUTROOT}" \
  --snapshot-every 5 \
  --snapshot-after 1 \
  --snapshot-final

echo "[CHECK] ${OUTROOT}/${TAG}/snapshots/final/O.dvg"
test -f "${OUTROOT}/${TAG}/snapshots/final/O.dvg"

echo "[DONE] ${OUTROOT}/${TAG}"
