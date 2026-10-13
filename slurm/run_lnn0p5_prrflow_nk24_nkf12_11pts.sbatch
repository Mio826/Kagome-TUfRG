#!/bin/bash
#SBATCH --job-name=lnn05PRR24
#SBATCH --account=gts-ikimchi3
#SBATCH --partition=cpu-medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/lnn05PRR24_%A_%a.out
#SBATCH --error=logs/lnn05PRR24_%A_%a.err

set -eo pipefail

cd /storage/home/hcoda1/4/mli826/scratch/divERGe_test_run/runs

source ~/.bashrc
export PATH="/storage/scratch1/4/mli826/conda_envs/diverge-test/bin:$PATH"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

# ============================================================
# 1. Physical parameters
# ============================================================

LAMBDA_NN=0.5
LAMBDA_NNN=0.0

# Real NN hopping t=1: inherited from the previous
# runner/model default. No change to the hopping builder.

# ============================================================
# 2. Read interaction grid
# ============================================================

GRID="scan_lnn0p5_prrflow_nk24_nkf12_11pts.csv"

LINE=$(awk -F, -v id="${SLURM_ARRAY_TASK_ID}" \
  'NR>1 && $1==id {print $0}' "$GRID")

if [ -z "$LINE" ]; then
    echo "[ERROR] Missing array index ${SLURM_ARRAY_TASK_ID}"
    exit 1
fi

U=$(echo "$LINE" | cut -d, -f2)
V=$(echo "$LINE" | cut -d, -f3)

Utag=${U/./p}
Vtag=${V/./p}

# ============================================================
# 3. Output paths
# ============================================================

OUTROOT="/storage/home/hcoda1/4/mli826/scratch/divERGe_test_run/runs/lnn0p5-prrflow-nk24-nkf12-11pts"

TAG="lnn0p5_prrflow_nk24_nkf12_U${Utag}_V${Vtag}"

mkdir -p "$OUTROOT"

echo "=================================================="
echo "Kagome TUfRG strong NN SOC"
echo "U=$U V=$V"
echo "t=1.0 (model default)"
echo "lambda_nn=$LAMBDA_NN"
echo "lambda_nnn=$LAMBDA_NNN"
echo "nk=24 nkf=12"
echo "NITER=100 MAXVERT=80"
echo "snapshot_every=5 snapshot_after=1"
echo "OUTROOT=$OUTROOT"
echo "TAG=$TAG"
echo "=================================================="

# ============================================================
# 4. Run TUfRG
# ============================================================

python TUfRG_runner_snapshots.py \
    --U "$U" \
    --V "$V" \
    --lambda-nn "$LAMBDA_NN" \
    --lambda-nnn "$LAMBDA_NNN" \
    --filling-mode fixed_filling \
    --target-filling 0.4166666666666667 \
    --nk 24 \
    --nkf 12 \
    --tu-distance 2.01 \
    --niter 100 \
    --maxvert 80 \
    --tu-channel-calc-project 0 \
    --vertex-convention official_s1minus1 \
    --tag "$TAG" \
    --outdir "$OUTROOT" \
    --snapshot-every 5 \
    --snapshot-after 1 \
    --snapshot-final

# ============================================================
# 5. Verify outputs
# ============================================================

CASE="$OUTROOT/$TAG"

test -f "$CASE/F.dat"
test -f "$CASE/metadata.json"
test -f "$CASE/snapshots/final/O.dvg"

echo "[SUCCESS] Flow finished: $CASE"
