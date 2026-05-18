#!/bin/bash
#SBATCH --job-name=bridge-q0-variants
#SBATCH --account=gts-ikimchi3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --qos=inferno
#SBATCH --output=logs/bridge-q0-variants-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=miaoli@gatech.edu

set -euo pipefail

module load anaconda3
conda activate /storage/scratch1/4/mli826/conda_envs/diverge-test

cd /storage/scratch1/4/mli826/divERGe_test_run/runs
mkdir -p logs

ROOT=${ROOT:-bridge_spinful_prr_raw_s1minus1}
SCRIPT=${SCRIPT:-inspect_bridge_q0_variants.py}

python -u "$SCRIPT" "$ROOT" \
  --case U1p5_V0p0 \
  --case U0p0_V1p5 \
  --out bridge_q0_variant_inspection \
  --top 40

echo "Done. Outputs are in $ROOT/bridge_q0_variant_inspection/"
