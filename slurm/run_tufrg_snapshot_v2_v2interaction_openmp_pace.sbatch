#!/bin/bash
#SBATCH --job-name=tufrg_v2_snap
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=24:00:00

set -euo pipefail

# Submit from the runs/ directory, e.g.:
#   sbatch --export=ALL,U=3,V1=1.5,V2=0.3,LAMBDA_NN=0.0,TAG=test_v2 slurm/run_tufrg_snapshot_v2_v2interaction_openmp_pace.sbatch
RUN_DIR="${RUN_DIR:-$SLURM_SUBMIT_DIR}"
cd "$RUN_DIR"
mkdir -p slurm/logs

# Optional conda activation.  If your sbatch environment already has the right
# python/divERGe environment loaded, leave CONDA_ENV empty.
if [[ -n "${CONDA_ENV:-}" ]]; then
  if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
  fi
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"

# Physical parameters.
U="${U:-1.5}"
V1="${V1:-${V:-0.0}}"   # V is kept as a backward-compatible alias for V1.
V2="${V2:-0.0}"
LAMBDA_NN="${LAMBDA_NN:-0.0}"
LAMBDA_NNN="${LAMBDA_NNN:-0.0}"

# Filling / mesh / flow.
FILLING_MODE="${FILLING_MODE:-fixed_mu}"
MU="${MU:-0.0}"
TARGET_FILLING="${TARGET_FILLING:-0.4166666666666667}"
FILLING_NGRID="${FILLING_NGRID:-120}"
NK="${NK:-18}"
NKF="${NKF:-12}"
TU_DISTANCE="${TU_DISTANCE:-2.01}"
NITER="${NITER:-100}"
MAXVERT="${MAXVERT:-80.0}"

# Snapshot controls.
SNAPSHOT_EVERY="${SNAPSHOT_EVERY:-5}"
SNAPSHOT_AFTER="${SNAPSHOT_AFTER:-1}"
SNAPSHOT_AT="${SNAPSHOT_AT:-}"
SNAPSHOT_ROOT="${SNAPSHOT_ROOT:-snapshots}"
SNAPSHOT_FINAL="${SNAPSHOT_FINAL:-1}"

# Output controls.
OUTDIR="${OUTDIR:-km_v2_spinful_runs}"
TAG="${TAG:-U${U}_V1${V1}_V2${V2}_lnn${LAMBDA_NN}_lnnn${LAMBDA_NNN}_nk${NK}_nkf${NKF}}"
VERTEX_CONVENTION="${VERTEX_CONVENTION:-official_s1minus1}"
TU_CHANNEL_CALC_PROJECT="${TU_CHANNEL_CALC_PROJECT:-0}"

args=(
  python -u TUfRG_runner_snapshots_v2.py
  --U "$U"
  --V1 "$V1"
  --V2 "$V2"
  --lambda-nn "$LAMBDA_NN"
  --lambda-nnn "$LAMBDA_NNN"
  --filling-mode "$FILLING_MODE"
  --mu "$MU"
  --target-filling "$TARGET_FILLING"
  --filling-ngrid "$FILLING_NGRID"
  --nk "$NK"
  --nkf "$NKF"
  --tu-distance "$TU_DISTANCE"
  --niter "$NITER"
  --maxvert "$MAXVERT"
  --tu-channel-calc-project "$TU_CHANNEL_CALC_PROJECT"
  --vertex-convention "$VERTEX_CONVENTION"
  --outdir "$OUTDIR"
  --tag "$TAG"
  --snapshot-every "$SNAPSHOT_EVERY"
  --snapshot-after "$SNAPSHOT_AFTER"
  --snapshot-root "$SNAPSHOT_ROOT"
)

if [[ -n "$SNAPSHOT_AT" ]]; then
  args+=(--snapshot-at "$SNAPSHOT_AT")
fi
if [[ "$SNAPSHOT_FINAL" == "1" || "$SNAPSHOT_FINAL" == "true" || "$SNAPSHOT_FINAL" == "True" ]]; then
  args+=(--snapshot-final)
fi

printf 'Running command:\n'
printf ' %q' "${args[@]}"
printf '\n\n'
"${args[@]}"
