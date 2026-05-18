#!/bin/bash
#SBATCH --job-name=tufrg_smoke
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G

# ============================================================
# PACE sbatch script: quick smoke test for spinful kagome Kane-Mele TUFRG
# with calibrated diagnosis pipeline.
# ============================================================
# Required files in the same directory:
#   TUfRG_model.py
#   TUfRG_runner.py
#   TUfRG_instability.py
#   TUfRG_formfactor_from_modes.py
# Optional:
#   TUfRG_realspace_from_modes.py
#
# Default submission:
#   sbatch run_smoke.sbatch
#
# Override parameters from terminal, e.g.:
#   sbatch --export=ALL,U=0.2,V=0.1,LAMBDA_NN=0.05,TAG=smoke_U0p2_V0p1 run_smoke.sbatch
# ============================================================

set -euo pipefail
mkdir -p logs

U="${U:-0.2}"
V="${V:-0.0}"
LAMBDA_NN="${LAMBDA_NN:-0.05}"
LAMBDA_NNN="${LAMBDA_NNN:-0.0}"
FILLING_MODE="${FILLING_MODE:-fixed_mu}"
MU="${MU:-0.0}"
TARGET_FILLING="${TARGET_FILLING:-0.4166666666666667}"
FILLING_NGRID="${FILLING_NGRID:-60}"

NK="${NK:-4}"
NKF="${NKF:-1}"
TU_DISTANCE="${TU_DISTANCE:-1.01}"
NITER="${NITER:-3}"
MAXVERT="${MAXVERT:-50.0}"

LAMBDA0="${LAMBDA0:-50.0}"
DLAMBDA="${DLAMBDA:--5.0}"
DLAMBDA_FAC="${DLAMBDA_FAC:-0.1}"
DLAMBDA_FAC_SCALE="${DLAMBDA_FAC_SCALE:-1.0}"

DIAG_N_MODES="${DIAG_N_MODES:-4}"
FF_TOP_COMPONENTS="${FF_TOP_COMPONENTS:-8}"
FF_MAX_ROWS="${FF_MAX_ROWS:-80}"
RUN_REALSPACE="${RUN_REALSPACE:-0}"
REALSPACE_FILTER="${REALSPACE_FILTER:---family ph --sector ph-sz --q Gamma --mode 0 --observable density,bond,current --supercell 2 2 --no-plot}"

TAG="${TAG:-pace_smoke_U${U}_V${V}_l1${LAMBDA_NN}_l2${LAMBDA_NNN}_nk${NK}_nkf${NKF}}"
OUTDIR="${OUTDIR:-km_spinful_runs}"
CONDA_ENV="${CONDA_ENV:-/storage/scratch1/4/mli826/conda_envs/diverge-test}"

echo "============================================================"
echo "Smoke job started"
echo "hostname: $(hostname)"
echo "date:     $(date)"
echo "pwd:      $(pwd)"
echo "job id:   ${SLURM_JOB_ID:-none}"
echo "============================================================"
echo "Parameters: U=${U} V=${V} lambda_nn=${LAMBDA_NN} lambda_nnn=${LAMBDA_NNN} nk=${NK} nkf=${NKF} tag=${TAG}"

module purge || true
module load anaconda3/2023.03 || module load anaconda3/2022.05.0.1 || true
source activate "$CONDA_ENV"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

python - <<'PY'
import sys, importlib.util
print("sys.executable:", sys.executable)
print("python version:", sys.version)
print("diverge spec:", importlib.util.find_spec("diverge"))
import diverge
print("diverge:", diverge.__file__)
PY

for f in TUfRG_model.py TUfRG_runner.py TUfRG_instability.py TUfRG_formfactor_from_modes.py; do
  if [[ ! -f "$f" ]]; then
    echo "ERROR: missing required file $f"
    exit 1
  fi
  python -m py_compile "$f"
done

python -u TUfRG_runner.py \
  --U "$U" \
  --V "$V" \
  --lambda-nn "$LAMBDA_NN" \
  --lambda-nnn "$LAMBDA_NNN" \
  --filling-mode "$FILLING_MODE" \
  --mu "$MU" \
  --target-filling "$TARGET_FILLING" \
  --filling-ngrid "$FILLING_NGRID" \
  --nk "$NK" \
  --nkf "$NKF" \
  --tu-distance "$TU_DISTANCE" \
  --niter "$NITER" \
  --maxvert "$MAXVERT" \
  --Lambda "$LAMBDA0" \
  --dLambda "$DLAMBDA" \
  --dLambda-fac "$DLAMBDA_FAC" \
  --dLambda-fac-scale "$DLAMBDA_FAC_SCALE" \
  --tag "$TAG" \
  --outdir "$OUTDIR"

CASE_DIR="${OUTDIR}/${TAG}"
if [[ ! -d "$CASE_DIR" || ! -f "$CASE_DIR/O.dvg" ]]; then
  echo "ERROR: CASE_DIR or O.dvg missing: ${CASE_DIR}"
  exit 1
fi

python -u TUfRG_instability.py "$CASE_DIR" \
  --q Gamma M1 M2 M3 \
  --n-modes "$DIAG_N_MODES" \
  --out instability

python -u TUfRG_formfactor_from_modes.py "$CASE_DIR" \
  --instability-dir instability \
  --out formfactor_from \
  --top-components "$FF_TOP_COMPONENTS" \
  --max-rows "$FF_MAX_ROWS"

if [[ "$RUN_REALSPACE" == "1" ]]; then
  if [[ ! -f TUfRG_realspace_from_modes.py ]]; then
    echo "ERROR: RUN_REALSPACE=1 but TUfRG_realspace_from_modes.py is missing."
    exit 1
  fi
  python -m py_compile TUfRG_realspace_from_modes.py
  # shellcheck disable=SC2086
  python -u TUfRG_realspace_from_modes.py "$CASE_DIR" \
    --instability-dir instability \
    --out realspace_from \
    $REALSPACE_FILTER
fi

echo "============================================================"
echo "Smoke job finished"
echo "Output folder: ${CASE_DIR}"
echo "============================================================"
