#!/bin/bash
#SBATCH --job-name=tufrg_real
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

# ============================================================
# PACE sbatch script: production-style spinful kagome Kane-Mele TUFRG run
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
#   sbatch run.sbatch
#
# Override parameters from terminal, e.g.:
#   sbatch --export=ALL,U=1.0,V=0.5,LAMBDA_NN=0.05,NK=18,NKF=3,TAG=U1_V0p5_lnn005_nk18 run.sbatch
# ============================================================

set -euo pipefail
mkdir -p logs

# -------------------------
# User-overridable physics parameters
# -------------------------
U="${U:-1.0}"
V="${V:-0.0}"
LAMBDA_NN="${LAMBDA_NN:-0.05}"
LAMBDA_NNN="${LAMBDA_NNN:-0.0}"
FILLING_MODE="${FILLING_MODE:-fixed_mu}"
MU="${MU:-0.0}"
TARGET_FILLING="${TARGET_FILLING:-0.4166666666666667}"
FILLING_NGRID="${FILLING_NGRID:-120}"

# -------------------------
# User-overridable numerical parameters
# -------------------------
NK="${NK:-12}"
NKF="${NKF:-2}"
TU_DISTANCE="${TU_DISTANCE:-1.01}"
NITER="${NITER:-80}"
MAXVERT="${MAXVERT:-50.0}"

LAMBDA0="${LAMBDA0:-50.0}"
DLAMBDA="${DLAMBDA:--5.0}"
DLAMBDA_FAC="${DLAMBDA_FAC:-0.1}"
DLAMBDA_FAC_SCALE="${DLAMBDA_FAC_SCALE:-1.0}"

# Diagnosis parameters
DIAG_N_MODES="${DIAG_N_MODES:-8}"
FF_TOP_COMPONENTS="${FF_TOP_COMPONENTS:-12}"
FF_MAX_ROWS="${FF_MAX_ROWS:-120}"
RUN_REALSPACE="${RUN_REALSPACE:-0}"
REALSPACE_FILTER="${REALSPACE_FILTER:---family ph --sector ph-sz --q Gamma --mode 0 --observable density,bond,current --supercell 2 2}"

TAG="${TAG:-real_U${U}_V${V}_l1${LAMBDA_NN}_l2${LAMBDA_NNN}_nk${NK}_nkf${NKF}_d${TU_DISTANCE}}"
OUTDIR="${OUTDIR:-km_spinful_runs}"
CONDA_ENV="${CONDA_ENV:-/storage/scratch1/4/mli826/conda_envs/diverge-test}"

# -------------------------
# Environment
# -------------------------
echo "============================================================"
echo "Job started"
echo "hostname: $(hostname)"
echo "date:     $(date)"
echo "pwd:      $(pwd)"
echo "job id:   ${SLURM_JOB_ID:-none}"
echo "cpus:     ${SLURM_CPUS_PER_TASK:-unset}"
echo "============================================================"

echo "Parameters:"
echo "  U=${U} V=${V} lambda_nn=${LAMBDA_NN} lambda_nnn=${LAMBDA_NNN}"
echo "  filling_mode=${FILLING_MODE} mu=${MU} target_filling=${TARGET_FILLING}"
echo "  nk=${NK} nkf=${NKF} tu_distance=${TU_DISTANCE}"
echo "  niter=${NITER} maxvert=${MAXVERT} Lambda=${LAMBDA0} dLambda=${DLAMBDA}"
echo "  tag=${TAG} outdir=${OUTDIR}"
echo "  diag_n_modes=${DIAG_N_MODES} ff_max_rows=${FF_MAX_ROWS}"

module purge || true
module load anaconda3/2023.03 || module load anaconda3/2022.05.0.1 || true
source activate "$CONDA_ENV"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

python - <<'PY'
import sys, platform, importlib.util
print("sys.executable:", sys.executable)
print("python version:", sys.version)
print("platform:", platform.platform())
spec = importlib.util.find_spec("diverge")
print("diverge spec:", spec)
import diverge
print("diverge:", diverge.__file__)
print("has init:", hasattr(diverge, "init"), "has finalize:", hasattr(diverge, "finalize"))
PY

for f in TUfRG_model.py TUfRG_runner.py TUfRG_instability.py TUfRG_formfactor_from_modes.py; do
  if [[ ! -f "$f" ]]; then
    echo "ERROR: missing required file $f"
    exit 1
  fi
  python -m py_compile "$f"
done

RUN_CMD=(
  python -u TUfRG_runner.py
  --U "$U"
  --V "$V"
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
  --Lambda "$LAMBDA0"
  --dLambda "$DLAMBDA"
  --dLambda-fac "$DLAMBDA_FAC"
  --dLambda-fac-scale "$DLAMBDA_FAC_SCALE"
  --tag "$TAG"
  --outdir "$OUTDIR"
)

echo "============================================================"
echo "Running TUfRG flow:"
printf ' %q' "${RUN_CMD[@]}"
echo
echo "============================================================"
"${RUN_CMD[@]}"

CASE_DIR="${OUTDIR}/${TAG}"
if [[ ! -d "$CASE_DIR" || ! -f "$CASE_DIR/O.dvg" ]]; then
  echo "ERROR: CASE_DIR or O.dvg missing: ${CASE_DIR}"
  exit 1
fi

echo "============================================================"
echo "Running calibrated Level-1 instability diagnosis on ${CASE_DIR}"
echo "============================================================"
python -u TUfRG_instability.py "$CASE_DIR" \
  --q Gamma M1 M2 M3 \
  --n-modes "$DIAG_N_MODES" \
  --out instability

echo "============================================================"
echo "Running Level-2 form-factor classification from v2 modes"
echo "============================================================"
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
  echo "============================================================"
  echo "Running optional Level-3 real-space reconstruction"
  echo "REALSPACE_FILTER=${REALSPACE_FILTER}"
  echo "============================================================"
  # shellcheck disable=SC2086
  python -u TUfRG_realspace_from_modes.py "$CASE_DIR" \
    --instability-dir instability \
    --out realspace_from \
    $REALSPACE_FILTER
fi

echo "============================================================"
echo "Job finished"
echo "date: $(date)"
echo "Output folder: ${CASE_DIR}"
echo "Main diagnosis outputs:"
echo "  ${CASE_DIR}/instability/instability_summary.csv"
echo "  ${CASE_DIR}/formfactor_from/formfactor_summary.csv"
echo "============================================================"
