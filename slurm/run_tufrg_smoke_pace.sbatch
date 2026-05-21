#!/bin/bash
#SBATCH --job-name=tufrg_smoke
#SBATCH --account=gts-ikimchi3
#SBATCH --qos=inferno
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=miaoli@gatech.edu


set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

# -------------------------
# User-overridable parameters
# -------------------------
U="${U:-0.2}"
V="${V:-0.0}"
LAMBDA_NN="${LAMBDA_NN:-0.05}"
LAMBDA_NNN="${LAMBDA_NNN:-0.0}"
FILLING_MODE="${FILLING_MODE:-fixed_mu}"
MU="${MU:-0.0}"
TARGET_FILLING="${TARGET_FILLING:-0.4166666666666667}"
FILLING_NGRID="${FILLING_NGRID:-120}"

NK="${NK:-4}"
NKF="${NKF:-1}"
TU_DISTANCE="${TU_DISTANCE:-1.01}"
NITER="${NITER:-3}"
MAXVERT="${MAXVERT:-50.0}"

LAMBDA0="${LAMBDA0:-50.0}"
DLAMBDA="${DLAMBDA:--5.0}"
DLAMBDA_FAC="${DLAMBDA_FAC:-0.1}"
DLAMBDA_FAC_SCALE="${DLAMBDA_FAC_SCALE:-1.0}"

TAG="${TAG:-pace_smoke_U${U}_V${V}_l1${LAMBDA_NN}_l2${LAMBDA_NNN}_nk${NK}_nkf${NKF}}"
OUTDIR="${OUTDIR:-km_spinful_runs}"
CONDA_ENV="${CONDA_ENV:-/storage/scratch1/4/mli826/conda_envs/diverge-test}"
RUN_DIAGNOSIS="${RUN_DIAGNOSIS:-1}"

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

module load anaconda3
conda activate "$CONDA_ENV"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

for f in TUfRG_model.py TUfRG_runner.py; do
  if [[ ! -f "$f" ]]; then
    echo "ERROR: missing required file $f in $(pwd)"
    exit 1
  fi
done

python - <<'PY'
import sys, platform, importlib.util
print("sys.executable:", sys.executable)
print("python version:", sys.version)
print("platform:", platform.platform())
spec = importlib.util.find_spec("diverge")
print("diverge spec:", spec)
import diverge
print("diverge:", diverge.__file__)
PY

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
  --tu-channel-calc-project 0
  --vertex-convention official_s1minus1
)

echo "============================================================"
echo "Running flow:"
printf ' %q' "${RUN_CMD[@]}"
echo
echo "============================================================"
"${RUN_CMD[@]}"

CASE_DIR="${OUTDIR}/${TAG}"

if [[ "$RUN_DIAGNOSIS" == "1" && -d "$CASE_DIR" && -f TUfRG_instability.py ]]; then
  echo "============================================================"
  echo "Running Level-1 diagnosis on ${CASE_DIR}"
  echo "============================================================"
  python -u TUfRG_instability.py "$CASE_DIR" --q Gamma M1 M2 M3 --n-modes 4 --out instability

  if [[ -f TUfRG_formfactor_from_modes.py ]]; then
    echo "============================================================"
    echo "Running Level-2 form-factor diagnosis on ${CASE_DIR}"
    echo "============================================================"
    python -u TUfRG_formfactor_from_modes.py "$CASE_DIR" --instability-dir instability --out formfactor --top-components 12 --max-rows 80
  fi
else
  echo "Skipping diagnosis. RUN_DIAGNOSIS=${RUN_DIAGNOSIS}, CASE_DIR=${CASE_DIR}"
fi

echo "============================================================"
echo "Job finished"
echo "date: $(date)"
echo "Output folder: ${CASE_DIR}"
echo "============================================================"
