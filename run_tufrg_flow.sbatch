#!/bin/bash
#SBATCH -J kagome_tufrg
#SBATCH -A gts-ikimchi3
#SBATCH -p cpu-medium
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH -t 24:00:00
#SBATCH -o logs/kagome_flow_%j.out
#SBATCH -e logs/kagome_flow_%j.err
set -euo pipefail

RUNS=/storage/home/hcoda1/4/mli826/scratch/divERGe_test_run/runs
PY=/storage/scratch1/4/mli826/conda_envs/diverge-test/bin/python
cd "$RUNS"

: "${TAG:?Set a unique TAG before sbatch}"
: "${OUTROOT:?Set OUTROOT for NEW flow outputs}"
: "${FILLING_MODE:?Choose fixed_filling or fixed_mu explicitly}"
: "${NITER:?Choose NITER explicitly from the intended flow protocol}"
: "${MAXVERT:?Choose MAXVERT explicitly from the intended flow protocol}"
: "${NK:?Choose nk explicitly}"
: "${NKF:?Choose nkf explicitly}"
: "${TU_DISTANCE:?Choose TU FF distance explicitly}"
case "$TAG" in *[!a-zA-Z0-9_.-]*|'') echo 'TAG may only use letters/digits/._-' >&2; exit 2;; esac
[[ "$NITER" =~ ^[0-9]+$ ]] && (( NITER > 0 )) || { echo 'NITER must be positive integer' >&2; exit 2; }
[[ "$FILLING_MODE" == fixed_filling || "$FILLING_MODE" == fixed_mu ]] || { echo 'Invalid filling mode' >&2; exit 2; }
[[ "${LAMBDA_RASHBA:-0}" == 0 || "${LAMBDA_RASHBA:-0}" == 0.0 ]] || { echo 'Diagnosis v1 does not support Rashba' >&2; exit 2; }

mkdir -p "$OUTROOT"
OUTROOT="$(cd "$OUTROOT" && pwd -P)"
case "$OUTROOT" in
  /storage/scratch1/4/mli826/divERGe_test_run/runs/*|/storage/home/hcoda1/4/mli826/scratch/divERGe_test_run/runs/*) ;;
  *) echo 'OUTROOT must be a NEW subdirectory of this project runs folder' >&2; exit 2;;
esac
if [[ -e "$OUTROOT/$TAG" ]]; then
    echo "REFUSING to overwrite existing case: $OUTROOT/$TAG" >&2
    exit 2
fi

ARGS=(
  --tag "$TAG" --outdir "$OUTROOT"
  --U "${U:-0}" --V1 "${V1:-0}" --V2 "${V2:-0}" --V3 "${V3:-0}"
  --lambda-nn "${LAMBDA_NN:-0}" --lambda-nnn "${LAMBDA_NNN:-0}" --lambda-rashba 0
  --t "${T:-1}" --mu "${MU:-0}"
  --filling-mode "$FILLING_MODE" --target-filling "${TARGET_FILLING:-0.4166666666666667}"
  --filling-ngrid "${FILLING_NGRID:-120}"
  --nk "$NK" --nkf "$NKF" --tu-distance "$TU_DISTANCE"
  --niter "$NITER" --maxvert "$MAXVERT"
  --dLambda-fac "${DLAMBDA_FAC:-0.1}" --dLambda-fac-scale "${DLAMBDA_FAC_SCALE:-1.0}"
  --flow-mode tu --channels PCD --tu-channel-calc-project 0
  --vertex-convention official_s1minus1
)
if [[ -n "${LAMBDA0:-}" ]]; then ARGS+=(--Lambda "$LAMBDA0"); fi
if [[ -n "${DLAMBDA0:-}" ]]; then ARGS+=(--dLambda "$DLAMBDA0"); fi
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=8
printf 'FRG_INPUT %q ' "$PY" "$RUNS/TUfRG_runner.py" "${ARGS[@]}"
printf '\nOUTPUT_DIR %s\n' "$OUTROOT/$TAG"
"$PY" "$RUNS/TUfRG_runner.py" "${ARGS[@]}"
[[ -s "$OUTROOT/$TAG/O.dvg" && -s "$OUTROOT/$TAG/T.dvg" && -s "$OUTROOT/$TAG/metadata.json" ]]
echo 'FRG_FILES_WRITTEN — run Diagnosis as the separate next step'
