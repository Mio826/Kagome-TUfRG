#!/usr/bin/env bash
# Usage: bash run_gl.sh ARCHIVE.zip SNAPSHOT [MAP.json]
# Environment overrides: T, MESH, NMAT, OUT
set -euo pipefail
if [[ $# -lt 2 ]]; then
  echo "Usage: bash run_gl.sh ARCHIVE.zip SNAPSHOT [FF_MAP.json]" >&2
  exit 2
fi
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARCHIVE="$(realpath "$1")"
SNAPSHOT="$2"
T="${T:-0.12}"
MESH="${MESH:-12}"
NMAT="${NMAT:-256}"
OUT="${OUT:-results/${SNAPSHOT}_T${T}_N${MESH}.json}"
MAP="${3:-}"
TMP=""
if [[ -z "$MAP" ]]; then
  # Nk=30 Stage3 diagnostic ZIP already contains its matching FF index map.
  TMP="$(mktemp --suffix=.json)"
  trap 'rm -f "$TMP"' EXIT
  python - "$ARCHIVE" "$TMP" <<'PY'
import sys,zipfile
with zipfile.ZipFile(sys.argv[1]) as z:
    if 'ff_map.json' not in z.namelist():
        raise SystemExit('Archive has no ff_map.json; pass the matching map as argument 3')
    open(sys.argv[2],'wb').write(z.read('ff_map.json'))
PY
  MAP="$TMP"
else
  MAP="$(realpath "$MAP")"
fi
cd "$ROOT"
python gl_gateB_reusable.py \
  --map "$MAP" --archive "$ARCHIVE" --snapshot "$SNAPSHOT" \
  --T "$T" --mesh "$MESH" --nmat "$NMAT" --out "$OUT"
