#!/usr/bin/env bash
set -euo pipefail
# shellcheck disable=SC1091
source "$(dirname "$0")/_env.sh"

AQTOP="${AMPHI_AQTOP:-/tmp/simple.solv.aqtop}"
if [[ ! -f "$AQTOP" ]]; then
  echo "ionize.sh: expected solvated .aqtop at $AQTOP (run solvate first)"
  exit 1
fi
amphi_bin ionize "$AQTOP" -o "${AQTOP%.aqtop}.ion.aqtop"
