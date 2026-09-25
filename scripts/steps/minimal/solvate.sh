#!/usr/bin/env bash
set -euo pipefail
# shellcheck disable=SC1091
source "$(dirname "$0")/_env.sh"

AQTOP="${AMPHI_AQTOP:-/tmp/simple.aqtop}"
OUT="${AQTOP%.aqtop}.solv.aqtop"
amphi_bin solvate "$AQTOP" -o "$OUT"
export AMPHI_AQTOP="$OUT"
