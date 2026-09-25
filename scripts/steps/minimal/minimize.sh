#!/usr/bin/env bash
set -euo pipefail
# shellcheck disable=SC1091
source "$(dirname "$0")/_env.sh"

AQTOP="${AMPHI_AQTOP:-/tmp/simple.solv.ion.aqtop}"
if [[ ! -f "$AQTOP" ]]; then
  AQTOP="${AMPHI_AQTOP:-/tmp/simple.solv.aqtop}"
fi
amphi_bin minimize "$AQTOP" --steps 20
