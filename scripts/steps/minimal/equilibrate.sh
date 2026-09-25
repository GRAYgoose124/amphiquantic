#!/usr/bin/env bash
set -euo pipefail
# shellcheck disable=SC1091
source "$(dirname "$0")/_env.sh"

AQTOP="${AMPHI_AQTOP:-/tmp/simple.solv.ion.min.aqtop}"
amphi_bin equilibrate "$AQTOP" --steps 10 --restrain --restraint-k 500 2>/dev/null || \
  amphi_bin equilibrate "${AQTOP%.min.aqtop}.aqtop" --steps 10 2>/dev/null || true
