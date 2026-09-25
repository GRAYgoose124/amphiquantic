#!/usr/bin/env bash
set -euo pipefail
# shellcheck disable=SC1091
source "$(dirname "$0")/_env.sh"

AQTOP="${AMPHI_AQTOP:-/tmp/simple.solv.aqtop}"
if [[ ! -f "$AQTOP" ]]; then
  echo "simulate.sh: expected .aqtop at $AQTOP"
  exit 1
fi
amphi_bin simulate "$AQTOP" --steps 10
