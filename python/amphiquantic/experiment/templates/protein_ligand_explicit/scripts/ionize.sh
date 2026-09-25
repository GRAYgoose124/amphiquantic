#!/usr/bin/env bash
set -euo pipefail
: "${AMPI_ROOT:?AMPI_ROOT must be set by protocol.sh}"
# shellcheck disable=SC1091
source "$AMPI_ROOT/scripts/amphi_env.sh"
amphi_setup "$AMPI_ROOT"

TOP="$AMPHI_RUN_DIR/topology"
amphi_bin ionize "$TOP/solvated.aqtop" -o "$TOP/ionized.aqtop"
