#!/usr/bin/env bash
set -euo pipefail
: "${AMPI_ROOT:?AMPI_ROOT must be set by protocol.sh}"
# shellcheck disable=SC1091
source "$AMPI_ROOT/scripts/amphi_env.sh"
amphi_setup "$AMPI_ROOT"

TOP="$AMPHI_RUN_DIR/topology"
amphi_bin prepare_implicit "$TOP/complex.aqtop" -o "$TOP/implicit.aqtop"
amphi_bin ionize "$TOP/implicit.aqtop" -o "$TOP/ionized.aqtop"
