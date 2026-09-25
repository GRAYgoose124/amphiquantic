#!/usr/bin/env bash
set -euo pipefail
: "${AMPI_ROOT:?AMPI_ROOT must be set by protocol.sh}"
# shellcheck disable=SC1091
source "$AMPI_ROOT/scripts/amphi_env.sh"
amphi_setup "$AMPI_ROOT"

TOP="$AMPHI_RUN_DIR/topology"
STEPS="$(amphi_py -c "
from amphiquantic.experiment.config import read_step_count
print(read_step_count('$AMPHI_EXPERIMENT_ROOT', 'minimize', 1000))
")"
amphi_bin minimize "$TOP/ionized.aqtop" -o "$TOP/minimized.aqtop" --steps "$STEPS"
