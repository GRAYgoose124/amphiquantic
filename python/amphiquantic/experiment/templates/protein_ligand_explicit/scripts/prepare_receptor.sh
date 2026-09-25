#!/usr/bin/env bash
set -euo pipefail
: "${AMPI_ROOT:?AMPI_ROOT must be set by protocol.sh}"
# shellcheck disable=SC1091
source "$AMPI_ROOT/scripts/amphi_env.sh"
amphi_setup "$AMPI_ROOT"

HOLO="${AMPHI_EXPERIMENT_ROOT}/inputs/receptor.pdb"
APO="${AMPHI_EXPERIMENT_ROOT}/inputs/receptor_apo.pdb"

if [[ ! -f "$HOLO" ]]; then
  echo "ERROR: missing holo receptor.pdb (run fetch first)" >&2
  exit 1
fi

amphi_bin prepare_receptor "$HOLO" -o "$APO" --protonate
