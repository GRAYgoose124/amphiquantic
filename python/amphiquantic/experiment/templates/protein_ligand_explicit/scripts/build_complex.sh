#!/usr/bin/env bash
set -euo pipefail
: "${AMPI_ROOT:?AMPI_ROOT must be set by protocol.sh}"
# shellcheck disable=SC1091
source "$AMPI_ROOT/scripts/amphi_env.sh"
amphi_setup "$AMPI_ROOT"

TOP="$AMPHI_RUN_DIR/topology"
DOCK="$AMPHI_RUN_DIR/docking"

if [[ -f "$DOCK/poses.pdbqt" ]]; then
  amphi_bin build_complex "$TOP/receptor.aqtop" "$TOP/ligand.aqtop" \
    -o "$TOP/complex.aqtop" --ligand-pose "$DOCK/poses.pdbqt"
else
  amphi_bin build_complex "$TOP/receptor.aqtop" "$TOP/ligand.aqtop" \
    -o "$TOP/complex.aqtop"
fi
