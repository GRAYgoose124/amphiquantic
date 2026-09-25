#!/usr/bin/env bash
set -euo pipefail
: "${AMPI_ROOT:?AMPI_ROOT must be set by protocol.sh}"
# shellcheck disable=SC1091
source "$AMPI_ROOT/scripts/amphi_env.sh"
amphi_setup "$AMPI_ROOT"

TOP="$AMPHI_RUN_DIR/topology"
REC="${AMPHI_EXPERIMENT_ROOT}/inputs/receptor.pdb"
LIG="${AMPHI_EXPERIMENT_ROOT}/inputs/ligand.smi"
LIG_AQTOP="${AMPHI_EXPERIMENT_ROOT}/inputs/ligand.aqtop"

amphi_bin parameterize "$REC" -o "$TOP/receptor.aqtop" --bootstrap

if [[ -f "$LIG_AQTOP" ]]; then
  cp "$LIG_AQTOP" "$TOP/ligand.aqtop"
elif [[ -f "$LIG" ]]; then
  LIG_SMILES="$(tr -d '[:space:]' < "$LIG")"
  amphi_bin parameterize "$LIG_SMILES" --from-smiles --bootstrap -o "$TOP/ligand.aqtop"
else
  echo "ERROR: no ligand.smi or ligand.aqtop in inputs/" >&2
  exit 1
fi
