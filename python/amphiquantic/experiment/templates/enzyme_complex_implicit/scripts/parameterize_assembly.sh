#!/usr/bin/env bash
set -euo pipefail
: "${AMPI_ROOT:?AMPI_ROOT must be set by protocol.sh}"
# shellcheck disable=SC1091
source "$AMPI_ROOT/scripts/amphi_env.sh"
amphi_setup "$AMPI_ROOT"

TOP="$AMPHI_RUN_DIR/topology"
INPUTS="${AMPHI_EXPERIMENT_ROOT}/inputs"
REC="$INPUTS/receptor.pdb"

amphi_bin parameterize "$REC" -o "$TOP/receptor.aqtop" --bootstrap

if [[ -f "$INPUTS/ligand.aqtop" ]]; then
  cp "$INPUTS/ligand.aqtop" "$TOP/ligand.aqtop"
elif [[ -f "$INPUTS/ligand.smi" ]]; then
  LIG_SMILES="$(tr -d '[:space:]' < "$INPUTS/ligand.smi")"
  amphi_bin parameterize "$LIG_SMILES" --from-smiles --bootstrap -o "$TOP/ligand.aqtop"
fi

if [[ -f "$INPUTS/cofactor.smi" ]]; then
  COFACTOR_SMILES="$(tr -d '[:space:]' < "$INPUTS/cofactor.smi")"
  amphi_bin parameterize "$COFACTOR_SMILES" --from-smiles --bootstrap -o "$TOP/cofactor.aqtop"
fi

if [[ -f "$INPUTS/chain_b.pdb" ]]; then
  amphi_bin parameterize "$INPUTS/chain_b.pdb" -o "$TOP/chain_b.aqtop" --bootstrap
  if amphi_bin dock_assembly "$REC" "$INPUTS/chain_b.pdb" -o "$TOP/assembly.pdb" 2>/dev/null; then
    amphi_bin parameterize "$TOP/assembly.pdb" -o "$TOP/receptor.aqtop" --bootstrap
  fi
fi
