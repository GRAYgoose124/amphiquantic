#!/usr/bin/env bash
set -euo pipefail
: "${AMPI_ROOT:?AMPI_ROOT must be set by protocol.sh}"
# shellcheck disable=SC1091
source "$AMPI_ROOT/scripts/amphi_env.sh"
amphi_setup "$AMPI_ROOT"

TOP="$AMPHI_RUN_DIR/topology"
APO="${AMPHI_EXPERIMENT_ROOT}/inputs/receptor_apo.pdb"
LIG="${AMPHI_EXPERIMENT_ROOT}/inputs/ligand.smi"
LIG_AQTOP="${AMPHI_EXPERIMENT_ROOT}/inputs/ligand.aqtop"

REC_MODE="$(amphi_py -c "
import yaml
from pathlib import Path
cfg = yaml.safe_load((Path('$AMPHI_EXPERIMENT_ROOT') / 'config.yaml').read_text())
print(cfg.get('parameterization', {}).get('receptor', 'protein_ff'))
")"
LIG_MODE="$(amphi_py -c "
import yaml
from pathlib import Path
cfg = yaml.safe_load((Path('$AMPHI_EXPERIMENT_ROOT') / 'config.yaml').read_text())
print(cfg.get('parameterization', {}).get('ligand', 'openff'))
")"

if [[ "$REC_MODE" == "protein_ff" ]]; then
  amphi_bin parameterize "$APO" -o "$TOP/receptor.aqtop" --protein
elif [[ "$REC_MODE" == "bootstrap" ]]; then
  amphi_bin parameterize "$APO" -o "$TOP/receptor.aqtop" --bootstrap
else
  amphi_bin parameterize "$APO" -o "$TOP/receptor.aqtop" --openff
fi

if [[ -f "$LIG_AQTOP" ]]; then
  cp "$LIG_AQTOP" "$TOP/ligand.aqtop"
elif [[ -f "$LIG" ]]; then
  LIG_SMILES="$(tr -d '[:space:]' < "$LIG")"
  if [[ "$LIG_MODE" == "openff" ]]; then
    amphi_bin parameterize "$LIG_SMILES" --from-smiles --openff -o "$TOP/ligand.aqtop"
  else
    amphi_bin parameterize "$LIG_SMILES" --from-smiles --bootstrap -o "$TOP/ligand.aqtop"
  fi
else
  echo "ERROR: no ligand.smi or ligand.aqtop in inputs/" >&2
  exit 1
fi
