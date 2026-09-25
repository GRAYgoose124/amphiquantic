#!/usr/bin/env bash
set -euo pipefail
: "${AMPI_ROOT:?AMPI_ROOT must be set by protocol.sh}"
# shellcheck disable=SC1091
source "$AMPI_ROOT/scripts/amphi_env.sh"
amphi_setup "$AMPI_ROOT"

REC="${AMPHI_EXPERIMENT_ROOT}/inputs/receptor.pdb"
if [[ ! -f "$REC" ]]; then
  amphi_py -c "
import yaml
from pathlib import Path
from amphiquantic.structure import fetch_pdb, normalize_pdb_id
root = Path('$AMPHI_EXPERIMENT_ROOT')
cfg = yaml.safe_load((root / 'config.yaml').read_text())
ref = cfg.get('inputs', {}).get('receptor', '')
if len(str(ref)) == 4 and str(ref).isalnum():
    pdb_id = normalize_pdb_id(str(ref))
    text = fetch_pdb(pdb_id)
    (root / 'inputs' / 'receptor.pdb').write_text(text)
    print(f'fetched receptor {pdb_id}')
else:
    raise SystemExit(f'receptor input missing: {ref}')
"
fi
