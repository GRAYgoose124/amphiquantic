#!/usr/bin/env bash
set -euo pipefail
: "${AMPI_ROOT:?AMPI_ROOT must be set by protocol.sh}"
# shellcheck disable=SC1091
source "$AMPI_ROOT/scripts/amphi_env.sh"
amphi_setup "$AMPI_ROOT"

DOCK="$AMPHI_RUN_DIR/docking"
REC="${AMPHI_EXPERIMENT_ROOT}/inputs/receptor_apo.pdb"
REF="${AMPHI_EXPERIMENT_ROOT}/inputs/reference_ligand.pdb"
LIG="${AMPHI_EXPERIMENT_ROOT}/inputs/ligand.smi"

if [[ ! -f "$LIG" ]]; then
  echo "==> skip dock: no ligand.smi"
  exit 0
fi

amphi_py -c "
from pathlib import Path
from amphiquantic.docking.binding_site import auto_docking_box, write_box_json
ref = Path('$REF')
rec = '$REC'
box = auto_docking_box(rec, reference_ligand=ref if ref.is_file() else None)
write_box_json(box, '$DOCK/box.json')
"

DOCK_ARGS=(--out "$DOCK/poses.pdbqt")
if [[ -f "$REF" ]]; then
  DOCK_ARGS+=(--box from-ligand --reference-ligand "$REF")
else
  DOCK_ARGS+=(--box auto)
fi

if ! amphi_bin dock_ligand "$REC" "$LIG" "${DOCK_ARGS[@]}"; then
  if [[ "${AMPHI_REQUIRE_DOCK:-0}" == "1" ]]; then
    echo "ERROR: docking required but failed (need uv sync --extra docking and vina/smina on PATH)" >&2
    exit 1
  fi
  echo "==> dock skipped (set AMPHI_REQUIRE_DOCK=1 to fail hard)"
fi
