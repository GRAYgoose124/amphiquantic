#!/usr/bin/env bash
set -euo pipefail
EXPERIMENT_ROOT="{{EXPERIMENT_ROOT}}"
AMPI_ROOT="{{AMPI_ROOT}}"
export AMPHI_UV_EXTRAS="${AMPHI_UV_EXTRAS:-docking}"
# shellcheck disable=SC1091
source "$AMPI_ROOT/scripts/amphi_env.sh"
amphi_setup "$AMPI_ROOT"

export AMPHI_EXPERIMENT_ROOT="$EXPERIMENT_ROOT"
export AMPHI_REQUIRE_DOCK="${AMPHI_REQUIRE_DOCK:-1}"
export AMPI_ROOT

RUN_DIR="$(amphi_py -c "
from pathlib import Path
from amphiquantic.experiment.layout import new_run_dir, ensure_run_layout
from amphiquantic.experiment.manifest import read_manifest, write_run_json
root = Path('$EXPERIMENT_ROOT')
run = new_run_dir(root)
ensure_run_layout(run)
m = read_manifest(root)
write_run_json(run, experiment_id=m['id'])
print(run)
")"
export AMPHI_RUN_DIR="$RUN_DIR"

for step in fetch parameterize_assembly dock build_complex render_dock prepare_implicit minimize equilibrate simulate render; do
  echo "==> $step"
  bash "$EXPERIMENT_ROOT/scripts/${step}.sh" 2>&1 | tee "$AMPHI_RUN_DIR/logs/${step}.log"
done

echo "Enzyme complex run complete: $AMPHI_RUN_DIR"
