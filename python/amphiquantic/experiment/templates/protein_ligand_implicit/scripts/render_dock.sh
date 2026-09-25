#!/usr/bin/env bash
set -euo pipefail
: "${AMPI_ROOT:?AMPI_ROOT must be set by protocol.sh}"
# shellcheck disable=SC1091
source "$AMPI_ROOT/scripts/amphi_env.sh"
amphi_setup "$AMPI_ROOT"

if [[ "${AMPHI_SKIP_RENDER:-0}" == "1" ]]; then
  echo "==> skip render_dock: AMPHI_SKIP_RENDER=1"
  exit 0
fi

COMPLEX="$AMPHI_RUN_DIR/topology/complex.aqtop"
OUT="$AMPHI_RUN_DIR/render/dock_preview.mp4"

if [[ ! -f "$COMPLEX" ]]; then
  echo "==> skip render_dock: no complex.aqtop"
  exit 0
fi

amphi_bin render_structure "$COMPLEX" -o "$OUT" --frames 36 --fps 12
