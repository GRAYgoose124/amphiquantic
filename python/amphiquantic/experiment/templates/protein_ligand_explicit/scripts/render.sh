#!/usr/bin/env bash
set -euo pipefail
: "${AMPI_ROOT:?AMPI_ROOT must be set by protocol.sh}"
# shellcheck disable=SC1091
source "$AMPI_ROOT/scripts/amphi_env.sh"
amphi_setup "$AMPI_ROOT"

if [[ "${AMPHI_SKIP_RENDER:-0}" == "1" ]]; then
  echo "==> skip render: AMPHI_SKIP_RENDER=1"
  exit 0
fi

TRJ="$AMPHI_RUN_DIR/trajectories/md.aqtrj"
TOP="$AMPHI_RUN_DIR/topology/equil.aqtop"
OUT="$AMPHI_RUN_DIR/render/movie.mp4"

if [[ ! -f "$TRJ" ]]; then
  echo "==> skip render: no md.aqtrj"
  exit 0
fi

amphi_bin render_traj "$TRJ" "$TOP" -o "$OUT" --stride 50 --fps 12 --solute-only
