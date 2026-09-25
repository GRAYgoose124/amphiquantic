#!/usr/bin/env bash
set -euo pipefail
# shellcheck disable=SC1091
source "$(dirname "$0")/_env.sh"

PDB="${1:-tests/fixtures/ala.pdb}"
AQTOP="${2:-/tmp/simple.aqtop}"

amphi_bin parameterize "$PDB" -o "$AQTOP" --bootstrap
export AMPHI_AQTOP="$AQTOP"
