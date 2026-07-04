#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"

PDB="${1:-tests/fixtures/ala.pdb}"
AQTOP="${2:-/tmp/simple.aqtop}"

bin/parameterize "$PDB" -o "$AQTOP" --bootstrap
bin/solvate "$AQTOP" -o "${AQTOP%.aqtop}.solv.aqtop"
