#!/usr/bin/env bash
# A6-lite — Dipeptide + ethanol ligand (no docking)
set -euo pipefail
# shellcheck disable=SC1091
source "$(dirname "$0")/_env.sh"
OUT="${1:-output/examples/complex_lite}"
RECEPTOR="${2:-tests/fixtures/complex/ala_dipeptide.pdb}"
LIGAND="${3:-CCO}"
mkdir -p "$OUT"

bash "$AMPI_ROOT/scripts/steps/complex/build.sh" "$RECEPTOR" "$LIGAND" "$OUT"
echo "A6-lite complete: $OUT"
