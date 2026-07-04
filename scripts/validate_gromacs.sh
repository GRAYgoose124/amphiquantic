#!/usr/bin/env bash
# FOSS validation harness: compare amphiquantic .aqtop energy to GROMACS when available.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

AQTOP="${1:-tests/fixtures/methane.aqtop}"
OUTDIR="${2:-/tmp/amphi-gmx-val}"

if ! command -v gmx >/dev/null 2>&1 && ! command -v gromacs >/dev/null 2>&1; then
  echo "GROMACS not installed — skipping CLI validation (export-only mode)"
  mkdir -p "$OUTDIR"
  uv run python tests/validation/export_for_gromacs.py "$AQTOP" "$OUTDIR"
  exit 0
fi

GMX=$(command -v gmx || command -v gromacs)
mkdir -p "$OUTDIR"
uv run python tests/validation/export_for_gromacs.py "$AQTOP" "$OUTDIR"

echo "GROMACS validation stub: structures exported to $OUTDIR"
echo "Run manual grompp/mdrun against $OUTDIR/system.gro when topology export is extended."
