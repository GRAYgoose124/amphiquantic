#!/usr/bin/env bash
# FOSS validation harness: export structures for GROMACS comparison.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

AQTOP="${1:-tests/fixtures/methane.aqtop}"
OUTDIR="${2:-/tmp/amphi-gmx-val}"

mkdir -p "$OUTDIR"
uv run python tests/validation/export_for_gromacs.py "$AQTOP" "$OUTDIR"

if ! command -v gmx >/dev/null 2>&1 && ! command -v gromacs >/dev/null 2>&1; then
  echo "GROMACS not installed — exported coordinates to $OUTDIR/system.gro"
  echo "Tier B: install GROMACS and compare single-point energy manually."
  exit 0
fi

GMX=$(command -v gmx || command -v gromacs)
echo "Exported $OUTDIR/system.gro"
echo "Next: build matching .top and run:"
echo "  $GMX grompp -f em.mdp -c $OUTDIR/system.gro -p $OUTDIR/system.top -o $OUTDIR/tpr"
echo "  $GMX mdrun -deffnm $OUTDIR/em -nb cpu"
