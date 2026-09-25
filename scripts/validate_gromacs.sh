#!/usr/bin/env bash
# FOSS validation harness: export .aqtop → GROMACS and compare energies when available.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

AQTOP="${1:-tests/fixtures/complex/dipeptide_solv.aqtop}"
OUTDIR="${2:-/tmp/amphi-gmx-val}"

mkdir -p "$OUTDIR"
uv run python tests/validation/export_for_gromacs.py "$AQTOP" "$OUTDIR"

echo "Exported to $OUTDIR:"
ls -1 "$OUTDIR"/system.gro "$OUTDIR"/system.top "$OUTDIR"/*.mdp 2>/dev/null || true

if ! command -v gmx >/dev/null 2>&1 && ! command -v gromacs >/dev/null 2>&1; then
  echo "GROMACS not installed — topology and run files ready for manual grompp."
  exit 0
fi

GMX=$(command -v gmx || command -v gromacs)
TPR="$OUTDIR/em.tpr"
echo "==> grompp"
"$GMX" grompp -f "$OUTDIR/em.mdp" -c "$OUTDIR/system.gro" -p "$OUTDIR/system.top" -o "$TPR" -maxwarn 10

echo "==> mdrun (EM validation)"
"$GMX" mdrun -deffnm "$OUTDIR/em" -nb cpu -nsteps 0

echo "==> amphiquantic vs GROMACS energy"
uv run python -c "
from pathlib import Path
from amphiquantic.validation.gromacs_energy import compare_em_energy
r = compare_em_energy(Path('$AQTOP'), Path('$OUTDIR'))
print(f\"  amphiquantic E = {r['amphi_energy']:.2f} kJ/mol\")
print(f\"  GROMACS E      = {r['gmx_energy']:.2f} kJ/mol\")
print(f\"  relative diff  = {r['relative_diff']:.3f}\")
if not r['within_rtol']:
    raise SystemExit('Energy comparison outside tolerance (rtol=0.5)')
print('Energy comparison: OK')
"

echo "GROMACS validation: grompp + mdrun + energy comparison OK"
