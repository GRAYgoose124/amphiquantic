#!/usr/bin/env bash
# Complex workflow: prep → dock → parameterize → solvate → ionize → MD
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

RECEPTOR="${1:?receptor PDB required}"
LIGAND="${2:-}"
OUTDIR="${3:-output/complex}"
mkdir -p "$OUTDIR"

echo "==> parameterize receptor"
bin/parameterize "$RECEPTOR" -o "$OUTDIR/receptor.aqtop" --bootstrap

if [[ -n "$LIGAND" ]]; then
  echo "==> parameterize ligand"
  bin/parameterize "$LIGAND" --from-smiles -o "$OUTDIR/ligand.aqtop" --bootstrap

  echo "==> dock ligand (optional; requires --extra docking + vina)"
  if bin/dock_ligand "$RECEPTOR" "$LIGAND" --out "$OUTDIR/poses.pdbqt" 2>/dev/null; then
    echo "==> apply docked pose"
    bin/build_complex "$OUTDIR/receptor.aqtop" "$OUTDIR/ligand.aqtop" \
      -o "$OUTDIR/complex.aqtop" --ligand-pose "$OUTDIR/poses.pdbqt"
  else
    echo "==> merge without docking"
    bin/build_complex "$OUTDIR/receptor.aqtop" "$OUTDIR/ligand.aqtop" \
      -o "$OUTDIR/complex.aqtop"
  fi
else
  cp "$OUTDIR/receptor.aqtop" "$OUTDIR/complex.aqtop"
fi

echo "==> solvate"
bin/solvate "$OUTDIR/complex.aqtop" -o "$OUTDIR/solvated.aqtop"

echo "==> ionize"
bin/ionize "$OUTDIR/solvated.aqtop" -o "$OUTDIR/ionized.aqtop"

echo "==> minimize"
bin/minimize "$OUTDIR/ionized.aqtop" -o "$OUTDIR/minimized.aqtop"

echo "==> equilibrate"
bin/equilibrate "$OUTDIR/minimized.aqtop" --steps 100 -o "$OUTDIR/equil.aqtop"

echo "==> simulate"
bin/simulate "$OUTDIR/equil.aqtop" --steps 100

echo "Complex workflow complete: $OUTDIR"
