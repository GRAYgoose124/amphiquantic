#!/usr/bin/env bash
# Complex workflow: prep → dock → parameterize → solvate → ionize → MD
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/scripts/amphi_env.sh"
amphi_setup "$ROOT"

RECEPTOR="${1:-${AMPHI_RECEPTOR:?receptor PDB path or RCSB ID required}}"
LIGAND="${2:-${AMPHI_LIGAND:-}}"
OUTDIR="${3:-${AMPHI_COMPLEX_OUT:-output/complex}}"
mkdir -p "$OUTDIR"

count_atoms() {
  amphi_py -c "from rustquantic import load_topology; print(load_topology('$1').num_atoms())"
}

net_charge() {
  amphi_py -c "from rustquantic import load_topology; print(f'{load_topology(\"$1\").net_charge():.4f}')"
}

if [[ "$RECEPTOR" =~ ^[0-9][A-Za-z0-9]{3}$ ]]; then
  echo "==> fetch receptor from RCSB ($RECEPTOR)"
  amphi_bin fetch_structure "$RECEPTOR" -o "$OUTDIR/rcsb" --format pdb
  RECEPTOR="$OUTDIR/rcsb/${RECEPTOR,,}.pdb"
fi

echo "==> parameterize receptor"
if [[ "${AMPHI_USE_OPENFF:-0}" == "1" ]]; then
  amphi_bin parameterize "$RECEPTOR" -o "$OUTDIR/receptor.aqtop" --protein
else
  amphi_bin parameterize "$RECEPTOR" -o "$OUTDIR/receptor.aqtop" --bootstrap
fi
N0=$(count_atoms "$OUTDIR/receptor.aqtop")
echo "    receptor atoms: $N0"

if [[ -n "$LIGAND" ]]; then
  if [[ ! -f "$LIGAND" ]]; then
    echo "==> resolve ligand ($LIGAND)"
    amphi_bin fetch_ligand -q "$LIGAND" -o "$OUTDIR/ligand.smi"
    LIGAND="$OUTDIR/ligand.smi"
  fi
  echo "==> parameterize ligand"
  if [[ "${AMPHI_USE_OPENFF:-0}" == "1" ]]; then
    amphi_bin parameterize "$LIGAND" --from-smiles -o "$OUTDIR/ligand.aqtop" --openff
  else
    amphi_bin parameterize "$LIGAND" --from-smiles -o "$OUTDIR/ligand.aqtop" --bootstrap
  fi

  echo "==> dock ligand (optional; requires --extra docking + vina)"
  if amphi_bin dock_ligand "$RECEPTOR" "$LIGAND" --out "$OUTDIR/poses.pdbqt" 2>/dev/null; then
    echo "==> apply docked pose"
    amphi_bin build_complex "$OUTDIR/receptor.aqtop" "$OUTDIR/ligand.aqtop" \
      -o "$OUTDIR/complex.aqtop" --ligand-pose "$OUTDIR/poses.pdbqt"
  else
    echo "==> merge without docking"
    amphi_bin build_complex "$OUTDIR/receptor.aqtop" "$OUTDIR/ligand.aqtop" \
      -o "$OUTDIR/complex.aqtop"
  fi
else
  cp "$OUTDIR/receptor.aqtop" "$OUTDIR/complex.aqtop"
fi

N_COMPLEX=$(count_atoms "$OUTDIR/complex.aqtop")
echo "    complex atoms: $N_COMPLEX"

echo "==> solvate"
amphi_bin solvate "$OUTDIR/complex.aqtop" -o "$OUTDIR/solvated.aqtop"
N_SOLV=$(count_atoms "$OUTDIR/solvated.aqtop")
if [[ "$N_SOLV" -le "$N_COMPLEX" ]]; then
  echo "ERROR: solvation did not increase atom count ($N_COMPLEX -> $N_SOLV)" >&2
  exit 1
fi
echo "    solvated atoms: $N_SOLV"

echo "==> ionize"
amphi_bin ionize "$OUTDIR/solvated.aqtop" -o "$OUTDIR/ionized.aqtop"
Q=$(net_charge "$OUTDIR/ionized.aqtop")
echo "    net charge after ionize: $Q"
amphi_py -c "import sys; q=float('$Q'); sys.exit(0 if abs(q) < 0.5 else 1)" || {
  echo "ERROR: system not neutralized (|q| >= 0.5)" >&2
  exit 1
}

echo "==> minimize"
amphi_bin minimize "$OUTDIR/ionized.aqtop" -o "$OUTDIR/minimized.aqtop"

echo "==> equilibrate"
amphi_bin equilibrate "$OUTDIR/minimized.aqtop" --steps 100 \
  -o "$OUTDIR/equil.aqtop" --trajectory "$OUTDIR/equil.aqtrj" --restrain --restraint-k 500

echo "==> simulate"
amphi_bin simulate "$OUTDIR/equil.aqtop" --steps 100 \
  --trajectory "$OUTDIR/md.aqtrj" -o "$OUTDIR/md.aqtop"

echo "Complex workflow complete: $OUTDIR ($N_SOLV atoms)"
