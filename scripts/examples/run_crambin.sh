#!/usr/bin/env bash
# A5 — Crambin (1CRN) from RCSB
set -euo pipefail
# shellcheck disable=SC1091
source "$(dirname "$0")/_env.sh"
OUT="${1:-output/examples/crambin}"
PDB_ID="${2:-1CRN}"
mkdir -p "$OUT"

echo "==> fetch $PDB_ID"
amphi_bin fetch_structure "$PDB_ID" -o "$OUT/rcsb" --format pdb
PDB="$OUT/rcsb/${PDB_ID,,}.pdb"

echo "==> parameterize"
amphi_bin parameterize "$PDB" -o "$OUT/protein.aqtop" --bootstrap

echo "==> solvate + ionize"
amphi_bin solvate "$OUT/protein.aqtop" -o "$OUT/solvated.aqtop"
amphi_bin ionize "$OUT/solvated.aqtop" -o "$OUT/ionized.aqtop"

echo "==> minimize"
amphi_bin minimize "$OUT/ionized.aqtop" -o "$OUT/minimized.aqtop" --steps 20

amphi_py -c "
from rustquantic import load_topology
t = load_topology('$OUT/ionized.aqtop')
print(f'atoms={t.num_atoms()} Q={t.net_charge():.4f}')
"
echo "A5 complete: $OUT"
