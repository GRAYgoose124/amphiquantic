#!/usr/bin/env bash
# A3 — Ethanol in explicit water
set -euo pipefail
# shellcheck disable=SC1091
source "$(dirname "$0")/_env.sh"
OUT="${1:-output/examples/ethanol_solv}"
mkdir -p "$OUT"

echo "==> parameterize ethanol"
amphi_bin parameterize "CCO" --from-smiles -o "$OUT/ethanol.aqtop" --bootstrap

echo "==> solvate"
amphi_bin solvate "$OUT/ethanol.aqtop" -o "$OUT/solvated.aqtop"

echo "==> ionize"
amphi_bin ionize "$OUT/solvated.aqtop" -o "$OUT/ionized.aqtop"

echo "==> minimize + equilibrate"
amphi_bin minimize "$OUT/ionized.aqtop" -o "$OUT/minimized.aqtop" --steps 20
amphi_bin equilibrate "$OUT/minimized.aqtop" --steps 30 -o "$OUT/equil.aqtop" \
  --restrain --restraint-k 500

amphi_py -c "
from rustquantic import load_topology
t = load_topology('$OUT/ionized.aqtop')
print(f'atoms={t.num_atoms()} net_charge={t.net_charge():.4f}')
"
echo "A3 complete: $OUT"
