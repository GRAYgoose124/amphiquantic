#!/usr/bin/env bash
# A2 — NaCl(aq) from water box + ionize
set -euo pipefail
# shellcheck disable=SC1091
source "$(dirname "$0")/_env.sh"
OUT="${1:-output/examples/nacl_aq}"
mkdir -p "$OUT"

WATER="${2:-tests/fixtures/applications/water_box/solvated.aqtop}"
cp "$WATER" "$OUT/solvated.aqtop"

echo "==> ionize (add counter-ions)"
amphi_bin ionize "$OUT/solvated.aqtop" -o "$OUT/ionized.aqtop"

echo "==> GROMACS export"
amphi_py tests/validation/export_for_gromacs.py "$OUT/ionized.aqtop" "$OUT/gmx"

amphi_py -c "
from rustquantic import load_topology
t = load_topology('$OUT/ionized.aqtop')
print(f'atoms={t.num_atoms()} net_charge={t.net_charge():.6f}')
"
echo "A2 complete: $OUT"
