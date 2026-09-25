#!/usr/bin/env bash
# A1 — Pure TIP3P water box example
set -euo pipefail
# shellcheck disable=SC1091
source "$(dirname "$0")/_env.sh"
OUT="${1:-output/examples/water_box}"
mkdir -p "$OUT"

echo "==> solvate empty box (TIP3P grid)"
amphi_py -c "
from amphiquantic.topology.io import save_topology
from amphiquantic.topology.solvate import solvate_topology
from amphiquantic.validation.physics import density_g_cm3
empty = {'version': 1, 'metadata': {}, 'box': {'lx': 0, 'ly': 0, 'lz': 0, 'pbc': False},
         'atoms': [], 'bonds': [], 'angles': [], 'dihedrals': [], 'impropers': [], 'exclusions': []}
solv = solvate_topology(empty, padding=1.2, spacing=0.31)
save_topology('$OUT/solvated.aqtop', solv)
print(f'atoms={len(solv[\"atoms\"])} density={density_g_cm3(solv):.3f} g/cm3')
"

echo "==> ionize"
amphi_bin ionize "$OUT/solvated.aqtop" -o "$OUT/ionized.aqtop"

echo "==> minimize"
amphi_bin minimize "$OUT/ionized.aqtop" -o "$OUT/minimized.aqtop" --steps 20

echo "==> equilibrate"
amphi_bin equilibrate "$OUT/minimized.aqtop" --steps 50 -o "$OUT/equil.aqtop" \
  --trajectory "$OUT/equil.aqtrj"

amphi_py -c "
from amphiquantic.validation.trajectory_analysis import parse_aqtrj, mean_temperature_from_traj
frames = parse_aqtrj('$OUT/equil.aqtrj')
natoms = frames[0].natoms if frames else 0
print(f'frames={len(frames)} T_avg={mean_temperature_from_traj(frames, natoms):.1f} K')
"
echo "A1 complete: $OUT"
