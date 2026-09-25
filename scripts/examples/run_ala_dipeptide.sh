#!/usr/bin/env bash
# A4 — Alanine dipeptide in water
set -euo pipefail
# shellcheck disable=SC1091
source "$(dirname "$0")/_env.sh"
OUT="${1:-output/examples/ala_dipeptide}"
PDB="${2:-tests/fixtures/complex/ala_dipeptide.pdb}"
mkdir -p "$OUT"

echo "==> parameterize dipeptide"
amphi_bin parameterize "$PDB" -o "$OUT/dipeptide.aqtop" --bootstrap

echo "==> solvate + ionize"
amphi_bin solvate "$OUT/dipeptide.aqtop" -o "$OUT/solvated.aqtop"
amphi_bin ionize "$OUT/solvated.aqtop" -o "$OUT/ionized.aqtop"

echo "==> minimize"
amphi_bin minimize "$OUT/ionized.aqtop" -o "$OUT/minimized.aqtop" --steps 30

echo "==> equilibrate (restrained)"
amphi_bin equilibrate "$OUT/minimized.aqtop" --steps 50 -o "$OUT/equil.aqtop" \
  --trajectory "$OUT/equil.aqtrj" --restrain --restraint-k 500

echo "==> short production MD"
amphi_bin simulate "$OUT/equil.aqtop" --steps 50 --trajectory "$OUT/md.aqtrj"

amphi_py -c "
from amphiquantic.validation.trajectory_analysis import parse_aqtrj, energy_drift, mean_temperature_from_traj
from rustquantic import load_topology
frames = parse_aqtrj('$OUT/equil.aqtrj')
natoms = frames[0].natoms if frames else load_topology('$OUT/ionized.aqtop').num_atoms()
print(f'equil_frames={len(frames)} T={mean_temperature_from_traj(frames, natoms):.0f}K drift={energy_drift(frames):.2f} kJ/mol/frame')
t = load_topology('$OUT/ionized.aqtop')
print(f'atoms={natoms} Q={t.net_charge():.4f}')
"
echo "A4 complete: $OUT"
