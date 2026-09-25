#!/usr/bin/env bash
# A6 — Full protein-ligand from RCSB + PubChem (optional docking)
set -euo pipefail
# shellcheck disable=SC1091
source "$(dirname "$0")/_env.sh"
OUT="${1:-output/examples/protein_ligand}"
RECEPTOR_ID="${2:-1CRN}"
LIGAND="${3:-ethanol}"
mkdir -p "$OUT"

echo "==> fetch receptor $RECEPTOR_ID"
amphi_bin fetch_structure "$RECEPTOR_ID" -o "$OUT/rcsb" --format pdb
RECEPTOR="$OUT/rcsb/${RECEPTOR_ID,,}.pdb"

echo "==> resolve ligand"
amphi_bin fetch_ligand -q "$LIGAND" -o "$OUT/ligand.smi"

bash "$AMPI_ROOT/scripts/steps/complex/build.sh" "$RECEPTOR" "$OUT/ligand.smi" "$OUT"
echo "A6 complete: $OUT"
