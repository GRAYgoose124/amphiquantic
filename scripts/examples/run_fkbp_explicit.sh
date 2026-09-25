#!/usr/bin/env bash
# A9 — FKBP12 (1FKF) + FK506 explicit TIP3P experiment with docking + MD render
set -euo pipefail
# shellcheck disable=SC1091
source "$(dirname "$0")/_env.sh"

export AMPHI_UV_EXTRAS="${AMPHI_UV_EXTRAS:-docking,openff,protein}"

echo "==> sync extras: $AMPHI_UV_EXTRAS"
uv sync --extra docking --extra openff --extra protein
uv run maturin develop --manifest-path rust/Cargo.toml

EXP="$(uv run python bin/new_experiment \
  --protocol protein_ligand_explicit \
  --receptor 1FKF \
  --ligand-from-pdb 1FKF:FK5 \
  --ligand-resname FK5)"

echo "==> experiment: $EXP"
uv run python bin/run_experiment "$EXP"
echo "A9 complete: $EXP"
