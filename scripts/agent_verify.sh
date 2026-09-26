#!/usr/bin/env bash
# Agent verification gate — run before claiming work is complete.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

FAST=0
GPU_ONLY=0
PYTEST_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --fast) FAST=1; shift ;;
    --gpu) GPU_ONLY=1; shift ;;
    -h|--help)
      echo "Usage: $0 [--fast] [--gpu]"
      echo "  --fast  Skip slow, openff, tier_b markers"
      echo "  --gpu   Run only gpu-marked tests (after sync/build)"
      exit 0
      ;;
    *) PYTEST_ARGS+=("$1"); shift ;;
  esac
done

echo "==> amphiquantic agent verify (root: $ROOT)"

echo "==> uv sync"
uv sync --inexact  # --inexact: keep optional extras (openff, rdkit) installed

echo "==> maturin develop"
uv run maturin develop --manifest-path rust/Cargo.toml

echo "==> cargo test"
(cd rust && PYO3_PYTHON="$ROOT/.venv/bin/python" cargo test)

MARKER_EXPR="tier_a"
if [[ "$FAST" -eq 1 ]]; then
  MARKER_EXPR="tier_a and not slow and not openff and not tier_b and not pme and not complex"
fi
if [[ "$GPU_ONLY" -eq 1 ]]; then
  MARKER_EXPR="gpu"
fi

echo "==> pytest -m \"$MARKER_EXPR\""
uv run pytest -m "$MARKER_EXPR" "${PYTEST_ARGS[@]}"

if [[ "$FAST" -eq 0 && "$GPU_ONLY" -eq 0 ]]; then
  if [[ "${AMPHI_SKIP_OPENFF:-}" != "1" ]] && uv run python -c "import openff.toolkit" 2>/dev/null; then
    echo "==> pytest -m openff"
    uv run pytest -m openff "${PYTEST_ARGS[@]}"
  else
    echo "==> skip openff (install: uv sync --extra openff)"
  fi
fi

echo ""
echo "agent_verify: PASS (marker=$MARKER_EXPR)"
