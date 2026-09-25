#!/usr/bin/env bash
# Shared bootstrap: uv venv + rustquantic. Source from repo scripts.
# Usage: source "$ROOT/scripts/amphi_env.sh" && amphi_setup "$ROOT"
#
# Optional: AMPHI_UV_EXTRAS=comma-separated extras (e.g. docking,openff)

_amphi_uv_sync_args() {
  AMPI_UV_SYNC_ARGS=()
  local extras="${AMPHI_UV_EXTRAS:-}"
  [[ -z "$extras" ]] && return 0
  local e
  IFS=',' read -ra _extras <<< "$extras"
  for e in "${_extras[@]}"; do
    e="${e#"${e%%[![:space:]]*}"}"
    e="${e%"${e##*[![:space:]]}"}"
    [[ -n "$e" ]] && AMPI_UV_SYNC_ARGS+=(--extra "$e")
  done
}

_amphi_missing_uv_extras() {
  local extras="${AMPHI_UV_EXTRAS:-}"
  [[ -z "$extras" ]] && return 1
  if [[ "$extras" == *docking* ]] && ! uv run python -c "import rdkit; from meeko import MoleculePreparation" 2>/dev/null; then
    return 0
  fi
  if [[ "$extras" == *openff* ]] && ! uv run python -c "import openff.units; import openff.interchange; from openff.toolkit import ForceField; ForceField('openff-2.2.1.offxml')" 2>/dev/null; then
    return 0
  fi
  if [[ "$extras" == *protein* ]] && ! uv run python -c "import openmmforcefields" 2>/dev/null; then
    return 0
  fi
  return 1
}

_amphi_ensure_uv_extras() {
  _amphi_uv_sync_args
  if _amphi_missing_uv_extras; then
    echo "==> uv sync (${AMPHI_UV_EXTRAS})"
    uv sync "${AMPI_UV_SYNC_ARGS[@]}"
  fi
}

_amphi_ensure_rustquantic() {
  if ! uv run python -c "import rustquantic" 2>/dev/null; then
    echo "==> building rustquantic extension"
    uv run maturin develop --manifest-path rust/Cargo.toml
  fi
}

amphi_setup() {
  AMPI_ROOT="${1:?repo root required}"
  cd "$AMPI_ROOT"
  _amphi_uv_sync_args
  if [[ ! -d .venv ]]; then
    echo "==> bootstrapping: uv sync + maturin develop"
    uv sync "${AMPI_UV_SYNC_ARGS[@]}"
    _amphi_ensure_rustquantic
  else
    _amphi_ensure_uv_extras
    _amphi_ensure_rustquantic
  fi
  export AMPI_ROOT
  if [[ -n "${AMPHI_EXPERIMENT_ROOT:-}" ]]; then
    export AMPHI_EXPERIMENT_ROOT
  fi
  if [[ -n "${AMPHI_RUN_DIR:-}" ]]; then
    export AMPHI_RUN_DIR
  fi
}

# Run bin/<name> with project venv (not system python3).
amphi_bin() {
  local name="$1"
  shift
  uv run python "$AMPI_ROOT/bin/$name" "$@"
}

# Inline python in project venv.
amphi_py() {
  uv run python "$@"
}
