#!/usr/bin/env bash
# Source from scripts/examples/*.sh — sets AMPI_ROOT and bootstraps venv.
AMPI_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck disable=SC1091
source "$AMPI_ROOT/scripts/amphi_env.sh"
amphi_setup "$AMPI_ROOT"
