#!/usr/bin/env bash
# Protein–ligand implicit workflow via experiment runner
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/scripts/amphi_env.sh"
amphi_setup "$ROOT"

EXP="${1:?experiment uuid path required}"
amphi_bin run_experiment "$EXP"
