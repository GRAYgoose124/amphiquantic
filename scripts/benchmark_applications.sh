#!/usr/bin/env bash
# A7 — Benchmark application fixtures (CPU / hybrid)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/scripts/amphi_env.sh"
amphi_setup "$ROOT"

export AMPHI_HYBRID="${AMPHI_HYBRID:-1}"
STEPS="${1:-100}"

bench_one() {
  local fixture="$1"
  local backend="$2"
  echo "--- $fixture backend=$backend ---"
  env AMPHI_FORCE_BACKEND="$backend" AMPHI_HYBRID=1 uv run python -c "
import os, time
from rustquantic import load_topology, minimize_topology
path = '$fixture'
top = load_topology(path)
t0 = time.perf_counter()
_, energy, iters = minimize_topology(path, None, $STEPS, 0.001)
dt = time.perf_counter() - t0
sps = iters / dt if dt > 0 else 0
print(f'atoms={top.num_atoms()} iters={iters} E={energy:.2f} time={dt:.3f}s steps_per_s={sps:.1f} backend={os.environ.get(\"AMPHI_FORCE_BACKEND\")}')
"
}

FIXTURES=(
  "tests/fixtures/applications/water_box/solvated.aqtop"
  "tests/fixtures/applications/ala_dipeptide/ala_dipeptide_solv.aqtop"
)

for f in "${FIXTURES[@]}"; do
  [[ -f "$f" ]] || continue
  bench_one "$f" cpu
  bench_one "$f" hybrid
done
echo "benchmark_applications: done"
