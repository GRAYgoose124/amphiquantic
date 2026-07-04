#!/usr/bin/env bash
# RTX 4060M / hybrid MD benchmark harness
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export AMPHI_HYBRID=1

echo "==> Building extension"
uv sync
uv run maturin develop --manifest-path rust/Cargo.toml

FIXTURE="${1:-tests/fixtures/complex/dipeptide_solv.aqtop}"
STEPS="${2:-500}"

if [[ ! -f "$FIXTURE" ]]; then
  echo "Fixture missing: $FIXTURE — using minimal test topology"
  uv run python -c "
from rustquantic import Topology
t = Topology.from_pdb_types([(0,0,0),(0.15,0,0)], ['C','C'], [(0,1)])
t.write('/tmp/bench.aqtop')
"
  FIXTURE=/tmp/bench.aqtop
fi

echo "==> Benchmark: $FIXTURE ($STEPS steps)"
/usr/bin/time -f "wall=%e sec" env AMPHI_HYBRID=1 \
  uv run python -c "
import os, time
from rustquantic import load_topology, minimize_topology
path = '$FIXTURE'
top = load_topology(path)
print(f'atoms={top.num_atoms()}')
t0 = time.perf_counter()
_, energy, iters = minimize_topology(path, None, $STEPS, 0.001)
dt = time.perf_counter() - t0
print(f'iterations={iters} energy={energy:.4f} time={dt:.3f}s')
print(f'backend=hybrid AMPHI_HYBRID={os.environ.get(\"AMPHI_HYBRID\",\"\")}')
"

echo "benchmark_4060: done"
