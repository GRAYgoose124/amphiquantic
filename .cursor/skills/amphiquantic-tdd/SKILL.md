---
name: amphiquantic-tdd
description: TDD workflow for the amphiquantic MD engine (Rust rustquantic + Python amphiquantic). Use when implementing features or fixing bugs.
---

# amphiquantic TDD

1. Read `AGENTS.md`, `agents/tdd-protocol.md`, and `docs/STATUS.md` (current state, known bugs, next work).
2. RED: add a failing test (Rust `#[cfg(test)]` in the module, or pytest in `tests/` with a tier marker).
3. GREEN: minimal fix; run targeted tests.
4. VERIFY: `./scripts/agent_verify.sh --fast` each iteration; `./scripts/agent_verify.sh` before handoff.
   - Rust only: `cd rust && cargo test --release`
   - GPU: `uv run pytest -m gpu` (no GPU? install lavapipe — see docs/STATUS.md)
   - Benchmarks: `cd rust && cargo test --release -- --ignored --test-threads=1`

## Physics invariants (tests enforce these — keep them)
- Pair force scalars mean |F|/r and are multiplied by the separation vector.
- Forces must equal −dE/dx (`total_forces_match_finite_difference`).
- CPU is the reference; GPU / hybrid / gpu-resident must match it (parity tests).
- 1-4 pairs are scaled (LJ 0.5, Coulomb 1/1.2); excluded pairs get the PME erf correction once per unordered pair.
