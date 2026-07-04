# Amphiquantic — Agent Guide

This repo is a **wgpu-first DIY MD engine** (Rust `rustquantic` + Python `amphiquantic`). Agents must follow the TDD protocol before claiming work is complete.

## Quick start

```bash
cd /path/to/amphiquantic   # directory name must be amphiquantic for ./run
./scripts/agent_verify.sh  # full agent verification gate
```

Fast loop while iterating:

```bash
./scripts/agent_verify.sh --fast
```

## Architecture (where to edit)

| Layer | Path | Role |
|-------|------|------|
| Topology / param | `python/amphiquantic/topology/` | `.aqtop` I/O, OpenFF, bootstrap |
| Rust engine | `rust/src/` | forces, integrator, minimize, topology |
| wgpu kernels | `rust/data/shaders/` | WGSL force shaders |
| Workflow CLIs | `bin/` | parameterize, solvate, ionize, minimize, equilibrate, simulate |
| Orchestration | `scripts/steps/` | bash workflow steps |
| Tests | `tests/` | pytest tiers + validation |

**Primary compute:** hybrid (`AMPHI_HYBRID=1`, default) — CPU bonded/PME + GPU nonbonded. **Reference:** CPU-only (`AMPHI_FORCE_BACKEND=cpu`).

## TDD protocol (mandatory)

1. **Read** the relevant tier in [`agents/tdd-protocol.md`](agents/tdd-protocol.md).
2. **RED** — add or extend a failing test in `tests/` *before* implementation when feasible.
3. **GREEN** — implement minimal code; run targeted tests.
4. **REFACTOR** — clean up; re-run verification.
5. **VERIFY** — run `./scripts/agent_verify.sh` and confirm all gates pass.

Do not skip verification. Do not edit [`AGENTS.md`](AGENTS.md) or plan files unless explicitly asked.

## Verification gates

| Gate | Command | When |
|------|---------|------|
| Fast | `./scripts/agent_verify.sh --fast` | Every iteration |
| Standard | `./scripts/agent_verify.sh` | Before handoff / PR |
| Rust only | `cd rust && cargo test` | Rust-only changes |
| GPU parity | `uv run pytest -m gpu` | wgpu / force kernel changes |
| OpenFF | `uv sync --extra openff && uv run pytest -m openff` | Parameterization changes |

See [`agents/verification-checklist.md`](agents/verification-checklist.md) for the full checklist.

## Environment

- **Python:** 3.10–3.13 (`.python-version` → 3.13). Use `uv`, not bare `pip`.
- **Build:** `uv sync` then `uv run maturin develop --manifest-path rust/Cargo.toml`
- **Never** create `__init__.py` at repo root (shadows `amphiquantic` package).

### Key env vars

| Variable | Values | Purpose |
|----------|--------|---------|
| `AMPHI_FORCE_BACKEND` | `hybrid` (default), `cpu`, `gpu` | Force evaluation backend |
| `AMPHI_HYBRID` | `1` (default on), `0` | CPU bonded + GPU nonbonded parallel |
| `AMPHI_SKIP_OPENFF` | `1` | Skip OpenFF tests in CI |
| `AMPHI_DATA_PATH` | path | YAML/shaders data (set by `./build`) |

## Test tiers (pytest markers)

| Marker | Meaning |
|--------|---------|
| `tier_a` | Fast CI — run always |
| `tier_b` | Manual / GROMACS — optional locally |
| `rust` | Needs `rustquantic` extension |
| `gpu` | Needs wgpu / GPU backend |
| `openff` | Needs `uv sync --extra openff` |
| `hybrid` | Hybrid CPU+GPU force path |
| `pme` | PME electrostatics (slow) |
| `docking` | Docking orchestration tests |
| `complex` | Full complex fixtures (tier_b) |
| `slow` | Long-running; skip with `--fast` |

## Skill

Project skill: [`.cursor/skills/amphiquantic-tdd/SKILL.md`](.cursor/skills/amphiquantic-tdd/SKILL.md) — read this when implementing features or fixing bugs.

## FOSS policy

OpenFF / published constants only for parameters. GROMACS/CHARMM as **external CLI validators** only — never link proprietary libraries. See [`docs/validation.md`](docs/validation.md).
