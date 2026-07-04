# Agent Verification Checklist

Copy this checklist into your handoff when completing a task.

## Pre-flight

- [ ] Working directory is repo root (`amphiquantic/`)
- [ ] `uv sync` completed without errors
- [ ] `uv run maturin develop --manifest-path rust/Cargo.toml` completed (if Rust changed)
- [ ] No stray `__init__.py` at repo root

## Automated gates

```bash
./scripts/agent_verify.sh --fast   # during development
./scripts/agent_verify.sh          # before handoff
```

- [ ] `--fast` passes
- [ ] Full verify passes (or document intentional skips)

## Layer-specific

### Rust / wgpu

- [ ] `cd rust && cargo test`
- [ ] `cd rust && cargo check` clean
- [ ] If forces changed: `uv run pytest -m gpu` passes

### Python / topology

- [ ] `uv run pytest tests/agent/` passes
- [ ] Bootstrap parameterization works: `./bin/parameterize /tmp/x.pdb -o /tmp/x.aqtop`

### OpenFF (optional extra)

- [ ] `uv sync --extra openff`
- [ ] `uv run pytest -m openff` passes

### Workflow

- [ ] `./build` (steps 0–1) passes
- [ ] Any new step has a matching `scripts/steps/minimal/*.sh` and documents in README

## Manual smoke (when touching end-to-end path)

```bash
./bin/parameterize tests/fixtures/ala.pdb -o /tmp/smoke.aqtop
./bin/minimize /tmp/smoke.aqtop -o /tmp/smoke.min.aqtop
```

- [ ] Commands succeed without traceback

## Handoff template

```
## Verification
- agent_verify: PASS (full / fast)
- cargo test: PASS
- pytest: X passed, Y skipped
- Notes: ...
```
