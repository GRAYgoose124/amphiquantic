# Test layout for agents

Run the verification gate:

```bash
./scripts/agent_verify.sh --fast
```

## Directories

| Path | Purpose |
|------|---------|
| `tests/agent/` | Harness smoke + workflow contract |
| `tests/validation/` | Physics, OpenFF, CPU/GPU parity, application examples |
| `tests/fixtures/` | Small committed inputs (PDB, `.aqtop`) |
| `tests/fixtures/applications/` | Application catalog fixtures (A1–A4, A2 ionized) |

## Adding an application example

1. Add fixture under `tests/fixtures/applications/<id>/` with README + `.aqtop`
2. Add runner in `scripts/examples/run_<id>.sh`
3. Add tier_a tests in `tests/validation/test_applications.py`
4. Document in `docs/applications.md` and `docs/validation.md`

## Markers

Register new tests with `@pytest.mark.tier_a` unless they are slow, OpenFF-only, network, or manual GROMACS (`tier_b`).

See [AGENTS.md](../AGENTS.md), [docs/applications.md](../docs/applications.md), and [agents/tdd-protocol.md](../agents/tdd-protocol.md).
