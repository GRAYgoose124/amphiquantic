# Test layout for agents

Run the verification gate:

```bash
./scripts/agent_verify.sh
```

## Directories

| Path | Purpose |
|------|---------|
| `tests/agent/` | Harness smoke + workflow contract |
| `tests/validation/` | Physics, OpenFF, CPU/GPU parity |
| `tests/fixtures/` | Small committed inputs (PDB, `.aqtop`) |

## Markers

Register new tests with `@pytest.mark.tier_a` unless they are slow, OpenFF-only, or manual GROMACS (`tier_b`).

See [AGENTS.md](../AGENTS.md) and [agents/tdd-protocol.md](../agents/tdd-protocol.md).
