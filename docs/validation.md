# FOSS validation benchmark catalog

## Tier A (CI)

| System | Metric | Criterion |
|--------|--------|-----------|
| Methane (OpenFF) | atom count | 5 atoms from SMILES `C` |
| Force parity | CPU vs GPU minimize energy | same order of magnitude |
| Water box (future) | density | within 2% of 1.0 g/cm³ |

## Tier B (manual)

Export `.gro` via `tests/validation/export_for_gromacs.py` and compare to GROMACS `mdrun` when installed.

Run: `./scripts/validate_gromacs.sh tests/fixtures/methane.aqtop`

## Agent gate

Before handoff, agents run:

```bash
./scripts/agent_verify.sh
```

See [AGENTS.md](../AGENTS.md).

## FOSS policy

Parameters from OpenFF `.offxml` only. GROMACS/CHARMM used as external CLI validators, never linked.
