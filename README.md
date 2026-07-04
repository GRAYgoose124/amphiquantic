# Amphiquantic

FOSS molecular dynamics: Rust engine (`rustquantic`) + Python workflow (`amphiquantic`).

## Quick start

```bash
uv sync
uv run maturin develop --manifest-path rust/Cargo.toml
./scripts/agent_verify.sh --fast
```

## Minimal workflow

```bash
# Parameterize (OpenFF when installed, else --bootstrap)
./bin/parameterize "CCO" --from-smiles -o ligand.aqtop

# Solvate + neutralize (preserves FF terms)
./bin/solvate ligand.aqtop -o solvated.aqtop
./bin/ionize solvated.aqtop -o ionized.aqtop

# MD
./bin/minimize ionized.aqtop -o min.aqtop
./bin/equilibrate min.aqtop --steps 500
./bin/simulate min.eq.aqtop --steps 1000
```

## Complex workflow

```bash
./scripts/steps/complex/build.sh tests/fixtures/complex/ala_dipeptide.pdb 'CCO' output/complex
```

See [docs/complex_workflow.md](docs/complex_workflow.md).

## Environment

| Variable | Default | Purpose |
|----------|---------|---------|
| `AMPHI_HYBRID` | `1` | CPU bonded/PME + GPU nonbonded |
| `AMPHI_FORCE_BACKEND` | hybrid | `cpu`, `gpu`, or hybrid |

## Validation

```bash
./scripts/agent_verify.sh
./scripts/validate_gromacs.sh tests/fixtures/methane.aqtop  # Tier B, needs GROMACS
```

See [AGENTS.md](AGENTS.md) and [docs/validation.md](docs/validation.md).
