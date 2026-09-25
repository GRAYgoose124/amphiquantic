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
# Implicit experiment (dock → GB MD → MP4)
uv run python bin/new_experiment --protocol protein_ligand_implicit \
  --receptor tests/fixtures/complex/ala_dipeptide.pdb --ligand CCO
uv run python bin/run_experiment experiments/<uuid>

# Legacy explicit path
./scripts/steps/complex/build.sh tests/fixtures/complex/ala_dipeptide.pdb 'CCO' output/complex
```

See [docs/experiments.md](docs/experiments.md).

## Structure & ligand fetch

```bash
./bin/fetch_structure 1CRN -o output/rcsb              # PDB + mmCIF from RCSB
./bin/fetch_ligand -q ethanol -o ligand.smi            # PubChem → SMILES
./bin/fetch_ligand --from-pdb 1HTM:HEM -o heme.pdb     # HETATM from structure
```

See [docs/complex_workflow.md](docs/complex_workflow.md).

## Application examples

Runnable recipes for common MD scenarios:

```bash
./scripts/examples/run_water_box.sh
./scripts/examples/run_ethanol_solvated.sh
./scripts/examples/run_ala_dipeptide.sh
./scripts/examples/run_complex_lite.sh
```

See [docs/applications.md](docs/applications.md) and committed fixtures in `tests/fixtures/applications/`.

## Environment

| Variable | Default | Purpose |
|----------|---------|---------|
| `AMPHI_HYBRID` | `1` | CPU bonded/PME + GPU nonbonded |
| `AMPHI_FORCE_BACKEND` | hybrid | `cpu`, `gpu`, or hybrid |

## Validation

```bash
./scripts/agent_verify.sh --fast
./scripts/validate_gromacs.sh tests/fixtures/applications/ala_dipeptide/ala_dipeptide_solv.aqtop
uv run pytest -m tier_b tests/validation/test_applications.py
uv run pytest -m complex tests/validation/test_complex_workflow.py
./scripts/benchmark_applications.sh
```

See [AGENTS.md](AGENTS.md), [docs/validation.md](docs/validation.md), and [docs/applications.md](docs/applications.md).
