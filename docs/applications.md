# Practical application examples

Curated MD use cases for validating amphiquantic end-to-end. Each example has committed fixtures (offline CI), a runnable script, and observable pass criteria.

## Quick run

```bash
./scripts/examples/run_water_box.sh
./scripts/examples/run_ethanol_solvated.sh
./scripts/examples/run_ala_dipeptide.sh
./scripts/examples/run_complex_lite.sh
```

Scripts auto-bootstrap via `uv sync` + `maturin develop` if `.venv` is missing. Use `uv run pytest` for tests (not bare `pytest`).

Outputs land in `output/examples/<id>/`.

## Catalog

| ID | Application | Script | Tier | Fixture |
|----|-------------|--------|------|---------|
| A1 | TIP3P water box | `run_water_box.sh` | A | `tests/fixtures/applications/water_box/solvated.aqtop` |
| A3 | Ethanol in water | `run_ethanol_solvated.sh` | A | `tests/fixtures/applications/ethanol_solv/` |
| A4 | Ala dipeptide solvated | `run_ala_dipeptide.sh` | A | `tests/fixtures/applications/ala_dipeptide/` |
| A6-lite | Dipeptide + ligand | `run_complex_lite.sh` | B | built at runtime |
| A2 | NaCl(aq) | `run_nacl_aq.sh` | B | `tests/fixtures/applications/nacl_aq/` |
| A5 | Crambin (1CRN) | `run_crambin.sh` | B | RCSB fetch |
| A7 | GPU benchmark | `benchmark_applications.sh` | B | A1 + A4 fixtures |
| A8 | Protein–ligand implicit experiment | `bin/new_experiment` + `bin/run_experiment` | B | runtime in `experiments/{uuid}/` |
| A9 | FKBP12 + FK506 explicit experiment | `run_fkbp_explicit.sh` | B | RCSB 1FKF + OpenFF |

## A8 — Protein–ligand implicit experiment

```bash
uv run python bin/new_experiment --protocol protein_ligand_implicit \
  --receptor tests/fixtures/complex/ala_dipeptide.pdb --ligand CCO
uv run python bin/run_experiment experiments/<uuid>
```

Produces docked complex, implicit GB MD, trajectory, dock preview MP4, and MD movie MP4. See [experiments.md](experiments.md).

## A9 — FKBP12 + FK506 explicit experiment (1FKF)

```bash
./scripts/examples/run_fkbp_explicit.sh
```

Or manually:

```bash
uv sync --extra docking --extra openff --extra protein
uv run maturin develop --manifest-path rust/Cargo.toml
uv run python bin/new_experiment --protocol protein_ligand_explicit \
  --receptor 1FKF --ligand-from-pdb 1FKF:FK5 --ligand-resname FK5
uv run python bin/run_experiment experiments/<uuid>
```

Produces apo receptor, Vina docking at co-crystal site, TIP3P solvation, ~500k-step explicit MD, dock preview MP4, and solute-only MD movie MP4.

## Observables

| Metric | Module |
|--------|--------|
| Mass density | `amphiquantic.validation.physics.density_g_cm3` |
| Single-point energy | `rustquantic.topology_energy` |
| Trajectory temperature / drift | `amphiquantic.validation.trajectory_analysis` |
| GROMACS cross-check | `./scripts/validate_gromacs.sh <fixture>` |

## Tests

```bash
./scripts/agent_verify.sh --fast          # tier_a application tests
uv run pytest -m tier_b tests/validation/test_applications.py
uv run pytest -m complex tests/validation/test_complex_workflow.py
```

See [validation.md](validation.md) for gate criteria.
