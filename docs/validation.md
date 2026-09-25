# FOSS validation benchmark catalog

## Application examples

See [applications.md](applications.md) for runnable recipes (`scripts/examples/`) and committed fixtures under `tests/fixtures/applications/`.

| ID | Fixture | Tier A criterion |
|----|---------|------------------|
| A1 | `water_box/solvated.aqtop` | density 0.85–1.15 g/cm³ |
| A3 | `ethanol_solv/ethanol_solv_ion.aqtop` | >10 atoms, \|Q\|<0.5 |
| A4 | `ala_dipeptide/ala_dipeptide_solv.aqtop` | neutral, equil epot finite |
| A2 | `nacl_aq/ionized.aqtop` | \|Q\|<0.01 (tier_b) |

## Tier A (CI)

| System | Metric | Criterion |
|--------|--------|-----------|
| Application A1 | water density | 0.85–1.15 g/cm³ |
| Application A3/A4 | pipeline smoke | solvate, ionize, short MD |
| Methane (bootstrap) | atom count | 5 atoms in `tests/fixtures/methane.aqtop` |
| Force parity | CPU vs GPU energy | within 50% on methane fixture |
| mmCIF parse | ala_mini.cif → PDB | 4 atoms, coords preserved |
| Ligand DB | literal SMILES / HETATM extract | resolve + extract without network |
| GROMACS export | all application fixtures | `.gro`/`.top`/`.mdp` written |
| Restart I/O | `--restart` from `.aqtrj` | last-frame coords applied |
| Exclusions on read | empty exclusions + bonds | auto-built on `Topology::read` |
| Implicit GB (A8) | `prepare_implicit` + `topology_energy` | finite energy on methane |
| Experiment layout | `new_experiment` | manifest + inputs + protocol.sh |

## Tier B (manual / network)

| Check | Command |
|-------|---------|
| Application GROMACS energy | `uv run pytest -m tier_b tests/validation/test_applications.py::test_app_gromacs_energy_oracle` |
| NaCl neutralization | `pytest -m tier_b tests/validation/test_applications.py::test_app_nacl_neutral` |
| Crambin fetch | `./scripts/examples/run_crambin.sh` |
| NPT density (loose) | `pytest -m tier_b -m slow tests/validation/test_applications.py::test_app_npt_water_density` |
| RCSB mmCIF live | `pytest -m tier_b tests/validation/test_structure_fetch.py` |
| GROMACS grompp + energy | `./scripts/validate_gromacs.sh tests/fixtures/applications/ala_dipeptide/ala_dipeptide_solv.aqtop` |
| Complex workflow | `pytest -m complex tests/validation/test_complex_workflow.py` |
| Protein–ligand experiment | `uv run python bin/new_experiment --protocol protein_ligand_implicit ...` |
| Docking (Vina) | `uv run pytest -m tier_b -m docking tests/validation/test_docking_binding.py` |
| Enzyme experiment template | `uv run pytest -m tier_b tests/validation/test_experiment_enzyme.py` |
| GPU benchmark | `./scripts/benchmark_applications.sh` |

## Agent gate

```bash
./scripts/agent_verify.sh --fast
```

See [AGENTS.md](../AGENTS.md).

## FOSS policy

Parameters from OpenFF `.offxml` when installed; bootstrap fallback otherwise. GROMACS used as external CLI validator only.

## Known limitations

- PME uses direct k-space sum (FFT grid planned; `rustfft` in Cargo.toml unused).
- GPU path lacks explicit 1–4 LJ scaling (CPU applies 0.5/0.8333).
- Restart loads coordinates + box only (velocities reinitialized).
