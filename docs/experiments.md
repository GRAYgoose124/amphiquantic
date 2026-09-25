# Experiment runs

UUID-scoped, repeatable pipelines with frozen protocols and dated outputs.

## Quick start

```bash
uv sync --extra docking
uv run maturin develop --manifest-path rust/Cargo.toml

# Create experiment (implicit GB protein–ligand)
uv run python bin/new_experiment \
  --protocol protein_ligand_implicit \
  --receptor tests/fixtures/complex/ala_dipeptide.pdb \
  --ligand CCO

# Run (creates experiments/{uuid}/outputs/{YYYYMMDD-HHMMSS}/)
uv run python bin/run_experiment experiments/<uuid>
```

## Layout

```text
experiments/{uuid}/
  manifest.yaml       # id, protocol, config, git_rev
  protocol.sh         # frozen pipeline
  config.yaml
  inputs/             # receptor, ligand, optional cofactor
  scripts/            # step scripts snapshot
  outputs/{datestamp}/
    topology/         # *.aqtop
    trajectories/     # *.aqtrj
    docking/          # poses.pdbqt, box.json
    render/           # dock_preview.mp4, movie.mp4, dock_frames/, frames/
    logs/
    run.json
```

Environment variables set during a run:

| Variable | Meaning |
|----------|---------|
| `AMPHI_EXPERIMENT_ROOT` | `experiments/{uuid}/` |
| `AMPHI_RUN_DIR` | `outputs/{datestamp}/` |
| `AMPHI_REQUIRE_DOCK` | `1` = fail if Vina unavailable |
| `AMPHI_SKIP_RENDER` | `1` = skip dock preview and MD movie steps |

## Render outputs

Implicit experiment runs produce two videos under `render/`:

| File | Source | Description |
|------|--------|-------------|
| `dock_preview.mp4` | `topology/complex.aqtop` | Rotating view of docked complex (post-merge) |
| `movie.mp4` | `trajectories/md.aqtrj` | MD trajectory with molecule-aware coloring |

If `ffmpeg` is not installed, PNG frame directories (`dock_frames/`, `frames/`) are written instead.

## Protocol templates

| Template | Solvation | Steps |
|----------|-----------|-------|
| `protein_ligand_implicit` | OBC-II GB (Rust) | param → dock → merge → dock preview → implicit → MD → render |
| `protein_ligand_explicit` | TIP3P + PME | fetch → apo prep → param → dock → merge → dock preview → solvate → ionize → min → equil → MD → solute render |
| `enzyme_complex_implicit` | OBC-II GB | + optional cofactor / chain_b assembly |

## Prerequisites

**Implicit + docking:**
```bash
uv sync --extra docking
uv run maturin develop --manifest-path rust/Cargo.toml
```

**Explicit protein–ligand (OpenFF + ff14SB):**
```bash
uv sync --extra docking --extra openff --extra protein
uv run maturin develop --manifest-path rust/Cargo.toml
```

`bin/run_experiment` sets `AMPHI_UV_EXTRAS=docking,openff,protein` for `protein_ligand_explicit`.

**AutoDock Vina** (system binary on `PATH`):

| Platform | Install |
|----------|---------|
| Arch (AUR often fails on static Boost) | Download [vina_1.2.7_linux_x86_64](https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/v1.2.7/vina_1.2.7_linux_x86_64), `chmod +x`, move to `~/.local/bin/vina` |
| Debian/Ubuntu | `sudo apt install autodock-vina` |
| Conda | `conda install -c conda-forge autodock-vina` |

Optional: `ffmpeg` for MP4 encoding.

To run without Vina (merge ligand without docking): `AMPHI_REQUIRE_DOCK=0 uv run python bin/run_experiment experiments/<uuid>`

## Explicit refinement path

```bash
uv run python bin/new_experiment --protocol protein_ligand_explicit \
  --receptor 1FKF --ligand-from-pdb 1FKF:FK5 --ligand-resname FK5
uv run python bin/run_experiment experiments/<uuid>
```

Default config: 1FKF/FK5, ff14SB receptor + OpenFF ligand, 500k MD steps. Override `steps.simulate` in `config.yaml` for shorter runs. MD render uses `--solute-only` (receptor + ligand, no solvent).

See [applications.md](applications.md) — A9 `run_fkbp_explicit.sh`.

## See also

- [complex_workflow.md](complex_workflow.md)
- [applications.md](applications.md) — A8 protein_ligand_implicit
