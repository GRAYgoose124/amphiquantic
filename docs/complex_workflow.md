# Enzymatic complex workflow

End-to-end recipe for enzyme + ligand. **Implicit-first (OBC-II GB)** via experiments; explicit TIP3P remains for refinement.

## Experiment-based pipeline (recommended)

```bash
uv sync --extra docking
uv run maturin develop --manifest-path rust/Cargo.toml

uv run python bin/new_experiment \
  --protocol protein_ligand_implicit \
  --receptor receptor.pdb \
  --ligand 'CCO'

uv run python bin/run_experiment experiments/<uuid>
```

Outputs land in `experiments/<uuid>/outputs/{datestamp}/` including topology, trajectory, docking poses, and `render/movie.mp4`. See [experiments.md](experiments.md).

## Implicit vs explicit

| Path | Template | Solvation |
|------|----------|-----------|
| Fast (default) | `protein_ligand_implicit` | Rust OBC-II GB — no water box |
| Refinement | `protein_ligand_explicit` | TIP3P + PME via `build.sh` |

## Legacy explicit workflow

```bash
uv sync --extra openff --extra protein --extra docking
uv run maturin develop --manifest-path rust/Cargo.toml
```

Optional external tools: Vina/Smina, Meeko, PDBFixer, LightDock.

## Hybrid execution

```bash
export AMPHI_HYBRID=1   # CPU bonded + PME + GPU nonbonded (default on non-cpu backend)
export AMPHI_FORCE_BACKEND=cpu  # CI reference path
```

## Workflow

```bash
# 1. Parameterize protein
bin/parameterize receptor.pdb -o receptor.aqtop

# 2. Dock ligand
bin/dock_ligand receptor.pdb ligand.smi --out poses.pdbqt

# 3. Optional protein-protein assembly
bin/dock_assembly chainA.pdb chainB.pdb -o dimer.pdb

# 4. Merge components
uv run python -c "
from amphiquantic.topology.complex_builder import build_complex
build_complex(['receptor.aqtop', 'ligand.aqtop'], 'complex.aqtop')
"

# 5. Solvate + ionize
bin/solvate complex.aqtop -o solvated.aqtop
bin/ionize solvated.aqtop -o ionized.aqtop

# 6. MD
bin/minimize ionized.aqtop -o minimized.aqtop
bin/equilibrate minimized.aqtop --steps 50000
bin/simulate minimized.aqtop --steps 500000
```

Or run the orchestrated script:

```bash
./scripts/steps/complex/build.sh receptor.pdb 'CCO' output/complex
```

## Benchmark (4060M)

```bash
./scripts/benchmark_4060.sh tests/fixtures/complex/dipeptide_solv.aqtop 500
```

## Validation

See [validation.md](validation.md) Tier A/B gates. GROMACS oracle:

```bash
./scripts/validate_gromacs.sh
```
