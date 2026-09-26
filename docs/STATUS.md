# Engine status and handoff notes

Last updated: 2026-09-26 (branch `upgrade-1`). Read this before starting work.

## Backends

| `AMPHI_FORCE_BACKEND` | State |
|---|---|
| `cpu` | Reference implementation. Finite-difference tested. |
| `hybrid` (default) | CPU bonded + PME, GPU nonbonded. Parity-tested against CPU. |
| `gpu` | Per-step GPU nonbonded. Parity-tested against CPU. |
| `gpu-resident` | Whole step on the GPU; see `docs/gpu_resident.md`. Has known bugs (below). |

## Features

- **Ensembles** (`rust/src/integrator`): Langevin (BAOAB), Berendsen, V-rescale and Nosé–Hoover chain thermostats; Berendsen and Monte Carlo barostats; virial pressure; COM motion removal; degrees of freedom exclude constraints.
- **Electrostatics** (`rust/src/electrostatics/pme.rs`): smooth PME with order-4 B-splines and `rustfft`; α and grid are chosen automatically (`ewald_rtol`, `fourier_spacing`).
- **Constraints** (`rust/src/constraints`):
  - analytic SETTLE (Miyamoto–Kollman) for positions and velocities
  - LINCS, with SHAKE/RATTLE as a fallback
  - hydrogen mass repartitioning (HMR), allowing 4 fs steps
- **Performance:** rayon parallel forces and cell-list neighbour lists; per-atom exclusion lists.
- **Minimiser** (`rust/src/minimize`): L-BFGS by default, steepest descent available; `--minimizer`, `--emtol`.
- **I/O** (`rust/src/trajectory`): `.aqtrj` (default), DCD, multi-model PDB, XYZ, energy-log CSV, binary checkpoint.
- **Analysis** (`python/amphiquantic/analysis`, `bin/analyze`): RMSD, RMSF, Rg, RDF, MSD/diffusion, hydrogen bonds, energy statistics.
- **API change:** `simulate_topology`, `equilibrate_topology` and `simulate_topology_gpu_resident` return `(topology, final_T, final_P)`.

## Known bugs and gaps (highest priority first)

1. **gpu-resident Langevin + SETTLE diverges** on multi-water periodic boxes: kinetic energy runs away within a few hundred steps. V-rescale + SETTLE works. Unfinished debugging work is in `docs/wip/gpu-langevin-settle-debug.patch`. Suspects:
   - fixed-point i32 force-accumulation overflow at dense packing
   - SETTLE reference frame per half-drift
   - noise and velocity-constraint ordering
2. **gpu-resident pressure omits the SETTLE/SHAKE constraint virial**, so pressure and Berendsen NPT are wrong for water. The MC barostat is unaffected (it decides from energies). Unfinished work is in `docs/wip/gpu-constraint-virial.patch`. The CPU reference is `integrator::compute_constraint_virial`.
3. **gpu-resident pressure** reads positions back and calls the CPU virial at barostat/output cadence. A GPU virial reduction is still to do.
4. **gpu-resident box size:** the cell grid and PME grid are fixed at construction, so long NPT runs with large box drift need a rebuild.
5. **Loose tolerances:**
   - SPME vs. direct Ewald test in Python: 0.5%
   - combined gpu-resident PME parity: a few ‰
   - MC barostat test does not check density convergence
6. **Only orthorhombic boxes** are supported; there is no triclinic `SimulationBox`.

Apply a WIP patch with `git apply docs/wip/<name>.patch`. The patches are against `upgrade-1` at `438c025` and are unverified.

## Bugs fixed on this branch

Results produced before these fixes are unreliable:

- CPU nonbonded pair forces and the virial were scaled by r.
- 1-4 LJ/Coulomb scaling was missing in the `gpu` / `hybrid` shader.
- The PME excluded-pair correction was applied twice.
- The Ewald self/background energy was double-counted in reported energies.

## Testing

```bash
uv sync --extra openff          # optional extras; agent_verify uses --inexact so they persist
./scripts/agent_verify.sh --fast
cd rust && cargo test --release
uv run pytest -m gpu            # needs a wgpu adapter
cd rust && cargo test --release -- --ignored --test-threads=1   # benchmarks (wall-clock; run alone)
```

If there is no GPU, use a software Vulkan adapter:

```bash
sudo apt-get install -y mesa-vulkan-drivers libvulkan1
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json WGPU_BACKEND=vulkan
```

Lavapipe throughput numbers mean nothing for real GPUs. The last measurement was 0.48 ns/day at 5184 atoms.

## Suggested next work

1. Fix the two gpu-resident bugs above.
2. Benchmark on real GPU hardware.
3. Add a GPU virial reduction.
4. Add a GPU FFT for PME.
5. Add triclinic boxes.
6. Add CMAP and Urey–Bradley terms.
7. Add restraints, FEP/TI with soft-core potentials, and replica exchange.
