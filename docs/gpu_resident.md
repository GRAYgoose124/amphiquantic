# GPU-resident MD step (`AMPHI_FORCE_BACKEND=gpu-resident`)

A GPU-resident backend keeps positions, velocities, forces, per-atom
parameters and the neighbor cell list in GPU buffers across the whole run and
only copies back to the host at logging cadence, in the style of
OpenMM/GROMACS-GPU. This backend is opt-in; the default (`hybrid`, CPU
bonded + GPU nonbonded per-step) and the CPU reference path are unchanged.

Code:

- `rust/src/forces/gpu_resident/mod.rs` — the engine (`GpuResidentEngine`,
  `run_gpu_resident`) and the single-shot `ForceBackend::GpuResident`
  compatibility path.
- `rust/data/shaders/gpu_resident.wgsl` — the WGSL compute kernels.
- `rustquantic::simulate_topology_gpu_resident` (PyO3) — the dedicated
  multi-step entry point used by `bin/simulate --backend gpu-resident` and
  `bin/equilibrate --backend gpu-resident`.

## What's actually resident

Per MD step, entirely on the GPU:

- Cell list build: a GPU counting pass (`count_cells`) assigns each atom to a
  cell, a GPU scatter pass (`scatter_cells`) sorts atom indices into
  contiguous per-cell runs.
- Nonbonded LJ + real-space Ewald (`erfc`) via a 3x3x3 cell-neighbor stencil
  (`nonbonded`), single precision, with **fixed-point (`i32`, scale
  `2^16`) atomic force accumulation** so bonded and nonbonded kernels can
  both write into the same per-atom force without races or accumulation-order
  sensitivity — this is the same idea as OpenMM's mixed-precision fixed-point
  accumulators, at `i32` rather than `i64` scope for simplicity.
- Bonded harmonic bond, angle and periodic-torsion dihedral kernels
  (`bond_forces`, `angle_forces`, `dihedral_forces`), also accumulating into
  the fixed-point buffer. `dihedral_forces` covers both proper and improper
  torsions (they're chained into one dispatch, matching how
  `cpu::add_dihedral_forces` treats them identically) and is a direct WGSL
  port of that CPU function's math — same intermediate cross products
  (`n2`, `n3`, `m1`), same `atan2`-based dihedral angle, same
  `f_i`/`f_j`/`f_k`/`f_l` force redistribution.
- GPU potential-energy reduction: the `bond_forces`, `angle_forces`,
  `dihedral_forces` and `nonbonded` kernels each also write their
  contribution to a per-item (`bond_energy`/`angle_energy`/
  `dihedral_energy`, one `f32` per bond/angle/dihedral) or per-atom
  (`nb_energy`, one `f32` per atom, each atom's slot holding half the sum of
  its pairwise LJ+real-space-Ewald energies so summing over atoms double-
  counts correctly) buffer on every dispatch — the extra ALU cost is
  negligible next to the force math already being done. `potential_energy_gpu()`
  reads these four buffers back and sums them in `f64` on the host, and is
  only called on output steps (`GpuResidentEngine::run`'s
  `output_interval` cadence), not every step. This is a **per-item/per-atom
  readback**, not a from-scratch workgroup-shared-memory partial-sum
  reduction — a documented simplification in the spirit of the cell-list
  prefix-scan simplification below: the buffers are already the size of the
  system (comparable to the position/velocity readbacks the engine already
  does), so this isn't a new bottleneck, but it doesn't get a true
  logarithmic on-GPU reduction. Virial/pressure reduction is not yet part of
  this (tracked as a follow-up; see "What stays on the CPU").
- Velocity-Verlet integration (`kick_half`, `drift`, `kick_half` again),
  kinetic-energy reduction (`ke_reduce`), and thermostat velocity scaling
  (`scale_velocities`).
- Neighbor-list-rebuild decision via a GPU max-displacement reduction
  (`max_displacement`, `atomicMax` over squared displacement vs. the last
  rebuild's reference positions) compared to `skin/2` on the host (a single
  `u32` readback, not a bulk position copy).

## What stays on the CPU (by design, and why)

- **PME reciprocal-space electrostatics** stays on the CPU (existing SPME),
  as the task explicitly allows. It is **not** overlapped asynchronously with
  the GPU nonbonded kernel in this implementation — each step does a
  synchronous position readback, CPU PME + dihedral evaluation, then uploads
  the combined force delta back via `add_external_forces`. True overlap
  (dispatching the next GPU nonbonded pass while last step's PME runs on a
  CPU thread, then reconciling) is a natural follow-up once the two are
  running as separate submissions instead of the current per-step
  request/response readback.
- **Dihedral/improper bonded terms** are now evaluated on the GPU (see
  "What's actually resident" above) — this was previously a per-step CPU
  round trip (`cpu::compute_dihedral_forces`) and is no longer.
- **SETTLE** (rigid 3-site water) is applied via the existing, validated
  `constraints::apply_settle_analytic` / `apply_settle_velocity` CPU
  routines, once per step, on a position/velocity readback — rather than a
  new WGSL reimplementation of the Miyamoto–Kollman closed-form solve. This
  keeps constrained-water correctness anchored to code already covered by
  `settle_preserves_water_geometry`, `settle_velocity_orthogonal_to_bonds`,
  etc., at the cost of one CPU round trip per step for constrained systems.
- **Solute LINCS/SHAKE constraints and barostats are not implemented** in the
  GPU-resident path at all (`--backend gpu-resident` refuses `--npt` /
  `--barostat` at the CLI). Systems needing those still use `hybrid` (the
  default) or `cpu`.

## Deviations from the "textbook" design (and why)

- **Exclusions**: a small fixed-capacity (`MAX_EXCL = 8`) per-atom exclusion
  list uploaded from `topology.exclusions`, rather than a per-tile bitmask.
  Correct for the same reason the CPU/GPU per-pair backends' exclusion lists
  are (bonded exclusion lists are tiny), but doesn't get the SIMD-friendly
  bitmask-test property a real tile-bitmask design would.
- **1-4 (dihedral end-atom) LJ/Coulomb scaling**: the CPU nonbonded path
  (`cpu::compute_nonbonded_forces`) applies `LJ_14_SCALE`/`COULOMB_14_SCALE`
  to the `(i, l)` pair of every dihedral (they're nonbonded-interacting, not
  excluded, just scaled). The GPU-resident `nonbonded` kernel does not — it
  only checks the exclusion list, so a 1-4 pair that isn't also a 1-2/1-3
  exclusion gets full-strength LJ/Coulomb instead of the scaled value. This
  is a real, pre-existing correctness gap for any topology whose dihedral
  end-atoms are also within the nonbonded cutoff (the test topologies added
  for the dihedral-kernel and energy-reduction parity tests below sidestep
  it by using zero charge/epsilon, isolating the bonded-kernel math being
  tested). Fixing it means uploading a per-atom (or per-pair) 1-4 partner
  table to the `nonbonded` kernel the same way exclusions are uploaded;
  tracked as follow-up work, not fixed in this change.
- **Prefix scan**: the exclusive scan over per-cell atom counts (needed to
  turn per-cell counts into per-cell start offsets for the counting-sort
  scatter) is done by reading the small `cell_count` buffer back to the host,
  scanning it there, and writing `cell_start` back. The number of cells is
  orders of magnitude smaller than the number of atoms, so this is not a
  bottleneck, but it is not a from-scratch GPU Blelloch/Hillis-Steele scan.
- **RNG for stochastic thermostats**: the task asked for a GPU Philox
  generator. What's implemented is a plain 64-bit xorshift used host-side to
  perturb the (currently scalar, not per-atom) V-rescale/Langevin scale
  factor — **not** a per-atom Philox-4x32-10 kernel producing per-DOF
  Gaussian noise for a true BAOAB Langevin O-step. The `thermostat="langevin"`
  path here is a velocity-rescaling approximation, not full BAOAB; treat it
  as a placeholder until a real per-atom counter-based RNG kernel lands.
- **Energy accounting**: the "GPU-resident" run loop now reports (in
  `GpuResidentReport`/the CSV energy log, at `output_interval` cadence) the
  full potential energy: the GPU-reduced bond + angle + dihedral + LJ +
  real-space-Coulomb energy (`potential_energy_gpu()`) plus the CPU-computed
  PME reciprocal-space energy, its self/background correction, and the
  excluded-pair real-space subtraction PME requires (folded in via the same
  `compute_cpu_extra` closure that supplies the CPU-side force
  contribution). This is now directly comparable to the `hybrid`/`cpu`
  backends' energy logs for a given topology (see
  `potential_energy_matches_cpu_reference`, a non-periodic parity test
  against `cpu::compute_forces_with_pme`, <1e-4 relative). Virial/pressure
  reduction on the GPU is not yet implemented (kinetic + potential energy
  only); a scalar pressure readout for this backend is a follow-up.
- **Trajectory writing**: `simulate_topology_gpu_resident` only writes a
  single trajectory frame (the final structure) rather than a frame per
  `output_interval` like `simulate_topology` does — the run loop doesn't
  stream intermediate structures back to the host today, only energy
  samples.

## A note on a since-fixed pre-existing observation

An earlier version of this document flagged an apparent factor-of-`r`
normalization discrepancy in `cpu::compute_nonbonded_forces` /
`cpu::compute_lj_forces` (their pair-force scalars were being multiplied by
the unnormalized separation vector `dr`, magnitude `r`, instead of being
pre-divided by `r`) and, as a workaround, checked the GPU-resident
nonbonded parity test against an independent closed-form reference instead
of the CPU path directly. That CPU bug has since been fixed ("Fix CPU
nonbonded pair forces off by a factor of r", finite-difference-verified via
`total_forces_match_finite_difference`), so the workaround is gone:
`nonbonded_lj_matches_cpu_reference`, `dihedral_forces_match_cpu_reference`
and `potential_energy_matches_cpu_reference` all compare directly against
`cpu::compute_forces_with_pme` now.

## Precision model

Positions/velocities/forces/parameters are `f32` in GPU buffers. Force
accumulation across kernels (nonbonded + bonded, potentially touching the
same atom) goes through `i32` fixed-point atomics at a `2^16` scale factor,
which is exact for forces roughly below `±32768` in the simulation's native
units and rounds to `~1.5e-5` below that — adequate for the LJ/Ewald/bonded
force magnitudes seen in practice, but this backend does not attempt
double-precision or `i64` fixed-point accumulation the way OpenMM's "mixed
precision" mode can.

## How to run the tests / benchmark

Rust parity + drift tests (no Python needed):

```bash
cd rust
cargo test --release forces::gpu_resident
```

These build a real `wgpu` device and skip themselves (with a message) if no
adapter is found — they are not `#[ignore]`d so `cargo test --release` always
attempts them.

Python/pytest (marked `gpu`):

```bash
uv run pytest -m gpu tests/validation/test_gpu_resident.py -v
```

The (ignored-by-default) throughput benchmark:

```bash
uv run pytest tests/validation/test_gpu_resident.py -k benchmark -m gpu -s --no-skip
# or just remove/override the @pytest.mark.skip locally when benchmarking
```

It prints `ns/day` for a small LJ cluster; compare against the same system
run through `hybrid`/`cpu` via `bin/simulate` to see relative throughput on
your hardware.

### Running without real GPU hardware (software Vulkan / lavapipe)

If no GPU is available, a software Vulkan implementation (Mesa's `lavapipe`)
lets `wgpu` find an adapter so the `gpu`-marked tests actually execute
(slowly) instead of skipping:

```bash
apt-get install -y mesa-vulkan-drivers vulkan-tools libvulkan1
export XDG_RUNTIME_DIR=/tmp/xdgrt   # vulkan loader wants this set; any writable dir
mkdir -p "$XDG_RUNTIME_DIR"
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json
export WGPU_BACKEND=vulkan
cargo test --release forces::gpu_resident
uv run pytest -m gpu tests/validation/test_gpu_resident.py -v
```

This is how the GPU-resident code in this change was actually exercised
(see the PR/task notes): every kernel above — cell list build, tiled
nonbonded LJ+Ewald with fixed-point atomics, bond/angle/dihedral kernels,
the per-item/per-atom potential-energy reduction, velocity-Verlet
integration, max-displacement rebuild trigger, KE reduction, V-rescale
scaling — ran and produced correct results on `lavapipe`/`llvmpipe` (Mesa's
software Vulkan/CPU compute implementation), not just compiled. The Phase 1
dihedral-kernel and energy-reduction work specifically added
`dihedral_forces_match_cpu_reference` and
`potential_energy_matches_cpu_reference` (both passing on lavapipe, <1e-4
relative error against `cpu::compute_forces_with_pme`), alongside the
existing `nonbonded_lj_matches_cpu_reference` (now also compared directly
against the CPU path — see the note above), `bond_forces_match_cpu_reference`
and `nve_energy_drift_bounded_over_1000_steps`; `uv run pytest -m gpu` also
ran green on the same lavapipe adapter. It is much slower than real GPU
hardware and is a correctness check, not a performance one; the owner
should re-run the benchmark on real hardware to get meaningful `ns/day`
numbers.

## CLI

```bash
bin/simulate topo.aqtop --backend gpu-resident --steps 5000 --dt 0.002 \
    --thermostat vrescale --temperature 300 --energy-log out.csv
bin/equilibrate topo.aqtop --backend gpu-resident --steps 2000 --thermostat langevin
```

`--backend gpu-resident` on either CLI refuses `--npt`/`--barostat` (and, on
`equilibrate`, `--restrain`) since those aren't implemented in this backend
yet, and only accepts `--thermostat` in `{none, vrescale, langevin}`.
`--backend {cpu,gpu,hybrid}` just sets `$AMPHI_FORCE_BACKEND` (and
`$AMPHI_HYBRID=0` for non-hybrid) before calling the existing
`simulate_topology`/`equilibrate_topology`, matching prior behavior when
`--backend` is omitted entirely.
