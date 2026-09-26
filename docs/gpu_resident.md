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
- **1-4 (dihedral end-atom) LJ/Coulomb scaling**: the `nonbonded` kernel now
  carries a small fixed-capacity per-atom "1-4 partner" table (`pairs14`,
  `MAX_14 = 8`, built by `build_14_table`/`cpu::build_14_pairs`, uploaded the
  same way exclusions are) and an `is_14` lookup exactly mirroring
  `is_excluded`. A pair found in that table gets `LJ_14_SCALE`/
  `COULOMB_14_SCALE` applied to its LJ epsilon and Coulomb `qq`, matching
  `cpu::compute_nonbonded_forces`'s convention bit-for-bit. This closes the
  gap flagged in an earlier version of this document (see below).
- **SETTLE** (rigid 3-site water) and **SHAKE** (solute H-bond constraints)
  now run entirely on the GPU, with no per-step position/velocity readback
  (see "Constraints" below).

## Constraints: SETTLE + SHAKE, entirely GPU-resident

- **SETTLE** (`settle_position`, `settle_velocity` in `gpu_resident.wgsl`) is
  a literal, one-thread-per-water WGSL port of
  `constraints::settle_one`/`constraints::apply_settle_velocity` — the same
  canonical-frame Miyamoto & Kollman (1992) closed-form solve (steps 1-5:
  build the local frame from the *reference* O-H/O-H vectors, solve the
  planar triangle for `sinphi`/`sinpsi`/`sintheta`, rotate back), not an
  iterative approximation. `snapshot_settle_ref` copies the current
  positions into a `settle_ref_positions` buffer on the GPU immediately
  before each step's `kick_half`+`drift` (the CPU version's `reference`
  parameter — "positions at the start of the step, already on the
  constraint manifold" — now never leaves the GPU). `settle_position` reads
  that snapshot plus the drifted (unconstrained) positions and writes the
  corrected positions in place; `settle_velocity` similarly corrects
  velocities after the second half-kick. Parity: `settle_position_matches_
  cpu_reference` checks a perturbed water against
  `constraints::apply_settle_analytic` to <1e-5; `settle_velocity_
  orthogonal_to_bonds_on_gpu` and `constrained_nve_1000_steps_bounded_drift`
  check the GPU path's own invariants (zero bond-length time-derivative,
  bounded O-H/H-H drift over 1000 steps) the way the existing CPU SETTLE
  tests do.
- **SHAKE** (`shake_correction_pass`/`apply_shake_correction` for positions,
  `shake_velocity_pass`/`apply_shake_velocity_correction` for the RATTLE
  velocity projection) constrains solute H-bonds
  (`ConstraintSet::shake_bonds`, uploaded as `shake_idx`/`shake_r0`). Unlike
  `constraints::apply_shake`, which is a sequential Gauss-Seidel iteration
  over bonds (fine on one CPU thread, since each bond's correction is
  visible to the next), the GPU dispatches every bond's correction
  concurrently, so this is a **Jacobi** variant: each bond thread computes
  its correction from the same position snapshot and atomically accumulates
  a fixed-point delta per atom (reusing the `pos_correction_fp` scratch
  buffer, cleared after each apply), and a second kernel applies the
  accumulated deltas. `GpuResidentEngine::run` dispatches this
  correct/apply pair a fixed 25 times per half-step (`SHAKE_ITERATIONS`)
  rather than reading back a convergence residual the way
  `constraints::apply_shake`'s `SHAKE_TOLERANCE` loop does — a documented
  simplification (fixed iteration count trades a small amount of
  unnecessary work on already-converged constraints for avoiding a
  per-iteration readback; 25 Jacobi iterations comfortably converges typical
  X-H bond corrections given the position deltas involved). The velocity
  pass is a direct, single-iteration-per-dispatch (also repeated 25x) port
  of `constraints::apply_rattle`'s `shake_bonds` loop.
- **No per-step readback for either.** Before this change, both constraint
  types required a synchronous position (and, for SETTLE, velocity) readback
  every step; now the whole kick/drift/constrain/kick cycle stays on the GPU
  when PME is off. See `zero_readbacks_on_non_output_steps_without_pme`
  below and the "Readback accounting" section.

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
- **SETTLE and solute-H-bond SHAKE constraints** are now evaluated on the
  GPU (see "Constraints" below) — this was previously a per-step CPU round
  trip (`constraints::apply_settle_analytic`/`apply_settle_velocity`) for
  water, and unimplemented at all for solute H-bonds. LINCS specifically
  (as opposed to SHAKE) and **barostats are still not implemented** in the
  GPU-resident path (`--backend gpu-resident` refuses `--npt`/`--barostat`
  at the CLI). Systems needing NPT/a barostat still use `hybrid` (the
  default) or `cpu`; systems needing solute H-bond constraints can now stay
  on `gpu-resident` (SHAKE, not LINCS, is used there regardless of
  `ConstraintAlgorithm`).

## Deviations from the "textbook" design (and why)

- **Exclusions**: a small fixed-capacity (`MAX_EXCL = 8`) per-atom exclusion
  list uploaded from `topology.exclusions`, rather than a per-tile bitmask.
  Correct for the same reason the CPU/GPU per-pair backends' exclusion lists
  are (bonded exclusion lists are tiny), but doesn't get the SIMD-friendly
  bitmask-test property a real tile-bitmask design would.
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

## The same 1-4 gap in the plain `gpu`/`hybrid` backend

The `gpu-resident` nonbonded kernel wasn't the only place missing 1-4
LJ/Coulomb scaling: the older per-pair `forces.wgsl` kernel (used by both
the standalone `gpu` backend and `hybrid`'s GPU nonbonded half, via
`GpuForceEngine::compute_nonbonded` in `rust/src/forces/gpu/mod.rs`) had the
same gap — its pair list (`build_neighbor_list`'s `nl.pairs`) already
excludes 1-2/1-3 pairs but includes 1-4 pairs at full strength, with no
scale factor anywhere in the kernel or its host-side energy sum. Since
`hybrid` is this repo's **default** compute backend (`AMPHI_HYBRID=1`), this
was a real correctness gap for any topology with dihedrals run through the
default path, not just an edge case. Fixed the same way as `gpu-resident`:
`forces.wgsl` gained a `pair_scale: array<vec2<f32>>` buffer (one
`(lj_scale, coulomb_scale)` pair per neighbor-list entry, `(1,1)` normally
and `(LJ_14_SCALE, COULOMB_14_SCALE)` for a 1-4 pair, built host-side from
`cpu::build_14_pairs` exactly like `compute_nonbonded_forces` does), and the
host-side energy sum in `GpuForceEngine::compute_nonbonded` now applies the
same scale. `hybrid`/`gpu` share this one code path, so the fix covers both
backends from a single change.

## Readback accounting

`GpuResidentEngine` now tracks a `readback_count` (an `AtomicU64`,
incremented only by `read_positions()`/`read_velocities()` — the O(n)
per-atom GPU->CPU transfers that PME and, before this change, CPU
SETTLE/dihedral folding needed every step). Small constant-size scalar
readbacks (`max_disp` for the neighbor-rebuild trigger, `cell_count` for the
prefix-scan simplification, the per-item/per-atom energy-reduction buffers
on output steps) are deliberately **not** counted — they're either O(1),
bounded by cell count rather than atom count, or already gated to output
steps only, so counting them would muddy the specific claim this metric
backs: with PME off, a non-output step now does zero large per-atom
readbacks. `zero_readbacks_on_non_output_steps_without_pme` asserts this by
running the same non-periodic, unconstrained/water topology for two
different step counts (5 and 200) with `output_interval: 0` (no output
steps at all) and checking `readback_count()`'s delta is identical
(specifically, the 2 unavoidable end-of-run reads that build the returned
topology) regardless of step count — proving no readback crept back into
the per-step loop. PME reciprocal-space evaluation still forces one
`read_positions()`/step when the system is periodic (`needs_pme`); that
readback is explicitly out of scope for this change and is a Phase 3 item
(see "What stays on the CPU").

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
nonbonded LJ+Ewald (with 1-4 scaling) with fixed-point atomics,
bond/angle/dihedral kernels, the per-item/per-atom potential-energy
reduction, velocity-Verlet integration, max-displacement rebuild trigger,
KE reduction, V-rescale scaling, and now SETTLE/SHAKE (position + velocity,
Jacobi-parallel for SHAKE) — ran and produced correct results on
`lavapipe`/`llvmpipe` (Mesa's software Vulkan/CPU compute implementation),
not just compiled. Rust tests, all passing on lavapipe (`cargo test
--release forces::gpu_resident`, 9 tests):
`nonbonded_lj_matches_cpu_reference` (now with a realistic charged
dihedral-bearing topology exercising the 1-4 scaling path, compared
directly against `cpu::compute_forces_with_pme`), `bond_forces_match_
cpu_reference`, `dihedral_forces_match_cpu_reference`, `potential_energy_
matches_cpu_reference` (all <1e-4 relative), `settle_position_matches_
cpu_reference` (<1e-5 vs `constraints::apply_settle_analytic`),
`settle_velocity_orthogonal_to_bonds_on_gpu`, `nve_energy_drift_bounded_
over_1000_steps`, `constrained_nve_1000_steps_bounded_drift` (rigid-water
bond/H-H distances held within 1e-4 nm over 1000 steps), and
`zero_readbacks_on_non_output_steps_without_pme`. `uv run pytest -m gpu`
also ran green on the same lavapipe adapter. It is much slower than real
GPU hardware and is a correctness check, not a performance one; the owner
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
