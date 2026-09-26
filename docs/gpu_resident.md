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
- Bonded harmonic bond and angle kernels (`bond_forces`, `angle_forces`),
  also accumulating into the fixed-point buffer.
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
- **Dihedral/improper bonded terms** are likewise evaluated on the CPU each
  step (`cpu::compute_dihedral_forces`, a new function factored out of the
  existing `compute_bonded_forces` so it doesn't double-count bonds/angles
  already done on the GPU) and folded in the same way as PME.
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
- **Energy accounting**: the "GPU-resident" run loop reports (in
  `GpuResidentReport`/the CSV energy log) the kinetic energy from the GPU
  reduction, and the potential energy from the CPU-computed
  dihedral+PME/exclusion-correction terms only — the GPU-computed
  bond/angle/nonbonded potential energy is **not currently reduced and
  reported** (only the forces are used). A GPU potential-energy reduction
  kernel analogous to `ke_reduce` is the natural next step; until then,
  `total_energy`/drift-based diagnostics from this backend undercount the
  true potential energy and should not be compared directly against the
  `hybrid`/`cpu` backends' energy logs.
- **Trajectory writing**: `simulate_topology_gpu_resident` only writes a
  single trajectory frame (the final structure) rather than a frame per
  `output_interval` like `simulate_topology` does — the run loop doesn't
  stream intermediate structures back to the host today, only energy
  samples.

## A note on an unrelated pre-existing observation

While writing a GPU-vs-CPU parity test for the new nonbonded kernel, the two
independent, already-shipped GPU kernels (`forces.wgsl`'s per-pair kernel,
used by the `gpu`/`hybrid` backends) and the standard closed-form LJ vector
force (`F_i = 24ε(2(σ/r)^12 − (σ/r)^6) / r² · r_ij`) agree with each other and
with this new kernel — but `cpu::compute_nonbonded_forces` /
`cpu::compute_lj_forces` multiply their (already `1/r`-scaled)
`lj_force_scalar` by the **unnormalized** separation vector `dr` (magnitude
`r`) instead of a unit vector, which looks like a pre-existing factor-of-`r`
normalization discrepancy in the plain CPU path, unrelated to this change.
It was not touched here (out of scope, high blast radius); the new
GPU-resident parity test (`nonbonded_lj_matches_cpu_reference`, despite its
name — see its comments) is checked against the closed-form reference and
the existing GPU kernel's convention instead, to avoid inheriting a possibly
wrong answer. This is worth a dedicated follow-up investigation.

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
nonbonded LJ+Ewald with fixed-point atomics, bonded kernels, velocity-Verlet
integration, max-displacement rebuild trigger, KE reduction, V-rescale
scaling — ran and produced correct results on `llvmpipe` (a software CPU
rasterizer/compute implementation), not just compiled. It is much slower
than real GPU hardware and is a correctness check, not a performance one;
the owner should re-run the benchmark on real hardware to get meaningful
`ns/day` numbers.

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
