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

## GPU RNG + thermostats: Philox4x32-10, BAOAB Langevin, V-rescale

- **Philox4x32-10** (Salmon, Moraes, Dror & Shaw, 2011; the Random123
  algorithm) is implemented twice, kept in lockstep: `rust/src/random/mod.rs`
  (host reference, checked against the two published Random123
  `kat_vectors` entries in `philox_matches_random123_kat_vectors`) and
  `gpu_resident.wgsl`'s `philox4x32_10`/`mulhilo32`/`box_muller` (the same
  round structure, ported to `u32`/`vec4<u32>` — WGSL has no native 64-bit
  integer type, so `mulhilo32` does the 32x32->64 product via 16-bit-limb
  long multiplication instead of a native widening multiply).
  `wgsl_philox_matches_random123_kat_vectors` dispatches the actual WGSL
  kernel (via a small `philox_debug` entry point + a `philox_test_uniform`
  input buffer) for both KAT vectors and checks it against the Rust
  reference bit-for-bit, on real lavapipe hardware, not just in isolation.
  Every atom/DOF gets an independent draw from `(key, counter)` alone —
  `key = (seed, 0)` fixed for the run, `counter = (step, atom_index, 0, 0)`
  — so there's no shared RNG state and no dependency on dispatch order.
- **BAOAB Langevin** (`GpuResidentEngine::baoab_langevin_step`,
  `langevin_o_step` in the shader) replaces the old scalar velocity-rescale
  approximation with the real algorithm (Leimkuhler & Matthews): B (half
  kick) - A (half drift) - O (per-atom Philox+Box-Muller velocity
  randomization, `v' = c1*v + c2*sqrt(kB*T/m)*N(0,1)`, `c1 = exp(-gamma*dt)`,
  `c2 = sqrt(1-c1^2)`) - A (half drift) - B (half kick), with SETTLE/SHAKE
  position constraints re-applied after each A and velocity constraints
  re-applied after O and after each B. `langevin_mean_temperature_within_
  3_percent_of_target` validates this end-to-end (a periodic 64-ion
  LJ+Coulomb+PME fluid, no bonds) to the same 3%-of-target acceptance
  criterion `integrator::tests::vrescale_reproduces_target_mean_temperature`
  uses for the CPU thermostat.
  **Known limitation, found and not yet fixed**: the SETTLE+Langevin
  coupling (the two-reference-snapshot scheme around the O-step, described
  above) was unstable for denser, multi-water periodic systems in testing —
  kinetic energy ran away to unphysical values within a few hundred steps,
  confirmed not to be an initial-overlap artifact. A single rigid water
  under the same thermostat keeps its SETTLE bond lengths exactly correct
  throughout (`langevin_settle_keeps_water_rigid`) and shows large but
  bounded temperature fluctuations consistent with its very small number of
  degrees of freedom, not a runaway — so the instability specifically needs
  more waters/denser packing to reproduce, and the root cause (most likely
  in how per-atom noise interacts with the RATTLE-style velocity projection
  across two separate half-drift reference frames) hasn't been isolated
  yet. **`thermostat = Langevin` should be treated as validated only for
  systems without SETTLE/SHAKE constraints until this is fixed**;
  `thermostat = VRescale` (which doesn't restructure the integration around
  SETTLE — see below) has no such issue, as
  `constrained_nve_1000_steps_bounded_drift` and the existing SETTLE parity
  tests show.
- **V-rescale** keeps its GPU KE reduction (`kinetic_energy()` — "only the
  scalar comes back" was already true before this change) and now draws its
  one stochastic scalar input from the tested Philox/Box-Muller pair
  (`crate::random::philox4x32_10`/`box_muller`, host-side, keyed by
  `(seed, step)`) instead of an ad hoc xorshift generator. This thermostat
  path is unaffected by the Langevin/SETTLE issue above, since it doesn't
  change the integrator's structure (still the original single-drift
  kick/drift/kick loop with a post-hoc velocity rescale).

## PME: GPU spreading + CPU FFT, no per-atom readback

PME reciprocal-space electrostatics is a hybrid: **GPU B-spline charge
spreading and force gathering, CPU FFT**, chosen explicitly over a from
-scratch GPU FFT (a legitimate alternative the task allowed) because
`electrostatics::pme` already has a well-tested, exact `rustfft`-based
pipeline (`fft3d`, `bspline_moduli`, the influence-function multiply) that a
new WGSL radix-2/mixed-radix FFT would have to reproduce bit-for-bit to stay
numerically equivalent to the `cpu`/`hybrid` backends — implementing that
from scratch was judged higher-risk than reusing already-correct code, for
the same amount of "no position readback" benefit. Concretely, each step
(`GpuResidentEngine::pme_and_exclusion_step`):

1. `pme_clear_grid` zeroes a fixed-point (`i32`, `PME_FP_SCALE = 2^20`)
   charge-density grid (`q_grid_fp`, size `nx*ny*nz`).
2. `pme_spread` (one thread per atom) computes each atom's fractional grid
   coordinate and cardinal B-spline weights — `fill_bspline`/`atom_spline_dim`
   in `gpu_resident.wgsl` are a literal port of `electrostatics::pme`'s
   `fill_bspline`/`atom_splines` (same de Boor recursion, same weight/grid
   -point convention) — and atomically adds `charge * weight` to the
   relevant `order^3` grid cells.
3. The host reads back **the grid**, not positions: `q_grid_fp` is
   `nx*ny*nz` fixed-point values, a size set by the Fourier grid spacing
   and independent of atom count (a few hundred KB for a typical box), not
   an `O(N)` transfer. This is what "positions never come back per step"
   means in practice here — the transferred quantity is the charge density,
   which the GPU already reduced from per-atom contributions.
4. `electrostatics::pme::pme_recip_from_grid` — the exact forward-FFT /
   B-spline-moduli-corrected influence-function / inverse-FFT pipeline
   `compute_pme_forces` itself now calls (refactored so both share one
   implementation) — turns the charge grid into a potential grid
   (`2*Re(theta_q)` after the inverse FFT) plus the scalar reciprocal
   energy/virial, on the CPU, in `f64`.
5. The host uploads the potential grid; `pme_gather` (one thread per atom)
   recomputes the same B-spline weights/derivatives and reads the uploaded
   grid to compute each atom's force, atomically adding it to the shared
   `force_fp` fixed-point accumulator (same one bonded/nonbonded write
   into) — a literal port of `compute_pme_forces`'s force-gather loop.
6. `exclusion_correction` (one thread per atom) applies the erf-based
   real-space subtraction PME requires for excluded/1-4 pairs, from the
   same per-atom `exclusions`/`pairs14` tables the nonbonded kernel already
   has — a literal, per-atom-half-energy port of
   `electrostatics::excluded_pair_correction`.

The self-energy/neutralizing-background correction
(`ewald_energy_correction_with_alpha`) depends only on charges and box
volume, not positions, so it's computed once at engine construction
(`pme_self_bg_energy`) and reused every step for free.

**A real bug found and fixed along the way**: writing the parity test for
this (`pme_forces_and_energy_match_cpu_reference`) caught a pre-existing 2x
error in `electrostatics::excluded_pair_correction` itself — it iterated
`topology.exclusions` directly, which stores each excluded bond in *both*
directions (`build_exclusions` pushes `[i,j]` and `[j,i]`), applying every
pair's correction twice. This affected the `cpu`/`hybrid` backends' PME
exclusion correction too, not just `gpu-resident` — any periodic system
with bonded exclusions had its exclusion-correction energy and forces
double what they should have been. Fixed by deduplicating to unique
unordered pairs before applying the correction; see that function's updated
doc comment and `exclusion_correction_kernel_matches_cpu_reference_isolated`.

Parity: `pme_forces_and_energy_match_cpu_reference` compares against
`cpu::compute_forces_with_pme` + the self/background correction (the same
"full PME energy" convention `forces::compute_forces` uses) on both an
exclusion-free and an exclusion/1-4-bearing periodic system — forces to a
few-per-mille and energy to <5e-4 relative (looser than the isolated
per-kernel parity tests' 1e-4, since this exercises bond+angle+dihedral+
nonbonded+PME+exclusion together, in `f32` throughout the GPU side vs `f64`
throughout the CPU side — see that test's doc comment for why that gap is
expected and not itself evidence of a further bug).
`zero_readbacks_on_non_output_steps_with_pme` confirms the "no position
readback" property end to end: a periodic run's `readback_count()` doesn't
grow with step count.

Not done: overlapping the CPU FFT with the *next* step's GPU nonbonded
dispatch (they currently run sequentially: GPU spread -> CPU FFT -> GPU
gather, blocking); that overlap is a natural follow-up once the grid
round trip is on its own submission rather than synchronous.

## What stays on the CPU (by design, and why)

- **Dihedral/improper bonded terms** are now evaluated on the GPU (see
  "What's actually resident" above) — this was previously a per-step CPU
  round trip (`cpu::compute_dihedral_forces`) and is no longer.
- **SETTLE and solute-H-bond SHAKE constraints** are now evaluated on the
  GPU (see "Constraints" below) — this was previously a per-step CPU round
  trip (`constraints::apply_settle_analytic`/`apply_settle_velocity`) for
  water, and unimplemented at all for solute H-bonds. LINCS specifically
  (as opposed to SHAKE) is still not implemented in the GPU-resident path;
  systems needing solute H-bond constraints stay on `gpu-resident` fine
  (SHAKE, not LINCS, is used there regardless of `ConstraintAlgorithm`).
  **NPT (Berendsen and Monte Carlo barostats) is now implemented** — see
  "NPT: virial, pressure and barostats" below; `--npt`/`--barostat` are no
  longer refused on `--backend gpu-resident`.

## NPT: virial, pressure and barostats

`--backend gpu-resident --barostat {berendsen,montecarlo}` (or `--npt` on
`bin/simulate`, which defaults to Berendsen) is now supported, via
`GpuResidentEngine::pressure`/`apply_berendsen_barostat_gpu`/
`apply_mc_barostat_gpu` in `rust/src/forces/gpu_resident/mod.rs`. This is
implemented with a deliberately different strategy than a from-scratch WGSL
virial-reduction kernel per force term:

- **Virial/pressure**: rather than adding per-pair/per-bond virial
  accumulator buffers to every kernel (`nonbonded`, `bond_forces`, the PME
  gather, the SETTLE/SHAKE correction kernels — a large, bind-group-touching
  change), `GpuResidentEngine::pressure` reads back the current positions
  (`snapshot_topology`, one `O(n)` transfer — the same kind of readback the
  engine already does at `output_interval` cadence) into a CPU `Topology`
  and calls the exact, already-tested `cpu::compute_virial` (2-body bonded +
  nonbonded LJ/real-space-Coulomb via the CPU neighbor list, plus, when
  periodic, the PME reciprocal-space virial from `pme_recip_from_grid` and
  the excluded/1-4 correction's virial — all bit-for-bit the same formula
  `integrator::compute_step_forces` uses). Kinetic energy still comes from
  the GPU KE reduction (`kinetic_energy()`) — only the pairwise/PME virial
  term is computed on the CPU, and only at the cadence pressure is actually
  needed (output steps and barostat-attempt steps), never in the per-step
  integration loop. `pressure_matches_cpu_reference_on_water_snapshot`
  checks this against the identical formula computed entirely on the CPU
  (`(2*KE + W) / (3V) * PRESSURE_CONV`) on a periodic water snapshot, to
  <1% relative. This intentionally does not include the constraint-virial
  correction term `integrator::compute_constraint_virial` adds mid-step (a
  further simplification — SETTLE/SHAKE's own kernels don't currently
  accumulate a virial contribution), so a running trajectory's
  `GpuResidentReport::pressure` will be a somewhat noisier/biased estimate
  than the CPU integrator's for tightly constrained (heavily SETTLE'd)
  systems; it is exact for the same snapshot the CPU path evaluates. A
  documented follow-up: fold constraint-virial accumulation into the
  `settle_position`/`shake_correction_pass`/`apply_shake_correction` kernels
  the way the CPU integrator does after `apply_constraints`.
- **Berendsen barostat** (`apply_berendsen_barostat_gpu`): a literal port of
  `integrator::apply_berendsen_barostat` — same `pressure`-based scale
  factor, same `.clamp(0.98, 1.02)` per-step limit — applied via a position
  read/atomic-scale/write round trip (not a new WGSL scaling kernel; the
  round trip only happens once a step and is the same size as the
  input/output the engine already reads back for logging).
- **Monte Carlo barostat** (`apply_mc_barostat_gpu`): a literal port of
  `integrator::apply_mc_barostat` — the same random +/-2% volume move,
  molecule-COM scaling when the topology carries more than one
  `molecule_id` (else atomic scaling), and the same Metropolis acceptance
  criterion `-(dE + p*dV - N*kT*ln(V'/V))/kT`. Per the task's suggested
  approach, the trial energy before/after the volume move comes from a
  **GPU energy re-evaluation** (`evaluate_energy_gpu`: a fresh
  `compute_nonbonded_and_bonded` + `pme_and_exclusion_step` dispatch +
  `potential_energy_gpu()` readback), not a CPU `compute_forces` call — the
  MC barostat stays "GPU-resident" for its most expensive step (two full
  force/energy evaluations per attempt), only the positions/box round-trip
  through the host. On rejection, positions and box are restored and forces
  are re-evaluated once more so the (unchanged) state's force buffers stay
  consistent for the next integration step.
- **Box updates and the neighbor list**: both barostats update
  `GpuResidentEngine::box_dims` (the source of truth for the sim-params
  uniform used by the nonbonded cell stencil and, since `pme_and_exclusion_
  step` reads `self.box_dims` directly, PME's reciprocal-space box) and
  force an immediate `rebuild_neighbor_list()`. The neighbor **cell grid**
  itself (`cells`/`num_cells`, and the buffers sized from it) is fixed at
  engine construction from the *initial* box/cutoff and is not resized as
  the box fluctuates — a documented simplification acceptable for the
  small (a few percent) per-move volume changes both barostats make; a
  barostat run that drifts the box far from its starting size over a very
  long run would eventually want a resize, which isn't implemented. The PME
  grid (`pme_grid`, B-spline order) is likewise kept fixed — "update the
  reciprocal box, not the grid" — since `pme_recip_from_grid` already takes
  `box_dims` as a parameter independent of the grid's cell counts.
- **Where it's plumbed through**: `GpuResidentEngine::run`/`run_gpu_resident`
  take `barostat: GpuResidentBarostat, target_pressure, tau_p,
  compressibility, mc_interval` (mirroring `integrator::MdState`'s fields);
  `simulate_topology_gpu_resident` (PyO3) exposes them as
  `barostat`/`pressure`/`tau_p`/`compressibility`/`mc_interval` and now
  returns `(topology, final_temperature, final_pressure)`; `bin/simulate`
  and `bin/equilibrate` no longer refuse `--npt`/`--barostat` for
  `--backend gpu-resident`. `GpuResidentRunResult` also carries
  `mc_attempts`/`mc_accepts` so callers (and
  `mc_barostat_moves_density_toward_target_with_sane_acceptance`) can check
  the acceptance ratio is sane.
- **Not done**: constraint-virial accumulation in the SETTLE/SHAKE kernels
  (noted above), a from-scratch GPU virial reduction (the CPU-snapshot
  approach was judged lower-risk and reuses already-tested code, the same
  tradeoff PME reciprocal-space made), and anisotropic/semi-isotropic box
  scaling (isotropic only, matching the CPU integrator's `Barostat` enum).

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
- **Virial/pressure and barostats are now implemented** (see "NPT: virial,
  pressure and barostats" above) — `GpuResidentReport` now also carries
  `pressure`, and `--npt`/`--barostat {berendsen,montecarlo}` work on
  `--backend gpu-resident`. The virial itself is computed via a CPU-snapshot
  reuse of `cpu::compute_virial`, not a from-scratch WGSL virial-reduction
  kernel per force term, and does not (yet) include the constraint-virial
  correction the CPU integrator adds mid-step — see that section for the
  detail and the follow-up this leaves.
- **Energy accounting**: the "GPU-resident" run loop now reports (in
  `GpuResidentReport`/the CSV energy log, at `output_interval` cadence) the
  full potential energy: the GPU-reduced bond + angle + dihedral + LJ +
  real-space-Coulomb energy (`potential_energy_gpu()`) plus the CPU-computed
  PME reciprocal-space energy, its self/background correction, and the
  excluded-pair real-space subtraction PME requires. This is now directly
  comparable to the `hybrid`/`cpu` backends' energy logs for a given
  topology (see `potential_energy_matches_cpu_reference`, a non-periodic
  parity test, and `pme_forces_and_energy_match_cpu_reference` for the
  periodic/PME case) — other than the missing virial noted above, so
  there's still no pressure readout for this backend.
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
uv run pytest tests/validation/test_gpu_resident.py -k benchmark -m gpu -s
# (skipped by default via @pytest.mark.skip; comment that decorator out
# locally to actually run it — pytest has no built-in "un-skip" flag)
```

It builds a periodic TIP3P water box (`_tip3p_water_box_aqtop`, 12 waters/
axis = 1728 waters = 5184 atoms by default — the low end of a "5-20k atoms"
target, chosen so it still finishes in well under a minute on
lavapipe/software Vulkan; raise `n_per_axis` for a bigger box on real GPU
hardware) with both SETTLE and PME active — the full path this change
completes — and prints `ns/day`; compare against the same system run
through `hybrid`/`cpu` via `bin/simulate` to see relative throughput on
your hardware. Measured on lavapipe (this container, `AGENTS.md`'s
sanctioned software-Vulkan setup): **5184 atoms, 200 steps @ 0.5fs in
17.9s -> 0.48 ns/day**. This is a software CPU rasterizer's compute path,
not a GPU — it is not a useful absolute number (a real discrete GPU running
the same kernels would be expected to be one to several orders of
magnitude faster; GROMACS/OpenMM-class GPU MD for a system this size
typically reaches tens to low hundreds of ns/day on real hardware) — its
only purpose here is confirming the full GPU-resident path (SETTLE + PME +
all bonded/nonbonded kernels) runs to completion end-to-end at a
production-scale-adjacent atom count, not just on the few-atom unit-test
topologies. The owner should re-run this on real GPU hardware for a
meaningful number.

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
(see the PR/task notes): every kernel — cell list build, tiled nonbonded
LJ+Ewald (with 1-4 scaling) with fixed-point atomics, bond/angle/dihedral
kernels, the per-item/per-atom potential-energy reduction, velocity-Verlet
and BAOAB Langevin integration, max-displacement rebuild trigger, KE
reduction, V-rescale scaling, SETTLE/SHAKE (position + velocity,
Jacobi-parallel for SHAKE), Philox4x32-10 + Box-Muller, and PME GPU
spreading/gathering + exclusion correction — ran and produced correct
results on `lavapipe`/`llvmpipe` (Mesa's software Vulkan/CPU compute
implementation), not just compiled. Rust tests, all passing on lavapipe
(`cargo test --release forces::gpu_resident`, 18 tests): the 9 from earlier
phases plus `pme_forces_and_energy_match_cpu_reference`,
`exclusion_correction_kernel_matches_cpu_reference_isolated`,
`zero_readbacks_on_non_output_steps_with_pme`,
`wgsl_philox_matches_random123_kat_vectors`,
`langevin_mean_temperature_within_3_percent_of_target`,
`langevin_settle_keeps_water_rigid` (plus `rust/src/random/mod.rs`'s own
`philox_matches_random123_kat_vectors` and two supporting RNG tests). `uv
run pytest -m gpu` also ran green on the same lavapipe adapter. It is much
slower than real GPU hardware and is a correctness check, not a performance
one; the owner should re-run the benchmark on real hardware to get
meaningful `ns/day`
numbers.

## CLI

```bash
bin/simulate topo.aqtop --backend gpu-resident --steps 5000 --dt 0.002 \
    --thermostat vrescale --temperature 300 --energy-log out.csv
bin/equilibrate topo.aqtop --backend gpu-resident --steps 2000 --thermostat langevin
bin/simulate topo.aqtop --backend gpu-resident --steps 5000 --dt 0.002 \
    --thermostat vrescale --temperature 300 --npt --barostat montecarlo --pressure 1.0
```

`--backend gpu-resident` on either CLI refuses position restraints on
`equilibrate` (`--restrain`) since those aren't implemented in this backend
yet, and only accepts `--thermostat` in `{none, vrescale, langevin}`.
`--npt`/`--barostat {none, berendsen, montecarlo}` are supported (see "NPT:
virial, pressure and barostats" above). `--backend {cpu,gpu,hybrid}` just
sets `$AMPHI_FORCE_BACKEND` (and
`$AMPHI_HYBRID=0` for non-hybrid) before calling the existing
`simulate_topology`/`equilibrate_topology`, matching prior behavior when
`--backend` is omitted entirely.
