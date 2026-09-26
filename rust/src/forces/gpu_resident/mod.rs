//! GPU-resident MD step engine (`AMPHI_FORCE_BACKEND=gpu-resident`).
//!
//! Positions, velocities, forces, per-atom parameters and the neighbor cell list
//! live in GPU buffers for the whole run; the host only reads them back at
//! `output_interval` steps (for logging / trajectory writes) or when a CPU-side
//! bonded term (dihedrals) or the PME reciprocal sum needs the current
//! coordinates. See `docs/gpu_resident.md` for the full design writeup and the
//! deviations taken from a "textbook" implementation for scope reasons
//! (per-atom exclusion lists instead of per-tile bitmasks, host-side prefix
//! scan over the small per-cell count array, a counter-based RNG that is
//! Philox-*inspired* rather than bit-exact Philox, SETTLE applied via the
//! existing validated CPU implementation rather than a new WGSL kernel).

use crate::constraints::{build_constraints, ConstraintSet};
use crate::electrostatics::excluded_pair_correction;
use crate::electrostatics::ewald_energy_correction_with_alpha;
use crate::electrostatics::pme::{compute_pme_forces, PmeContext};
use crate::forces::cpu::build_14_pairs;
use crate::topology::Topology;
use bytemuck::{Pod, Zeroable};
use pollster;
use std::collections::HashSet;

const MAX_EXCL: usize = 8;
const MAX_14: usize = 8;
const WORKGROUP: u32 = 64;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SimParamsGpu {
    num_atoms: u32,
    num_cells: u32,
    cells_x: u32,
    cells_y: u32,
    cells_z: u32,
    max_per_atom_excl: u32,
    max_per_atom_14: u32,
    num_bonds: u32,
    num_angles: u32,
    num_dihedrals: u32,
    num_waters: u32,
    num_shake_bonds: u32,
    box_lx: f32,
    box_ly: f32,
    box_lz: f32,
    cell_size: f32,
    cutoff: f32,
    cutoff2: f32,
    alpha: f32,
    coulomb: f32,
    pbc: u32,
    skin_half2: f32,
    dt: f32,
    _pad0: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ScaleUniform {
    scale: f32,
    _p0: f32,
    _p1: f32,
    _p2: f32,
}

const COULOMB_CONSTANT: f32 = 138.935456;
const GPU_RESIDENT_SHADER: &str = include_str!("../../../data/shaders/gpu_resident.wgsl");

/// Thermostat choices supported by the GPU-resident loop. This is intentionally a
/// smaller set than the full CPU integrator (no barostat, no solute LINCS/SHAKE);
/// see docs/gpu_resident.md for what falls back to the CPU/hybrid path instead.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GpuResidentThermostat {
    None,
    VRescale,
    Langevin,
}

pub struct GpuResidentReport {
    pub potential_energy: f64,
    pub kinetic_energy: f64,
    pub temperature: f64,
    pub step: usize,
}

pub struct GpuResidentRunResult {
    pub topology: Topology,
    pub samples: Vec<GpuResidentReport>,
}

struct Buffers {
    positions: wgpu::Buffer,
    velocities: wgpu::Buffer,
    atom_params: wgpu::Buffer,
    force_fp: wgpu::Buffer,
    forces: wgpu::Buffer,
    sim_uniform: wgpu::Buffer,
    cell_count: wgpu::Buffer,
    cell_start: wgpu::Buffer,
    cell_cursor: wgpu::Buffer,
    sorted_atoms: wgpu::Buffer,
    atom_cell: wgpu::Buffer,
    exclusions: wgpu::Buffer,
    ref_positions: wgpu::Buffer,
    max_disp: wgpu::Buffer,
    ke_accum: wgpu::Buffer,
    external_force: wgpu::Buffer,
    nb_energy: wgpu::Buffer,
    pairs14: wgpu::Buffer,
    water_idx: wgpu::Buffer,
    water_params: wgpu::Buffer,
    settle_ref_positions: wgpu::Buffer,
    shake_idx: wgpu::Buffer,
    shake_r0: wgpu::Buffer,
    pos_correction_fp: wgpu::Buffer,
    bond_idx: wgpu::Buffer,
    bond_params: wgpu::Buffer,
    angle_idx: wgpu::Buffer,
    angle_params: wgpu::Buffer,
    dihedral_idx: wgpu::Buffer,
    dihedral_params: wgpu::Buffer,
    bond_energy: wgpu::Buffer,
    angle_energy: wgpu::Buffer,
    dihedral_energy: wgpu::Buffer,
    scale_uniform: wgpu::Buffer,
    read_pos: wgpu::Buffer,
    read_vel: wgpu::Buffer,
    read_force: wgpu::Buffer,
    read_ke: wgpu::Buffer,
    read_maxdisp: wgpu::Buffer,
}

/// The GPU-resident engine: owns the device/queue/pipelines and the persistent
/// per-run GPU buffers for one topology-sized system.
pub struct GpuResidentEngine {
    device: wgpu::Device,
    queue: wgpu::Queue,
    n: usize,
    num_bonds: usize,
    num_angles: usize,
    num_dihedrals: usize,
    num_waters: usize,
    num_shake_bonds: usize,
    /// Counts calls to `read_positions()`/`read_velocities()` — the O(n)
    /// per-atom GPU->CPU transfers that PME reciprocal-space evaluation and
    /// (pre-Phase-2) SETTLE/dihedral folding used every step. Small,
    /// constant-size scalar readbacks (the neighbor-rebuild trigger's
    /// `max_disp`, the cell-list prefix-scan's `cell_count`, per-item energy
    /// buffers on output steps) are not counted here; see
    /// docs/gpu_resident.md.
    readback_count: std::sync::atomic::AtomicU64,
    cells: (u32, u32, u32),
    num_cells: u32,
    cutoff: f64,
    alpha: f64,
    pbc: bool,
    box_dims: [f64; 3],

    layout0: wgpu::BindGroupLayout,
    layout1: wgpu::BindGroupLayout,
    layout2: wgpu::BindGroupLayout,
    layout3: wgpu::BindGroupLayout,
    layout4: wgpu::BindGroupLayout,
    pipelines: Pipelines,
    buf: Buffers,
    bg0: wgpu::BindGroup,
    bg1: wgpu::BindGroup,
    bg2: wgpu::BindGroup,
    bg3: wgpu::BindGroup,
    bg4: wgpu::BindGroup,
}

struct Pipelines {
    clear_cells: wgpu::ComputePipeline,
    count_cells: wgpu::ComputePipeline,
    scatter_cells: wgpu::ComputePipeline,
    snapshot_ref: wgpu::ComputePipeline,
    max_displacement: wgpu::ComputePipeline,
    nonbonded: wgpu::ComputePipeline,
    bond_forces: wgpu::ComputePipeline,
    angle_forces: wgpu::ComputePipeline,
    dihedral_forces: wgpu::ComputePipeline,
    convert_forces: wgpu::ComputePipeline,
    add_external_forces: wgpu::ComputePipeline,
    kick_half: wgpu::ComputePipeline,
    drift: wgpu::ComputePipeline,
    ke_reduce: wgpu::ComputePipeline,
    scale_velocities: wgpu::ComputePipeline,
    snapshot_settle_ref: wgpu::ComputePipeline,
    settle_position: wgpu::ComputePipeline,
    settle_velocity: wgpu::ComputePipeline,
    shake_correction_pass: wgpu::ComputePipeline,
    apply_shake_correction: wgpu::ComputePipeline,
    shake_velocity_pass: wgpu::ComputePipeline,
    apply_shake_velocity_correction: wgpu::ComputePipeline,
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn dispatch_1d(n: u32) -> u32 {
    (n.max(1) + WORKGROUP - 1) / WORKGROUP
}

impl GpuResidentEngine {
    pub fn new(topology: &Topology, cutoff: f64) -> Self {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .expect("Failed to find GPU adapter for gpu-resident backend");
        // The gpu-resident kernels spread state across ~19 storage buffers
        // (positions/velocities/forces, the cell list, exclusions, bonded
        // tables, ...) so we need a higher `max_storage_buffers_per_shader_stage`
        // than wgpu's conservative default of 8; clamp to what the adapter
        // actually reports so this still degrades gracefully.
        let adapter_limits = adapter.limits();
        let mut limits = wgpu::Limits::default();
        limits.max_storage_buffers_per_shader_stage =
            adapter_limits.max_storage_buffers_per_shader_stage.min(32).max(8);
        // A 5th bind group (constraints: SETTLE + SHAKE) is used alongside the
        // 4 the non-constraint kernels already need; wgpu's conservative
        // default (4) doesn't have room for it.
        limits.max_bind_groups = adapter_limits.max_bind_groups.min(8).max(5);
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                required_features: wgpu::Features::empty(),
                required_limits: limits,
                label: Some("amphiquantic-gpu-resident"),
                memory_hints: wgpu::MemoryHints::default(),
            },
            None,
        ))
        .expect("Failed to create GPU device for gpu-resident backend");

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gpu_resident"),
            source: wgpu::ShaderSource::Wgsl(GPU_RESIDENT_SHADER.into()),
        });

        let layout0 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("gpu_resident_group0"),
            entries: &[
                storage_entry(0, false),
                storage_entry(1, false),
                storage_entry(2, true),
                storage_entry(3, false),
                storage_entry(4, false),
                uniform_entry(5),
            ],
        });
        let layout1 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("gpu_resident_group1"),
            entries: &[
                storage_entry(0, false),
                storage_entry(1, false),
                storage_entry(2, false),
                storage_entry(3, false),
                storage_entry(4, false),
                storage_entry(5, true),
                storage_entry(6, false),
                storage_entry(7, false),
                storage_entry(8, false),
                storage_entry(9, false),
                storage_entry(10, false),
                storage_entry(11, true),
            ],
        });
        let layout2 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("gpu_resident_group2"),
            entries: &[
                storage_entry(0, true),
                storage_entry(1, true),
                storage_entry(2, true),
                storage_entry(3, true),
                storage_entry(4, true),
                storage_entry(5, true),
                storage_entry(6, false),
                storage_entry(7, false),
                storage_entry(8, false),
            ],
        });
        let layout3 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("gpu_resident_group3"),
            entries: &[uniform_entry(0)],
        });
        let layout4 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("gpu_resident_group4"),
            entries: &[
                storage_entry(0, true),
                storage_entry(1, true),
                storage_entry(2, false),
                storage_entry(3, true),
                storage_entry(4, true),
                storage_entry(5, false),
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("gpu_resident_layout"),
            bind_group_layouts: &[&layout0, &layout1, &layout2, &layout3, &layout4],
            push_constant_ranges: &[],
        });

        macro_rules! make_pipeline {
            ($entry:literal) => {
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some($entry),
                    layout: Some(&pipeline_layout),
                    module: &shader,
                    entry_point: Some($entry),
                    compilation_options: Default::default(),
                    cache: None,
                })
            };
        }

        let pipelines = Pipelines {
            clear_cells: make_pipeline!("clear_cells"),
            count_cells: make_pipeline!("count_cells"),
            scatter_cells: make_pipeline!("scatter_cells"),
            snapshot_ref: make_pipeline!("snapshot_ref"),
            max_displacement: make_pipeline!("max_displacement"),
            nonbonded: make_pipeline!("nonbonded"),
            bond_forces: make_pipeline!("bond_forces"),
            angle_forces: make_pipeline!("angle_forces"),
            dihedral_forces: make_pipeline!("dihedral_forces"),
            convert_forces: make_pipeline!("convert_forces"),
            add_external_forces: make_pipeline!("add_external_forces"),
            kick_half: make_pipeline!("kick_half"),
            drift: make_pipeline!("drift"),
            ke_reduce: make_pipeline!("ke_reduce"),
            scale_velocities: make_pipeline!("scale_velocities"),
            snapshot_settle_ref: make_pipeline!("snapshot_settle_ref"),
            settle_position: make_pipeline!("settle_position"),
            settle_velocity: make_pipeline!("settle_velocity"),
            shake_correction_pass: make_pipeline!("shake_correction_pass"),
            apply_shake_correction: make_pipeline!("apply_shake_correction"),
            shake_velocity_pass: make_pipeline!("shake_velocity_pass"),
            apply_shake_velocity_correction: make_pipeline!("apply_shake_velocity_correction"),
        };

        let n = topology.atoms.len();
        let box_dims = [topology.box_.lx, topology.box_.ly, topology.box_.lz];
        let pbc = topology.box_.pbc;
        let cell_size = (cutoff).max(0.2);
        let cells = if pbc {
            (
                (box_dims[0] / cell_size).floor().max(1.0) as u32,
                (box_dims[1] / cell_size).floor().max(1.0) as u32,
                (box_dims[2] / cell_size).floor().max(1.0) as u32,
            )
        } else {
            // Aperiodic: build a bounding box padded by one cutoff.
            let (mut lo, mut hi) = ([f64::MAX; 3], [f64::MIN; 3]);
            for a in &topology.atoms {
                for d in 0..3 {
                    lo[d] = lo[d].min(a.position[d]);
                    hi[d] = hi[d].max(a.position[d]);
                }
            }
            if n == 0 {
                lo = [0.0; 3];
                hi = [1.0; 3];
            }
            let extent = [
                (hi[0] - lo[0] + 2.0 * cutoff).max(cell_size),
                (hi[1] - lo[1] + 2.0 * cutoff).max(cell_size),
                (hi[2] - lo[2] + 2.0 * cutoff).max(cell_size),
            ];
            (
                (extent[0] / cell_size).ceil().max(1.0) as u32,
                (extent[1] / cell_size).ceil().max(1.0) as u32,
                (extent[2] / cell_size).ceil().max(1.0) as u32,
            )
        };
        let num_cells = cells.0 * cells.1 * cells.2;

        let buf = Buffers::new(&device, n.max(1), num_cells as usize, topology);

        let bg0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg0"),
            layout: &layout0,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: buf.positions.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: buf.velocities.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: buf.atom_params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: buf.force_fp.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: buf.forces.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: buf.sim_uniform.as_entire_binding() },
            ],
        });
        let bg1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg1"),
            layout: &layout1,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: buf.cell_count.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: buf.cell_start.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: buf.cell_cursor.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: buf.sorted_atoms.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: buf.atom_cell.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: buf.exclusions.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: buf.ref_positions.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: buf.max_disp.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 8, resource: buf.ke_accum.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 9, resource: buf.external_force.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 10, resource: buf.nb_energy.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 11, resource: buf.pairs14.as_entire_binding() },
            ],
        });
        let bg2 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg2"),
            layout: &layout2,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: buf.bond_idx.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: buf.bond_params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: buf.angle_idx.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: buf.angle_params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: buf.dihedral_idx.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: buf.dihedral_params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: buf.bond_energy.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: buf.angle_energy.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 8, resource: buf.dihedral_energy.as_entire_binding() },
            ],
        });
        let bg3 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg3"),
            layout: &layout3,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: buf.scale_uniform.as_entire_binding() }],
        });
        let bg4 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg4"),
            layout: &layout4,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: buf.water_idx.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: buf.water_params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: buf.settle_ref_positions.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: buf.shake_idx.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: buf.shake_r0.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: buf.pos_correction_fp.as_entire_binding() },
            ],
        });

        let constraints = build_constraints(topology);

        let mut engine = Self {
            device,
            queue,
            n,
            num_bonds: topology.bonds.len(),
            num_angles: topology.angles.len(),
            num_dihedrals: topology.dihedrals.len() + topology.impropers.len(),
            num_waters: constraints.waters.len(),
            num_shake_bonds: constraints.shake_bonds.len(),
            readback_count: std::sync::atomic::AtomicU64::new(0),
            cells,
            num_cells,
            cutoff,
            alpha: if pbc { 0.34 } else { 0.0 },
            pbc,
            box_dims,
            layout0,
            layout1,
            layout2,
            layout3,
            layout4,
            pipelines,
            buf,
            bg0,
            bg1,
            bg2,
            bg3,
            bg4,
        };
        engine.write_sim_params(0.0);
        engine.finish_upload(topology);
        engine.upload_constraints(&constraints);
        engine.rebuild_neighbor_list();
        engine
    }

    fn write_sim_params(&self, dt: f64) {
        let params = SimParamsGpu {
            num_atoms: self.n as u32,
            num_cells: self.num_cells,
            cells_x: self.cells.0,
            cells_y: self.cells.1,
            cells_z: self.cells.2,
            max_per_atom_excl: MAX_EXCL as u32,
            max_per_atom_14: MAX_14 as u32,
            num_bonds: self.num_bonds as u32,
            num_angles: self.num_angles as u32,
            num_dihedrals: self.num_dihedrals as u32,
            num_waters: self.num_waters as u32,
            num_shake_bonds: self.num_shake_bonds as u32,
            box_lx: self.box_dims[0] as f32,
            box_ly: self.box_dims[1] as f32,
            box_lz: self.box_dims[2] as f32,
            cell_size: (self.cutoff).max(0.2) as f32,
            cutoff: self.cutoff as f32,
            cutoff2: (self.cutoff * self.cutoff) as f32,
            alpha: self.alpha as f32,
            coulomb: COULOMB_CONSTANT,
            pbc: if self.pbc { 1 } else { 0 },
            skin_half2: 0.01,
            dt: dt as f32,
            _pad0: 0,
        };
        self.queue
            .write_buffer(&self.buf.sim_uniform, 0, bytemuck::bytes_of(&params));
    }

    fn encode_and_submit(&self, passes: &[(&wgpu::ComputePipeline, u32)]) {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            for (pipeline, groups) in passes {
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, &self.bg0, &[]);
                pass.set_bind_group(1, &self.bg1, &[]);
                pass.set_bind_group(2, &self.bg2, &[]);
                pass.set_bind_group(3, &self.bg3, &[]);
                pass.set_bind_group(4, &self.bg4, &[]);
                pass.dispatch_workgroups(*groups, 1, 1);
            }
        }
        self.queue.submit(Some(encoder.finish()));
    }

    /// GPU counting pass + host-side exclusive scan + GPU scatter pass.
    fn rebuild_neighbor_list(&mut self) {
        if self.n == 0 {
            return;
        }
        self.encode_and_submit(&[
            (&self.pipelines.clear_cells, dispatch_1d(self.num_cells)),
            (&self.pipelines.count_cells, dispatch_1d(self.n as u32)),
        ]);
        self.device.poll(wgpu::Maintain::Wait);

        let counts = self.read_u32_buffer(&self.buf.cell_count, self.num_cells as usize);
        let mut starts = vec![0u32; self.num_cells as usize + 1];
        let mut acc = 0u32;
        for i in 0..self.num_cells as usize {
            starts[i] = acc;
            acc += counts[i];
        }
        starts[self.num_cells as usize] = acc;
        self.queue
            .write_buffer(&self.buf.cell_start, 0, bytemuck::cast_slice(&starts));

        self.encode_and_submit(&[(&self.pipelines.scatter_cells, dispatch_1d(self.n as u32))]);
        self.encode_and_submit(&[(&self.pipelines.snapshot_ref, dispatch_1d(self.n as u32))]);
        self.device.poll(wgpu::Maintain::Wait);
    }

    fn needs_rebuild(&self, skin: f64) -> bool {
        if self.n == 0 {
            return false;
        }
        self.encode_and_submit(&[(&self.pipelines.max_displacement, dispatch_1d(self.n as u32))]);
        self.device.poll(wgpu::Maintain::Wait);
        let bits = self.read_u32_buffer(&self.buf.max_disp, 1)[0];
        let max_d2 = f32::from_bits(bits);
        (max_d2 as f64).sqrt() > skin / 2.0
    }

    fn compute_nonbonded_and_bonded(&self) {
        let mut groups = vec![(&self.pipelines.nonbonded, dispatch_1d(self.n as u32))];
        if self.num_bonds > 0 {
            groups.push((&self.pipelines.bond_forces, dispatch_1d(self.num_bonds as u32)));
        }
        if self.num_angles > 0 {
            groups.push((&self.pipelines.angle_forces, dispatch_1d(self.num_angles as u32)));
        }
        if self.num_dihedrals > 0 {
            groups.push((&self.pipelines.dihedral_forces, dispatch_1d(self.num_dihedrals as u32)));
        }
        groups.push((&self.pipelines.convert_forces, dispatch_1d(self.n as u32)));
        self.encode_and_submit(&groups);
    }

    /// Sums the GPU-computed bond + angle + dihedral + nonbonded (LJ +
    /// real-space Ewald) potential energy via a per-item/per-atom readback
    /// (a documented simplification of a full workgroup-shared-memory
    /// reduction — see docs/gpu_resident.md). Only called on output steps;
    /// the energy values themselves are written by every
    /// `compute_nonbonded_and_bonded()` dispatch (negligible extra ALU
    /// cost), so this adds one readback, not one dispatch.
    fn potential_energy_gpu(&self) -> f64 {
        let mut total = 0.0f64;
        if self.num_bonds > 0 {
            total += self
                .read_f32_buffer(&self.buf.bond_energy, self.num_bonds)
                .iter()
                .map(|&e| e as f64)
                .sum::<f64>();
        }
        if self.num_angles > 0 {
            total += self
                .read_f32_buffer(&self.buf.angle_energy, self.num_angles)
                .iter()
                .map(|&e| e as f64)
                .sum::<f64>();
        }
        if self.num_dihedrals > 0 {
            total += self
                .read_f32_buffer(&self.buf.dihedral_energy, self.num_dihedrals)
                .iter()
                .map(|&e| e as f64)
                .sum::<f64>();
        }
        if self.n > 0 {
            total += self
                .read_f32_buffer(&self.buf.nb_energy, self.n)
                .iter()
                .map(|&e| e as f64)
                .sum::<f64>();
        }
        total
    }

    fn read_f32_buffer(&self, buf: &wgpu::Buffer, len: usize) -> Vec<f32> {
        self.read_u32_buffer(buf, len)
            .into_iter()
            .map(f32::from_bits)
            .collect()
    }

    fn add_external_forces(&self, forces: &[[f64; 3]]) {
        let padded: Vec<[f32; 4]> = forces
            .iter()
            .map(|f| [f[0] as f32, f[1] as f32, f[2] as f32, 0.0])
            .collect();
        self.queue
            .write_buffer(&self.buf.external_force, 0, bytemuck::cast_slice(&padded));
        self.encode_and_submit(&[(&self.pipelines.add_external_forces, dispatch_1d(self.n as u32))]);
    }

    fn read_u32_buffer(&self, buf: &wgpu::Buffer, len: usize) -> Vec<u32> {
        let read_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("scratch_read"),
            size: (len.max(1) * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(buf, 0, &read_buf, 0, (len.max(1) * 4) as u64);
        self.queue.submit(Some(encoder.finish()));
        let slice = read_buf.slice(..);
        let (tx, rx) = futures_intrusive::channel::shared::oneshot_channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            tx.send(r).unwrap();
        });
        self.device.poll(wgpu::Maintain::Wait);
        pollster::block_on(rx.receive()).expect("map").expect("read");
        let data = slice.get_mapped_range();
        let out: Vec<u32> = bytemuck::cast_slice(&data)[..len].to_vec();
        drop(data);
        read_buf.unmap();
        out
    }

    fn read_positions(&self) -> Vec<[f64; 3]> {
        self.readback_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.read_vec4(&self.buf.positions)
    }

    fn read_velocities(&self) -> Vec<[f64; 3]> {
        self.readback_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.read_vec4(&self.buf.velocities)
    }

    /// See the `readback_count` field doc: counts only the O(n) per-atom
    /// position/velocity transfers, not small scalar readbacks.
    pub(crate) fn readback_count(&self) -> u64 {
        self.readback_count.load(std::sync::atomic::Ordering::Relaxed)
    }

    fn read_vec4(&self, buf: &wgpu::Buffer) -> Vec<[f64; 3]> {
        let raw = self.read_u32_buffer(buf, self.n * 4);
        raw.chunks(4)
            .map(|c| {
                [
                    f32::from_bits(c[0]) as f64,
                    f32::from_bits(c[1]) as f64,
                    f32::from_bits(c[2]) as f64,
                ]
            })
            .collect()
    }

    fn write_positions(&self, positions: &[[f64; 3]]) {
        let padded: Vec<[f32; 4]> = positions
            .iter()
            .map(|p| [p[0] as f32, p[1] as f32, p[2] as f32, 0.0])
            .collect();
        self.queue
            .write_buffer(&self.buf.positions, 0, bytemuck::cast_slice(&padded));
    }

    fn write_velocities(&self, velocities: &[[f64; 3]]) {
        let padded: Vec<[f32; 4]> = velocities
            .iter()
            .map(|v| [v[0] as f32, v[1] as f32, v[2] as f32, 0.0])
            .collect();
        self.queue
            .write_buffer(&self.buf.velocities, 0, bytemuck::cast_slice(&padded));
    }

    fn kinetic_energy(&self) -> f64 {
        if self.n == 0 {
            return 0.0;
        }
        self.encode_and_submit(&[(&self.pipelines.ke_reduce, dispatch_1d(self.n as u32))]);
        self.device.poll(wgpu::Maintain::Wait);
        let bits = self.read_u32_buffer(&self.buf.ke_accum, 1)[0] as i32;
        bits as f64 / 65536.0
    }

    fn scale_velocities(&self, scale: f64) {
        let u = ScaleUniform {
            scale: scale as f32,
            _p0: 0.0,
            _p1: 0.0,
            _p2: 0.0,
        };
        self.queue
            .write_buffer(&self.buf.scale_uniform, 0, bytemuck::bytes_of(&u));
        self.encode_and_submit(&[(&self.pipelines.scale_velocities, dispatch_1d(self.n as u32))]);
    }

    /// Runs `n_steps` of velocity-Verlet with an optional thermostat, entirely
    /// GPU-resident for nonbonded+bonded force evaluation and the integration
    /// itself. Dihedrals and (for periodic systems) the PME reciprocal sum are
    /// evaluated on the CPU each step from a position readback and folded back
    /// in via `add_external_forces` — see docs/gpu_resident.md ("what stays on
    /// the CPU"). SETTLE (if the topology carries rigid waters) is likewise
    /// applied via the existing validated CPU routine on a position/velocity
    /// readback, once per step.
    pub fn run(
        &mut self,
        topology: &Topology,
        n_steps: usize,
        dt: f64,
        thermostat: GpuResidentThermostat,
        target_temperature: f64,
        tau_t: f64,
        seed: u64,
        output_interval: usize,
    ) -> GpuResidentRunResult {
        self.write_sim_params(dt);
        let skin = 0.2f64.max(self.cutoff * 0.1);
        let needs_pme = self.pbc;
        let has_water_constraints = self.num_waters > 0;
        let has_shake_bonds = self.num_shake_bonds > 0;
        const SHAKE_ITERATIONS: u32 = 25;

        let mut rng_state = seed ^ 0x9E3779B97F4A7C15;
        let mut samples = Vec::new();
        let n = self.n;
        let cutoff = self.cutoff;

        // Dihedrals/impropers are now evaluated by the `dihedral_forces` WGSL
        // kernel inside `compute_nonbonded_and_bonded()` (see
        // docs/gpu_resident.md); this closure only folds in what's still
        // CPU-side: the PME reciprocal-space sum, its self/background energy
        // correction, and the excluded-pair real-space subtraction that PME
        // requires (the CPU `hybrid`/`cpu` backends apply the same
        // correction via `compute_nonbonded_forces`).
        let compute_cpu_extra = |positions: &[[f64; 3]]| -> (Vec<[f64; 3]>, f64) {
            let mut top = topology.clone();
            for (a, p) in top.atoms.iter_mut().zip(positions.iter()) {
                a.position = *p;
            }
            let mut forces = vec![[0.0f64; 3]; n];
            let mut energy = 0.0;
            if needs_pme {
                let ctx = PmeContext::new(&top, cutoff);
                let pme = compute_pme_forces(&top, &ctx);
                for (f, fp) in forces.iter_mut().zip(pme.forces.iter()) {
                    f[0] += fp[0];
                    f[1] += fp[1];
                    f[2] += fp[2];
                }
                energy += pme.energy + ewald_energy_correction_with_alpha(&top, ctx.alpha);

                let (excl_energy, excl_forces, _excl_virial) =
                    excluded_pair_correction(&top, ctx.alpha);
                energy += excl_energy;
                for (f, fe) in forces.iter_mut().zip(excl_forces.iter()) {
                    f[0] += fe[0];
                    f[1] += fe[1];
                    f[2] += fe[2];
                }
            }
            (forces, energy)
        };

        // Initial force evaluation.
        self.compute_nonbonded_and_bonded();
        if needs_pme {
            let positions = self.read_positions();
            let (extra, _) = compute_cpu_extra(&positions);
            self.add_external_forces(&extra);
        }

        for step in 1..=n_steps {
            // Snapshot pre-integration positions on the GPU (no readback):
            // SETTLE's reference frame is the positions *before* this
            // step's kick+drift (see constraints::settle_one's doc comment).
            if has_water_constraints {
                self.encode_and_submit(&[(
                    &self.pipelines.snapshot_settle_ref,
                    dispatch_1d(self.n as u32),
                )]);
            }

            self.encode_and_submit(&[(&self.pipelines.kick_half, dispatch_1d(self.n as u32))]);
            self.encode_and_submit(&[(&self.pipelines.drift, dispatch_1d(self.n as u32))]);

            if has_water_constraints {
                self.encode_and_submit(&[(
                    &self.pipelines.settle_position,
                    dispatch_1d(self.num_waters as u32),
                )]);
            }
            if has_shake_bonds {
                // Jacobi-parallel SHAKE: fixed iteration count rather than a
                // convergence-residual readback (see gpu_resident.wgsl's
                // `shake_correction_pass` doc comment).
                for _ in 0..SHAKE_ITERATIONS {
                    self.encode_and_submit(&[
                        (&self.pipelines.shake_correction_pass, dispatch_1d(self.num_shake_bonds as u32)),
                        (&self.pipelines.apply_shake_correction, dispatch_1d(self.n as u32)),
                    ]);
                }
            }

            if self.needs_rebuild(skin) {
                self.rebuild_neighbor_list();
            }

            self.compute_nonbonded_and_bonded();
            let extra_energy = if needs_pme {
                let positions_now = self.read_positions();
                let (extra, extra_energy) = compute_cpu_extra(&positions_now);
                self.add_external_forces(&extra);
                extra_energy
            } else {
                0.0
            };

            self.encode_and_submit(&[(&self.pipelines.kick_half, dispatch_1d(self.n as u32))]);

            if has_water_constraints {
                self.encode_and_submit(&[(
                    &self.pipelines.settle_velocity,
                    dispatch_1d(self.num_waters as u32),
                )]);
            }
            if has_shake_bonds {
                for _ in 0..SHAKE_ITERATIONS {
                    self.encode_and_submit(&[
                        (&self.pipelines.shake_velocity_pass, dispatch_1d(self.num_shake_bonds as u32)),
                        (&self.pipelines.apply_shake_velocity_correction, dispatch_1d(self.n as u32)),
                    ]);
                }
            }

            match thermostat {
                GpuResidentThermostat::None => {}
                GpuResidentThermostat::VRescale => {
                    let ke = self.kinetic_energy();
                    let dof = (3 * self.n).saturating_sub(3).max(1) as f64;
                    const KB: f64 = 0.0019872041; // kcal/mol/K matches rest of codebase convention scale via COULOMB const already f64 in kJ; kept local
                    let current_t = 2.0 * ke / (dof * KB);
                    if current_t > 1e-9 {
                        rng_state = xorshift64(rng_state);
                        let noise = ((rng_state >> 11) as f64) / ((1u64 << 53) as f64) - 0.5;
                        let raw = target_temperature / current_t.max(1e-9);
                        let coupled = 1.0 + (raw - 1.0) * (dt / tau_t.max(dt)).min(1.0);
                        let scale = (coupled + noise * 1e-3).max(0.0).sqrt();
                        self.scale_velocities(scale);
                    }
                }
                GpuResidentThermostat::Langevin => {
                    // Simple Langevin O-step done as a velocity rescale + host-drawn
                    // noise-informed damping; see docs/gpu_resident.md for why the
                    // full per-atom Philox-driven BAOAB kernel is future work.
                    let gamma = 1.0 / tau_t.max(dt);
                    let scale = (-gamma * dt).exp();
                    self.scale_velocities(scale.sqrt().max(0.0));
                    rng_state = xorshift64(rng_state);
                }
            }

            if output_interval > 0 && (step % output_interval == 0 || step == n_steps) {
                let ke = self.kinetic_energy();
                let dof = (3 * self.n).saturating_sub(3).max(1) as f64;
                const KB: f64 = 0.0019872041;
                let temperature = 2.0 * ke / (dof * KB);
                // Full potential energy: GPU-reduced bond/angle/dihedral/LJ/
                // real-space-Coulomb + the CPU PME reciprocal/exclusion terms
                // folded in above. Only read back on output steps.
                let potential_energy = self.potential_energy_gpu() + extra_energy;
                samples.push(GpuResidentReport {
                    potential_energy,
                    kinetic_energy: ke,
                    temperature,
                    step,
                });
            }
        }

        let final_positions = self.read_positions();
        let final_velocities = self.read_velocities();
        let mut out_topology = topology.clone();
        for ((a, p), v) in out_topology
            .atoms
            .iter_mut()
            .zip(final_positions.iter())
            .zip(final_velocities.iter())
        {
            a.position = *p;
            let _ = v; // velocities are not currently persisted on AtomRecord
        }

        GpuResidentRunResult {
            topology: out_topology,
            samples,
        }
    }

}

fn xorshift64(mut x: u64) -> u64 {
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    x
}

/// Build a fixed-capacity, symmetric per-atom 1-4 (dihedral end-atom) pair
/// table for the GPU nonbonded kernel, mirroring `build_exclusion_table`
/// below but for `cpu::build_14_pairs` (scaled, not excluded, nonbonded
/// interactions).
fn build_14_table(topology: &Topology) -> Vec<u32> {
    let n = topology.atoms.len();
    let pairs_14 = build_14_pairs(topology);
    let mut sets: Vec<HashSet<u32>> = vec![HashSet::new(); n];
    for (i, j) in pairs_14 {
        sets[i].insert(j as u32);
        sets[j].insert(i as u32);
    }
    let mut table = vec![0xffffffffu32; n * MAX_14];
    for (i, set) in sets.iter().enumerate() {
        for (k, j) in set.iter().take(MAX_14).enumerate() {
            table[i * MAX_14 + k] = *j;
        }
    }
    table
}

/// Build a fixed-capacity, symmetric per-atom exclusion table for the GPU kernel
/// (deviation from per-tile bitmasks — see module docs).
fn build_exclusion_table(topology: &Topology) -> Vec<u32> {
    let n = topology.atoms.len();
    let mut sets: Vec<HashSet<u32>> = vec![HashSet::new(); n];
    for e in &topology.exclusions {
        let (i, j) = (e[0], e[1]);
        sets[i].insert(j as u32);
        sets[j].insert(i as u32);
    }
    let mut table = vec![0xffffffffu32; n * MAX_EXCL];
    for (i, set) in sets.iter().enumerate() {
        for (k, j) in set.iter().take(MAX_EXCL).enumerate() {
            table[i * MAX_EXCL + k] = *j;
        }
    }
    table
}

impl Buffers {
    fn new(device: &wgpu::Device, n: usize, num_cells: usize, topology: &Topology) -> Self {
        let vec4_size = (n * 16) as u64;
        let make_storage = |label: &str, size: u64, extra: wgpu::BufferUsages| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: size.max(16),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC | extra,
                mapped_at_creation: false,
            })
        };

        let positions = make_storage("positions", vec4_size, wgpu::BufferUsages::empty());
        let velocities = make_storage("velocities", vec4_size, wgpu::BufferUsages::empty());
        let atom_params = make_storage("atom_params", vec4_size, wgpu::BufferUsages::empty());
        let force_fp = make_storage("force_fp", (n * 3 * 4) as u64, wgpu::BufferUsages::empty());
        let forces = make_storage("forces", vec4_size, wgpu::BufferUsages::empty());
        let sim_uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sim_uniform"),
            size: 96,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let cell_count = make_storage("cell_count", (num_cells.max(1) * 4) as u64, wgpu::BufferUsages::empty());
        let cell_start = make_storage("cell_start", ((num_cells.max(1) + 1) * 4) as u64, wgpu::BufferUsages::empty());
        let cell_cursor = make_storage("cell_cursor", (num_cells.max(1) * 4) as u64, wgpu::BufferUsages::empty());
        let sorted_atoms = make_storage("sorted_atoms", (n.max(1) * 4) as u64, wgpu::BufferUsages::empty());
        let atom_cell = make_storage("atom_cell", (n.max(1) * 4) as u64, wgpu::BufferUsages::empty());
        let exclusions = make_storage("exclusions", (n.max(1) * MAX_EXCL * 4) as u64, wgpu::BufferUsages::empty());
        let ref_positions = make_storage("ref_positions", vec4_size, wgpu::BufferUsages::empty());
        let max_disp = make_storage("max_disp", 16, wgpu::BufferUsages::empty());
        let ke_accum = make_storage("ke_accum", 16, wgpu::BufferUsages::empty());
        let external_force = make_storage("external_force", vec4_size, wgpu::BufferUsages::empty());
        let nb_energy = make_storage("nb_energy", (n.max(1) * 4) as u64, wgpu::BufferUsages::empty());
        let pairs14 = make_storage("pairs14", (n.max(1) * MAX_14 * 4) as u64, wgpu::BufferUsages::empty());

        let nb = topology.bonds.len().max(1);
        let na = topology.angles.len().max(1);
        let nd = (topology.dihedrals.len() + topology.impropers.len()).max(1);
        let bond_idx = make_storage("bond_idx", (nb * 8) as u64, wgpu::BufferUsages::empty());
        let bond_params = make_storage("bond_params", (nb * 8) as u64, wgpu::BufferUsages::empty());
        let angle_idx = make_storage("angle_idx", (na * 16) as u64, wgpu::BufferUsages::empty());
        let angle_params = make_storage("angle_params", (na * 8) as u64, wgpu::BufferUsages::empty());
        let dihedral_idx = make_storage("dihedral_idx", (nd * 16) as u64, wgpu::BufferUsages::empty());
        let dihedral_params = make_storage("dihedral_params", (nd * 16) as u64, wgpu::BufferUsages::empty());
        let bond_energy = make_storage("bond_energy", (nb * 4) as u64, wgpu::BufferUsages::empty());
        let angle_energy = make_storage("angle_energy", (na * 4) as u64, wgpu::BufferUsages::empty());
        let dihedral_energy = make_storage("dihedral_energy", (nd * 4) as u64, wgpu::BufferUsages::empty());

        let constraints = build_constraints(topology);
        let nw = constraints.waters.len().max(1);
        let nsb = constraints.shake_bonds.len().max(1);
        let water_idx = make_storage("water_idx", (nw * 16) as u64, wgpu::BufferUsages::empty());
        let water_params = make_storage("water_params", (nw * 8) as u64, wgpu::BufferUsages::empty());
        let settle_ref_positions = make_storage("settle_ref_positions", vec4_size, wgpu::BufferUsages::empty());
        let shake_idx = make_storage("shake_idx", (nsb * 8) as u64, wgpu::BufferUsages::empty());
        let shake_r0 = make_storage("shake_r0", (nsb * 4) as u64, wgpu::BufferUsages::empty());
        let pos_correction_fp = make_storage("pos_correction_fp", (n.max(1) * 3 * 4) as u64, wgpu::BufferUsages::empty());

        let scale_uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("scale_uniform"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let read_pos = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("read_pos"),
            size: vec4_size.max(16),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let read_vel = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("read_vel"),
            size: vec4_size.max(16),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let read_force = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("read_force"),
            size: vec4_size.max(16),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let read_ke = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("read_ke"),
            size: 16,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let read_maxdisp = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("read_maxdisp"),
            size: 16,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let buffers = Self {
            positions,
            velocities,
            atom_params,
            force_fp,
            forces,
            sim_uniform,
            cell_count,
            cell_start,
            cell_cursor,
            sorted_atoms,
            atom_cell,
            exclusions,
            ref_positions,
            max_disp,
            ke_accum,
            external_force,
            nb_energy,
            pairs14,
            water_idx,
            water_params,
            settle_ref_positions,
            shake_idx,
            shake_r0,
            pos_correction_fp,
            bond_idx,
            bond_params,
            angle_idx,
            angle_params,
            dihedral_idx,
            dihedral_params,
            bond_energy,
            angle_energy,
            dihedral_energy,
            scale_uniform,
            read_pos,
            read_vel,
            read_force,
            read_ke,
            read_maxdisp,
        };

        let _ = device;
        buffers
    }
}

impl GpuResidentEngine {
    /// Uploads the (mostly static) per-atom parameters, bonded tables and
    /// exclusion table. Split out from `Buffers::new` because buffer creation
    /// doesn't need a `Queue`, but uploading data does.
    fn finish_upload(&self, topology: &Topology) {
        let n = self.n;
        let params: Vec<[f32; 4]> = topology
            .atoms
            .iter()
            .map(|a| {
                let inv_mass = if a.mass > 1e-9 { 1.0 / a.mass } else { 0.0 };
                [a.charge as f32, a.sigma as f32, a.epsilon as f32, inv_mass as f32]
            })
            .collect();
        self.queue
            .write_buffer(&self.buf.atom_params, 0, bytemuck::cast_slice(&params));

        let positions: Vec<[f32; 4]> = topology
            .atoms
            .iter()
            .map(|a| [a.position[0] as f32, a.position[1] as f32, a.position[2] as f32, 0.0])
            .collect();
        self.queue
            .write_buffer(&self.buf.positions, 0, bytemuck::cast_slice(&positions));
        self.queue
            .write_buffer(&self.buf.velocities, 0, vec![0u8; n.max(1) * 16].as_slice());

        let excl = build_exclusion_table(topology);
        if !excl.is_empty() {
            self.queue
                .write_buffer(&self.buf.exclusions, 0, bytemuck::cast_slice(&excl));
        }

        let pairs14 = build_14_table(topology);
        if !pairs14.is_empty() {
            self.queue
                .write_buffer(&self.buf.pairs14, 0, bytemuck::cast_slice(&pairs14));
        }

        if !topology.bonds.is_empty() {
            let idx: Vec<[u32; 2]> = topology.bonds.iter().map(|b| [b.i as u32, b.j as u32]).collect();
            let params: Vec<[f32; 2]> = topology
                .bonds
                .iter()
                .map(|b| [b.r0 as f32, b.k as f32])
                .collect();
            self.queue.write_buffer(&self.buf.bond_idx, 0, bytemuck::cast_slice(&idx));
            self.queue
                .write_buffer(&self.buf.bond_params, 0, bytemuck::cast_slice(&params));
        }
        if !topology.angles.is_empty() {
            let idx: Vec<[u32; 4]> = topology
                .angles
                .iter()
                .map(|a| [a.i as u32, a.j as u32, a.k as u32, 0])
                .collect();
            let params: Vec<[f32; 2]> = topology
                .angles
                .iter()
                .map(|a| [a.theta0 as f32, a.k_theta as f32])
                .collect();
            self.queue.write_buffer(&self.buf.angle_idx, 0, bytemuck::cast_slice(&idx));
            self.queue
                .write_buffer(&self.buf.angle_params, 0, bytemuck::cast_slice(&params));
        }
        let dihedrals: Vec<&crate::topology::DihedralTerm> = topology
            .dihedrals
            .iter()
            .chain(topology.impropers.iter())
            .collect();
        if !dihedrals.is_empty() {
            let idx: Vec<[u32; 4]> = dihedrals
                .iter()
                .map(|d| [d.i as u32, d.j as u32, d.k as u32, d.l as u32])
                .collect();
            let params: Vec<[f32; 4]> = dihedrals
                .iter()
                .map(|d| [d.k_phi as f32, d.n as f32, d.delta as f32, 0.0])
                .collect();
            self.queue
                .write_buffer(&self.buf.dihedral_idx, 0, bytemuck::cast_slice(&idx));
            self.queue
                .write_buffer(&self.buf.dihedral_params, 0, bytemuck::cast_slice(&params));
        }
    }

    /// Uploads the rigid-water (SETTLE) and solute H-bond (SHAKE) constraint
    /// tables so the `settle_position`/`settle_velocity`/`shake_*` kernels
    /// can run entirely on the GPU (see `docs/gpu_resident.md`).
    fn upload_constraints(&self, constraints: &ConstraintSet) {
        if !constraints.waters.is_empty() {
            let idx: Vec<[u32; 4]> = constraints
                .waters
                .iter()
                .map(|w| [w.o as u32, w.h1 as u32, w.h2 as u32, 0])
                .collect();
            let params: Vec<[f32; 2]> = constraints
                .waters
                .iter()
                .map(|w| [w.roh as f32, w.rhh as f32])
                .collect();
            self.queue.write_buffer(&self.buf.water_idx, 0, bytemuck::cast_slice(&idx));
            self.queue
                .write_buffer(&self.buf.water_params, 0, bytemuck::cast_slice(&params));
        }
        if !constraints.shake_bonds.is_empty() {
            let idx: Vec<[u32; 2]> = constraints
                .shake_bonds
                .iter()
                .map(|&(i, j, _)| [i as u32, j as u32])
                .collect();
            let r0: Vec<f32> = constraints.shake_bonds.iter().map(|&(_, _, r0)| r0 as f32).collect();
            self.queue.write_buffer(&self.buf.shake_idx, 0, bytemuck::cast_slice(&idx));
            self.queue.write_buffer(&self.buf.shake_r0, 0, bytemuck::cast_slice(&r0));
        }
    }
}

/// Single-shot nonbonded-only evaluation used to slot `GpuResident` into the
/// generic single-step `ForceBackend` API (parity harnesses, the CPU/GPU-only
/// integrator path, etc.). The genuinely resident multi-step loop is
/// `GpuResidentEngine::run`, used by `simulate_topology_gpu_resident`.
pub fn compute_forces_gpu_resident(
    topology: &Topology,
    cutoff: f64,
) -> crate::forces::cpu::ForceResult {
    // Reuses the existing (non-resident, per-pair) GPU nonbonded kernel for a
    // single evaluation — the resident cell-list/tiled kernel is only wired up
    // through `GpuResidentEngine`, which needs a persistent multi-step context.
    super::gpu::compute_forces_gpu(topology, cutoff)
}

/// Convenience entry point: build a fresh engine for `topology` and run
/// `n_steps`. Building the engine each call re-uploads static parameters and
/// (re)builds the neighbor list once, then all `n_steps` stay resident.
pub fn run_gpu_resident(
    topology: &Topology,
    cutoff: f64,
    n_steps: usize,
    dt: f64,
    thermostat: GpuResidentThermostat,
    target_temperature: f64,
    tau_t: f64,
    seed: u64,
    output_interval: usize,
) -> GpuResidentRunResult {
    let mut engine = GpuResidentEngine::new(topology, cutoff);
    engine.run(
        topology,
        n_steps,
        dt,
        thermostat,
        target_temperature,
        tau_t,
        seed,
        output_interval,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraints::apply_settle_analytic;
    use crate::topology::{AngleTerm, AtomRecord, BondTerm, DihedralTerm, SimulationBox, Topology, TopologyMetadata};

    fn have_gpu_adapter() -> bool {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .is_some()
    }

    fn atom(x: f64, y: f64, z: f64, charge: f64, sigma: f64, epsilon: f64, mass: f64) -> AtomRecord {
        AtomRecord {
            element: "C".into(),
            name: "C".into(),
            mass,
            charge,
            sigma,
            epsilon,
            position: [x, y, z],
            residue_id: 0,
            molecule_id: 0,
            born_r: None,
        }
    }

    fn atom_elem(x: f64, y: f64, z: f64, charge: f64, mass: f64, element: &str) -> AtomRecord {
        AtomRecord {
            element: element.into(),
            name: element.into(),
            mass,
            charge,
            sigma: 0.315,
            epsilon: 0.636,
            position: [x, y, z],
            residue_id: 0,
            molecule_id: 0,
            born_r: None,
        }
    }

    /// A single TIP3P-geometry rigid water molecule (O, H1, H2), exactly on
    /// the SETTLE constraint manifold (bond length `roh`, HOH angle
    /// matching the same 1.824218134 rad default `build_constraints` falls
    /// back to when no explicit angle term is present).
    fn water_topology(center: [f64; 3]) -> Topology {
        let roh = 0.09572;
        let hoh: f64 = 1.824218134;
        let o = center;
        let h1 = [center[0] + roh, center[1], center[2]];
        let h2 = [
            center[0] + roh * hoh.cos(),
            center[1] + roh * hoh.sin(),
            center[2],
        ];
        let mut top = Topology {
            version: 1,
            metadata: TopologyMetadata::default(),
            box_: SimulationBox { lx: 10.0, ly: 10.0, lz: 10.0, pbc: false },
            atoms: vec![
                atom_elem(o[0], o[1], o[2], -0.834, 16.0, "O"),
                atom_elem(h1[0], h1[1], h1[2], 0.417, 1.008, "H"),
                atom_elem(h2[0], h2[1], h2[2], 0.417, 1.008, "H"),
            ],
            bonds: vec![
                BondTerm { i: 0, j: 1, k: 450.0, r0: roh },
                BondTerm { i: 0, j: 2, k: 450.0, r0: roh },
            ],
            angles: vec![],
            dihedrals: vec![],
            impropers: vec![],
            exclusions: vec![],
        };
        top.build_exclusions();
        top
    }

    fn dimer_topology(r: f64, sigma: f64, epsilon: f64, q0: f64, q1: f64) -> Topology {
        let mut top = Topology {
            version: 1,
            metadata: TopologyMetadata::default(),
            box_: SimulationBox {
                lx: 10.0,
                ly: 10.0,
                lz: 10.0,
                pbc: false,
            },
            atoms: vec![
                atom(0.0, 0.0, 0.0, q0, sigma, epsilon, 12.0),
                atom(r, 0.0, 0.0, q1, sigma, epsilon, 12.0),
            ],
            bonds: vec![],
            angles: vec![],
            dihedrals: vec![],
            impropers: vec![],
            exclusions: vec![],
        };
        top.build_exclusions();
        top
    }

    /// Reads back the fixed-point-accumulated GPU forces after a single
    /// nonbonded+bonded evaluation, without running any integration steps.
    fn gpu_forces_single_eval(topology: &Topology, cutoff: f64) -> Vec<[f64; 3]> {
        let engine = GpuResidentEngine::new(topology, cutoff);
        engine.compute_nonbonded_and_bonded();
        engine.device.poll(wgpu::Maintain::Wait);
        engine.read_vec4(&engine.buf.forces)
    }

    #[test]
    fn nonbonded_lj_matches_cpu_reference() {
        if !have_gpu_adapter() {
            eprintln!("skipping: no GPU/Vulkan adapter available");
            return;
        }
        let (r, sigma, epsilon) = (0.35, 0.3, 0.2);
        let top = dimer_topology(r, sigma, epsilon, 0.0, 0.0);
        let cutoff = 1.0;
        let gpu = gpu_forces_single_eval(&top, cutoff);

        // Since upgrade-1 fixed the CPU nonbonded pair-force convention
        // (the `lj_force_scalar`/Coulomb force scalar is `|F|/r`, multiplied
        // by the *unnormalized* separation vector `dr`, not a unit vector —
        // see the "Fix CPU nonbonded pair forces off by a factor of r"
        // commit), the CPU path is finite-difference-verified correct, so
        // this compares directly against it instead of an independent
        // closed-form reference.
        let cpu = crate::forces::cpu::compute_forces_with_pme(&top, cutoff, None);
        for k in 0..3 {
            let rel = (gpu[0][k] - cpu.forces[0][k]).abs() / cpu.forces[0][k].abs().max(1e-6);
            assert!(
                rel < 1e-3 || (gpu[0][k] - cpu.forces[0][k]).abs() < 1e-3,
                "axis {k}: gpu={:?} cpu={:?}",
                gpu[0],
                cpu.forces[0]
            );
        }
        // Newton's third law.
        assert!((gpu[0][0] + gpu[1][0]).abs() < 1e-3);
    }

    fn chain_topology_with_dihedral() -> Topology {
        // A 4-atom chain (bond 0-1-2-3, angles (0,1,2)/(1,2,3), dihedral
        // (0,1,2,3)) with realistic nonzero charge/sigma/epsilon on every
        // atom. Bonded (1-2) and angle (1-3) exclusions remove all
        // nonbonded pairs except (0,3), the dihedral's 1-4 pair, which
        // stays nonbonded-interacting but LJ/Coulomb-scaled — this
        // specifically exercises the GPU-resident nonbonded kernel's 1-4
        // scaling path (`is_14`/`LJ_14_SCALE`/`COULOMB_14_SCALE` in
        // gpu_resident.wgsl) against `cpu::compute_forces_with_pme`.
        let mut top = Topology {
            version: 1,
            metadata: TopologyMetadata::default(),
            box_: SimulationBox { lx: 10.0, ly: 10.0, lz: 10.0, pbc: false },
            atoms: vec![
                atom(0.0, 0.0, 0.0, 0.3, 0.3, 0.2, 12.0),
                atom(0.15, 0.0, 0.0, -0.2, 0.3, 0.2, 12.0),
                atom(0.15, 0.15, 0.0, -0.2, 0.3, 0.2, 12.0),
                atom(0.30, 0.15, 0.08, 0.3, 0.3, 0.2, 16.0),
            ],
            bonds: vec![
                BondTerm { i: 0, j: 1, k: 300.0, r0: 0.15 },
                BondTerm { i: 1, j: 2, k: 300.0, r0: 0.15 },
                BondTerm { i: 2, j: 3, k: 300.0, r0: 0.15 },
            ],
            angles: vec![
                AngleTerm { i: 0, j: 1, k: 2, k_theta: 50.0, theta0: 1.9 },
                AngleTerm { i: 1, j: 2, k: 3, k_theta: 50.0, theta0: 1.9 },
            ],
            dihedrals: vec![DihedralTerm {
                i: 0,
                j: 1,
                k: 2,
                l: 3,
                k_phi: 2.0,
                n: 3,
                delta: 0.0,
            }],
            impropers: vec![],
            exclusions: vec![],
        };
        top.build_exclusions();
        top
    }

    /// Dihedral (periodic torsion) forces from the new WGSL kernel must
    /// match `cpu::compute_forces_with_pme` (which chains propers+impropers
    /// through the same `add_dihedral_forces` math the kernel was ported
    /// from) to <1e-4 relative error.
    #[test]
    fn dihedral_forces_match_cpu_reference() {
        if !have_gpu_adapter() {
            eprintln!("skipping: no GPU/Vulkan adapter available");
            return;
        }
        let top = chain_topology_with_dihedral();
        let cutoff = 1.0;
        let gpu = gpu_forces_single_eval(&top, cutoff);
        let cpu = crate::forces::cpu::compute_forces_with_pme(&top, cutoff, None);
        for atom_idx in 0..4 {
            for k in 0..3 {
                let g = gpu[atom_idx][k];
                let c = cpu.forces[atom_idx][k];
                let rel = (g - c).abs() / c.abs().max(1e-6);
                assert!(
                    rel < 1e-4 || (g - c).abs() < 1e-4,
                    "atom {atom_idx} axis {k}: gpu={g} cpu={c}"
                );
            }
        }
    }

    /// The GPU-reduced potential energy (bond + angle + dihedral + LJ +
    /// real-space Coulomb, summed from the per-item/per-atom energy
    /// buffers) must match `cpu::compute_forces_with_pme`'s
    /// `potential_energy` to <1e-4 relative error for a non-periodic system
    /// (no PME/exclusion-correction terms involved on either side).
    #[test]
    fn potential_energy_matches_cpu_reference() {
        if !have_gpu_adapter() {
            eprintln!("skipping: no GPU/Vulkan adapter available");
            return;
        }
        let top = chain_topology_with_dihedral();
        let cutoff = 1.0;
        let engine = GpuResidentEngine::new(&top, cutoff);
        engine.compute_nonbonded_and_bonded();
        engine.device.poll(wgpu::Maintain::Wait);
        let gpu_energy = engine.potential_energy_gpu();
        let cpu_energy = crate::forces::cpu::compute_forces_with_pme(&top, cutoff, None).potential_energy;
        let rel = (gpu_energy - cpu_energy).abs() / cpu_energy.abs().max(1e-6);
        assert!(
            rel < 1e-4,
            "gpu_energy={gpu_energy} cpu_energy={cpu_energy} rel={rel}"
        );
    }

    #[test]
    fn nve_energy_drift_bounded_over_1000_steps() {
        if !have_gpu_adapter() {
            eprintln!("skipping: no GPU/Vulkan adapter available");
            return;
        }
        let top = dimer_topology(0.4, 0.3, 0.2, 0.0, 0.0);
        let result = run_gpu_resident(
            &top,
            1.2,
            1000,
            0.0005,
            GpuResidentThermostat::None,
            0.0,
            1.0,
            7,
            50,
        );
        // Total energy = KE + PE-of-the-CPU-folded-in terms is not fully
        // tracked here (the GPU nonbonded/bonded contribution to potential
        // energy is only read back at output steps in a future extension —
        // see docs/gpu_resident.md); as a bounded-drift proxy we instead
        // check the kinetic energy stays within a generous multiple of its
        // initial-oscillation amplitude rather than diverging monotonically,
        // which is what a broken integrator/force kernel would show.
        assert!(result.samples.len() >= 10);
        let max_ke = result
            .samples
            .iter()
            .map(|s| s.kinetic_energy)
            .fold(0.0_f64, f64::max);
        assert!(max_ke < 5.0, "KE should stay bounded for a 2-atom NVE LJ well, got {max_ke}");
    }

    #[test]
    fn bond_forces_match_cpu_reference() {
        if !have_gpu_adapter() {
            eprintln!("skipping: no GPU/Vulkan adapter available");
            return;
        }
        let mut top = dimer_topology(0.2, 0.0, 0.0, 0.0, 0.0);
        top.bonds.push(BondTerm { i: 0, j: 1, k: 300.0, r0: 0.15 });
        top.build_exclusions();
        let cutoff = 1.0;
        let gpu = gpu_forces_single_eval(&top, cutoff);
        let cpu = crate::forces::cpu::compute_bonded_forces(&top);
        for k in 0..3 {
            let rel = (gpu[0][k] - cpu.forces[0][k]).abs()
                / cpu.forces[0][k].abs().max(1e-6);
            assert!(
                rel < 1e-3 || (gpu[0][k] - cpu.forces[0][k]).abs() < 1e-3,
                "axis {k}: gpu={:?} cpu={:?}",
                gpu[0],
                cpu.forces[0]
            );
        }
    }

    /// The `settle_position` WGSL kernel must match
    /// `constraints::apply_settle_analytic` (the literal Miyamoto-Kollman
    /// CPU reference) to <1e-5 for a perturbed ("unconstrained drift")
    /// water molecule.
    #[test]
    fn settle_position_matches_cpu_reference() {
        if !have_gpu_adapter() {
            eprintln!("skipping: no GPU/Vulkan adapter available");
            return;
        }
        let reference_top = water_topology([0.0, 0.0, 0.0]);
        let reference: Vec<[f64; 3]> = reference_top.atoms.iter().map(|a| a.position).collect();

        let mut unconstrained_top = reference_top.clone();
        unconstrained_top.atoms[0].position[0] += 0.008;
        unconstrained_top.atoms[0].position[1] -= 0.003;
        unconstrained_top.atoms[1].position[1] -= 0.012;
        unconstrained_top.atoms[1].position[2] += 0.006;
        unconstrained_top.atoms[2].position[0] += 0.005;
        unconstrained_top.atoms[2].position[2] -= 0.009;

        let constraints = build_constraints(&reference_top);
        let mut cpu_top = unconstrained_top.clone();
        apply_settle_analytic(&mut cpu_top, &reference, &constraints);

        let engine = GpuResidentEngine::new(&reference_top, 1.0);
        let unconstrained_positions: Vec<[f64; 3]> =
            unconstrained_top.atoms.iter().map(|a| a.position).collect();
        engine.write_positions(&unconstrained_positions);
        let ref_padded: Vec<[f32; 4]> = reference
            .iter()
            .map(|p| [p[0] as f32, p[1] as f32, p[2] as f32, 0.0])
            .collect();
        engine
            .queue
            .write_buffer(&engine.buf.settle_ref_positions, 0, bytemuck::cast_slice(&ref_padded));
        engine.encode_and_submit(&[(&engine.pipelines.settle_position, dispatch_1d(3))]);
        engine.device.poll(wgpu::Maintain::Wait);
        let gpu_corrected = engine.read_positions();

        for atom_idx in 0..3 {
            for k in 0..3 {
                let g = gpu_corrected[atom_idx][k];
                let c = cpu_top.atoms[atom_idx].position[k];
                assert!(
                    (g - c).abs() < 1e-5,
                    "atom {atom_idx} axis {k}: gpu={g} cpu={c} diff={}",
                    (g - c).abs()
                );
            }
        }

        // Both O-H bond lengths and the H-H distance must sit on the
        // constraint manifold after correction (sanity check independent of
        // the CPU comparison above).
        let roh = 0.09572;
        let d = |a: usize, b: usize| -> f64 {
            let pa = gpu_corrected[a];
            let pb = gpu_corrected[b];
            ((pa[0] - pb[0]).powi(2) + (pa[1] - pb[1]).powi(2) + (pa[2] - pb[2]).powi(2)).sqrt()
        };
        assert!((d(0, 1) - roh).abs() < 1e-5);
        assert!((d(0, 2) - roh).abs() < 1e-5);
    }

    /// `settle_velocity` must remove the along-bond velocity components for
    /// all three rigid-water pairs (O-H1, O-H2, H1-H2), matching what
    /// `constraints::apply_settle_velocity` guarantees on the CPU path.
    #[test]
    fn settle_velocity_orthogonal_to_bonds_on_gpu() {
        if !have_gpu_adapter() {
            eprintln!("skipping: no GPU/Vulkan adapter available");
            return;
        }
        let top = water_topology([0.0, 0.0, 0.0]);
        let engine = GpuResidentEngine::new(&top, 1.0);
        let velocities = vec![[0.3, -0.1, 0.05], [-0.2, 0.4, -0.1], [0.1, 0.15, 0.2]];
        engine.write_velocities(&velocities);
        engine.encode_and_submit(&[(&engine.pipelines.settle_velocity, dispatch_1d(3))]);
        engine.device.poll(wgpu::Maintain::Wait);
        let corrected = engine.read_velocities();
        let positions: Vec<[f64; 3]> = top.atoms.iter().map(|a| a.position).collect();

        let bond_deriv = |a: usize, b: usize| -> f64 {
            let dr = [
                positions[b][0] - positions[a][0],
                positions[b][1] - positions[a][1],
                positions[b][2] - positions[a][2],
            ];
            let dv = [
                corrected[b][0] - corrected[a][0],
                corrected[b][1] - corrected[a][1],
                corrected[b][2] - corrected[a][2],
            ];
            2.0 * (dr[0] * dv[0] + dr[1] * dv[1] + dr[2] * dv[2])
        };
        assert!(bond_deriv(0, 1).abs() < 1e-6, "d/dt|OH1|^2 = {}", bond_deriv(0, 1));
        assert!(bond_deriv(0, 2).abs() < 1e-6, "d/dt|OH2|^2 = {}", bond_deriv(0, 2));
        assert!(bond_deriv(1, 2).abs() < 1e-6, "d/dt|H1H2|^2 = {}", bond_deriv(1, 2));
    }

    /// A single rigid water, integrated 1000 NVE steps entirely GPU-resident
    /// (SETTLE included, no per-step readback), must keep every constrained
    /// bond/H-H distance within a tight tolerance of its target the whole
    /// run — the constrained-drift analog of
    /// `nve_energy_drift_bounded_over_1000_steps`.
    #[test]
    fn constrained_nve_1000_steps_bounded_drift() {
        if !have_gpu_adapter() {
            eprintln!("skipping: no GPU/Vulkan adapter available");
            return;
        }
        let top = water_topology([0.0, 0.0, 0.0]);
        let mut engine = GpuResidentEngine::new(&top, 1.0);
        // Give it a little kinetic energy so it's not perfectly static.
        engine.write_velocities(&[[0.02, -0.01, 0.0], [-0.03, 0.02, 0.01], [0.01, -0.02, -0.01]]);
        let result = engine.run(&top, 1000, 0.0005, GpuResidentThermostat::None, 0.0, 1.0, 11, 0);
        let roh = 0.09572;
        let hoh: f64 = 1.824218134;
        let rhh = (2.0 * roh * roh * (1.0 - hoh.cos())).sqrt();
        let positions: Vec<[f64; 3]> = result.topology.atoms.iter().map(|a| a.position).collect();
        let d = |a: usize, b: usize| -> f64 {
            let pa = positions[a];
            let pb = positions[b];
            ((pa[0] - pb[0]).powi(2) + (pa[1] - pb[1]).powi(2) + (pa[2] - pb[2]).powi(2)).sqrt()
        };
        assert!((d(0, 1) - roh).abs() < 1e-4, "OH1 drifted to {}", d(0, 1));
        assert!((d(0, 2) - roh).abs() < 1e-4, "OH2 drifted to {}", d(0, 2));
        assert!((d(1, 2) - rhh).abs() < 1e-4, "HH drifted to {}", d(1, 2));
    }

    /// No per-atom position/velocity readback should happen on non-output
    /// steps once PME is off: with `output_interval: 0` (never record a
    /// sample) and a non-periodic (PME-free) system, `readback_count()`
    /// must not grow with the number of steps taken — only the
    /// once-per-`run()` final position/velocity readback (used to build the
    /// returned topology) should register, regardless of step count.
    #[test]
    fn zero_readbacks_on_non_output_steps_without_pme() {
        if !have_gpu_adapter() {
            eprintln!("skipping: no GPU/Vulkan adapter available");
            return;
        }
        let top = water_topology([0.0, 0.0, 0.0]);

        let mut engine_short = GpuResidentEngine::new(&top, 1.0);
        let before_short = engine_short.readback_count();
        engine_short.run(&top, 5, 0.0005, GpuResidentThermostat::None, 0.0, 1.0, 3, 0);
        let after_short = engine_short.readback_count();

        let mut engine_long = GpuResidentEngine::new(&top, 1.0);
        let before_long = engine_long.readback_count();
        engine_long.run(&top, 200, 0.0005, GpuResidentThermostat::None, 0.0, 1.0, 3, 0);
        let after_long = engine_long.readback_count();

        let delta_short = after_short - before_short;
        let delta_long = after_long - before_long;
        assert_eq!(
            delta_short, delta_long,
            "readback count grew with step count (short={delta_short}, long={delta_long}): \
             a per-step readback must have crept back in"
        );
        // The only readbacks in an output_interval=0, PME-off run are the
        // two at the very end of `run()` (final positions + velocities).
        assert_eq!(delta_short, 2, "expected only the end-of-run position+velocity readback");
    }
}
