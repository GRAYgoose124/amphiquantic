use crate::forces::cpu::{build_14_pairs, ForceResult, COULOMB_14_SCALE, LJ_14_SCALE};
use crate::neighbor::build_neighbor_list;
use crate::topology::Topology;
use bytemuck::{Pod, Zeroable};
use pollster;
use std::sync::Mutex;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuAtomParams {
    charge: f32,
    sigma: f32,
    epsilon: f32,
    _pad: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuSimParams {
    num_atoms: u32,
    num_pairs: u32,
    cutoff: f32,
    box_lx: f32,
    box_ly: f32,
    box_lz: f32,
    pbc: u32,
    use_screened: u32,
    alpha: f32,
}

const FORCE_SHADER: &str = include_str!("../../../data/shaders/forces.wgsl");
const COULOMB_CONSTANT: f32 = 138.935456;

pub struct GpuForceEngine {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    state: Mutex<GpuBufferState>,
}

struct GpuBufferState {
    max_atoms: usize,
    max_pairs: usize,
    coord_buf: wgpu::Buffer,
    param_buf: wgpu::Buffer,
    pair_buf: wgpu::Buffer,
    pair_scale_buf: wgpu::Buffer,
    force_buf: wgpu::Buffer,
    read_buf: wgpu::Buffer,
    sim_buf: wgpu::Buffer,
}

impl GpuForceEngine {
    pub fn new() -> Self {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .expect("Failed to find GPU adapter");
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                label: Some("amphiquantic-gpu"),
                memory_hints: wgpu::MemoryHints::default(),
            },
            None,
        ))
        .expect("Failed to create GPU device");

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("forces"),
            source: wgpu::ShaderSource::Wgsl(FORCE_SHADER.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                storage_entry(0, false),
                storage_entry(1, true),
                storage_entry(2, true),
                storage_entry(3, true),
                uniform_entry(4),
                storage_entry(5, true),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let state = GpuBufferState::new(&device, 256, 4096);

        Self {
            device,
            queue,
            pipeline,
            bind_group_layout,
            state: Mutex::new(state),
        }
    }

    pub fn compute_nonbonded(
        &self,
        topology: &Topology,
        cutoff: f64,
        use_screened: bool,
        alpha: f64,
    ) -> ForceResult {
        let n = topology.atoms.len();
        if n == 0 {
            return ForceResult {
                forces: vec![],
                potential_energy: 0.0,
            };
        }

        let nl = build_neighbor_list(topology, cutoff);
        let coords_f32: Vec<[f32; 4]> = topology
            .atoms
            .iter()
            .map(|a| [a.position[0] as f32, a.position[1] as f32, a.position[2] as f32, 0.0])
            .collect();
        let params: Vec<GpuAtomParams> = topology
            .atoms
            .iter()
            .map(|a| GpuAtomParams {
                charge: a.charge as f32,
                sigma: a.sigma as f32,
                epsilon: a.epsilon as f32,
                _pad: 0.0,
            })
            .collect();
        let mut pair_indices: Vec<u32> = nl
            .pairs
            .iter()
            .flat_map(|(i, j)| [*i as u32, *j as u32])
            .collect();
        if pair_indices.is_empty() {
            pair_indices = vec![0, 0];
        }
        let num_pairs = nl.pairs.len() as u32;

        // Dihedral 1-4 pairs get scaled LJ/Coulomb instead of full strength
        // (matches cpu::compute_nonbonded_forces's LJ_14_SCALE/COULOMB_14_SCALE
        // convention). Every other pair gets (1.0, 1.0).
        let pairs_14 = build_14_pairs(topology);
        let mut pair_scale: Vec<[f32; 2]> = nl
            .pairs
            .iter()
            .map(|&(i, j)| {
                if pairs_14.contains(&(i.min(j), i.max(j))) {
                    [LJ_14_SCALE as f32, COULOMB_14_SCALE as f32]
                } else {
                    [1.0f32, 1.0f32]
                }
            })
            .collect();
        if pair_scale.is_empty() {
            pair_scale = vec![[1.0, 1.0]];
        }

        let sim_params = GpuSimParams {
            num_atoms: n as u32,
            num_pairs,
            cutoff: cutoff as f32,
            box_lx: topology.box_.lx as f32,
            box_ly: topology.box_.ly as f32,
            box_lz: topology.box_.lz as f32,
            pbc: if topology.box_.pbc { 1 } else { 0 },
            use_screened: if use_screened { 1 } else { 0 },
            alpha: alpha as f32,
        };

        let mut state = self.state.lock().unwrap();
        state.ensure_capacity(&self.device, n, pair_indices.len() / 2);

        self.queue
            .write_buffer(&state.coord_buf, 0, bytemuck::cast_slice(&coords_f32));
        self.queue
            .write_buffer(&state.param_buf, 0, bytemuck::cast_slice(&params));
        self.queue
            .write_buffer(&state.pair_buf, 0, bytemuck::cast_slice(&pair_indices));
        self.queue
            .write_buffer(&state.pair_scale_buf, 0, bytemuck::cast_slice(&pair_scale));
        self.queue
            .write_buffer(&state.sim_buf, 0, bytemuck::bytes_of(&sim_params));

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: state.force_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: state.coord_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: state.param_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: state.pair_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: state.sim_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: state.pair_scale_buf.as_entire_binding(),
                },
            ],
            label: None,
        });

        let mut encoder =
            self.device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(((n as u32) + 63) / 64, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&state.force_buf, 0, &state.read_buf, 0, (n * 16) as u64);
        self.queue.submit(Some(encoder.finish()));

        let slice = state.read_buf.slice(..);
        let (tx, rx) = futures_intrusive::channel::shared::oneshot_channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            tx.send(r).unwrap();
        });
        self.device.poll(wgpu::Maintain::Wait);
        pollster::block_on(rx.receive()).expect("map").expect("read");

        let data = slice.get_mapped_range();
        let raw: &[[f32; 4]] = bytemuck::cast_slice(&data);
        let forces: Vec<[f64; 3]> = raw
            .iter()
            .map(|f| [f[0] as f64, f[1] as f64, f[2] as f64])
            .collect();
        drop(data);
        state.read_buf.unmap();

        let mut potential_energy = 0.0f64;
        for &(i, j) in &nl.pairs {
            let pi = topology.atoms[i].position;
            let pj = topology.atoms[j].position;
            let qi = topology.atoms[i].charge;
            let qj = topology.atoms[j].charge;
            let sigma = 0.5 * (topology.atoms[i].sigma + topology.atoms[j].sigma);
            let mut epsilon = (topology.atoms[i].epsilon * topology.atoms[j].epsilon).sqrt();
            let mut coulomb_scale = 1.0;
            if pairs_14.contains(&(i.min(j), i.max(j))) {
                epsilon *= LJ_14_SCALE;
                coulomb_scale = COULOMB_14_SCALE;
            }
            let mut dr = [pj[0] - pi[0], pj[1] - pi[1], pj[2] - pi[2]];
            dr = crate::neighbor::minimum_image(dr, &topology.box_);
            let r2 = dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2];
            let r = r2.sqrt().max(1e-12);
            if sigma > 1e-8 && epsilon > 1e-12 {
                let sr = sigma / r;
                let sr6 = sr.powi(6);
                let sr12 = sr6 * sr6;
                potential_energy += 4.0 * epsilon * (sr12 - sr6);
            }
            if use_screened {
                let (e, _) = crate::electrostatics::ewald::screened_coulomb_energy_force(
                    qi, qj, r, r2, alpha, coulomb_scale,
                );
                potential_energy += e;
            } else {
                potential_energy += COULOMB_CONSTANT as f64 * coulomb_scale * qi * qj / r;
            }
        }

        ForceResult {
            forces,
            potential_energy,
        }
    }
}

impl GpuBufferState {
    fn new(device: &wgpu::Device, max_atoms: usize, max_pairs: usize) -> Self {
        let coord_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("coords"),
            size: (max_atoms * 16) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let param_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("params"),
            size: (max_atoms * 16) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let pair_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pairs"),
            size: (max_pairs.max(1) * 2 * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let pair_scale_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pair_scale"),
            size: (max_pairs.max(1) * 2 * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let force_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("forces"),
            size: (max_atoms * 16) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let read_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("forces_read"),
            size: (max_atoms * 16) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let sim_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sim"),
            size: 48,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            max_atoms,
            max_pairs,
            coord_buf,
            param_buf,
            pair_buf,
            pair_scale_buf,
            force_buf,
            read_buf,
            sim_buf,
        }
    }

    fn ensure_capacity(&mut self, device: &wgpu::Device, n_atoms: usize, n_pairs: usize) {
        if n_atoms <= self.max_atoms && n_pairs <= self.max_pairs {
            return;
        }
        let max_atoms = n_atoms.next_power_of_two().max(256);
        let max_pairs = n_pairs.next_power_of_two().max(4096);
        *self = Self::new(device, max_atoms, max_pairs);
    }
}

pub fn compute_forces_gpu(topology: &Topology, cutoff: f64) -> ForceResult {
    static ENGINE: std::sync::OnceLock<GpuForceEngine> = std::sync::OnceLock::new();
    let screened = topology.box_.pbc;
    ENGINE
        .get_or_init(GpuForceEngine::new)
        .compute_nonbonded(topology, cutoff, screened, 0.34)
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
