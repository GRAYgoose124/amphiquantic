use crate::forces::cpu::ForceResult;
use crate::neighbor::build_neighbor_list;
use crate::topology::Topology;
use bytemuck::{Pod, Zeroable};
use pollster;
use wgpu::util::DeviceExt;

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
    _pad: f32,
}

const FORCE_SHADER: &str = include_str!("../../../data/shaders/forces.wgsl");

pub fn compute_forces_gpu(topology: &Topology, cutoff: f64) -> ForceResult {
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
    let mut pair_indices: Vec<u32> = nl.pairs.iter().flat_map(|(i, j)| [*i as u32, *j as u32]).collect();
    if pair_indices.is_empty() {
        // wgpu rejects zero-length storage buffers; shader skips pairs when num_pairs == 0
        pair_indices = vec![0, 0];
    }
    let num_pairs = nl.pairs.len() as u32;

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
            label: None,
            memory_hints: wgpu::MemoryHints::default(),
        },
        None,
    ))
    .expect("Failed to create GPU device");

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("forces"),
        source: wgpu::ShaderSource::Wgsl(FORCE_SHADER.into()),
    });

    let coord_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("coords"),
        contents: bytemuck::cast_slice(&coords_f32),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let param_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("params"),
        contents: bytemuck::cast_slice(&params),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let pair_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("pairs"),
        contents: bytemuck::cast_slice(&pair_indices),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let force_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("forces"),
        size: (n * std::mem::size_of::<[f32; 4]>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let read_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("forces_read"),
        size: (n * std::mem::size_of::<[f32; 4]>()) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let sim_params = GpuSimParams {
        num_atoms: n as u32,
        num_pairs,
        cutoff: cutoff as f32,
        _pad: 0.0,
    };
    let sim_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("sim"),
        contents: bytemuck::bytes_of(&sim_params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            storage_entry(0, false),
            storage_entry(1, true),
            storage_entry(2, true),
            storage_entry(3, true),
            uniform_entry(4),
        ],
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: force_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: coord_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: param_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: pair_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: sim_buf.as_entire_binding(),
            },
        ],
        label: None,
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&layout],
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

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(((n as u32) + 63) / 64, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&force_buf, 0, &read_buf, 0, (n * 16) as u64);
    queue.submit(Some(encoder.finish()));

    let slice = read_buf.slice(..);
    let (tx, rx) = futures_intrusive::channel::shared::oneshot_channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        tx.send(r).unwrap();
    });
    device.poll(wgpu::Maintain::Wait);
    pollster::block_on(rx.receive()).expect("map").expect("read");

    let data = slice.get_mapped_range();
    let raw: &[[f32; 4]] = bytemuck::cast_slice(&data);
    let forces: Vec<[f64; 3]> = raw
        .iter()
        .map(|f| [f[0] as f64, f[1] as f64, f[2] as f64])
        .collect();
    drop(data);
    read_buf.unmap();

    let cpu = crate::forces::cpu::compute_forces(topology, cutoff);
    ForceResult {
        forces,
        potential_energy: cpu.potential_energy,
    }
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
