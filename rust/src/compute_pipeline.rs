use pyo3::prelude::*;
use wgpu::util::DeviceExt;
use pollster;
use bytemuck;
use bytemuck::{Pod, Zeroable};

use crate::utilities::shader::MINIMIZE_SHADER;
use crate::utilities::shader::SIMULATE_SHADER;
use crate::utilities::shader::RELAX_SHADER;

#[pyclass]
#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct AtomPipelineParams {
    pub step_size: f32,
    pub max_steps: u32,
    pub process_type: u32, // 0 for relaxation, 1 for minimization, 2 for simulation
}



pub fn run_atom_pipeline(
    coords: &Vec<[f64; 3]>,
    atom_types: &Vec<String>,
    bonds: &Vec<(usize, usize)>,
    params: AtomPipelineParams,
) -> Vec<[f64; 3]> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        dx12_shader_compiler: wgpu::Dx12Compiler::Fxc,
        flags: wgpu::InstanceFlags::empty(),
        gles_minor_version: wgpu::Gles3MinorVersion::Automatic,
    });

    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    })).expect("Failed to find an appropriate adapter");

    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::default(),
        label: None,
    }, None)).expect("Failed to create device");

    // pick shader
    let shader = match params.process_type {
        0 => device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(RELAX_SHADER.as_str().into()),
        }),
        1 => device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(MINIMIZE_SHADER.as_str().into()),
        }),
        2 => device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(SIMULATE_SHADER.as_str().into()),
        }),
        _ => panic!("Invalid process type"),
    };

    let coord_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Coordinate Buffer"),
        contents: bytemuck::cast_slice(&coords),
        usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::STORAGE,
    });

    let result_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Result Buffer"),
        size: (coords.len() * std::mem::size_of::<[f64; 3]>()) as wgpu::BufferAddress,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let bonds_bytes: Vec<u8> = bonds.iter().flat_map(|(a, b)| vec![*a as u8, *b as u8]).collect();
    let bonds_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Bonds Buffer"),
        contents: &bonds_bytes,
        usage: wgpu::BufferUsages::STORAGE,
    });

    let atom_types_bytes: Vec<u8> = atom_types.iter().flat_map(|s| s.bytes()).collect();
    let atom_types_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Atom Types Buffer"),
        contents: &atom_types_bytes,
        usage: wgpu::BufferUsages::STORAGE,
    });

    let param_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Parameter Buffer"),
        contents: bytemuck::cast_slice(&[params]),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
        label: None,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: coord_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: param_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: atom_types_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: bonds_buffer.as_entire_binding(),
            },
        ],
        label: None,
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: "main",
        compilation_options: Default::default(),
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        cpass.set_pipeline(&compute_pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.dispatch_workgroups((coords.len() as u32 + 63) / 64, 1, 1);
    }

    encoder.copy_buffer_to_buffer(&coord_buffer, 0, &result_buffer, 0, (coords.len() * std::mem::size_of::<[f64; 3]>()) as wgpu::BufferAddress);

    queue.submit(Some(encoder.finish()));

    // Read back the data
    let buffer_slice = result_buffer.slice(..);
    let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
        sender.send(result).unwrap();
    });

    device.poll(wgpu::Maintain::Wait);

    if let Some(Ok(())) = pollster::block_on(receiver.receive()) {
        let data = buffer_slice.get_mapped_range();
        let result: Vec<[f64; 3]> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        result_buffer.unmap();
        result
    } else {
        panic!("Failed to read result buffer");
    }
}
