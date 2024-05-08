use wgpu::util::DeviceExt;
use pollster;
use bytemuck;
use bytemuck::{Pod, Zeroable};

/// Parameters required for energy minimization
#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct MinimizationParams {
    pub step_size: f32,
    pub max_steps: u32,
}





/// Perform energy minimization using a GPU
pub fn minimize_energy(
    coords: &mut Vec<(f64, f64, f64)>,
    params: MinimizationParams,
) {
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default())).unwrap();
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default(), None)).unwrap();

    let shader = include_str!("../data/gpu/minimize.wgsl");

    // Load the shader
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Minimization Shader"),
        source: wgpu::ShaderSource::Wgsl(shader.into()),
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Pipeline Layout"),
        bind_group_layouts: &[],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Compute Pipeline"),
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: "main",
        compilation_options: Default::default(),
    });

    // Prepare data for GPU
    let coords_data: Vec<f32> = coords.iter().flat_map(|(x, y, z)| vec![*x as f32, *y as f32, *z as f32]).collect();
    let buffer_size = (coords_data.len() * std::mem::size_of::<f32>()) as u64;

    let coords_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Coords Buffer"),
        contents: bytemuck::cast_slice(&coords_data),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
    });

    let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Params Buffer"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Bind Group Layout"),
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
        ],
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Bind Group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: coords_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: params_buffer.as_entire_binding(),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Command Encoder"),
    });

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Compute Pass"),
            timestamp_writes: None
        });
        compute_pass.set_pipeline(&pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        compute_pass.dispatch_workgroups((coords.len() as u32 + 63) / 64, 1, 1);
    }

    queue.submit(Some(encoder.finish()));

    // Read back data
    let buffer_slice = coords_buffer.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |result| sender.send(result).unwrap());
    device.poll(wgpu::Maintain::Wait);

    if receiver.recv().unwrap().is_ok() {
        let data = buffer_slice.get_mapped_range();
        let updated_coords: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        coords.clear();
        for chunk in updated_coords.chunks(3) {
            coords.push((chunk[0] as f64, chunk[1] as f64, chunk[2] as f64));
        }
    }
    coords_buffer.unmap();
}
