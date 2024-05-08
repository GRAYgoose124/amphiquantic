struct Params {
    step_size: f32,
    max_steps: u32,
};

@group(0) @binding(0) var<storage, read_write> coords: array<vec3<f32>>;
@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    if (index >= arrayLength(&coords)) {
        return;
    }

    var step: u32 = 0u;
    while (step < params.max_steps) {
        // Dummy logic
        coords[index] = coords[index] + vec3<f32>(params.step_size, params.step_size, params.step_size);
        step = step + 1u;
    }
}
