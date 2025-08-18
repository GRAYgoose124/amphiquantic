struct SimulationParams {
    step_size: f32;
    max_steps: u32;
    process_type: u32;
};


@group(0) @binding(0) var<storage, read_write> coords: array<vec3<f32>>;
@group(0) @binding(1) var<uniform> params: SimulationParams;
@group(0) @binding(2) var<storage, read> atom_types: array<u32>;
@group(0) @binding(3) var<storage, read> bond_indices: array<u32>;

fn lennard_jones_force(pos1: vec3<f32>, pos2: vec3<f32>) -> vec3<f32> {
    let r = distance(pos1, pos2);
    let r_inv = 1.0 / r;
    let r_inv6 = r_inv * r_inv * r_inv * r_inv * r_inv * r_inv;
    let r_inv12 = r_inv6 * r_inv6;
    let force_scalar = 24.0 * (2.0 * r_inv12 - r_inv6) * r_inv;
    return force_scalar * (pos1 - pos2);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    if (index >= arrayLength(&coords)) {
        return;
    }

    var step: u32 = 0u;
    while (step < params.max_steps) {
        var force: vec3<f32> = vec3<f32>(0.0, 0.0, 0.0);

        for (var i: u32 = 0u; i < arrayLength(&coords); i = i + 1u) {
            if (i != index) {
                force = force + lennard_jones_force(coords[index], coords[i]);
            }
        }

        coords[index] = coords[index] + force * params.step_size;
        step = step + 1u;
    }
}
