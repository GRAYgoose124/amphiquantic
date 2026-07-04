struct SimulationParams {
    num_atoms: u32,
    num_pairs: u32,
    cutoff: f32,
    _pad: f32,
}

struct AtomParams {
    charge: f32,
    sigma: f32,
    epsilon: f32,
    _pad: f32,
}

@group(0) @binding(0) var<storage, read_write> forces: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> coords: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> atom_params: array<AtomParams>;
@group(0) @binding(3) var<storage, read> pairs: array<u32>;
@group(0) @binding(4) var<uniform> sim: SimulationParams;

const COULOMB: f32 = 138.935456;

fn lj_force_energy(pos_i: vec3<f32>, pos_j: vec3<f32>, sigma: f32, epsilon: f32) -> vec4<f32> {
    let dr = pos_j - pos_i;
    let r2 = dot(dr, dr);
    let r = max(sqrt(r2), 1e-12);
    let sr = sigma / r;
    let sr2 = sr * sr;
    let sr6 = sr2 * sr2 * sr2;
    let sr12 = sr6 * sr6;
    let force_scalar = 24.0 * epsilon * (2.0 * sr12 - sr6) / r;
    let f = force_scalar * dr;
    let energy = 4.0 * epsilon * (sr12 - sr6);
    return vec4<f32>(f, energy);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let atom = gid.x;
    if (atom >= sim.num_atoms) {
        return;
    }

    var total_force = vec3<f32>(0.0);
    var pos_i = coords[atom].xyz;
    let pi = atom_params[atom];

    var p: u32 = 0u;
    while (p < sim.num_pairs) {
        let i = pairs[p * 2u];
        let j = pairs[p * 2u + 1u];
        if (i == atom || j == atom) {
            let other = select(j, i, i == atom);
            let pj = atom_params[other];
            let sigma = 0.5 * (pi.sigma + pj.sigma);
            let epsilon = sqrt(pi.epsilon * pj.epsilon);
            let pos_j = coords[other].xyz;
            var dr = pos_j - pos_i;
            if (i == atom) {
                dr = -dr;
            }
            let r2 = max(dot(dr, dr), 1e-12);
            let r = sqrt(r2);
            let sr = sigma / r;
            let sr6 = pow(sr, 6.0);
            let sr12 = sr6 * sr6;
            let lj = 24.0 * epsilon * (2.0 * sr12 - sr6) / r;
            let coulomb = COULOMB * pi.charge * pj.charge / r2;
            let sign = select(-1.0, 1.0, i == atom);
            total_force += sign * (lj + coulomb) * normalize(dr);
        }
        p = p + 1u;
    }

    forces[atom] = vec4<f32>(total_force, 0.0);
}
