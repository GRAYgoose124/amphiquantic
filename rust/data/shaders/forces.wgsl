struct SimulationParams {
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
// Per-pair (lj_scale, coulomb_scale): (1,1) normally, (0.5, 1/1.2) for a
// dihedral 1-4 pair — matches cpu::compute_nonbonded_forces's
// LJ_14_SCALE/COULOMB_14_SCALE convention exactly.
@group(0) @binding(5) var<storage, read> pair_scale: array<vec2<f32>>;

const COULOMB: f32 = 138.935456;

fn erfc_approx(x: f32) -> f32 {
    let t = 1.0 / (1.0 + 0.5 * abs(x));
    let tau = t * exp(-x * x - 1.26551223
        + t * (1.00002368
        + t * (0.37409196
        + t * (0.09678418
        + t * (-0.18628806
        + t * (0.27886807
        + t * (-1.13520398
        + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))));
    return select(2.0 - tau, tau, x >= 0.0);
}

fn mic(dr: vec3<f32>) -> vec3<f32> {
    if (sim.pbc == 0u) {
        return dr;
    }
    var out = dr;
    if (out.x > 0.5 * sim.box_lx) { out.x -= sim.box_lx; }
    if (out.x < -0.5 * sim.box_lx) { out.x += sim.box_lx; }
    if (out.y > 0.5 * sim.box_ly) { out.y -= sim.box_ly; }
    if (out.y < -0.5 * sim.box_ly) { out.y += sim.box_ly; }
    if (out.z > 0.5 * sim.box_lz) { out.z -= sim.box_lz; }
    if (out.z < -0.5 * sim.box_lz) { out.z += sim.box_lz; }
    return out;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let atom = gid.x;
    if (atom >= sim.num_atoms) {
        return;
    }

    var total_force = vec3<f32>(0.0);
    let pos_i = coords[atom].xyz;
    let pi = atom_params[atom];

    var p: u32 = 0u;
    while (p < sim.num_pairs) {
        let i = pairs[p * 2u];
        let j = pairs[p * 2u + 1u];
        if (i == atom || j == atom) {
            let other = select(j, i, i == atom);
            let pj = atom_params[other];
            let scale = pair_scale[p];
            let sigma = 0.5 * (pi.sigma + pj.sigma);
            let epsilon = sqrt(pi.epsilon * pj.epsilon) * scale.x;
            var dr = coords[other].xyz - pos_i;
            dr = mic(dr);
            if (i == atom) {
                dr = -dr;
            }
            let r2 = max(dot(dr, dr), 1e-12);
            let r = sqrt(r2);
            let sr = sigma / r;
            let sr6 = pow(sr, 6.0);
            let sr12 = sr6 * sr6;
            let lj = 24.0 * epsilon * (2.0 * sr12 - sr6) / r;
            var coulomb = COULOMB * scale.y * pi.charge * pj.charge / r2;
            if (sim.use_screened == 1u) {
                let arg = sim.alpha * r;
                let erfc_val = erfc_approx(arg);
                coulomb = COULOMB * scale.y * pi.charge * pj.charge * (
                    erfc_val / r2 + 2.0 * sim.alpha * exp(-arg * arg) / (sqrt(3.14159265) * r)
                );
            }
            let sign = select(-1.0, 1.0, i == atom);
            total_force += sign * (lj + coulomb) * normalize(dr);
        }
        p = p + 1u;
    }

    forces[atom] = vec4<f32>(total_force, 0.0);
}
