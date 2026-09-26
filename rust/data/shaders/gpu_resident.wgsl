// GPU-resident MD step kernels.
//
// Design notes (see docs/gpu_resident.md for the full writeup):
//  - Forces are accumulated in fixed-point (i32, FP_SCALE below) via atomicAdd so that
//    bonded and nonbonded kernels can both contribute to the same atom without races
//    and so accumulation order does not change results (determinism, OpenMM-style
//    "mixed precision" substitute for true double-precision atomics).
//  - The cell list uses a GPU counting pass + GPU scatter pass; the small exclusive
//    prefix sum over per-cell counts is done host-side (the cell count is orders of
//    magnitude smaller than the atom count, so this is not a bottleneck) — this is a
//    documented simplification of a fully GPU-side prefix scan.
//  - Exclusions are carried as a small fixed-capacity per-atom list (MAX_EXCL) rather
//    than a full per-tile bitmask; documented deviation from the tile-bitmask design
//    for scope reasons.

struct SimParams {
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
    pme_grid_x: u32,
    pme_grid_y: u32,
    pme_grid_z: u32,
    pme_order: u32,
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

const FP_SCALE: f32 = 65536.0;
const MAX_EXCL: u32 = 8u;
const MAX_14: u32 = 8u;
const LJ_14_SCALE: f32 = 0.5;
const COULOMB_14_SCALE: f32 = 1.0 / 1.2;

@group(0) @binding(0) var<storage, read_write> positions: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> velocities: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> atom_params: array<vec4<f32>>; // charge, sigma, epsilon, inv_mass
@group(0) @binding(3) var<storage, read_write> force_fp: array<atomic<i32>>; // 3*num_atoms
@group(0) @binding(4) var<storage, read_write> forces: array<vec4<f32>>;
@group(0) @binding(5) var<uniform> sim: SimParams;

@group(1) @binding(0) var<storage, read_write> cell_count: array<atomic<u32>>;
@group(1) @binding(1) var<storage, read_write> cell_start: array<u32>;
@group(1) @binding(2) var<storage, read_write> cell_cursor: array<atomic<u32>>;
@group(1) @binding(3) var<storage, read_write> sorted_atoms: array<u32>;
@group(1) @binding(4) var<storage, read_write> atom_cell: array<u32>;
@group(1) @binding(5) var<storage, read> exclusions: array<u32>; // MAX_EXCL per atom, 0xffffffff = empty
@group(1) @binding(6) var<storage, read_write> ref_positions: array<vec4<f32>>;
@group(1) @binding(7) var<storage, read_write> max_disp: array<atomic<u32>>; // bitcast f32, 1 element
@group(1) @binding(8) var<storage, read_write> ke_accum: array<atomic<i32>>; // fixed point, 1 element
@group(1) @binding(9) var<storage, read_write> external_force: array<vec4<f32>>; // CPU-side PME contribution
@group(1) @binding(10) var<storage, read_write> nb_energy: array<f32>; // per-atom, 0.5*sum(pair energy) so summing over atoms gives the total
@group(1) @binding(11) var<storage, read> pairs14: array<u32>; // MAX_14 per atom, 0xffffffff = empty; dihedral (i,l) 1-4 partners

@group(2) @binding(0) var<storage, read> bond_idx: array<vec2<u32>>;
@group(2) @binding(1) var<storage, read> bond_params: array<vec2<f32>>; // r0, k
@group(2) @binding(2) var<storage, read> angle_idx: array<vec4<u32>>; // i, j, k, pad
@group(2) @binding(3) var<storage, read> angle_params: array<vec2<f32>>; // theta0(rad), k
@group(2) @binding(4) var<storage, read> dihedral_idx: array<vec4<u32>>; // i, j, k, l
@group(2) @binding(5) var<storage, read> dihedral_params: array<vec4<f32>>; // k_phi, n, delta, pad
@group(2) @binding(6) var<storage, read_write> bond_energy: array<f32>;
@group(2) @binding(7) var<storage, read_write> angle_energy: array<f32>;
@group(2) @binding(8) var<storage, read_write> dihedral_energy: array<f32>;

// ---- constraints (group 4): SETTLE (rigid water) + SHAKE (solute H-bonds) ----
@group(4) @binding(0) var<storage, read> water_idx: array<vec4<u32>>; // o, h1, h2, pad
@group(4) @binding(1) var<storage, read> water_params: array<vec2<f32>>; // roh, rhh
@group(4) @binding(2) var<storage, read_write> settle_ref_positions: array<vec4<f32>>; // snapshotted before kick+drift
@group(4) @binding(3) var<storage, read> shake_idx: array<vec2<u32>>; // i, j
@group(4) @binding(4) var<storage, read> shake_r0: array<f32>;
@group(4) @binding(5) var<storage, read_write> pos_correction_fp: array<atomic<i32>>; // 3*num_atoms, fixed point

// ---- PME (group 5): GPU B-spline charge spreading + force gather; the
// forward FFT / influence-function multiply / inverse FFT happens on the
// CPU (electrostatics::pme::pme_recip_from_grid) between a grid download
// and a grid upload — never a per-atom position readback. See
// docs/gpu_resident.md "PME: GPU spreading + CPU FFT". ----
@group(5) @binding(0) var<storage, read_write> q_grid_fp: array<atomic<i32>>; // nx*ny*nz, fixed point charge density
@group(5) @binding(1) var<storage, read> pme_potential_grid: array<f32>; // nx*ny*nz, 2*Re(theta_q) after inverse FFT
@group(5) @binding(2) var<storage, read_write> excl_energy: array<f32>; // per-atom, 0.5*sum(excluded/1-4 correction energy)
@group(5) @binding(3) var<storage, read_write> philox_debug_out: array<vec4<u32>>; // 1 element, KAT-test output

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

fn mic(dr_in: vec3<f32>) -> vec3<f32> {
    var dr = dr_in;
    if (sim.pbc == 0u) {
        return dr;
    }
    if (dr.x > 0.5 * sim.box_lx) { dr.x -= sim.box_lx; }
    if (dr.x < -0.5 * sim.box_lx) { dr.x += sim.box_lx; }
    if (dr.y > 0.5 * sim.box_ly) { dr.y -= sim.box_ly; }
    if (dr.y < -0.5 * sim.box_ly) { dr.y += sim.box_ly; }
    if (dr.z > 0.5 * sim.box_lz) { dr.z -= sim.box_lz; }
    if (dr.z < -0.5 * sim.box_lz) { dr.z += sim.box_lz; }
    return dr;
}

fn wrap_pos(p_in: vec3<f32>) -> vec3<f32> {
    var p = p_in;
    if (sim.pbc == 0u) {
        return p;
    }
    p.x = p.x - sim.box_lx * floor(p.x / sim.box_lx);
    p.y = p.y - sim.box_ly * floor(p.y / sim.box_ly);
    p.z = p.z - sim.box_lz * floor(p.z / sim.box_lz);
    return p;
}

fn cell_of(p: vec3<f32>) -> vec3<i32> {
    let w = wrap_pos(p);
    var cx = i32(floor(w.x / sim.cell_size));
    var cy = i32(floor(w.y / sim.cell_size));
    var cz = i32(floor(w.z / sim.cell_size));
    cx = clamp(cx, 0, i32(sim.cells_x) - 1);
    cy = clamp(cy, 0, i32(sim.cells_y) - 1);
    cz = clamp(cz, 0, i32(sim.cells_z) - 1);
    return vec3<i32>(cx, cy, cz);
}

fn cell_index(c: vec3<i32>) -> u32 {
    return u32(c.x) + u32(sim.cells_x) * (u32(c.y) + u32(sim.cells_y) * u32(c.z));
}

fn add_force_fp(atom: u32, f: vec3<f32>) {
    atomicAdd(&force_fp[atom * 3u + 0u], i32(f.x * FP_SCALE));
    atomicAdd(&force_fp[atom * 3u + 1u], i32(f.y * FP_SCALE));
    atomicAdd(&force_fp[atom * 3u + 2u], i32(f.z * FP_SCALE));
}

// ---- cell list build ----

@compute @workgroup_size(64)
fn clear_cells(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= sim.num_cells) { return; }
    atomicStore(&cell_count[idx], 0u);
    atomicStore(&cell_cursor[idx], 0u);
}

@compute @workgroup_size(64)
fn count_cells(@builtin(global_invocation_id) gid: vec3<u32>) {
    let atom = gid.x;
    if (atom >= sim.num_atoms) { return; }
    let c = cell_index(cell_of(positions[atom].xyz));
    atom_cell[atom] = c;
    atomicAdd(&cell_count[c], 1u);
}

// cell_start is written host-side after reading back cell_count (exclusive scan).
@compute @workgroup_size(64)
fn scatter_cells(@builtin(global_invocation_id) gid: vec3<u32>) {
    let atom = gid.x;
    if (atom >= sim.num_atoms) { return; }
    let c = atom_cell[atom];
    let slot = cell_start[c] + atomicAdd(&cell_cursor[c], 1u);
    sorted_atoms[slot] = atom;
}

@compute @workgroup_size(64)
fn snapshot_ref(@builtin(global_invocation_id) gid: vec3<u32>) {
    let atom = gid.x;
    if (atom >= sim.num_atoms) { return; }
    ref_positions[atom] = positions[atom];
    if (atom == 0u) {
        atomicStore(&max_disp[0], 0u);
    }
}

@compute @workgroup_size(64)
fn max_displacement(@builtin(global_invocation_id) gid: vec3<u32>) {
    let atom = gid.x;
    if (atom >= sim.num_atoms) { return; }
    let d = mic(positions[atom].xyz - ref_positions[atom].xyz);
    let d2 = dot(d, d);
    atomicMax(&max_disp[0], bitcast<u32>(d2));
}

// ---- nonbonded (tiled by cell neighbor stencil) ----

fn is_excluded(i: u32, j: u32) -> bool {
    let base = i * MAX_EXCL;
    for (var k = 0u; k < sim.max_per_atom_excl; k = k + 1u) {
        let e = exclusions[base + k];
        if (e == j) { return true; }
        if (e == 0xffffffffu) { return false; }
    }
    return false;
}

fn is_14(i: u32, j: u32) -> bool {
    let base = i * MAX_14;
    for (var k = 0u; k < sim.max_per_atom_14; k = k + 1u) {
        let e = pairs14[base + k];
        if (e == j) { return true; }
        if (e == 0xffffffffu) { return false; }
    }
    return false;
}

@compute @workgroup_size(64)
fn nonbonded(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= sim.num_atoms) { return; }
    let pi = positions[i].xyz;
    let qi = atom_params[i].x;
    let sigi = atom_params[i].y;
    let epsi = atom_params[i].z;
    let ci = cell_of(pi);

    var facc = vec3<f32>(0.0, 0.0, 0.0);
    var eacc = 0.0;

    for (var dz = -1; dz <= 1; dz = dz + 1) {
        for (var dy = -1; dy <= 1; dy = dy + 1) {
            for (var dx = -1; dx <= 1; dx = dx + 1) {
                var nx = ci.x + dx;
                var ny = ci.y + dy;
                var nz = ci.z + dz;
                if (sim.pbc != 0u) {
                    nx = (nx + i32(sim.cells_x)) % i32(sim.cells_x);
                    ny = (ny + i32(sim.cells_y)) % i32(sim.cells_y);
                    nz = (nz + i32(sim.cells_z)) % i32(sim.cells_z);
                } else {
                    if (nx < 0 || ny < 0 || nz < 0 || nx >= i32(sim.cells_x) || ny >= i32(sim.cells_y) || nz >= i32(sim.cells_z)) {
                        continue;
                    }
                }
                let nc = cell_index(vec3<i32>(nx, ny, nz));
                let start = cell_start[nc];
                let end = cell_start[nc + 1u];
                for (var s = start; s < end; s = s + 1u) {
                    let j = sorted_atoms[s];
                    if (j == i) { continue; }
                    if (is_excluded(i, j)) { continue; }
                    let dr = mic(positions[j].xyz - pi);
                    let r2 = dot(dr, dr);
                    if (r2 >= sim.cutoff2 || r2 < 1e-12) { continue; }
                    let r = sqrt(r2);
                    let qj = atom_params[j].x;
                    let sigj = atom_params[j].y;
                    let epsj = atom_params[j].z;

                    // Dihedral 1-4 pairs are nonbonded-interacting but scaled
                    // (not excluded) — matches cpu::compute_nonbonded_forces's
                    // LJ_14_SCALE/COULOMB_14_SCALE convention exactly.
                    var lj_scale = 1.0;
                    var coul_scale = 1.0;
                    if (is_14(i, j)) {
                        lj_scale = LJ_14_SCALE;
                        coul_scale = COULOMB_14_SCALE;
                    }

                    var fscalar = 0.0;
                    if (sigi > 1e-8 && epsj > 1e-12 && epsi > 1e-12) {
                        let sigma = 0.5 * (sigi + sigj);
                        let epsilon = sqrt(epsi * epsj) * lj_scale;
                        let sr = sigma / r;
                        let sr6 = pow(sr, 6.0);
                        let sr12 = sr6 * sr6;
                        fscalar += 24.0 * epsilon * (2.0 * sr12 - sr6) / r2;
                        eacc += 0.5 * 4.0 * epsilon * (sr12 - sr6);
                    }
                    if (sim.alpha > 0.0) {
                        let ar = sim.alpha * r;
                        let erfc_v = erfc_approx(ar);
                        let expfac = exp(-ar * ar);
                        let qq = sim.coulomb * coul_scale * qi * qj;
                        let e_deriv = qq * (erfc_v / r + (2.0 * sim.alpha / sqrt(3.14159265) ) * expfac) / r2;
                        fscalar += e_deriv;
                        eacc += 0.5 * qq * erfc_v / r;
                    } else {
                        let qq = sim.coulomb * coul_scale * qi * qj;
                        fscalar += qq / (r2 * r);
                        eacc += 0.5 * qq / r;
                    }
                    facc -= fscalar * dr;
                }
            }
        }
    }
    add_force_fp(i, facc);
    nb_energy[i] = eacc;
}

// ---- bonded ----

@compute @workgroup_size(64)
fn bond_forces(@builtin(global_invocation_id) gid: vec3<u32>) {
    let b = gid.x;
    if (b >= sim.num_bonds) { return; }
    let idx = bond_idx[b];
    let p = bond_params[b];
    let r0 = p.x;
    let k = p.y;
    let dr = mic(positions[idx.y].xyz - positions[idx.x].xyz);
    let r = max(length(dr), 1e-8);
    let dr_mag = r - r0;
    let fscalar = -2.0 * k * dr_mag / r;
    let f = fscalar * dr;
    add_force_fp(idx.x, -f);
    add_force_fp(idx.y, f);
    bond_energy[b] = k * dr_mag * dr_mag;
}

@compute @workgroup_size(64)
fn angle_forces(@builtin(global_invocation_id) gid: vec3<u32>) {
    let a = gid.x;
    if (a >= sim.num_angles) { return; }
    let idx = angle_idx[a];
    let p = angle_params[a];
    let theta0 = p.x;
    let k = p.y;
    let rij = mic(positions[idx.x].xyz - positions[idx.y].xyz);
    let rkj = mic(positions[idx.z].xyz - positions[idx.y].xyz);
    let lij = max(length(rij), 1e-8);
    let lkj = max(length(rkj), 1e-8);
    var cos_t = dot(rij, rkj) / (lij * lkj);
    cos_t = clamp(cos_t, -1.0, 1.0);
    let theta = acos(cos_t);
    let sin_t = max(sqrt(1.0 - cos_t * cos_t), 1e-6);
    let dtheta = theta - theta0;
    let dvdt = 2.0 * k * dtheta;
    let coef = -dvdt / sin_t;

    let fi = coef * (rkj / (lij * lkj) - cos_t * rij / (lij * lij));
    let fk = coef * (rij / (lij * lkj) - cos_t * rkj / (lkj * lkj));
    let fj = -(fi + fk);
    add_force_fp(idx.x, fi);
    add_force_fp(idx.y, fj);
    add_force_fp(idx.z, fk);
    angle_energy[a] = k * dtheta * dtheta;
}

// ---- dihedrals (proper + improper periodic torsions; the CPU reference,
// `cpu::add_dihedral_forces`, chains propers and impropers into one list and
// uses the same functional form for both, so a single kernel below covers
// both, exactly matching the CPU math term-for-term). ----

@compute @workgroup_size(64)
fn dihedral_forces(@builtin(global_invocation_id) gid: vec3<u32>) {
    let d = gid.x;
    if (d >= sim.num_dihedrals) { return; }
    let idx = dihedral_idx[d];
    let p = dihedral_params[d];
    let k_phi = p.x;
    let n = p.y;
    let delta = p.z;

    let pi = positions[idx.x].xyz;
    let pj = positions[idx.y].xyz;
    let pk = positions[idx.z].xyz;
    let pl = positions[idx.w].xyz;

    let b1 = pj - pi;
    let b2 = pk - pj;
    let b3 = pl - pk;

    let n2 = cross(b1, b2);
    let n3 = cross(b2, b3);
    let len_b2 = max(length(b2), 1e-12);
    let m1 = cross(n2, b2 / len_b2);

    let x = dot(n2, n3);
    let y = dot(m1, n3);
    let phi = atan2(y, x);
    let angle_term = n * phi - delta;
    let e = k_phi * (1.0 + cos(angle_term));
    dihedral_energy[d] = e;

    let d_e_d_phi = k_phi * n * sin(angle_term);
    let inv_n2 = 1.0 / max(length(n2), 1e-12);
    let inv_n3 = 1.0 / max(length(n3), 1e-12);

    let f_i = n2 * (-d_e_d_phi * len_b2 * inv_n2);
    let f_l = n3 * (d_e_d_phi * len_b2 * inv_n3);
    let b2dot = max(dot(b2, b2), 1e-12);
    let f_j = f_i * (-1.0 + dot(b3, b2) / b2dot);
    let f_k = f_l * (-1.0 + dot(b1, b2) / b2dot);

    add_force_fp(idx.x, f_i);
    add_force_fp(idx.y, f_j);
    add_force_fp(idx.z, f_k);
    add_force_fp(idx.w, f_l);
}

// ---- convert / clear ----

@compute @workgroup_size(64)
fn convert_forces(@builtin(global_invocation_id) gid: vec3<u32>) {
    let atom = gid.x;
    if (atom >= sim.num_atoms) { return; }
    let fx = f32(atomicLoad(&force_fp[atom * 3u + 0u])) / FP_SCALE;
    let fy = f32(atomicLoad(&force_fp[atom * 3u + 1u])) / FP_SCALE;
    let fz = f32(atomicLoad(&force_fp[atom * 3u + 2u])) / FP_SCALE;
    forces[atom] = vec4<f32>(fx, fy, fz, 0.0);
    atomicStore(&force_fp[atom * 3u + 0u], 0);
    atomicStore(&force_fp[atom * 3u + 1u], 0);
    atomicStore(&force_fp[atom * 3u + 2u], 0);
}

// add an externally-computed force contribution (e.g. CPU dihedrals / PME reciprocal)
// straight into the f32 force buffer.
@compute @workgroup_size(64)
fn add_external_forces(@builtin(global_invocation_id) gid: vec3<u32>) {
    let atom = gid.x;
    if (atom >= sim.num_atoms) { return; }
    forces[atom] = forces[atom] + external_force[atom];
}

// ---- integration (velocity Verlet, split kick/drift) ----

@compute @workgroup_size(64)
fn kick_half(@builtin(global_invocation_id) gid: vec3<u32>) {
    let atom = gid.x;
    if (atom >= sim.num_atoms) { return; }
    let inv_m = atom_params[atom].w;
    velocities[atom] = vec4<f32>(velocities[atom].xyz + 0.5 * sim.dt * forces[atom].xyz * inv_m, 0.0);
}

@compute @workgroup_size(64)
fn drift(@builtin(global_invocation_id) gid: vec3<u32>) {
    let atom = gid.x;
    if (atom >= sim.num_atoms) { return; }
    positions[atom] = vec4<f32>(positions[atom].xyz + sim.dt * velocities[atom].xyz, 0.0);
}

// BAOAB's "A" sub-step (dt/2 drift, applied twice around the O-step).
@compute @workgroup_size(64)
fn half_drift(@builtin(global_invocation_id) gid: vec3<u32>) {
    let atom = gid.x;
    if (atom >= sim.num_atoms) { return; }
    positions[atom] = vec4<f32>(positions[atom].xyz + 0.5 * sim.dt * velocities[atom].xyz, 0.0);
}

@compute @workgroup_size(64)
fn ke_reduce(@builtin(global_invocation_id) gid: vec3<u32>) {
    let atom = gid.x;
    if (atom == 0u) {
        atomicStore(&ke_accum[0], 0);
    }
    storageBarrier();
    if (atom >= sim.num_atoms) { return; }
    let inv_m = atom_params[atom].w;
    if (inv_m <= 0.0) { return; }
    let mass = 1.0 / inv_m;
    let v2 = dot(velocities[atom].xyz, velocities[atom].xyz);
    let ke = 0.5 * mass * v2;
    atomicAdd(&ke_accum[0], i32(ke * FP_SCALE));
}

struct ScaleUniform {
    scale: f32,
    _p0: f32,
    _p1: f32,
    _p2: f32,
}
@group(3) @binding(0) var<uniform> scale_u: ScaleUniform;

struct LangevinUniform {
    seed0: u32,
    seed1: u32,
    step: u32,
    target_temperature: f32,
    gamma: f32,
    _p0: f32,
    _p1: f32,
    _p2: f32,
}
@group(3) @binding(1) var<uniform> langevin_u: LangevinUniform;

struct PhiloxTestUniform {
    ctr0: u32,
    ctr1: u32,
    ctr2: u32,
    ctr3: u32,
    key0: u32,
    key1: u32,
    _p0: u32,
    _p1: u32,
}
@group(3) @binding(2) var<uniform> philox_test: PhiloxTestUniform;

@compute @workgroup_size(64)
fn scale_velocities(@builtin(global_invocation_id) gid: vec3<u32>) {
    let atom = gid.x;
    if (atom >= sim.num_atoms) { return; }
    velocities[atom] = vec4<f32>(velocities[atom].xyz * scale_u.scale, 0.0);
}

// ---- Philox4x32-10 counter-based RNG (Salmon, Moraes, Dror & Shaw, 2011 —
// the Random123 algorithm) + Box-Muller normals, for the BAOAB Langevin
// O-step below. Bit-exact port of rust/src/random/mod.rs's
// `philox4x32_10`/`box_muller` (that module's `philox_matches_random123_
// kat_vectors` test checks the same round structure against the published
// Random123 reference vectors; `philox_debug` below lets a Rust test
// dispatch this exact WGSL implementation and compare its output to that
// same Rust reference, cross-validating the port on real
// hardware/lavapipe). Every atom/DOF gets its own independent draw from
// `(key, counter)` alone: `key = (seed0, seed1)` is fixed for the whole run,
// `counter = (step, atom_index, 0, 0)`, so no shared RNG state is needed and
// results don't depend on dispatch/thread order.
const PHILOX_M0: u32 = 0xD2511F53u;
const PHILOX_M1: u32 = 0xCD9E8D57u;
const PHILOX_W0: u32 = 0x9E3779B9u;
const PHILOX_W1: u32 = 0xBB67AE85u;

// Returns (lo, hi) of the full 64-bit product a*b, via 16-bit-limb long
// multiplication (WGSL has no native 64-bit integer type); u32 arithmetic
// wraps mod 2^32 per the WGSL spec, which is exactly what a lo/hi split
// needs.
fn mulhilo32(a: u32, b: u32) -> vec2<u32> {
    let alo = a & 0xffffu;
    let ahi = a >> 16u;
    let blo = b & 0xffffu;
    let bhi = b >> 16u;

    let p0 = alo * blo;
    let p1 = alo * bhi;
    let p2 = ahi * blo;
    let p3 = ahi * bhi;

    let carry = ((p0 >> 16u) + (p1 & 0xffffu) + (p2 & 0xffffu)) >> 16u;

    let lo = p0 + ((p1 & 0xffffu) << 16u) + ((p2 & 0xffffu) << 16u);
    let hi = p3 + (p1 >> 16u) + (p2 >> 16u) + carry;
    return vec2<u32>(lo, hi);
}

fn philox4x32_10(counter_in: vec4<u32>, key_in: vec2<u32>) -> vec4<u32> {
    var c = counter_in;
    var k = key_in;
    for (var r = 0u; r < 10u; r = r + 1u) {
        let hilo0 = mulhilo32(PHILOX_M0, c.x);
        let hilo1 = mulhilo32(PHILOX_M1, c.z);
        let lo0 = hilo0.x;
        let hi0 = hilo0.y;
        let lo1 = hilo1.x;
        let hi1 = hilo1.y;
        c = vec4<u32>(hi1 ^ c.y ^ k.x, lo1, hi0 ^ c.w ^ k.y, lo0);
        k.x = k.x + PHILOX_W0;
        k.y = k.y + PHILOX_W1;
    }
    return c;
}

fn box_muller(u1: u32, u2: u32) -> vec2<f32> {
    let r1 = max((f32(u1) + 0.5) / 4294967296.0, 1e-30);
    let r2 = (f32(u2) + 0.5) / 4294967296.0;
    let radius = sqrt(-2.0 * log(r1));
    let theta = 2.0 * 3.14159265 * r2;
    return vec2<f32>(radius * cos(theta), radius * sin(theta));
}

@compute @workgroup_size(1)
fn philox_debug(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x != 0u) { return; }
    let ctr = vec4<u32>(philox_test.ctr0, philox_test.ctr1, philox_test.ctr2, philox_test.ctr3);
    let key = vec2<u32>(philox_test.key0, philox_test.key1);
    philox_debug_out[0] = philox4x32_10(ctr, key);
}

// ---- BAOAB Langevin: full per-atom O-step on the GPU, no readback. ----
// The B (kick) and A (drift) sub-steps reuse the existing `kick_half`/
// `drift` kernels (called twice each, around this O-step, by
// `GpuResidentEngine::run`); SETTLE/SHAKE position and velocity constraints
// are re-applied after each A and after this O exactly as they are for the
// unthermostatted integrator, so this is compatible with rigid
// water/solute H-bonds. v_new = c1*v + c2*sqrt(kB*T/m)*N(0,1) per DOF,
// c1 = exp(-gamma*dt), c2 = sqrt(1 - c1^2) — the standard BAOAB
// Ornstein-Uhlenbeck velocity update (Leimkuhler & Matthews).
const LANGEVIN_KB: f32 = 0.0019872041; // kcal/mol/K, matches the rest of the codebase

@compute @workgroup_size(64)
fn langevin_o_step(@builtin(global_invocation_id) gid: vec3<u32>) {
    let atom = gid.x;
    if (atom >= sim.num_atoms) { return; }
    let inv_m = atom_params[atom].w;
    if (inv_m <= 0.0) { return; }
    let mass = 1.0 / inv_m;

    let c1 = exp(-langevin_u.gamma * sim.dt);
    let c2 = sqrt(max(1.0 - c1 * c1, 0.0) * LANGEVIN_KB * langevin_u.target_temperature / mass);

    let ctr = vec4<u32>(langevin_u.step, atom, 0u, 0u);
    let key = vec2<u32>(langevin_u.seed0, langevin_u.seed1);
    let r = philox4x32_10(ctr, key);
    let bm1 = box_muller(r.x, r.y);
    let bm2 = box_muller(r.z, r.w);
    let noise = vec3<f32>(bm1.x, bm1.y, bm2.x);

    velocities[atom] = vec4<f32>(velocities[atom].xyz * c1 + noise * c2, 0.0);
}

// ---- constraints: SETTLE (literal Miyamoto-Kollman, ported from
// constraints::settle_one / constraints::apply_settle_velocity) + SHAKE
// (Jacobi-parallel iteration of constraints::apply_shake) — no per-step
// readback: `settle_ref_positions` is snapshotted on-GPU right before the
// kick+drift that needs constraining, and both position/velocity
// corrections are applied in place. ----

@compute @workgroup_size(64)
fn snapshot_settle_ref(@builtin(global_invocation_id) gid: vec3<u32>) {
    let atom = gid.x;
    if (atom >= sim.num_atoms) { return; }
    settle_ref_positions[atom] = positions[atom];
}

fn atom_mass(atom: u32) -> f32 {
    return 1.0 / max(atom_params[atom].w, 1e-12);
}

// One thread per water molecule; literal port of
// constraints::settle_one's canonical-frame closed-form solve.
@compute @workgroup_size(64)
fn settle_position(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = gid.x;
    if (w >= sim.num_waters) { return; }
    let idx = water_idx[w];
    let o = idx.x;
    let h1 = idx.y;
    let h2 = idx.z;
    let par = water_params[w];
    let roh = par.x;
    let rhh = par.y;

    let m0 = atom_mass(o);
    let m1 = atom_mass(h1);
    let m2 = atom_mass(h2);

    let apos0 = settle_ref_positions[o].xyz;
    let apos1 = settle_ref_positions[h1].xyz;
    let apos2 = settle_ref_positions[h2].xyz;
    let xp0_new = positions[o].xyz;
    let xp1_new = positions[h1].xyz;
    let xp2_new = positions[h2].xyz;
    var xp0 = xp0_new - apos0;
    var xp1 = xp1_new - apos1;
    var xp2 = xp2_new - apos2;

    // --- Step1: A1' ---
    let b0 = apos1 - apos0;
    let c0 = apos2 - apos0;
    let inv_total_mass = 1.0 / (m0 + m1 + m2);

    let com = (xp0 * m0 + (b0 + xp1) * m1 + (c0 + xp2) * m2) * inv_total_mass;

    let a1 = xp0 - com;
    let b1 = b0 + xp1 - com;
    let c1 = c0 + xp2 - com;

    let aks_zd = cross(b0, c0);
    let aks_xd = cross(a1, aks_zd);
    let aks_yd = cross(aks_zd, aks_xd);

    let axlng = length(aks_xd);
    let aylng = length(aks_yd);
    let azlng = max(length(aks_zd), 1e-20);

    let trns_x = aks_xd / max(axlng, 1e-20);
    let trns_y = aks_yd / max(aylng, 1e-20);
    let trns_z = aks_zd / azlng;

    let xb0d = dot(trns_x, b0);
    let yb0d = dot(trns_y, b0);
    let xc0d = dot(trns_x, c0);
    let yc0d = dot(trns_y, c0);
    let za1d = dot(trns_z, a1);
    let xb1d = dot(trns_x, b1);
    let yb1d = dot(trns_y, b1);
    let zb1d = dot(trns_z, b1);
    let xc1d = dot(trns_x, c1);
    let yc1d = dot(trns_y, c1);
    let zc1d = dot(trns_z, c1);

    // --- Step2: A2' ---
    let rc = 0.5 * rhh;
    var rb = sqrt(max(roh * roh - rc * rc, 0.0));
    let ra = rb * (m1 + m2) * inv_total_mass;
    rb -= ra;
    let sinphi = clamp(za1d / ra, -1.0, 1.0);
    let cosphi = sqrt(max(1.0 - sinphi * sinphi, 0.0));
    let sinpsi = clamp((zb1d - zc1d) / (2.0 * rc * cosphi), -1.0, 1.0);
    let cospsi = sqrt(max(1.0 - sinpsi * sinpsi, 0.0));

    let ya2d = ra * cosphi;
    var xb2d = -rc * cospsi;
    let yb2d = -rb * cosphi - rc * sinpsi * sinphi;
    let yc2d = -rb * cosphi + rc * sinpsi * sinphi;
    let xb2d2 = xb2d * xb2d;
    let hh2 = 4.0 * xb2d2 + (yb2d - yc2d) * (yb2d - yc2d) + (zb1d - zc1d) * (zb1d - zc1d);
    let deltx = 2.0 * xb2d + sqrt(max(4.0 * xb2d2 - hh2 + rhh * rhh, 0.0));
    xb2d -= deltx * 0.5;

    // --- Step3: al, be, ga ---
    let alpha = xb2d * (xb0d - xc0d) + yb0d * yb2d + yc0d * yc2d;
    let beta = xb2d * (yc0d - yb0d) + xb0d * yb2d + xc0d * yc2d;
    let gamma = xb0d * yb1d - xb1d * yb0d + xc0d * yc1d - xc1d * yc0d;

    let al2be2 = alpha * alpha + beta * beta;
    let sintheta = clamp(
        (alpha * gamma - beta * sqrt(max(al2be2 - gamma * gamma, 0.0))) / al2be2,
        -1.0,
        1.0,
    );

    // --- Step4: A3' ---
    let costheta = sqrt(max(1.0 - sintheta * sintheta, 0.0));
    let xa3d = -ya2d * sintheta;
    let ya3d = ya2d * costheta;
    let za3d = za1d;
    let xb3d = xb2d * costheta - yb2d * sintheta;
    let yb3d = xb2d * sintheta + yb2d * costheta;
    let zb3d = zb1d;
    let xc3d = -xb2d * costheta - yc2d * sintheta;
    let yc3d = -xb2d * sintheta + yc2d * costheta;
    let zc3d = zc1d;

    // --- Step5: A3 (rotate back to lab frame) ---
    let a3 = xa3d * trns_x + ya3d * trns_y + za3d * trns_z;
    let b3 = xb3d * trns_x + yb3d * trns_y + zb3d * trns_z;
    let c3 = xc3d * trns_x + yc3d * trns_y + zc3d * trns_z;

    xp0 = com + a3;
    xp1 = com + b3 - b0;
    xp2 = com + c3 - c0;

    positions[o] = vec4<f32>(xp0 + apos0, 0.0);
    positions[h1] = vec4<f32>(xp1 + apos1, 0.0);
    positions[h2] = vec4<f32>(xp2 + apos2, 0.0);
}

// One thread per water molecule; literal port of
// constraints::apply_settle_velocity's RATTLE-analog linear solve.
@compute @workgroup_size(64)
fn settle_velocity(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = gid.x;
    if (w >= sim.num_waters) { return; }
    let idx = water_idx[w];
    let o = idx.x;
    let h1 = idx.y;
    let h2 = idx.z;

    let apos0 = positions[o].xyz;
    let apos1 = positions[h1].xyz;
    let apos2 = positions[h2].xyz;
    let m_a = atom_mass(o);
    let m_b = atom_mass(h1);
    let m_c = atom_mass(h2);
    var v0 = velocities[o].xyz;
    var v1 = velocities[h1].xyz;
    var v2 = velocities[h2].xyz;

    let e_ab = normalize(apos1 - apos0);
    let e_bc = normalize(apos2 - apos1);
    let e_ca = normalize(apos0 - apos2);

    let v_ab = dot(v1 - v0, e_ab);
    let v_bc = dot(v2 - v1, e_bc);
    let v_ca = dot(v0 - v2, e_ca);

    let c_a = -dot(e_ab, e_ca);
    let c_b = -dot(e_ab, e_bc);
    let c_c = -dot(e_bc, e_ca);
    let s2a = 1.0 - c_a * c_a;
    let s2b = 1.0 - c_b * c_b;
    let s2c = 1.0 - c_c * c_c;

    let mabc_inv = 1.0 / (m_a * m_b * m_c);
    let denom = (((s2a * m_b + s2b * m_a) * m_c
        + (s2a * m_b * m_b + 2.0 * (c_a * c_b * c_c + 1.0) * m_a * m_b + s2b * m_a * m_a))
        * m_c
        + s2c * m_a * m_b * (m_a + m_b))
        * mabc_inv;
    let tab = ((c_b * c_c * m_a - c_a * m_b - c_a * m_c) * v_ca
        + (c_a * c_c * m_b - c_b * m_c - c_b * m_a) * v_bc
        + (s2c * m_a * m_a * m_b * m_b * mabc_inv + (m_a + m_b + m_c)) * v_ab)
        / denom;
    let tbc = ((c_a * c_b * m_c - c_c * m_b - c_c * m_a) * v_ca
        + (s2a * m_b * m_b * m_c * m_c * mabc_inv + (m_a + m_b + m_c)) * v_bc
        + (c_a * c_c * m_b - c_b * m_a - c_b * m_c) * v_ab)
        / denom;
    let tca = ((s2b * m_a * m_a * m_c * m_c * mabc_inv + (m_a + m_b + m_c)) * v_ca
        + (c_a * c_b * m_c - c_c * m_b - c_c * m_a) * v_bc
        + (c_b * c_c * m_a - c_a * m_b - c_a * m_c) * v_ab)
        / denom;

    v0 += (e_ab * tab - e_ca * tca) / m_a;
    v1 += (e_bc * tbc - e_ab * tab) / m_b;
    v2 += (e_ca * tca - e_bc * tbc) / m_c;

    velocities[o] = vec4<f32>(v0, 0.0);
    velocities[h1] = vec4<f32>(v1, 0.0);
    velocities[h2] = vec4<f32>(v2, 0.0);
}

// ---- SHAKE (Jacobi-parallel) for solute H-bond constraints ----
// GPU threads run concurrently, unlike the CPU's sequential Gauss-Seidel
// `apply_shake`, so this is a Jacobi variant: every bond computes its
// correction from the same snapshot of positions and atomically
// accumulates a fixed-point delta per atom; a second kernel applies the
// accumulated deltas and clears the accumulator. The host dispatches this
// pair a fixed number of times per step (see `GpuResidentEngine::run`)
// rather than reading back a convergence residual.
@compute @workgroup_size(64)
fn shake_correction_pass(@builtin(global_invocation_id) gid: vec3<u32>) {
    let b = gid.x;
    if (b >= sim.num_shake_bonds) { return; }
    let idx = shake_idx[b];
    let r0 = shake_r0[b];
    let pi = positions[idx.x].xyz;
    let pj = positions[idx.y].xyz;
    let dr = pj - pi;
    let r = max(length(dr), 1e-12);
    let err = r - r0;
    let mi = atom_mass(idx.x);
    let mj = atom_mass(idx.y);
    let inv_mass = 1.0 / mi + 1.0 / mj;
    let corr = err / (2.0 * r * inv_mass);
    let dc = corr * dr;
    atomicAdd(&pos_correction_fp[idx.x * 3u + 0u], i32((dc.x / mi) * FP_SCALE));
    atomicAdd(&pos_correction_fp[idx.x * 3u + 1u], i32((dc.y / mi) * FP_SCALE));
    atomicAdd(&pos_correction_fp[idx.x * 3u + 2u], i32((dc.z / mi) * FP_SCALE));
    atomicAdd(&pos_correction_fp[idx.y * 3u + 0u], i32(-(dc.x / mj) * FP_SCALE));
    atomicAdd(&pos_correction_fp[idx.y * 3u + 1u], i32(-(dc.y / mj) * FP_SCALE));
    atomicAdd(&pos_correction_fp[idx.y * 3u + 2u], i32(-(dc.z / mj) * FP_SCALE));
}

@compute @workgroup_size(64)
fn apply_shake_correction(@builtin(global_invocation_id) gid: vec3<u32>) {
    let atom = gid.x;
    if (atom >= sim.num_atoms) { return; }
    let dx = f32(atomicLoad(&pos_correction_fp[atom * 3u + 0u])) / FP_SCALE;
    let dy = f32(atomicLoad(&pos_correction_fp[atom * 3u + 1u])) / FP_SCALE;
    let dz = f32(atomicLoad(&pos_correction_fp[atom * 3u + 2u])) / FP_SCALE;
    positions[atom] = positions[atom] + vec4<f32>(dx, dy, dz, 0.0);
    atomicStore(&pos_correction_fp[atom * 3u + 0u], 0);
    atomicStore(&pos_correction_fp[atom * 3u + 1u], 0);
    atomicStore(&pos_correction_fp[atom * 3u + 2u], 0);
}

// RATTLE velocity projection for the same solute H-bond constraints
// (constraints::apply_rattle's `shake_bonds` loop, literally ported); reuses
// `pos_correction_fp` as scratch (cleared by the apply kernel each pass, and
// never live at the same time as a position-correction pass).
@compute @workgroup_size(64)
fn shake_velocity_pass(@builtin(global_invocation_id) gid: vec3<u32>) {
    let b = gid.x;
    if (b >= sim.num_shake_bonds) { return; }
    let idx = shake_idx[b];
    let pi = positions[idx.x].xyz;
    let pj = positions[idx.y].xyz;
    let dr = pj - pi;
    let r2 = dot(dr, dr);
    let vi = velocities[idx.x].xyz;
    let mi = atom_mass(idx.x);
    let mj = atom_mass(idx.y);
    let inv_mass = 1.0 / mi + 1.0 / mj;
    let dot_v = dot(vi, dr);
    let corr = dot_v / max(r2 * inv_mass, 1e-20);
    let dc = corr * dr;
    atomicAdd(&pos_correction_fp[idx.x * 3u + 0u], i32(-(dc.x / mi) * FP_SCALE));
    atomicAdd(&pos_correction_fp[idx.x * 3u + 1u], i32(-(dc.y / mi) * FP_SCALE));
    atomicAdd(&pos_correction_fp[idx.x * 3u + 2u], i32(-(dc.z / mi) * FP_SCALE));
    atomicAdd(&pos_correction_fp[idx.y * 3u + 0u], i32((dc.x / mj) * FP_SCALE));
    atomicAdd(&pos_correction_fp[idx.y * 3u + 1u], i32((dc.y / mj) * FP_SCALE));
    atomicAdd(&pos_correction_fp[idx.y * 3u + 2u], i32((dc.z / mj) * FP_SCALE));
}

@compute @workgroup_size(64)
fn apply_shake_velocity_correction(@builtin(global_invocation_id) gid: vec3<u32>) {
    let atom = gid.x;
    if (atom >= sim.num_atoms) { return; }
    let dx = f32(atomicLoad(&pos_correction_fp[atom * 3u + 0u])) / FP_SCALE;
    let dy = f32(atomicLoad(&pos_correction_fp[atom * 3u + 1u])) / FP_SCALE;
    let dz = f32(atomicLoad(&pos_correction_fp[atom * 3u + 2u])) / FP_SCALE;
    velocities[atom] = velocities[atom] + vec4<f32>(dx, dy, dz, 0.0);
    atomicStore(&pos_correction_fp[atom * 3u + 0u], 0);
    atomicStore(&pos_correction_fp[atom * 3u + 1u], 0);
    atomicStore(&pos_correction_fp[atom * 3u + 2u], 0);
}

// ---- PME: GPU B-spline charge spreading + force gather ----
// Literal port of electrostatics::pme's `fill_bspline` (Cardinal B-spline
// weights/derivatives, de Boor recursion) and `atom_splines`
// (fractional-coordinate + base-grid-index setup), so the spread/gather
// weights match the CPU `pme_recip_from_grid` pipeline exactly. MAX_PME_ORDER
// mirrors `PmeContext::with_params`'s `order.clamp(3, 8)`.
const MAX_PME_ORDER: u32 = 8u;
const PME_FP_SCALE: f32 = 1048576.0;

fn fill_bspline(w: f32, order: u32, arr: ptr<function, array<f32, 8>>, darr: ptr<function, array<f32, 8>>) {
    for (var i = 0u; i < MAX_PME_ORDER; i = i + 1u) {
        (*arr)[i] = 0.0;
        (*darr)[i] = 0.0;
    }
    (*arr)[1] = w;
    (*arr)[0] = 1.0 - w;

    for (var k = 3u; k < order; k = k + 1u) {
        let div = 1.0 / (f32(k) - 1.0);
        (*arr)[k - 1u] = div * w * (*arr)[k - 2u];
        for (var j = 1u; j < (k - 1u); j = j + 1u) {
            (*arr)[k - 1u - j] = div * ((w + f32(j)) * (*arr)[k - 2u - j] + (f32(k) - f32(j) - w) * (*arr)[k - 1u - j]);
        }
        (*arr)[0] = div * (1.0 - w) * (*arr)[0];
    }

    (*darr)[0] = -(*arr)[0];
    for (var j = 1u; j < order; j = j + 1u) {
        (*darr)[j] = (*arr)[j - 1u] - (*arr)[j];
    }

    let k = order;
    let div = 1.0 / (f32(k) - 1.0);
    (*arr)[k - 1u] = div * w * (*arr)[k - 2u];
    for (var j = 1u; j < (k - 1u); j = j + 1u) {
        (*arr)[k - 1u - j] = div * ((w + f32(j)) * (*arr)[k - 2u - j] + (f32(k) - f32(j) - w) * (*arr)[k - 1u - j]);
    }
    (*arr)[0] = div * (1.0 - w) * (*arr)[0];
}

// Per-dimension fractional coordinate + B-spline weights/derivatives,
// reversed so `w[i]`/`dw[i]` line up with grid point `(base - i)`, matching
// `atom_splines`'s convention exactly.
fn atom_spline_dim(
    pos_d: f32,
    length_d: f32,
    grid_d: u32,
    order: u32,
    base: ptr<function, i32>,
    w: ptr<function, array<f32, 8>>,
    dw: ptr<function, array<f32, 8>>,
) {
    let l = max(length_d, 1e-12);
    var s = pos_d / l;
    s = s - floor(s);
    let u = s * f32(grid_d);
    let u0 = floor(u);
    let frac = clamp(u - u0, 0.0, 1.0 - 1e-6);
    *base = i32(u0);
    var arr: array<f32, 8>;
    var darr: array<f32, 8>;
    fill_bspline(frac, order, &arr, &darr);
    for (var i = 0u; i < order; i = i + 1u) {
        (*w)[i] = arr[order - 1u - i];
        (*dw)[i] = darr[order - 1u - i];
    }
}

fn pme_grid_idx(base: i32, i: u32, dim: u32) -> u32 {
    let raw = base - i32(i);
    let m = raw % i32(dim);
    return u32(select(m, m + i32(dim), m < 0));
}

@compute @workgroup_size(64)
fn pme_clear_grid(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = sim.pme_grid_x * sim.pme_grid_y * sim.pme_grid_z;
    if (idx >= total) { return; }
    atomicStore(&q_grid_fp[idx], 0);
}

@compute @workgroup_size(64)
fn pme_spread(@builtin(global_invocation_id) gid: vec3<u32>) {
    let atom = gid.x;
    if (atom >= sim.num_atoms) { return; }
    let q = atom_params[atom].x;
    if (abs(q) < 1e-12) { return; }
    let pos = positions[atom].xyz;

    var basex: i32; var wx: array<f32, 8>; var dwx: array<f32, 8>;
    var basey: i32; var wy: array<f32, 8>; var dwy: array<f32, 8>;
    var basez: i32; var wz: array<f32, 8>; var dwz: array<f32, 8>;
    atom_spline_dim(pos.x, sim.box_lx, sim.pme_grid_x, sim.pme_order, &basex, &wx, &dwx);
    atom_spline_dim(pos.y, sim.box_ly, sim.pme_grid_y, sim.pme_order, &basey, &wy, &dwy);
    atom_spline_dim(pos.z, sim.box_lz, sim.pme_grid_z, sim.pme_order, &basez, &wz, &dwz);

    for (var ix = 0u; ix < sim.pme_order; ix = ix + 1u) {
        let gx = pme_grid_idx(basex, ix, sim.pme_grid_x);
        let wxv = wx[ix];
        if (wxv == 0.0) { continue; }
        for (var iy = 0u; iy < sim.pme_order; iy = iy + 1u) {
            let gy = pme_grid_idx(basey, iy, sim.pme_grid_y);
            let wxy = wxv * wy[iy];
            if (wxy == 0.0) { continue; }
            let row = (gx * sim.pme_grid_y + gy) * sim.pme_grid_z;
            for (var iz = 0u; iz < sim.pme_order; iz = iz + 1u) {
                let gz = pme_grid_idx(basez, iz, sim.pme_grid_z);
                let weight = wxy * wz[iz];
                atomicAdd(&q_grid_fp[row + gz], i32(q * weight * PME_FP_SCALE));
            }
        }
    }
}

@compute @workgroup_size(64)
fn pme_gather(@builtin(global_invocation_id) gid: vec3<u32>) {
    let atom = gid.x;
    if (atom >= sim.num_atoms) { return; }
    let q = atom_params[atom].x;
    if (abs(q) < 1e-12) { return; }
    let pos = positions[atom].xyz;

    var basex: i32; var wx: array<f32, 8>; var dwx: array<f32, 8>;
    var basey: i32; var wy: array<f32, 8>; var dwy: array<f32, 8>;
    var basez: i32; var wz: array<f32, 8>; var dwz: array<f32, 8>;
    atom_spline_dim(pos.x, sim.box_lx, sim.pme_grid_x, sim.pme_order, &basex, &wx, &dwx);
    atom_spline_dim(pos.y, sim.box_ly, sim.pme_grid_y, sim.pme_order, &basey, &wy, &dwy);
    atom_spline_dim(pos.z, sim.box_lz, sim.pme_grid_z, sim.pme_order, &basez, &wz, &dwz);

    let scale_x = f32(sim.pme_grid_x) / sim.box_lx;
    let scale_y = f32(sim.pme_grid_y) / sim.box_ly;
    let scale_z = f32(sim.pme_grid_z) / sim.box_lz;

    var f = vec3<f32>(0.0, 0.0, 0.0);
    for (var ix = 0u; ix < sim.pme_order; ix = ix + 1u) {
        let gx = pme_grid_idx(basex, ix, sim.pme_grid_x);
        let wxv = wx[ix];
        let dwxv = dwx[ix];
        for (var iy = 0u; iy < sim.pme_order; iy = iy + 1u) {
            let gy = pme_grid_idx(basey, iy, sim.pme_grid_y);
            let wyv = wy[iy];
            let dwyv = dwy[iy];
            let row = (gx * sim.pme_grid_y + gy) * sim.pme_grid_z;
            for (var iz = 0u; iz < sim.pme_order; iz = iz + 1u) {
                let gz = pme_grid_idx(basez, iz, sim.pme_grid_z);
                let wzv = wz[iz];
                let dwzv = dwz[iz];
                let g = pme_potential_grid[row + gz];
                f.x -= q * dwxv * wyv * wzv * scale_x * g;
                f.y -= q * wxv * dwyv * wzv * scale_y * g;
                f.z -= q * wxv * wyv * dwzv * scale_z * g;
            }
        }
    }
    add_force_fp(atom, f);
}

// ---- Exclusion / 1-4 real-space correction for PME ----
// The reciprocal sum implicitly includes the full q_i*q_j/r interaction for
// every pair, including bonded exclusions and 1-4 pairs whose direct-space
// term was zeroed or scaled; this subtracts erf(alpha*r)/r times
// (1 - scale) for each, on the GPU, from the same per-atom exclusion/1-4
// tables the nonbonded kernel already uses — a literal port of
// electrostatics::excluded_pair_correction, per-atom rather than per-pair
// (each atom visits its own exclusion/1-4 partners and applies half the
// correction, so summing over atoms gives the same total as summing once
// per unique pair).
@compute @workgroup_size(64)
fn exclusion_correction(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= sim.num_atoms) { return; }
    if (sim.alpha <= 0.0) {
        excl_energy[i] = 0.0;
        return;
    }
    let pi = positions[i].xyz;
    let qi = atom_params[i].x;
    var eacc = 0.0;
    var facc = vec3<f32>(0.0, 0.0, 0.0);

    for (var k = 0u; k < sim.max_per_atom_excl; k = k + 1u) {
        let j = exclusions[i * MAX_EXCL + k];
        if (j == 0xffffffffu) { break; }
        let qj = atom_params[j].x;
        if (qi == 0.0 || qj == 0.0) { continue; }
        let dr = mic(positions[j].xyz - pi);
        let r2 = max(dot(dr, dr), 1e-12);
        let r = sqrt(r2);
        let arg = sim.alpha * r;
        let erf_val = 1.0 - erfc_approx(arg);
        let pref = sim.coulomb * qi * qj; // one_minus_scale = 1.0 (full exclusion)
        let e = -pref * erf_val / r;
        eacc += 0.5 * e;
        let derf_dr = (2.0 * sim.alpha / sqrt(3.14159265)) * exp(-arg * arg);
        let fscalar = pref * (derf_dr / r - erf_val / r2) / r;
        facc -= fscalar * dr;
    }
    for (var k = 0u; k < sim.max_per_atom_14; k = k + 1u) {
        let j = pairs14[i * MAX_14 + k];
        if (j == 0xffffffffu) { break; }
        let qj = atom_params[j].x;
        let one_minus_scale = 1.0 - COULOMB_14_SCALE;
        if (qi == 0.0 || qj == 0.0) { continue; }
        let dr = mic(positions[j].xyz - pi);
        let r2 = max(dot(dr, dr), 1e-12);
        let r = sqrt(r2);
        let arg = sim.alpha * r;
        let erf_val = 1.0 - erfc_approx(arg);
        let pref = sim.coulomb * qi * qj * one_minus_scale;
        let e = -pref * erf_val / r;
        eacc += 0.5 * e;
        let derf_dr = (2.0 * sim.alpha / sqrt(3.14159265)) * exp(-arg * arg);
        let fscalar = pref * (derf_dr / r - erf_val / r2) / r;
        facc -= fscalar * dr;
    }

    excl_energy[i] = eacc;
    add_force_fp(i, facc);
}
