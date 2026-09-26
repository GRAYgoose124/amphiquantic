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

@compute @workgroup_size(64)
fn scale_velocities(@builtin(global_invocation_id) gid: vec3<u32>) {
    let atom = gid.x;
    if (atom >= sim.num_atoms) { return; }
    velocities[atom] = vec4<f32>(velocities[atom].xyz * scale_u.scale, 0.0);
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
