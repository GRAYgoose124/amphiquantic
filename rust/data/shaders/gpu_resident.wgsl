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
    num_bonds: u32,
    num_angles: u32,
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
@group(1) @binding(9) var<storage, read_write> external_force: array<vec4<f32>>; // CPU-side dihedral/PME contribution

@group(2) @binding(0) var<storage, read> bond_idx: array<vec2<u32>>;
@group(2) @binding(1) var<storage, read> bond_params: array<vec2<f32>>; // r0, k
@group(2) @binding(2) var<storage, read> angle_idx: array<vec4<u32>>; // i, j, k, pad
@group(2) @binding(3) var<storage, read> angle_params: array<vec2<f32>>; // theta0(rad), k

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

                    var fscalar = 0.0;
                    if (sigi > 1e-8 && epsj > 1e-12 && epsi > 1e-12) {
                        let sigma = 0.5 * (sigi + sigj);
                        let epsilon = sqrt(epsi * epsj);
                        let sr = sigma / r;
                        let sr6 = pow(sr, 6.0);
                        let sr12 = sr6 * sr6;
                        fscalar += 24.0 * epsilon * (2.0 * sr12 - sr6) / r2;
                    }
                    if (sim.alpha > 0.0) {
                        let ar = sim.alpha * r;
                        let erfc_v = erfc_approx(ar);
                        let expfac = exp(-ar * ar);
                        let qq = sim.coulomb * qi * qj;
                        let e_deriv = qq * (erfc_v / r + (2.0 * sim.alpha / sqrt(3.14159265) ) * expfac) / r2;
                        fscalar += e_deriv;
                    } else {
                        let qq = sim.coulomb * qi * qj;
                        fscalar += qq / (r2 * r);
                    }
                    facc -= fscalar * dr;
                }
            }
        }
    }
    add_force_fp(i, facc);
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
    let fscalar = -2.0 * k * (r - r0) / r;
    let f = fscalar * dr;
    add_force_fp(idx.x, -f);
    add_force_fp(idx.y, f);
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
    let dvdt = 2.0 * k * (theta - theta0);
    let coef = -dvdt / sin_t;

    let fi = coef * (rkj / (lij * lkj) - cos_t * rij / (lij * lij));
    let fk = coef * (rij / (lij * lkj) - cos_t * rkj / (lkj * lkj));
    let fj = -(fi + fk);
    add_force_fp(idx.x, fi);
    add_force_fp(idx.y, fj);
    add_force_fp(idx.z, fk);
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
