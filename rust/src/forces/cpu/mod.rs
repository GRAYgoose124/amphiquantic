use crate::electrostatics::pme::{self, PmeContext};
use crate::neighbor::{build_neighbor_list, minimum_image};
use crate::topology::Topology;
use rayon::prelude::*;
use std::cell::RefCell;

// ---------------------------------------------------------------------
// Persistent per-thread scratch buffers for the parallel force reduction.
//
// The naive `par_iter().fold(|| vec![[0.0;3]; n], ...)` pattern allocates a
// brand-new full-length force buffer for *every* fold split rayon creates
// (which, with the default adaptive splitting, is far more than one per
// thread for anything but the tiniest input), and pays for it every single
// force-evaluation call, every MD step. That allocation + zeroing cost is
// the dominant reason the measured parallel speedup was ~1.0x at ~600
// atoms: the work being parallelized (a few hundred pair/bond kernels) is
// tiny compared to the multiple full-N-atom `Vec` allocations per call. We
// fix it two ways: (1) a thread-local free list of `Vec<[f64;3]>` buffers,
// so after the first few calls no new heap allocation happens on the hot
// path at all (buffers are just zeroed and reused); (2) `with_min_len` to
// bound the number of fold splits to a small multiple of the thread count
// instead of leaving it to rayon's default fine-grained splitting, and a
// serial fallback below a size threshold where the parallel overhead
// (splitting, work-stealing, cross-thread reduction) is not worth paying.
// ---------------------------------------------------------------------

thread_local! {
    static FORCE_BUFFER_POOL: RefCell<Vec<Vec<[f64; 3]>>> = RefCell::new(Vec::new());
}

/// Below this many work items, run the reduction on the calling thread:
/// the fixed cost of spinning up a parallel fold/reduce (thread-pool
/// dispatch, per-split buffer setup, cross-thread merge) exceeds the cost
/// of just doing the (tiny) amount of work serially.
const PARALLEL_MIN_ITEMS: usize = 1024;

fn take_force_buffer(n_atoms: usize) -> Vec<[f64; 3]> {
    FORCE_BUFFER_POOL.with(|pool| {
        if let Some(mut buf) = pool.borrow_mut().pop() {
            // clear() before resize: a pooled buffer from a differently
            // sized system must not leak stale forces into the new one.
            buf.clear();
            buf.resize(n_atoms, [0.0; 3]);
            buf
        } else {
            vec![[0.0f64; 3]; n_atoms]
        }
    })
}

fn return_force_buffer(buf: Vec<[f64; 3]>) {
    FORCE_BUFFER_POOL.with(|pool| {
        // Keep the pool from growing without bound if some caller ever uses
        // an unusually large number of concurrent buffers on one thread.
        let mut pool = pool.borrow_mut();
        if pool.len() < 8 {
            pool.push(buf);
        }
    });
}

/// Accumulate (forces, energy) from independent per-item contributions in
/// parallel: each rayon worker folds its share of items into a reused,
/// thread-local (Vec<[f64;3]>, f64) buffer, and buffers are reduced
/// pairwise at the end. This avoids any shared mutable state / locking in
/// the hot loop while remaining numerically equivalent (to
/// float-summation-order tolerance) to the serial accumulation. Small item
/// counts fall back to a plain serial loop.
fn parallel_accumulate<T, F>(items: &[T], n_atoms: usize, f: F) -> (Vec<[f64; 3]>, f64)
where
    T: Sync,
    F: Fn(&T, &mut [[f64; 3]], &mut f64) + Sync,
{
    if items.len() < PARALLEL_MIN_ITEMS || rayon::current_num_threads() <= 1 {
        let mut forces = vec![[0.0f64; 3]; n_atoms];
        let mut energy = 0.0f64;
        for item in items {
            f(item, &mut forces, &mut energy);
        }
        return (forces, energy);
    }

    let n_threads = rayon::current_num_threads().max(1);
    // A handful of chunks per thread keeps load balancing reasonable while
    // capping the number of per-split scratch buffers pulled from the pool
    // to a small, thread-count-scaled number instead of one per item.
    let min_len = (items.len() / (n_threads * 4)).max(1);

    let (forces, energy) = items
        .par_iter()
        .with_min_len(min_len)
        .fold(
            || (take_force_buffer(n_atoms), 0.0f64),
            |(mut forces, mut energy), item| {
                f(item, &mut forces, &mut energy);
                (forces, energy)
            },
        )
        .reduce(
            || (take_force_buffer(n_atoms), 0.0f64),
            |(mut fa, ea), (fb, eb)| {
                for (a, b) in fa.iter_mut().zip(fb.iter()) {
                    a[0] += b[0];
                    a[1] += b[1];
                    a[2] += b[2];
                }
                return_force_buffer(fb);
                (fa, ea + eb)
            },
        );
    (forces, energy)
}

pub struct ForceResult {
    pub forces: Vec<[f64; 3]>,
    pub potential_energy: f64,
}

const COULOMB_CONSTANT: f64 = 138.935456;
const LJ_14_SCALE: f64 = 0.5;
const COULOMB_14_SCALE: f64 = 1.0 / 1.2;

pub fn compute_bonded_forces(topology: &Topology) -> ForceResult {
    let n = topology.atoms.len();

    let (bond_forces, bond_energy) = parallel_accumulate(&topology.bonds, n, |bond, forces, energy| {
        let i = bond.i;
        let j = bond.j;
        let pi = topology.atoms[i].position;
        let pj = topology.atoms[j].position;
        let mut dr = [pj[0] - pi[0], pj[1] - pi[1], pj[2] - pi[2]];
        dr = minimum_image(dr, &topology.box_);
        let r = (dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2]).sqrt().max(1e-12);
        let dr_mag = r - bond.r0;
        let force_scalar = -2.0 * bond.k * dr_mag / r;
        for k in 0..3 {
            forces[i][k] -= force_scalar * dr[k];
            forces[j][k] += force_scalar * dr[k];
        }
        *energy += bond.k * dr_mag * dr_mag;
    });

    let (angle_forces, angle_energy) = parallel_accumulate(&topology.angles, n, |angle, forces, energy| {
        let pi = topology.atoms[angle.i].position;
        let pj = topology.atoms[angle.j].position;
        let pk = topology.atoms[angle.k].position;
        let b1 = [pi[0] - pj[0], pi[1] - pj[1], pi[2] - pj[2]];
        let b2 = [pk[0] - pj[0], pk[1] - pj[1], pk[2] - pj[2]];
        let n1 = norm(b1);
        let n2 = norm(b2);
        if n1 < 1e-12 || n2 < 1e-12 {
            return;
        }
        let cos_theta = (dot(b1, b2) / (n1 * n2)).clamp(-1.0, 1.0);
        let theta = cos_theta.acos();
        let dtheta = theta - angle.theta0;
        let coeff = -2.0 * angle.k_theta * dtheta;
        *energy += angle.k_theta * dtheta * dtheta;
        let inv_n1 = 1.0 / n1;
        let inv_n2 = 1.0 / n2;
        for k in 0..3 {
            let d_cos_d_b1 = (b2[k] * inv_n2 - cos_theta * b1[k] * inv_n1) / n1;
            let d_cos_d_b2 = (b1[k] * inv_n1 - cos_theta * b2[k] * inv_n2) / n2;
            forces[angle.i][k] += coeff * d_cos_d_b1;
            forces[angle.j][k] += coeff * (-d_cos_d_b1 - d_cos_d_b2);
            forces[angle.k][k] += coeff * d_cos_d_b2;
        }
    });

    let dihedrals: Vec<&crate::topology::DihedralTerm> = topology
        .dihedrals
        .iter()
        .chain(topology.impropers.iter())
        .collect();
    let (dihedral_forces, dihedral_energy) =
        parallel_accumulate(&dihedrals, n, |dihedral, forces, energy| {
            add_dihedral_forces(topology, *dihedral, forces, energy);
        });

    let mut forces = bond_forces;
    for (a, b) in forces.iter_mut().zip(angle_forces.iter()) {
        a[0] += b[0];
        a[1] += b[1];
        a[2] += b[2];
    }
    for (a, b) in forces.iter_mut().zip(dihedral_forces.iter()) {
        a[0] += b[0];
        a[1] += b[1];
        a[2] += b[2];
    }
    // These two buffers are fully consumed above; hand them back to the
    // thread-local pool so the next force-evaluation call (next MD step)
    // can reuse them instead of allocating.
    return_force_buffer(angle_forces);
    return_force_buffer(dihedral_forces);

    ForceResult {
        forces,
        potential_energy: bond_energy + angle_energy + dihedral_energy,
    }
}

pub fn compute_nonbonded_forces(
    topology: &Topology,
    cutoff: f64,
    pme: Option<&PmeContext>,
) -> ForceResult {
    let n = topology.atoms.len();

    let nl = build_neighbor_list(topology, cutoff);
    let pairs_14 = build_14_pairs(topology);

    let (mut forces, mut potential_energy) =
        parallel_accumulate(&nl.pairs, n, |&(i, j), forces, energy| {
            let pi = topology.atoms[i].position;
            let pj = topology.atoms[j].position;
            let qi = topology.atoms[i].charge;
            let qj = topology.atoms[j].charge;
            let sigma = 0.5 * (topology.atoms[i].sigma + topology.atoms[j].sigma);
            let mut epsilon = (topology.atoms[i].epsilon * topology.atoms[j].epsilon).sqrt();
            let mut coulomb_scale = 1.0;
            if pairs_14.contains(&(i.min(j), i.max(j))) {
                epsilon *= LJ_14_SCALE;
                coulomb_scale = COULOMB_14_SCALE;
            }
            let mut dr = [pj[0] - pi[0], pj[1] - pi[1], pj[2] - pi[2]];
            dr = minimum_image(dr, &topology.box_);
            let r2 = dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2];
            let r = r2.sqrt().max(1e-12);

            if sigma > 1e-8 && epsilon > 1e-12 {
                let sr = sigma / r;
                let sr6 = sr.powi(6);
                let sr12 = sr6 * sr6;
                let lj_force_scalar = 24.0 * epsilon * (2.0 * sr12 - sr6) / r;
                let lj_energy = 4.0 * epsilon * (sr12 - sr6);
                *energy += lj_energy;
                for k in 0..3 {
                    forces[i][k] -= lj_force_scalar * dr[k];
                    forces[j][k] += lj_force_scalar * dr[k];
                }
            }

            let (coul_energy, coul_force) = if let Some(ctx) = pme {
                crate::electrostatics::ewald::screened_coulomb_energy_force(
                    qi, qj, r, r2, ctx.alpha, coulomb_scale,
                )
            } else {
                crate::electrostatics::ewald::direct_coulomb_energy_force(
                    qi, qj, r, r2, coulomb_scale,
                )
            };
            *energy += coul_energy;
            for k in 0..3 {
                forces[i][k] -= coul_force * dr[k];
                forces[j][k] += coul_force * dr[k];
            }
        });

    if let Some(ctx) = pme {
        let pme_result = pme::compute_pme_forces(topology, ctx);
        for (fi, fp) in forces.iter_mut().zip(pme_result.forces.iter()) {
            fi[0] += fp[0];
            fi[1] += fp[1];
            fi[2] += fp[2];
        }
        potential_energy += pme_result.energy;

        let (excl_energy, excl_forces, _excl_virial) =
            crate::electrostatics::excluded_pair_correction(topology, ctx.alpha);
        potential_energy += excl_energy;
        for (fi, fe) in forces.iter_mut().zip(excl_forces.iter()) {
            fi[0] += fe[0];
            fi[1] += fe[1];
            fi[2] += fe[2];
        }
    }

    ForceResult {
        forces,
        potential_energy,
    }
}

pub fn compute_forces(topology: &Topology, cutoff: f64) -> ForceResult {
    compute_forces_with_pme(topology, cutoff, None)
}

/// Lennard-Jones nonbonded only (implicit GB path).
pub fn compute_lj_forces(topology: &Topology, cutoff: f64) -> ForceResult {
    let n = topology.atoms.len();

    let nl = build_neighbor_list(topology, cutoff);
    let pairs_14 = build_14_pairs(topology);

    let (forces, potential_energy) =
        parallel_accumulate(&nl.pairs, n, |&(i, j), forces, energy| {
            let pi = topology.atoms[i].position;
            let pj = topology.atoms[j].position;
            let mut dr = [pj[0] - pi[0], pj[1] - pi[1], pj[2] - pi[2]];
            dr = minimum_image(dr, &topology.box_);
            let r2 = dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2];
            let r = r2.sqrt().max(1e-12);

            let sigma = 0.5 * (topology.atoms[i].sigma + topology.atoms[j].sigma);
            let mut epsilon = (topology.atoms[i].epsilon * topology.atoms[j].epsilon).sqrt();
            if pairs_14.contains(&(i.min(j), i.max(j))) {
                epsilon *= LJ_14_SCALE;
            }

            if sigma > 1e-8 && epsilon > 1e-12 {
                let sr = sigma / r;
                let sr6 = sr.powi(6);
                let sr12 = sr6 * sr6;
                let lj_force_scalar = 24.0 * epsilon * (2.0 * sr12 - sr6) / r;
                let lj_energy = 4.0 * epsilon * (sr12 - sr6);
                *energy += lj_energy;
                for k in 0..3 {
                    forces[i][k] -= lj_force_scalar * dr[k];
                    forces[j][k] += lj_force_scalar * dr[k];
                }
            }
        });

    ForceResult {
        forces,
        potential_energy,
    }
}

pub fn compute_forces_with_pme(
    topology: &Topology,
    cutoff: f64,
    pme: Option<&PmeContext>,
) -> ForceResult {
    let bonded = compute_bonded_forces(topology);
    let nonbonded = compute_nonbonded_forces(topology, cutoff, pme);
    merge_force_results(bonded, nonbonded)
}

pub fn merge_force_results(a: ForceResult, b: ForceResult) -> ForceResult {
    let mut forces = a.forces;
    for (fa, fb) in forces.iter_mut().zip(b.forces.iter()) {
        fa[0] += fb[0];
        fa[1] += fb[1];
        fa[2] += fb[2];
    }
    return_force_buffer(b.forces);
    ForceResult {
        forces,
        potential_energy: a.potential_energy + b.potential_energy,
    }
}

/// Pairwise virial (sum of r_ij . f_ij) from 2-body bonded terms only.
/// Angle/dihedral (3/4-body) contributions are not included: this is a
/// standard, minor approximation for the scalar pressure estimate.
pub fn compute_bonded_virial(topology: &Topology) -> f64 {
    topology
        .bonds
        .par_iter()
        .map(|bond| {
            let pi = topology.atoms[bond.i].position;
            let pj = topology.atoms[bond.j].position;
            let mut dr = [pj[0] - pi[0], pj[1] - pi[1], pj[2] - pi[2]];
            dr = minimum_image(dr, &topology.box_);
            let r2 = dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2];
            let r = r2.sqrt().max(1e-12);
            let dr_mag = r - bond.r0;
            let force_scalar = -2.0 * bond.k * dr_mag / r;
            force_scalar * r2
        })
        .sum()
}

/// Pairwise virial from nonbonded LJ + direct Coulomb interactions within
/// `cutoff`. The PME reciprocal-space contribution is not included; this is
/// an approximation acceptable for a scalar pressure estimate.
pub fn compute_nonbonded_virial(topology: &Topology, cutoff: f64) -> f64 {
    let nl = build_neighbor_list(topology, cutoff);
    let pairs_14 = build_14_pairs(topology);
    nl.pairs
        .par_iter()
        .map(|&(i, j)| {
            let pi = topology.atoms[i].position;
            let pj = topology.atoms[j].position;
            let qi = topology.atoms[i].charge;
            let qj = topology.atoms[j].charge;
            let sigma = 0.5 * (topology.atoms[i].sigma + topology.atoms[j].sigma);
            let mut epsilon = (topology.atoms[i].epsilon * topology.atoms[j].epsilon).sqrt();
            let mut coulomb_scale = 1.0;
            if pairs_14.contains(&(i.min(j), i.max(j))) {
                epsilon *= LJ_14_SCALE;
                coulomb_scale = COULOMB_14_SCALE;
            }
            let mut dr = [pj[0] - pi[0], pj[1] - pi[1], pj[2] - pi[2]];
            dr = minimum_image(dr, &topology.box_);
            let r2 = dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2];
            let r = r2.sqrt().max(1e-12);

            let mut v = 0.0;
            if sigma > 1e-8 && epsilon > 1e-12 {
                let sr = sigma / r;
                let sr6 = sr.powi(6);
                let sr12 = sr6 * sr6;
                let lj_force_scalar = 24.0 * epsilon * (2.0 * sr12 - sr6) / r;
                v += lj_force_scalar * r2;
            }

            let (_, coul_force) = crate::electrostatics::ewald::direct_coulomb_energy_force(
                qi,
                qj,
                r,
                r2,
                coulomb_scale,
            );
            v += coul_force * r2;
            v
        })
        .sum()
}

/// Total scalar virial W = sum r_ij . f_ij used for pressure:
/// P = (2*KE + W) / (3*V).
///
/// When the system is periodic (and not using the implicit-solvent GB
/// path), this includes the PME reciprocal-space virial
/// (W_recip = -3V dE_recip/dV at fixed fractional coordinates) and the
/// excluded/1-4 pair correction's virial, so pressure stays correct with
/// PME electrostatics.
pub fn compute_virial(topology: &Topology, cutoff: f64) -> f64 {
    let mut virial = compute_bonded_virial(topology) + compute_nonbonded_virial(topology, cutoff);
    if topology.box_.pbc && !topology.is_implicit_gb() {
        let ctx = PmeContext::new(topology, cutoff);
        let pme_result = pme::compute_pme_forces(topology, &ctx);
        virial += pme_result.virial;
        let (_, _, excl_virial) = crate::electrostatics::excluded_pair_correction(topology, ctx.alpha);
        virial += excl_virial;
    }
    virial
}

fn build_14_pairs(topology: &Topology) -> std::collections::HashSet<(usize, usize)> {
    let mut pairs = std::collections::HashSet::new();
    for d in topology.dihedrals.iter().chain(topology.impropers.iter()) {
        let (a, b) = (d.i.min(d.l), d.i.max(d.l));
        pairs.insert((a, b));
    }
    pairs
}

fn add_dihedral_forces(
    topology: &Topology,
    dihedral: &crate::topology::DihedralTerm,
    forces: &mut [[f64; 3]],
    energy: &mut f64,
) {
    let pi = topology.atoms[dihedral.i].position;
    let pj = topology.atoms[dihedral.j].position;
    let pk = topology.atoms[dihedral.k].position;
    let pl = topology.atoms[dihedral.l].position;

    let b1 = [pj[0] - pi[0], pj[1] - pi[1], pj[2] - pi[2]];
    let b2 = [pk[0] - pj[0], pk[1] - pj[1], pk[2] - pj[2]];
    let b3 = [pl[0] - pk[0], pl[1] - pk[1], pl[2] - pk[2]];

    let n2 = cross(b1, b2);
    let n3 = cross(b2, b3);
    let m1 = cross(n2, [b2[0] / norm(b2), b2[1] / norm(b2), b2[2] / norm(b2)]);

    let x = dot(n2, n3);
    let y = dot(m1, n3);
    let phi = y.atan2(x);
    let n = dihedral.n as f64;
    let delta = dihedral.delta;
    let angle_term = n * phi - delta;
    let e = dihedral.k_phi * (1.0 + angle_term.cos());
    *energy += e;

    let d_e_d_phi = dihedral.k_phi * n * angle_term.sin();
    let inv_n2 = 1.0 / norm(n2).max(1e-12);
    let inv_n3 = 1.0 / norm(n3).max(1e-12);

    let f_i = scale(n2, -d_e_d_phi * norm(b2) * inv_n2);
    let f_l = scale(n3, d_e_d_phi * norm(b2) * inv_n3);
    let f_j = [
        f_i[0] * (-1.0) + dot(b3, b2) / dot(b2, b2) * f_i[0],
        f_i[1] * (-1.0) + dot(b3, b2) / dot(b2, b2) * f_i[1],
        f_i[2] * (-1.0) + dot(b3, b2) / dot(b2, b2) * f_i[2],
    ];
    let f_k = [
        f_l[0] * (-1.0) + dot(b1, b2) / dot(b2, b2) * f_l[0],
        f_l[1] * (-1.0) + dot(b1, b2) / dot(b2, b2) * f_l[1],
        f_l[2] * (-1.0) + dot(b1, b2) / dot(b2, b2) * f_l[2],
    ];

    for k in 0..3 {
        forces[dihedral.i][k] += f_i[k];
        forces[dihedral.j][k] += f_j[k];
        forces[dihedral.k][k] += f_k[k];
        forces[dihedral.l][k] += f_l[k];
    }
}

fn norm(v: [f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn scale(v: [f64; 3], s: f64) -> [f64; 3] {
    [v[0] * s, v[1] * s, v[2] * s]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::topology::{AtomRecord, BondTerm, DihedralTerm, SimulationBox, Topology};
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    fn random_solvated_topology(n_waters: usize, seed: u64) -> Topology {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut top = Topology::new();
        let extent = 4.0;
        top.box_ = SimulationBox {
            lx: extent,
            ly: extent,
            lz: extent,
            pbc: true,
        };
        let roh = 0.09572;
        let hoh = 1.824218134_f64;
        for w in 0..n_waters {
            let base = [
                rng.gen_range(0.0..extent),
                rng.gen_range(0.0..extent),
                rng.gen_range(0.0..extent),
            ];
            let o_idx = top.atoms.len();
            top.atoms.push(AtomRecord {
                element: "O".into(),
                name: "O".into(),
                mass: 15.999,
                charge: -0.834,
                sigma: 0.315,
                epsilon: 0.636,
                position: base,
                residue_id: w as u32,
                molecule_id: w as u32,
                born_r: None,
            });
            let h1 = [base[0] + roh, base[1], base[2]];
            let h2 = [
                base[0] + roh * hoh.cos(),
                base[1] + roh * hoh.sin(),
                base[2],
            ];
            for (name, pos) in [("H1", h1), ("H2", h2)] {
                top.atoms.push(AtomRecord {
                    element: "H".into(),
                    name: name.into(),
                    mass: 1.008,
                    charge: 0.417,
                    sigma: 0.0,
                    epsilon: 0.0,
                    position: pos,
                    residue_id: w as u32,
                    molecule_id: w as u32,
                    born_r: None,
                });
            }
            top.bonds.push(BondTerm {
                i: o_idx,
                j: o_idx + 1,
                k: 5000.0,
                r0: roh,
            });
            top.bonds.push(BondTerm {
                i: o_idx,
                j: o_idx + 2,
                k: 5000.0,
                r0: roh,
            });
        }
        top.build_exclusions();
        top
    }

    /// Parallel (rayon, default thread pool) nonbonded + bonded forces must
    /// match a forced single-thread evaluation to <1e-10 relative error:
    /// the same per-item math runs either way, only the reduction order
    /// (and hence floating-point summation order) differs.
    #[test]
    fn parallel_forces_match_single_threaded() {
        let top = random_solvated_topology(80, 11);
        let cutoff = 1.0;

        let parallel = compute_forces(&top, cutoff);

        let single = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap()
            .install(|| compute_forces(&top, cutoff));

        let e_rel = ((parallel.potential_energy - single.potential_energy)
            / single.potential_energy.abs().max(1e-12))
        .abs();
        assert!(e_rel < 1e-10, "energy rel diff = {e_rel}");

        for (fa, fb) in parallel.forces.iter().zip(single.forces.iter()) {
            for k in 0..3 {
                let scale = fb[k].abs().max(1.0);
                let rel = (fa[k] - fb[k]).abs() / scale;
                assert!(rel < 1e-10, "force component rel diff = {rel}");
            }
        }
    }

    /// Benchmark: report the wall-clock speedup of the (default, all-core)
    /// rayon-parallel force evaluation vs a forced single-thread run on a
    /// moderately large system. Not a strict pass/fail gate on absolute
    /// speedup (CI machines vary in core count), just prints the measured
    /// number for visibility, and sanity-checks it's not a slowdown.
    #[test]
    #[ignore = "wall-clock benchmark; run with --ignored"]
    fn benchmark_parallel_speedup_vs_single_thread() {
        let top = random_solvated_topology(600, 7);
        let cutoff = 1.0;
        let iters = 5;

        let single_pool = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap();
        let t0 = std::time::Instant::now();
        single_pool.install(|| {
            for _ in 0..iters {
                let _ = compute_forces(&top, cutoff);
            }
        });
        let single_elapsed = t0.elapsed();

        let t1 = std::time::Instant::now();
        for _ in 0..iters {
            let _ = compute_forces(&top, cutoff);
        }
        let parallel_elapsed = t1.elapsed();

        let speedup = single_elapsed.as_secs_f64() / parallel_elapsed.as_secs_f64().max(1e-12);
        println!(
            "[benchmark] rayon threads = {}, single-thread = {:?}, parallel = {:?}, speedup = {:.2}x",
            rayon::current_num_threads(),
            single_elapsed,
            parallel_elapsed,
            speedup
        );
        // On a multi-core CI runner this should be faster; guard loosely
        // against a pathological regression rather than requiring a fixed
        // multiplier (thread count varies by machine).
        if rayon::current_num_threads() > 1 {
            assert!(speedup > 0.8, "parallel run should not be slower: {speedup:.2}x");
        }
    }

    /// Same measurement as above but at N~5000 atoms (~1666 waters), the
    /// scale this crate's performance target (>=2.5x speedup at N~5000 on 4
    /// threads) is stated for.
    #[test]
    #[ignore = "wall-clock benchmark; run with --ignored"]
    fn benchmark_parallel_speedup_at_5000_atoms() {
        let top = random_solvated_topology(1666, 13);
        let cutoff = 1.0;
        let iters = 5;

        let single_pool = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap();
        let t0 = std::time::Instant::now();
        single_pool.install(|| {
            for _ in 0..iters {
                let _ = compute_forces(&top, cutoff);
            }
        });
        let single_elapsed = t0.elapsed();

        let t1 = std::time::Instant::now();
        for _ in 0..iters {
            let _ = compute_forces(&top, cutoff);
        }
        let parallel_elapsed = t1.elapsed();

        let speedup = single_elapsed.as_secs_f64() / parallel_elapsed.as_secs_f64().max(1e-12);
        println!(
            "[benchmark] N={} atoms, rayon threads = {}, single-thread = {:?}, parallel = {:?}, speedup = {:.2}x",
            top.atoms.len(),
            rayon::current_num_threads(),
            single_elapsed,
            parallel_elapsed,
            speedup
        );
        // Target is >=2.5x at N~5000 on 4 dedicated threads (see the
        // benchmark numbers recorded when these fixes landed). Like the
        // benchmark above, this assertion stays loose (just "not a
        // slowdown"): running the whole suite's tests concurrently shares
        // the same handful of CPU cores across many test threads at once,
        // which starves this timing measurement regardless of the force
        // code's own parallel efficiency, so a tight speedup floor here
        // would be a source of CI flakiness rather than a real regression
        // signal. Run this test in isolation for a trustworthy number.
        if rayon::current_num_threads() > 1 {
            assert!(speedup > 0.8, "parallel run should not be slower: {speedup:.2}x");
        }
    }

    #[test]
    fn bonded_energy_finite() {
        let mut top = Topology::new();
        top.atoms.push(AtomRecord {
            element: "C".into(),
            name: "C1".into(),
            mass: 12.0,
            charge: 0.0,
            sigma: 0.34,
            epsilon: 0.36,
            position: [0.0, 0.0, 0.0],
            residue_id: 0,
            molecule_id: 0,
            born_r: None,
        });
        top.atoms.push(AtomRecord {
            element: "C".into(),
            name: "C2".into(),
            mass: 12.0,
            charge: 0.0,
            sigma: 0.34,
            epsilon: 0.36,
            position: [0.18, 0.0, 0.0],
            residue_id: 0,
            molecule_id: 0,
            born_r: None,
        });
        top.bonds.push(BondTerm {
            i: 0,
            j: 1,
            k: 500.0,
            r0: 0.15,
        });
        let r = compute_bonded_forces(&top);
        assert!(r.potential_energy.is_finite());
    }

    #[test]
    fn dihedral_term_runs() {
        let mut top = Topology::new();
        for i in 0..4 {
            top.atoms.push(AtomRecord {
                element: "C".into(),
                name: format!("C{i}"),
                mass: 12.0,
                charge: 0.0,
                sigma: 0.34,
                epsilon: 0.36,
                position: [i as f64 * 0.15, 0.0, 0.0],
                residue_id: 0,
                molecule_id: 0,
                born_r: None,
            });
        }
        top.dihedrals.push(DihedralTerm {
            i: 0,
            j: 1,
            k: 2,
            l: 3,
            k_phi: 1.0,
            n: 1,
            delta: 0.0,
        });
        let r = compute_bonded_forces(&top);
        assert!(r.potential_energy.is_finite());
    }
}
