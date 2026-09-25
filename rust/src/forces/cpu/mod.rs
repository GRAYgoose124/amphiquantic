use crate::electrostatics::pme::{self, PmeContext};
use crate::neighbor::{build_neighbor_list, minimum_image};
use crate::topology::Topology;

pub struct ForceResult {
    pub forces: Vec<[f64; 3]>,
    pub potential_energy: f64,
}

const COULOMB_CONSTANT: f64 = 138.935456;
const LJ_14_SCALE: f64 = 0.5;
const COULOMB_14_SCALE: f64 = 1.0 / 1.2;

pub fn compute_bonded_forces(topology: &Topology) -> ForceResult {
    let n = topology.atoms.len();
    let mut forces = vec![[0.0f64; 3]; n];
    let mut potential_energy = 0.0;

    for bond in &topology.bonds {
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
        potential_energy += bond.k * dr_mag * dr_mag;
    }

    for angle in &topology.angles {
        let pi = topology.atoms[angle.i].position;
        let pj = topology.atoms[angle.j].position;
        let pk = topology.atoms[angle.k].position;
        let b1 = [pi[0] - pj[0], pi[1] - pj[1], pi[2] - pj[2]];
        let b2 = [pk[0] - pj[0], pk[1] - pj[1], pk[2] - pj[2]];
        let n1 = norm(b1);
        let n2 = norm(b2);
        if n1 < 1e-12 || n2 < 1e-12 {
            continue;
        }
        let cos_theta = (dot(b1, b2) / (n1 * n2)).clamp(-1.0, 1.0);
        let theta = cos_theta.acos();
        let dtheta = theta - angle.theta0;
        let coeff = -2.0 * angle.k_theta * dtheta;
        potential_energy += angle.k_theta * dtheta * dtheta;
        let inv_n1 = 1.0 / n1;
        let inv_n2 = 1.0 / n2;
        for k in 0..3 {
            let d_cos_d_b1 = (b2[k] * inv_n2 - cos_theta * b1[k] * inv_n1) / n1;
            let d_cos_d_b2 = (b1[k] * inv_n1 - cos_theta * b2[k] * inv_n2) / n2;
            forces[angle.i][k] += coeff * d_cos_d_b1;
            forces[angle.j][k] += coeff * (-d_cos_d_b1 - d_cos_d_b2);
            forces[angle.k][k] += coeff * d_cos_d_b2;
        }
    }

    for dihedral in topology
        .dihedrals
        .iter()
        .chain(topology.impropers.iter())
    {
        add_dihedral_forces(topology, dihedral, &mut forces, &mut potential_energy);
    }

    ForceResult {
        forces,
        potential_energy,
    }
}

pub fn compute_nonbonded_forces(
    topology: &Topology,
    cutoff: f64,
    pme: Option<&PmeContext>,
) -> ForceResult {
    let n = topology.atoms.len();
    let mut forces = vec![[0.0f64; 3]; n];
    let mut potential_energy = 0.0;

    let nl = build_neighbor_list(topology, cutoff);
    let pairs_14 = build_14_pairs(topology);

    for &(i, j) in &nl.pairs {
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
            potential_energy += lj_energy;
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
            crate::electrostatics::ewald::direct_coulomb_energy_force(qi, qj, r, r2, coulomb_scale)
        };
        potential_energy += coul_energy;
        for k in 0..3 {
            forces[i][k] -= coul_force * dr[k];
            forces[j][k] += coul_force * dr[k];
        }
    }

    if let Some(ctx) = pme {
        let pme_result = pme::compute_pme_forces(topology, ctx);
        for (fi, fp) in forces.iter_mut().zip(pme_result.forces.iter()) {
            fi[0] += fp[0];
            fi[1] += fp[1];
            fi[2] += fp[2];
        }
        potential_energy += pme_result.energy;
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
    let mut forces = vec![[0.0f64; 3]; n];
    let mut potential_energy = 0.0;

    let nl = build_neighbor_list(topology, cutoff);
    let pairs_14 = build_14_pairs(topology);

    for &(i, j) in &nl.pairs {
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
            potential_energy += lj_energy;
            for k in 0..3 {
                forces[i][k] -= lj_force_scalar * dr[k];
                forces[j][k] += lj_force_scalar * dr[k];
            }
        }
    }

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
    ForceResult {
        forces,
        potential_energy: a.potential_energy + b.potential_energy,
    }
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
    use crate::topology::{AtomRecord, BondTerm, DihedralTerm, Topology};

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
