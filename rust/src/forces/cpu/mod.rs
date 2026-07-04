use crate::neighbor::{build_neighbor_list, minimum_image};
use crate::topology::Topology;

pub struct ForceResult {
    pub forces: Vec<[f64; 3]>,
    pub potential_energy: f64,
}

pub fn compute_forces(topology: &Topology, cutoff: f64) -> ForceResult {
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
        let cos_theta = dot(b1, b2) / (n1 * n2).clamp(-1.0, 1.0);
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

    let nl = build_neighbor_list(topology, cutoff);
    let coulomb_constant = 138.935456;
    for &(i, j) in &nl.pairs {
        let pi = topology.atoms[i].position;
        let pj = topology.atoms[j].position;
        let qi = topology.atoms[i].charge;
        let qj = topology.atoms[j].charge;
        let sigma = 0.5 * (topology.atoms[i].sigma + topology.atoms[j].sigma);
        let epsilon = (topology.atoms[i].epsilon * topology.atoms[j].epsilon).sqrt();
        let mut dr = [pj[0] - pi[0], pj[1] - pi[1], pj[2] - pi[2]];
        dr = minimum_image(dr, &topology.box_);
        let r2 = dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2];
        let r = r2.sqrt().max(1e-12);
        let sr = sigma / r;
        let sr6 = sr * sr * sr * sr * sr * sr;
        let sr12 = sr6 * sr6;
        let lj_force_scalar = 24.0 * epsilon * (2.0 * sr12 - sr6) / r;
        let lj_energy = 4.0 * epsilon * (sr12 - sr6);
        let coulomb_force = coulomb_constant * qi * qj / r2;
        let coulomb_energy = coulomb_constant * qi * qj / r;
        let force_scalar = lj_force_scalar + coulomb_force;
        for k in 0..3 {
            forces[i][k] -= force_scalar * dr[k];
            forces[j][k] += force_scalar * dr[k];
        }
        potential_energy += lj_energy + coulomb_energy;
    }

    ForceResult {
        forces,
        potential_energy,
    }
}

fn norm(v: [f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
