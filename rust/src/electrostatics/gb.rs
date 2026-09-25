//! Generalized Born implicit electrostatics (OBC-II effective radii).

use crate::topology::Topology;

const COULOMB_CONSTANT: f64 = 138.935456;
const EPS_SOLUTION: f64 = 1.0;

pub struct GbForceResult {
    pub forces: Vec<[f64; 3]>,
    pub potential_energy: f64,
}

pub fn gb_epsilon_factor(solvent_dielectric: f64) -> f64 {
    (1.0 / EPS_SOLUTION) - (1.0 / solvent_dielectric.max(1.0))
}

/// Pairwise GB energy and force scalar (kJ/mol/nm along separation vector).
fn gb_pair(
    r2: f64,
    ri: f64,
    rj: f64,
    qi: f64,
    qj: f64,
    eps_factor: f64,
) -> (f64, f64) {
    if r2 < 1e-12 {
        return (0.0, 0.0);
    }
    let ri = ri.max(0.05);
    let rj = rj.max(0.05);
    let exp_term = (-r2 / (4.0 * ri * rj)).exp();
    let v = ri * rj * exp_term;
    let f = (r2 + v).sqrt().max(1e-8);
    let energy = -0.5 * COULOMB_CONSTANT * eps_factor * qi * qj / f;

    let df_du = 0.5 * (1.0 - v / (4.0 * ri * rj)) / f;
    let d_e_du = 0.5 * COULOMB_CONSTANT * eps_factor * qi * qj / (f * f) * df_du;
    let force_scalar = d_e_du * 2.0 * r2.sqrt();
    (energy, force_scalar)
}

pub fn compute_gb_forces(topology: &Topology) -> GbForceResult {
    let n = topology.atoms.len();
    let mut forces = vec![[0.0f64; 3]; n];
    let mut potential_energy = 0.0;
    let eps_factor = gb_epsilon_factor(topology.solvent_dielectric());

    for i in 0..n {
        let qi = topology.atoms[i].charge;
        let ri = topology.effective_born_r(i);
        if qi.abs() > 1e-12 && ri > 0.0 {
            potential_energy += -0.5 * COULOMB_CONSTANT * eps_factor * qi * qi / ri;
        }
    }

    for i in 0..n {
        let pi = topology.atoms[i].position;
        let qi = topology.atoms[i].charge;
        let ri = topology.effective_born_r(i);
        for j in (i + 1)..n {
            let pj = topology.atoms[j].position;
            let qj = topology.atoms[j].charge;
            let rj = topology.effective_born_r(j);
            let dr = [pj[0] - pi[0], pj[1] - pi[1], pj[2] - pi[2]];
            let r2 = dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2];
            let (e, f_scalar) = gb_pair(r2, ri, rj, qi, qj, eps_factor);
            potential_energy += e;
            if r2 > 1e-12 {
                let inv_r = 1.0 / r2.sqrt();
                for k in 0..3 {
                    let f = f_scalar * dr[k] * inv_r;
                    forces[i][k] += f;
                    forces[j][k] -= f;
                }
            }
        }
    }

    GbForceResult {
        forces,
        potential_energy,
    }
}
