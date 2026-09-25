use crate::topology::Topology;

pub struct PmeResult {
    pub forces: Vec<[f64; 3]>,
    pub energy: f64,
}

pub struct PmeContext {
    pub alpha: f64,
    pub grid_size: [usize; 3],
    pub cutoff: f64,
}

impl PmeContext {
    pub fn new(_topology: &Topology, cutoff: f64) -> Self {
        Self {
            alpha: 0.34,
            grid_size: [32, 32, 32],
            cutoff,
        }
    }
}

/// Reciprocal-space Ewald/PME-style sum (CPU reference; direct k-sum).
pub fn compute_pme_forces(topology: &Topology, ctx: &PmeContext) -> PmeResult {
    let n = topology.atoms.len();
    if n == 0 || !topology.box_.pbc {
        return PmeResult {
            forces: vec![],
            energy: 0.0,
        };
    }

    let [kx_max, ky_max, kz_max] = ctx.grid_size;
    let volume = topology.box_.lx * topology.box_.ly * topology.box_.lz;
    let prefactor = 138.935456;
    let two_pi = 2.0 * std::f64::consts::PI;
    let alpha2 = ctx.alpha * ctx.alpha;
    let mut energy = 0.0f64;
    let mut forces = vec![[0.0f64; 3]; n];

    for ix in 0..kx_max {
        for iy in 0..ky_max {
            for iz in 0..kz_max {
                if ix == 0 && iy == 0 && iz == 0 {
                    continue;
                }
                let mx = if ix <= kx_max / 2 {
                    ix as f64
                } else {
                    ix as f64 - kx_max as f64
                };
                let my = if iy <= ky_max / 2 {
                    iy as f64
                } else {
                    iy as f64 - ky_max as f64
                };
                let mz = if iz <= kz_max / 2 {
                    iz as f64
                } else {
                    iz as f64 - kz_max as f64
                };
                let kx = two_pi * mx / topology.box_.lx;
                let ky = two_pi * my / topology.box_.ly;
                let kz = two_pi * mz / topology.box_.lz;
                let k2 = kx * kx + ky * ky + kz * kz;
                if k2 < 1e-12 {
                    continue;
                }
                let exp_term = (-k2 / (4.0 * alpha2)).exp() / k2;

                let mut sk = 0.0;
                let mut ck = 0.0;
                for atom in &topology.atoms {
                    let phase = kx * atom.position[0]
                        + ky * atom.position[1]
                        + kz * atom.position[2];
                    sk += atom.charge * phase.sin();
                    ck += atom.charge * phase.cos();
                }
                let qk2 = sk * sk + ck * ck;
                energy += prefactor * 0.5 * exp_term * qk2 / volume;

                let coeff = prefactor * exp_term / volume;
                for (ai, atom) in topology.atoms.iter().enumerate() {
                    let phase = kx * atom.position[0]
                        + ky * atom.position[1]
                        + kz * atom.position[2];
                    let s = phase.sin();
                    let c = phase.cos();
                    let q = atom.charge;
                    forces[ai][0] += coeff * q * kx * s;
                    forces[ai][1] += coeff * q * ky * s;
                    forces[ai][2] += coeff * q * kz * s;
                    let _ = c;
                }
            }
        }
    }

    PmeResult { forces, energy }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::topology::{AtomRecord, SimulationBox, Topology};

    #[test]
    fn ion_pair_finite() {
        let mut top = Topology::new();
        top.box_ = SimulationBox {
            lx: 3.0,
            ly: 3.0,
            lz: 3.0,
            pbc: true,
        };
        top.atoms.push(AtomRecord {
            element: "Na".into(),
            name: "Na".into(),
            mass: 22.99,
            charge: 1.0,
            sigma: 0.25,
            epsilon: 0.01,
            position: [0.5, 0.5, 0.5],
            residue_id: 0,
            molecule_id: 0,
            born_r: None,
        });
        top.atoms.push(AtomRecord {
            element: "Cl".into(),
            name: "Cl".into(),
            mass: 35.45,
            charge: -1.0,
            sigma: 0.35,
            epsilon: 0.01,
            position: [2.0, 2.0, 2.0],
            residue_id: 0,
            molecule_id: 0,
            born_r: None,
        });
        let ctx = PmeContext::new(&top, 1.0);
        let r = compute_pme_forces(&top, &ctx);
        assert!(r.energy.is_finite());
        assert_eq!(r.forces.len(), 2);
    }
}
