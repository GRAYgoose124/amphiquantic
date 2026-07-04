pub mod ewald;
pub mod pme;

use crate::topology::Topology;

/// Ewald self-energy and neutralizing background correction (CPU reference).
pub fn ewald_energy_correction(topology: &Topology) -> f64 {
    if !topology.box_.pbc {
        return 0.0;
    }
    let volume = topology.box_.lx * topology.box_.ly * topology.box_.lz;
    if volume <= 0.0 {
        return 0.0;
    }
    let alpha = 0.34;
    let prefactor = -alpha / (2.0 * std::f64::consts::PI.sqrt() * volume.sqrt());
    let q_sum: f64 = topology.atoms.iter().map(|a| a.charge).sum();
    let q_sq_sum: f64 = topology.atoms.iter().map(|a| a.charge * a.charge).sum();
    prefactor * (q_sq_sum - q_sum * q_sum)
}

pub fn apply_pbc(pos: &mut [f64; 3], box_: &crate::topology::SimulationBox) {
    if !box_.pbc {
        return;
    }
    let lengths = [box_.lx, box_.ly, box_.lz];
    for (p, l) in pos.iter_mut().zip(lengths.iter()) {
        if *l <= 0.0 {
            continue;
        }
        *p -= (*p / l).floor() * l;
    }
}

pub fn pme_reciprocal_energy(topology: &Topology) -> f64 {
    if !topology.box_.pbc {
        return 0.0;
    }
    let ctx = pme::PmeContext::new(topology, 1.0);
    pme::compute_pme_forces(topology, &ctx).energy
}
