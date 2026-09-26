pub mod ewald;
pub mod gb;
pub mod pme;

use crate::topology::Topology;

const COULOMB_CONSTANT: f64 = 138.935456;

/// Ewald self-energy and neutralizing-background correction (CPU
/// reference), using the same Ewald splitting parameter `alpha` as the
/// paired real-space (erfc) and reciprocal-space (PME) terms:
///
///   E_self = -ke * alpha/sqrt(pi) * sum(q_i^2)
///   E_bg   = -ke * pi / (2 * alpha^2 * V) * (sum q_i)^2
///
/// `E_bg` only matters for non-neutral systems (it cancels the divergent
/// k=0 term implicitly assumed by the reciprocal sum); for a neutral
/// system it vanishes exactly.
pub fn ewald_energy_correction_with_alpha(topology: &Topology, alpha: f64) -> f64 {
    if !topology.box_.pbc {
        return 0.0;
    }
    let volume = topology.box_.lx * topology.box_.ly * topology.box_.lz;
    if volume <= 0.0 {
        return 0.0;
    }
    let q_sum: f64 = topology.atoms.iter().map(|a| a.charge).sum();
    let q_sq_sum: f64 = topology.atoms.iter().map(|a| a.charge * a.charge).sum();
    let self_energy = -COULOMB_CONSTANT * alpha / std::f64::consts::PI.sqrt() * q_sq_sum;
    let background =
        -COULOMB_CONSTANT * std::f64::consts::PI / (2.0 * alpha * alpha * volume) * q_sum * q_sum;
    self_energy + background
}

/// Convenience wrapper that derives `alpha` from `cutoff` the same way
/// `pme::PmeContext::new` does, so callers that only have a cutoff (not an
/// existing `PmeContext`) stay consistent with the reciprocal-space term.
pub fn ewald_energy_correction_for_cutoff(topology: &Topology, cutoff: f64) -> f64 {
    let alpha = pme::ewald_alpha_from_rtol(cutoff, pme::DEFAULT_EWALD_RTOL);
    ewald_energy_correction_with_alpha(topology, alpha)
}

/// Backwards-compatible entry point using the default real-space cutoff of
/// 1.0 nm (matches historical callers that did not have a cutoff at hand).
pub fn ewald_energy_correction(topology: &Topology) -> f64 {
    ewald_energy_correction_for_cutoff(topology, 1.0)
}

/// Excluded-pair and scaled-1-4 correction for PME: the reciprocal sum
/// implicitly includes the full q_i*q_j/r interaction for *every* pair,
/// including bonded exclusions and 1-4 pairs whose direct-space term was
/// zeroed out or scaled. This subtracts the erf(alpha*r)/r component the
/// reciprocal sum contributed for those pairs, times (1 - scale), so the
/// net pairwise interaction matches what the force field requires
/// (0 for full exclusions, `scale` * q_i*q_j/r for 1-4 pairs).
pub fn excluded_pair_correction(
    topology: &Topology,
    alpha: f64,
) -> (f64, Vec<[f64; 3]>, f64) {
    let n = topology.atoms.len();
    let mut forces = vec![[0.0f64; 3]; n];
    let mut energy = 0.0f64;
    let mut virial = 0.0f64;
    if !topology.box_.pbc {
        return (energy, forces, virial);
    }

    let mut pairs: Vec<(usize, usize, f64)> = Vec::new();
    for e in &topology.exclusions {
        pairs.push((e[0], e[1], 0.0));
    }
    let coulomb_14_scale = 1.0 / 1.2;
    let mut seen14 = std::collections::HashSet::new();
    for d in topology.dihedrals.iter().chain(topology.impropers.iter()) {
        let (a, b) = (d.i.min(d.l), d.i.max(d.l));
        if seen14.insert((a, b)) {
            pairs.push((a, b, coulomb_14_scale));
        }
    }

    for (i, j, scale) in pairs {
        if i >= n || j >= n || i == j {
            continue;
        }
        let qi = topology.atoms[i].charge;
        let qj = topology.atoms[j].charge;
        let one_minus_scale = 1.0 - scale;
        if qi == 0.0 || qj == 0.0 || one_minus_scale == 0.0 {
            continue;
        }
        let pi_pos = topology.atoms[i].position;
        let pj_pos = topology.atoms[j].position;
        let mut dr = [
            pj_pos[0] - pi_pos[0],
            pj_pos[1] - pi_pos[1],
            pj_pos[2] - pi_pos[2],
        ];
        dr = crate::neighbor::minimum_image(dr, &topology.box_);
        let r2 = dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2];
        let r = r2.sqrt().max(1e-12);
        let arg = alpha * r;
        let erf_val = 1.0 - ewald::erfc(arg);
        let pref = COULOMB_CONSTANT * qi * qj * one_minus_scale;
        let e = -pref * erf_val / r;
        energy += e;
        // force_scalar = -dE/dr, matching the sign convention used by
        // `ewald::screened_coulomb_energy_force` / `direct_coulomb_energy_force`.
        let derf_dr = (2.0 * alpha / std::f64::consts::PI.sqrt()) * (-arg * arg).exp();
        let force_scalar = pref * (derf_dr / r - erf_val / r2) / r;
        for k in 0..3 {
            forces[i][k] -= force_scalar * dr[k];
            forces[j][k] += force_scalar * dr[k];
        }
        virial += force_scalar * r2;
    }

    (energy, forces, virial)
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
