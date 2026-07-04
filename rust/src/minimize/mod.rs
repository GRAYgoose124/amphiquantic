use crate::electrostatics::ewald_energy_correction;
use crate::forces::{backend_from_env, compute_forces};
use crate::topology::Topology;

pub struct MinimizeResult {
    pub topology: Topology,
    pub final_energy: f64,
    pub iterations: usize,
}

pub fn minimize(
    mut topology: Topology,
    cutoff: f64,
    max_steps: usize,
    step_size: f64,
    tolerance: f64,
) -> MinimizeResult {
    let backend = backend_from_env();
    let mut final_energy = 0.0;
    let mut iterations = 0;

    for step in 0..max_steps {
        let result = compute_forces(&topology, cutoff, backend.clone());
        final_energy = result.potential_energy + ewald_energy_correction(&topology);
        iterations = step + 1;

        let max_force = result
            .forces
            .iter()
            .map(|f| (f[0] * f[0] + f[1] * f[1] + f[2] * f[2]).sqrt())
            .fold(0.0, f64::max);

        if max_force < tolerance {
            break;
        }

        for (atom, force) in topology.atoms.iter_mut().zip(result.forces.iter()) {
            atom.position[0] += step_size * force[0];
            atom.position[1] += step_size * force[1];
            atom.position[2] += step_size * force[2];
            if topology.box_.pbc {
                crate::electrostatics::apply_pbc(&mut atom.position, &topology.box_);
            }
        }
    }

    MinimizeResult {
        topology,
        final_energy,
        iterations,
    }
}
