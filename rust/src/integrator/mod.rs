use crate::forces::{backend_from_env, compute_forces};
use crate::topology::Topology;
use rand::Rng;

pub struct MdState {
    pub topology: Topology,
    pub velocities: Vec<[f64; 3]>,
    pub temperature: f64,
    pub timestep: f64,
    pub friction: f64,
}

pub struct MdStepResult {
    pub potential_energy: f64,
    pub kinetic_energy: f64,
    pub temperature: f64,
}

pub fn initialize_velocities(topology: &Topology, temperature: f64) -> Vec<[f64; 3]> {
    let mut rng = rand::thread_rng();
    let k_b = 0.008314462618;
    topology
        .atoms
        .iter()
        .map(|atom| {
            let scale = (k_b * temperature / atom.mass).sqrt();
            [
                rng.gen::<f64>() * scale,
                rng.gen::<f64>() * scale,
                rng.gen::<f64>() * scale,
            ]
        })
        .collect()
}

pub fn langevin_step(state: &mut MdState, cutoff: f64) -> MdStepResult {
    let backend = backend_from_env();
    let forces = compute_forces(&state.topology, cutoff, backend);
    let k_b = 0.008314462618;
    let mut rng = rand::thread_rng();

    for ((atom, vel), force) in state
        .topology
        .atoms
        .iter_mut()
        .zip(state.velocities.iter_mut())
        .zip(forces.forces.iter())
    {
        let noise = [rng.gen::<f64>(), rng.gen::<f64>(), rng.gen::<f64>()];
        let c1 = 1.0 - state.friction * state.timestep;
        let c2 = state.timestep / atom.mass;
        vel[0] = c1 * vel[0] + c2 * force[0]
            + noise[0] * (2.0 * k_b * state.temperature / atom.mass).sqrt();
        vel[1] = c1 * vel[1] + c2 * force[1]
            + noise[1] * (2.0 * k_b * state.temperature / atom.mass).sqrt();
        vel[2] = c1 * vel[2] + c2 * force[2]
            + noise[2] * (2.0 * k_b * state.temperature / atom.mass).sqrt();
        atom.position[0] += state.timestep * vel[0];
        atom.position[1] += state.timestep * vel[1];
        atom.position[2] += state.timestep * vel[2];
    }

    let kinetic: f64 = state
        .velocities
        .iter()
        .zip(state.topology.atoms.iter())
        .map(|(v, a)| 0.5 * a.mass * (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]))
        .sum();
    let dof = (3 * state.topology.atoms.len()).max(1) as f64;

    MdStepResult {
        potential_energy: forces.potential_energy,
        kinetic_energy: kinetic,
        temperature: 2.0 * kinetic / (dof * k_b),
    }
}

pub fn velocity_verlet_step(state: &mut MdState, cutoff: f64) -> MdStepResult {
    let backend = backend_from_env();
    let forces = compute_forces(&state.topology, cutoff, backend);

    for ((atom, vel), force) in state
        .topology
        .atoms
        .iter_mut()
        .zip(state.velocities.iter_mut())
        .zip(forces.forces.iter())
    {
        vel[0] += 0.5 * state.timestep * force[0] / atom.mass;
        vel[1] += 0.5 * state.timestep * force[1] / atom.mass;
        vel[2] += 0.5 * state.timestep * force[2] / atom.mass;
        atom.position[0] += state.timestep * vel[0];
        atom.position[1] += state.timestep * vel[1];
        atom.position[2] += state.timestep * vel[2];
    }

    let forces2 = compute_forces(&state.topology, cutoff, backend_from_env());
    for (vel, (atom, force)) in state
        .velocities
        .iter_mut()
        .zip(state.topology.atoms.iter().zip(forces2.forces.iter()))
    {
        vel[0] += 0.5 * state.timestep * force[0] / atom.mass;
        vel[1] += 0.5 * state.timestep * force[1] / atom.mass;
        vel[2] += 0.5 * state.timestep * force[2] / atom.mass;
    }

    let k_b = 0.008314462618;
    let kinetic: f64 = state
        .velocities
        .iter()
        .zip(state.topology.atoms.iter())
        .map(|(v, a)| 0.5 * a.mass * (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]))
        .sum();
    let dof = (3 * state.topology.atoms.len()).max(1) as f64;

    MdStepResult {
        potential_energy: forces2.potential_energy,
        kinetic_energy: kinetic,
        temperature: 2.0 * kinetic / (dof * k_b),
    }
}
