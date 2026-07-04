use crate::constraints::{apply_constraints, apply_rattle, build_constraints, ConstraintSet};
use crate::electrostatics::apply_pbc;
use crate::forces::{backend_from_env, compute_forces, ForceBackend};
use crate::neighbor::NeighborListManager;
use crate::topology::Topology;
use rand::Rng;

pub struct Restraint {
    pub atom: usize,
    pub k: f64,
    pub position: [f64; 3],
}

pub struct MdState {
    pub topology: Topology,
    pub velocities: Vec<[f64; 3]>,
    pub temperature: f64,
    pub timestep: f64,
    pub friction: f64,
    pub pressure: f64,
    pub step_count: u64,
    pub constraints: Option<ConstraintSet>,
    pub use_constraints: bool,
    pub npt: bool,
    pub restraints: Vec<Restraint>,
    pub neighbor_manager: NeighborListManager,
    pub last_virial: f64,
}

pub struct MdStepResult {
    pub potential_energy: f64,
    pub kinetic_energy: f64,
    pub temperature: f64,
}

impl MdState {
    pub fn new(topology: Topology, temperature: f64) -> Self {
        let constraints = build_constraints(&topology);
        Self {
            velocities: initialize_velocities(&topology, temperature),
            neighbor_manager: NeighborListManager::new(1.0, 0.2),
            topology,
            temperature,
            timestep: 0.002,
            friction: 10.0,
            pressure: 1.0,
            step_count: 0,
            constraints: Some(constraints),
            use_constraints: true,
            npt: false,
            restraints: Vec::new(),
            last_virial: 0.0,
        }
    }
}

pub fn initialize_velocities(topology: &Topology, temperature: f64) -> Vec<[f64; 3]> {
    let mut rng = rand::thread_rng();
    let k_b = 0.008314462618;
    let mut velocities: Vec<[f64; 3]> = topology
        .atoms
        .iter()
        .map(|atom| {
            let scale = (k_b * temperature / atom.mass).sqrt();
            let u1: f64 = rng.gen::<f64>().max(1e-12);
            let u2: f64 = rng.gen();
            let mag = scale * (-2.0 * u1.ln()).sqrt();
            let theta = 2.0 * std::f64::consts::PI * u2;
            [mag * theta.cos(), mag * theta.sin(), scale * rng.gen::<f64>()]
        })
        .collect();

    let total_mass: f64 = topology.atoms.iter().map(|a| a.mass).sum();
    let mut p = [0.0; 3];
    for (v, a) in velocities.iter().zip(topology.atoms.iter()) {
        for k in 0..3 {
            p[k] += a.mass * v[k];
        }
    }
    for v in velocities.iter_mut() {
        for k in 0..3 {
            v[k] -= p[k] / total_mass;
        }
    }
    velocities
}

fn compute_step_forces(state: &mut MdState, cutoff: f64) -> crate::forces::cpu::ForceResult {
    state.neighbor_manager.maybe_rebuild(&state.topology, cutoff);
    let use_pme = state.topology.box_.pbc && state.step_count % 2 == 0;
    let backend = backend_from_env();
    let mut result = match backend {
        ForceBackend::Hybrid => {
            crate::forces::hybrid::compute_forces_hybrid(&state.topology, cutoff, use_pme)
        }
        other => compute_forces(&state.topology, cutoff, other),
    };

    if !state.restraints.is_empty() {
        for r in &state.restraints {
            let pos = state.topology.atoms[r.atom].position;
            let dr = [
                pos[0] - r.position[0],
                pos[1] - r.position[1],
                pos[2] - r.position[2],
            ];
            let f = [-2.0 * r.k * dr[0], -2.0 * r.k * dr[1], -2.0 * r.k * dr[2]];
            result.forces[r.atom][0] += f[0];
            result.forces[r.atom][1] += f[1];
            result.forces[r.atom][2] += f[2];
            result.potential_energy += r.k * (dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2]);
        }
    }

    state.last_virial = compute_virial(&state.topology, &result.forces);
    result
}

fn compute_virial(topology: &Topology, forces: &[[f64; 3]]) -> f64 {
    let mut virial = 0.0;
    for bond in &topology.bonds {
        let i = bond.i;
        let j = bond.j;
        let pi = topology.atoms[i].position;
        let pj = topology.atoms[j].position;
        let dr = [pj[0] - pi[0], pj[1] - pi[1], pj[2] - pi[2]];
        virial += dr[0] * forces[i][0] + dr[1] * forces[i][1] + dr[2] * forces[i][2];
    }
    virial / 3.0
}

pub fn langevin_step(state: &mut MdState, cutoff: f64) -> MdStepResult {
    let forces = compute_step_forces(state, cutoff);
    let k_b = 0.008314462618;
    let mut rng = rand::thread_rng();
    let gamma = state.friction;
    let dt = state.timestep;

    for ((atom, vel), force) in state
        .topology
        .atoms
        .iter_mut()
        .zip(state.velocities.iter_mut())
        .zip(forces.forces.iter())
    {
        let c1 = (-gamma * dt).exp();
        let c2 = (1.0 - c1) / (gamma * atom.mass);
        let noise_scale = (k_b * state.temperature * (1.0 - c1 * c1) / atom.mass).sqrt();
        let noise = [rng.gen::<f64>(), rng.gen::<f64>(), rng.gen::<f64>()];
        vel[0] = c1 * vel[0] + c2 * force[0] + noise[0] * noise_scale;
        vel[1] = c1 * vel[1] + c2 * force[1] + noise[1] * noise_scale;
        vel[2] = c1 * vel[2] + c2 * force[2] + noise[2] * noise_scale;
        atom.position[0] += dt * vel[0];
        atom.position[1] += dt * vel[1];
        atom.position[2] += dt * vel[2];
        if state.topology.box_.pbc {
            apply_pbc(&mut atom.position, &state.topology.box_);
        }
    }

    if state.use_constraints {
        if let Some(ref c) = state.constraints.clone() {
            apply_constraints(&mut state.topology, c);
            apply_rattle(&state.topology, &mut state.velocities, c);
        }
    }

    if state.npt {
        apply_berendsen_barostat(state);
    }

    state.step_count += 1;
    kinetic_stats(state, forces.potential_energy)
}

pub fn velocity_verlet_step(state: &mut MdState, cutoff: f64) -> MdStepResult {
    let forces = compute_step_forces(state, cutoff);

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
        if state.topology.box_.pbc {
            apply_pbc(&mut atom.position, &state.topology.box_);
        }
    }

    if state.use_constraints {
        if let Some(ref c) = state.constraints.clone() {
            apply_constraints(&mut state.topology, c);
        }
    }

    let forces2 = compute_step_forces(state, cutoff);
    for (vel, (atom, force)) in state
        .velocities
        .iter_mut()
        .zip(state.topology.atoms.iter().zip(forces2.forces.iter()))
    {
        vel[0] += 0.5 * state.timestep * force[0] / atom.mass;
        vel[1] += 0.5 * state.timestep * force[1] / atom.mass;
        vel[2] += 0.5 * state.timestep * force[2] / atom.mass;
    }

    if state.use_constraints {
        if let Some(ref c) = state.constraints.clone() {
            apply_rattle(&state.topology, &mut state.velocities, c);
        }
    }

    if state.npt {
        apply_berendsen_barostat(state);
    }

    state.step_count += 1;
    kinetic_stats(state, forces2.potential_energy)
}

fn apply_berendsen_barostat(state: &mut MdState) {
    let box_ = &mut state.topology.box_;
    if !box_.pbc || box_.lx <= 0.0 {
        return;
    }
    let volume = box_.lx * box_.ly * box_.lz;
    let k_b = 0.008314462618;
    let dof = (3 * state.topology.atoms.len()).max(1) as f64;
    let kinetic: f64 = state
        .velocities
        .iter()
        .zip(state.topology.atoms.iter())
        .map(|(v, a)| 0.5 * a.mass * (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]))
        .sum();
    let temp = 2.0 * kinetic / (dof * k_b);
    let ideal = dof * k_b * temp / volume;
    let pressure = ideal + state.last_virial / volume;
    let tau = 1000.0;
    let compressibility = 4.5e-5;
    let scale = 1.0
        - compressibility * (state.pressure - pressure) * state.timestep / tau;
    let scale = scale.clamp(0.995, 1.005);
    box_.lx *= scale;
    box_.ly *= scale;
    box_.lz *= scale;
    for atom in state.topology.atoms.iter_mut() {
        atom.position[0] *= scale;
        atom.position[1] *= scale;
        atom.position[2] *= scale;
    }
}

fn kinetic_stats(state: &MdState, potential_energy: f64) -> MdStepResult {
    let k_b = 0.008314462618;
    let kinetic: f64 = state
        .velocities
        .iter()
        .zip(state.topology.atoms.iter())
        .map(|(v, a)| 0.5 * a.mass * (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]))
        .sum();
    let dof = (3 * state.topology.atoms.len()).max(1) as f64;
    MdStepResult {
        potential_energy,
        kinetic_energy: kinetic,
        temperature: 2.0 * kinetic / (dof * k_b),
    }
}
