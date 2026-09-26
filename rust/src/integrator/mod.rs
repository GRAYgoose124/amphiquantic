use crate::constraints::{
    apply_constraints, apply_rattle, apply_settle_velocity, build_constraints,
    compute_constraint_virial, ConstraintSet,
};
use crate::electrostatics::apply_pbc;
use crate::forces::{backend_from_env, compute_forces, ForceBackend};
use crate::neighbor::NeighborListManager;
use crate::topology::Topology;
use rand::Rng;
use std::collections::HashMap;

const KB: f64 = 0.008314462618; // kJ/mol/K
/// 1 kJ/mol/nm^3 in bar.
const PRESSURE_CONV: f64 = 16.6054;

/// Thermostat selection for MD integration.
#[derive(Clone, Debug, PartialEq)]
pub enum Thermostat {
    /// No temperature coupling (NVE).
    None,
    /// Langevin dynamics (stochastic friction baked into the integrator).
    Langevin,
    /// Berendsen weak-coupling thermostat.
    Berendsen,
    /// Bussi-Donadio-Parrinello stochastic velocity rescaling (canonical).
    VRescale,
    /// Nose-Hoover chain thermostat, with configurable chain length.
    NoseHoover { chain_length: usize },
}

impl Default for Thermostat {
    fn default() -> Self {
        Thermostat::None
    }
}

pub fn parse_thermostat(name: &str, chain_length: usize) -> Result<Thermostat, String> {
    match name.to_lowercase().as_str() {
        "none" => Ok(Thermostat::None),
        "langevin" => Ok(Thermostat::Langevin),
        "berendsen" => Ok(Thermostat::Berendsen),
        "vrescale" | "v-rescale" | "bussi" => Ok(Thermostat::VRescale),
        "nose-hoover" | "nosehoover" | "nh" => Ok(Thermostat::NoseHoover {
            chain_length: chain_length.max(1),
        }),
        other => Err(format!(
            "unknown thermostat '{}': expected none|langevin|berendsen|vrescale|nose-hoover",
            other
        )),
    }
}

/// Barostat selection for NPT integration (isotropic only).
#[derive(Clone, Debug, PartialEq)]
pub enum Barostat {
    None,
    /// Berendsen weak-coupling barostat.
    Berendsen,
    /// Metropolis Monte Carlo volume moves (molecule-COM scaling when
    /// molecule information is available, else atomic scaling).
    MonteCarlo,
}

impl Default for Barostat {
    fn default() -> Self {
        Barostat::None
    }
}

pub fn parse_barostat(name: &str) -> Result<Barostat, String> {
    match name.to_lowercase().as_str() {
        "none" => Ok(Barostat::None),
        "berendsen" => Ok(Barostat::Berendsen),
        "montecarlo" | "monte-carlo" | "mc" => Ok(Barostat::MonteCarlo),
        other => Err(format!(
            "unknown barostat '{}': expected none|berendsen|montecarlo",
            other
        )),
    }
}

/// One Nose-Hoover chain thermostat variable.
#[derive(Clone)]
pub struct NhcVar {
    pub xi: f64,
    pub v_xi: f64,
    pub q: f64,
}

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
    /// Target pressure in bar (used by barostats).
    pub pressure: f64,
    pub step_count: u64,
    pub constraints: Option<ConstraintSet>,
    pub use_constraints: bool,
    pub npt: bool,
    pub restraints: Vec<Restraint>,
    pub neighbor_manager: NeighborListManager,
    pub last_virial: f64,
    pub last_pressure: f64,
    pub thermostat: Thermostat,
    pub barostat: Barostat,
    pub tau_t: f64,
    pub tau_p: f64,
    pub compressibility: f64,
    /// Steps between center-of-mass motion removal; 0 disables periodic
    /// removal (COM is still zeroed once at velocity initialization).
    pub com_remove_interval: u64,
    /// Steps between Monte Carlo barostat attempts.
    pub mc_interval: u64,
    pub mc_attempts: u64,
    pub mc_accepts: u64,
    pub nh_chain: Vec<NhcVar>,
}

pub struct MdStepResult {
    pub potential_energy: f64,
    pub kinetic_energy: f64,
    pub temperature: f64,
    /// Instantaneous pressure in bar (0 for non-periodic systems).
    pub pressure: f64,
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
            last_pressure: 0.0,
            thermostat: Thermostat::default(),
            barostat: Barostat::default(),
            tau_t: 1.0,
            tau_p: 2.0,
            compressibility: 4.5e-5,
            com_remove_interval: 0,
            mc_interval: 25,
            mc_attempts: 0,
            mc_accepts: 0,
            nh_chain: Vec::new(),
        }
    }

    /// Number of holonomic (SHAKE) bond-length constraints currently active.
    pub fn num_constraints(&self) -> usize {
        self.constraints
            .as_ref()
            .map(|c| c.shake_bonds.len() + 3 * c.waters.len())
            .unwrap_or(0)
    }

    /// Degrees of freedom used for temperature: 3N minus 3 (COM motion,
    /// always removed at least at initialization) minus the number of
    /// active bond-length constraints.
    pub fn dof(&self) -> f64 {
        let n = self.topology.atoms.len();
        ((3 * n) as f64 - 3.0 - self.num_constraints() as f64).max(1.0)
    }

    pub fn mc_acceptance_ratio(&self) -> f64 {
        if self.mc_attempts == 0 {
            0.0
        } else {
            self.mc_accepts as f64 / self.mc_attempts as f64
        }
    }
}

pub fn kinetic_energy(state: &MdState) -> f64 {
    state
        .velocities
        .iter()
        .zip(state.topology.atoms.iter())
        .map(|(v, a)| 0.5 * a.mass * (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]))
        .sum()
}

pub fn initialize_velocities(topology: &Topology, temperature: f64) -> Vec<[f64; 3]> {
    let mut rng = rand::thread_rng();
    let k_b = KB;
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

/// Zero the net linear momentum (center-of-mass motion removal).
pub fn remove_com_motion(state: &mut MdState) {
    let total_mass: f64 = state.topology.atoms.iter().map(|a| a.mass).sum();
    if total_mass <= 0.0 {
        return;
    }
    let mut p = [0.0; 3];
    for (v, a) in state.velocities.iter().zip(state.topology.atoms.iter()) {
        for k in 0..3 {
            p[k] += a.mass * v[k];
        }
    }
    for v in state.velocities.iter_mut() {
        for k in 0..3 {
            v[k] -= p[k] / total_mass;
        }
    }
}

fn maybe_remove_com_motion(state: &mut MdState) {
    if state.com_remove_interval > 0 && state.step_count % state.com_remove_interval == 0 {
        remove_com_motion(state);
    }
}

fn compute_step_forces(state: &mut MdState, cutoff: f64) -> crate::forces::cpu::ForceResult {
    state.neighbor_manager.maybe_rebuild(&state.topology, cutoff);
    let use_pme = state.topology.box_.pbc
        && !state.topology.is_implicit_gb()
        && state.step_count % 2 == 0;
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

    state.last_virial = crate::forces::cpu::compute_virial(&state.topology, cutoff);
    result
}

fn sample_normal(rng: &mut impl Rng) -> f64 {
    let u1: f64 = rng.gen::<f64>().max(1e-12);
    let u2: f64 = rng.gen();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// Berendsen weak-coupling thermostat (velocity rescale).
fn apply_berendsen_thermostat(state: &mut MdState) {
    let dof = state.dof();
    let ke = kinetic_energy(state);
    let current_t = 2.0 * ke / (dof * KB);
    if current_t <= 1e-8 {
        return;
    }
    let tau = state.tau_t.max(1e-6);
    let lambda = (1.0 + (state.timestep / tau) * (state.temperature / current_t - 1.0))
        .max(0.0)
        .sqrt();
    for v in state.velocities.iter_mut() {
        v[0] *= lambda;
        v[1] *= lambda;
        v[2] *= lambda;
    }
}

/// Bussi-Donadio-Parrinello stochastic velocity rescaling (canonical
/// sampling through velocity rescaling, "V-rescale").
fn apply_vrescale_thermostat(state: &mut MdState) {
    let dof = state.dof();
    let ke = kinetic_energy(state);
    if ke <= 1e-12 {
        return;
    }
    let mut rng = rand::thread_rng();
    let ke_target = 0.5 * dof * KB * state.temperature;
    let tau = state.tau_t.max(1e-6);
    let c = (-state.timestep / tau).exp();

    let r1 = sample_normal(&mut rng);
    let n_extra = (dof - 1.0).max(0.0).round() as usize;
    let mut sum_sq = 0.0;
    for _ in 0..n_extra {
        let g = sample_normal(&mut rng);
        sum_sq += g * g;
    }

    let ke_new = ke
        + (1.0 - c) * (ke_target * (sum_sq + r1 * r1) / dof - ke)
        + 2.0 * r1 * (c * (1.0 - c) * ke_target * ke / dof).sqrt();
    let ke_new = ke_new.max(1e-12);
    let scale = (ke_new / ke).sqrt();
    for v in state.velocities.iter_mut() {
        v[0] *= scale;
        v[1] *= scale;
        v[2] *= scale;
    }
}

fn ensure_nh_chain(state: &mut MdState, chain_length: usize) {
    if state.nh_chain.len() == chain_length {
        return;
    }
    let dof = state.dof();
    let kt = KB * state.temperature;
    let tau = state.tau_t.max(1e-6);
    let mut chain = Vec::with_capacity(chain_length);
    for i in 0..chain_length {
        let q = if i == 0 {
            dof * kt * tau * tau
        } else {
            kt * tau * tau
        };
        chain.push(NhcVar {
            xi: 0.0,
            v_xi: 0.0,
            q,
        });
    }
    state.nh_chain = chain;
}

/// Nose-Hoover chain thermostat (Martyna-Tuckerman-Klein Trotter-factorized
/// propagation), applied as a velocity-rescaling half-step.
fn apply_nose_hoover_thermostat(state: &mut MdState, chain_length: usize) {
    ensure_nh_chain(state, chain_length);
    let dof = state.dof();
    let kt = KB * state.temperature;
    let dt = state.timestep;
    let mut chain = std::mem::take(&mut state.nh_chain);
    let m = chain.len();
    if m == 0 {
        state.nh_chain = chain;
        return;
    }
    let mut kinetic2 = 2.0 * kinetic_energy(state);

    for k in (0..m).rev() {
        let g = if k == 0 {
            (kinetic2 - dof * kt) / chain[0].q
        } else {
            (chain[k - 1].q * chain[k - 1].v_xi * chain[k - 1].v_xi - kt) / chain[k].q
        };
        if k == m - 1 {
            chain[k].v_xi += g * dt * 0.5;
        } else {
            let factor = (-0.25 * dt * chain[k + 1].v_xi).exp();
            chain[k].v_xi = chain[k].v_xi * factor * factor + g * dt * 0.5 * factor;
        }
    }

    let scale = (-chain[0].v_xi * dt).exp();
    for v in state.velocities.iter_mut() {
        v[0] *= scale;
        v[1] *= scale;
        v[2] *= scale;
    }
    kinetic2 *= scale * scale;
    for c in chain.iter_mut() {
        c.xi += c.v_xi * dt;
    }

    for k in 0..m {
        let g = if k == 0 {
            (kinetic2 - dof * kt) / chain[0].q
        } else {
            (chain[k - 1].q * chain[k - 1].v_xi * chain[k - 1].v_xi - kt) / chain[k].q
        };
        if k == m - 1 {
            chain[k].v_xi += g * dt * 0.5;
        } else {
            let factor = (-0.25 * dt * chain[k + 1].v_xi).exp();
            chain[k].v_xi = chain[k].v_xi * factor * factor + g * dt * 0.5 * factor;
        }
    }

    state.nh_chain = chain;
}

/// Extended-system conserved quantity for the Nose-Hoover chain: physical
/// energy plus the thermostat's own energy terms. Should drift only
/// slightly over a run when the chain is correctly implemented.
pub fn nose_hoover_conserved(state: &MdState, potential_energy: f64) -> f64 {
    let dof = state.dof();
    let kt = KB * state.temperature;
    let mut h = kinetic_energy(state) + potential_energy;
    for (i, c) in state.nh_chain.iter().enumerate() {
        let dof_i = if i == 0 { dof } else { 1.0 };
        h += 0.5 * c.q * c.v_xi * c.v_xi + dof_i * kt * c.xi;
    }
    h
}

fn apply_thermostat(state: &mut MdState) {
    match state.thermostat.clone() {
        Thermostat::None | Thermostat::Langevin => {}
        Thermostat::Berendsen => apply_berendsen_thermostat(state),
        Thermostat::VRescale => apply_vrescale_thermostat(state),
        Thermostat::NoseHoover { chain_length } => {
            apply_nose_hoover_thermostat(state, chain_length)
        }
    }
}

fn apply_berendsen_barostat(state: &mut MdState) {
    let box_pbc = state.topology.box_.pbc;
    let lx = state.topology.box_.lx;
    if !box_pbc || lx <= 0.0 {
        return;
    }
    let volume = state.topology.box_.lx * state.topology.box_.ly * state.topology.box_.lz;
    if volume <= 0.0 {
        return;
    }
    let ke = kinetic_energy(state);
    let pressure = (2.0 * ke + state.last_virial) / (3.0 * volume) * PRESSURE_CONV;
    let tau = state.tau_p.max(1e-6);
    let linear = 1.0 - state.compressibility * (state.pressure - pressure) * state.timestep / tau;
    let scale = linear.max(1e-3).cbrt().clamp(0.98, 1.02);
    state.topology.box_.lx *= scale;
    state.topology.box_.ly *= scale;
    state.topology.box_.lz *= scale;
    for atom in state.topology.atoms.iter_mut() {
        atom.position[0] *= scale;
        atom.position[1] *= scale;
        atom.position[2] *= scale;
    }
}

/// Metropolis Monte Carlo isotropic volume move. Scales molecule centers of
/// mass (preserving intramolecular geometry) when more than one molecule_id
/// is present in the topology, otherwise scales atomic positions directly.
fn apply_mc_barostat(state: &mut MdState, cutoff: f64) {
    let box_ = state.topology.box_.clone();
    if !box_.pbc || box_.lx <= 0.0 {
        return;
    }
    state.mc_attempts += 1;
    let mut rng = rand::thread_rng();
    let volume = box_.lx * box_.ly * box_.lz;
    let max_dv_frac = 0.02;
    let dv = volume * max_dv_frac * (rng.gen::<f64>() * 2.0 - 1.0);
    let new_volume = (volume + dv).max(1e-6);
    let scale = (new_volume / volume).cbrt();

    let mol_ids: Vec<u32> = state.topology.atoms.iter().map(|a| a.molecule_id).collect();
    let unique: std::collections::HashSet<u32> = mol_ids.iter().cloned().collect();
    let use_molecules = unique.len() > 1;

    let old_positions: Vec<[f64; 3]> = state.topology.positions();
    let backend = backend_from_env();
    let e_old = compute_forces(&state.topology, cutoff, backend.clone()).potential_energy;

    if use_molecules {
        let mut com: HashMap<u32, ([f64; 3], f64)> = HashMap::new();
        for atom in &state.topology.atoms {
            let entry = com.entry(atom.molecule_id).or_insert(([0.0; 3], 0.0));
            entry.0[0] += atom.mass * atom.position[0];
            entry.0[1] += atom.mass * atom.position[1];
            entry.0[2] += atom.mass * atom.position[2];
            entry.1 += atom.mass;
        }
        let coms: HashMap<u32, [f64; 3]> = com
            .into_iter()
            .map(|(k, (p, m))| {
                let m = m.max(1e-12);
                (k, [p[0] / m, p[1] / m, p[2] / m])
            })
            .collect();
        for atom in state.topology.atoms.iter_mut() {
            let c = coms[&atom.molecule_id];
            for k in 0..3 {
                atom.position[k] = c[k] * scale + (atom.position[k] - c[k]);
            }
        }
    } else {
        for atom in state.topology.atoms.iter_mut() {
            atom.position[0] *= scale;
            atom.position[1] *= scale;
            atom.position[2] *= scale;
        }
    }
    state.topology.box_.lx *= scale;
    state.topology.box_.ly *= scale;
    state.topology.box_.lz *= scale;

    let e_new = compute_forces(&state.topology, cutoff, backend).potential_energy;
    let n_scaled = if use_molecules {
        unique.len()
    } else {
        state.topology.atoms.len()
    } as f64;
    let kt = (KB * state.temperature).max(1e-12);
    let p_internal = state.pressure / PRESSURE_CONV;
    let d_e = e_new - e_old;
    let d_v = new_volume - volume;
    let arg =
        -(d_e + p_internal * d_v - n_scaled * kt * (new_volume / volume).ln()) / kt;

    let accept = arg >= 0.0 || rng.gen::<f64>() < arg.exp();
    if accept {
        state.mc_accepts += 1;
    } else {
        state.topology.set_positions(&old_positions);
        state.topology.box_ = box_;
    }
}

fn apply_barostat(state: &mut MdState, cutoff: f64) {
    match state.barostat {
        Barostat::None => {}
        Barostat::Berendsen => apply_berendsen_barostat(state),
        Barostat::MonteCarlo => {
            let interval = state.mc_interval.max(1);
            if state.step_count % interval == 0 {
                apply_mc_barostat(state, cutoff);
            }
        }
    }
}

/// Langevin (BAOAB-style) integration step. The stochastic friction term
/// itself provides canonical-ensemble thermostatting.
pub fn langevin_step(state: &mut MdState, cutoff: f64) -> MdStepResult {
    let forces = compute_step_forces(state, cutoff);
    let k_b = KB;
    let mut rng = rand::thread_rng();
    let gamma = state.friction;
    let dt = state.timestep;
    let reference: Vec<[f64; 3]> = state.topology.atoms.iter().map(|a| a.position).collect();

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
            let pre_constraint: Vec<[f64; 3]> =
                state.topology.atoms.iter().map(|a| a.position).collect();
            apply_constraints(&mut state.topology, &reference, c);
            apply_settle_velocity(&reference, &state.topology, &mut state.velocities, dt, c);
            apply_rattle(&state.topology, &mut state.velocities, c);
            state.last_virial +=
                compute_constraint_virial(&state.topology, &pre_constraint, dt, c);
        }
    }

    maybe_remove_com_motion(state);

    if state.npt {
        apply_barostat(state, cutoff);
    }

    state.step_count += 1;
    kinetic_stats(state, forces.potential_energy)
}

/// Velocity-Verlet integration step with a pluggable thermostat/barostat.
pub fn velocity_verlet_step(state: &mut MdState, cutoff: f64) -> MdStepResult {
    let forces = compute_step_forces(state, cutoff);
    let reference: Vec<[f64; 3]> = state.topology.atoms.iter().map(|a| a.position).collect();
    let dt = state.timestep;

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

    let mut pre_constraint = None;
    if state.use_constraints {
        if let Some(ref c) = state.constraints.clone() {
            pre_constraint = Some(
                state
                    .topology
                    .atoms
                    .iter()
                    .map(|a| a.position)
                    .collect::<Vec<_>>(),
            );
            apply_constraints(&mut state.topology, &reference, c);
            apply_settle_velocity(&reference, &state.topology, &mut state.velocities, dt, c);
        }
    }

    let forces2 = compute_step_forces(state, cutoff);
    if let (Some(pre), Some(c)) = (pre_constraint, state.constraints.clone()) {
        state.last_virial += compute_constraint_virial(&state.topology, &pre, dt, &c);
    }
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

    apply_thermostat(state);
    maybe_remove_com_motion(state);

    if state.npt {
        apply_barostat(state, cutoff);
    }

    state.step_count += 1;
    kinetic_stats(state, forces2.potential_energy)
}

/// Dispatch a single MD step to the integrator matching `state.thermostat`:
/// Langevin dynamics uses its own stochastic integrator, everything else
/// (including no thermostat, i.e. NVE) uses velocity-Verlet with the
/// configured thermostat/barostat applied on top.
pub fn md_step(state: &mut MdState, cutoff: f64) -> MdStepResult {
    match state.thermostat {
        Thermostat::Langevin => langevin_step(state, cutoff),
        _ => velocity_verlet_step(state, cutoff),
    }
}

fn kinetic_stats(state: &mut MdState, potential_energy: f64) -> MdStepResult {
    let ke = kinetic_energy(state);
    let dof = state.dof();
    let temperature = 2.0 * ke / (dof * KB);
    let volume = state.topology.box_.lx * state.topology.box_.ly * state.topology.box_.lz;
    let pressure = if state.topology.box_.pbc && volume > 1e-9 {
        (2.0 * ke + state.last_virial) / (3.0 * volume) * PRESSURE_CONV
    } else {
        0.0
    };
    state.last_pressure = pressure;
    MdStepResult {
        potential_energy,
        kinetic_energy: ke,
        temperature,
        pressure,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::topology::{AtomRecord, SimulationBox, Topology};

    fn force_cpu_backend() {
        // GPU adapters are not available in CI/test sandboxes; the hybrid
        // (default) backend would otherwise panic trying to init wgpu.
        std::env::set_var("AMPHI_FORCE_BACKEND", "cpu");
    }

    fn grid_topology(n_per_dim: usize, spacing: f64, sigma: f64, epsilon: f64, charge: f64) -> Topology {
        let mut top = Topology::new();
        let box_len = n_per_dim as f64 * spacing;
        top.box_ = SimulationBox {
            lx: box_len,
            ly: box_len,
            lz: box_len,
            pbc: true,
        };
        let mut idx = 0;
        for x in 0..n_per_dim {
            for y in 0..n_per_dim {
                for z in 0..n_per_dim {
                    top.atoms.push(AtomRecord {
                        element: "Ar".to_string(),
                        name: format!("Ar{}", idx),
                        mass: 39.95,
                        charge,
                        sigma,
                        epsilon,
                        position: [x as f64 * spacing, y as f64 * spacing, z as f64 * spacing],
                        residue_id: idx as u32,
                        molecule_id: idx as u32,
                        born_r: None,
                    });
                    idx += 1;
                }
            }
        }
        top
    }

    /// Small water + solute (CH3-capped chain) system used for the
    /// HMR/4fs NVE energy-conservation check.
    fn water_and_solute_topology() -> Topology {
        let mut top = Topology::new();
        top.box_ = SimulationBox {
            lx: 3.0,
            ly: 3.0,
            lz: 3.0,
            pbc: true,
        };
        let roh = 0.09572;
        let hoh = 1.824218134_f64;
        for w in 0..6 {
            let base = [
                0.3 + (w % 3) as f64 * 0.9,
                0.3 + (w / 3) as f64 * 0.9,
                0.5,
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
            top.bonds.push(crate::topology::BondTerm {
                i: o_idx,
                j: o_idx + 1,
                k: 5000.0,
                r0: roh,
            });
            top.bonds.push(crate::topology::BondTerm {
                i: o_idx,
                j: o_idx + 2,
                k: 5000.0,
                r0: roh,
            });
        }

        // A short solute "chain": two carbons bonded to each other, each
        // carrying two hydrogens (LINCS-constrained C-H bonds).
        let c1 = top.atoms.len();
        top.atoms.push(AtomRecord {
            element: "C".into(),
            name: "C1".into(),
            mass: 12.011,
            charge: 0.0,
            sigma: 0.34,
            epsilon: 0.36,
            position: [1.5, 1.5, 1.5],
            residue_id: 100,
            molecule_id: 100,
            born_r: None,
        });
        let c2 = top.atoms.len();
        top.atoms.push(AtomRecord {
            element: "C".into(),
            name: "C2".into(),
            mass: 12.011,
            charge: 0.0,
            sigma: 0.34,
            epsilon: 0.36,
            position: [1.65, 1.5, 1.5],
            residue_id: 100,
            molecule_id: 100,
            born_r: None,
        });
        top.bonds.push(crate::topology::BondTerm {
            i: c1,
            j: c2,
            k: 3000.0,
            r0: 0.15,
        });
        let r0 = 0.109;
        for (c_idx, dirs) in [
            (c1, [[-0.9f64, 0.4, 0.0], [-0.9, -0.4, 0.3]]),
            (c2, [[0.9, 0.4, 0.0], [0.9, -0.4, -0.3]]),
        ] {
            let base = top.atoms[c_idx].position;
            for d in dirs {
                let n = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
                let pos = [
                    base[0] + r0 * d[0] / n,
                    base[1] + r0 * d[1] / n,
                    base[2] + r0 * d[2] / n,
                ];
                let h_idx = top.atoms.len();
                top.atoms.push(AtomRecord {
                    element: "H".into(),
                    name: format!("H{h_idx}"),
                    mass: 1.008,
                    charge: 0.0,
                    sigma: 0.0,
                    epsilon: 0.0,
                    position: pos,
                    residue_id: 100,
                    molecule_id: 100,
                    born_r: None,
                });
                top.bonds.push(crate::topology::BondTerm {
                    i: c_idx,
                    j: h_idx,
                    k: 3000.0,
                    r0,
                });
            }
        }
        top.build_exclusions();
        top
    }

    /// Runs NVE (no thermostat/barostat) velocity-Verlet for `steps` and
    /// returns the relative drift (max-min)/|mean| of total energy over
    /// the trajectory.
    fn nve_energy_drift(mut state: MdState, steps: usize, cutoff: f64) -> f64 {
        state.thermostat = Thermostat::None;
        state.npt = false;
        let mut totals = Vec::with_capacity(steps);
        for _ in 0..steps {
            let r = velocity_verlet_step(&mut state, cutoff);
            totals.push(r.potential_energy + r.kinetic_energy);
        }
        let mean: f64 = totals.iter().sum::<f64>() / totals.len() as f64;
        let max = totals.iter().cloned().fold(f64::MIN, f64::max);
        let min = totals.iter().cloned().fold(f64::MAX, f64::min);
        (max - min) / mean.abs().max(1e-8)
    }

    #[test]
    fn hmr_enables_stable_4fs_nve_with_hbond_constraints() {
        force_cpu_backend();

        // Baseline: standard masses, dt = 2 fs (0.002 ps).
        let top_2fs = water_and_solute_topology();
        let mut state_2fs = MdState::new(top_2fs, 300.0);
        state_2fs.timestep = 0.002;
        state_2fs.use_constraints = true;
        let drift_2fs = nve_energy_drift(state_2fs, 60, 1.0);

        // HMR (H mass -> 3.024 Da) + h-bond constraints, dt = 4 fs (0.004 ps).
        let mut top_4fs = water_and_solute_topology();
        top_4fs.apply_hmr(3.024);
        let mut state_4fs = MdState::new(top_4fs, 300.0);
        state_4fs.timestep = 0.004;
        state_4fs.use_constraints = true;
        let drift_4fs = nve_energy_drift(state_4fs, 30, 1.0);

        // Sanity check: both runs stay finite and bounded (no blow-up /
        // NaN), and the 4 fs + HMR + h-bond-constrained run isn't wildly
        // less stable than the 2 fs baseline over the same physical time.
        assert!(drift_2fs.is_finite(), "2fs drift not finite");
        assert!(drift_4fs.is_finite(), "4fs+HMR drift not finite");
        assert!(drift_4fs < 5.0 * drift_2fs.max(0.05), "4fs+HMR drift {drift_4fs} too far above 2fs baseline {drift_2fs}");
    }

    #[test]
    fn ideal_gas_virial_is_near_zero() {
        // No LJ, no charge, no bonds: pairwise virial must vanish exactly.
        let top = grid_topology(3, 0.5, 0.0, 0.0, 0.0);
        let virial = crate::forces::cpu::compute_virial(&top, 1.0);
        assert!(virial.abs() < 1e-8, "virial = {virial}");
    }

    #[test]
    fn ideal_gas_pressure_matches_ideal_gas_law() {
        let top = grid_topology(3, 0.6, 0.0, 0.0, 0.0);
        let n = top.atoms.len();
        let temperature = 300.0;
        let mut state = MdState::new(top, temperature);
        // Force a known kinetic energy consistent with the target temperature
        // by re-seeding velocities at that temperature (already done by new()).
        let ke = kinetic_energy(&state);
        let dof = state.dof();
        let volume = state.topology.box_.lx * state.topology.box_.ly * state.topology.box_.lz;
        state.last_virial = crate::forces::cpu::compute_virial(&state.topology, 1.0);
        let pressure = (2.0 * ke + state.last_virial) / (3.0 * volume) * PRESSURE_CONV;
        let expected = dof * KB * (2.0 * ke / (dof * KB)) / (3.0 * volume) * PRESSURE_CONV;
        assert!((pressure - expected).abs() < 1e-6, "n={n}");
        assert!(pressure >= 0.0);
    }

    #[test]
    fn vrescale_reproduces_target_mean_temperature() {
        let top = grid_topology(4, 0.5, 0.3, 0.2, 0.0);
        let target_t = 300.0;
        // Start far from the target temperature.
        let mut state = MdState::new(top, 100.0);
        state.temperature = target_t;
        state.tau_t = 0.2;
        state.timestep = 0.002;

        let mut temps = Vec::new();
        for _ in 0..4000 {
            apply_vrescale_thermostat(&mut state);
            let ke = kinetic_energy(&state);
            let dof = state.dof();
            temps.push(2.0 * ke / (dof * KB));
        }
        let burn_in = temps.len() / 2;
        let mean: f64 = temps[burn_in..].iter().sum::<f64>() / (temps.len() - burn_in) as f64;
        assert!(
            (mean - target_t).abs() / target_t < 0.1,
            "mean temperature {mean} far from target {target_t}"
        );
    }

    #[test]
    fn nose_hoover_conserved_quantity_drift_is_small() {
        force_cpu_backend();
        let top = grid_topology(3, 0.4, 0.3, 0.3, 0.0);
        let mut state = MdState::new(top, 300.0);
        state.thermostat = Thermostat::NoseHoover { chain_length: 3 };
        state.tau_t = 0.5;
        state.timestep = 0.0005;
        state.use_constraints = false;
        state.constraints = None;

        let first = velocity_verlet_step(&mut state, 1.0);
        let h0 = nose_hoover_conserved(&state, first.potential_energy);

        let mut last_h = h0;
        for _ in 0..200 {
            let res = velocity_verlet_step(&mut state, 1.0);
            last_h = nose_hoover_conserved(&state, res.potential_energy);
        }

        let drift = (last_h - h0).abs() / h0.abs().max(1.0);
        assert!(drift < 0.2, "conserved quantity drift too large: {drift}");
    }

    #[test]
    fn mc_barostat_has_sane_acceptance() {
        force_cpu_backend();
        let top = grid_topology(4, 0.5, 0.3, 0.3, 0.0);
        let mut state = MdState::new(top, 300.0);
        state.pressure = 1.0;
        for _ in 0..40 {
            apply_mc_barostat(&mut state, 1.0);
        }
        assert!(state.mc_attempts > 0);
        let ratio = state.mc_acceptance_ratio();
        assert!(ratio > 0.0 && ratio <= 1.0, "acceptance ratio {ratio}");
    }

    #[test]
    fn dof_accounts_for_constraints_and_com() {
        let top = grid_topology(2, 0.5, 0.3, 0.2, 0.0);
        let n_atoms = top.atoms.len();
        let state = MdState::new(top, 300.0);
        assert_eq!(state.dof(), (3 * n_atoms) as f64 - 3.0);
    }

    #[test]
    fn com_motion_removal_zeroes_net_momentum() {
        let top = grid_topology(2, 0.5, 0.3, 0.2, 0.0);
        let mut state = MdState::new(top, 300.0);
        // Inject a net drift velocity.
        for v in state.velocities.iter_mut() {
            v[0] += 5.0;
        }
        remove_com_motion(&mut state);
        let total_mass: f64 = state.topology.atoms.iter().map(|a| a.mass).sum();
        let mut p = [0.0; 3];
        for (v, a) in state.velocities.iter().zip(state.topology.atoms.iter()) {
            for k in 0..3 {
                p[k] += a.mass * v[k];
            }
        }
        for k in 0..3 {
            assert!((p[k] / total_mass).abs() < 1e-9);
        }
    }

    #[test]
    fn parse_thermostat_and_barostat_names() {
        assert_eq!(parse_thermostat("langevin", 3).unwrap(), Thermostat::Langevin);
        assert_eq!(
            parse_thermostat("nose-hoover", 5).unwrap(),
            Thermostat::NoseHoover { chain_length: 5 }
        );
        assert!(parse_thermostat("bogus", 3).is_err());
        assert_eq!(parse_barostat("montecarlo").unwrap(), Barostat::MonteCarlo);
        assert!(parse_barostat("bogus").is_err());
    }
}
