use pyo3::prelude::*;
use pyo3::wrap_pymodule;

mod bonds;
mod builder;
mod compute_pipeline;
mod constraints;
mod electrostatics;
mod forces;
mod integrator;
mod minimize;
mod neighbor;
mod pdb;
mod random;
mod topology;
mod trajectory;
mod utilities;

use builder::builder as build;
use compute_pipeline::{run_atom_pipeline, run_md, AtomPipelineParams};
use constraints::{build_constraints_with_algorithm, ConstraintAlgorithm};
use integrator::{md_step, parse_barostat, parse_thermostat, Barostat, MdState, Restraint};
use minimize::minimize_with;
use forces::{backend_from_env, compute_forces};
use pdb::PdbFilePy;
use topology::{SimulationBox, Topology, TopologyPy};
use trajectory::{
    read_checkpoint, read_dcd, write_checkpoint, Checkpoint, EnergyLogWriter, TrajectoryFormat,
    TrajectoryReader, TrajectoryWriter,
};

/// Default hydrogen mass (Da) used by hydrogen mass repartitioning when
/// `--hmr` is requested without an explicit mass.
const DEFAULT_HMR_MASS: f64 = 3.024;

/// Set the global rayon thread pool size, if requested and not already set
/// for this process (rayon's global pool can only be configured once).
fn apply_thread_override(threads: Option<usize>) -> PyResult<()> {
    if let Some(n) = threads {
        // Building the global pool a second time (e.g. a second simulate()
        // call in the same process) errors; that's fine to ignore since the
        // pool is already configured (possibly by an earlier call).
        let _ = rayon::ThreadPoolBuilder::new().num_threads(n).build_global();
    }
    Ok(())
}

/// Apply `--dt` / `--constraints` / `--constraint-algorithm` overrides to an
/// already-constructed `MdState`.
fn apply_dt_and_constraint_overrides(
    state: &mut MdState,
    dt: Option<f64>,
    constraints_mode: Option<&str>,
    constraint_algorithm: Option<&str>,
) -> PyResult<()> {
    if let Some(dt) = dt {
        state.timestep = dt;
    }
    if let Some(mode) = constraints_mode {
        match mode {
            "none" => state.use_constraints = false,
            "h-bonds" => state.use_constraints = true,
            other => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "unknown --constraints value '{other}' (expected 'none' or 'h-bonds')"
                )))
            }
        }
    }
    if let Some(algo) = constraint_algorithm {
        let algorithm = match algo {
            "lincs" => ConstraintAlgorithm::Lincs,
            "shake" => ConstraintAlgorithm::Shake,
            other => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "unknown --constraint-algorithm value '{other}' (expected 'lincs' or 'shake')"
                )))
            }
        };
        state.constraints = Some(build_constraints_with_algorithm(&state.topology, algorithm));
    }
    Ok(())
}

#[pyfunction]
fn load_topology(path: &str) -> PyResult<TopologyPy> {
    TopologyPy::read(path)
}

#[pyfunction]
#[pyo3(signature = (path, output=None))]
fn ionize_topology(path: &str, output: Option<&str>) -> PyResult<TopologyPy> {
    let mut top = Topology::read(path).map_err(|e| pyo3::exceptions::PyIOError::new_err(e))?;
    top.neutralize();
    if let Some(out) = output {
        top.write(out)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e))?;
    }
    Ok(TopologyPy { inner: top })
}

/// Slow, independent direct-Ewald reference total electrostatic energy
/// (real-space minimum-image + brute-force reciprocal sum + self term),
/// for validating the fast SPME path against. Intended for small test
/// systems only (O(N*K^3)).
#[pyfunction]
#[pyo3(signature = (path, cutoff=1.0, ewald_rtol=1e-5, k_max=12))]
fn direct_ewald_reference_energy(
    path: &str,
    cutoff: f64,
    ewald_rtol: f64,
    k_max: i64,
) -> PyResult<f64> {
    let top = Topology::read(path).map_err(|e| pyo3::exceptions::PyIOError::new_err(e))?;
    Ok(crate::electrostatics::pme::direct_ewald_total_energy(
        &top, cutoff, ewald_rtol, k_max,
    ))
}

#[pyfunction]
fn topology_energy(path: &str) -> PyResult<f64> {
    let top = Topology::read(path).map_err(|e| pyo3::exceptions::PyIOError::new_err(e))?;
    let backend = backend_from_env();
    // `compute_forces` already folds in the Ewald self-/background-energy
    // correction for periodic (PME) systems internally (see
    // `forces::compute_forces`); adding it again here double-counted it
    // for every periodic topology.
    let result = compute_forces(&top, 1.0, backend);
    Ok(result.potential_energy)
}

#[pyfunction]
#[pyo3(signature = (path, output=None, max_steps=None, step_size=None, emtol=None, minimizer=None))]
fn minimize_topology(
    path: &str,
    output: Option<&str>,
    max_steps: Option<usize>,
    step_size: Option<f64>,
    emtol: Option<f64>,
    minimizer: Option<&str>,
) -> PyResult<(TopologyPy, f64, usize)> {
    let top = Topology::read(path).map_err(|e| pyo3::exceptions::PyIOError::new_err(e))?;
    let kind = match minimizer.unwrap_or("lbfgs") {
        "lbfgs" => crate::minimize::MinimizerKind::LBfgs,
        "sd" | "steepest_descent" => crate::minimize::MinimizerKind::SteepestDescent,
        other => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "unknown minimizer '{other}', expected 'lbfgs' or 'sd'"
            )))
        }
    };
    let result = minimize_with(
        top,
        1.0,
        max_steps.unwrap_or(100),
        step_size.unwrap_or(0.001),
        emtol.unwrap_or(10.0),
        kind,
    );
    if let Some(out) = output {
        result
            .topology
            .write(out)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e))?;
    }
    Ok((
        TopologyPy {
            inner: result.topology,
        },
        result.final_energy,
        result.iterations,
    ))
}

fn make_traj_writer(
    output_traj: Option<&str>,
    traj_format: Option<&str>,
) -> PyResult<Option<TrajectoryWriter>> {
    match output_traj {
        None => Ok(None),
        Some(path) => match traj_format {
            None => Ok(Some(TrajectoryWriter::new(path))),
            Some(fmt) => {
                let format = TrajectoryFormat::parse(fmt)
                    .map_err(pyo3::exceptions::PyValueError::new_err)?;
                Ok(Some(
                    TrajectoryWriter::with_format(path, format)
                        .map_err(pyo3::exceptions::PyIOError::new_err)?,
                ))
            }
        },
    }
}

fn write_final_checkpoint(
    checkpoint_out: Option<&str>,
    state: &MdState,
    step: u64,
    seed: u64,
) -> PyResult<()> {
    if let Some(path) = checkpoint_out {
        let ckpt = Checkpoint {
            step,
            seed,
            box_: state.topology.box_.clone(),
            positions: state.topology.atoms.iter().map(|a| a.position).collect(),
            velocities: state.velocities.clone(),
        };
        write_checkpoint(path, &ckpt).map_err(pyo3::exceptions::PyIOError::new_err)?;
    }
    Ok(())
}

#[pyfunction]
#[pyo3(signature = (
    path,
    output_traj=None,
    steps=None,
    temperature=None,
    restart_traj=None,
    restraint_k=None,
    thermostat=None,
    barostat=None,
    pressure=None,
    tau_t=None,
    tau_p=None,
    com_remove_interval=None,
    nh_chain_length=None,
    mc_interval=None,
    traj_format=None,
    checkpoint_out=None,
    checkpoint_in=None,
    energy_log=None,
    dt=None,
    hmr=None,
    constraints=None,
    constraint_algorithm=None,
    threads=None
))]
fn equilibrate_topology(
    path: &str,
    output_traj: Option<&str>,
    steps: Option<u64>,
    temperature: Option<f64>,
    restart_traj: Option<&str>,
    restraint_k: Option<f64>,
    thermostat: Option<&str>,
    barostat: Option<&str>,
    pressure: Option<f64>,
    tau_t: Option<f64>,
    tau_p: Option<f64>,
    com_remove_interval: Option<u64>,
    nh_chain_length: Option<usize>,
    mc_interval: Option<u64>,
    traj_format: Option<&str>,
    checkpoint_out: Option<&str>,
    checkpoint_in: Option<&str>,
    energy_log: Option<&str>,
    dt: Option<f64>,
    hmr: Option<bool>,
    constraints: Option<&str>,
    constraint_algorithm: Option<&str>,
    threads: Option<usize>,
) -> PyResult<(TopologyPy, f64, f64)> {
    apply_thread_override(threads)?;
    let mut top = Topology::read(path).map_err(|e| pyo3::exceptions::PyIOError::new_err(e))?;
    if hmr.unwrap_or(false) {
        top.apply_hmr(DEFAULT_HMR_MASS);
    }
    let reference: Vec<[f64; 3]> = top.atoms.iter().map(|a| a.position).collect();
    if let Some(restart) = restart_traj {
        TrajectoryReader::new(restart)
            .apply_last_frame(&mut top)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;
    }
    let n_steps = steps.unwrap_or(100);
    let temp = temperature.unwrap_or(300.0);
    let mut state = MdState::new(top.clone(), temp);
    apply_dt_and_constraint_overrides(&mut state, dt, constraints, constraint_algorithm)?;
    let mut seed = 0u64;
    if let Some(ckpt_path) = checkpoint_in {
        let ckpt = read_checkpoint(ckpt_path).map_err(pyo3::exceptions::PyIOError::new_err)?;
        if ckpt.positions.len() == state.topology.atoms.len() {
            for (atom, pos) in state.topology.atoms.iter_mut().zip(ckpt.positions.iter()) {
                atom.position = *pos;
            }
            state.velocities = ckpt.velocities.clone();
            seed = ckpt.seed;
        }
    }
    state.thermostat = parse_thermostat(thermostat.unwrap_or("langevin"), nh_chain_length.unwrap_or(3))
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    let requested_npt = barostat.is_some();
    state.barostat = if requested_npt {
        parse_barostat(barostat.unwrap_or("none")).map_err(pyo3::exceptions::PyValueError::new_err)?
    } else {
        Barostat::None
    };
    state.npt = requested_npt;
    if let Some(p) = pressure {
        state.pressure = p;
    }
    if let Some(t) = tau_t {
        state.tau_t = t;
    }
    if let Some(t) = tau_p {
        state.tau_p = t;
    }
    if let Some(c) = com_remove_interval {
        state.com_remove_interval = c;
    }
    if let Some(m) = mc_interval {
        state.mc_interval = m;
    }
    if let Some(k) = restraint_k {
        if k > 0.0 {
            for (atom, pos) in reference.iter().enumerate() {
                state.restraints.push(Restraint {
                    atom,
                    k,
                    position: *pos,
                });
            }
        }
    }
    let mut writer = make_traj_writer(output_traj, traj_format)?;
    let mut elog = energy_log
        .map(|p| EnergyLogWriter::new(p, false))
        .transpose()
        .map_err(pyo3::exceptions::PyIOError::new_err)?;
    let mut last_temp = temp;
    let mut last_pressure = 0.0;
    for step in 0..n_steps {
        let res = md_step(&mut state, 1.0);
        last_temp = res.temperature;
        last_pressure = res.pressure;
        if let Some(w) = writer.as_mut() {
            w.write_frame(step, &state.topology, res.potential_energy, res.kinetic_energy)
                .map_err(|e| pyo3::exceptions::PyIOError::new_err(e))?;
        }
        if let Some(log) = elog.as_mut() {
            log.write_row(step, step as f64 * 0.001, res.potential_energy, res.kinetic_energy, res.temperature)
                .map_err(pyo3::exceptions::PyIOError::new_err)?;
        }
    }
    write_final_checkpoint(checkpoint_out, &state, n_steps, seed)?;
    top = state.topology;
    Ok((TopologyPy { inner: top }, last_temp, last_pressure))
}

#[pyfunction]
#[pyo3(signature = (
    path,
    output_traj=None,
    steps=None,
    temperature=None,
    restart_traj=None,
    npt=None,
    thermostat=None,
    barostat=None,
    pressure=None,
    tau_t=None,
    tau_p=None,
    com_remove_interval=None,
    nh_chain_length=None,
    mc_interval=None,
    traj_format=None,
    checkpoint_out=None,
    checkpoint_in=None,
    energy_log=None,
    dt=None,
    hmr=None,
    constraints=None,
    constraint_algorithm=None,
    threads=None
))]
fn simulate_topology(
    path: &str,
    output_traj: Option<&str>,
    steps: Option<u64>,
    temperature: Option<f64>,
    restart_traj: Option<&str>,
    npt: Option<bool>,
    thermostat: Option<&str>,
    barostat: Option<&str>,
    pressure: Option<f64>,
    tau_t: Option<f64>,
    tau_p: Option<f64>,
    com_remove_interval: Option<u64>,
    nh_chain_length: Option<usize>,
    mc_interval: Option<u64>,
    traj_format: Option<&str>,
    checkpoint_out: Option<&str>,
    checkpoint_in: Option<&str>,
    energy_log: Option<&str>,
    dt: Option<f64>,
    hmr: Option<bool>,
    constraints: Option<&str>,
    constraint_algorithm: Option<&str>,
    threads: Option<usize>,
) -> PyResult<(TopologyPy, f64, f64)> {
    apply_thread_override(threads)?;
    let mut top = Topology::read(path).map_err(|e| pyo3::exceptions::PyIOError::new_err(e))?;
    if hmr.unwrap_or(false) {
        top.apply_hmr(DEFAULT_HMR_MASS);
    }
    if let Some(restart) = restart_traj {
        TrajectoryReader::new(restart)
            .apply_last_frame(&mut top)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;
    }
    let n_steps = steps.unwrap_or(100);
    let temp = temperature.unwrap_or(300.0);
    let mut state = MdState::new(top.clone(), temp);
    state.use_constraints = true;
    apply_dt_and_constraint_overrides(&mut state, dt, constraints, constraint_algorithm)?;
    let mut seed = 0u64;
    if let Some(ckpt_path) = checkpoint_in {
        let ckpt = read_checkpoint(ckpt_path).map_err(pyo3::exceptions::PyIOError::new_err)?;
        if ckpt.positions.len() == state.topology.atoms.len() {
            for (atom, pos) in state.topology.atoms.iter_mut().zip(ckpt.positions.iter()) {
                atom.position = *pos;
            }
            state.velocities = ckpt.velocities.clone();
            seed = ckpt.seed;
        }
    }
    state.thermostat = parse_thermostat(thermostat.unwrap_or("none"), nh_chain_length.unwrap_or(3))
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    let requested_npt = npt.unwrap_or(false) || barostat.is_some();
    state.barostat = if requested_npt {
        parse_barostat(barostat.unwrap_or("berendsen"))
            .map_err(pyo3::exceptions::PyValueError::new_err)?
    } else {
        Barostat::None
    };
    state.npt = requested_npt;
    if let Some(p) = pressure {
        state.pressure = p;
    }
    if let Some(t) = tau_t {
        state.tau_t = t;
    }
    if let Some(t) = tau_p {
        state.tau_p = t;
    }
    if let Some(c) = com_remove_interval {
        state.com_remove_interval = c;
    }
    if let Some(m) = mc_interval {
        state.mc_interval = m;
    }
    let mut writer = make_traj_writer(output_traj, traj_format)?;
    let mut elog = energy_log
        .map(|p| EnergyLogWriter::new(p, false))
        .transpose()
        .map_err(pyo3::exceptions::PyIOError::new_err)?;
    let mut last_temp = temp;
    let mut last_pressure = 0.0;
    for step in 0..n_steps {
        let res = md_step(&mut state, 1.0);
        last_temp = res.temperature;
        last_pressure = res.pressure;
        if let Some(w) = writer.as_mut() {
            w.write_frame(step, &state.topology, res.potential_energy, res.kinetic_energy)
                .map_err(|e| pyo3::exceptions::PyIOError::new_err(e))?;
        }
        if let Some(log) = elog.as_mut() {
            log.write_row(step, step as f64 * 0.002, res.potential_energy, res.kinetic_energy, res.temperature)
                .map_err(pyo3::exceptions::PyIOError::new_err)?;
        }
    }
    write_final_checkpoint(checkpoint_out, &state, n_steps, seed)?;
    top = state.topology;
    Ok((TopologyPy { inner: top }, last_temp, last_pressure))
}

/// GPU-resident production MD: positions/velocities/forces stay in GPU
/// buffers for the whole run (see `forces::gpu_resident`). Supports a
/// narrower feature set than `simulate_topology` — NVE, Langevin, and
/// V-rescale only; no barostat, no solute LINCS/SHAKE (h-bonds constraints
/// use CPU SETTLE for water only). See docs/gpu_resident.md.
#[pyfunction]
#[pyo3(signature = (
    path,
    output_traj=None,
    steps=None,
    temperature=None,
    dt=None,
    thermostat=None,
    tau_t=None,
    seed=None,
    energy_log=None,
    cutoff=None,
))]
fn simulate_topology_gpu_resident(
    path: &str,
    output_traj: Option<&str>,
    steps: Option<u64>,
    temperature: Option<f64>,
    dt: Option<f64>,
    thermostat: Option<&str>,
    tau_t: Option<f64>,
    seed: Option<u64>,
    energy_log: Option<&str>,
    cutoff: Option<f64>,
) -> PyResult<(TopologyPy, f64)> {
    let top = Topology::read(path).map_err(|e| pyo3::exceptions::PyIOError::new_err(e))?;
    let n_steps = steps.unwrap_or(100) as usize;
    let temp = temperature.unwrap_or(300.0);
    let dt_val = dt.unwrap_or(0.002);
    let cutoff_val = cutoff.unwrap_or(1.0);
    let thermostat_kind = match thermostat.unwrap_or("none") {
        "none" => forces::gpu_resident::GpuResidentThermostat::None,
        "vrescale" => forces::gpu_resident::GpuResidentThermostat::VRescale,
        "langevin" => forces::gpu_resident::GpuResidentThermostat::Langevin,
        other => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "gpu-resident backend supports thermostat in {{none, vrescale, langevin}}, got {other}"
            )))
        }
    };
    let result = forces::gpu_resident::run_gpu_resident(
        &top,
        cutoff_val,
        n_steps,
        dt_val,
        thermostat_kind,
        temp,
        tau_t.unwrap_or(1.0),
        seed.unwrap_or(0),
        10,
    );
    if let Some(log_path) = energy_log {
        let mut log = EnergyLogWriter::new(log_path, false).map_err(pyo3::exceptions::PyIOError::new_err)?;
        for sample in &result.samples {
            log.write_row(
                sample.step as u64,
                sample.step as f64 * dt_val,
                sample.potential_energy,
                sample.kinetic_energy,
                sample.temperature,
            )
            .map_err(pyo3::exceptions::PyIOError::new_err)?;
        }
    }
    let final_temp = result.samples.last().map(|s| s.temperature).unwrap_or(temp);
    if let Some(traj_path) = output_traj {
        let mut writer = make_traj_writer(Some(traj_path), None)?;
        if let Some(w) = writer.as_mut() {
            let last = result.samples.last();
            w.write_frame(
                n_steps as u64,
                &result.topology,
                last.map(|s| s.potential_energy).unwrap_or(0.0),
                last.map(|s| s.kinetic_energy).unwrap_or(0.0),
            )
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e))?;
        }
    }
    Ok((TopologyPy { inner: result.topology }, final_temp))
}

#[pyfunction]
fn read_dcd_trajectory(
    path: &str,
) -> PyResult<(usize, Vec<Vec<(f64, f64, f64)>>, Vec<(f64, f64, f64)>)> {
    let traj = read_dcd(path).map_err(pyo3::exceptions::PyIOError::new_err)?;
    let frames: Vec<Vec<(f64, f64, f64)>> = traj
        .frames
        .iter()
        .map(|f| {
            f.positions
                .iter()
                .map(|p| (p[0] as f64, p[1] as f64, p[2] as f64))
                .collect()
        })
        .collect();
    let boxes: Vec<(f64, f64, f64)> = traj
        .frames
        .iter()
        .map(|f| (f.cell[0], f.cell[2], f.cell[5]))
        .collect();
    Ok((traj.natoms, frames, boxes))
}

#[pyfunction]
fn write_checkpoint_file(
    path: &str,
    positions: Vec<(f64, f64, f64)>,
    velocities: Vec<(f64, f64, f64)>,
    box_lengths: (f64, f64, f64),
    pbc: bool,
    step: u64,
    seed: u64,
) -> PyResult<()> {
    let ckpt = Checkpoint {
        step,
        seed,
        box_: SimulationBox {
            lx: box_lengths.0,
            ly: box_lengths.1,
            lz: box_lengths.2,
            pbc,
        },
        positions: positions.into_iter().map(|p| [p.0, p.1, p.2]).collect(),
        velocities: velocities.into_iter().map(|v| [v.0, v.1, v.2]).collect(),
    };
    write_checkpoint(path, &ckpt).map_err(pyo3::exceptions::PyIOError::new_err)
}

#[allow(clippy::type_complexity)]
#[pyfunction]
fn read_checkpoint_file(
    path: &str,
) -> PyResult<(
    Vec<(f64, f64, f64)>,
    Vec<(f64, f64, f64)>,
    (f64, f64, f64),
    bool,
    u64,
    u64,
)> {
    let ckpt = read_checkpoint(path).map_err(pyo3::exceptions::PyIOError::new_err)?;
    Ok((
        ckpt.positions.iter().map(|p| (p[0], p[1], p[2])).collect(),
        ckpt.velocities.iter().map(|v| (v[0], v[1], v[2])).collect(),
        (ckpt.box_.lx, ckpt.box_.ly, ckpt.box_.lz),
        ckpt.box_.pbc,
        ckpt.step,
        ckpt.seed,
    ))
}

#[pymodule]
fn simulate(_py: Python, m: Bound<PyModule>) -> PyResult<()> {
    #[pyfn(m, name = "run_atom_pipeline")]
    fn rap(
        coords: Vec<(f64, f64, f64)>,
        atom_types: Vec<String>,
        bonds: Vec<(usize, usize)>,
        params: AtomPipelineParams,
    ) -> Vec<(f64, f64, f64)> {
        let coords_vec: Vec<[f64; 3]> = coords.iter().map(|c| [c.0, c.1, c.2]).collect();
        run_atom_pipeline(&coords_vec, &atom_types, &bonds, params)
            .iter()
            .map(|c| (c[0], c[1], c[2]))
            .collect()
    }

    #[pyfn(m, name = "run_simulation")]
    fn run_simulation_py(
        coords: Vec<(f64, f64, f64)>,
        atom_types: Vec<String>,
        bonds: Vec<(usize, usize)>,
    ) -> PyResult<PdbFilePy> {
        let out = run_md(&coords, &atom_types, &bonds, 2, 100, 0.002);
        Ok(PdbFilePy {
            coords: out.iter().map(|c| (c[0], c[1], c[2])).collect(),
            atom_types,
            bonds,
        })
    }

    #[pyfn(m, name = "run_minimization")]
    fn run_minimization_py(
        coords: Vec<(f64, f64, f64)>,
        atom_types: Vec<String>,
        bonds: Vec<(usize, usize)>,
    ) -> PyResult<PdbFilePy> {
        let out = run_md(&coords, &atom_types, &bonds, 1, 50, 0.001);
        Ok(PdbFilePy {
            coords: out.iter().map(|c| (c[0], c[1], c[2])).collect(),
            atom_types,
            bonds,
        })
    }

    #[pyfn(m, name = "run_relaxation")]
    fn run_relaxation_py(
        coords: Vec<(f64, f64, f64)>,
        atom_types: Vec<String>,
        bonds: Vec<(usize, usize)>,
    ) -> PyResult<PdbFilePy> {
        let out = run_md(&coords, &atom_types, &bonds, 0, 50, 0.002);
        Ok(PdbFilePy {
            coords: out.iter().map(|c| (c[0], c[1], c[2])).collect(),
            atom_types,
            bonds,
        })
    }

    Ok(())
}

#[pymodule]
fn rustquantic(_py: Python, m: Bound<PyModule>) -> PyResult<()> {
    m.add_class::<PdbFilePy>()?;
    m.add_class::<TopologyPy>()?;
    m.add_function(wrap_pyfunction_bound!(topology_energy, &m)?)?;
    m.add_function(wrap_pyfunction_bound!(direct_ewald_reference_energy, &m)?)?;
    m.add_function(wrap_pyfunction_bound!(ionize_topology, &m)?)?;
    m.add_function(wrap_pyfunction_bound!(load_topology, &m)?)?;
    m.add_function(wrap_pyfunction_bound!(minimize_topology, &m)?)?;
    m.add_function(wrap_pyfunction_bound!(equilibrate_topology, &m)?)?;
    m.add_function(wrap_pyfunction_bound!(simulate_topology, &m)?)?;
    m.add_function(wrap_pyfunction_bound!(simulate_topology_gpu_resident, &m)?)?;
    m.add_function(wrap_pyfunction_bound!(read_dcd_trajectory, &m)?)?;
    m.add_function(wrap_pyfunction_bound!(write_checkpoint_file, &m)?)?;
    m.add_function(wrap_pyfunction_bound!(read_checkpoint_file, &m)?)?;
    m.add_wrapped(wrap_pymodule!(crate::utilities::utilities))?;
    m.add_wrapped(wrap_pymodule!(build))?;
    m.add_wrapped(wrap_pymodule!(simulate))?;
    topology::register_module(&m)?;
    Ok(())
}
