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
mod topology;
mod trajectory;
mod utilities;

use builder::builder as build;
use compute_pipeline::{run_atom_pipeline, run_md, AtomPipelineParams};
use integrator::{langevin_step, velocity_verlet_step, MdState, Restraint};
use minimize::minimize;
use forces::{backend_from_env, compute_forces};
use electrostatics::ewald_energy_correction;
use pdb::PdbFilePy;
use topology::{SimulationBox, Topology, TopologyPy};
use trajectory::{
    read_checkpoint, read_dcd, write_checkpoint, Checkpoint, EnergyLogWriter, TrajectoryFormat,
    TrajectoryReader, TrajectoryWriter,
};

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

#[pyfunction]
fn topology_energy(path: &str) -> PyResult<f64> {
    let top = Topology::read(path).map_err(|e| pyo3::exceptions::PyIOError::new_err(e))?;
    let backend = backend_from_env();
    let result = compute_forces(&top, 1.0, backend);
    Ok(result.potential_energy + ewald_energy_correction(&top))
}

#[pyfunction]
#[pyo3(signature = (path, output=None, max_steps=None, step_size=None))]
fn minimize_topology(
    path: &str,
    output: Option<&str>,
    max_steps: Option<usize>,
    step_size: Option<f64>,
) -> PyResult<(TopologyPy, f64, usize)> {
    let top = Topology::read(path).map_err(|e| pyo3::exceptions::PyIOError::new_err(e))?;
    let result = minimize(
        top,
        1.0,
        max_steps.unwrap_or(100),
        step_size.unwrap_or(0.001),
        1e-4,
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
#[pyo3(signature = (path, output_traj=None, steps=None, temperature=None, restart_traj=None, restraint_k=None, traj_format=None, checkpoint_out=None, checkpoint_in=None, energy_log=None))]
fn equilibrate_topology(
    path: &str,
    output_traj: Option<&str>,
    steps: Option<u64>,
    temperature: Option<f64>,
    restart_traj: Option<&str>,
    restraint_k: Option<f64>,
    traj_format: Option<&str>,
    checkpoint_out: Option<&str>,
    checkpoint_in: Option<&str>,
    energy_log: Option<&str>,
) -> PyResult<TopologyPy> {
    let mut top = Topology::read(path).map_err(|e| pyo3::exceptions::PyIOError::new_err(e))?;
    let reference: Vec<[f64; 3]> = top.atoms.iter().map(|a| a.position).collect();
    if let Some(restart) = restart_traj {
        TrajectoryReader::new(restart)
            .apply_last_frame(&mut top)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;
    }
    let n_steps = steps.unwrap_or(100);
    let temp = temperature.unwrap_or(300.0);
    let mut state = MdState::new(top.clone(), temp);
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
    for step in 0..n_steps {
        let res = langevin_step(&mut state, 1.0);
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
    Ok(TopologyPy { inner: top })
}

#[pyfunction]
#[pyo3(signature = (path, output_traj=None, steps=None, temperature=None, restart_traj=None, npt=None, traj_format=None, checkpoint_out=None, checkpoint_in=None, energy_log=None))]
fn simulate_topology(
    path: &str,
    output_traj: Option<&str>,
    steps: Option<u64>,
    temperature: Option<f64>,
    restart_traj: Option<&str>,
    npt: Option<bool>,
    traj_format: Option<&str>,
    checkpoint_out: Option<&str>,
    checkpoint_in: Option<&str>,
    energy_log: Option<&str>,
) -> PyResult<TopologyPy> {
    let mut top = Topology::read(path).map_err(|e| pyo3::exceptions::PyIOError::new_err(e))?;
    if let Some(restart) = restart_traj {
        TrajectoryReader::new(restart)
            .apply_last_frame(&mut top)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;
    }
    let n_steps = steps.unwrap_or(100);
    let temp = temperature.unwrap_or(300.0);
    let mut state = MdState::new(top.clone(), temp);
    state.use_constraints = true;
    state.npt = npt.unwrap_or(false);
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
    let mut writer = make_traj_writer(output_traj, traj_format)?;
    let mut elog = energy_log
        .map(|p| EnergyLogWriter::new(p, false))
        .transpose()
        .map_err(pyo3::exceptions::PyIOError::new_err)?;
    for step in 0..n_steps {
        let res = velocity_verlet_step(&mut state, 1.0);
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
    Ok(TopologyPy { inner: top })
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
    m.add_function(wrap_pyfunction_bound!(ionize_topology, &m)?)?;
    m.add_function(wrap_pyfunction_bound!(load_topology, &m)?)?;
    m.add_function(wrap_pyfunction_bound!(minimize_topology, &m)?)?;
    m.add_function(wrap_pyfunction_bound!(equilibrate_topology, &m)?)?;
    m.add_function(wrap_pyfunction_bound!(simulate_topology, &m)?)?;
    m.add_function(wrap_pyfunction_bound!(read_dcd_trajectory, &m)?)?;
    m.add_function(wrap_pyfunction_bound!(write_checkpoint_file, &m)?)?;
    m.add_function(wrap_pyfunction_bound!(read_checkpoint_file, &m)?)?;
    m.add_wrapped(wrap_pymodule!(crate::utilities::utilities))?;
    m.add_wrapped(wrap_pymodule!(build))?;
    m.add_wrapped(wrap_pymodule!(simulate))?;
    topology::register_module(&m)?;
    Ok(())
}
