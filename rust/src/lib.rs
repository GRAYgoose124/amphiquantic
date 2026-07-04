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
use integrator::{langevin_step, velocity_verlet_step, MdState};
use minimize::minimize;
use pdb::PdbFilePy;
use topology::{Topology, TopologyPy};
use trajectory::TrajectoryWriter;

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

#[pyfunction]
#[pyo3(signature = (path, output_traj=None, steps=None, temperature=None))]
fn equilibrate_topology(
    path: &str,
    output_traj: Option<&str>,
    steps: Option<u64>,
    temperature: Option<f64>,
) -> PyResult<TopologyPy> {
    let mut top = Topology::read(path).map_err(|e| pyo3::exceptions::PyIOError::new_err(e))?;
    let n_steps = steps.unwrap_or(100);
    let temp = temperature.unwrap_or(300.0);
    let mut state = MdState::new(top.clone(), temp);
    let mut writer = output_traj.map(TrajectoryWriter::new);
    for step in 0..n_steps {
        let res = langevin_step(&mut state, 1.0);
        if let Some(w) = writer.as_mut() {
            w.write_frame(step, &state.topology, res.potential_energy, res.kinetic_energy)
                .map_err(|e| pyo3::exceptions::PyIOError::new_err(e))?;
        }
    }
    top = state.topology;
    Ok(TopologyPy { inner: top })
}

#[pyfunction]
#[pyo3(signature = (path, output_traj=None, steps=None, temperature=None))]
fn simulate_topology(
    path: &str,
    output_traj: Option<&str>,
    steps: Option<u64>,
    temperature: Option<f64>,
) -> PyResult<TopologyPy> {
    let mut top = Topology::read(path).map_err(|e| pyo3::exceptions::PyIOError::new_err(e))?;
    let n_steps = steps.unwrap_or(100);
    let temp = temperature.unwrap_or(300.0);
    let mut state = MdState::new(top.clone(), temp);
    state.use_constraints = true;
    state.npt = false;
    let mut writer = output_traj.map(TrajectoryWriter::new);
    for step in 0..n_steps {
        let res = velocity_verlet_step(&mut state, 1.0);
        if let Some(w) = writer.as_mut() {
            w.write_frame(step, &state.topology, res.potential_energy, res.kinetic_energy)
                .map_err(|e| pyo3::exceptions::PyIOError::new_err(e))?;
        }
    }
    top = state.topology;
    Ok(TopologyPy { inner: top })
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
    m.add_function(wrap_pyfunction_bound!(ionize_topology, &m)?)?;
    m.add_function(wrap_pyfunction_bound!(load_topology, &m)?)?;
    m.add_function(wrap_pyfunction_bound!(minimize_topology, &m)?)?;
    m.add_function(wrap_pyfunction_bound!(equilibrate_topology, &m)?)?;
    m.add_function(wrap_pyfunction_bound!(simulate_topology, &m)?)?;
    m.add_wrapped(wrap_pymodule!(crate::utilities::utilities))?;
    m.add_wrapped(wrap_pymodule!(build))?;
    m.add_wrapped(wrap_pymodule!(simulate))?;
    topology::register_module(&m)?;
    Ok(())
}
