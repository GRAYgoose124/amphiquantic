use pyo3::prelude::*;
use pyo3::wrap_pymodule;

use pyo3::Python;


mod atom;
mod bonds;
mod pdb;
mod builder;
mod minimization;
mod simulation;
mod utilities;


use pdb::PdbFilePy;
use builder::builder as build;
use simulation::{run_simulation, SimulationParams};
use minimization::{minimize_energy, MinimizationParams};


#[pymodule]
fn simulate(_py: Python, m: Bound<PyModule>) -> PyResult<()> {
    #[pyfn(m, name = "minimize_energy")]
    fn minimize_energy_py(coords: Vec<(f64, f64, f64)>, step_size: f32, max_steps: u32) -> Vec<(f64, f64, f64)> {
        let params = MinimizationParams {
            step_size,
            max_steps,
        };

        let mut coords_vec: Vec<[f64; 3]> = coords.iter().map(|c| [c.0, c.1, c.2]).collect();
        minimize_energy(&mut coords_vec, params);
        coords_vec.iter().map(|c| (c[0], c[1], c[2])).collect()
    }
    
    #[pyfn(m, name = "run_simulation")]
    fn run_simulation_py(coords: Vec<(f64, f64, f64)>, time_step: f32, num_steps: u32) -> Vec<(f64, f64, f64)> {
        let coords = coords;
        let params = SimulationParams {
            step_size: time_step,
            max_steps: num_steps,
        };

        let mut coords_vec: Vec<[f64; 3]> = coords.iter().map(|c| [c.0, c.1, c.2]).collect();
        run_simulation(&mut coords_vec, params);
        coords_vec.iter().map(|c| (c[0], c[1], c[2])).collect()
    }

    Ok(())
}

#[pymodule]
fn rustquantic(_py: Python, m: Bound<PyModule>) -> PyResult<()> {
    m.add_class::<PdbFilePy>()?;
    m.add_wrapped(wrap_pymodule!(crate::utilities::utilities))?;
    m.add_wrapped(wrap_pymodule!(build))?;
    m.add_wrapped(wrap_pymodule!(simulate))?;

    Ok(())
}
