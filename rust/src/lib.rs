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
fn simulate(_py: Python, m: &PyModule) -> PyResult<()> {
    #[pyfn(m, name = "minimize_energy")]
    fn minimize_energy_py(coords: Vec<(f64, f64, f64)>, step_size: f32, max_steps: u32) -> Vec<(f64, f64, f64)> {
        let mut coords = coords;
        let params = MinimizationParams {
            step_size,
            max_steps,
        };
        minimize_energy(&mut coords, params);
        coords
    }
    
    #[pyfn(m, name = "run_simulation")]
    fn run_simulation_py(coords: Vec<(f64, f64, f64)>, time_step: f32, num_steps: u32) -> Vec<(f64, f64, f64)> {
        let mut coords = coords;
        let params = SimulationParams {
            time_step,
            num_steps,
        };
        run_simulation(&mut coords, params);
        coords
    }

    Ok(())
}

#[pymodule]
fn rustquantic(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PdbFilePy>()?;
    m.add_wrapped(wrap_pymodule!(crate::utilities::utilities))?;
    m.add_wrapped(wrap_pymodule!(build))?;
    m.add_wrapped(wrap_pymodule!(simulate))?;

    Ok(())
}
