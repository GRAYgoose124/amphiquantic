use pyo3::prelude::*;
use pyo3::wrap_pymodule;
use pyo3::Python;
use pyo3::types::PyDict;
use ndarray::Array2;
use numpy::{PyArray2, IntoPyArray};
use std::collections::HashSet;

mod atom;
mod bonds;
mod pdb;
mod utils;
mod builder;
mod minimization;
mod simulation;

use bonds::load_bond_data;
use atom::load_atom_data;
use pdb::PdbFilePy;
use builder::builder as build;
use simulation::{run_simulation, SimulationParams};
use minimization::{minimize_energy, MinimizationParams};
use utils::{get_data_path, get_bond_distances_path, get_atom_properties_path};

#[pymodule]
fn utilities(_py: Python, m: &PyModule) -> PyResult<()> {
    #[pyfn(m, name = "get_data_path")]
    fn get_data_path_py(_py: Python) -> String {
        get_data_path()
    }

    #[pyfn(m, name = "get_bond_distances_path")]
    fn get_bond_distances_path_py(_py: Python) -> String {
        get_bond_distances_path()
    }

    #[pyfn(m, name = "get_atom_properties_path")]
    fn get_atom_properties_path_py(_py: Python) -> String {
        get_atom_properties_path()
    }

    #[pyfn(m, name = "load_bond_distances")]
    fn load_bond_data_py(_py: Python) -> Py<PyDict> {
        let bond_data = load_bond_data();
        let dict = PyDict::new(_py);
        for (key, value) in bond_data.iter() {
            dict.set_item(key, value).unwrap();
        }
        dict.into()
    }

    #[pyfn(m, name = "load_atom_properties")]
    fn load_atom_data_py(_py: Python) -> Py<PyDict> {
        let atom_data = load_atom_data();
        let dict = PyDict::new(_py);
        for (key, value) in atom_data.iter() {
            dict.set_item(key, value.to_object(_py)).unwrap();
        }
        dict.into()
    }



    m.add_wrapped(wrap_pyfunction!(get_data_path_py))?;
    m.add_wrapped(wrap_pyfunction!(get_bond_distances_path_py))?;
    m.add_wrapped(wrap_pyfunction!(get_atom_properties_path_py))?;

    Ok(())
}

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
    m.add_wrapped(wrap_pymodule!(utilities))?;
    m.add_wrapped(wrap_pymodule!(build))?;
    m.add_wrapped(wrap_pymodule!(simulate))?;

    Ok(())
}
