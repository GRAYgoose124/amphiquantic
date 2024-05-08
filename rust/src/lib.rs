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

use bonds::{determine_bonds, load_bond_data};
use atom::load_atom_data;
use pdb::PdbFilePy;
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
fn rustquantic(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PdbFilePy>()?;
    m.add_wrapped(wrap_pymodule!(utilities))?;

    Ok(())
}
