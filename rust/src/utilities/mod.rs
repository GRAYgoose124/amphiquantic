use std::env;
use pyo3::prelude::*;
use pyo3::types::PyDict;

pub mod atom;
pub mod shader;
pub mod bonds;


use bonds::{load_bond_data, get_bond_distances_path};
use atom::{load_atom_data, get_atom_properties_path};

pub(crate) fn get_data_path() -> String {
    // append bond_distances.yml to the data path
     env::var("PDBVIZ_DATA_PATH").unwrap_or_else(|_| "rust/data".to_string())
}





#[pymodule]
pub(crate) fn utilities(_py: Python, m: &PyModule) -> PyResult<()> {
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