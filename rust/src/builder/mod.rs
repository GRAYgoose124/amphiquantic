use pyo3::prelude::*;
use pyo3::wrap_pymodule;
use pyo3::Python;
use pyo3::types::PyDict;

pub mod ions;
pub mod solvation;

use ions::add_ions;
use solvation::solvate_box;

#[pymodule]
fn builder(_py: Python, m: &PyModule) -> PyResult<()> {
    #[pyfn(m, name = "add_ions")]
    fn add_ions_py(coords: Vec<(f64, f64, f64)>, atom_types: Vec<String>, ion: &str, number: usize) -> (Vec<(f64, f64, f64)>, Vec<String>) {
        let mut coords = coords;
        let mut atom_types = atom_types;
        add_ions(&mut coords, &mut atom_types, ion, number);
        (coords, atom_types)
    }

    #[pyfn(m, name = "solvate_box")]
    fn solvate_box_py(coords: Vec<(f64, f64, f64)>, atom_types: Vec<String>, box_size: f64) -> (Vec<(f64, f64, f64)>, Vec<String>) {
        let mut coords = coords;
        let mut atom_types = atom_types;
        solvate_box(&mut coords, &mut atom_types, box_size);
        (coords, atom_types)
    }

    Ok(())
}