use pyo3::prelude::*;
use pyo3::wrap_pymodule;
use pyo3::Python;
use ndarray::Array2;
use numpy::{PyArray2, IntoPyArray};
use std::collections::HashSet;

mod atom;
mod bonds;
mod pdb;
mod utils;

use bonds::determine_bonds;
use pdb::{parse_pdb_file, adjust_coordinates};
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

    Ok(())
}

#[pymodule]
fn pdbviz(_py: Python, m: &PyModule) -> PyResult<()> {
    #[pyfn(m, name = "determine_bonds")]
    fn determine_bonds_py(
        py: Python,
        coords: Vec<(f64, f64, f64)>,
        atom_types: Vec<String>,
    ) -> (Py<PyArray2<usize>>, Vec<(usize, usize)>, HashSet<(String, String)>) {
        let (bonds_vec, near_bonds, missing) = determine_bonds(coords, atom_types);
        let bonds_array = Array2::from_shape_vec((bonds_vec.len(), 2), bonds_vec.into_iter().flatten().collect::<Vec<_>>()).unwrap();
        let bonds_numpy = bonds_array.into_pyarray(py).to_owned();
        
        (bonds_numpy, near_bonds, missing)
    }

    #[pyfn(m, name = "parse_pdb_file")]
    fn parse_pdb_file_py(
        _py: Python,
        file_path: &str,
    ) -> (Vec<(f64, f64, f64)>, Vec<String>) {
        parse_pdb_file(file_path)
    }

    #[pyfn(m, name = "adjust_coordinates")]
    fn adjust_coordinates_py(
        py: Python,
        raw_coords: Vec<(f64, f64, f64)>,
        fill_size: (f64, f64),
        margin: (f64, f64),
    ) -> Py<PyArray2<f64>> {
        let adjusted_coords = adjust_coordinates(raw_coords, fill_size, margin);
        let adjusted_coords_array = Array2::from_shape_vec((adjusted_coords.len(), 3), adjusted_coords.into_iter().flat_map(|(x, y, z)| vec![x, y, z]).collect()).unwrap();
        adjusted_coords_array.into_pyarray(py).to_owned()
    }
   
    m.add_wrapped(wrap_pymodule!(utilities))?;

    Ok(())
}
