use pyo3::prelude::*;
use pyo3::types::PyTuple;
use std::fs::File;
use std::io::{BufReader, BufRead, BufWriter, Write};
use std::collections::HashSet;


use crate::bonds::determine_bonds;

#[pyclass]
pub struct PdbFilePy {
    #[pyo3(get)]
    pub coords: Vec<(f64, f64, f64)>,
    #[pyo3(get)]
    pub atom_types: Vec<String>,
    #[pyo3(get)]
    pub bonds: Vec<(usize, usize)>,
}


#[pyfunction]
pub fn parse_pdb(file_path: &str) -> PdbFilePy {
    let file = File::open(file_path).unwrap();
    let reader = BufReader::new(file);
    let mut coords = Vec::new();
    let mut atom_types = Vec::new();
    let mut bonds = Vec::new();
    for line in reader.lines() {
        let line = line.unwrap();
        // Handle both ATOM and HETATM records
        if line.starts_with("ATOM") || line.starts_with("HETATM") {
            let atom_type = line[76..78].trim().to_string();
            let x = line[30..38].trim().parse::<f64>().unwrap();
            let y = line[38..46].trim().parse::<f64>().unwrap();
            let z = line[46..54].trim().parse::<f64>().unwrap();
            coords.push((x, y, z));
            atom_types.push(atom_type);
        }
        // Handle CONECT records for bonds
        if line.starts_with("CONECT") {
            let mut split = line.split_whitespace();
            split.next();
            let from = split.next().unwrap().parse::<usize>().unwrap() - 1;
            let mut to = Vec::new();
            while let Some(t) = split.next() {
                to.push(t.parse::<usize>().unwrap() - 1);
            }
            for t in to {
                bonds.push((from, t));
            }
        }
    }

    PdbFilePy {
        coords,
        atom_types,
        bonds,
    }
}

#[pyfunction]
pub fn adjust_coordinates_tuple(
    coords: Vec<(f64, f64, f64)>,
    fill_size: &PyTuple,
    margin: &PyTuple
) -> Vec<(f64, f64, f64)> {
    let fill_size: (f64, f64) = (
        fill_size.get_item(0).unwrap().extract().unwrap(),
        fill_size.get_item(1).unwrap().extract().unwrap()
    );
    let margin: (f64, f64) = (
        margin.get_item(0).unwrap().extract().unwrap(),
        margin.get_item(1).unwrap().extract().unwrap()
    );

    let min_vals = (
        coords.iter().map(|c| c.0).min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap(),
        coords.iter().map(|c| c.1).min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap(),
        coords.iter().map(|c| c.2).min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap(),
    );
    let max_vals = (
        coords.iter().map(|c| c.0).max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap(),
        coords.iter().map(|c| c.1).max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap(),
        coords.iter().map(|c| c.2).max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap(),
    );
    let fill_size: (f64, f64) = (
        fill_size.0 + margin.0 * 2.0,
        fill_size.1 + margin.1 * 2.0,
    );
    let scale: (f64, f64) = (
        fill_size.0 / (max_vals.0 - min_vals.0),
        fill_size.1 / (max_vals.1 - min_vals.1),
    );
    let mut adjusted_coords = Vec::new();
    for (x, y, z) in &coords {
        let x = (x - min_vals.0) * scale.0 + margin.0;
        let y = (y - min_vals.1) * scale.1 + margin.1;
        adjusted_coords.push((x, y, *z));
    }
    adjusted_coords
}

// write pdb - with optional bonds
pub fn write_pdb(file_path: &str, coords: Vec<(f64, f64, f64)>, atom_types: Vec<String>, bonds: Option<Vec<(usize, usize)>>) {
    let file = File::create(file_path).unwrap();
    let mut writer = BufWriter::new(file);
    for (i, (x, y, z)) in coords.iter().enumerate() {
        let atom_type = atom_types[i].clone();
        writeln!(writer, "ATOM  {:>5} {:<3} MOL     1    {:>8.3} {:>8.3} {:>8.3}  1.00  0.00          {:>2}", i + 1, atom_type, x, y, z, atom_type).unwrap();
    }
    if let Some(bonds) = bonds {
        for (from, to) in bonds {
            writeln!(writer, "CONECT {:>5} {:>5}", from + 1, to + 1).unwrap();
        }
    }
}

#[pymethods]
impl PdbFilePy {
    #[new]
    pub fn new(coords: Vec<(f64, f64, f64)>, atom_types: Vec<String>, bonds: Vec<(usize, usize)>) -> Self {
        PdbFilePy {
            coords,
            atom_types,
            bonds,
        }
    }


    #[staticmethod]
    pub fn parse(file_path: &str) -> PdbFilePy {
        parse_pdb(file_path)
    }


    pub fn adjust_coordinates(&mut self, fill_size: &PyTuple, margin: &PyTuple) {
        self.coords = adjust_coordinates_tuple(self.coords.clone(), fill_size, margin);
    }

    pub fn determine_bonds(&self) -> (Vec<(usize, usize)>, Vec<(usize, usize)>, HashSet<(String, String)>) {
        determine_bonds(self.coords.clone(), self.atom_types.clone())
    }

    pub fn set_bonds(&mut self, bonds: Vec<(usize, usize)>) {
        self.bonds = bonds;
    }

    pub fn write(&self, file_path: &str, write_bonds: bool) {
        write_pdb(file_path, self.coords.clone(), self.atom_types.clone(), if write_bonds { Some(self.bonds.clone()) } else { None });
    }
}