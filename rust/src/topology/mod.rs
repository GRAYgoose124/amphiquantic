use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;

pub const AQTOP_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TopologyMetadata {
    pub openff_version: Option<String>,
    pub source: Option<String>,
    pub smiles: Option<String>,
}

impl Default for TopologyMetadata {
    fn default() -> Self {
        Self {
            openff_version: None,
            source: None,
            smiles: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SimulationBox {
    pub lx: f64,
    pub ly: f64,
    pub lz: f64,
    pub pbc: bool,
}

impl Default for SimulationBox {
    fn default() -> Self {
        Self {
            lx: 0.0,
            ly: 0.0,
            lz: 0.0,
            pbc: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AtomRecord {
    pub element: String,
    pub name: String,
    pub mass: f64,
    pub charge: f64,
    pub sigma: f64,
    pub epsilon: f64,
    pub position: [f64; 3],
    pub residue_id: u32,
    pub molecule_id: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BondTerm {
    pub i: usize,
    pub j: usize,
    pub k: f64,
    pub r0: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AngleTerm {
    pub i: usize,
    pub j: usize,
    pub k: usize,
    pub k_theta: f64,
    pub theta0: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DihedralTerm {
    pub i: usize,
    pub j: usize,
    pub k: usize,
    pub l: usize,
    pub k_phi: f64,
    pub n: i32,
    pub delta: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Topology {
    pub version: u32,
    pub metadata: TopologyMetadata,
    #[serde(rename = "box")]
    pub box_: SimulationBox,
    pub atoms: Vec<AtomRecord>,
    pub bonds: Vec<BondTerm>,
    pub angles: Vec<AngleTerm>,
    pub dihedrals: Vec<DihedralTerm>,
    pub exclusions: Vec<[usize; 2]>,
}

impl Topology {
    pub fn new() -> Self {
        Self {
            version: AQTOP_VERSION,
            metadata: TopologyMetadata::default(),
            box_: SimulationBox::default(),
            atoms: Vec::new(),
            bonds: Vec::new(),
            angles: Vec::new(),
            dihedrals: Vec::new(),
            exclusions: Vec::new(),
        }
    }

    pub fn net_charge(&self) -> f64 {
        self.atoms.iter().map(|a| a.charge).sum()
    }

    pub fn positions(&self) -> Vec<[f64; 3]> {
        self.atoms.iter().map(|a| a.position).collect()
    }

    pub fn set_positions(&mut self, coords: &[[f64; 3]]) {
        for (atom, coord) in self.atoms.iter_mut().zip(coords.iter()) {
            atom.position = *coord;
        }
    }

    pub fn read(path: impl AsRef<Path>) -> Result<Self, String> {
        let file = File::open(path.as_ref()).map_err(|e| e.to_string())?;
        let reader = BufReader::new(file);
        serde_json::from_reader(reader).map_err(|e| e.to_string())
    }

    pub fn write(&self, path: impl AsRef<Path>) -> Result<(), String> {
        let file = File::create(path.as_ref()).map_err(|e| e.to_string())?;
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, self).map_err(|e| e.to_string())
    }

    pub fn from_pdb_types(
        coords: &[(f64, f64, f64)],
        atom_types: &[String],
        bonds: &[(usize, usize)],
    ) -> Self {
        let mut top = Self::new();
        for (idx, ((x, y, z), element)) in coords.iter().zip(atom_types.iter()).enumerate() {
            top.atoms.push(AtomRecord {
                element: element.clone(),
                name: format!("{}{}", element, idx + 1),
                mass: default_mass(element),
                charge: 0.0,
                sigma: 0.34,
                epsilon: 0.36,
                position: [*x, *y, *z],
                residue_id: 0,
                molecule_id: 0,
            });
        }
        for &(i, j) in bonds {
            top.bonds.push(BondTerm {
                i,
                j,
                k: 500.0,
                r0: 0.15,
            });
        }
        top.build_exclusions();
        top
    }

    pub fn build_exclusions(&mut self) {
        self.exclusions.clear();
        for bond in &self.bonds {
            self.exclusions.push([bond.i, bond.j]);
            self.exclusions.push([bond.j, bond.i]);
        }
        for angle in &self.angles {
            self.exclusions.push([angle.i, angle.k]);
            self.exclusions.push([angle.k, angle.i]);
        }
    }

    pub fn to_pdb_types(&self) -> (Vec<(f64, f64, f64)>, Vec<String>, Vec<(usize, usize)>) {
        let coords: Vec<_> = self
            .atoms
            .iter()
            .map(|a| (a.position[0], a.position[1], a.position[2]))
            .collect();
        let atom_types: Vec<_> = self.atoms.iter().map(|a| a.element.clone()).collect();
        let bonds: Vec<_> = self.bonds.iter().map(|b| (b.i, b.j)).collect();
        (coords, atom_types, bonds)
    }
}

impl Default for Topology {
    fn default() -> Self {
        Self::new()
    }
}

fn default_mass(element: &str) -> f64 {
    match element {
        "H" => 1.008,
        "C" => 12.011,
        "N" => 14.007,
        "O" => 15.999,
        "S" => 32.06,
        "P" => 30.974,
        "Na" => 22.99,
        "Cl" => 35.45,
        "Ca" => 40.078,
        _ => 1.0,
    }
}

#[pyclass(name = "Topology")]
#[derive(Clone)]
pub struct TopologyPy {
    pub(crate) inner: Topology,
}

#[pymethods]
impl TopologyPy {
    #[new]
    fn new() -> Self {
        Self {
            inner: Topology::new(),
        }
    }

    #[staticmethod]
    pub fn read(path: &str) -> PyResult<Self> {
        Ok(Self {
            inner: Topology::read(path).map_err(|e| pyo3::exceptions::PyIOError::new_err(e))?,
        })
    }

    fn write(&self, path: &str) -> PyResult<()> {
        self.inner
            .write(path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e))
    }

    fn net_charge(&self) -> f64 {
        self.inner.net_charge()
    }

    fn num_atoms(&self) -> usize {
        self.inner.atoms.len()
    }

    fn to_pdb_types(&self) -> (Vec<(f64, f64, f64)>, Vec<String>, Vec<(usize, usize)>) {
        self.inner.to_pdb_types()
    }

    #[staticmethod]
    fn from_pdb_types(
        coords: Vec<(f64, f64, f64)>,
        atom_types: Vec<String>,
        bonds: Vec<(usize, usize)>,
    ) -> Self {
        Self {
            inner: Topology::from_pdb_types(&coords, &atom_types, &bonds),
        }
    }

    fn get_inner(&self) -> TopologyPy {
        self.clone()
    }
}

pub fn register_module(_parent: &Bound<'_, PyModule>) -> PyResult<()> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_json() {
        let mut top = Topology::new();
        top.atoms.push(AtomRecord {
            element: "C".into(),
            name: "C1".into(),
            mass: 12.0,
            charge: 0.0,
            sigma: 0.34,
            epsilon: 0.36,
            position: [0.0, 0.0, 0.0],
            residue_id: 0,
            molecule_id: 0,
        });
        let json = serde_json::to_string(&top).unwrap();
        let parsed: Topology = serde_json::from_str(&json).unwrap();
        assert_eq!(top, parsed);
    }
}
