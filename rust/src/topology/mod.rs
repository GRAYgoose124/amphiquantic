use pyo3::prelude::*;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;

pub const AQTOP_VERSION: u32 = 1;

/// OBC-II effective Born radius from intrinsic radius (nm).
fn obc2_effective_born_r(rho: f64) -> f64 {
    let rho = rho.max(0.05);
    let one = rho;
    let psi = one * one * rho;
    let psi = rho - psi;
    let tanh = (psi * 0.8).tanh();
    (rho - 0.005909 * tanh).max(0.05)
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TopologyMetadata {
    pub openff_version: Option<String>,
    pub source: Option<String>,
    pub smiles: Option<String>,
    #[serde(default)]
    pub solvation_model: Option<String>,
    #[serde(default)]
    pub dielectric: Option<f64>,
}

impl Default for TopologyMetadata {
    fn default() -> Self {
        Self {
            openff_version: None,
            source: None,
            smiles: None,
            solvation_model: None,
            dielectric: None,
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
    #[serde(default)]
    pub born_r: Option<f64>,
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
    #[serde(default)]
    pub impropers: Vec<DihedralTerm>,
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
            impropers: Vec::new(),
            exclusions: Vec::new(),
        }
    }

    pub fn net_charge(&self) -> f64 {
        self.atoms.iter().map(|a| a.charge).sum()
    }

    pub fn is_implicit_gb(&self) -> bool {
        matches!(
            self.metadata.solvation_model.as_deref(),
            Some("gb_obc2") | Some("gb")
        )
    }

    pub fn solvent_dielectric(&self) -> f64 {
        self.metadata.dielectric.unwrap_or(78.5)
    }

    pub fn effective_born_r(&self, idx: usize) -> f64 {
        let atom = &self.atoms[idx];
        let rho = atom.born_r.unwrap_or(0.15);
        obc2_effective_born_r(rho)
    }

    /// Virtual charge neutralization for implicit solvent (no ion atoms).
    pub fn neutralize_virtual(&mut self) {
        let charge = self.net_charge();
        if charge.abs() < 1e-9 || self.atoms.is_empty() {
            return;
        }
        let delta = -charge / self.atoms.len() as f64;
        for atom in &mut self.atoms {
            atom.charge += delta;
        }
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
        let mut top: Self = serde_json::from_reader(reader).map_err(|e| e.to_string())?;
        if top.exclusions.is_empty() && (!top.bonds.is_empty() || !top.angles.is_empty()) {
            top.build_exclusions();
        }
        Ok(top)
    }

    pub fn write(&self, path: impl AsRef<Path>) -> Result<(), String> {
        let mut top = self.clone();
        top.sanitize_non_finite();
        let file = File::create(path.as_ref()).map_err(|e| e.to_string())?;
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, &top).map_err(|e| e.to_string())
    }

    /// Replace NaN/Inf coords and box lengths so JSON round-trips stay valid.
    pub fn sanitize_non_finite(&mut self) {
        for atom in &mut self.atoms {
            for c in &mut atom.position {
                if !c.is_finite() {
                    *c = 0.0;
                }
            }
        }
        if !self.box_.lx.is_finite() {
            self.box_.lx = 0.0;
        }
        if !self.box_.ly.is_finite() {
            self.box_.ly = 0.0;
        }
        if !self.box_.lz.is_finite() {
            self.box_.lz = 0.0;
        }
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
                born_r: None,
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

    /// Add counter-ions in the simulation box, preserving all bonded terms.
    pub fn add_ions(&mut self, ion: &str, count: usize) {
        if count == 0 {
            return;
        }
        let params = ion_params(ion);
        let mol_id = self
            .atoms
            .iter()
            .map(|a| a.molecule_id)
            .max()
            .unwrap_or(0)
            + 1;
        let (min_x, max_x, min_y, max_y, min_z, max_z) = self.bounding_box();
        let mut rng = rand::thread_rng();
        for idx in 0..count {
            let pos = random_position_in_box(
                min_x, max_x, min_y, max_y, min_z, max_z,
                &self.atoms,
                &mut rng,
            );
            self.atoms.push(AtomRecord {
                element: params.element.to_string(),
                name: format!("{}{}", params.element, self.atoms.len() + 1),
                mass: params.mass,
                charge: params.charge,
                sigma: params.sigma,
                epsilon: params.epsilon,
                position: pos,
                residue_id: 0,
                molecule_id: mol_id + idx as u32,
                born_r: None,
            });
        }
    }

    pub fn neutralize(&mut self) {
        if self.is_implicit_gb() {
            self.neutralize_virtual();
            return;
        }
        let charge = self.net_charge();
        let n = charge.round().abs() as usize;
        if n == 0 {
            return;
        }
        let ion = if charge > 0.0 { "Cl" } else { "Na" };
        self.add_ions(ion, n);
    }

    fn bounding_box(&self) -> (f64, f64, f64, f64, f64, f64) {
        if self.box_.pbc && self.box_.lx > 0.0 {
            return (0.0, self.box_.lx, 0.0, self.box_.ly, 0.0, self.box_.lz);
        }
        let xs: Vec<f64> = self.atoms.iter().map(|a| a.position[0]).collect();
        let ys: Vec<f64> = self.atoms.iter().map(|a| a.position[1]).collect();
        let zs: Vec<f64> = self.atoms.iter().map(|a| a.position[2]).collect();
        let pad = 0.5;
        (
            xs.iter().copied().fold(f64::INFINITY, f64::min) - pad,
            xs.iter().copied().fold(f64::NEG_INFINITY, f64::max) + pad,
            ys.iter().copied().fold(f64::INFINITY, f64::min) - pad,
            ys.iter().copied().fold(f64::NEG_INFINITY, f64::max) + pad,
            zs.iter().copied().fold(f64::INFINITY, f64::min) - pad,
            zs.iter().copied().fold(f64::NEG_INFINITY, f64::max) + pad,
        )
    }
}

impl Default for Topology {
    fn default() -> Self {
        Self::new()
    }
}

fn default_mass(element: &str) -> f64 {
    ion_params(element).mass
}

struct IonParams {
    element: &'static str,
    mass: f64,
    charge: f64,
    sigma: f64,
    epsilon: f64,
}

fn ion_params(element: &str) -> IonParams {
    match element {
        "Na" => IonParams {
            element: "Na",
            mass: 22.99,
            charge: 1.0,
            sigma: 0.259,
            epsilon: 0.011,
        },
        "Cl" => IonParams {
            element: "Cl",
            mass: 35.45,
            charge: -1.0,
            sigma: 0.347,
            epsilon: 1.108,
        },
        "H" => IonParams {
            element: "H",
            mass: 1.008,
            charge: 0.0,
            sigma: 0.106,
            epsilon: 0.066,
        },
        "C" => IonParams {
            element: "C",
            mass: 12.011,
            charge: 0.0,
            sigma: 0.34,
            epsilon: 0.36,
        },
        "N" => IonParams {
            element: "N",
            mass: 14.007,
            charge: 0.0,
            sigma: 0.325,
            epsilon: 0.711,
        },
        "O" => IonParams {
            element: "O",
            mass: 15.999,
            charge: 0.0,
            sigma: 0.296,
            epsilon: 0.879,
        },
        "S" => IonParams {
            element: "S",
            mass: 32.06,
            charge: 0.0,
            sigma: 0.356,
            epsilon: 1.046,
        },
        "P" => IonParams {
            element: "P",
            mass: 30.974,
            charge: 0.0,
            sigma: 0.374,
            epsilon: 0.836,
        },
        "Ca" => IonParams {
            element: "Ca",
            mass: 40.078,
            charge: 2.0,
            sigma: 0.299,
            epsilon: 0.011,
        },
        _ => IonParams {
            element: "X",
            mass: 1.0,
            charge: 0.0,
            sigma: 0.34,
            epsilon: 0.36,
        },
    }
}

fn random_position_in_box<R: Rng>(
    min_x: f64,
    max_x: f64,
    min_y: f64,
    max_y: f64,
    min_z: f64,
    max_z: f64,
    atoms: &[AtomRecord],
    rng: &mut R,
) -> [f64; 3] {
    for _ in 0..100 {
        let x = rng.gen_range(min_x..max_x);
        let y = rng.gen_range(min_y..max_y);
        let z = rng.gen_range(min_z..max_z);
        let too_close = atoms.iter().any(|a| {
            let dx = a.position[0] - x;
            let dy = a.position[1] - y;
            let dz = a.position[2] - z;
            (dx * dx + dy * dy + dz * dz).sqrt() < 2.0
        });
        if !too_close {
            return [x, y, z];
        }
    }
    [
        rng.gen_range(min_x..max_x),
        rng.gen_range(min_y..max_y),
        rng.gen_range(min_z..max_z),
    ]
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
            born_r: None,
        });
        let json = serde_json::to_string(&top).unwrap();
        let parsed: Topology = serde_json::from_str(&json).unwrap();
        assert_eq!(top, parsed);
    }

    #[test]
    fn read_builds_missing_exclusions() {
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
            born_r: None,
        });
        top.atoms.push(AtomRecord {
            element: "C".into(),
            name: "C2".into(),
            mass: 12.0,
            charge: 0.0,
            sigma: 0.34,
            epsilon: 0.36,
            position: [0.15, 0.0, 0.0],
            residue_id: 0,
            molecule_id: 0,
            born_r: None,
        });
        top.bonds.push(BondTerm {
            i: 0,
            j: 1,
            k: 500.0,
            r0: 0.15,
        });
        top.exclusions.clear();
        let path = std::env::temp_dir().join("amphi_excl_test.aqtop");
        top.write(&path).unwrap();
        let loaded = Topology::read(&path).unwrap();
        assert!(!loaded.exclusions.is_empty());
        let _ = std::fs::remove_file(path);
    }
}
