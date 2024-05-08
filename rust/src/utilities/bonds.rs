use std::collections::{HashMap};
use std::fs::File;
use std::io::Read;
use serde::{Deserialize, Serialize};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::utilities::{get_data_path};


#[derive(Serialize, Deserialize, Debug)]
pub struct BondDistances {
    bond_distances: HashMap<String, (f64, f64)>,
}

impl ToPyObject for BondDistances {
    fn to_object(&self, py: Python) -> PyObject {
        let dict = PyDict::new(py);
        for (key, value) in self.bond_distances.iter() {
            dict.set_item(key, value).unwrap();
        }
        dict.into()
    }
}

lazy_static::lazy_static! {
    pub(crate) static ref BOND_DISTANCES: HashMap<String, (f64, f64)> = load_bond_data();
    pub(crate) static ref AVG_BOND_DISTANCES: HashMap<String, f64> = {
        let mut avg_bond_distances = HashMap::new();
        for (k, (min, max)) in BOND_DISTANCES.iter() {
            avg_bond_distances.insert(k.clone(), (min + max) / 2.0);
        }
        avg_bond_distances
    };
}

pub(crate) fn get_bond_distances_path() -> String {
    let data_path = get_data_path();
    format!("{}/bond_distances.yml", data_path)
}


pub(crate) fn load_bond_data() -> HashMap<String, (f64, f64)> {
    let path = get_bond_distances_path();
    println!("Loading bond data from: {}", path);
    let mut file = File::open(path).unwrap();
    let mut contents = String::new();
    file.read_to_string(&mut contents).unwrap();
    let data: BondDistances = serde_yaml::from_str(&contents).unwrap();
    data.bond_distances
}

