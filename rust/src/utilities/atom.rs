use std::collections::{HashMap, HashSet};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::Read;

use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::utilities::{get_data_path};

// Define the atom properties as structs
#[derive(Serialize, Deserialize, Debug)]
pub struct AtomProperties {
    color: (f32, f32, f32),
    radius: f32,
    valence: usize,
}

impl ToPyObject for AtomProperties {
    fn to_object(&self, py: Python) -> PyObject {
        let dict = PyDict::new(py);
        dict.set_item("color", self.color.to_object(py)).unwrap();
        dict.set_item("radius", self.radius.to_object(py)).unwrap();
        dict.set_item("valence", self.valence.to_object(py)).unwrap();
        dict.into()
    }
}



lazy_static::lazy_static! {
    pub(crate) static ref ATOM_PROPERTIES: HashMap<String, AtomProperties> = load_atom_data();
}
pub(crate) fn get_atom_properties_path() -> String {
    let data_path = get_data_path();
    format!("{}/atom_properties.yml", data_path)
}

pub(crate) fn load_atom_data() -> HashMap<String, AtomProperties> {
    let path = get_atom_properties_path();
    println!("Loading atom data from: {}", path);
    let mut file = File::open(path).unwrap();
    let mut contents = String::new();
    file.read_to_string(&mut contents).unwrap();
    let data: HashMap<String, AtomProperties> = serde_yaml::from_str(&contents).unwrap();
    data
}