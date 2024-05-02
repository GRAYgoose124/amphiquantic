use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::Read;
use serde::{Deserialize, Serialize};


use crate::utils::get_bond_distances_path;

#[derive(Serialize, Deserialize, Debug)]
struct BondDistances {
    bond_distances: HashMap<String, (f64, f64)>,
}

fn load_bond_data() -> HashMap<String, (f64, f64)> {
    let path = get_bond_distances_path();
    println!("Loading bond data from: {}", path);
    let mut file = File::open(path).unwrap();
    let mut contents = String::new();
    file.read_to_string(&mut contents).unwrap();
    let data: BondDistances = serde_yaml::from_str(&contents).unwrap();
    data.bond_distances
}

lazy_static::lazy_static! {
    static ref BOND_DISTANCES: HashMap<String, (f64, f64)> = load_bond_data();
    static ref AVG_BOND_DISTANCES: HashMap<String, f64> = {
        let mut avg_bond_distances = HashMap::new();
        for (k, (min, max)) in BOND_DISTANCES.iter() {
            avg_bond_distances.insert(k.clone(), (min + max) / 2.0);
        }
        avg_bond_distances
    };
}

pub(crate) fn determine_bonds(coords: Vec<(f64, f64, f64)>, atom_types: Vec<String>) -> (Vec<[usize; 2]>, Vec<(usize, usize)>, HashSet<(String, String)>) {
    let bond_distances = BOND_DISTANCES.iter().map(|(k, v)| (k.clone(), v.clone())).collect::<HashMap<String, (f64, f64)>>();
    let avg_bond_distances = AVG_BOND_DISTANCES.iter().map(|(k, v)| (k.clone(), *v)).collect::<HashMap<String, f64>>();

    let threshold = 0.2;
    let mut bonds_vec = Vec::new();
    let mut missing = HashSet::new();
    let mut near_bonds = Vec::new();
    let num_atoms = coords.len();

    for i in 0..num_atoms {
        for j in i + 1..num_atoms {
            let pair = format!("{}-{}", atom_types[i], atom_types[j]);
            let dist = ((coords[i].0 - coords[j].0).powi(2) +
                        (coords[i].1 - coords[j].1).powi(2) +
                        (coords[i].2 - coords[j].2).powi(2)).sqrt();
    
            if let Some((min_dist, max_dist)) = bond_distances.get(&pair).or_else(|| bond_distances.get(&format!("{}-{}", atom_types[j], atom_types[i]))) {
                if *min_dist <= dist && dist <= *max_dist {
                    bonds_vec.push([i, j]);
                }
            } else {
                let typical_dist_key = avg_bond_distances.get(&pair).or_else(|| avg_bond_distances.get(&format!("{}-{}", atom_types[j], atom_types[i])));
                if let Some(&typ) = typical_dist_key {
                    if (dist - typ).abs() <= threshold {
                        near_bonds.push((i, j));
                    }
                } else {
                    missing.insert((atom_types[i].clone(), atom_types[j].clone()));
                }
            }
        }
    }
    
    (bonds_vec, near_bonds, missing)
}
