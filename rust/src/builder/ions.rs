use rand::Rng;
use std::collections::HashMap;

use crate::utilities::bonds::{BOND_DISTANCES, AVG_BOND_DISTANCES};

/// Add ions to a structure to neutralize the system or achieve a specific concentration.
pub fn add_ions(
    coords: &mut Vec<(f64, f64, f64)>,
    atom_types: &mut Vec<String>,
    ion: &str,
    number: usize,
) {
    let mut rng = rand::thread_rng();
    let min_x = coords.iter().map(|c| c.0).min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    let min_y = coords.iter().map(|c| c.1).min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    let min_z = coords.iter().map(|c| c.2).min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    let max_x = coords.iter().map(|c| c.0).max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    let max_y = coords.iter().map(|c| c.1).max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    let max_z = coords.iter().map(|c| c.2).max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();

    for _ in 0..number {
        let x = rng.gen_range(min_x..max_x);
        let y = rng.gen_range(min_y..max_y);
        let z = rng.gen_range(min_z..max_z);
        coords.push((x, y, z));
        atom_types.push(ion.to_string());
    }
}


// Find realistic locations for ions in a structure.

pub fn find_possible_ion_locations(
    coords: &Vec<(f64, f64, f64)>,
    atom_types: &mut Vec<String>,
) -> Vec<(f64, f64, f64)> {
   // don't randomly place, use bond data to find possible locations
    let _bond_distances = BOND_DISTANCES.iter().map(|(k, v)| (k.clone(), v.clone())).collect::<HashMap<String, (f64, f64)>>();
    let avg_bond_distances = AVG_BOND_DISTANCES.iter().map(|(k, v)| (k.clone(), *v)).collect::<HashMap<String, f64>>();

    let threshold = 0.2;
    let mut possible_locations = vec![];
    let num_atoms = coords.len();

    for i in 0..num_atoms {
        for j in 0..num_atoms {
            if i == j {
                continue;
            }
            let pair = format!("{}-{}", atom_types[i], atom_types[j]);
            let typical_dist_key = avg_bond_distances.get(&pair).or_else(|| avg_bond_distances.get(&format!("{}-{}", atom_types[j], atom_types[i])));
            if let Some(&typ) = typical_dist_key {
                let dist = ((coords[i].0 - coords[j].0).powi(2) +
                            (coords[i].1 - coords[j].1).powi(2) +
                            (coords[i].2 - coords[j].2).powi(2)).sqrt();
                if (dist - typ).abs() <= threshold {
                    let new_coord = (
                        (coords[i].0 + coords[j].0) / 2.0,
                        (coords[i].1 + coords[j].1) / 2.0,
                        (coords[i].2 + coords[j].2) / 2.0,
                    );
                    possible_locations.push(new_coord);
                }
            }
        }
    }

    possible_locations
}