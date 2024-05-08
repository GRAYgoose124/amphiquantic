use rand::Rng;

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
