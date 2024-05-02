use std::fs::File;
use std::io::{BufReader, BufRead};


pub(crate) fn parse_pdb_file(file_path: &str) -> ((Vec<(f64, f64, f64)>, Vec<String>), Vec<(usize, usize)>) {
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
    ((coords, atom_types), bonds)
}


pub(crate) fn adjust_coordinates(
    raw_coords: Vec<(f64, f64, f64)>,
    fill_size: (f64, f64),
    margin: (f64, f64),
) -> Vec<(f64, f64, f64)> {
    let coords = raw_coords.clone();
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
    let fill_size = (
        fill_size.0 + margin.0 * 2.0,
        fill_size.1 + margin.1 * 2.0,
    );
    let scale = (
        fill_size.0 / (max_vals.0 - min_vals.0),
        fill_size.1 / (max_vals.1 - min_vals.1),
    );
    let mut adjusted_coords = Vec::new();
    for (x, y, z) in coords {
        let x = (x - min_vals.0) * scale.0 + margin.0;
        let y = (y - min_vals.1) * scale.1 + margin.1;
        adjusted_coords.push((x, y, z));
    }
    adjusted_coords
}