use rand::Rng;

/// Add a box of water (or other solvent) around a molecular system.
pub fn solvate_box(
    coords: &mut Vec<(f64, f64, f64)>,
    atom_types: &mut Vec<String>,
    box_size: f64,
) {
    let mut rng = rand::thread_rng();
    let min_x = coords.iter().map(|c| c.0).min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    let min_y = coords.iter().map(|c| c.1).min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    let min_z = coords.iter().map(|c| c.2).min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    let max_x = coords.iter().map(|c| c.0).max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    let max_y = coords.iter().map(|c| c.1).max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    let max_z = coords.iter().map(|c| c.2).max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();

    let x_offset = max_x + box_size;
    let y_offset = max_y + box_size;
    let z_offset = max_z + box_size;
    let x_start = min_x - box_size;
    let y_start = min_y - box_size;
    let z_start = min_z - box_size;

    // Assuming TIP3P water model with a fixed geometry
    let tip3p_geometry = [
        ("O", (0.0, 0.0, 0.0)),
        ("H1", (0.9572, 0.0, 0.0)),
        ("H2", (-0.2399872, 0.92662721, 0.0)),
    ];

    for x in (x_start as isize..x_offset as isize).step_by(3) {
        for y in (y_start as isize..y_offset as isize).step_by(3) {
            for z in (z_start as isize..z_offset as isize).step_by(3) {
                let mut water_coords = vec![];
                for (atom, (dx, dy, dz)) in &tip3p_geometry {
                    let new_coord = (
                        x as f64 + dx + rng.gen_range(-0.1..0.1),
                        y as f64 + dy + rng.gen_range(-0.1..0.1),
                        z as f64 + dz + rng.gen_range(-0.1..0.1),
                    );
                    water_coords.push((atom.to_string(), new_coord));
                }

                // Ensure water molecules do not overlap with the existing molecule
                let overlaps = coords.iter().any(|&(cx, cy, cz)| {
                    water_coords.iter().any(|(_, (wx, wy, wz))| {
                        let dx = cx - wx;
                        let dy = cy - wy;
                        let dz = cz - wz;
                        (dx * dx + dy * dy + dz * dz).sqrt() < 2.0
                    })
                });

                if !overlaps {
                    for (atom, coord) in water_coords {
                        atom_types.push(atom);
                        coords.push(coord);
                    }
                }
            }
        }
    }
}
