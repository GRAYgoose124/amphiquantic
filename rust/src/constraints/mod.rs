use crate::topology::Topology;

const SHAKE_TOLERANCE: f64 = 1e-6;
const SHAKE_MAX_ITER: usize = 100;

#[derive(Clone)]
pub struct ConstraintSet {
    pub shake_bonds: Vec<(usize, usize, f64)>,
    pub water_oxygens: Vec<usize>,
}

pub fn build_constraints(topology: &Topology) -> ConstraintSet {
    let mut shake_bonds = Vec::new();
    for bond in &topology.bonds {
        let ei = topology.atoms[bond.i].element.as_str();
        let ej = topology.atoms[bond.j].element.as_str();
        if ei == "H" || ej == "H" {
            shake_bonds.push((bond.i, bond.j, bond.r0));
        }
    }

    let mut water_oxygens = Vec::new();
    for (idx, atom) in topology.atoms.iter().enumerate() {
        if atom.element != "O" {
            continue;
        }
        let h_neighbors: Vec<usize> = topology
            .bonds
            .iter()
            .filter(|b| b.i == idx || b.j == idx)
            .filter_map(|b| {
                let other = if b.i == idx { b.j } else { b.i };
                if topology.atoms[other].element == "H" {
                    Some(other)
                } else {
                    None
                }
            })
            .collect();
        if h_neighbors.len() == 2 {
            water_oxygens.push(idx);
        }
    }

    ConstraintSet {
        shake_bonds,
        water_oxygens,
    }
}

pub fn apply_shake(topology: &mut Topology, constraints: &ConstraintSet) {
    for _ in 0..SHAKE_MAX_ITER {
        let mut max_err = 0.0f64;
        for &(i, j, r0) in &constraints.shake_bonds {
            let pi = topology.atoms[i].position;
            let pj = topology.atoms[j].position;
            let dr = [pj[0] - pi[0], pj[1] - pi[1], pj[2] - pi[2]];
            let r2 = dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2];
            let r = r2.sqrt().max(1e-12);
            let err = r - r0;
            max_err = max_err.max(err.abs());
            if err.abs() < SHAKE_TOLERANCE {
                continue;
            }
            let mi = topology.atoms[i].mass;
            let mj = topology.atoms[j].mass;
            let inv_mass = 1.0 / mi + 1.0 / mj;
            let corr = err / (2.0 * r * inv_mass);
            for k in 0..3 {
                let dc = corr * dr[k];
                topology.atoms[i].position[k] += dc / mi;
                topology.atoms[j].position[k] -= dc / mj;
            }
        }
        if max_err < SHAKE_TOLERANCE {
            break;
        }
    }
}

pub fn apply_settle(topology: &mut Topology, constraints: &ConstraintSet) {
    let oh = 0.9572;
    let hoh = 1.824218134_f64;
    for &o in &constraints.water_oxygens {
        let hs: Vec<usize> = topology
            .bonds
            .iter()
            .filter(|b| b.i == o || b.j == o)
            .filter_map(|b| {
                let other = if b.i == o { b.j } else { b.i };
                if topology.atoms[other].element == "H" {
                    Some(other)
                } else {
                    None
                }
            })
            .collect();
        if hs.len() != 2 {
            continue;
        }
        let h1 = hs[0];
        let h2 = hs[1];
        let po = topology.atoms[o].position;
        let mut p1 = topology.atoms[h1].position;
        let mut p2 = topology.atoms[h2].position;
        let d1 = [p1[0] - po[0], p1[1] - po[1], p1[2] - po[2]];
        let d2 = [p2[0] - po[0], p2[1] - po[1], p2[2] - po[2]];
        let r1 = (d1[0] * d1[0] + d1[1] * d1[1] + d1[2] * d1[2]).sqrt().max(1e-12);
        let r2 = (d2[0] * d2[0] + d2[1] * d2[1] + d2[2] * d2[2]).sqrt().max(1e-12);
        let cos_a = (d1[0] * d2[0] + d1[1] * d2[1] + d1[2] * d2[2]) / (r1 * r2);
        let angle_err = cos_a.clamp(-1.0, 1.0).acos() - hoh;
        if angle_err.abs() > SHAKE_TOLERANCE || (r1 - oh).abs() > SHAKE_TOLERANCE {
            for k in 0..3 {
                p1[k] = po[k] + oh * d1[k] / r1;
                p2[k] = po[k] + oh * d2[k] / r2;
            }
            topology.atoms[h1].position = p1;
            topology.atoms[h2].position = p2;
        }
    }
}

pub fn apply_constraints(topology: &mut Topology, constraints: &ConstraintSet) {
    apply_settle(topology, constraints);
    apply_shake(topology, constraints);
}

pub fn apply_rattle(
    topology: &Topology,
    velocities: &mut [[f64; 3]],
    constraints: &ConstraintSet,
) {
    for &(i, j, _r0) in &constraints.shake_bonds {
        let pi = topology.atoms[i].position;
        let pj = topology.atoms[j].position;
        let dr = [pj[0] - pi[0], pj[1] - pi[1], pj[2] - pi[2]];
        let r2 = dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2];
        let dot_v = velocities[i][0] * dr[0]
            + velocities[i][1] * dr[1]
            + velocities[i][2] * dr[2];
        let mi = topology.atoms[i].mass;
        let mj = topology.atoms[j].mass;
        let inv_mass = 1.0 / mi + 1.0 / mj;
        let corr = dot_v / (r2 * inv_mass);
        for k in 0..3 {
            let dc = corr * dr[k];
            velocities[i][k] -= dc / mi;
            velocities[j][k] += dc / mj;
        }
    }
}
