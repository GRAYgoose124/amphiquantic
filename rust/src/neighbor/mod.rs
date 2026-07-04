use crate::topology::Topology;

pub struct NeighborList {
    pub pairs: Vec<(usize, usize)>,
}

pub fn build_neighbor_list(topology: &Topology, cutoff: f64) -> NeighborList {
    let n = topology.atoms.len();
    let mut pairs = Vec::new();
    let cutoff_sq = cutoff * cutoff;

    let is_excluded = |a: usize, b: usize| -> bool {
        topology.exclusions.iter().any(|e| e[0] == a && e[1] == b)
    };

    for i in 0..n {
        for j in (i + 1)..n {
            if is_excluded(i, j) {
                continue;
            }
            let pi = topology.atoms[i].position;
            let pj = topology.atoms[j].position;
            let dx = pi[0] - pj[0];
            let dy = pi[1] - pj[1];
            let dz = pi[2] - pj[2];
            if dx * dx + dy * dy + dz * dz <= cutoff_sq {
                pairs.push((i, j));
            }
        }
    }

    NeighborList { pairs }
}

pub fn minimum_image(delta: [f64; 3], box_: &crate::topology::SimulationBox) -> [f64; 3] {
    if !box_.pbc {
        return delta;
    }
    let mut out = delta;
    let lengths = [box_.lx, box_.ly, box_.lz];
    for (d, l) in out.iter_mut().zip(lengths.iter()) {
        if *l <= 0.0 {
            continue;
        }
        if *d > 0.5 * l {
            *d -= l;
        } else if *d < -0.5 * l {
            *d += l;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::topology::{AtomRecord, Topology};

    #[test]
    fn finds_close_pairs() {
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
        top.atoms.push(AtomRecord {
            element: "C".into(),
            name: "C2".into(),
            mass: 12.0,
            charge: 0.0,
            sigma: 0.34,
            epsilon: 0.36,
            position: [0.2, 0.0, 0.0],
            residue_id: 0,
            molecule_id: 0,
        });
        let nl = build_neighbor_list(&top, 1.0);
        assert_eq!(nl.pairs.len(), 1);
    }
}
