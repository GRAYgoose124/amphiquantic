use crate::topology::Topology;

pub struct NeighborList {
    pub pairs: Vec<(usize, usize)>,
}

/// Cell-linked neighbor list with minimum image convention (O(N) average).
pub fn build_neighbor_list(topology: &Topology, cutoff: f64) -> NeighborList {
    let n = topology.atoms.len();
    if n == 0 {
        return NeighborList { pairs: vec![] };
    }

    let cutoff_sq = cutoff * cutoff;
    let is_excluded = |a: usize, b: usize| -> bool {
        topology.exclusions.iter().any(|e| e[0] == a && e[1] == b)
    };

    if !topology.box_.pbc {
        return build_naive(topology, cutoff_sq, is_excluded);
    }

    let box_ = &topology.box_;
    let lx = box_.lx.max(1e-6);
    let ly = box_.ly.max(1e-6);
    let lz = box_.lz.max(1e-6);
    let cell_size = cutoff;
    let nx = ((lx / cell_size).ceil() as i32).max(1);
    let ny = ((ly / cell_size).ceil() as i32).max(1);
    let nz = ((lz / cell_size).ceil() as i32).max(1);

    let mut cells: std::collections::HashMap<(i32, i32, i32), Vec<usize>> =
        std::collections::HashMap::new();

    for (idx, atom) in topology.atoms.iter().enumerate() {
        let p = atom.position;
        let cx = ((p[0] / lx).rem_euclid(1.0) * nx as f64) as i32 % nx;
        let cy = ((p[1] / ly).rem_euclid(1.0) * ny as f64) as i32 % ny;
        let cz = ((p[2] / lz).rem_euclid(1.0) * nz as f64) as i32 % nz;
        cells.entry((cx, cy, cz)).or_default().push(idx);
    }

    let mut pairs = Vec::new();
    for (&(cx, cy, cz), members) in &cells {
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    let nc = (
                        (cx + dx).rem_euclid(nx),
                        (cy + dy).rem_euclid(ny),
                        (cz + dz).rem_euclid(nz),
                    );
                    let Some(neighbors) = cells.get(&nc) else {
                        continue;
                    };
                    for &i in members {
                        for &j in neighbors {
                            if j <= i {
                                continue;
                            }
                            if is_excluded(i, j) {
                                continue;
                            }
                            let pi = topology.atoms[i].position;
                            let pj = topology.atoms[j].position;
                            let dr = minimum_image(
                                [pj[0] - pi[0], pj[1] - pi[1], pj[2] - pi[2]],
                                box_,
                            );
                            let r2 = dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2];
                            if r2 <= cutoff_sq {
                                pairs.push((i, j));
                            }
                        }
                    }
                }
            }
        }
    }

    NeighborList { pairs }
}

fn build_naive<F>(topology: &Topology, cutoff_sq: f64, is_excluded: F) -> NeighborList
where
    F: Fn(usize, usize) -> bool,
{
    let n = topology.atoms.len();
    let mut pairs = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            if is_excluded(i, j) {
                continue;
            }
            let pi = topology.atoms[i].position;
            let pj = topology.atoms[j].position;
            let mut dr = [pj[0] - pi[0], pj[1] - pi[1], pj[2] - pi[2]];
            dr = minimum_image(dr, &topology.box_);
            let r2 = dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2];
            if r2 <= cutoff_sq {
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

/// Verlet skin neighbor list manager — rebuilds when atoms move beyond skin/2.
pub struct NeighborListManager {
    cutoff: f64,
    skin: f64,
    reference_positions: Vec<[f64; 3]>,
    list: NeighborList,
}

impl NeighborListManager {
    pub fn new(cutoff: f64, skin: f64) -> Self {
        Self {
            cutoff,
            skin,
            reference_positions: Vec::new(),
            list: NeighborList { pairs: vec![] },
        }
    }

    pub fn maybe_rebuild(&mut self, topology: &Topology, cutoff: f64) {
        self.cutoff = cutoff;
        let needs_rebuild = self.reference_positions.len() != topology.atoms.len()
            || self.max_displacement(topology) > self.skin * 0.5;
        if needs_rebuild {
            self.list = build_neighbor_list(topology, self.cutoff + self.skin);
            self.reference_positions = topology.atoms.iter().map(|a| a.position).collect();
        }
    }

    pub fn pairs(&self) -> &[(usize, usize)] {
        &self.list.pairs
    }

    fn max_displacement(&self, topology: &Topology) -> f64 {
        self.reference_positions
            .iter()
            .zip(topology.atoms.iter())
            .map(|(r, a)| {
                let dr = [
                    a.position[0] - r[0],
                    a.position[1] - r[1],
                    a.position[2] - r[2],
                ];
                (dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2]).sqrt()
            })
            .fold(0.0, f64::max)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::topology::{AtomRecord, SimulationBox, Topology};

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

    #[test]
    fn cell_list_pbc() {
        let mut top = Topology::new();
        top.box_ = SimulationBox {
            lx: 3.0,
            ly: 3.0,
            lz: 3.0,
            pbc: true,
        };
        for i in 0..4 {
            top.atoms.push(AtomRecord {
                element: "C".into(),
                name: format!("C{i}"),
                mass: 12.0,
                charge: 0.0,
                sigma: 0.34,
                epsilon: 0.36,
                position: [i as f64 * 0.5, 0.0, 0.0],
                residue_id: 0,
                molecule_id: 0,
            });
        }
        let nl = build_neighbor_list(&top, 1.5);
        assert!(!nl.pairs.is_empty());
    }
}
