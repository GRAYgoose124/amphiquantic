use crate::topology::Topology;
use std::collections::{HashMap, HashSet};

pub struct NeighborList {
    pub pairs: Vec<(usize, usize)>,
}

/// Below this atom count the brute-force O(N^2) builder is used directly;
/// above it we switch to the O(N) cell-linked list builder. Cell overhead
/// dominates for tiny systems, so the crossover keeps small-system tests
/// fast while giving large systems linear scaling.
pub const CELL_LIST_THRESHOLD: usize = 200;

/// Neighbor list with minimum image convention.
///
/// For `n < CELL_LIST_THRESHOLD` atoms this falls back to a brute-force
/// O(N^2) scan (cheaper than the cell-list bookkeeping at that scale).
/// Above the threshold it builds an O(N) cell-linked list, for both
/// periodic (orthorhombic, with minimum-image wraparound) and
/// non-periodic (bounded, no wraparound) systems.
pub fn build_neighbor_list(topology: &Topology, cutoff: f64) -> NeighborList {
    let n = topology.atoms.len();
    if n == 0 {
        return NeighborList { pairs: vec![] };
    }

    let exclusions = build_exclusion_set(topology);
    let cutoff_sq = cutoff * cutoff;

    if n < CELL_LIST_THRESHOLD {
        return build_naive(topology, cutoff_sq, &exclusions);
    }

    if topology.box_.pbc {
        build_cell_list_periodic(topology, cutoff, cutoff_sq, &exclusions)
    } else {
        build_cell_list_aperiodic(topology, cutoff, cutoff_sq, &exclusions)
    }
}

/// Force the brute-force O(N^2) builder regardless of system size. Used by
/// parity tests to compare against the cell-list builder.
pub fn build_neighbor_list_naive(topology: &Topology, cutoff: f64) -> NeighborList {
    let exclusions = build_exclusion_set(topology);
    build_naive(topology, cutoff * cutoff, &exclusions)
}

/// Force the cell-linked-list builder regardless of system size. Used by
/// parity tests to compare against the brute-force builder on small systems.
pub fn build_neighbor_list_cells(topology: &Topology, cutoff: f64) -> NeighborList {
    let exclusions = build_exclusion_set(topology);
    if topology.atoms.is_empty() {
        return NeighborList { pairs: vec![] };
    }
    if topology.box_.pbc {
        build_cell_list_periodic(topology, cutoff, cutoff * cutoff, &exclusions)
    } else {
        build_cell_list_aperiodic(topology, cutoff, cutoff * cutoff, &exclusions)
    }
}

fn build_exclusion_set(topology: &Topology) -> HashSet<(usize, usize)> {
    topology
        .exclusions
        .iter()
        .map(|e| (e[0], e[1]))
        .collect()
}

fn build_naive(
    topology: &Topology,
    cutoff_sq: f64,
    exclusions: &HashSet<(usize, usize)>,
) -> NeighborList {
    let n = topology.atoms.len();
    let mut pairs = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            if exclusions.contains(&(i, j)) {
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

/// Cell-linked list for a periodic orthorhombic box: cells wrap around with
/// minimum-image convention, so this is correct even when the box is only
/// slightly larger than the cutoff (falls back to a 3x3x3 or smaller grid).
fn build_cell_list_periodic(
    topology: &Topology,
    cutoff: f64,
    cutoff_sq: f64,
    exclusions: &HashSet<(usize, usize)>,
) -> NeighborList {
    let box_ = &topology.box_;
    let lx = box_.lx.max(1e-6);
    let ly = box_.ly.max(1e-6);
    let lz = box_.lz.max(1e-6);
    let cell_size = cutoff.max(1e-6);
    let nx = ((lx / cell_size).floor() as i32).max(1);
    let ny = ((ly / cell_size).floor() as i32).max(1);
    let nz = ((lz / cell_size).floor() as i32).max(1);

    let mut cells: HashMap<(i32, i32, i32), Vec<usize>> = HashMap::new();
    for (idx, atom) in topology.atoms.iter().enumerate() {
        let p = atom.position;
        let cx = ((p[0] / lx).rem_euclid(1.0) * nx as f64) as i32 % nx;
        let cy = ((p[1] / ly).rem_euclid(1.0) * ny as f64) as i32 % ny;
        let cz = ((p[2] / lz).rem_euclid(1.0) * nz as f64) as i32 % nz;
        cells.entry((cx, cy, cz)).or_default().push(idx);
    }

    // When a dimension has fewer than 3 cells, the 27-cell stencil would
    // visit the same neighboring cell more than once via wraparound; dedupe
    // the offsets actually visited for each axis to avoid double-counting.
    let offsets = |n_cells: i32| -> Vec<i32> {
        if n_cells >= 3 {
            vec![-1, 0, 1]
        } else if n_cells == 2 {
            vec![0, 1]
        } else {
            vec![0]
        }
    };
    let ox = offsets(nx);
    let oy = offsets(ny);
    let oz = offsets(nz);

    let mut pairs = Vec::new();
    let mut seen_cell_pairs: HashSet<((i32, i32, i32), (i32, i32, i32))> = HashSet::new();
    for (&(cx, cy, cz), members) in &cells {
        for &dx in &ox {
            for &dy in &oy {
                for &dz in &oz {
                    let nc = (
                        (cx + dx).rem_euclid(nx),
                        (cy + dy).rem_euclid(ny),
                        (cz + dz).rem_euclid(nz),
                    );
                    let this_cell = (cx, cy, cz);
                    // Each unordered pair of cells should only be scanned once.
                    let key = if this_cell <= nc {
                        (this_cell, nc)
                    } else {
                        (nc, this_cell)
                    };
                    if this_cell != nc && !seen_cell_pairs.insert(key) {
                        continue;
                    }
                    let Some(neighbors) = cells.get(&nc) else {
                        continue;
                    };
                    for &i in members {
                        for &j in neighbors {
                            if this_cell == nc && j <= i {
                                continue;
                            }
                            let (a, b) = if i < j { (i, j) } else { (j, i) };
                            if a == b || exclusions.contains(&(a, b)) {
                                continue;
                            }
                            let pi = topology.atoms[a].position;
                            let pj = topology.atoms[b].position;
                            let dr = minimum_image(
                                [pj[0] - pi[0], pj[1] - pi[1], pj[2] - pi[2]],
                                box_,
                            );
                            let r2 = dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2];
                            if r2 <= cutoff_sq {
                                pairs.push((a, b));
                            }
                        }
                    }
                }
            }
        }
    }

    pairs.sort_unstable();
    pairs.dedup();
    NeighborList { pairs }
}

/// Cell-linked list for a non-periodic system: cells are laid out over the
/// atoms' bounding box with no wraparound, so only the 27 (or fewer, at the
/// edges) adjacent cells are ever visited per atom.
fn build_cell_list_aperiodic(
    topology: &Topology,
    cutoff: f64,
    cutoff_sq: f64,
    exclusions: &HashSet<(usize, usize)>,
) -> NeighborList {
    let n = topology.atoms.len();
    let cell_size = cutoff.max(1e-6);

    let mut min = [f64::INFINITY; 3];
    let mut max = [f64::NEG_INFINITY; 3];
    for atom in &topology.atoms {
        for k in 0..3 {
            min[k] = min[k].min(atom.position[k]);
            max[k] = max[k].max(atom.position[k]);
        }
    }

    let cell_of = |p: [f64; 3]| -> (i64, i64, i64) {
        (
            ((p[0] - min[0]) / cell_size).floor() as i64,
            ((p[1] - min[1]) / cell_size).floor() as i64,
            ((p[2] - min[2]) / cell_size).floor() as i64,
        )
    };

    let mut cells: HashMap<(i64, i64, i64), Vec<usize>> = HashMap::new();
    for idx in 0..n {
        let c = cell_of(topology.atoms[idx].position);
        cells.entry(c).or_default().push(idx);
    }

    let mut pairs = Vec::new();
    for (&(cx, cy, cz), members) in &cells {
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    // Only scan each unordered cell pair once: visit the
                    // neighbor cell only if it's lexicographically >= this
                    // cell (this also naturally covers the self-cell case).
                    let nc = (cx + dx, cy + dy, cz + dz);
                    if nc < (cx, cy, cz) {
                        continue;
                    }
                    let Some(neighbors) = cells.get(&nc) else {
                        continue;
                    };
                    let same_cell = nc == (cx, cy, cz);
                    for &i in members {
                        for &j in neighbors {
                            if same_cell && j <= i {
                                continue;
                            }
                            let (a, b) = if i < j { (i, j) } else { (j, i) };
                            if a == b || exclusions.contains(&(a, b)) {
                                continue;
                            }
                            let pi = topology.atoms[a].position;
                            let pj = topology.atoms[b].position;
                            let dr = [pj[0] - pi[0], pj[1] - pi[1], pj[2] - pi[2]];
                            let r2 = dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2];
                            if r2 <= cutoff_sq {
                                pairs.push((a, b));
                            }
                        }
                    }
                }
            }
        }
    }

    pairs.sort_unstable();
    pairs.dedup();
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
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    fn make_atom(position: [f64; 3], name: &str) -> AtomRecord {
        AtomRecord {
            element: "C".into(),
            name: name.into(),
            mass: 12.0,
            charge: 0.0,
            sigma: 0.34,
            epsilon: 0.36,
            position,
            residue_id: 0,
            molecule_id: 0,
            born_r: None,
        }
    }

    fn sort_pairs(nl: &NeighborList) -> Vec<(usize, usize)> {
        let mut v = nl.pairs.clone();
        v.sort_unstable();
        v
    }

    #[test]
    fn finds_close_pairs() {
        let mut top = Topology::new();
        top.atoms.push(make_atom([0.0, 0.0, 0.0], "C1"));
        top.atoms.push(make_atom([0.2, 0.0, 0.0], "C2"));
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
            top.atoms
                .push(make_atom([i as f64 * 0.5, 0.0, 0.0], &format!("C{i}")));
        }
        let nl = build_neighbor_list(&top, 1.5);
        assert!(!nl.pairs.is_empty());
    }

    fn random_topology(n: usize, pbc: bool, seed: u64) -> Topology {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut top = Topology::new();
        let extent = 5.0;
        top.box_ = SimulationBox {
            lx: extent,
            ly: extent,
            lz: extent,
            pbc,
        };
        for i in 0..n {
            let p = [
                rng.gen_range(0.0..extent),
                rng.gen_range(0.0..extent),
                rng.gen_range(0.0..extent),
            ];
            top.atoms.push(make_atom(p, &format!("A{i}")));
        }
        // Add some bonded exclusions so the parity test actually exercises them.
        for i in 0..n {
            if i + 1 < n && i % 3 == 0 {
                top.bonds.push(crate::topology::BondTerm {
                    i,
                    j: i + 1,
                    r0: 0.1,
                    k: 100.0,
                });
            }
        }
        top.build_exclusions();
        top
    }

    #[test]
    fn cell_list_matches_naive_periodic_random() {
        for seed in 0..5 {
            let top = random_topology(150, true, seed);
            let cutoff = 1.2;
            let naive = build_neighbor_list_naive(&top, cutoff);
            let cells = build_neighbor_list_cells(&top, cutoff);
            assert_eq!(
                sort_pairs(&naive),
                sort_pairs(&cells),
                "mismatch at seed {seed} (periodic)"
            );
        }
    }

    #[test]
    fn cell_list_matches_naive_nonperiodic_random() {
        for seed in 0..5 {
            let top = random_topology(150, false, seed);
            let cutoff = 1.2;
            let naive = build_neighbor_list_naive(&top, cutoff);
            let cells = build_neighbor_list_cells(&top, cutoff);
            assert_eq!(
                sort_pairs(&naive),
                sort_pairs(&cells),
                "mismatch at seed {seed} (non-periodic)"
            );
        }
    }

    #[test]
    fn auto_dispatch_uses_cell_list_above_threshold() {
        // Above the threshold, build_neighbor_list must still match the
        // brute-force result (it just takes the cell-list code path).
        let top = random_topology(CELL_LIST_THRESHOLD + 50, true, 42);
        let cutoff = 1.0;
        let naive = build_neighbor_list_naive(&top, cutoff);
        let auto = build_neighbor_list(&top, cutoff);
        assert_eq!(sort_pairs(&naive), sort_pairs(&auto));
    }

    #[test]
    fn exclusions_are_honored_exactly() {
        let mut top = Topology::new();
        top.atoms.push(make_atom([0.0, 0.0, 0.0], "C1"));
        top.atoms.push(make_atom([0.1, 0.0, 0.0], "C2"));
        top.atoms.push(make_atom([0.2, 0.0, 0.0], "C3"));
        top.bonds.push(crate::topology::BondTerm {
            i: 0,
            j: 1,
            r0: 0.1,
            k: 100.0,
        });
        top.build_exclusions();
        let nl = build_neighbor_list(&top, 1.0);
        // 0-1 excluded (bonded), 1-2 and 0-2 remain.
        assert!(!nl.pairs.contains(&(0, 1)));
        assert!(nl.pairs.contains(&(1, 2)));
        assert!(nl.pairs.contains(&(0, 2)));
    }

    #[test]
    fn neighbor_list_manager_rebuild_matches_direct_build() {
        let top = random_topology(80, true, 7);
        let mut mgr = NeighborListManager::new(1.0, 0.2);
        mgr.maybe_rebuild(&top, 1.0);
        let direct = build_neighbor_list(&top, 1.2);
        assert_eq!(sort_pairs(&direct), {
            let mut v = mgr.pairs().to_vec();
            v.sort_unstable();
            v
        });
    }
}
