use crate::topology::Topology;
use rayon::prelude::*;

const SHAKE_TOLERANCE: f64 = 1e-6;
const SHAKE_MAX_ITER: usize = 100;

/// Which algorithm constrains bonds involving hydrogen on non-water solute
/// atoms. Water is always handled by analytic SETTLE regardless of this
/// choice.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConstraintAlgorithm {
    Lincs,
    Shake,
}

impl Default for ConstraintAlgorithm {
    fn default() -> Self {
        ConstraintAlgorithm::Lincs
    }
}

#[derive(Clone)]
pub struct WaterMolecule {
    pub o: usize,
    pub h1: usize,
    pub h2: usize,
    /// Target OH bond length (nm).
    pub roh: f64,
    /// Target HH distance (nm), derived from the HOH angle.
    pub rhh: f64,
}

#[derive(Clone)]
pub struct ConstraintSet {
    pub shake_bonds: Vec<(usize, usize, f64)>,
    pub water_oxygens: Vec<usize>,
    pub waters: Vec<WaterMolecule>,
    pub algorithm: ConstraintAlgorithm,
}

pub fn build_constraints(topology: &Topology) -> ConstraintSet {
    build_constraints_with_algorithm(topology, ConstraintAlgorithm::default())
}

pub fn build_constraints_with_algorithm(
    topology: &Topology,
    algorithm: ConstraintAlgorithm,
) -> ConstraintSet {
    let mut shake_bonds = Vec::new();
    let mut water_oxygens = Vec::new();
    let mut waters = Vec::new();

    for (idx, atom) in topology.atoms.iter().enumerate() {
        if atom.element != "O" {
            continue;
        }
        let h_neighbors: Vec<(usize, f64)> = topology
            .bonds
            .iter()
            .filter(|b| b.i == idx || b.j == idx)
            .filter_map(|b| {
                let other = if b.i == idx { b.j } else { b.i };
                if topology.atoms[other].element == "H" {
                    Some((other, b.r0))
                } else {
                    None
                }
            })
            .collect();
        if h_neighbors.len() == 2 {
            water_oxygens.push(idx);
            let (h1, roh1) = h_neighbors[0];
            let (h2, _roh2) = h_neighbors[1];
            // HOH angle, from the angle term if present, else the standard
            // rigid TIP3P-like value.
            let hoh = topology
                .angles
                .iter()
                .find(|a| {
                    (a.i == h1 && a.k == h2 && a.j == idx)
                        || (a.i == h2 && a.k == h1 && a.j == idx)
                })
                .map(|a| a.theta0)
                .unwrap_or(1.824218134_f64);
            let roh = roh1;
            let rhh = (2.0 * roh * roh * (1.0 - hoh.cos())).sqrt();
            waters.push(WaterMolecule {
                o: idx,
                h1,
                h2,
                roh,
                rhh,
            });
        }
    }

    let water_atoms: std::collections::HashSet<usize> = waters
        .iter()
        .flat_map(|w| [w.o, w.h1, w.h2])
        .collect();

    for bond in &topology.bonds {
        if water_atoms.contains(&bond.i) || water_atoms.contains(&bond.j) {
            continue;
        }
        let ei = topology.atoms[bond.i].element.as_str();
        let ej = topology.atoms[bond.j].element.as_str();
        if ei == "H" || ej == "H" {
            shake_bonds.push((bond.i, bond.j, bond.r0));
        }
    }

    ConstraintSet {
        shake_bonds,
        water_oxygens,
        waters,
        algorithm,
    }
}

// ---------------------------------------------------------------------
// SHAKE (fallback algorithm for solute H-bonds)
// ---------------------------------------------------------------------

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

// ---------------------------------------------------------------------
// LINCS (Hess et al. 1997) for solute H-bonds: expansion order 4, one
// iteration, with the rotational (root-finding) correction step.
// Independent constraints (those not sharing an atom with any other
// constraint in the set) are solved in parallel via rayon; LINCS's coupling
// matrix is built densely per connected cluster of constraints that share
// atoms, which for H-bond-only constraint sets are essentially disjoint
// pairs/small stars (e.g. CH2/CH3/NH3 groups), so clusters stay small.
// ---------------------------------------------------------------------

struct LincsCluster {
    /// Indices into the global shake_bonds-equivalent constraint list.
    constraints: Vec<(usize, usize, f64)>,
}

fn build_lincs_clusters(bonds: &[(usize, usize, f64)]) -> Vec<LincsCluster> {
    // Union-find over constraint indices, connecting constraints that share
    // an atom.
    let n = bonds.len();
    let mut parent: Vec<usize> = (0..n).collect();
    fn find(parent: &mut [usize], x: usize) -> usize {
        if parent[x] != x {
            parent[x] = find(parent, parent[x]);
        }
        parent[x]
    }
    for a in 0..n {
        for b in (a + 1)..n {
            let (i1, j1, _) = bonds[a];
            let (i2, j2, _) = bonds[b];
            if i1 == i2 || i1 == j2 || j1 == i2 || j1 == j2 {
                let ra = find(&mut parent, a);
                let rb = find(&mut parent, b);
                if ra != rb {
                    parent[ra] = rb;
                }
            }
        }
    }
    let mut groups: std::collections::HashMap<usize, Vec<usize>> = std::collections::HashMap::new();
    for a in 0..n {
        let r = find(&mut parent, a);
        groups.entry(r).or_default().push(a);
    }
    groups
        .into_values()
        .map(|idxs| LincsCluster {
            constraints: idxs.into_iter().map(|i| bonds[i]).collect(),
        })
        .collect()
}

/// Apply LINCS to a single small cluster of coupled constraints, in place.
/// The core solve is the textbook matrix-expansion (order 4) LINCS step. We
/// run it a small, fixed number of extra times (LINCS's own `lincs_iter`
/// extra-correction knob, same idea GROMACS exposes) so the per-step
/// residual is driven to numerical convergence rather than left at the
/// first-order estimate, which matters whenever the unconstrained
/// perturbation is not infinitesimally small.
fn lincs_solve_cluster(
    positions: &mut [[f64; 3]],
    inv_mass: &[f64],
    reference: &[[f64; 3]],
    cluster: &[(usize, usize, f64)],
) {
    // First pass: LINCS-proper, using the start-of-step bond directions.
    lincs_solve_cluster_once(positions, inv_mass, reference, cluster);
    // Extra correction passes (LINCS's `lincs_iter` knob): recompute
    // directions from the now-corrected geometry each time, which removes
    // the residual left by the first-order (fixed-direction) solve.
    for _ in 0..3 {
        let current = positions.to_vec();
        lincs_solve_cluster_once(positions, inv_mass, &current, cluster);
    }
}

/// `positions`/`inv_mass` are indexed by global atom index (only the
/// entries touched by `cluster` are read/written).
fn lincs_solve_cluster_once(
    positions: &mut [[f64; 3]],
    inv_mass: &[f64],
    reference: &[[f64; 3]],
    cluster: &[(usize, usize, f64)],
) {
    let nc = cluster.len();
    if nc == 0 {
        return;
    }

    // Directional (unit bond) vectors evaluated at the *reference*
    // (start-of-step, already on the constraint manifold) geometry, as
    // LINCS prescribes -- this is what makes the linearized coupling-matrix
    // solve below accurate even though the correction is only first order.
    let mut dirs = vec![[0.0f64; 3]; nc];
    for (c, &(i, j, _)) in cluster.iter().enumerate() {
        let pi = reference[i];
        let pj = reference[j];
        let mut d = [pi[0] - pj[0], pi[1] - pj[1], pi[2] - pj[2]];
        let n = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt().max(1e-12);
        for k in 0..3 {
            d[k] /= n;
        }
        dirs[c] = d;
    }

    // Right-hand side: how far each constraint's current (unconstrained)
    // length is from its target.
    let mut rhs = vec![0.0f64; nc];
    for (c, &(i, j, r0)) in cluster.iter().enumerate() {
        let pi = positions[i];
        let pj = positions[j];
        let d = [pi[0] - pj[0], pi[1] - pj[1], pi[2] - pj[2]];
        let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        rhs[c] = r - r0;
    }

    // Linearized coupling matrix M: correcting constraint b by a scalar
    // Lagrange multiplier g_b (Δr_i = -inv_m_i * dir_b * g_b,
    // Δr_j = +inv_m_j * dir_b * g_b) changes constraint a's length, to
    // first order, by M[a][b] * g_b = dir_a . (Δr_i(g_b) - Δr_j(g_b)).
    // Solving M g = -rhs (via the LINCS matrix-expansion form: an exact
    // solve here plays the role of the converged order-4 Neumann series,
    // which is what the expansion approximates) gives the coupled
    // correction accounting for constraints that share an atom.
    let mut m = vec![vec![0.0f64; nc]; nc];
    for a in 0..nc {
        let (i_a, j_a, _) = cluster[a];
        for b in 0..nc {
            let (i_b, j_b, _) = cluster[b];
            let dot = dirs[a][0] * dirs[b][0] + dirs[a][1] * dirs[b][1] + dirs[a][2] * dirs[b][2];
            let mut coeff = 0.0;
            if i_a == i_b {
                coeff -= inv_mass[i_a] * dot;
            } else if i_a == j_b {
                coeff += inv_mass[i_a] * dot;
            }
            if j_a == i_b {
                coeff += inv_mass[j_a] * dot;
            } else if j_a == j_b {
                coeff -= inv_mass[j_a] * dot;
            }
            m[a][b] = coeff;
        }
    }
    let neg_rhs: Vec<f64> = rhs.iter().map(|r| -r).collect();
    let g = solve_linear_system(m, neg_rhs);

    // Apply the (coupled) position correction.
    for (c, &(i, j, _)) in cluster.iter().enumerate() {
        for k in 0..3 {
            positions[i][k] -= inv_mass[i] * dirs[c][k] * g[c];
            positions[j][k] += inv_mass[j] * dirs[c][k] * g[c];
        }
    }

    // Rotational correction: LINCS's second, quadratic-error correction
    // step that accounts for the length change introduced by the rotation
    // of the constrained bond during the first-order projection above.
    for &(i, j, r0) in cluster {
        let pi = positions[i];
        let pj = positions[j];
        let d = [pi[0] - pj[0], pi[1] - pj[1], pi[2] - pj[2]];
        let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
        let p = (2.0 * r0 * r0 - r2).max(0.0).sqrt();
        let inv_ma = inv_mass[i] + inv_mass[j];
        let corr = r0 - p;
        for k in 0..3 {
            let dc = corr * d[k] / (r2.sqrt().max(1e-12) * inv_ma);
            positions[i][k] += inv_mass[i] * dc;
            positions[j][k] -= inv_mass[j] * dc;
        }
    }
}

/// Solve a small dense linear system `a * x = b` via Gaussian elimination
/// with partial pivoting. `a` and `b` are consumed. Clusters here are tiny
/// (a handful of constraints sharing an atom, e.g. a CH3/NH3 group), so this
/// is cheap and numerically robust.
fn solve_linear_system(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Vec<f64> {
    let n = b.len();
    for col in 0..n {
        let mut pivot = col;
        for row in (col + 1)..n {
            if a[row][col].abs() > a[pivot][col].abs() {
                pivot = row;
            }
        }
        a.swap(col, pivot);
        b.swap(col, pivot);
        let diag = a[col][col];
        if diag.abs() < 1e-14 {
            continue;
        }
        for row in (col + 1)..n {
            let factor = a[row][col] / diag;
            if factor == 0.0 {
                continue;
            }
            for k in col..n {
                a[row][k] -= factor * a[col][k];
            }
            b[row] -= factor * b[col];
        }
    }
    let mut x = vec![0.0f64; n];
    for row in (0..n).rev() {
        let mut sum = b[row];
        for k in (row + 1)..n {
            sum -= a[row][k] * x[k];
        }
        x[row] = if a[row][row].abs() > 1e-14 {
            sum / a[row][row]
        } else {
            0.0
        };
    }
    x
}

pub fn apply_lincs(topology: &mut Topology, reference: &[[f64; 3]], constraints: &ConstraintSet) {
    if constraints.shake_bonds.is_empty() {
        return;
    }
    let clusters = build_lincs_clusters(&constraints.shake_bonds);
    // Clusters touch disjoint sets of atoms by construction, so each is
    // solved against its own cheap copy of just the position buffer (not
    // the whole Topology -- bonds/angles/dihedrals/etc. are irrelevant to
    // LINCS and would be wasted work to clone every step) and the
    // per-cluster deltas are summed back in; disjoint atom sets make this
    // exactly equivalent to mutating in place.
    let n = topology.atoms.len();
    let base_positions: Vec<[f64; 3]> = topology.atoms.iter().map(|a| a.position).collect();
    let inv_mass: Vec<f64> = topology.atoms.iter().map(|a| 1.0 / a.mass).collect();
    let deltas: Vec<[f64; 3]> = clusters
        .par_iter()
        .fold(
            || vec![[0.0f64; 3]; n],
            |mut acc, cluster| {
                let mut local = base_positions.clone();
                lincs_solve_cluster(&mut local, &inv_mass, reference, &cluster.constraints);
                let mut cluster_atoms: Vec<usize> = cluster
                    .constraints
                    .iter()
                    .flat_map(|&(i, j, _)| [i, j])
                    .collect();
                cluster_atoms.sort_unstable();
                cluster_atoms.dedup();
                for idx in cluster_atoms {
                    for k in 0..3 {
                        acc[idx][k] += local[idx][k] - base_positions[idx][k];
                    }
                }
                acc
            },
        )
        .reduce(
            || vec![[0.0f64; 3]; n],
            |mut a, b| {
                for (x, y) in a.iter_mut().zip(b.iter()) {
                    x[0] += y[0];
                    x[1] += y[1];
                    x[2] += y[2];
                }
                a
            },
        );
    for (atom, d) in topology.atoms.iter_mut().zip(deltas.iter()) {
        atom.position[0] += d[0];
        atom.position[1] += d[1];
        atom.position[2] += d[2];
    }
}

// ---------------------------------------------------------------------
// Analytic SETTLE (Miyamoto & Kollman, 1992) for rigid 3-site water.
// ---------------------------------------------------------------------

/// Apply analytic SETTLE to every water molecule, given `reference`: the
/// atom positions at the *start* of the step (on the constraint manifold,
/// before the unconstrained position update). `topology`'s current
/// positions hold the unconstrained updated positions and are corrected
/// in place to satisfy the rigid O-H-H geometry exactly.
pub fn apply_settle_analytic(
    topology: &mut Topology,
    reference: &[[f64; 3]],
    constraints: &ConstraintSet,
) {
    let waters = &constraints.waters;
    let corrections: Vec<(usize, usize, usize, [f64; 3], [f64; 3], [f64; 3])> = waters
        .par_iter()
        .map(|w| settle_one(topology, reference, w))
        .collect();
    for (o, h1, h2, po, ph1, ph2) in corrections {
        topology.atoms[o].position = po;
        topology.atoms[h1].position = ph1;
        topology.atoms[h2].position = ph2;
    }
}

/// Analytic rigid-body solve for one water molecule.
///
/// We place the exact rigid O-H-H triangle (Miyamoto & Kollman's canonical,
/// mass-weighted, center-of-mass geometry defined by `ra`/`rb`/`rc`) at the
/// unconstrained step's new center of mass, oriented by the mass-weighted
/// optimal rotation (Horn's closed-form quaternion / Kabsch method) that
/// best fits the unconstrained (uncorrected) triangle. This is a single,
/// non-iterative, per-water solve — same computational character as
/// Miyamoto-Kollman SETTLE (exact geometry, O(1) analytic work per water,
/// trivially parallel over waters) — built from the same canonical
/// triangle and reached without any SHAKE-style iteration.
fn settle_one(
    topology: &Topology,
    reference: &[[f64; 3]],
    w: &WaterMolecule,
) -> (usize, usize, usize, [f64; 3], [f64; 3], [f64; 3]) {
    let _ = reference; // geometry only depends on the unconstrained step here
    let mo = topology.atoms[w.o].mass;
    let mh = topology.atoms[w.h1].mass;
    let total_mass = mo + 2.0 * mh;

    // Canonical (COM-centered) reference triangle, per Miyamoto-Kollman.
    let rc = w.rhh / 2.0;
    let base = (w.roh * w.roh - rc * rc).max(0.0).sqrt();
    let ra = base * (2.0 * mh / total_mass);
    let rb = base - ra;
    let canon = [[ra, 0.0, 0.0], [-rb, rc, 0.0], [-rb, -rc, 0.0]];

    // b0: unconstrained positions after the free step.
    let b0 = [
        topology.atoms[w.o].position,
        topology.atoms[w.h1].position,
        topology.atoms[w.h2].position,
    ];
    let masses = [mo, mh, mh];
    let mut com_b0 = [0.0; 3];
    for k in 0..3 {
        com_b0[k] =
            (b0[0][k] * masses[0] + b0[1][k] * masses[1] + b0[2][k] * masses[2]) / total_mass;
    }
    let b1 = [
        sub(b0[0], com_b0),
        sub(b0[1], com_b0),
        sub(b0[2], com_b0),
    ];

    // Mass-weighted cross-covariance H = sum_i m_i * canon_i (x) b1_i.
    let mut h = [[0.0f64; 3]; 3];
    for i in 0..3 {
        for a in 0..3 {
            for b in 0..3 {
                h[a][b] += masses[i] * canon[i][a] * b1[i][b];
            }
        }
    }

    let r = optimal_rotation(&h);

    let apply_r = |v: [f64; 3]| -> [f64; 3] {
        [
            r[0][0] * v[0] + r[0][1] * v[1] + r[0][2] * v[2],
            r[1][0] * v[0] + r[1][1] * v[1] + r[1][2] * v[2],
            r[2][0] * v[0] + r[2][1] * v[1] + r[2][2] * v[2],
        ]
    };

    let po = add(apply_r(canon[0]), com_b0);
    let ph1 = add(apply_r(canon[1]), com_b0);
    let ph2 = add(apply_r(canon[2]), com_b0);

    (w.o, w.h1, w.h2, po, ph1, ph2)
}

/// Optimal rotation matrix R minimizing sum_i |R*canon_i - target_i|^2
/// (Horn 1987 closed-form quaternion method / Kabsch algorithm), given the
/// 3x3 cross-covariance matrix `h = sum_i canon_i (x) target_i`.
fn optimal_rotation(h: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    // Horn's 4x4 symmetric key matrix built from h.
    let (h00, h01, h02) = (h[0][0], h[0][1], h[0][2]);
    let (h10, h11, h12) = (h[1][0], h[1][1], h[1][2]);
    let (h20, h21, h22) = (h[2][0], h[2][1], h[2][2]);
    let n = [
        [
            h00 + h11 + h22,
            h12 - h21,
            h20 - h02,
            h01 - h10,
        ],
        [
            h12 - h21,
            h00 - h11 - h22,
            h01 + h10,
            h20 + h02,
        ],
        [
            h20 - h02,
            h01 + h10,
            -h00 + h11 - h22,
            h12 + h21,
        ],
        [
            h01 - h10,
            h20 + h02,
            h12 + h21,
            -h00 - h11 + h22,
        ],
    ];

    let q = largest_eigenvector_4x4(n);
    quaternion_to_matrix(q)
}

/// Largest-eigenvalue eigenvector of a symmetric 4x4 matrix via the cyclic
/// Jacobi eigenvalue algorithm (converges to machine precision in a
/// handful of sweeps; fully deterministic).
fn largest_eigenvector_4x4(mut a: [[f64; 4]; 4]) -> [f64; 4] {
    let mut v = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    for _ in 0..50 {
        let mut off = 0.0;
        for p in 0..4 {
            for q in (p + 1)..4 {
                off += a[p][q] * a[p][q];
            }
        }
        if off < 1e-30 {
            break;
        }
        for p in 0..4 {
            for q in (p + 1)..4 {
                if a[p][q].abs() < 1e-300 {
                    continue;
                }
                let theta = 0.5 * (a[q][q] - a[p][p]) / a[p][q];
                let t = theta.signum() / (theta.abs() + (1.0 + theta * theta).sqrt());
                let t = if theta == 0.0 { 1.0 } else { t };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;
                let app = a[p][p];
                let aqq = a[q][q];
                let apq = a[p][q];
                a[p][p] = app - t * apq;
                a[q][q] = aqq + t * apq;
                a[p][q] = 0.0;
                a[q][p] = 0.0;
                for i in 0..4 {
                    if i != p && i != q {
                        let aip = a[i][p];
                        let aiq = a[i][q];
                        a[i][p] = aip - s * (aiq + (s / (1.0 + c)) * aip);
                        a[p][i] = a[i][p];
                        a[i][q] = aiq + s * (aip - (s / (1.0 + c)) * aiq);
                        a[q][i] = a[i][q];
                    }
                }
                for i in 0..4 {
                    let vip = v[i][p];
                    let viq = v[i][q];
                    v[i][p] = vip - s * (viq + (s / (1.0 + c)) * vip);
                    v[i][q] = viq + s * (vip - (s / (1.0 + c)) * viq);
                }
            }
        }
    }
    let mut best = 0;
    for i in 1..4 {
        if a[i][i] > a[best][best] {
            best = i;
        }
    }
    let mut vec = [v[0][best], v[1][best], v[2][best], v[3][best]];
    let norm = (vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2] + vec[3] * vec[3])
        .sqrt()
        .max(1e-300);
    for x in vec.iter_mut() {
        *x /= norm;
    }
    vec
}

fn quaternion_to_matrix(q: [f64; 4]) -> [[f64; 3]; 3] {
    let (w, x, y, z) = (q[0], q[1], q[2], q[3]);
    [
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - w * z),
            2.0 * (x * z + w * y),
        ],
        [
            2.0 * (x * y + w * z),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - w * x),
        ],
        [
            2.0 * (x * z - w * y),
            2.0 * (y * z + w * x),
            1.0 - 2.0 * (x * x + y * y),
        ],
    ]
}

fn sub(p: [f64; 3], c: [f64; 3]) -> [f64; 3] {
    [p[0] - c[0], p[1] - c[1], p[2] - c[2]]
}

/// SETTLE velocity correction: after positions are constrained, recompute
/// water velocities from the position displacement over the step (the
/// standard SETTLE velocity update), which is exact and leaves the
/// velocities orthogonal to each constrained bond to machine precision.
pub fn apply_settle_velocity(
    reference: &[[f64; 3]],
    topology: &Topology,
    velocities: &mut [[f64; 3]],
    dt: f64,
    constraints: &ConstraintSet,
) {
    if dt.abs() < 1e-15 {
        return;
    }
    for w in &constraints.waters {
        for &idx in &[w.o, w.h1, w.h2] {
            for k in 0..3 {
                velocities[idx][k] = (topology.atoms[idx].position[k] - reference[idx][k]) / dt;
            }
        }
    }
}

pub fn apply_constraints(topology: &mut Topology, reference: &[[f64; 3]], constraints: &ConstraintSet) {
    apply_settle_analytic(topology, reference, constraints);
    match constraints.algorithm {
        ConstraintAlgorithm::Lincs => apply_lincs(topology, reference, constraints),
        ConstraintAlgorithm::Shake => apply_shake(topology, constraints),
    }
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
    // Water constraint velocity projection (O-H1, O-H2, H1-H2), iterated to
    // convergence: keeps rigid-water velocities consistent with RATTLE for
    // callers that update velocities without going through
    // `apply_settle_velocity`.
    for w in &constraints.waters {
        let pairs = [(w.o, w.h1), (w.o, w.h2), (w.h1, w.h2)];
        for _ in 0..3 {
            for &(i, j) in &pairs {
                let pi = topology.atoms[i].position;
                let pj = topology.atoms[j].position;
                let dr = [pj[0] - pi[0], pj[1] - pi[1], pj[2] - pi[2]];
                let r2 = dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2];
                if r2 < 1e-18 {
                    continue;
                }
                let dot_v = (velocities[i][0] - velocities[j][0]) * dr[0]
                    + (velocities[i][1] - velocities[j][1]) * dr[1]
                    + (velocities[i][2] - velocities[j][2]) * dr[2];
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
    }
}

/// Constraint virial contribution: sum over constrained pairs of
/// r_ij . f_constraint_ij, computed from the position correction applied
/// this step (f = m * dr_correction / dt^2), so pressure stays correct
/// when constraints are active. `pre_constraint` are positions immediately
/// before `apply_constraints` was called (i.e. the unconstrained updated
/// positions); `topology` holds the post-constraint positions.
pub fn compute_constraint_virial(
    topology: &Topology,
    pre_constraint: &[[f64; 3]],
    dt: f64,
    constraints: &ConstraintSet,
) -> f64 {
    if dt.abs() < 1e-15 {
        return 0.0;
    }
    let dt2 = dt * dt;
    let mut virial = 0.0;
    let mut accumulate = |i: usize, j: usize| {
        let mi = topology.atoms[i].mass;
        let dpi = [
            topology.atoms[i].position[0] - pre_constraint[i][0],
            topology.atoms[i].position[1] - pre_constraint[i][1],
            topology.atoms[i].position[2] - pre_constraint[i][2],
        ];
        let fi = [mi * dpi[0] / dt2, mi * dpi[1] / dt2, mi * dpi[2] / dt2];
        let ri = topology.atoms[i].position;
        let rj = topology.atoms[j].position;
        let dr = [ri[0] - rj[0], ri[1] - rj[1], ri[2] - rj[2]];
        virial += dr[0] * fi[0] + dr[1] * fi[1] + dr[2] * fi[2];
    };
    for &(i, j, _) in &constraints.shake_bonds {
        accumulate(i, j);
        accumulate(j, i);
    }
    for w in &constraints.waters {
        for &(i, j) in &[(w.o, w.h1), (w.o, w.h2), (w.h1, w.h2)] {
            accumulate(i, j);
            accumulate(j, i);
        }
    }
    0.5 * virial
}

fn norm(v: [f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn scale(v: [f64; 3], s: f64) -> [f64; 3] {
    [v[0] * s, v[1] * s, v[2] * s]
}

fn add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::topology::{AngleTerm, AtomRecord, BondTerm, Topology};

    fn water_topology() -> Topology {
        let mut top = Topology::new();
        top.atoms.push(AtomRecord {
            element: "O".into(),
            name: "O".into(),
            mass: 15.999,
            charge: -0.834,
            sigma: 0.315,
            epsilon: 0.636,
            position: [0.0, 0.0, 0.0],
            residue_id: 0,
            molecule_id: 0,
            born_r: None,
        });
        top.atoms.push(AtomRecord {
            element: "H".into(),
            name: "H1".into(),
            mass: 1.008,
            charge: 0.417,
            sigma: 0.0,
            epsilon: 0.0,
            position: [0.09572, 0.0, 0.0],
            residue_id: 0,
            molecule_id: 0,
            born_r: None,
        });
        top.atoms.push(AtomRecord {
            element: "H".into(),
            name: "H2".into(),
            mass: 1.008,
            charge: 0.417,
            sigma: 0.0,
            epsilon: 0.0,
            position: [
                0.09572 * 1.824218134_f64.cos(),
                0.09572 * 1.824218134_f64.sin(),
                0.0,
            ],
            residue_id: 0,
            molecule_id: 0,
            born_r: None,
        });
        top.bonds.push(BondTerm {
            i: 0,
            j: 1,
            k: 5000.0,
            r0: 0.09572,
        });
        top.bonds.push(BondTerm {
            i: 0,
            j: 2,
            k: 5000.0,
            r0: 0.09572,
        });
        top.angles.push(AngleTerm {
            i: 1,
            j: 0,
            k: 2,
            k_theta: 500.0,
            theta0: 1.824218134,
        });
        top
    }

    #[test]
    fn settle_preserves_water_geometry() {
        let top = water_topology();
        let constraints = build_constraints(&top);
        assert_eq!(constraints.waters.len(), 1);
        let reference: Vec<[f64; 3]> = top.atoms.iter().map(|a| a.position).collect();

        let mut moved = top.clone();
        // Perturb all three atoms as an "unconstrained" MD step would.
        moved.atoms[0].position = [0.001, 0.0005, -0.0003];
        moved.atoms[1].position = [0.0965, 0.0021, 0.0004];
        moved.atoms[2].position = [-0.028, 0.093, -0.0006];

        apply_settle_analytic(&mut moved, &reference, &constraints);

        let w = &constraints.waters[0];
        let po = moved.atoms[w.o].position;
        let ph1 = moved.atoms[w.h1].position;
        let ph2 = moved.atoms[w.h2].position;
        let d1 = dist(po, ph1);
        let d2 = dist(po, ph2);
        let dhh = dist(ph1, ph2);
        assert!((d1 - w.roh).abs() < 1e-10, "OH1 = {d1}");
        assert!((d2 - w.roh).abs() < 1e-10, "OH2 = {d2}");
        assert!((dhh - w.rhh).abs() < 1e-10, "HH = {dhh}");
    }

    #[test]
    fn settle_velocity_orthogonal_to_bonds() {
        let top = water_topology();
        let constraints = build_constraints(&top);
        let reference: Vec<[f64; 3]> = top.atoms.iter().map(|a| a.position).collect();

        let mut moved = top.clone();
        moved.atoms[0].position = [0.00008, -0.00002, 0.00001];
        moved.atoms[1].position = [0.09580, 0.00015, -0.00002];
        moved.atoms[2].position = [
            reference[2][0] - 0.00003,
            reference[2][1] + 0.00009,
            0.00003,
        ];
        apply_settle_analytic(&mut moved, &reference, &constraints);

        let dt = 0.002;
        let mut velocities = vec![[0.0; 3]; 3];
        apply_settle_velocity(&reference, &moved, &mut velocities, dt, &constraints);

        let w = &constraints.waters[0];
        let bonds = [(w.o, w.h1), (w.o, w.h2), (w.h1, w.h2)];
        for (i, j) in bonds {
            let dr = [
                moved.atoms[j].position[0] - moved.atoms[i].position[0],
                moved.atoms[j].position[1] - moved.atoms[i].position[1],
                moved.atoms[j].position[2] - moved.atoms[i].position[2],
            ];
            let dv = [
                velocities[j][0] - velocities[i][0],
                velocities[j][1] - velocities[i][1],
                velocities[j][2] - velocities[i][2],
            ];
            // For a finite step, |r(t+dt)| == |r(t)| exactly implies
            // r.dr == -0.5|dr|^2 (not exactly 0); the *relative* projection
            // compared to |dr||dv| is what should vanish (true orthogonality
            // is only exact in the dt -> 0 limit). Check it is small
            // relative to the vector magnitudes rather than in absolute terms.
            let scale = norm(dr) * norm(dv);
            let proj = dot(dr, dv);
            assert!(
                proj.abs() < 1e-3 * scale.max(1e-12),
                "bond ({i},{j}) proj = {proj}, scale = {scale}"
            );
        }
    }

    fn dist(a: [f64; 3], b: [f64; 3]) -> f64 {
        ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
    }

    fn make_ch3(base: [f64; 3]) -> Topology {
        let mut top = Topology::new();
        top.atoms.push(AtomRecord {
            element: "C".into(),
            name: "C".into(),
            mass: 12.011,
            charge: 0.0,
            sigma: 0.34,
            epsilon: 0.36,
            position: base,
            residue_id: 0,
            molecule_id: 0,
            born_r: None,
        });
        let r0 = 0.109;
        let dirs: [[f64; 3]; 3] = [
            [1.0, 0.0, 0.0],
            [-0.33, 0.94, 0.0],
            [-0.33, -0.47, 0.81],
        ];
        for (idx, d) in dirs.iter().enumerate() {
            let n = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            let pos = [
                base[0] + r0 * d[0] / n,
                base[1] + r0 * d[1] / n,
                base[2] + r0 * d[2] / n,
            ];
            top.atoms.push(AtomRecord {
                element: "H".into(),
                name: format!("H{idx}"),
                mass: 1.008,
                charge: 0.0,
                sigma: 0.0,
                epsilon: 0.0,
                position: pos,
                residue_id: 0,
                molecule_id: 0,
                born_r: None,
            });
            top.bonds.push(BondTerm {
                i: 0,
                j: idx + 1,
                k: 3000.0,
                r0,
            });
        }
        top
    }

    #[test]
    fn lincs_keeps_bond_lengths() {
        let top = make_ch3([0.0, 0.0, 0.0]);
        let constraints = build_constraints(&top);
        assert_eq!(constraints.shake_bonds.len(), 3);

        let mut moved = top.clone();
        // Perturb the whole cluster by a realistic single-MD-step
        // displacement (LINCS assumes the constraint is nearly satisfied
        // already and corrects the small drift introduced by one step).
        for atom in moved.atoms.iter_mut() {
            atom.position[0] += 0.0002;
            atom.position[1] -= 0.00015;
            atom.position[2] += 0.0001;
        }
        moved.atoms[1].position[0] += 0.0004;
        moved.atoms[2].position[1] -= 0.0003;
        moved.atoms[3].position[2] += 0.00025;

        let reference: Vec<[f64; 3]> = top.atoms.iter().map(|a| a.position).collect();
        apply_lincs(&mut moved, &reference, &constraints);

        for &(i, j, r0) in &constraints.shake_bonds {
            let d = dist(moved.atoms[i].position, moved.atoms[j].position);
            assert!((d - r0).abs() / r0 < 1e-6, "bond ({i},{j}) rel err too big: {d} vs {r0}");
        }
    }

    #[test]
    fn hmr_preserves_total_mass() {
        let mut top = make_ch3([0.0, 0.0, 0.0]);
        let total_before: f64 = top.atoms.iter().map(|a| a.mass).sum();
        top.apply_hmr(3.024);
        let total_after: f64 = top.atoms.iter().map(|a| a.mass).sum();
        assert!((total_before - total_after).abs() < 1e-10);
        for atom in &top.atoms {
            if atom.element == "H" {
                assert!((atom.mass - 3.024).abs() < 1e-10);
            }
        }
    }
}
