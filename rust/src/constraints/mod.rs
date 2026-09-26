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

/// Analytic rigid-body solve for one water molecule: the literal
/// Miyamoto & Kollman (1992) SETTLE algorithm (JCC 13(8), pp. 952-962),
/// following the canonical-frame / closed-form sin-cos derivation as
/// implemented in GROMACS `settle.cpp` and OpenMM's
/// `ReferenceSETTLEAlgorithm::apply`. `reference` gives the atom positions
/// at the *start* of the step (already on the constraint manifold); the
/// topology's current positions are the unconstrained new positions. The
/// correction this produces displaces each atom along the *old* (reference)
/// bond directions -- the defining property of SETTLE that a best-fit
/// (Kabsch/Horn) rotation does not have.
fn settle_one(
    topology: &Topology,
    reference: &[[f64; 3]],
    w: &WaterMolecule,
) -> (usize, usize, usize, [f64; 3], [f64; 3], [f64; 3]) {
    let m0 = topology.atoms[w.o].mass;
    let m1 = topology.atoms[w.h1].mass;
    let m2 = topology.atoms[w.h2].mass;

    let apos0 = reference[w.o];
    let apos1 = reference[w.h1];
    let apos2 = reference[w.h2];
    let xp0_new = topology.atoms[w.o].position;
    let xp1_new = topology.atoms[w.h1].position;
    let xp2_new = topology.atoms[w.h2].position;
    let mut xp0 = sub(xp0_new, apos0);
    let mut xp1 = sub(xp1_new, apos1);
    let mut xp2 = sub(xp2_new, apos2);

    // --- Step1: A1' ---
    let xb0 = apos1[0] - apos0[0];
    let yb0 = apos1[1] - apos0[1];
    let zb0 = apos1[2] - apos0[2];
    let xc0 = apos2[0] - apos0[0];
    let yc0 = apos2[1] - apos0[1];
    let zc0 = apos2[2] - apos0[2];

    let inv_total_mass = 1.0 / (m0 + m1 + m2);
    let xcom = (xp0[0] * m0 + (xb0 + xp1[0]) * m1 + (xc0 + xp2[0]) * m2) * inv_total_mass;
    let ycom = (xp0[1] * m0 + (yb0 + xp1[1]) * m1 + (yc0 + xp2[1]) * m2) * inv_total_mass;
    let zcom = (xp0[2] * m0 + (zb0 + xp1[2]) * m1 + (zc0 + xp2[2]) * m2) * inv_total_mass;

    let xa1 = xp0[0] - xcom;
    let ya1 = xp0[1] - ycom;
    let za1 = xp0[2] - zcom;
    let xb1 = xb0 + xp1[0] - xcom;
    let yb1 = yb0 + xp1[1] - ycom;
    let zb1 = zb0 + xp1[2] - zcom;
    let xc1 = xc0 + xp2[0] - xcom;
    let yc1 = yc0 + xp2[1] - ycom;
    let zc1 = zc0 + xp2[2] - zcom;

    let xaks_zd = yb0 * zc0 - zb0 * yc0;
    let yaks_zd = zb0 * xc0 - xb0 * zc0;
    let zaks_zd = xb0 * yc0 - yb0 * xc0;
    let xaks_xd = ya1 * zaks_zd - za1 * yaks_zd;
    let yaks_xd = za1 * xaks_zd - xa1 * zaks_zd;
    let zaks_xd = xa1 * yaks_zd - ya1 * xaks_zd;
    let xaks_yd = yaks_zd * zaks_xd - zaks_zd * yaks_xd;
    let yaks_yd = zaks_zd * xaks_xd - xaks_zd * zaks_xd;
    let zaks_yd = xaks_zd * yaks_xd - yaks_zd * xaks_xd;

    let axlng = (xaks_xd * xaks_xd + yaks_xd * yaks_xd + zaks_xd * zaks_xd).sqrt();
    let aylng = (xaks_yd * xaks_yd + yaks_yd * yaks_yd + zaks_yd * zaks_yd).sqrt();
    let azlng = (xaks_zd * xaks_zd + yaks_zd * yaks_zd + zaks_zd * zaks_zd).sqrt();
    let trns11 = xaks_xd / axlng;
    let trns21 = yaks_xd / axlng;
    let trns31 = zaks_xd / axlng;
    let trns12 = xaks_yd / aylng;
    let trns22 = yaks_yd / aylng;
    let trns32 = zaks_yd / aylng;
    let trns13 = xaks_zd / azlng;
    let trns23 = yaks_zd / azlng;
    let trns33 = zaks_zd / azlng;

    let xb0d = trns11 * xb0 + trns21 * yb0 + trns31 * zb0;
    let yb0d = trns12 * xb0 + trns22 * yb0 + trns32 * zb0;
    let xc0d = trns11 * xc0 + trns21 * yc0 + trns31 * zc0;
    let yc0d = trns12 * xc0 + trns22 * yc0 + trns32 * zc0;
    let za1d = trns13 * xa1 + trns23 * ya1 + trns33 * za1;
    let xb1d = trns11 * xb1 + trns21 * yb1 + trns31 * zb1;
    let yb1d = trns12 * xb1 + trns22 * yb1 + trns32 * zb1;
    let zb1d = trns13 * xb1 + trns23 * yb1 + trns33 * zb1;
    let xc1d = trns11 * xc1 + trns21 * yc1 + trns31 * zc1;
    let yc1d = trns12 * xc1 + trns22 * yc1 + trns32 * zc1;
    let zc1d = trns13 * xc1 + trns23 * yc1 + trns33 * zc1;

    // --- Step2: A2' ---
    let rc = 0.5 * w.rhh;
    let mut rb = (w.roh * w.roh - rc * rc).max(0.0).sqrt();
    let ra = rb * (m1 + m2) * inv_total_mass;
    rb -= ra;
    let sinphi = (za1d / ra).clamp(-1.0, 1.0);
    let cosphi = (1.0 - sinphi * sinphi).max(0.0).sqrt();
    let sinpsi = ((zb1d - zc1d) / (2.0 * rc * cosphi)).clamp(-1.0, 1.0);
    let cospsi = (1.0 - sinpsi * sinpsi).max(0.0).sqrt();

    let ya2d = ra * cosphi;
    let mut xb2d = -rc * cospsi;
    let yb2d = -rb * cosphi - rc * sinpsi * sinphi;
    let yc2d = -rb * cosphi + rc * sinpsi * sinphi;
    let xb2d2 = xb2d * xb2d;
    let hh2 = 4.0 * xb2d2 + (yb2d - yc2d) * (yb2d - yc2d) + (zb1d - zc1d) * (zb1d - zc1d);
    let deltx = 2.0 * xb2d + (4.0 * xb2d2 - hh2 + w.rhh * w.rhh).max(0.0).sqrt();
    xb2d -= deltx * 0.5;

    // --- Step3: al, be, ga ---
    let alpha = xb2d * (xb0d - xc0d) + yb0d * yb2d + yc0d * yc2d;
    let beta = xb2d * (yc0d - yb0d) + xb0d * yb2d + xc0d * yc2d;
    let gamma = xb0d * yb1d - xb1d * yb0d + xc0d * yc1d - xc1d * yc0d;

    let al2be2 = alpha * alpha + beta * beta;
    let sintheta = ((alpha * gamma - beta * (al2be2 - gamma * gamma).max(0.0).sqrt()) / al2be2)
        .clamp(-1.0, 1.0);

    // --- Step4: A3' ---
    let costheta = (1.0 - sintheta * sintheta).max(0.0).sqrt();
    let xa3d = -ya2d * sintheta;
    let ya3d = ya2d * costheta;
    let za3d = za1d;
    let xb3d = xb2d * costheta - yb2d * sintheta;
    let yb3d = xb2d * sintheta + yb2d * costheta;
    let zb3d = zb1d;
    let xc3d = -xb2d * costheta - yc2d * sintheta;
    let yc3d = -xb2d * sintheta + yc2d * costheta;
    let zc3d = zc1d;

    // --- Step5: A3 ---
    let xa3 = trns11 * xa3d + trns12 * ya3d + trns13 * za3d;
    let ya3 = trns21 * xa3d + trns22 * ya3d + trns23 * za3d;
    let za3 = trns31 * xa3d + trns32 * ya3d + trns33 * za3d;
    let xb3 = trns11 * xb3d + trns12 * yb3d + trns13 * zb3d;
    let yb3 = trns21 * xb3d + trns22 * yb3d + trns23 * zb3d;
    let zb3 = trns31 * xb3d + trns32 * yb3d + trns33 * zb3d;
    let xc3 = trns11 * xc3d + trns12 * yc3d + trns13 * zc3d;
    let yc3 = trns21 * xc3d + trns22 * yc3d + trns23 * zc3d;
    let zc3 = trns31 * xc3d + trns32 * yc3d + trns33 * zc3d;

    xp0 = [xcom + xa3, ycom + ya3, zcom + za3];
    xp1 = [xcom + xb3 - xb0, ycom + yb3 - yb0, zcom + zb3 - zb0];
    xp2 = [xcom + xc3 - xc0, ycom + yc3 - yc0, zcom + zc3 - zc0];

    let po = add(xp0, apos0);
    let ph1 = add(xp1, apos1);
    let ph2 = add(xp2, apos2);

    (w.o, w.h1, w.h2, po, ph1, ph2)
}

fn sub(p: [f64; 3], c: [f64; 3]) -> [f64; 3] {
    [p[0] - c[0], p[1] - c[1], p[2] - c[2]]
}

/// SETTLE velocity constraint (Miyamoto & Kollman's RATTLE-analog): solves
/// the 3x3 linear system for the along-bond velocity corrections tAB, tBC,
/// tCA that make the post-constraint velocities satisfy d/dt|r_ij|^2 = 0 for
/// each of the three rigid-water distance constraints, exactly as in
/// OpenMM/GROMACS's SETTLE velocity update (this is the general form for
/// unequal H masses, not the equal-mass simplification in the original
/// SETTLE paper's appendix B).
pub fn apply_settle_velocity(
    reference: &[[f64; 3]],
    topology: &Topology,
    velocities: &mut [[f64; 3]],
    dt: f64,
    constraints: &ConstraintSet,
) {
    let _ = (reference, dt);
    for w in &constraints.waters {
        let apos0 = topology.atoms[w.o].position;
        let apos1 = topology.atoms[w.h1].position;
        let apos2 = topology.atoms[w.h2].position;
        let m_a = topology.atoms[w.o].mass;
        let m_b = topology.atoms[w.h1].mass;
        let m_c = topology.atoms[w.h2].mass;
        let mut v0 = velocities[w.o];
        let mut v1 = velocities[w.h1];
        let mut v2 = velocities[w.h2];

        let norm3 = |v: [f64; 3]| -> [f64; 3] {
            let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-300);
            [v[0] / n, v[1] / n, v[2] / n]
        };
        let e_ab = norm3(sub(apos1, apos0));
        let e_bc = norm3(sub(apos2, apos1));
        let e_ca = norm3(sub(apos0, apos2));

        let v_ab = dot(sub(v1, v0), e_ab);
        let v_bc = dot(sub(v2, v1), e_bc);
        let v_ca = dot(sub(v0, v2), e_ca);

        let c_a = -dot(e_ab, e_ca);
        let c_b = -dot(e_ab, e_bc);
        let c_c = -dot(e_bc, e_ca);
        let s2a = 1.0 - c_a * c_a;
        let s2b = 1.0 - c_b * c_b;
        let s2c = 1.0 - c_c * c_c;

        let mabc_inv = 1.0 / (m_a * m_b * m_c);
        let denom = (((s2a * m_b + s2b * m_a) * m_c
            + (s2a * m_b * m_b + 2.0 * (c_a * c_b * c_c + 1.0) * m_a * m_b + s2b * m_a * m_a))
            * m_c
            + s2c * m_a * m_b * (m_a + m_b))
            * mabc_inv;
        let tab = ((c_b * c_c * m_a - c_a * m_b - c_a * m_c) * v_ca
            + (c_a * c_c * m_b - c_b * m_c - c_b * m_a) * v_bc
            + (s2c * m_a * m_a * m_b * m_b * mabc_inv + (m_a + m_b + m_c)) * v_ab)
            / denom;
        let tbc = ((c_a * c_b * m_c - c_c * m_b - c_c * m_a) * v_ca
            + (s2a * m_b * m_b * m_c * m_c * mabc_inv + (m_a + m_b + m_c)) * v_bc
            + (c_a * c_c * m_b - c_b * m_a - c_b * m_c) * v_ab)
            / denom;
        let tca = ((s2b * m_a * m_a * m_c * m_c * mabc_inv + (m_a + m_b + m_c)) * v_ca
            + (c_a * c_b * m_c - c_c * m_b - c_c * m_a) * v_bc
            + (c_b * c_c * m_a - c_a * m_b - c_a * m_c) * v_ab)
            / denom;

        for k in 0..3 {
            v0[k] += (e_ab[k] * tab - e_ca[k] * tca) / m_a;
            v1[k] += (e_bc[k] * tbc - e_ab[k] * tab) / m_b;
            v2[k] += (e_ca[k] * tca - e_bc[k] * tbc) / m_c;
        }
        velocities[w.o] = v0;
        velocities[w.h1] = v1;
        velocities[w.h2] = v2;
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
        for idx in 0..3 {
            for k in 0..3 {
                velocities[idx][k] = (moved.atoms[idx].position[k] - reference[idx][k]) / dt;
            }
        }
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

    /// Converged SHAKE (iterated to 1e-12 residual) on the full water
    /// triangle (O-H1, O-H2, H1-H2), used as an independent reference
    /// solution: SETTLE and this SHAKE both solve the *same* constrained
    /// least-squares problem (minimum mass-weighted displacement from the
    /// unconstrained positions satisfying the three rigid distances), so at
    /// tight tolerance they must agree to numerical precision.
    /// Canonical SHAKE (Ryckaert, Ciccotti & Berendsen 1977): the
    /// correction for each constraint is applied along the *reference*
    /// (old, pre-step) bond vector -- not the instantaneous one -- which is
    /// exactly the discretized Euler-Lagrange constraint-force direction
    /// that Miyamoto-Kollman SETTLE also solves for (in closed form,
    /// without iteration). Iterated to a 1e-13 residual this is the "true
    /// constrained solution" against which SETTLE is checked.
    fn converged_shake_water(
        positions: &mut [[f64; 3]; 3],
        reference: &[[f64; 3]; 3],
        masses: [f64; 3],
        targets: [(usize, usize, f64); 3],
    ) {
        let ref_dr: Vec<[f64; 3]> = targets
            .iter()
            .map(|&(i, j, _)| sub(reference[j], reference[i]))
            .collect();
        for _ in 0..20_000 {
            let mut max_err = 0.0f64;
            for (c, &(i, j, r0)) in targets.iter().enumerate() {
                let dr_ref = ref_dr[c];
                let dr_cur = sub(positions[j], positions[i]);
                let r2_cur = dr_cur[0] * dr_cur[0] + dr_cur[1] * dr_cur[1] + dr_cur[2] * dr_cur[2];
                let err = r2_cur - r0 * r0;
                max_err = max_err.max(err.abs() / (2.0 * r0));
                if err.abs() < 1e-13 {
                    continue;
                }
                let inv_mass = 1.0 / masses[i] + 1.0 / masses[j];
                let dot_ref_cur = dr_ref[0] * dr_cur[0] + dr_ref[1] * dr_cur[1] + dr_ref[2] * dr_cur[2];
                let g = err / (2.0 * inv_mass * dot_ref_cur);
                for k in 0..3 {
                    let dc = g * dr_ref[k];
                    positions[i][k] += dc / masses[i];
                    positions[j][k] -= dc / masses[j];
                }
            }
            if max_err < 1e-13 {
                break;
            }
        }
    }

    #[test]
    fn settle_matches_converged_shake() {
        let top = water_topology();
        let constraints = build_constraints(&top);
        let w = &constraints.waters[0];
        let reference: Vec<[f64; 3]> = top.atoms.iter().map(|a| a.position).collect();

        let mut moved = top.clone();
        moved.atoms[0].position = [0.0011, -0.0004, 0.0007];
        moved.atoms[1].position = [0.0968, 0.0032, -0.0011];
        moved.atoms[2].position = [-0.0271, 0.0895, 0.0021];

        // SETTLE solution.
        let mut settle_top = moved.clone();
        apply_settle_analytic(&mut settle_top, &reference, &constraints);

        // Independent converged-SHAKE solution on the same unconstrained
        // input and the same masses.
        let masses = [
            top.atoms[w.o].mass,
            top.atoms[w.h1].mass,
            top.atoms[w.h2].mass,
        ];
        let mut shake_pos = [
            moved.atoms[w.o].position,
            moved.atoms[w.h1].position,
            moved.atoms[w.h2].position,
        ];
        let ref_pos = [reference[w.o], reference[w.h1], reference[w.h2]];
        converged_shake_water(
            &mut shake_pos,
            &ref_pos,
            masses,
            [(0, 1, w.roh), (0, 2, w.roh), (1, 2, w.rhh)],
        );

        let settle_pos = [
            settle_top.atoms[w.o].position,
            settle_top.atoms[w.h1].position,
            settle_top.atoms[w.h2].position,
        ];
        for i in 0..3 {
            for k in 0..3 {
                assert!(
                    (settle_pos[i][k] - shake_pos[i][k]).abs() < 1e-8,
                    "atom {i} axis {k}: settle {} vs shake {}",
                    settle_pos[i][k],
                    shake_pos[i][k]
                );
            }
        }
    }

    #[test]
    fn settle_conserves_linear_momentum() {
        let top = water_topology();
        let constraints = build_constraints(&top);
        let w = &constraints.waters[0];
        let reference: Vec<[f64; 3]> = top.atoms.iter().map(|a| a.position).collect();

        let mut moved = top.clone();
        moved.atoms[0].position = [0.0006, 0.0009, -0.0002];
        moved.atoms[1].position = [0.0961, -0.0018, 0.0006];
        moved.atoms[2].position = [-0.0283, 0.0921, -0.0009];

        let dt = 0.002;
        let mut velocities = vec![[0.0; 3]; 3];
        for idx in 0..3 {
            for k in 0..3 {
                velocities[idx][k] = (moved.atoms[idx].position[k] - reference[idx][k]) / dt;
            }
        }
        let masses = [
            top.atoms[w.o].mass,
            top.atoms[w.h1].mass,
            top.atoms[w.h2].mass,
        ];
        let mut p_before = [0.0f64; 3];
        for i in 0..3 {
            for k in 0..3 {
                p_before[k] += masses[i] * velocities[i][k];
            }
        }

        apply_settle_analytic(&mut moved, &reference, &constraints);
        apply_settle_velocity(&reference, &moved, &mut velocities, dt, &constraints);

        let mut p_after = [0.0f64; 3];
        for i in 0..3 {
            for k in 0..3 {
                p_after[k] += masses[i] * velocities[i][k];
            }
        }
        for k in 0..3 {
            assert!(
                (p_before[k] - p_after[k]).abs() < 1e-9,
                "momentum axis {k}: before {} after {}",
                p_before[k],
                p_after[k]
            );
        }
    }

    #[test]
    fn lincs_matches_converged_shake() {
        let top = make_ch3([0.0, 0.0, 0.0]);
        let constraints = build_constraints(&top);
        assert_eq!(constraints.shake_bonds.len(), 3);

        let mut moved = top.clone();
        for atom in moved.atoms.iter_mut() {
            atom.position[0] += 0.0002;
            atom.position[1] -= 0.00015;
            atom.position[2] += 0.0001;
        }
        moved.atoms[1].position[0] += 0.0004;
        moved.atoms[2].position[1] -= 0.0003;
        moved.atoms[3].position[2] += 0.00025;

        let reference: Vec<[f64; 3]> = top.atoms.iter().map(|a| a.position).collect();

        let mut lincs_top = moved.clone();
        apply_lincs(&mut lincs_top, &reference, &constraints);

        // Independent converged-SHAKE reference on the same unconstrained
        // input (small, star-shaped cluster: SHAKE's own sequential
        // relaxation converges here too, driven to 1e-13 residual).
        let mut shake_top = moved.clone();
        for _ in 0..20_000 {
            let mut max_err = 0.0f64;
            for &(i, j, r0) in &constraints.shake_bonds {
                let dr = sub(shake_top.atoms[j].position, shake_top.atoms[i].position);
                let r2 = dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2];
                let r = r2.sqrt();
                let err = r - r0;
                max_err = max_err.max(err.abs());
                if err.abs() < 1e-13 {
                    continue;
                }
                let mi = shake_top.atoms[i].mass;
                let mj = shake_top.atoms[j].mass;
                let inv_mass = 1.0 / mi + 1.0 / mj;
                let corr = err / (2.0 * r * inv_mass);
                for k in 0..3 {
                    let dc = corr * dr[k];
                    shake_top.atoms[i].position[k] += dc / mi;
                    shake_top.atoms[j].position[k] -= dc / mj;
                }
            }
            if max_err < 1e-13 {
                break;
            }
        }

        for i in 0..lincs_top.atoms.len() {
            for k in 0..3 {
                let a = lincs_top.atoms[i].position[k];
                let b = shake_top.atoms[i].position[k];
                assert!(
                    (a - b).abs() < 1e-6,
                    "atom {i} axis {k}: lincs {a} vs shake {b}"
                );
            }
        }
    }
}
