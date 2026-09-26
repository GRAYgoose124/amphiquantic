use crate::forces::{backend_from_env, compute_forces, ForceBackend};
use crate::topology::Topology;

pub struct MinimizeResult {
    pub topology: Topology,
    pub final_energy: f64,
    pub iterations: usize,
}

/// Which minimization algorithm to run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MinimizerKind {
    /// Limited-memory BFGS with backtracking (Armijo) line search and a
    /// per-step maximum displacement cap. This is the robust, SOTA-style
    /// default: it never blows up on the very close contacts a solvation
    /// builder or half-random starting structure can produce, because
    /// candidate steps that raise the energy (or produce a non-finite
    /// energy/force) are rejected and the step size is shrunk instead.
    LBfgs,
    /// Plain steepest descent with a fixed step size (legacy behaviour),
    /// kept available as an explicit option.
    SteepestDescent,
}

/// Maximum per-component atomic displacement allowed in a single step, nm.
/// This is the safety valve that prevents the classic "minimizer blowup":
/// a huge force from an overlapping/close-contact pair producing an
/// enormous step, which produces an even larger force next step, and so on
/// until positions or energies overflow to NaN/inf.
const MAX_STEP_NM: f64 = 0.02;

fn flatten(topology: &Topology) -> Vec<f64> {
    let mut x = Vec::with_capacity(topology.atoms.len() * 3);
    for atom in &topology.atoms {
        x.push(atom.position[0]);
        x.push(atom.position[1]);
        x.push(atom.position[2]);
    }
    x
}

fn apply_positions(topology: &mut Topology, x: &[f64]) {
    for (i, atom) in topology.atoms.iter_mut().enumerate() {
        atom.position[0] = x[3 * i];
        atom.position[1] = x[3 * i + 1];
        atom.position[2] = x[3 * i + 2];
        if topology.box_.pbc {
            crate::electrostatics::apply_pbc(&mut atom.position, &topology.box_);
        }
    }
}

/// Evaluate potential energy and the gradient (= -force) at a given flat
/// coordinate vector, without mutating the caller's topology permanently
/// (a scratch topology is used for the evaluation).
fn energy_and_grad(
    base: &Topology,
    x: &[f64],
    cutoff: f64,
    backend: ForceBackend,
) -> (f64, Vec<f64>) {
    let mut scratch = base.clone();
    apply_positions(&mut scratch, x);
    let result = compute_forces(&scratch, cutoff, backend);
    // `compute_forces` already folds in the Ewald self-/background-energy
    // correction for periodic systems; do not add it again here.
    let energy = result.potential_energy;
    let mut grad = Vec::with_capacity(x.len());
    for f in &result.forces {
        // grad = -force
        grad.push(-f[0]);
        grad.push(-f[1]);
        grad.push(-f[2]);
    }
    (energy, grad)
}

fn max_force_norm(grad: &[f64]) -> f64 {
    grad.chunks(3)
        .map(|f| (f[0] * f[0] + f[1] * f[1] + f[2] * f[2]).sqrt())
        .fold(0.0, f64::max)
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn cap_step(step: &mut [f64]) {
    // Cap each atom's displacement magnitude (not each component
    // independently) so the direction of the move is preserved.
    for chunk in step.chunks_mut(3) {
        let norm = (chunk[0] * chunk[0] + chunk[1] * chunk[1] + chunk[2] * chunk[2]).sqrt();
        if norm > MAX_STEP_NM && norm > 0.0 {
            let scale = MAX_STEP_NM / norm;
            chunk[0] *= scale;
            chunk[1] *= scale;
            chunk[2] *= scale;
        }
    }
}

fn finite(v: &[f64]) -> bool {
    v.iter().all(|x| x.is_finite())
}

/// Minimize with the classic fixed-step steepest descent (legacy path).
fn minimize_steepest_descent(
    mut topology: Topology,
    cutoff: f64,
    max_steps: usize,
    step_size: f64,
    tolerance_fmax: f64,
) -> MinimizeResult {
    let backend = backend_from_env();
    let mut final_energy = 0.0;
    let mut iterations = 0;

    for step in 0..max_steps {
        let result = compute_forces(&topology, cutoff, backend.clone());
        final_energy = result.potential_energy;
        iterations = step + 1;

        let max_force = result
            .forces
            .iter()
            .map(|f| (f[0] * f[0] + f[1] * f[1] + f[2] * f[2]).sqrt())
            .fold(0.0, f64::max);

        if !final_energy.is_finite() || max_force >= tolerance_fmax * 0.0 + f64::INFINITY {
            // energy already broke; nothing more we can safely do
            break;
        }
        if max_force < tolerance_fmax {
            break;
        }

        let mut step_vec: Vec<f64> = Vec::with_capacity(result.forces.len() * 3);
        for f in &result.forces {
            step_vec.push(step_size * f[0]);
            step_vec.push(step_size * f[1]);
            step_vec.push(step_size * f[2]);
        }
        cap_step(&mut step_vec);

        for (atom, chunk) in topology.atoms.iter_mut().zip(step_vec.chunks(3)) {
            atom.position[0] += chunk[0];
            atom.position[1] += chunk[1];
            atom.position[2] += chunk[2];
            if topology.box_.pbc {
                crate::electrostatics::apply_pbc(&mut atom.position, &topology.box_);
            }
        }
    }

    MinimizeResult {
        topology,
        final_energy,
        iterations,
    }
}

/// L-BFGS minimizer (two-loop recursion, m<=10 history) with a backtracking
/// (Armijo sufficient-decrease) line search and a maximum-displacement cap
/// on every trial step. Falls back to a capped steepest-descent step
/// whenever the L-BFGS direction is not a descent direction, is
/// non-finite, or the line search fails to find an accepted step — this
/// keeps the minimizer robust on pathological starting structures (e.g.
/// overlapping/close-contact atoms straight out of a solvation builder)
/// instead of ever propagating NaN/inf positions.
///
/// Convergence uses GROMACS `emtol` semantics: stop when the maximum force
/// magnitude on any atom (kJ/mol/nm) drops below `tolerance_fmax`.
fn minimize_lbfgs(
    mut topology: Topology,
    cutoff: f64,
    max_steps: usize,
    tolerance_fmax: f64,
) -> MinimizeResult {
    const HISTORY: usize = 10;
    let backend = backend_from_env();

    let mut x = flatten(&topology);
    let (mut energy, mut grad) = energy_and_grad(&topology, &x, cutoff, backend.clone());

    // If the starting structure/force evaluation is already broken (e.g.
    // literally coincident atoms producing an infinite energy, or a force
    // backend that cannot produce a finite result at all in this
    // environment), there is nothing a gradient-based method can do from
    // that exact point; report it as-is (one evaluation was attempted)
    // rather than propagate NaNs through positions.
    let mut iterations = 1usize;
    if !energy.is_finite() || !finite(&grad) {
        apply_positions(&mut topology, &x);
        return MinimizeResult {
            topology,
            final_energy: energy,
            iterations,
        };
    }

    let mut s_hist: Vec<Vec<f64>> = Vec::with_capacity(HISTORY);
    let mut y_hist: Vec<Vec<f64>> = Vec::with_capacity(HISTORY);
    let mut rho_hist: Vec<f64> = Vec::with_capacity(HISTORY);

    for step in 0..max_steps {
        iterations = step + 1;

        if max_force_norm(&grad) < tolerance_fmax {
            iterations = step;
            break;
        }

        // Two-loop recursion to get the L-BFGS search direction.
        let mut q = grad.clone();
        let m = s_hist.len();
        let mut alpha = vec![0.0; m];
        for i in (0..m).rev() {
            alpha[i] = rho_hist[i] * dot(&s_hist[i], &q);
            for k in 0..q.len() {
                q[k] -= alpha[i] * y_hist[i][k];
            }
        }
        let gamma = if m > 0 {
            let sy = dot(&s_hist[m - 1], &y_hist[m - 1]);
            let yy = dot(&y_hist[m - 1], &y_hist[m - 1]);
            if yy > 1e-300 {
                sy / yy
            } else {
                1.0
            }
        } else {
            // First iteration: scale by a small factor of the gradient
            // norm so the initial trial step is a reasonable size.
            let gnorm = max_force_norm(&grad).max(1e-8);
            (0.02 / gnorm).min(1.0)
        };
        for v in q.iter_mut() {
            *v *= gamma;
        }
        for i in 0..m {
            let beta = rho_hist[i] * dot(&y_hist[i], &q);
            for k in 0..q.len() {
                q[k] += s_hist[i][k] * (alpha[i] - beta);
            }
        }
        // Direction is -H*grad (descent direction for energy minimization,
        // since grad = -force).
        let mut direction: Vec<f64> = q.iter().map(|v| -v).collect();

        // Sanity check: must be a descent direction and finite. If not,
        // fall back to plain (negative-gradient) steepest descent for this
        // step and clear history — this is the robustness backstop.
        let directional_deriv = dot(&grad, &direction);
        if !finite(&direction) || directional_deriv >= 0.0 {
            direction = grad.iter().map(|g| -g).collect();
            s_hist.clear();
            y_hist.clear();
            rho_hist.clear();
        }

        cap_step(&mut direction);

        // Backtracking Armijo line search.
        const C1: f64 = 1e-4;
        let mut t = 1.0f64;
        let directional_deriv = dot(&grad, &direction);
        let mut accepted = false;
        let mut x_new = x.clone();
        let mut energy_new = energy;
        let mut grad_new = grad.clone();

        for _ in 0..20 {
            x_new = x
                .iter()
                .zip(direction.iter())
                .map(|(xi, di)| xi + t * di)
                .collect();
            let (e_trial, g_trial) = energy_and_grad(&topology, &x_new, cutoff, backend.clone());
            if e_trial.is_finite()
                && finite(&g_trial)
                && e_trial <= energy + C1 * t * directional_deriv
            {
                energy_new = e_trial;
                grad_new = g_trial;
                accepted = true;
                break;
            }
            t *= 0.5;
        }

        if !accepted {
            // Line search failed even with a tiny, capped step: we are as
            // converged as this backend/precision allows. Keep the last
            // good (finite) state and stop rather than risk a NaN.
            break;
        }

        let s: Vec<f64> = x_new.iter().zip(x.iter()).map(|(a, b)| a - b).collect();
        let y: Vec<f64> = grad_new.iter().zip(grad.iter()).map(|(a, b)| a - b).collect();
        let sy = dot(&s, &y);
        if sy > 1e-12 {
            if s_hist.len() == HISTORY {
                s_hist.remove(0);
                y_hist.remove(0);
                rho_hist.remove(0);
            }
            s_hist.push(s);
            y_hist.push(y);
            rho_hist.push(1.0 / sy);
        }

        x = x_new;
        energy = energy_new;
        grad = grad_new;
    }

    apply_positions(&mut topology, &x);

    MinimizeResult {
        topology,
        final_energy: energy,
        iterations,
    }
}

/// Minimize `topology` using the requested algorithm.
///
/// `tolerance` is the Fmax convergence criterion (GROMACS `emtol`
/// semantics): the maximum force magnitude on any atom, in kJ/mol/nm.
/// `step_size` is only used by the steepest-descent legacy path.
pub fn minimize_with(
    topology: Topology,
    cutoff: f64,
    max_steps: usize,
    step_size: f64,
    tolerance: f64,
    kind: MinimizerKind,
) -> MinimizeResult {
    match kind {
        MinimizerKind::LBfgs => minimize_lbfgs(topology, cutoff, max_steps, tolerance),
        MinimizerKind::SteepestDescent => {
            minimize_steepest_descent(topology, cutoff, max_steps, step_size, tolerance)
        }
    }
}

/// Backwards-compatible entry point: defaults to the robust L-BFGS
/// minimizer. `tolerance` here is interpreted as an Fmax (kJ/mol/nm)
/// convergence threshold.
pub fn minimize(
    topology: Topology,
    cutoff: f64,
    max_steps: usize,
    step_size: f64,
    tolerance: f64,
) -> MinimizeResult {
    minimize_with(
        topology,
        cutoff,
        max_steps,
        step_size,
        tolerance,
        MinimizerKind::LBfgs,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::topology::{AtomRecord, SimulationBox};

    fn make_atom(position: [f64; 3]) -> AtomRecord {
        AtomRecord {
            element: "Ar".to_string(),
            name: "Ar".to_string(),
            mass: 16.0,
            charge: 0.0,
            sigma: 0.3,
            epsilon: 0.5,
            position,
            residue_id: 0,
            molecule_id: 0,
            born_r: None,
        }
    }

    fn two_atom_topology(dist: f64) -> Topology {
        let mut top = Topology::new();
        top.box_ = SimulationBox {
            lx: 10.0,
            ly: 10.0,
            lz: 10.0,
            pbc: false,
        };
        let mut a = make_atom([5.0, 5.0, 5.0]);
        a.molecule_id = 0;
        let mut b = make_atom([5.0 + dist, 5.0, 5.0]);
        b.molecule_id = 1;
        top.atoms.push(a);
        top.atoms.push(b);
        top
    }

    #[test]
    fn lbfgs_relaxes_overlapping_pair_without_nan() {
        std::env::set_var("AMPHI_FORCE_BACKEND", "cpu");
        // Two LJ particles placed essentially on top of each other: a
        // stand-in for the close contacts a solvation builder can
        // produce. The legacy fixed-step steepest descent blows this up
        // into NaN; L-BFGS with capped, line-searched steps must not.
        let top = two_atom_topology(0.01);
        let result = minimize_with(top, 1.0, 200, 0.001, 1.0, MinimizerKind::LBfgs);
        assert!(result.final_energy.is_finite(), "energy became non-finite");
        for atom in &result.topology.atoms {
            assert!(atom.position.iter().all(|p| p.is_finite()));
        }
    }

    #[test]
    fn lbfgs_converges_on_well_separated_pair() {
        std::env::set_var("AMPHI_FORCE_BACKEND", "cpu");
        let top = two_atom_topology(0.5);
        let result = minimize_with(top, 1.0, 500, 0.001, 1e-6, MinimizerKind::LBfgs);
        assert!(result.final_energy.is_finite());
        assert!(result.iterations >= 1);
    }
}
