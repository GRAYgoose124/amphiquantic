//! Smooth Particle Mesh Ewald (SPME), following Essmann, Perera, Berkowitz,
//! Darden, Lee & Pedersen, J. Chem. Phys. 103, 8577 (1995).
//!
//! Charges are spread onto a regular grid with cardinal B-splines, the
//! resulting charge-density grid is Fourier transformed with `rustfft`, the
//! reciprocal-space energy/forces are evaluated via the Ewald influence
//! function (corrected for the B-spline smoothing through the B-spline
//! moduli), and the result is transformed back and gathered onto the atoms
//! using analytic spline derivatives. This turns the O(N*K^3) direct
//! reciprocal-space sum into an O(N + K^3 log K) computation.

use crate::topology::Topology;
use rayon::prelude::*;
use rustfft::{num_complex::Complex64, FftDirection, FftPlanner};
use std::f64::consts::PI;

const COULOMB_CONSTANT: f64 = 138.935456;

pub struct PmeResult {
    pub forces: Vec<[f64; 3]>,
    pub energy: f64,
    /// Scalar reciprocal-space virial contribution W = -3V dE/dV (fixed
    /// fractional coordinates), consistent with the sign convention used by
    /// `forces::cpu::compute_virial` (P = (2*KE + W) / (3*V)).
    pub virial: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct PmeContext {
    pub alpha: f64,
    pub grid_size: [usize; 3],
    pub cutoff: f64,
    pub order: usize,
}

/// Default Ewald real-space cutoff tolerance (GROMACS-style `ewald_rtol`).
pub const DEFAULT_EWALD_RTOL: f64 = 1e-5;
/// Default target reciprocal-grid spacing, nm (GROMACS `fourierspacing`).
pub const DEFAULT_FOURIER_SPACING: f64 = 0.12;
/// Default cardinal B-spline interpolation order (`pme-order`).
pub const DEFAULT_PME_ORDER: usize = 4;

impl PmeContext {
    /// Automatic, GROMACS-style parameter choice: alpha from `ewald_rtol`
    /// and the real-space cutoff, grid size from the fourier spacing
    /// (rounded up to an FFT-friendly 2/3/5/7-smooth size), spline order 4.
    pub fn new(topology: &Topology, cutoff: f64) -> Self {
        Self::with_params(
            topology,
            cutoff,
            DEFAULT_EWALD_RTOL,
            DEFAULT_FOURIER_SPACING,
            DEFAULT_PME_ORDER,
        )
    }

    pub fn with_order(topology: &Topology, cutoff: f64, order: usize) -> Self {
        Self::with_params(
            topology,
            cutoff,
            DEFAULT_EWALD_RTOL,
            DEFAULT_FOURIER_SPACING,
            order,
        )
    }

    pub fn with_params(
        topology: &Topology,
        cutoff: f64,
        ewald_rtol: f64,
        fourier_spacing: f64,
        order: usize,
    ) -> Self {
        let order = order.clamp(3, 8);
        let alpha = ewald_alpha_from_rtol(cutoff, ewald_rtol);
        let grid_size = [
            fft_grid_dim(topology.box_.lx, fourier_spacing, order),
            fft_grid_dim(topology.box_.ly, fourier_spacing, order),
            fft_grid_dim(topology.box_.lz, fourier_spacing, order),
        ];
        Self {
            alpha,
            grid_size,
            cutoff,
            order,
        }
    }
}

/// Solve erfc(alpha*cutoff) = rtol for alpha via bisection (monotonic,
/// well-conditioned). This is the standard GROMACS convention for choosing
/// the Ewald splitting parameter from a real-space cutoff.
pub fn ewald_alpha_from_rtol(cutoff: f64, rtol: f64) -> f64 {
    if cutoff <= 0.0 {
        return 0.3;
    }
    let rtol = rtol.clamp(1e-12, 0.5);
    let mut lo = 1e-4;
    let mut hi = 60.0 / cutoff;
    // erfc(lo*cutoff) should be > rtol, erfc(hi*cutoff) should be < rtol.
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        let val = super::ewald::erfc(mid * cutoff);
        if val > rtol {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

/// Round `n` up to the next integer whose only prime factors are 2, 3, 5, 7
/// (an "FFT-friendly" size for `rustfft`'s mixed-radix planner).
pub fn next_fft_friendly(mut n: usize) -> usize {
    if n < 1 {
        n = 1;
    }
    loop {
        let mut m = n;
        for p in [2usize, 3, 5, 7] {
            while m % p == 0 {
                m /= p;
            }
        }
        if m == 1 {
            return n;
        }
        n += 1;
    }
}

fn fft_grid_dim(length: f64, spacing: f64, order: usize) -> usize {
    if length <= 0.0 {
        return next_fft_friendly(order.max(4));
    }
    let spacing = spacing.max(1e-6);
    let raw = (length / spacing).ceil() as i64;
    let min_dim = (2 * order) as i64;
    let n = raw.max(min_dim).max(4) as usize;
    next_fft_friendly(n)
}

/// Cardinal B-spline weights and derivatives of order `order` at fractional
/// offset `w` in `[0, 1)`. Returns `(weights, d_weights/dw)`, each of length
/// `order`, following the classic recursive construction used throughout
/// the PME literature (Essmann et al. 1995, appendix).
fn fill_bspline(w: f64, order: usize) -> (Vec<f64>, Vec<f64>) {
    let mut arr = vec![0.0f64; order];
    arr[1] = w;
    arr[0] = 1.0 - w;

    for k in 3..order {
        let div = 1.0 / (k as f64 - 1.0);
        arr[k - 1] = div * w * arr[k - 2];
        for j in 1..(k - 1) {
            arr[k - 1 - j] =
                div * ((w + j as f64) * arr[k - 2 - j] + (k as f64 - j as f64 - w) * arr[k - 1 - j]);
        }
        arr[0] = div * (1.0 - w) * arr[0];
    }

    let mut darr = vec![0.0f64; order];
    darr[0] = -arr[0];
    for j in 1..order {
        darr[j] = arr[j - 1] - arr[j];
    }

    let k = order;
    let div = 1.0 / (k as f64 - 1.0);
    arr[k - 1] = div * w * arr[k - 2];
    for j in 1..(k - 1) {
        arr[k - 1 - j] =
            div * ((w + j as f64) * arr[k - 2 - j] + (k as f64 - j as f64 - w) * arr[k - 1 - j]);
    }
    arr[0] = div * (1.0 - w) * arr[0];

    (arr, darr)
}

/// |b(m)|^2 for each grid index m = 0..K-1 along one dimension: the inverse
/// squared modulus of the DFT of the (order-1)-fold zero-padded spline
/// values at integer knots. See Essmann et al. eq. (4.4).
pub(crate) fn bspline_moduli(order: usize, k: usize) -> Vec<f64> {
    let (arr, _) = fill_bspline(0.0, order);
    // `arr[i]` (unreversed) equals M_n(order-1-i); the modulus formula
    // needs padded[k] = M_n(k+1) for k = 0..order-2, i.e. the same values
    // in reverse order (see the comment in `atom_splines`).
    let mut padded = vec![Complex64::new(0.0, 0.0); k];
    for kk in 0..order.saturating_sub(1) {
        padded[kk] = Complex64::new(arr[order - 2 - kk], 0.0);
    }
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft(k, FftDirection::Forward);
    fft.process(&mut padded);

    let mut modulus_sq: Vec<f64> = padded.iter().map(|c| c.norm_sqr()).collect();

    // Standard fix for the (rare) even-order case where a modulus lands
    // exactly on a spline root (Nyquist-adjacent index): interpolate from
    // neighbours rather than dividing by ~0.
    let eps = 1e-7;
    for i in 0..k {
        if modulus_sq[i] < eps {
            let prev = modulus_sq[(i + k - 1) % k];
            let next = modulus_sq[(i + 1) % k];
            modulus_sq[i] = 0.5 * (prev + next);
        }
    }
    modulus_sq.iter().map(|v| 1.0 / v.max(1e-12)).collect()
}

struct AtomSpline {
    base: [isize; 3],
    w: [Vec<f64>; 3],
    dw: [Vec<f64>; 3],
}

fn atom_splines(topology: &Topology, grid: [usize; 3], order: usize) -> Vec<AtomSpline> {
    let lengths = [topology.box_.lx, topology.box_.ly, topology.box_.lz];
    topology
        .atoms
        .iter()
        .map(|atom| {
            let mut base = [0isize; 3];
            let mut w: [Vec<f64>; 3] = Default::default();
            let mut dw: [Vec<f64>; 3] = Default::default();
            for d in 0..3 {
                let l = lengths[d].max(1e-12);
                let mut s = atom.position[d] / l;
                s -= s.floor();
                let u = s * grid[d] as f64;
                let u0 = u.floor();
                let frac = (u - u0).clamp(0.0, 1.0 - 1e-14);
                base[d] = u0 as isize;
                // `fill_bspline` returns weights indexed so that arr[i]
                // corresponds to spline argument (order-1-i)+frac; the
                // spreading/gather convention below assigns grid point
                // (u0 - i) weight M_n(i + frac), i.e. index (order-1-i).
                // Reverse here once so `w[d][i]`/`dw[d][i]` line up directly
                // with grid point (u0 - i).
                let (mut wd, mut dwd) = fill_bspline(frac, order);
                wd.reverse();
                dwd.reverse();
                w[d] = wd;
                dw[d] = dwd;
            }
            AtomSpline { base, w, dw }
        })
        .collect()
}

fn grid_index(base: isize, i: usize, dim: usize) -> usize {
    let raw = base - i as isize;
    raw.rem_euclid(dim as isize) as usize
}

pub(crate) fn fft3d(data: &mut [Complex64], dims: [usize; 3], direction: FftDirection) {
    let [nx, ny, nz] = dims;
    let mut planner = FftPlanner::new();

    // Fastest axis (z): contiguous chunks, parallelizable.
    let fft_z = planner.plan_fft(nz, direction);
    data.par_chunks_mut(nz).for_each(|chunk| fft_z.process(chunk));

    // y axis: strided, gathered into a temporary line buffer per (ix, iz).
    let fft_y = planner.plan_fft(ny, direction);
    for ix in 0..nx {
        let mut buf = vec![Complex64::new(0.0, 0.0); ny];
        for iz in 0..nz {
            for iy in 0..ny {
                buf[iy] = data[(ix * ny + iy) * nz + iz];
            }
            fft_y.process(&mut buf);
            for iy in 0..ny {
                data[(ix * ny + iy) * nz + iz] = buf[iy];
            }
        }
    }

    // x axis: strided.
    let fft_x = planner.plan_fft(nx, direction);
    for iy in 0..ny {
        let mut buf = vec![Complex64::new(0.0, 0.0); nx];
        for iz in 0..nz {
            for ix in 0..nx {
                buf[ix] = data[(ix * ny + iy) * nz + iz];
            }
            fft_x.process(&mut buf);
            for ix in 0..nx {
                data[(ix * ny + iy) * nz + iz] = buf[ix];
            }
        }
    }
}

/// True SPME reciprocal-space energy and forces.
pub fn compute_pme_forces(topology: &Topology, ctx: &PmeContext) -> PmeResult {
    let n = topology.atoms.len();
    if n == 0 || !topology.box_.pbc {
        return PmeResult {
            forces: vec![],
            energy: 0.0,
            virial: 0.0,
        };
    }

    let grid = ctx.grid_size;
    let order = ctx.order;
    let [nx, ny, nz] = grid;
    let volume = topology.box_.lx * topology.box_.ly * topology.box_.lz;
    if volume <= 0.0 {
        return PmeResult {
            forces: vec![[0.0; 3]; n],
            energy: 0.0,
            virial: 0.0,
        };
    }
    let lengths = [topology.box_.lx, topology.box_.ly, topology.box_.lz];
    let alpha = ctx.alpha;

    let splines = atom_splines(topology, grid, order);

    // --- Charge spreading (serial: overlapping writes to shared grid). ---
    let mut q_grid: Vec<Complex64> = vec![Complex64::new(0.0, 0.0); nx * ny * nz];
    for (atom, sp) in topology.atoms.iter().zip(splines.iter()) {
        let q = atom.charge;
        if q == 0.0 {
            continue;
        }
        for ix in 0..order {
            let gx = grid_index(sp.base[0], ix, nx);
            let wx = sp.w[0][ix];
            if wx == 0.0 {
                continue;
            }
            for iy in 0..order {
                let gy = grid_index(sp.base[1], iy, ny);
                let wxy = wx * sp.w[1][iy];
                if wxy == 0.0 {
                    continue;
                }
                let row = (gx * ny + gy) * nz;
                for iz in 0..order {
                    let gz = grid_index(sp.base[2], iz, nz);
                    let weight = wxy * sp.w[2][iz];
                    q_grid[row + gz].re += q * weight;
                }
            }
        }
    }

    // --- Forward FFT, influence function, inverse FFT: factored into
    // `pme_recip_from_grid` so the GPU-resident backend's GPU-side charge
    // spreading can drive the exact same numerical pipeline. ---
    let charge_grid_re: Vec<f64> = q_grid.iter().map(|c| c.re).collect();
    let (potential_grid, energy, virial) =
        pme_recip_from_grid(&charge_grid_re, grid, order, alpha, lengths, volume);

    // --- Force gather: analytic derivative of E via the same splines. ---
    let forces: Vec<[f64; 3]> = topology
        .atoms
        .par_iter()
        .zip(splines.par_iter())
        .map(|(atom, sp)| {
            let q = atom.charge;
            let mut f = [0.0f64; 3];
            if q == 0.0 {
                return f;
            }
            let scale = [
                grid[0] as f64 / lengths[0],
                grid[1] as f64 / lengths[1],
                grid[2] as f64 / lengths[2],
            ];
            for ix in 0..order {
                let gx = grid_index(sp.base[0], ix, nx);
                let wx = sp.w[0][ix];
                let dwx = sp.dw[0][ix];
                for iy in 0..order {
                    let gy = grid_index(sp.base[1], iy, ny);
                    let wy = sp.w[1][iy];
                    let dwy = sp.dw[1][iy];
                    let row = (gx * ny + gy) * nz;
                    for iz in 0..order {
                        let gz = grid_index(sp.base[2], iz, nz);
                        let wz = sp.w[2][iz];
                        let dwz = sp.dw[2][iz];
                        // dE/dQ at this grid point is 2*Re(theta_q) after
                        // the inverse transform above (see derivation);
                        // `pme_recip_from_grid` already applies that factor.
                        let g = potential_grid[row + gz] as f64;
                        f[0] -= q * dwx * wy * wz * scale[0] * g;
                        f[1] -= q * wx * dwy * wz * scale[1] * g;
                        f[2] -= q * wx * wy * dwz * scale[2] * g;
                    }
                }
            }
            f
        })
        .collect();

    PmeResult {
        forces,
        energy,
        virial,
    }
}

/// The middle third of `compute_pme_forces` (forward FFT, B-spline-moduli-
/// corrected influence function, inverse FFT), factored out so a caller who
/// already has a charge grid spread some other way (in particular: the
/// GPU-resident backend's `pme_spread` WGSL kernel, which spreads charges
/// from GPU-resident positions with no host readback of positions
/// themselves) can drive the same numerically-exact CPU FFT pipeline
/// without needing atom positions/charges again. `charge_grid` is the
/// spread charge density, real-valued, row-major `(nx, ny, nz)` — exactly
/// what `compute_pme_forces`'s internal `q_grid` holds right before its
/// `fft3d(&mut q_grid, ...)` call, just without the imaginary part (always
/// zero for a real charge density).
///
/// Returns `(potential_grid, energy, virial)` where `potential_grid` is
/// `2 * Re(theta_q)` after the inverse FFT (row-major `(nx, ny, nz)`,
/// `f32`) — precisely the per-grid-point quantity `compute_pme_forces`'s
/// force-gather loop calls `g`, so a force-gather kernel/function fed this
/// grid reproduces `compute_pme_forces`'s forces exactly (same weights,
/// same derivative convention) given the same B-spline weights/derivatives
/// at each atom.
pub fn pme_recip_from_grid(
    charge_grid: &[f64],
    grid: [usize; 3],
    order: usize,
    alpha: f64,
    box_lengths: [f64; 3],
    volume: f64,
) -> (Vec<f32>, f64, f64) {
    let [nx, ny, nz] = grid;
    let mut q_grid: Vec<Complex64> = charge_grid.iter().map(|&re| Complex64::new(re, 0.0)).collect();

    fft3d(&mut q_grid, grid, FftDirection::Forward);

    let bmod_x = bspline_moduli(order, nx);
    let bmod_y = bspline_moduli(order, ny);
    let bmod_z = bspline_moduli(order, nz);

    let coeff = COULOMB_CONSTANT / (2.0 * PI * volume);
    let pi2 = PI * PI;

    let signed_freq = |i: usize, k: usize| -> f64 {
        if i <= k / 2 {
            i as f64
        } else {
            i as f64 - k as f64
        }
    };

    let mut theta_q = vec![Complex64::new(0.0, 0.0); nx * ny * nz];
    let mut energy = 0.0f64;
    let mut virial = 0.0f64;

    for ix in 0..nx {
        let mx = signed_freq(ix, nx) / box_lengths[0];
        for iy in 0..ny {
            let my = signed_freq(iy, ny) / box_lengths[1];
            let row = (ix * ny + iy) * nz;
            for iz in 0..nz {
                if ix == 0 && iy == 0 && iz == 0 {
                    continue;
                }
                let mz = signed_freq(iz, nz) / box_lengths[2];
                let m2 = mx * mx + my * my + mz * mz;
                if m2 < 1e-14 {
                    continue;
                }
                let bfac = bmod_x[ix] * bmod_y[iy] * bmod_z[iz];
                let theta = (-pi2 * m2 / (alpha * alpha)).exp() / m2 * bfac;
                let qhat = q_grid[row + iz];
                let e_term = coeff * theta * qhat.norm_sqr();
                energy += e_term;
                virial += e_term * (1.0 - 2.0 * pi2 * m2 / (alpha * alpha));
                theta_q[row + iz] = qhat * (coeff * theta);
            }
        }
    }

    fft3d(&mut theta_q, grid, FftDirection::Inverse);

    let potential_grid: Vec<f32> = theta_q.iter().map(|c| (2.0 * c.re) as f32).collect();
    (potential_grid, energy, virial)
}

/// Brute-force O(N*K^3) direct reciprocal Ewald sum plus the matching
/// real-space (minimum-image) and self-energy terms, using the exact same
/// alpha the SPME path would pick for `ewald_rtol`/`cutoff`. This is a slow
/// correctness reference only — used by tests to validate the fast SPME
/// implementation (`compute_pme_forces`) against an independent
/// implementation of the same Ewald sum, not a production code path.
pub fn direct_ewald_total_energy(
    topology: &Topology,
    cutoff: f64,
    ewald_rtol: f64,
    k_max: i64,
) -> f64 {
    let alpha = ewald_alpha_from_rtol(cutoff, ewald_rtol);
    let lengths = [topology.box_.lx, topology.box_.ly, topology.box_.lz];
    let volume = lengths[0] * lengths[1] * lengths[2];
    let coeff = COULOMB_CONSTANT / (2.0 * PI * volume);
    let pi2 = PI * PI;

    // Reciprocal-space sum.
    let mut recip_energy = 0.0;
    for mx in -k_max..=k_max {
        for my in -k_max..=k_max {
            for mz in -k_max..=k_max {
                if mx == 0 && my == 0 && mz == 0 {
                    continue;
                }
                let m = [
                    mx as f64 / lengths[0],
                    my as f64 / lengths[1],
                    mz as f64 / lengths[2],
                ];
                let m2 = m[0] * m[0] + m[1] * m[1] + m[2] * m[2];
                let theta = (-pi2 * m2 / (alpha * alpha)).exp() / m2;
                let mut sk = 0.0;
                let mut ck = 0.0;
                for atom in &topology.atoms {
                    let phase = 2.0
                        * PI
                        * (m[0] * atom.position[0]
                            + m[1] * atom.position[1]
                            + m[2] * atom.position[2]);
                    sk += atom.charge * phase.sin();
                    ck += atom.charge * phase.cos();
                }
                recip_energy += coeff * theta * (sk * sk + ck * ck);
            }
        }
    }

    // Real-space (minimum-image) sum.
    let mut real_energy = 0.0;
    let n = topology.atoms.len();
    for i in 0..n {
        for j in (i + 1)..n {
            let mut dr = [0.0; 3];
            for k in 0..3 {
                let mut d = topology.atoms[j].position[k] - topology.atoms[i].position[k];
                d -= (d / lengths[k]).round() * lengths[k];
                dr[k] = d;
            }
            let r2 = dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2];
            let r = r2.sqrt();
            if r >= cutoff {
                continue;
            }
            let (e, _) = crate::electrostatics::ewald::screened_coulomb_energy_force(
                topology.atoms[i].charge,
                topology.atoms[j].charge,
                r,
                r2,
                alpha,
                1.0,
            );
            real_energy += e;
        }
    }

    let self_energy: f64 = -COULOMB_CONSTANT * alpha / PI.sqrt()
        * topology.atoms.iter().map(|a| a.charge * a.charge).sum::<f64>();

    recip_energy + real_energy + self_energy
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::topology::{AtomRecord, SimulationBox, Topology};
    use rand::Rng;
    use std::f64::consts::PI;

    fn make_atom(charge: f64, position: [f64; 3]) -> AtomRecord {
        AtomRecord {
            element: "X".into(),
            name: "X".into(),
            mass: 1.0,
            charge,
            sigma: 0.0,
            epsilon: 0.0,
            position,
            residue_id: 0,
            molecule_id: 0,
            born_r: None,
        }
    }

    fn cubic_box(l: f64) -> SimulationBox {
        SimulationBox {
            lx: l,
            ly: l,
            lz: l,
            pbc: true,
        }
    }

    /// Independent, brute-force O(N*K^3) direct reciprocal Ewald sum, used
    /// only as a correctness reference for the SPME implementation above.
    /// Uses the same coefficient convention (ke/(2*pi*V), sum over the full
    /// signed-frequency grid) as `compute_pme_forces`.
    fn direct_ewald_reciprocal(topology: &Topology, alpha: f64, k_max: i64) -> (f64, Vec<[f64; 3]>) {
        let lengths = [topology.box_.lx, topology.box_.ly, topology.box_.lz];
        let volume = lengths[0] * lengths[1] * lengths[2];
        let coeff = COULOMB_CONSTANT / (2.0 * PI * volume);
        let pi2 = PI * PI;
        let n = topology.atoms.len();
        let mut energy = 0.0;
        let mut forces = vec![[0.0f64; 3]; n];

        for mx in -k_max..=k_max {
            for my in -k_max..=k_max {
                for mz in -k_max..=k_max {
                    if mx == 0 && my == 0 && mz == 0 {
                        continue;
                    }
                    let m = [
                        mx as f64 / lengths[0],
                        my as f64 / lengths[1],
                        mz as f64 / lengths[2],
                    ];
                    let m2 = m[0] * m[0] + m[1] * m[1] + m[2] * m[2];
                    let theta = (-pi2 * m2 / (alpha * alpha)).exp() / m2;

                    let mut sk = 0.0;
                    let mut ck = 0.0;
                    for atom in &topology.atoms {
                        let phase = 2.0 * PI
                            * (m[0] * atom.position[0]
                                + m[1] * atom.position[1]
                                + m[2] * atom.position[2]);
                        sk += atom.charge * phase.sin();
                        ck += atom.charge * phase.cos();
                    }
                    let s2 = sk * sk + ck * ck;
                    energy += coeff * theta * s2;

                    let pref = 2.0 * coeff * theta;
                    for (ai, atom) in topology.atoms.iter().enumerate() {
                        let phase = 2.0 * PI
                            * (m[0] * atom.position[0]
                                + m[1] * atom.position[1]
                                + m[2] * atom.position[2]);
                        let s = phase.sin();
                        let c = phase.cos();
                        // dE/dr_i from d(phase)/dr_i = 2*pi*m.
                        let dedphase = pref * atom.charge * (ck * (-s) + sk * c);
                        for k in 0..3 {
                            forces[ai][k] -= dedphase * 2.0 * PI * m[k];
                        }
                    }
                }
            }
        }
        (energy, forces)
    }

    #[test]
    fn ion_pair_finite() {
        let mut top = Topology::new();
        top.box_ = cubic_box(3.0);
        top.atoms.push(make_atom(1.0, [0.5, 0.5, 0.5]));
        top.atoms.push(make_atom(-1.0, [2.0, 2.0, 2.0]));
        let ctx = PmeContext::new(&top, 1.0);
        let r = compute_pme_forces(&top, &ctx);
        assert!(r.energy.is_finite());
        assert_eq!(r.forces.len(), 2);
    }

    #[test]
    fn spme_matches_direct_ewald_random_neutral_box() {
        let mut rng = rand::thread_rng();
        let l = 3.0;
        let mut top = Topology::new();
        top.box_ = cubic_box(l);
        let n_pairs = 12;
        for _ in 0..n_pairs {
            let pos: [f64; 3] = [
                rng.gen_range(0.0..l),
                rng.gen_range(0.0..l),
                rng.gen_range(0.0..l),
            ];
            top.atoms.push(make_atom(1.0, pos));
            let pos2: [f64; 3] = [
                rng.gen_range(0.0..l),
                rng.gen_range(0.0..l),
                rng.gen_range(0.0..l),
            ];
            top.atoms.push(make_atom(-1.0, pos2));
        }

        let alpha = ewald_alpha_from_rtol(1.0, 1e-5);
        let ctx = PmeContext::with_params(&top, 1.0, 1e-5, 0.02, 6);
        let spme = compute_pme_forces(&top, &ctx);
        let (direct_energy, direct_forces) = direct_ewald_reciprocal(&top, alpha, 12);

        let rel_err = ((spme.energy - direct_energy) / direct_energy).abs();
        assert!(
            rel_err < 1e-4,
            "energy mismatch: spme={} direct={} rel={}",
            spme.energy,
            direct_energy,
            rel_err
        );

        for (fa, fb) in spme.forces.iter().zip(direct_forces.iter()) {
            for k in 0..3 {
                let denom = fb[k].abs().max(1.0);
                assert!(
                    (fa[k] - fb[k]).abs() / denom < 1e-3,
                    "force mismatch: spme={:?} direct={:?}",
                    fa,
                    fb
                );
            }
        }
    }

    #[test]
    fn madelung_constant_nacl() {
        // Rock-salt NaCl lattice, single conventional cubic cell (4 NaCl
        // formula units), reduced coordinates with nearest-neighbour
        // distance a = 0.5 (box length L = 1.0). The Madelung constant is
        // defined by E_i = -M * ke * q^2 / a per ion; here we recover M via
        // the total lattice energy per ion pair.
        let l = 1.0;
        let a = 0.5;
        let mut top = Topology::new();
        top.box_ = cubic_box(l);
        let mut positions = Vec::new();
        for ix in 0..2 {
            for iy in 0..2 {
                for iz in 0..2 {
                    positions.push([ix as f64 * a, iy as f64 * a, iz as f64 * a]);
                }
            }
        }
        for p in &positions {
            let parity = ((p[0] / a).round() as i64
                + (p[1] / a).round() as i64
                + (p[2] / a).round() as i64)
                % 2;
            let q = if parity == 0 { 1.0 } else { -1.0 };
            top.atoms.push(make_atom(q, *p));
        }

        let cutoff = 0.49;
        let ctx = PmeContext::with_params(&top, cutoff, 1e-8, 0.01, 8);
        let recip = compute_pme_forces(&top, &ctx);

        // Real-space (minimum-image, single shell is enough given the
        // small, highly-neutral unit cell and Ewald screening) + self
        // energy, using the same alpha as the reciprocal sum.
        let mut real_energy = 0.0;
        for i in 0..top.atoms.len() {
            for j in (i + 1)..top.atoms.len() {
                let mut dr = [0.0; 3];
                for k in 0..3 {
                    let mut d = top.atoms[j].position[k] - top.atoms[i].position[k];
                    d -= (d / l).round() * l;
                    dr[k] = d;
                }
                let r2 = dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2];
                let r = r2.sqrt();
                let (e, _) = super::super::ewald::screened_coulomb_energy_force(
                    top.atoms[i].charge,
                    top.atoms[j].charge,
                    r,
                    r2,
                    ctx.alpha,
                    1.0,
                );
                real_energy += e;
            }
        }
        let self_energy: f64 = -COULOMB_CONSTANT * ctx.alpha / PI.sqrt()
            * top.atoms.iter().map(|a| a.charge * a.charge).sum::<f64>();

        let total = real_energy + recip.energy + self_energy;
        let per_ion = total / top.atoms.len() as f64;
        // Total lattice energy = (1/2) * sum_i E_i (each pair counted once),
        // so per-ion energy (total/N) is half of E_i = -M*ke*q^2/a; correct
        // for that factor of two here.
        let madelung = -2.0 * per_ion * a / COULOMB_CONSTANT;

        assert!(
            (madelung - 1.747565).abs() < 1e-4,
            "madelung constant off: got {madelung}"
        );
    }

    #[test]
    fn forces_match_finite_difference_gradient() {
        let mut top = Topology::new();
        top.box_ = cubic_box(2.5);
        top.atoms.push(make_atom(0.8, [0.4, 1.1, 0.9]));
        top.atoms.push(make_atom(-0.5, [1.6, 0.6, 1.8]));
        top.atoms.push(make_atom(-0.3, [2.0, 2.0, 0.3]));

        let ctx = PmeContext::new(&top, 1.0);
        let result = compute_pme_forces(&top, &ctx);

        let h = 1e-6;
        for ai in 0..top.atoms.len() {
            for k in 0..3 {
                let mut plus = top.clone();
                plus.atoms[ai].position[k] += h;
                let ep = compute_pme_forces(&plus, &ctx).energy;

                let mut minus = top.clone();
                minus.atoms[ai].position[k] -= h;
                let em = compute_pme_forces(&minus, &ctx).energy;

                let numeric = -(ep - em) / (2.0 * h);
                let analytic = result.forces[ai][k];
                assert!(
                    (numeric - analytic).abs() < 1e-3 * analytic.abs().max(1.0),
                    "atom {ai} dim {k}: numeric={numeric} analytic={analytic}"
                );
            }
        }
    }

    #[test]
    fn translation_invariance() {
        let mut top = Topology::new();
        top.box_ = cubic_box(3.0);
        top.atoms.push(make_atom(1.0, [0.3, 0.7, 1.1]));
        top.atoms.push(make_atom(-1.0, [1.9, 2.2, 0.4]));
        top.atoms.push(make_atom(0.5, [2.5, 0.9, 2.1]));
        top.atoms.push(make_atom(-0.5, [0.6, 1.6, 0.8]));

        // Use a fine grid/high order so the (unavoidable, grid-phase
        // dependent) B-spline discretization error is small enough to
        // check translation invariance to tight tolerance.
        let ctx = PmeContext::with_params(&top, 1.0, 1e-5, 0.02, 6);
        let base = compute_pme_forces(&top, &ctx);

        let shift = [0.37, 1.21, 2.05];
        let mut shifted = top.clone();
        let lengths = [top.box_.lx, top.box_.ly, top.box_.lz];
        for atom in shifted.atoms.iter_mut() {
            for k in 0..3 {
                let mut p = atom.position[k] + shift[k];
                p -= (p / lengths[k]).floor() * lengths[k];
                atom.position[k] = p;
            }
        }
        let shifted_result = compute_pme_forces(&shifted, &ctx);

        assert!((base.energy - shifted_result.energy).abs() / base.energy.abs() < 1e-4);
        for (fa, fb) in base.forces.iter().zip(shifted_result.forces.iter()) {
            for k in 0..3 {
                assert!(
                    (fa[k] - fb[k]).abs() < 1e-3 * fa[k].abs().max(1.0),
                    "fa={fa:?} fb={fb:?}"
                );
            }
        }
    }

    #[test]
    fn spme_is_much_faster_than_direct_sum_at_scale() {
        use std::time::Instant;
        let mut rng = rand::thread_rng();
        let l = 6.0;
        let mut top = Topology::new();
        top.box_ = cubic_box(l);
        let n = 3000;
        for i in 0..n {
            let q = if i % 2 == 0 { 1.0 } else { -1.0 };
            top.atoms.push(make_atom(
                q,
                [
                    rng.gen_range(0.0..l),
                    rng.gen_range(0.0..l),
                    rng.gen_range(0.0..l),
                ],
            ));
        }
        let ctx = PmeContext::new(&top, 1.0);

        let t0 = Instant::now();
        let _ = compute_pme_forces(&top, &ctx);
        let spme_time = t0.elapsed();

        // Direct sum at a modest k_max (comparable accuracy grid) on a
        // smaller atom subsample so the O(N*K^3) reference completes in a
        // reasonable time, then we scale the atom-loop cost linearly to
        // estimate the full-N direct-sum cost for the speedup comparison.
        let sample = 200;
        let mut sub = Topology::new();
        sub.box_ = top.box_.clone();
        sub.atoms = top.atoms[..sample].to_vec();
        let t1 = Instant::now();
        let _ = direct_ewald_reciprocal(&sub, ctx.alpha, 10);
        let direct_sample_time = t1.elapsed();
        let estimated_direct_time = direct_sample_time.mul_f64(n as f64 / sample as f64);

        eprintln!(
            "N={n}: SPME={spme_time:?} estimated direct={estimated_direct_time:?} speedup={:.1}x",
            estimated_direct_time.as_secs_f64() / spme_time.as_secs_f64()
        );
        assert!(
            spme_time < estimated_direct_time,
            "SPME ({spme_time:?}) should be much faster than the estimated \
             direct sum ({estimated_direct_time:?}) at N={n}"
        );
    }
}

#[cfg(test)]
mod fft_tests {
    use super::*;

    #[test]
    fn fft3d_roundtrip() {
        let dims = [4usize, 5, 6];
        let n: usize = dims.iter().product();
        let mut data: Vec<Complex64> = (0..n)
            .map(|i| Complex64::new(i as f64, (i as f64) * 0.3))
            .collect();
        let orig = data.clone();
        fft3d(&mut data, dims, FftDirection::Forward);
        fft3d(&mut data, dims, FftDirection::Inverse);
        for i in 0..n {
            let scaled = orig[i] * (n as f64);
            assert!(
                (data[i] - scaled).norm() < 1e-6,
                "i={i} got={:?} expect={:?}",
                data[i],
                scaled
            );
        }
    }
}
