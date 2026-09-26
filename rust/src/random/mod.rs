//! Philox4x32-10, a counter-based RNG (Salmon, Moraes, Dror & Shaw, 2011;
//! the algorithm implemented by the Random123 library). Used by the
//! GPU-resident backend's BAOAB Langevin thermostat (`gpu_resident.wgsl`'s
//! `philox4x32_10`, a line-for-line WGSL port of this) so every atom/DOF
//! gets an independent, reproducible Gaussian draw from `(key, counter)`
//! alone — no shared RNG state, no per-atom stream object, safe to call
//! from any GPU thread in any order.
//!
//! This Rust copy exists so the algorithm can be checked against the
//! published Random123 known-answer test vectors without needing a GPU
//! (`tests::philox_matches_random123_kat_vectors`), and as the reference
//! `philox_matches_wgsl_reference` cross-checks the WGSL kernel against.

const PHILOX_M0: u32 = 0xD2511F53;
const PHILOX_M1: u32 = 0xCD9E8D57;
const PHILOX_W0: u32 = 0x9E3779B9;
const PHILOX_W1: u32 = 0xBB67AE85;

fn mulhilo32(a: u32, b: u32) -> (u32, u32) {
    let product = (a as u64) * (b as u64);
    ((product >> 32) as u32, product as u32)
}

/// One Philox4x32-10 evaluation: `key` is the 2-word key (e.g.
/// `[seed_lo, seed_hi]`), `counter` is the 4-word counter (e.g.
/// `[step, atom_index, dof, 0]`); returns 4 pseudo-random `u32`s.
pub fn philox4x32_10(mut counter: [u32; 4], mut key: [u32; 2]) -> [u32; 4] {
    for _ in 0..10 {
        let (hi0, lo0) = mulhilo32(PHILOX_M0, counter[0]);
        let (hi1, lo1) = mulhilo32(PHILOX_M1, counter[2]);
        counter = [hi1 ^ counter[1] ^ key[0], lo1, hi0 ^ counter[3] ^ key[1], lo0];
        key[0] = key[0].wrapping_add(PHILOX_W0);
        key[1] = key[1].wrapping_add(PHILOX_W1);
    }
    counter
}

/// Converts a `u32` to a uniform `f64` in `[0, 1)` (53-bit-ish precision
/// isn't needed here; this just needs to avoid the exact endpoints for
/// `ln`/`sqrt` in Box-Muller).
fn u32_to_unit_f64(x: u32) -> f64 {
    ((x as f64) + 0.5) / (u32::MAX as f64 + 1.0)
}

/// Box-Muller transform: two independent uniforms -> two independent
/// standard-normal draws.
pub fn box_muller(u1: u32, u2: u32) -> (f64, f64) {
    let r1 = u32_to_unit_f64(u1).max(1e-300);
    let r2 = u32_to_unit_f64(u2);
    let radius = (-2.0 * r1.ln()).sqrt();
    let theta = 2.0 * std::f64::consts::PI * r2;
    (radius * theta.cos(), radius * theta.sin())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The two standard Random123 `kat_vectors` entries for philox4x32-10:
    /// all-zero key/counter, and all-`0xffffffff` key/counter. These are
    /// published, tool-independent reference values (see the Random123
    /// distribution's `kat_vectors` file / the paper's appendix), so a
    /// match here means this implementation is bit-exact Philox4x32-10, not
    /// merely "a" counter-based RNG.
    #[test]
    fn philox_matches_random123_kat_vectors() {
        let out = philox4x32_10([0, 0, 0, 0], [0, 0]);
        assert_eq!(out, [0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8]);

        let out = philox4x32_10(
            [0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff],
            [0xffffffff, 0xffffffff],
        );
        assert_eq!(out, [0x408f276d, 0x41c83b0e, 0xa20bc7c6, 0x6d5451fd]);
    }

    #[test]
    fn philox_is_deterministic_and_counter_sensitive() {
        let a = philox4x32_10([1, 2, 3, 4], [5, 6]);
        let b = philox4x32_10([1, 2, 3, 4], [5, 6]);
        assert_eq!(a, b, "same (key, counter) must give the same output");
        let c = philox4x32_10([1, 2, 3, 5], [5, 6]);
        assert_ne!(a, c, "changing the counter must change the output");
    }

    #[test]
    fn box_muller_produces_finite_normals_with_unit_ish_variance() {
        let mut sum = 0.0;
        let mut sumsq = 0.0;
        let mut count = 0.0;
        for i in 0..5000u32 {
            let out = philox4x32_10([0, i, 0, 0], [42, 7]);
            let (z0, z1) = box_muller(out[0], out[1]);
            for z in [z0, z1] {
                assert!(z.is_finite());
                sum += z;
                sumsq += z * z;
                count += 1.0;
            }
        }
        let mean = sum / count;
        let var = sumsq / count - mean * mean;
        assert!(mean.abs() < 0.1, "mean={mean}");
        assert!((var - 1.0).abs() < 0.2, "var={var}");
    }
}
