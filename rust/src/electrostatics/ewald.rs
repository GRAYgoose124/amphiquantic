/// Ewald screening helpers for split electrostatics.

pub fn erfc(x: f64) -> f64 {
    // Abramowitz & Stegun 7.1.26
    let t = 1.0 / (1.0 + 0.5 * x.abs());
    let tau = t
        * (-x * x - 1.26551223
            + t * (1.00002368
                + t * (0.37409196
                    + t * (0.09678418
                        + t * (-0.18628806
                            + t * (0.27886807
                                + t * (-1.13520398
                                    + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))))
        .exp();
    if x >= 0.0 { tau } else { 2.0 - tau }
}

pub fn screened_coulomb_energy_force(
    qi: f64,
    qj: f64,
    r: f64,
    r2: f64,
    alpha: f64,
    scale: f64,
) -> (f64, f64) {
    if r < 1e-12 {
        return (0.0, 0.0);
    }
    let pref = 138.935456 * scale * qi * qj;
    let arg = alpha * r;
    let erfc_val = erfc(arg);
    let energy = pref * erfc_val / r;
    let force_scalar = pref * (erfc_val / r2 + 2.0 * alpha * (-arg * arg).exp() / (std::f64::consts::PI.sqrt() * r));
    (energy, force_scalar)
}

pub fn direct_coulomb_energy_force(qi: f64, qj: f64, r: f64, r2: f64, scale: f64) -> (f64, f64) {
    let pref = 138.935456 * scale * qi * qj;
    (pref / r, pref / r2)
}
