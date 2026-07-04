use crate::forces::cpu::{self, ForceResult};
use crate::forces::gpu::GpuForceEngine;
use crate::topology::Topology;
use std::sync::OnceLock;

static GPU_ENGINE: OnceLock<GpuForceEngine> = OnceLock::new();

fn gpu_engine() -> &'static GpuForceEngine {
    GPU_ENGINE.get_or_init(GpuForceEngine::new)
}

/// Hybrid parallel: CPU bonded + GPU short-range (LJ + screened Coulomb) + CPU PME reciprocal.
pub fn compute_forces_hybrid(
    topology: &Topology,
    cutoff: f64,
    use_pme: bool,
) -> ForceResult {
    let pme_ctx = if use_pme && topology.box_.pbc {
        Some(crate::electrostatics::pme::PmeContext::new(topology, cutoff))
    } else {
        None
    };
    let alpha = pme_ctx.as_ref().map(|c| c.alpha).unwrap_or(0.34);
    let screened = use_pme && topology.box_.pbc;

    let (bonded, gpu_nb) = rayon::join(
        || cpu::compute_bonded_forces(topology),
        || gpu_engine().compute_nonbonded(topology, cutoff, screened, alpha),
    );

    let mut result = cpu::merge_force_results(bonded, gpu_nb);

    if let Some(ctx) = pme_ctx.as_ref() {
        let pme = crate::electrostatics::pme::compute_pme_forces(topology, ctx);
        for (f, fp) in result.forces.iter_mut().zip(pme.forces.iter()) {
            f[0] += fp[0];
            f[1] += fp[1];
            f[2] += fp[2];
        }
        result.potential_energy += pme.energy;
        result.potential_energy += crate::electrostatics::ewald_energy_correction(topology);
    }

    result
}

pub fn hybrid_enabled() -> bool {
    match std::env::var("AMPHI_HYBRID").as_deref() {
        Ok("0") | Ok("false") => false,
        Ok("1") | Ok("true") => true,
        _ => std::env::var("AMPHI_FORCE_BACKEND").as_deref() != Ok("cpu"),
    }
}
