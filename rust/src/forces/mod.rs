pub mod cpu;
pub mod gpu;
pub mod hybrid;

use crate::electrostatics::{ewald_energy_correction_with_alpha, excluded_pair_correction};
use crate::topology::Topology;
use cpu::ForceResult as CpuForceResult;

#[derive(Clone)]
pub enum ForceBackend {
    Cpu,
    Gpu,
    Hybrid,
}

pub fn compute_forces(topology: &Topology, cutoff: f64, backend: ForceBackend) -> CpuForceResult {
    if topology.is_implicit_gb() {
        let bonded = cpu::compute_bonded_forces(topology);
        let lj = cpu::compute_lj_forces(topology, cutoff);
        let gb = crate::electrostatics::gb::compute_gb_forces(topology);
        let mut result = cpu::merge_force_results(bonded, lj);
        for (f, fg) in result.forces.iter_mut().zip(gb.forces.iter()) {
            f[0] += fg[0];
            f[1] += fg[1];
            f[2] += fg[2];
        }
        result.potential_energy += gb.potential_energy;
        return result;
    }
    match backend {
        ForceBackend::Cpu => {
            if topology.box_.pbc {
                let ctx = crate::electrostatics::pme::PmeContext::new(topology, cutoff);
                let mut result =
                    cpu::compute_forces_with_pme(topology, cutoff, Some(&ctx));
                result.potential_energy += ewald_energy_correction_with_alpha(topology, ctx.alpha);
                result
            } else {
                cpu::compute_forces(topology, cutoff)
            }
        }
        ForceBackend::Gpu => {
            let bonded = cpu::compute_bonded_forces(topology);
            let gpu_nb = gpu::compute_forces_gpu(topology, cutoff);
            let mut result = cpu::merge_force_results(bonded, gpu_nb);
            if topology.box_.pbc {
                let ctx = crate::electrostatics::pme::PmeContext::new(topology, cutoff);
                let pme = crate::electrostatics::pme::compute_pme_forces(topology, &ctx);
                for (f, fp) in result.forces.iter_mut().zip(pme.forces.iter()) {
                    f[0] += fp[0];
                    f[1] += fp[1];
                    f[2] += fp[2];
                }
                result.potential_energy +=
                    pme.energy + ewald_energy_correction_with_alpha(topology, ctx.alpha);

                let (excl_energy, excl_forces, _) =
                    excluded_pair_correction(topology, ctx.alpha);
                result.potential_energy += excl_energy;
                for (f, fe) in result.forces.iter_mut().zip(excl_forces.iter()) {
                    f[0] += fe[0];
                    f[1] += fe[1];
                    f[2] += fe[2];
                }
            }
            result
        }
        ForceBackend::Hybrid => hybrid::compute_forces_hybrid(topology, cutoff, topology.box_.pbc),
    }
}

pub fn backend_from_env() -> ForceBackend {
    if hybrid::hybrid_enabled() {
        return ForceBackend::Hybrid;
    }
    match std::env::var("AMPHI_FORCE_BACKEND").as_deref() {
        Ok("cpu") => ForceBackend::Cpu,
        Ok("gpu") => ForceBackend::Gpu,
        _ => ForceBackend::Hybrid,
    }
}
