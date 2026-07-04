pub mod cpu;
pub mod gpu;

use crate::topology::Topology;
use cpu::ForceResult as CpuForceResult;

#[derive(Clone)]
pub enum ForceBackend {
    Cpu,
    Gpu,
}

pub fn compute_forces(topology: &Topology, cutoff: f64, backend: ForceBackend) -> CpuForceResult {
    match backend {
        ForceBackend::Cpu => cpu::compute_forces(topology, cutoff),
        ForceBackend::Gpu => {
            let gpu = gpu::compute_forces_gpu(topology, cutoff);
            CpuForceResult {
                forces: gpu.forces,
                potential_energy: gpu.potential_energy,
            }
        }
    }
}

pub fn backend_from_env() -> ForceBackend {
    match std::env::var("AMPHI_FORCE_BACKEND").as_deref() {
        Ok("cpu") => ForceBackend::Cpu,
        _ => ForceBackend::Gpu,
    }
}
