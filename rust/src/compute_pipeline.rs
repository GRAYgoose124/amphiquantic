use crate::integrator::{langevin_step, MdState};
use crate::minimize::minimize;
use crate::topology::Topology;
use bytemuck::{Pod, Zeroable};
use pyo3::prelude::*;

#[pyclass]
#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct AtomPipelineParams {
    pub step_size: f32,
    pub max_steps: u32,
    pub process_type: u32,
}

pub fn run_atom_pipeline(
    coords: &[[f64; 3]],
    atom_types: &[String],
    bonds: &[(usize, usize)],
    params: AtomPipelineParams,
) -> Vec<[f64; 3]> {
    run_md(
        &coords.iter().map(|c| (c[0], c[1], c[2])).collect::<Vec<_>>(),
        atom_types,
        bonds,
        params.process_type,
        params.max_steps as usize,
        params.step_size as f64,
    )
}

pub fn run_md(
    coords: &[(f64, f64, f64)],
    atom_types: &[String],
    bonds: &[(usize, usize)],
    process_type: u32,
    max_steps: usize,
    step_size: f64,
) -> Vec<[f64; 3]> {
    let mut top = Topology::from_pdb_types(coords, atom_types, bonds);

    match process_type {
        1 => {
            let result = minimize(top, 1.0, max_steps, step_size, 1e-3);
            result.topology.positions()
        }
        0 | 2 => {
            let temp = if process_type == 0 { 300.0 } else { 310.0 };
            let mut state = MdState::new(top, temp);
            state.timestep = step_size;
            state.friction = if process_type == 0 { 2.0 } else { 1.0 };
            for _ in 0..max_steps {
                langevin_step(&mut state, 1.0);
            }
            state.topology.positions()
        }
        _ => top.positions(),
    }
}
