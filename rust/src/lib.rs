use pyo3::prelude::*;
use pyo3::wrap_pymodule;
use pyo3::Python;

mod atom;
mod bonds;
mod pdb;
mod builder;
mod compute_pipeline;
mod utilities;

use pdb::PdbFilePy;
use builder::builder as build;
use compute_pipeline::{run_atom_pipeline, AtomPipelineParams};


#[pymodule]
fn simulate(_py: Python, m: Bound<PyModule>) -> PyResult<()> {
    #[pyfn(m, name = "run_atom_pipeline")]
    // params should be a dict compatible with ypthon dict
    fn rap(coords: Vec<(f64, f64, f64)>, atom_types: Vec<String>, bonds: Vec<(usize, usize)>, params: AtomPipelineParams) -> Vec<(f64, f64, f64)> {
        let coords_vec: Vec<[f64; 3]> = coords.iter().map(|c| [c.0, c.1, c.2]).collect();
        run_atom_pipeline(&coords_vec, &atom_types, &bonds, params).iter().map(|c| (c[0], c[1], c[2])).collect()
    }
    
    #[pyfn(m, name = "run_simulation")]
    fn run_simulation_py(coords: Vec<(f64, f64, f64)>, atom_types: Vec<String>, bonds: Vec<(usize, usize)>) -> PyResult<PdbFilePy> {
        let pdb = rap(coords, atom_types.clone(), bonds.clone(), AtomPipelineParams {
            step_size: 0.1,
            max_steps: 100,
            process_type: 2,
        });
        Ok(PdbFilePy {
            coords: pdb,
            atom_types: atom_types,
            bonds: bonds,
        })
    }

    #[pyfn(m, name = "run_minimization")]
    fn run_minimization_py(coords: Vec<(f64, f64, f64)>, atom_types: Vec<String>, bonds: Vec<(usize, usize)>) -> PyResult<PdbFilePy> {
        let pdb = rap(coords, atom_types.clone(), bonds.clone(), AtomPipelineParams {
            step_size: 0.1,
            max_steps: 100,
            process_type: 1,
        });
        Ok(PdbFilePy {
            coords: pdb,
            atom_types: atom_types,
            bonds: bonds,
        })
    }

    #[pyfn(m, name = "run_relaxation")]
    fn run_relaxation_py(coords: Vec<(f64, f64, f64)>, atom_types: Vec<String>, bonds: Vec<(usize, usize)>) -> PyResult<PdbFilePy> {
        let pdb = rap(coords, atom_types.clone(), bonds.clone(), AtomPipelineParams {
            step_size: 0.1,
            max_steps: 100,
            process_type: 0,
        });
        Ok(PdbFilePy {
            coords: pdb,
            atom_types: atom_types,
            bonds: bonds,
        })
    }

    Ok(())

}

#[pymodule]
fn rustquantic(_py: Python, m: Bound<PyModule>) -> PyResult<()> {
    m.add_class::<PdbFilePy>()?;
    m.add_wrapped(wrap_pymodule!(crate::utilities::utilities))?;
    m.add_wrapped(wrap_pymodule!(build))?;
    m.add_wrapped(wrap_pymodule!(simulate))?;

    Ok(())
}
