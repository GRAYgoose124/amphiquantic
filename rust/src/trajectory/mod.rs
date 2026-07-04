use crate::topology::Topology;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

pub struct TrajectoryWriter {
    path: String,
    started: bool,
}

impl TrajectoryWriter {
    pub fn new(path: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            started: false,
        }
    }

    pub fn write_frame(&mut self, step: u64, topology: &Topology, potential: f64, kinetic: f64) -> Result<(), String> {
        let file = File::options()
            .create(true)
            .append(true)
            .open(&self.path)
            .map_err(|e| e.to_string())?;
        let mut w = BufWriter::new(file);
        if !self.started {
            writeln!(w, "# aqtrj v1 step natoms epot ekin").map_err(|e| e.to_string())?;
            self.started = true;
        }
        writeln!(
            w,
            "FRAME {} {} {:.8} {:.8}",
            step,
            topology.atoms.len(),
            potential,
            kinetic
        )
        .map_err(|e| e.to_string())?;
        for atom in &topology.atoms {
            writeln!(
                w,
                "{:.6} {:.6} {:.6}",
                atom.position[0], atom.position[1], atom.position[2]
            )
            .map_err(|e| e.to_string())?;
        }
        Ok(())
    }
}

pub fn default_trajectory_path(base: &Path) -> String {
    base.with_extension("aqtrj")
        .to_string_lossy()
        .into_owned()
}
