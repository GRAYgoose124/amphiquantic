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
            writeln!(
                w,
                "# aqtrj v1 step natoms epot ekin box_lx box_ly box_lz pbc"
            )
            .map_err(|e| e.to_string())?;
            self.started = true;
        }
        writeln!(
            w,
            "FRAME {} {} {:.8} {:.8} {:.6} {:.6} {:.6} {}",
            step,
            topology.atoms.len(),
            potential,
            kinetic,
            topology.box_.lx,
            topology.box_.ly,
            topology.box_.lz,
            if topology.box_.pbc { 1 } else { 0 }
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

pub struct TrajectoryReader {
    path: String,
}

impl TrajectoryReader {
    pub fn new(path: impl Into<String>) -> Self {
        Self { path: path.into() }
    }

    pub fn last_frame_positions(&self, topology: &Topology) -> Result<Vec<[f64; 3]>, String> {
        let content = std::fs::read_to_string(&self.path).map_err(|e| e.to_string())?;
        let mut positions = topology.positions();
        let mut in_frame = false;
        let mut idx = 0;
        for line in content.lines() {
            if line.starts_with("FRAME ") {
                in_frame = true;
                idx = 0;
                continue;
            }
            if !in_frame || line.starts_with('#') {
                continue;
            }
            let parts: Vec<f64> = line
                .split_whitespace()
                .filter_map(|s| s.parse().ok())
                .collect();
            if parts.len() >= 3 && idx < positions.len() {
                positions[idx] = [parts[0], parts[1], parts[2]];
                idx += 1;
            }
        }
        Ok(positions)
    }
}
