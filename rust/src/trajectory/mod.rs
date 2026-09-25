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

#[derive(Default, Clone, Copy)]
struct FrameMeta {
    natoms: usize,
    lx: f64,
    ly: f64,
    lz: f64,
    pbc: bool,
}

impl TrajectoryReader {
    pub fn new(path: impl Into<String>) -> Self {
        Self { path: path.into() }
    }

    fn parse_last_frame(&self) -> Result<(FrameMeta, Vec<[f64; 3]>), String> {
        let content = std::fs::read_to_string(&self.path).map_err(|e| e.to_string())?;
        let mut meta = FrameMeta::default();
        let mut positions: Vec<[f64; 3]> = Vec::new();
        let mut in_frame = false;
        for line in content.lines() {
            if line.starts_with("FRAME ") {
                in_frame = true;
                positions.clear();
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 9 {
                    meta = FrameMeta {
                        natoms: parts[2].parse().unwrap_or(0),
                        lx: parts[5].parse().unwrap_or(0.0),
                        ly: parts[6].parse().unwrap_or(0.0),
                        lz: parts[7].parse().unwrap_or(0.0),
                        pbc: parts[8] != "0",
                    };
                }
                continue;
            }
            if !in_frame || line.starts_with('#') {
                continue;
            }
            let parts: Vec<f64> = line
                .split_whitespace()
                .filter_map(|s| s.parse().ok())
                .collect();
            if parts.len() >= 3 {
                positions.push([parts[0], parts[1], parts[2]]);
            }
        }
        if positions.is_empty() {
            return Err(format!("No frames found in trajectory: {}", self.path));
        }
        Ok((meta, positions))
    }

    pub fn last_frame_positions(&self, topology: &Topology) -> Result<Vec<[f64; 3]>, String> {
        let (meta, positions) = self.parse_last_frame()?;
        if meta.natoms > 0 && meta.natoms != topology.atoms.len() {
            return Err(format!(
                "Trajectory natoms {} != topology {}",
                meta.natoms,
                topology.atoms.len()
            ));
        }
        if positions.len() != topology.atoms.len() {
            return Err(format!(
                "Trajectory frame has {} coords, topology has {}",
                positions.len(),
                topology.atoms.len()
            ));
        }
        Ok(positions)
    }

    pub fn apply_last_frame(&self, topology: &mut Topology) -> Result<(), String> {
        let (meta, positions) = self.parse_last_frame()?;
        if meta.natoms > 0 && meta.natoms != topology.atoms.len() {
            return Err(format!(
                "Restart natoms {} != topology {}",
                meta.natoms,
                topology.atoms.len()
            ));
        }
        if positions.len() != topology.atoms.len() {
            return Err(format!(
                "Restart frame has {} coords, topology has {}",
                positions.len(),
                topology.atoms.len()
            ));
        }
        for (atom, pos) in topology.atoms.iter_mut().zip(positions) {
            atom.position = pos;
        }
        if meta.lx > 0.0 {
            topology.box_.lx = meta.lx;
            topology.box_.ly = meta.ly;
            topology.box_.lz = meta.lz;
            topology.box_.pbc = meta.pbc;
        }
        Ok(())
    }
}
