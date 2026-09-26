use crate::topology::Topology;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

pub mod checkpoint;
pub mod dcd;
pub mod energy_log;
pub mod pdb_writer;
pub mod xyz;

pub use checkpoint::{read_checkpoint, write_checkpoint, Checkpoint};
pub use dcd::{read_dcd, DcdFrame, DcdTrajectory, DcdWriter};
pub use energy_log::EnergyLogWriter;
pub use pdb_writer::PdbTrajectoryWriter;
pub use xyz::XyzWriter;

/// Trajectory output format selectable via `--traj-format`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrajectoryFormat {
    /// The original plain-text `aqtrj` format (default; unchanged).
    Aqtrj,
    /// Binary CHARMM/NAMD-style DCD, with unit cell.
    Dcd,
    /// Multi-model PDB (`MODEL`/`ENDMDL` per frame).
    Pdb,
    /// Plain XYZ.
    Xyz,
}

impl TrajectoryFormat {
    pub fn parse(name: &str) -> Result<Self, String> {
        match name.to_ascii_lowercase().as_str() {
            "aqtrj" => Ok(TrajectoryFormat::Aqtrj),
            "dcd" => Ok(TrajectoryFormat::Dcd),
            "pdb" => Ok(TrajectoryFormat::Pdb),
            "xyz" => Ok(TrajectoryFormat::Xyz),
            other => Err(format!(
                "unknown trajectory format '{other}' (expected aqtrj, dcd, pdb or xyz)"
            )),
        }
    }
}

enum FormatWriter {
    Aqtrj { started: bool },
    Dcd(Option<DcdWriter>),
    Pdb(PdbTrajectoryWriter),
    Xyz(XyzWriter),
}

pub struct TrajectoryWriter {
    path: String,
    format: TrajectoryFormat,
    inner: FormatWriter,
}

impl TrajectoryWriter {
    /// Original behavior: plain-text `aqtrj` format, unchanged.
    pub fn new(path: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            format: TrajectoryFormat::Aqtrj,
            inner: FormatWriter::Aqtrj { started: false },
        }
    }

    /// Create a writer for a specific trajectory format. `Pdb` and `Xyz`
    /// truncate the target file immediately; `Dcd` truncates it lazily on
    /// the first `write_frame` call (once atom count is known); `Aqtrj`
    /// keeps its original append-on-first-write behavior.
    pub fn with_format(path: impl Into<String>, format: TrajectoryFormat) -> Result<Self, String> {
        let path = path.into();
        let inner = match format {
            TrajectoryFormat::Aqtrj => FormatWriter::Aqtrj { started: false },
            TrajectoryFormat::Dcd => FormatWriter::Dcd(None),
            TrajectoryFormat::Pdb => FormatWriter::Pdb(PdbTrajectoryWriter::new(&path)?),
            TrajectoryFormat::Xyz => FormatWriter::Xyz(XyzWriter::new(&path)?),
        };
        Ok(Self {
            path,
            format,
            inner,
        })
    }

    pub fn format(&self) -> TrajectoryFormat {
        self.format
    }

    pub fn write_frame(
        &mut self,
        step: u64,
        topology: &Topology,
        potential: f64,
        kinetic: f64,
    ) -> Result<(), String> {
        match &mut self.inner {
            FormatWriter::Aqtrj { started } => {
                let file = File::options()
                    .create(true)
                    .append(true)
                    .open(&self.path)
                    .map_err(|e| e.to_string())?;
                let mut w = BufWriter::new(file);
                if !*started {
                    writeln!(
                        w,
                        "# aqtrj v1 step natoms epot ekin box_lx box_ly box_lz pbc"
                    )
                    .map_err(|e| e.to_string())?;
                    *started = true;
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
            FormatWriter::Dcd(writer) => {
                if writer.is_none() {
                    *writer = Some(DcdWriter::create(&self.path, topology.atoms.len(), 0.002)?);
                }
                let positions: Vec<[f64; 3]> = topology.atoms.iter().map(|a| a.position).collect();
                let box_ = &topology.box_;
                writer.as_mut().unwrap().write_frame(
                    &positions,
                    (
                        box_.lx.max(0.0),
                        box_.ly.max(0.0),
                        box_.lz.max(0.0),
                    ),
                )
            }
            FormatWriter::Pdb(writer) => writer.write_frame(topology),
            FormatWriter::Xyz(writer) => writer.write_frame(step, topology),
        }
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

#[cfg(test)]
mod format_tests {
    use super::*;
    use crate::topology::{AtomRecord, SimulationBox, Topology};

    fn sample_topology() -> Topology {
        let mut top = Topology::new();
        top.box_ = SimulationBox {
            lx: 12.0,
            ly: 12.0,
            lz: 12.0,
            pbc: true,
        };
        for i in 0..3 {
            top.atoms.push(AtomRecord {
                element: "C".into(),
                name: format!("C{i}"),
                mass: 12.0,
                charge: 0.0,
                sigma: 0.34,
                epsilon: 0.36,
                position: [i as f64, 0.0, 0.0],
                residue_id: 0,
                molecule_id: 0,
                born_r: None,
            });
        }
        top
    }

    #[test]
    fn aqtrj_default_format_unchanged() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("rq_traj_fmt_aqtrj_{}.aqtrj", std::process::id()));
        let path_str = path.to_string_lossy().to_string();

        let top = sample_topology();
        let mut writer = TrajectoryWriter::new(&path_str);
        assert_eq!(writer.format(), TrajectoryFormat::Aqtrj);
        writer.write_frame(0, &top, -10.0, 1.0).unwrap();

        let mut top2 = Topology::new();
        top2.atoms = top.atoms.clone();
        TrajectoryReader::new(&path_str)
            .apply_last_frame(&mut top2)
            .unwrap();
        assert_eq!(top2.atoms[1].position, top.atoms[1].position);

        let _ = std::fs::remove_file(&path_str);
    }

    #[test]
    fn dcd_format_round_trips_through_writer() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("rq_traj_fmt_dcd_{}.dcd", std::process::id()));
        let path_str = path.to_string_lossy().to_string();

        let top = sample_topology();
        let mut writer = TrajectoryWriter::with_format(&path_str, TrajectoryFormat::Dcd).unwrap();
        writer.write_frame(0, &top, -1.0, 0.5).unwrap();
        writer.write_frame(1, &top, -1.1, 0.6).unwrap();

        let traj = read_dcd(&path_str).unwrap();
        assert_eq!(traj.natoms, 3);
        assert_eq!(traj.frames.len(), 2);
        assert_eq!(traj.frames[0].cell[0], 12.0);

        let _ = std::fs::remove_file(&path_str);
    }

    #[test]
    fn pdb_format_writes_multimodel() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("rq_traj_fmt_pdb_{}.pdb", std::process::id()));
        let path_str = path.to_string_lossy().to_string();

        let top = sample_topology();
        let mut writer = TrajectoryWriter::with_format(&path_str, TrajectoryFormat::Pdb).unwrap();
        writer.write_frame(0, &top, 0.0, 0.0).unwrap();
        writer.write_frame(1, &top, 0.0, 0.0).unwrap();

        let contents = std::fs::read_to_string(&path_str).unwrap();
        assert_eq!(contents.matches("MODEL").count(), 2);

        let _ = std::fs::remove_file(&path_str);
    }

    #[test]
    fn xyz_format_writes_frames() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("rq_traj_fmt_xyz_{}.xyz", std::process::id()));
        let path_str = path.to_string_lossy().to_string();

        let top = sample_topology();
        let mut writer = TrajectoryWriter::with_format(&path_str, TrajectoryFormat::Xyz).unwrap();
        writer.write_frame(0, &top, 0.0, 0.0).unwrap();

        let contents = std::fs::read_to_string(&path_str).unwrap();
        assert!(contents.starts_with("3\n"));

        let _ = std::fs::remove_file(&path_str);
    }

    #[test]
    fn parse_rejects_unknown_format() {
        assert!(TrajectoryFormat::parse("bogus").is_err());
        assert_eq!(TrajectoryFormat::parse("DCD").unwrap(), TrajectoryFormat::Dcd);
    }
}
