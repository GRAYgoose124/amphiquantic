//! Multi-model PDB trajectory writer: each frame is written as a
//! `MODEL` / `ATOM...` / `ENDMDL` block, appended to the same file.

use crate::topology::Topology;
use std::fs::File;
use std::io::{BufWriter, Write};

pub struct PdbTrajectoryWriter {
    path: String,
    model_count: u32,
}

impl PdbTrajectoryWriter {
    /// Truncates any existing file at `path`.
    pub fn new(path: impl Into<String>) -> Result<Self, String> {
        let path = path.into();
        File::create(&path).map_err(|e| e.to_string())?;
        Ok(Self {
            path,
            model_count: 0,
        })
    }

    pub fn write_frame(&mut self, topology: &Topology) -> Result<(), String> {
        self.model_count += 1;
        let file = File::options()
            .append(true)
            .open(&self.path)
            .map_err(|e| e.to_string())?;
        let mut w = BufWriter::new(file);

        writeln!(w, "MODEL     {:>4}", self.model_count).map_err(|e| e.to_string())?;
        if topology.box_.lx > 0.0 {
            writeln!(
                w,
                "CRYST1{:>9.3}{:>9.3}{:>9.3}{:>7.2}{:>7.2}{:>7.2} P 1           1",
                topology.box_.lx, topology.box_.ly, topology.box_.lz, 90.0, 90.0, 90.0
            )
            .map_err(|e| e.to_string())?;
        }
        for (i, atom) in topology.atoms.iter().enumerate() {
            let [x, y, z] = atom.position;
            let name = if atom.element.len() > 2 {
                &atom.element[..2]
            } else {
                atom.element.as_str()
            };
            writeln!(
                w,
                "ATOM  {:>5} {:<3} MOL     1    {:>8.3}{:>8.3}{:>8.3}  1.00  0.00          {:>2}",
                i + 1,
                name,
                x,
                y,
                z,
                name
            )
            .map_err(|e| e.to_string())?;
        }
        writeln!(w, "ENDMDL").map_err(|e| e.to_string())?;
        Ok(())
    }

    pub fn model_count(&self) -> u32 {
        self.model_count
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::topology::{AtomRecord, SimulationBox, Topology};

    #[test]
    fn writes_multiple_models() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("rq_pdb_traj_test_{}.pdb", std::process::id()));
        let path_str = path.to_string_lossy().to_string();

        let mut top = Topology::new();
        top.box_ = SimulationBox {
            lx: 20.0,
            ly: 20.0,
            lz: 20.0,
            pbc: true,
        };
        top.atoms.push(AtomRecord {
            element: "C".into(),
            name: "C1".into(),
            mass: 12.0,
            charge: 0.0,
            sigma: 0.34,
            epsilon: 0.36,
            position: [1.0, 2.0, 3.0],
            residue_id: 0,
            molecule_id: 0,
            born_r: None,
        });

        let mut writer = PdbTrajectoryWriter::new(&path_str).unwrap();
        writer.write_frame(&top).unwrap();
        writer.write_frame(&top).unwrap();
        assert_eq!(writer.model_count(), 2);

        let contents = std::fs::read_to_string(&path_str).unwrap();
        assert_eq!(contents.matches("MODEL").count(), 2);
        assert_eq!(contents.matches("ENDMDL").count(), 2);
        assert!(contents.contains("CRYST1"));

        let _ = std::fs::remove_file(&path_str);
    }
}
