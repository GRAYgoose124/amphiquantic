//! Plain-text XYZ trajectory writer (one frame appended per call).

use crate::topology::Topology;
use std::fs::File;
use std::io::{BufWriter, Write};

pub struct XyzWriter {
    path: String,
}

impl XyzWriter {
    /// Truncates any existing file at `path`.
    pub fn new(path: impl Into<String>) -> Result<Self, String> {
        let path = path.into();
        File::create(&path).map_err(|e| e.to_string())?;
        Ok(Self { path })
    }

    pub fn write_frame(&mut self, step: u64, topology: &Topology) -> Result<(), String> {
        let file = File::options()
            .append(true)
            .open(&self.path)
            .map_err(|e| e.to_string())?;
        let mut w = BufWriter::new(file);
        writeln!(w, "{}", topology.atoms.len()).map_err(|e| e.to_string())?;
        writeln!(w, "step={step}").map_err(|e| e.to_string())?;
        for atom in &topology.atoms {
            writeln!(
                w,
                "{:<3} {:>12.6} {:>12.6} {:>12.6}",
                atom.element, atom.position[0], atom.position[1], atom.position[2]
            )
            .map_err(|e| e.to_string())?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::topology::{AtomRecord, Topology};

    #[test]
    fn writes_multiple_frames() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("rq_xyz_test_{}.xyz", std::process::id()));
        let path_str = path.to_string_lossy().to_string();

        let mut top = Topology::new();
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

        let mut writer = XyzWriter::new(&path_str).unwrap();
        writer.write_frame(0, &top).unwrap();
        writer.write_frame(1, &top).unwrap();

        let contents = std::fs::read_to_string(&path_str).unwrap();
        let lines: Vec<&str> = contents.lines().collect();
        assert_eq!(lines.len(), 6); // 2 frames x 3 lines each
        assert_eq!(lines[0], "1");
        assert!(lines[1].starts_with("step=0"));
        assert!(lines[3] == "1");
        assert!(lines[4].starts_with("step=1"));

        let _ = std::fs::remove_file(&path_str);
    }
}
