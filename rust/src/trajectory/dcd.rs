//! Binary DCD trajectory writer/reader (CHARMM/NAMD "CORD" format), with
//! unit cell support. Layout follows the classic Fortran-unformatted DCD
//! convention used by CHARMM, NAMD, VMD, MDAnalysis and mdtraj:
//!
//! ```text
//! record 1: "CORD" + 20 x int32 control block (icntrl)
//! record 2: ntitle (int32) + ntitle * 80-byte title strings
//! record 3: natoms (int32)
//! per frame:
//!   [record: 6 x float64 unit cell (A, gamma, B, beta, alpha, C)]  (if icntrl[10] != 0)
//!   record: natoms x float32 X
//!   record: natoms x float32 Y
//!   record: natoms x float32 Z
//! ```
//!
//! Each record is wrapped in 4-byte little-endian record-length markers
//! (the Fortran unformatted convention), one before and one after the
//! payload.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};

const NSET_OFFSET: u64 = 8; // byte offset of icntrl[0] (NSET) within the file

fn write_record(w: &mut impl Write, payload: &[u8]) -> Result<(), String> {
    let len = payload.len() as u32;
    w.write_all(&len.to_le_bytes()).map_err(|e| e.to_string())?;
    w.write_all(payload).map_err(|e| e.to_string())?;
    w.write_all(&len.to_le_bytes()).map_err(|e| e.to_string())?;
    Ok(())
}

fn read_record(r: &mut impl Read) -> Result<Vec<u8>, String> {
    let mut len_buf = [0u8; 4];
    r.read_exact(&mut len_buf).map_err(|e| e.to_string())?;
    let len = u32::from_le_bytes(len_buf) as usize;
    let mut payload = vec![0u8; len];
    r.read_exact(&mut payload).map_err(|e| e.to_string())?;
    let mut end_buf = [0u8; 4];
    r.read_exact(&mut end_buf).map_err(|e| e.to_string())?;
    let end_len = u32::from_le_bytes(end_buf) as usize;
    if end_len != len {
        return Err(format!(
            "DCD record length mismatch: start={len} end={end_len}"
        ));
    }
    Ok(payload)
}

pub struct DcdWriter {
    file: File,
    natoms: usize,
    nframes: u32,
}

impl DcdWriter {
    /// Create (truncating) a new DCD file and write its header. `natoms`
    /// must match every frame written afterwards. `timestep_ps` is stored
    /// informationally in the header (AKMA-ish float32 slot); readers that
    /// don't care about exact units can ignore it.
    pub fn create(path: &str, natoms: usize, timestep_ps: f32) -> Result<Self, String> {
        let mut file = File::create(path).map_err(|e| e.to_string())?;

        let mut icntrl = [0i32; 20];
        icntrl[0] = 0; // NSET, patched as frames are written
        icntrl[1] = 0; // ISTART
        icntrl[2] = 1; // NSAVC
        icntrl[7] = 0; // NDEGF
        icntrl[8] = 0; // NFREAT
        icntrl[9] = timestep_ps.to_bits() as i32; // DELTA (float32 bit pattern)
        icntrl[10] = 1; // WITH_UNIT_CELL
        icntrl[19] = 24; // CHARMM version

        let mut header_payload = Vec::with_capacity(84);
        header_payload.extend_from_slice(b"CORD");
        for v in icntrl.iter() {
            header_payload.extend_from_slice(&v.to_le_bytes());
        }
        write_record(&mut file, &header_payload)?;

        let title = format!("{:<80}", "Written by rustquantic");
        let mut title_payload = Vec::with_capacity(4 + 80);
        title_payload.extend_from_slice(&1i32.to_le_bytes()); // NTITLE
        title_payload.extend_from_slice(title.as_bytes());
        write_record(&mut file, &title_payload)?;

        let natoms_payload = (natoms as i32).to_le_bytes();
        write_record(&mut file, &natoms_payload)?;

        Ok(Self {
            file,
            natoms,
            nframes: 0,
        })
    }

    /// Write one frame. `box_lengths` is (a, b, c) in the same length units
    /// as the coordinates; angles are assumed 90 degrees (orthorhombic box).
    pub fn write_frame(
        &mut self,
        positions: &[[f64; 3]],
        box_lengths: (f64, f64, f64),
    ) -> Result<(), String> {
        if positions.len() != self.natoms {
            return Err(format!(
                "DCD frame has {} atoms, writer expects {}",
                positions.len(),
                self.natoms
            ));
        }

        // Unit cell record: A, GAMMA, B, BETA, ALPHA, C (CHARMM convention).
        let (a, b, c) = box_lengths;
        let cell = [a, 90.0_f64, b, 90.0_f64, 90.0_f64, c];
        let mut cell_payload = Vec::with_capacity(48);
        for v in cell.iter() {
            cell_payload.extend_from_slice(&v.to_le_bytes());
        }
        write_record(&mut self.file, &cell_payload)?;

        for axis in 0..3 {
            let mut payload = Vec::with_capacity(self.natoms * 4);
            for p in positions {
                payload.extend_from_slice(&(p[axis] as f32).to_le_bytes());
            }
            write_record(&mut self.file, &payload)?;
        }

        self.nframes += 1;
        let pos = self.file.stream_position().map_err(|e| e.to_string())?;
        self.file
            .seek(SeekFrom::Start(NSET_OFFSET))
            .map_err(|e| e.to_string())?;
        self.file
            .write_all(&(self.nframes as i32).to_le_bytes())
            .map_err(|e| e.to_string())?;
        self.file
            .seek(SeekFrom::Start(pos))
            .map_err(|e| e.to_string())?;
        Ok(())
    }

    pub fn nframes(&self) -> u32 {
        self.nframes
    }
}

pub struct DcdFrame {
    pub positions: Vec<[f32; 3]>,
    /// (A, GAMMA, B, BETA, ALPHA, C)
    pub cell: [f64; 6],
}

pub struct DcdTrajectory {
    pub natoms: usize,
    pub frames: Vec<DcdFrame>,
}

pub fn read_dcd(path: &str) -> Result<DcdTrajectory, String> {
    let mut file = File::open(path).map_err(|e| e.to_string())?;

    let header_payload = read_record(&mut file)?;
    if header_payload.len() < 84 || &header_payload[0..4] != b"CORD" {
        return Err("not a CORD-format DCD file".to_string());
    }
    let mut icntrl = [0i32; 20];
    for (i, chunk) in header_payload[4..84].chunks_exact(4).enumerate() {
        icntrl[i] = i32::from_le_bytes(chunk.try_into().unwrap());
    }
    let with_unit_cell = icntrl[10] != 0;

    let title_payload = read_record(&mut file)?;
    let ntitle = i32::from_le_bytes(title_payload[0..4].try_into().unwrap()).max(0) as usize;
    let _ = ntitle; // titles themselves are not needed downstream

    let natoms_payload = read_record(&mut file)?;
    let natoms = i32::from_le_bytes(natoms_payload[0..4].try_into().unwrap()).max(0) as usize;

    let mut frames = Vec::new();
    loop {
        let mut cell = [0.0f64; 6];
        if with_unit_cell {
            match read_record(&mut file) {
                Ok(cell_payload) => {
                    for (i, chunk) in cell_payload.chunks_exact(8).enumerate().take(6) {
                        cell[i] = f64::from_le_bytes(chunk.try_into().unwrap());
                    }
                }
                Err(_) => break, // clean EOF between frames
            }
        }

        let x_payload = match read_record(&mut file) {
            Ok(p) => p,
            Err(_) => break,
        };
        let y_payload = read_record(&mut file)?;
        let z_payload = read_record(&mut file)?;

        let read_f32s = |payload: &[u8]| -> Vec<f32> {
            payload
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect()
        };
        let xs = read_f32s(&x_payload);
        let ys = read_f32s(&y_payload);
        let zs = read_f32s(&z_payload);
        let mut positions = Vec::with_capacity(natoms);
        for i in 0..natoms.min(xs.len()).min(ys.len()).min(zs.len()) {
            positions.push([xs[i], ys[i], zs[i]]);
        }
        frames.push(DcdFrame { positions, cell });
    }

    Ok(DcdTrajectory { natoms, frames })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_single_frame() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("rq_dcd_test_{}.dcd", std::process::id()));
        let path_str = path.to_string_lossy().to_string();

        let positions = vec![[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [-1.5, 0.5, 9.25]];
        {
            let mut w = DcdWriter::create(&path_str, positions.len(), 0.002).unwrap();
            w.write_frame(&positions, (10.0, 10.0, 10.0)).unwrap();
        }

        let traj = read_dcd(&path_str).unwrap();
        assert_eq!(traj.natoms, 3);
        assert_eq!(traj.frames.len(), 1);
        let frame = &traj.frames[0];
        assert_eq!(frame.cell[0], 10.0);
        assert_eq!(frame.cell[2], 10.0);
        assert_eq!(frame.cell[5], 10.0);
        for (got, want) in frame.positions.iter().zip(positions.iter()) {
            for k in 0..3 {
                assert!((got[k] as f64 - want[k]).abs() < 1e-4);
            }
        }
        let _ = std::fs::remove_file(&path_str);
    }

    #[test]
    fn round_trip_multi_frame() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("rq_dcd_multi_{}.dcd", std::process::id()));
        let path_str = path.to_string_lossy().to_string();

        let natoms = 5;
        {
            let mut w = DcdWriter::create(&path_str, natoms, 0.001).unwrap();
            for f in 0..4 {
                let positions: Vec<[f64; 3]> = (0..natoms)
                    .map(|i| [i as f64 + f as f64, 0.0, 0.0])
                    .collect();
                w.write_frame(&positions, (20.0, 20.0, 20.0)).unwrap();
            }
            assert_eq!(w.nframes(), 4);
        }

        let traj = read_dcd(&path_str).unwrap();
        assert_eq!(traj.natoms, natoms);
        assert_eq!(traj.frames.len(), 4);
        for (f, frame) in traj.frames.iter().enumerate() {
            for (i, p) in frame.positions.iter().enumerate() {
                assert!((p[0] as f64 - (i as f64 + f as f64)).abs() < 1e-4);
            }
        }
        let _ = std::fs::remove_file(&path_str);
    }
}
