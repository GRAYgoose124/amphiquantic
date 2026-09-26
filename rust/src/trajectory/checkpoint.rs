//! Binary simulation checkpoint: positions, velocities, box, step and RNG
//! seed, with a round-trip reader. Format (little-endian):
//!
//! ```text
//! magic:     8 bytes  b"AQCKPT01"
//! step:      u64
//! seed:      u64
//! box lx,ly,lz: 3 x f64
//! box pbc:   u8 (0/1)
//! natoms:    u64
//! positions: natoms x 3 x f64
//! velocities: natoms x 3 x f64
//! ```

use crate::topology::SimulationBox;
use std::fs::File;
use std::io::{Read, Write};

const MAGIC: &[u8; 8] = b"AQCKPT01";

#[derive(Debug, Clone, PartialEq)]
pub struct Checkpoint {
    pub step: u64,
    pub seed: u64,
    pub box_: SimulationBox,
    pub positions: Vec<[f64; 3]>,
    pub velocities: Vec<[f64; 3]>,
}

pub fn write_checkpoint(path: &str, checkpoint: &Checkpoint) -> Result<(), String> {
    if checkpoint.positions.len() != checkpoint.velocities.len() {
        return Err(format!(
            "checkpoint positions ({}) and velocities ({}) length mismatch",
            checkpoint.positions.len(),
            checkpoint.velocities.len()
        ));
    }
    let mut file = File::create(path).map_err(|e| e.to_string())?;
    file.write_all(MAGIC).map_err(|e| e.to_string())?;
    file.write_all(&checkpoint.step.to_le_bytes())
        .map_err(|e| e.to_string())?;
    file.write_all(&checkpoint.seed.to_le_bytes())
        .map_err(|e| e.to_string())?;
    file.write_all(&checkpoint.box_.lx.to_le_bytes())
        .map_err(|e| e.to_string())?;
    file.write_all(&checkpoint.box_.ly.to_le_bytes())
        .map_err(|e| e.to_string())?;
    file.write_all(&checkpoint.box_.lz.to_le_bytes())
        .map_err(|e| e.to_string())?;
    file.write_all(&[if checkpoint.box_.pbc { 1u8 } else { 0u8 }])
        .map_err(|e| e.to_string())?;
    let natoms = checkpoint.positions.len() as u64;
    file.write_all(&natoms.to_le_bytes())
        .map_err(|e| e.to_string())?;
    for p in &checkpoint.positions {
        for v in p {
            file.write_all(&v.to_le_bytes()).map_err(|e| e.to_string())?;
        }
    }
    for v in &checkpoint.velocities {
        for c in v {
            file.write_all(&c.to_le_bytes()).map_err(|e| e.to_string())?;
        }
    }
    Ok(())
}

pub fn read_checkpoint(path: &str) -> Result<Checkpoint, String> {
    let mut file = File::open(path).map_err(|e| e.to_string())?;
    let mut magic = [0u8; 8];
    file.read_exact(&mut magic).map_err(|e| e.to_string())?;
    if &magic != MAGIC {
        return Err("not an amphiquantic checkpoint file".to_string());
    }

    let mut read_u64 = || -> Result<u64, String> {
        let mut buf = [0u8; 8];
        file.read_exact(&mut buf).map_err(|e| e.to_string())?;
        Ok(u64::from_le_bytes(buf))
    };
    let step = read_u64()?;
    let seed = read_u64()?;

    let mut read_f64 = |file: &mut File| -> Result<f64, String> {
        let mut buf = [0u8; 8];
        file.read_exact(&mut buf).map_err(|e| e.to_string())?;
        Ok(f64::from_le_bytes(buf))
    };
    let lx = read_f64(&mut file)?;
    let ly = read_f64(&mut file)?;
    let lz = read_f64(&mut file)?;
    let mut pbc_byte = [0u8; 1];
    file.read_exact(&mut pbc_byte).map_err(|e| e.to_string())?;
    let pbc = pbc_byte[0] != 0;

    let mut natoms_buf = [0u8; 8];
    file.read_exact(&mut natoms_buf)
        .map_err(|e| e.to_string())?;
    let natoms = u64::from_le_bytes(natoms_buf) as usize;

    let read_vec3 = |file: &mut File| -> Result<[f64; 3], String> {
        let mut out = [0.0; 3];
        for slot in out.iter_mut() {
            let mut buf = [0u8; 8];
            file.read_exact(&mut buf).map_err(|e| e.to_string())?;
            *slot = f64::from_le_bytes(buf);
        }
        Ok(out)
    };

    let mut positions = Vec::with_capacity(natoms);
    for _ in 0..natoms {
        positions.push(read_vec3(&mut file)?);
    }
    let mut velocities = Vec::with_capacity(natoms);
    for _ in 0..natoms {
        velocities.push(read_vec3(&mut file)?);
    }

    Ok(Checkpoint {
        step,
        seed,
        box_: SimulationBox { lx, ly, lz, pbc },
        positions,
        velocities,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("rq_ckpt_test_{}.bin", std::process::id()));
        let path_str = path.to_string_lossy().to_string();

        let ckpt = Checkpoint {
            step: 12345,
            seed: 987654321,
            box_: SimulationBox {
                lx: 30.0,
                ly: 30.0,
                lz: 30.0,
                pbc: true,
            },
            positions: vec![[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [-4.5, 5.5, -6.5]],
            velocities: vec![[0.1, 0.2, 0.3], [0.0, 0.0, 0.0], [1.0, -1.0, 0.5]],
        };

        write_checkpoint(&path_str, &ckpt).unwrap();
        let loaded = read_checkpoint(&path_str).unwrap();
        assert_eq!(loaded, ckpt);

        let _ = std::fs::remove_file(&path_str);
    }

    #[test]
    fn mismatched_lengths_rejected() {
        let ckpt = Checkpoint {
            step: 0,
            seed: 0,
            box_: SimulationBox::default(),
            positions: vec![[0.0, 0.0, 0.0]],
            velocities: vec![],
        };
        assert!(write_checkpoint("/tmp/should_not_be_created.bin", &ckpt).is_err());
    }
}
