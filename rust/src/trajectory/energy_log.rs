//! CSV energy log writer: step,time_ps,potential,kinetic,total,temperature
//! with optional pressure/volume columns when those values are supplied.

use std::fs::File;
use std::io::{BufWriter, Write};

pub struct EnergyLogWriter {
    path: String,
    header_written: bool,
    include_pressure_volume: bool,
}

impl EnergyLogWriter {
    /// Truncates any existing file at `path`. If `include_pressure_volume`
    /// is true, every row must be written via [`write_row_full`], otherwise
    /// use [`write_row`].
    pub fn new(path: impl Into<String>, include_pressure_volume: bool) -> Result<Self, String> {
        let path = path.into();
        File::create(&path).map_err(|e| e.to_string())?;
        Ok(Self {
            path,
            header_written: false,
            include_pressure_volume,
        })
    }

    fn ensure_header(&mut self) -> Result<(), String> {
        if self.header_written {
            return Ok(());
        }
        let file = File::options()
            .append(true)
            .open(&self.path)
            .map_err(|e| e.to_string())?;
        let mut w = BufWriter::new(file);
        if self.include_pressure_volume {
            writeln!(w, "step,time_ps,potential,kinetic,total,temperature,pressure,volume")
                .map_err(|e| e.to_string())?;
        } else {
            writeln!(w, "step,time_ps,potential,kinetic,total,temperature")
                .map_err(|e| e.to_string())?;
        }
        self.header_written = true;
        Ok(())
    }

    pub fn write_row(
        &mut self,
        step: u64,
        time_ps: f64,
        potential: f64,
        kinetic: f64,
        temperature: f64,
    ) -> Result<(), String> {
        self.ensure_header()?;
        let file = File::options()
            .append(true)
            .open(&self.path)
            .map_err(|e| e.to_string())?;
        let mut w = BufWriter::new(file);
        let total = potential + kinetic;
        if self.include_pressure_volume {
            writeln!(
                w,
                "{step},{time_ps:.6},{potential:.8},{kinetic:.8},{total:.8},{temperature:.4},,"
            )
            .map_err(|e| e.to_string())?;
        } else {
            writeln!(
                w,
                "{step},{time_ps:.6},{potential:.8},{kinetic:.8},{total:.8},{temperature:.4}"
            )
            .map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    pub fn write_row_full(
        &mut self,
        step: u64,
        time_ps: f64,
        potential: f64,
        kinetic: f64,
        temperature: f64,
        pressure: f64,
        volume: f64,
    ) -> Result<(), String> {
        self.ensure_header()?;
        let file = File::options()
            .append(true)
            .open(&self.path)
            .map_err(|e| e.to_string())?;
        let mut w = BufWriter::new(file);
        let total = potential + kinetic;
        writeln!(
            w,
            "{step},{time_ps:.6},{potential:.8},{kinetic:.8},{total:.8},{temperature:.4},{pressure:.6},{volume:.6}"
        )
        .map_err(|e| e.to_string())?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_columns() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("rq_energy_test_{}.csv", std::process::id()));
        let path_str = path.to_string_lossy().to_string();

        let mut w = EnergyLogWriter::new(&path_str, false).unwrap();
        w.write_row(0, 0.0, -100.0, 10.0, 300.0).unwrap();
        w.write_row(1, 0.002, -99.5, 10.2, 301.0).unwrap();

        let contents = std::fs::read_to_string(&path_str).unwrap();
        let lines: Vec<&str> = contents.lines().collect();
        assert_eq!(lines[0], "step,time_ps,potential,kinetic,total,temperature");
        assert_eq!(lines.len(), 3);
        assert!(lines[1].starts_with("0,0.000000,-100.00000000,10.00000000,-90.00000000,300.0000"));

        let _ = std::fs::remove_file(&path_str);
    }

    #[test]
    fn pressure_volume_columns() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("rq_energy_pv_test_{}.csv", std::process::id()));
        let path_str = path.to_string_lossy().to_string();

        let mut w = EnergyLogWriter::new(&path_str, true).unwrap();
        w.write_row_full(0, 0.0, -100.0, 10.0, 300.0, 1.0, 1000.0)
            .unwrap();

        let contents = std::fs::read_to_string(&path_str).unwrap();
        let lines: Vec<&str> = contents.lines().collect();
        assert_eq!(
            lines[0],
            "step,time_ps,potential,kinetic,total,temperature,pressure,volume"
        );
        assert!(lines[1].ends_with("1.000000,1000.000000"));

        let _ = std::fs::remove_file(&path_str);
    }
}
