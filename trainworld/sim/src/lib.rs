//! Simulation core, compiled to WebAssembly and run in a Web Worker.
//!
//! Skeleton for T-001: proves the Rust -> WASM -> worker -> main thread path. Data is kept as
//! flat arrays per field (structure of arrays), which is the layout the real core will use.

use wasm_bindgen::prelude::*;

/// T-005 demand kernel spike. Behind the `demand` feature so the app's build does not carry it.
#[cfg(feature = "demand")]
pub mod demand;

/// T-040 track model: geometry, validity, costs, edits, run-time profiles, capacity. The clock
/// worker's API is `track::wasm::TrackApi`.
pub mod track;

#[wasm_bindgen]
pub struct Sim {
    /// Per line: length in metres.
    line_len_m: Vec<f32>,
    /// Per train: which line it runs on.
    train_line: Vec<u32>,
    /// Per train: departure time from the line's start, in seconds.
    train_depart_s: Vec<f32>,
    speed_mps: f32,
}

#[wasm_bindgen]
impl Sim {
    #[wasm_bindgen(constructor)]
    pub fn new(n_lines: u32, trains_per_line: u32) -> Sim {
        let speed_mps = 15.0;
        let line_len_m: Vec<f32> = (0..n_lines).map(|i| 5_000.0 + (i % 40) as f32 * 500.0).collect();
        let mut train_line = Vec::with_capacity((n_lines * trains_per_line) as usize);
        let mut train_depart_s = Vec::with_capacity(train_line.capacity());
        for (line, &len) in line_len_m.iter().enumerate() {
            let headway_s = len / speed_mps / trains_per_line as f32;
            for k in 0..trains_per_line {
                train_line.push(line as u32);
                train_depart_s.push(k as f32 * headway_s);
            }
        }
        Sim { line_len_m, train_line, train_depart_s, speed_mps }
    }

    pub fn train_count(&self) -> u32 {
        self.train_line.len() as u32
    }

    /// Distance along its line for every train at time `t` (seconds). Closed form: nothing is
    /// stepped, so any time can be asked for and trains nobody looks at cost nothing.
    pub fn offsets(&self, t: f64, out: &mut [f32]) {
        for i in 0..self.train_line.len() {
            let len = self.line_len_m[self.train_line[i] as usize];
            let run = ((t as f32 - self.train_depart_s[i]) * self.speed_mps).rem_euclid(len);
            out[i] = run;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn offsets_stay_on_the_line() {
        let sim = Sim::new(3, 4);
        let mut out = vec![0.0; sim.train_count() as usize];
        sim.offsets(12_345.6, &mut out);
        for (i, &o) in out.iter().enumerate() {
            let len = sim.line_len_m[sim.train_line[i] as usize];
            assert!(o >= 0.0 && o < len);
        }
    }
}
