//! T-173: a small everyday opening for the browser to optimise its code on, in one demand worker
//! at startup, before the real cities open in every worker at once.
//!
//! A browser runs a WebAssembly function in its baseline compiler until the function has run a
//! while, then swaps in optimised code for later calls. With five workers opening New York's
//! everyday layer at once from baseline code, the optimised code came seconds late (7-8 s in
//! every worker instead of 1.5-2 s; one worker alone took 1.6 s). Compiled code is shared by the
//! workers of a page, so one worker opening this small city first leaves the code optimised for
//! all of them.

use super::{everyday::EverydayApi, pack};
use wasm_bindgen::prelude::*;

/// A synthetic city pack with the everyday arrays: a `side` x `side` grid of cells 400 m apart
/// (2 km zones: `side` 100 is 400 zones, 10,000 cells), people and jobs varying cell to cell.
fn pack(side: u32) -> (String, Vec<u8>) {
    let n = (side * side) as usize;
    let lump = |i: usize, k: usize| 50.0 + ((i * k) % 997) as f32;
    let city = pack::City {
        name: "warm-up".into(),
        h3: (1..=n as u64).collect(),
        x: (0..n).map(|i| (i % side as usize) as f32 * 400.0).collect(),
        y: (0..n).map(|i| (i / side as usize) as f32 * 400.0).collect(),
        pop: (0..n).map(|i| lump(i, 7919)).collect(),
        jobs: (0..n).map(|i| lump(i, 104729)).collect(),
        region: vec![],
        gravity: None,
    };
    let (header, mut bin) = pack::write(&city);
    let mut h: serde_json::Value = serde_json::from_str(&header).expect("pack::write's header");
    for (name, k) in [("errand_jobs", 15485863), ("leisure_jobs", 32452843)] {
        let offset = bin.len();
        bin.extend((0..n).flat_map(|i| lump(i, k).to_le_bytes()));
        h["arrays"].as_array_mut().expect("arrays").push(serde_json::json!({"name": name, "dtype": "f32", "per": "cell", "count": n, "offset": offset}));
    }
    (h.to_string(), bin)
}

/// Open the everyday layer of `pack(side)` and forget it. Returns the ms it took.
#[wasm_bindgen]
pub fn everyday_warm_up(side: u32) -> f64 {
    let t0 = super::clock::now_ms();
    let (header, bin) = pack(side);
    let _ = EverydayApi::new(&header, &bin).map(|a| a.info());
    super::clock::now_ms() - t0
}

#[cfg(test)]
mod tests {
    #[test]
    fn everyday_warm_up_pack_opens() {
        // an error would make the warm-up do nothing, silently
        let (header, bin) = super::pack(40);
        let info = super::EverydayApi::new(&header, &bin).unwrap().info();
        assert!(info[0] > 0.0 && info[3] > 0.0, "{info:?}");
        assert!(super::everyday_warm_up(20) >= 0.0);
    }
}
