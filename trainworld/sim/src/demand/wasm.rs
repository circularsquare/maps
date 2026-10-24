//! The wasm entry outside `DemandApi` (api.rs): the linear memory size.

use wasm_bindgen::prelude::*;

/// Current size of the wasm linear memory (it never shrinks, so this is also its peak).
#[wasm_bindgen]
pub fn wasm_memory_bytes() -> f64 {
    (core::arch::wasm32::memory_size(0) * 65536) as f64
}
