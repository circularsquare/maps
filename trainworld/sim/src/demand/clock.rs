//! Milliseconds since an arbitrary start: `Instant` natively, `performance.now()` in wasm
//! (`std::time::Instant` panics on wasm32-unknown-unknown).

#[cfg(not(target_arch = "wasm32"))]
pub fn now_ms() -> f64 {
    use std::sync::OnceLock;
    use std::time::Instant;
    static T0: OnceLock<Instant> = OnceLock::new();
    T0.get_or_init(Instant::now).elapsed().as_secs_f64() * 1000.0
}

#[cfg(target_arch = "wasm32")]
mod js {
    use wasm_bindgen::prelude::*;
    #[wasm_bindgen]
    extern "C" {
        #[wasm_bindgen(js_namespace = performance)]
        pub fn now() -> f64;
    }
}

#[cfg(target_arch = "wasm32")]
pub fn now_ms() -> f64 {
    js::now()
}
