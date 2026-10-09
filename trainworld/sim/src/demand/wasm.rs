//! wasm entries outside `DemandApi` (api.rs): the T-005 benchmark (feature `demand-bench`; the
//! app's build leaves it out) and the linear memory size.

#[cfg(feature = "demand-bench")]
use super::{bench, clock::now_ms, network, pack, synth};
use wasm_bindgen::prelude::*;

/// Run the benchmark. Empty `header` = synthetic city. `opts` like "exact,sweep,rounds=1,zone=2000".
#[cfg(feature = "demand-bench")]
#[wasm_bindgen]
pub fn demand_bench(header: &str, bin: &[u8], opts: &str) -> String {
    let mut o = bench::Opts::default();
    for kv in opts.split(',').filter(|s| !s.is_empty()) {
        let mut it = kv.splitn(2, '=');
        match (it.next().unwrap(), it.next()) {
            ("exact", _) => o.exact = true,
            ("solve", _) => o.solve_gravity = true,
            ("sweep", _) => {
                o.sweep = true;
                o.exact = true;
            }
            ("noedit", _) => o.edit = false,
            ("noperiods", _) => o.periods = false,
            ("zone", Some(v)) => o.zone_m = v.parse().unwrap_or(o.zone_m),
            ("band", Some(v)) => o.band_min = v.parse().unwrap_or(o.band_min),
            ("rounds", Some(v)) => o.crowd_rounds = v.parse().unwrap_or(1),
            ("theta", Some(v)) => o.theta = v.parse().ok(),
            ("walk", Some(v)) => o.walk_cutoff_m = v.parse().ok(),
            _ => {}
        }
    }
    let t0 = now_ms();
    let city = if header.is_empty() {
        synth::synth_city(5)
    } else {
        match pack::parse(header, bin) {
            Ok(c) => c,
            Err(e) => return format!("pack error: {e}"),
        }
    };
    let mut out = format!("load/generate {:.0} ms\n", now_ms() - t0);
    let net = network::synth_network();
    out.push_str(&bench::run(&city, &net, &o));
    out.push_str(&format!("wasm linear memory now {:.1} MB\n", wasm_memory_bytes() as f64 / 1048576.0));
    out
}

/// Current size of the wasm linear memory (it never shrinks, so this is also its peak).
#[wasm_bindgen]
pub fn wasm_memory_bytes() -> f64 {
    (core::arch::wasm32::memory_size(0) * 65536) as f64
}
