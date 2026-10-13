//! T-276: each zone pack's local trip legs a day with no rail (commutes kept in the pack and every
//! everyday purpose, both legs), for the far-zoomed per-zone bubble of a zone not opened in the
//! game (docs/zones.md 6). pipeline/zone_summary.py writes them into the registry.
//!
//!   cargo run --release --features demand --bin zone_summary -- data/packs/nyc.json [more packs]
//!
//! Prints one JSON object: {"<pack file stem>": legs, ...}.

use trainworld_sim::demand::{api::DemandApi, everyday::EverydayApi, pack};

fn main() {
    let mut out = serde_json::Map::new();
    for path in std::env::args().skip(1) {
        let (h, b) = pack::load_raw(&path).unwrap_or_else(|e| panic!("{e}"));
        let commute = DemandApi::new(&h, &b).unwrap_or_else(|e| panic!("{path}: {e}")).info()[2];
        let outings: f64 = if h.contains("\"errand_jobs\"") {
            let e = EverydayApi::new(&h, &b).unwrap_or_else(|e| panic!("{path}: {e}")).info();
            e.chunks_exact(3).map(|v| v[0]).sum()
        } else {
            0.0
        };
        let stem = std::path::Path::new(&path).file_stem().unwrap().to_string_lossy().to_string();
        eprintln!("{stem}: {commute:.0} commuters in the pack, {outings:.0} outings");
        out.insert(stem, serde_json::json!((2.0 * (commute + outings)).round()));
    }
    println!("{}", serde_json::Value::Object(out));
}
