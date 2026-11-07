//! T-2690: each start's trips inside circles round its centre, both ends inside, from every pack
//! the circles reach (pipeline/start_difficulty.py runs it and adds the packs up).
//!
//!   cargo run --release --features demand --bin start_difficulty -- STARTS.json PACK.json [more packs]
//!
//! STARTS.json: {"radii_km": [..], "starts": [{"id", "lon", "lat"}, ..]}. Prints one JSON line per
//! pack: {"pack", "starts": {"<id>": [per radius: commute legs, of them walked, outing legs, of them
//! walked, people, jobs]}}, or {"pack", "error"}. A weekday with no rail; trips between packs and
//! attraction visits are not counted.

use trainworld_sim::demand::{api::DemandApi, everyday::EverydayApi, pack};

const R_EARTH: f64 = 6371008.8;
const WEEKDAY: u32 = 0;

struct Start {
    id: String,
    lon: f64,
    lat: f64,
}

fn cell_f32(hv: &serde_json::Value, b: &[u8], name: &str, n: usize) -> Result<Vec<f32>, String> {
    let a = hv["arrays"].as_array().and_then(|v| v.iter().find(|a| a["name"] == name)).ok_or(format!("no {name}"))?;
    let off = a["offset"].as_u64().ok_or("bad offset")? as usize;
    if a["dtype"] != "f32" || a["count"].as_u64() != Some(n as u64) || off + 4 * n > b.len() {
        return Err(format!("{name}: not {n} f32"));
    }
    Ok(b[off..off + 4 * n].chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect())
}

fn one_pack(path: &str, starts: &[Start], radii: &[f64]) -> Result<serde_json::Map<String, serde_json::Value>, String> {
    let (h, b) = pack::load_raw(path)?;
    let hv: serde_json::Value = serde_json::from_str(&h).map_err(|e| e.to_string())?;
    let (olon, olat) = (hv["origin"]["lon"].as_f64().ok_or("no origin")?, hv["origin"]["lat"].as_f64().ok_or("no origin")?);
    let d = DemandApi::new(&h, &b)?;
    let xy = d.cell_xy();
    let n = xy.len() / 2;
    let pop = cell_f32(&hv, &b, "pop", n)?;
    let jobs = cell_f32(&hv, &b, "jobs", n)?;
    let (mut x0, mut y0, mut x1, mut y1) = (f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
    for c in 0..n {
        let (x, y) = (xy[2 * c] as f64, xy[2 * c + 1] as f64);
        x0 = x0.min(x);
        x1 = x1.max(x);
        y0 = y0.min(y);
        y1 = y1.max(y);
    }
    let rmax = radii.iter().cloned().fold(0.0, f64::max) * 1000.0;
    let k = olat.to_radians().cos();
    let mut out = serde_json::Map::new();
    let mut everyday: Option<Option<EverydayApi>> = None;
    for s in starts {
        let cx = R_EARTH * (s.lon - olon).to_radians() * k;
        let cy = R_EARTH * (s.lat - olat).to_radians();
        let gap = ((x0 - cx).max(cx - x1).max(0.0)).hypot((y0 - cy).max(cy - y1).max(0.0));
        if gap > rmax {
            continue;
        }
        let e = everyday.get_or_insert_with(|| if h.contains("\"errand_jobs\"") { EverydayApi::beside(&d, &h, &b).ok() } else { None });
        let mut per = vec![];
        for &r in radii {
            let r2 = (r * 1000.0).powi(2);
            let inside: Vec<bool> = (0..n).map(|c| (xy[2 * c] as f64 - cx).powi(2) + (xy[2 * c + 1] as f64 - cy).powi(2) <= r2).collect();
            let cells: Vec<u32> = (0..n as u32).filter(|&c| inside[c as usize]).collect();
            if cells.is_empty() {
                per.push(serde_json::json!([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]));
                continue;
            }
            // home end: the circle's homes, their far cells the job ends (rail 0 with no network)
            let f = d.flows_day(0, &cells, &[], WEEKDAY);
            let (mut commute, mut commute_walk) = (0f64, 0f64);
            if f.len() > 7 {
                let m = f[6] as usize;
                for i in 0..m {
                    let c = f[7 + i] as usize;
                    if c < n && inside[c] {
                        let (wk, dr) = (f[7 + 2 * m + i] as f64, f[7 + 3 * m + i] as f64);
                        commute += 2.0 * (wk + dr);
                        commute_walk += 2.0 * wk;
                    }
                }
            }
            let (outing, outing_walk) = e.as_ref().map_or((0.0, 0.0), |e| e.legs_within(&inside, WEEKDAY));
            let p: f64 = cells.iter().map(|&c| pop[c as usize] as f64).sum();
            let j: f64 = cells.iter().map(|&c| jobs[c as usize] as f64).sum();
            per.push(serde_json::json!([commute.round(), commute_walk.round(), outing.round(), outing_walk.round(), p.round(), j.round()]));
        }
        out.insert(s.id.clone(), serde_json::Value::Array(per));
    }
    Ok(out)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (spec, packs) = args.split_first().expect("usage: start_difficulty STARTS.json PACK.json [more]");
    let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(spec).unwrap_or_else(|e| panic!("{spec}: {e}"))).unwrap();
    let radii: Vec<f64> = v["radii_km"].as_array().unwrap().iter().map(|r| r.as_f64().unwrap()).collect();
    let starts: Vec<Start> = v["starts"]
        .as_array()
        .unwrap()
        .iter()
        .map(|s| Start { id: s["id"].as_str().unwrap().to_string(), lon: s["lon"].as_f64().unwrap(), lat: s["lat"].as_f64().unwrap() })
        .collect();
    for path in packs {
        let stem = std::path::Path::new(path).file_stem().unwrap().to_string_lossy().to_string();
        let line = match std::panic::catch_unwind(|| one_pack(path, &starts, &radii)) {
            Ok(Ok(m)) => serde_json::json!({"pack": stem, "starts": m}),
            Ok(Err(e)) => serde_json::json!({"pack": stem, "error": e}),
            Err(_) => serde_json::json!({"pack": stem, "error": "panic"}),
        };
        println!("{line}");
    }
}
