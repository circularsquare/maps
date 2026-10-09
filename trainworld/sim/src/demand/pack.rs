//! City pack format 1 (notes/T-004.md): a JSON header plus little-endian arrays back to back,
//! per cell and per 2 km gravity zone. The reader takes the arrays it knows and ignores the rest.

use serde_json::{json, Value};

pub const FORMAT: u64 = 1;

pub struct City {
    pub name: String,
    pub h3: Vec<u64>,
    /// Cell centre, metres east / north of the pack origin.
    pub x: Vec<f32>,
    pub y: Vec<f32>,
    /// The home-end and work-end weights of local commuting. From a pack with `commute_home`
    /// and `commute_work` (T-042: residents and jobs less those working from home) they are
    /// those arrays; otherwise the pack's `pop` and `jobs`. The gravity balances on them.
    pub pop: Vec<f32>,
    pub jobs: Vec<f32>,
    /// The zone gravity solved by the pipeline; None if the pack has none (then the kernel
    /// solves it, which is what the pipeline itself does).
    pub gravity: Option<PackGravity>,
}

impl City {
    pub fn len(&self) -> usize {
        self.x.len()
    }
    pub fn is_empty(&self) -> bool {
        self.x.is_empty()
    }
    pub fn totals(&self) -> (f64, f64) {
        let p = self.pop.iter().map(|&v| v as f64).sum();
        let j = self.jobs.iter().map(|&v| v as f64).sum();
        (p, j)
    }
}

/// Doubly constrained gravity between square zones, as shipped: zone trips
/// T_IJ = row_I * col_J * d_IJ^-decay_pow * exp(-d_IJ / decay_km), d by zone grid offset
/// (SPEC 4.4). In the header: `"decay": "exp"` (decay_pow 0) or `"pow_exp"` with `decay_pow`.
#[derive(Clone, Debug)]
pub struct PackGravity {
    pub zone_m: f32,
    pub decay_km: f32,
    pub decay_pow: f32,
    /// Lower-left corner of the zone grid, metres from the pack origin, and its size in zones.
    pub grid_x0: f32,
    pub grid_y0: f32,
    pub grid_w: i32,
    pub grid_h: i32,
    /// Per cell.
    pub cell_zone: Vec<u32>,
    /// Per zone: grid column and row, and the two balancing factors (A_I O_I and B_J D_J).
    pub gx: Vec<i32>,
    pub gy: Vec<i32>,
    pub row: Vec<f64>,
    pub col: Vec<f64>,
    pub iters: usize,
    pub max_err: f64,
    pub mean_km: f64,
    pub trips: f64,
}

fn dtype_size(dtype: &str) -> Result<usize, String> {
    Ok(match dtype {
        "u8" => 1,
        "u16" => 2,
        "f32" | "u32" | "i32" => 4,
        "f64" | "u64" => 8,
        _ => return Err(format!("dtype {dtype} not readable as numbers")),
    })
}

fn slice<'a>(bin: &'a [u8], off: usize, count: usize, dtype: &str) -> Result<&'a [u8], String> {
    let end = off + count * dtype_size(dtype)?;
    if end > bin.len() {
        return Err(format!("array at {off} runs past the end of the .bin ({end} > {})", bin.len()));
    }
    Ok(&bin[off..end])
}

fn read_f64s(bin: &[u8], off: usize, count: usize, dtype: &str) -> Result<Vec<f64>, String> {
    let b = slice(bin, off, count, dtype)?;
    Ok(match dtype {
        "f32" => b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap()) as f64).collect(),
        "f64" => b.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect(),
        "u32" => b.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap()) as f64).collect(),
        "i32" => b.chunks_exact(4).map(|c| i32::from_le_bytes(c.try_into().unwrap()) as f64).collect(),
        "u64" => b.chunks_exact(8).map(|c| u64::from_le_bytes(c.try_into().unwrap()) as f64).collect(),
        "u16" => b.chunks_exact(2).map(|c| u16::from_le_bytes(c.try_into().unwrap()) as f64).collect(),
        "u8" => b.iter().map(|&v| v as f64).collect(),
        _ => unreachable!(),
    })
}

fn read_f32s(bin: &[u8], off: usize, count: usize, dtype: &str) -> Result<Vec<f32>, String> {
    Ok(read_f64s(bin, off, count, dtype)?.into_iter().map(|v| v as f32).collect())
}

/// Integer arrays (ids, grid positions): any unsigned or signed integer dtype up to 32 bits.
fn read_ints(bin: &[u8], off: usize, count: usize, dtype: &str) -> Result<Vec<i64>, String> {
    let b = slice(bin, off, count, dtype)?;
    Ok(match dtype {
        "u8" => b.iter().map(|&v| v as i64).collect(),
        "u16" => b.chunks_exact(2).map(|c| u16::from_le_bytes(c.try_into().unwrap()) as i64).collect(),
        "u32" => b.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap()) as i64).collect(),
        "i32" => b.chunks_exact(4).map(|c| i32::from_le_bytes(c.try_into().unwrap()) as i64).collect(),
        _ => return Err(format!("dtype {dtype} is not an integer type")),
    })
}

/// Parse a pack from its header text and its .bin bytes.
pub fn parse(header: &str, bin: &[u8]) -> Result<City, String> {
    let v: Value = serde_json::from_str(header).map_err(|e| format!("header: {e}"))?;
    let format = v["format"].as_u64().ok_or("header has no format")?;
    if format != FORMAT {
        return Err(format!("pack format {format}, this reader knows {FORMAT}"));
    }
    let cells = v["cells"].as_u64().ok_or("header has no cells")? as usize;
    let zones = v["zones"].as_u64().unwrap_or(0) as usize;
    let name = v["city"].as_str().unwrap_or("?").to_string();
    let mut city = City { name, h3: vec![], x: vec![], y: vec![], pop: vec![], jobs: vec![], gravity: None };
    let (mut cz, mut gx, mut gy, mut row, mut col) = (vec![], vec![], vec![], vec![], vec![]);
    let (mut ch, mut cw) = (vec![], vec![]);
    for a in v["arrays"].as_array().ok_or("header has no arrays")? {
        let aname = a["name"].as_str().unwrap_or("");
        let dtype = a["dtype"].as_str().unwrap_or("");
        let per = a["per"].as_str().unwrap_or("cell");
        let off = a["offset"].as_u64().unwrap_or(0) as usize;
        let count = a["count"].as_u64().unwrap_or(0) as usize;
        let known = match per {
            "cell" => matches!(aname, "h3" | "x_m" | "y_m" | "pop" | "jobs" | "zone" | "commute_home" | "commute_work"),
            "zone" => matches!(aname, "zone_gx" | "zone_gy" | "zone_row" | "zone_col"),
            _ => false,
        };
        if !known {
            continue; // unknown arrays are allowed and ignored
        }
        let want = if per == "zone" { zones } else { cells };
        if count != want {
            return Err(format!("array {aname} has {count} entries, header says {want} {per}s"));
        }
        match aname {
            "h3" => {
                if dtype != "u64" {
                    return Err("h3 must be u64".into());
                }
                city.h3 = slice(bin, off, count, dtype)?.chunks_exact(8).map(|c| u64::from_le_bytes(c.try_into().unwrap())).collect();
            }
            "x_m" => city.x = read_f32s(bin, off, count, dtype)?,
            "y_m" => city.y = read_f32s(bin, off, count, dtype)?,
            "pop" => city.pop = read_f32s(bin, off, count, dtype)?,
            "jobs" => city.jobs = read_f32s(bin, off, count, dtype)?,
            "commute_home" => ch = read_f32s(bin, off, count, dtype)?,
            "commute_work" => cw = read_f32s(bin, off, count, dtype)?,
            "zone" => cz = read_ints(bin, off, count, dtype)?.into_iter().map(|v| v as u32).collect(),
            "zone_gx" => gx = read_ints(bin, off, count, dtype)?.into_iter().map(|v| v as i32).collect(),
            "zone_gy" => gy = read_ints(bin, off, count, dtype)?.into_iter().map(|v| v as i32).collect(),
            "zone_row" => row = read_f64s(bin, off, count, dtype)?,
            "zone_col" => col = read_f64s(bin, off, count, dtype)?,
            _ => unreachable!(),
        }
    }
    for (n, a) in [("x_m", &city.x), ("y_m", &city.y), ("pop", &city.pop), ("jobs", &city.jobs)] {
        if a.len() != cells {
            return Err(format!("pack lacks array {n}"));
        }
    }
    if city.h3.is_empty() {
        city.h3 = (0..cells as u64).collect();
    }
    match (ch.len() == cells, cw.len() == cells) {
        (true, true) => {
            city.pop = ch;
            city.jobs = cw;
        }
        (false, false) => {}
        _ => return Err("pack has one of commute_home, commute_work but not the other".into()),
    }
    let g = &v["gravity"];
    if g.is_object() {
        if cz.len() != cells || gx.len() != zones || gy.len() != zones || row.len() != zones || col.len() != zones {
            return Err("pack has a gravity block but lacks one of zone, zone_gx, zone_gy, zone_row, zone_col".into());
        }
        if cz.iter().any(|&z| z as usize >= zones) {
            return Err("a cell's zone id is out of range".into());
        }
        let num = |k: &str| g[k].as_f64().ok_or(format!("gravity block lacks {k}"));
        let decay_pow = match g["decay"].as_str().unwrap_or("") {
            "exp" => 0.0,
            "pow_exp" => num("decay_pow")? as f32,
            d => return Err(format!("gravity decay {d:?}, this reader knows \"exp\" and \"pow_exp\"")),
        };
        let o = g["grid_origin_m"].as_array().ok_or("gravity block lacks grid_origin_m")?;
        let s = g["grid"].as_array().ok_or("gravity block lacks grid")?;
        city.gravity = Some(PackGravity {
            zone_m: num("zone_m")? as f32,
            decay_km: num("decay_km")? as f32,
            decay_pow,
            grid_x0: o[0].as_f64().unwrap_or(0.0) as f32,
            grid_y0: o[1].as_f64().unwrap_or(0.0) as f32,
            grid_w: s[0].as_i64().unwrap_or(0) as i32,
            grid_h: s[1].as_i64().unwrap_or(0) as i32,
            cell_zone: cz,
            gx,
            gy,
            row,
            col,
            iters: num("iterations").unwrap_or(0.0) as usize,
            max_err: num("max_error").unwrap_or(0.0),
            mean_km: num("mean_trip_km")?,
            trips: num("trips")?,
        });
    }
    Ok(city)
}

/// Appends arrays to a .bin and their entries to the header's array list, 8-byte aligned.
struct Writer {
    bin: Vec<u8>,
    entries: Vec<Value>,
}

impl Writer {
    fn push(&mut self, name: &str, dtype: &str, per: &str, count: usize, bytes: Vec<u8>) {
        while self.bin.len() % 8 != 0 {
            self.bin.push(0);
        }
        self.entries.push(json!({ "name": name, "dtype": dtype, "per": per, "offset": self.bin.len(), "count": count }));
        self.bin.extend_from_slice(&bytes);
    }
}

fn le_f32(a: impl Iterator<Item = f32>) -> Vec<u8> {
    a.flat_map(|v| v.to_le_bytes()).collect()
}

/// Add (or replace) the zone gravity in a format-1 pack: the per-zone arrays go on the end of
/// the .bin, the `gravity` block and `zones` count into the header. Everything else in the
/// header (origin, totals, sources) is kept. Used by the pipeline (`bin/pack_gravity.rs`).
pub fn with_gravity(header: &str, bin: &[u8], g: &PackGravity) -> Result<(String, Vec<u8>), String> {
    let mut v: Value = serde_json::from_str(header).map_err(|e| format!("header: {e}"))?;
    let cells = v["cells"].as_u64().ok_or("header has no cells")? as usize;
    if g.cell_zone.len() != cells {
        return Err(format!("{} cell zone ids for {cells} cells", g.cell_zone.len()));
    }
    // drop any gravity arrays already there (their bytes stay as dead space; the pipeline
    // always starts from a cells-only pack, so in practice there are none)
    let mut arrays: Vec<Value> = v["arrays"].as_array().cloned().unwrap_or_default();
    arrays.retain(|a| !matches!(a["name"].as_str().unwrap_or(""), "zone" | "zone_gx" | "zone_gy" | "zone_row" | "zone_col"));
    let nz = g.gx.len();
    let mut w = Writer { bin: bin.to_vec(), entries: vec![] };
    w.push("zone", "u32", "cell", cells, g.cell_zone.iter().flat_map(|v| v.to_le_bytes()).collect());
    w.push("zone_gx", "u16", "zone", nz, g.gx.iter().flat_map(|&v| (v as u16).to_le_bytes()).collect());
    w.push("zone_gy", "u16", "zone", nz, g.gy.iter().flat_map(|&v| (v as u16).to_le_bytes()).collect());
    w.push("zone_row", "f32", "zone", nz, le_f32(g.row.iter().map(|&v| v as f32)));
    w.push("zone_col", "f32", "zone", nz, le_f32(g.col.iter().map(|&v| v as f32)));
    arrays.extend(w.entries);
    v["format"] = json!(FORMAT);
    v["zones"] = json!(nz);
    v["arrays"] = Value::Array(arrays);
    v["gravity"] = json!({
        "zone_m": g.zone_m,
        "grid_origin_m": [g.grid_x0, g.grid_y0],
        "grid": [g.grid_w, g.grid_h],
        "decay": if g.decay_pow == 0.0 { "exp" } else { "pow_exp" },
        "decay_km": g.decay_km,
        "decay_pow": g.decay_pow,
        "intrazonal_km": 0.52 * g.zone_m as f64 / 1000.0,
        "workers": "home-end weights (commute_home if the pack has it, else pop) scaled to the work-end total",
        "iterations": g.iters,
        "max_error": g.max_err,
        "mean_trip_km": g.mean_km,
        "trips": g.trips,
        "solver": "trainworld-sim kernel::gravity (sim/src/bin/pack_gravity.rs)",
    });
    let text = serde_json::to_string_pretty(&v).map_err(|e| e.to_string())? + "\n";
    Ok((text, w.bin))
}

/// Write a pack as (header, bin), with the gravity if the city has it. Used for synthetic packs
/// and tests, so the loader is exercised end to end.
pub fn write(city: &City) -> (String, Vec<u8>) {
    let n = city.len();
    let mut w = Writer { bin: Vec::with_capacity(n * 24), entries: vec![] };
    w.push("h3", "u64", "cell", n, city.h3.iter().flat_map(|v| v.to_le_bytes()).collect());
    for (name, a) in [("x_m", &city.x), ("y_m", &city.y), ("pop", &city.pop), ("jobs", &city.jobs)] {
        w.push(name, "f32", "cell", n, le_f32(a.iter().cloned()));
    }
    let (p, j) = city.totals();
    let v = json!({
        "format": FORMAT,
        "city": city.name,
        "built": "synthetic",
        "origin": { "lon": -73.985, "lat": 40.758 },
        "h3_res": 9,
        "cells": n,
        "arrays": w.entries,
        "totals": { "pop": p.round(), "jobs": j.round() },
        "sources": [ { "what": "everything", "name": "synthetic, trainworld-sim demand::synth", "url": "" } ],
    });
    let header = serde_json::to_string_pretty(&v).unwrap() + "\n";
    match &city.gravity {
        Some(g) => with_gravity(&header, &w.bin, g).unwrap(),
        None => (header, w.bin),
    }
}

/// Load `<stem>.json` + `<stem>.bin` from disk.
#[cfg(not(target_arch = "wasm32"))]
pub fn load(json_path: &str) -> Result<City, String> {
    let (header, bin) = load_raw(json_path)?;
    parse(&header, &bin)
}

/// The header text and .bin bytes of `<stem>.json`, unparsed.
#[cfg(not(target_arch = "wasm32"))]
pub fn load_raw(json_path: &str) -> Result<(String, Vec<u8>), String> {
    let header = std::fs::read_to_string(json_path).map_err(|e| format!("{json_path}: {e}"))?;
    let bin_path = json_path.strip_suffix(".json").unwrap_or(json_path).to_string() + ".bin";
    let bin = std::fs::read(&bin_path).map_err(|e| format!("{bin_path}: {e}"))?;
    Ok((header, bin))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small() -> City {
        City {
            name: "t".into(),
            h3: vec![10, 20, 30],
            x: vec![1.0, 2.0, 3.0],
            y: vec![-1.0, -2.0, -3.0],
            pop: vec![5.0, 0.0, 7.0],
            jobs: vec![0.0, 9.0, 1.0],
            gravity: None,
        }
    }

    #[test]
    fn roundtrip_and_ignores_unknown_arrays() {
        let city = small();
        let (h, mut b) = write(&city);
        // add an unknown array the reader must skip
        let off = b.len();
        b.extend_from_slice(&[0u8; 12]);
        let h = h.replacen(
            "\"arrays\": [\n",
            &format!("\"arrays\": [\n    {{ \"name\": \"road_kmh\", \"dtype\": \"f32\", \"per\": \"cell\", \"offset\": {off}, \"count\": 3 }},\n"),
            1,
        );
        let c = parse(&h, &b).unwrap();
        assert_eq!(c.h3, city.h3);
        assert_eq!(c.x, city.x);
        assert_eq!(c.y, city.y);
        assert_eq!(c.pop, city.pop);
        assert_eq!(c.jobs, city.jobs);
        assert!(c.gravity.is_none());
    }

    #[test]
    fn commute_arrays_become_the_demand_weights() {
        // T-042: commute_home / commute_work replace pop / jobs as the kernel's weights
        let city = small();
        let (h, mut b) = write(&city);
        let mut w = Writer { bin: std::mem::take(&mut b), entries: vec![] };
        w.push("commute_home", "f32", "cell", 3, le_f32([4.0f32, 0.0, 6.0].into_iter()));
        w.push("commute_work", "f32", "cell", 3, le_f32([0.0f32, 8.0, 0.5].into_iter()));
        let extra: Vec<String> = w.entries.iter().map(|e| e.to_string()).collect();
        let h2 = h.replacen("\"arrays\": [\n", &format!("\"arrays\": [\n{},\n", extra.join(",\n")), 1);
        let c = parse(&h2, &w.bin).unwrap();
        assert_eq!(c.pop, vec![4.0, 0.0, 6.0]);
        assert_eq!(c.jobs, vec![0.0, 8.0, 0.5]);
        // only one of the two is an error
        let h3 = h.replacen("\"arrays\": [\n", &format!("\"arrays\": [\n{},\n", extra[0]), 1);
        assert!(parse(&h3, &w.bin).is_err());
    }

    #[test]
    fn gravity_roundtrip() {
        let mut city = small();
        city.gravity = Some(PackGravity {
            zone_m: 2000.0,
            decay_km: 9.0,
            decay_pow: 0.5,
            grid_x0: 1.0,
            grid_y0: -3.0,
            grid_w: 1,
            grid_h: 2,
            cell_zone: vec![1, 0, 1],
            gx: vec![0, 0],
            gy: vec![1, 0],
            row: vec![0.5, 2.25],
            col: vec![3.0, 0.125],
            iters: 7,
            max_err: 1e-5,
            mean_km: 1.5,
            trips: 10.0,
        });
        let (h, b) = write(&city);
        let c = parse(&h, &b).unwrap();
        let (g, g0) = (c.gravity.unwrap(), city.gravity.unwrap());
        assert_eq!(g.cell_zone, g0.cell_zone);
        assert_eq!((g.gx, g.gy), (g0.gx, g0.gy));
        assert_eq!((g.row, g.col), (g0.row, g0.col));
        assert_eq!((g.grid_w, g.grid_h, g.zone_m, g.decay_km, g.decay_pow), (1, 2, 2000.0, 9.0, 0.5));
        assert_eq!(g.trips, 10.0);
    }
}
