//! T-918: Anita's ceiling tests (A, B), a typical metro (T), a slow line (S), and rail at fixed
//! multiples of the car's cost (R1.2 .. R4, the farming check), by home density, for each choice
//! curve. Built only in a scratch copy of the crate, as `everyday::t918_ceiling` (T-854's harness
//! brought to the current model).
//!
//!   T918_PACK=<pack .json> T918_OUT=out.json cargo test --release --features demand --lib t918 -- --ignored --nocapture

use super::*;
use crate::demand::api::{DemandApi, FARE_MIN_PER_USD};
use crate::demand::pack;
use serde_json::json;

const CELL: f32 = 0.1053;
const GROUP_EDGES: [f32; 3] = [3000.0, 10000.0, 20000.0];
const NG: usize = 5;
fn group(d: f32) -> usize {
    GROUP_EDGES.iter().position(|&e| d < e).unwrap_or(3)
}

#[derive(Clone, Copy)]
struct Scen {
    kmh: f32,
    walk_min: f32,
    wait_min: f32,
    base_usd: f32,
    km_usd: f32,
    /// > 0: rail costs this times the car's mean perceived cost instead
    ratio: f32,
    /// > 0: rail costs the car's mean plus this many minutes instead
    plus: f32,
}
const fn sc(kmh: f32, walk_min: f32, wait_min: f32, base_usd: f32, km_usd: f32) -> Scen {
    Scen { kmh, walk_min, wait_min, base_usd, km_usd, ratio: 0.0, plus: 0.0 }
}
const fn rs(ratio: f32) -> Scen {
    Scen { kmh: 0.0, walk_min: 0.0, wait_min: 0.0, base_usd: 0.0, km_usd: 0.0, ratio, plus: 0.0 }
}
const fn ps(plus: f32) -> Scen {
    Scen { kmh: 0.0, walk_min: 0.0, wait_min: 0.0, base_usd: 0.0, km_usd: 0.0, ratio: 0.0, plus }
}
const SCENS: [(&str, Scen); 12] = [
    ("A", sc(140.0, 3.0, 2.5, 0.0, 0.0)),
    ("B", sc(100.0, 3.0, 2.5, 1.5, 0.05)),
    ("T", sc(30.0, 6.0, 3.0, 1.5, 0.1)),
    ("S", sc(20.0, 10.0, 7.5, 1.5, 0.1)),
    ("R1.2", rs(1.2)),
    ("R1.5", rs(1.5)),
    ("R2", rs(2.0)),
    ("R3", rs(3.0)),
    ("R4", rs(4.0)),
    ("P30", ps(30.0)),
    ("P60", ps(60.0)),
    ("P120", ps(120.0)),
];
const NS: usize = SCENS.len();

fn rail_cost(s: &Scen, p: &Params, d_m: f32, car_mean: f32) -> f32 {
    if s.ratio > 0.0 {
        return s.ratio * car_mean;
    }
    if s.plus > 0.0 {
        return car_mean + s.plus;
    }
    let km = d_m / 1000.0;
    let walk = s.walk_min.min(p.walk_easy_min) * p.w_walk + (s.walk_min - p.walk_easy_min).max(0.0) * p.w_walk_far;
    km / s.kmh * 60.0 + 2.0 * walk + s.wait_min + (s.base_usd + s.km_usd * km) * FARE_MIN_PER_USD
}

/// (name, curve, cv_a, cv_w, cv_g, rail_floor)
const VARS: [(&str, u8, f32, f32, f32, f32); 17] = [
    ("mix15", 2, 1.1486, 0.15, 0.0, 0.0),
    ("ratio3", 4, 1.0, 0.0, 3.0, 0.0),
    ("ratio2.5", 4, 1.0, 0.0, 2.5, 0.0),
    ("today", 1, 1.0, 0.0, 0.0, 0.05),
    ("today-nob", 1, 1.0, 0.0, 0.0, 0.0),
    ("L07", 1, 0.7, 0.0, 0.0, 0.0),
    ("L07+b", 1, 0.7, 0.0, 0.0, 0.05),
    ("mix10", 2, 1.0942, 0.1, 0.0, 0.0),
    ("mix20", 2, 1.2088, 0.2, 0.0, 0.0),
    ("mix30", 2, 1.3507, 0.3, 0.0, 0.0),
    ("mix20+b", 2, 1.2088, 0.2, 0.0, 0.05),
    ("atan", 3, 0.8875, 0.0, 0.0, 0.0),
    ("atan+b", 3, 0.8875, 0.0, 0.0, 0.05),
    ("ratio3.5", 4, 1.0, 0.0, 3.476, 0.0),
    ("ratio5.5", 4, 1.0, 0.0, 5.485, 0.0),
    ("ratio3.5+b", 4, 1.0, 0.0, 3.476, 0.05),
    ("mix10+b", 2, 1.0942, 0.1, 0.0, 0.05),
];
const NV: usize = VARS.len();

fn var_params(p: &Params, vi: usize) -> Params {
    let (_, c, a, w, g, fl) = VARS[vi];
    Params { curve: c, cv_a: a, cv_w: w, cv_t: 0.25, cv_g: g, rail_floor: fl, rail_floor_ratio: 3.0, ..p.clone() }
}

/// [rail, walk, car]
fn shares(p: &Params, sc: &Scen, d_m: f32, cw: f32, drive: f32, park: (f32, f32, f32), home: f32, rk: f32) -> [f64; 3] {
    let (fixed, paid, free) = park;
    let road = p.car_drive(drive, rk) + fixed + home;
    let car_mean = road + (1.0 - free) * paid;
    let rail = rail_cost(sc, p, d_m, car_mean);
    let bt = p.mode_beta(road);
    let (r, w) = p.curve_choice_rw(road, rail, cw, bt, paid, free);
    [r, w, 1.0 - r - w]
}

struct Acc(Vec<[f64; 4]>);
impl Acc {
    fn add(&mut self, vi: usize, s: usize, g: usize, t: f64, sh: [f64; 3]) {
        for gg in [g, NG - 1] {
            let a = &mut self.0[(vi * NS + s) * NG + gg];
            a[0] += t;
            for k in 0..3 {
                a[k + 1] += t * sh[k];
            }
        }
    }
}

fn zone_dens(z: &Zones, pop: &[f32]) -> Vec<f32> {
    let mut pp = vec![0f64; z.n];
    for c in 0..pop.len() {
        pp[z.cell_zone[c] as usize] += pop[c] as f64;
    }
    let land = |i: usize| (z.ncell[i].max(1) as f32 * CELL).min((z.zone_m / 1000.0).powi(2));
    (0..z.n).map(|i| pp[i] as f32 / land(i)).collect()
}

#[test]
#[ignore]
fn t918() {
    let pack_path = std::env::var("T918_PACK").expect("T918_PACK");
    let out_path = std::env::var("T918_OUT").expect("T918_OUT");
    let t0 = std::time::Instant::now();
    let (h, b) = pack::load_raw(&pack_path).unwrap();
    let api = DemandApi::new(&h, &b).unwrap();
    let ev = EverydayApi::beside(&api, &h, &b).unwrap();
    let n = api.city.len();
    let pop: Vec<f32> = pack::cell_weights(&h, &b, "pop", n).unwrap().unwrap();
    let mut acc = Acc(vec![[0.0; 4]; NV * NS * NG]);
    // ---- commutes, by zone pair
    {
        let (z, gr, p0) = (&*api.z, &*api.gr, &api.base);
        let ps: Vec<Params> = (0..NV).map(|vi| var_params(p0, vi)).collect();
        let (_, dtab) = decay_table(z.w, z.h, z.zone_m, 1.0, 0.0);
        let place: Vec<(f32, f32)> = z.density().iter().map(|&x| p0.place_min_per_km(x)).collect();
        let park: Vec<(f32, f32, f32)> = (0..z.n).map(|j| p0.park_parts(z, j)).collect();
        let dres = zone_dens(z, &pop);
        let homes: Vec<f32> = dres.iter().map(|&d| p0.home_park_at(d)).collect();
        for i in 0..z.n {
            if gr.row[i] == 0.0 {
                continue;
            }
            let g = group(dres[i]);
            let base = gr.base(z, i);
            for j in 0..z.n {
                let t = 2.0 * gr.t(z, i, j);
                if t <= 0.0 {
                    continue;
                }
                let k = (base + gr.key(z, j)) as usize;
                let d_km = dtab[k];
                let cw = gr.walk_cost_at(i, j, k);
                let f = p0.drive_factor(i, j);
                let drive = p0.drive_min_f(d_km * 1000.0, f, place[i], place[j]);
                for (s, (_, sc)) in SCENS.iter().enumerate() {
                    for vi in 0..NV {
                        let sh = shares(&ps[vi], sc, d_km * 1000.0, cw, drive, park[j], homes[i], d_km * f);
                        acc.add(vi, s, g, t as f64, sh);
                    }
                }
            }
        }
    }
    eprintln!("commutes done");
    // ---- everyday samples, both directions; the home end's parking on both trips
    {
        let pz = &*ev.engine.z;
        let p0 = ev.ep.clone();
        let ps: Vec<Params> = (0..NV).map(|vi| var_params(&p0, vi)).collect();
        let place: Vec<(f32, f32)> = pz.density().iter().map(|&x| p0.place_min_per_km(x)).collect();
        let park: Vec<(f32, f32, f32)> = (0..pz.n).map(|j| p0.park_parts(pz, j)).collect();
        let dres = zone_dens(pz, &pop);
        let homes: Vec<f32> = dres.iter().map(|&d| p0.home_park_at(d)).collect();
        for k in 0..ev.destinations.len() {
            let Some((_, c, dd, w, _)) = ev.edge(k) else { continue };
            let dist = ev.distance(c, dd);
            let (hz, az) = (pz.cell_zone[c] as usize, pz.cell_zone[dd] as usize);
            let g = group(dres[hz]);
            let cw = p0.walk_trip_cost(dist * ev.walk_factor(c, dd) / 1000.0);
            for (s, (_, sc)) in SCENS.iter().enumerate() {
                for (u, v2) in [(hz, az), (az, hz)] {
                    let f = p0.drive_factor(u, v2);
                    let drive = p0.drive_min_f(dist, f, place[u], place[v2]);
                    for vi in 0..NV {
                        let sh = shares(&ps[vi], sc, dist, cw, drive, park[v2], homes[hz], dist * 0.001 * f);
                        acc.add(vi, s, g, w as f64, sh);
                    }
                }
            }
        }
    }
    let out = json!({
        "pack": pack_path, "groups": ["<3k", "3-10k", "10-20k", ">=20k", "all"],
        "scens": SCENS.iter().map(|s| s.0).collect::<Vec<_>>(),
        "vars": VARS.iter().map(|v| v.0).collect::<Vec<_>>(),
        "acc": acc.0.iter().map(|a| a.to_vec()).collect::<Vec<_>>(),
        "secs": t0.elapsed().as_secs_f64(),
    });
    std::fs::write(&out_path, serde_json::to_string(&out).unwrap()).unwrap();
}

/// The choice function's cost per call, today's lane expressions against each curve.
#[test]
#[ignore]
fn t918_bench() {
    let p0 = Params::default();
    let n = 2_000_000usize;
    let mut out = vec![];
    for vi in 0..NV {
        let p = var_params(&p0, vi);
        let t = std::time::Instant::now();
        let mut s = 0f64;
        for i in 0..n {
            let road = 20.0 + (i % 97) as f32;
            let rail = 15.0 + (i % 89) as f32;
            let bt = p.mode_beta(road);
            s += p.curve_choice(road, rail, 40.0 + (i % 13) as f32, bt, 10.0, 0.4) as f64;
        }
        out.push((VARS[vi].0, t.elapsed().as_secs_f64() * 1e9 / n as f64, s));
    }
    let p = Params::default();
    let t = std::time::Instant::now();
    let mut s = 0f64;
    for i in 0..n {
        let road = 20.0 + (i % 97) as f32;
        let rail = 15.0 + (i % 89) as f32;
        let bt = p.mode_beta(road);
        let (util, uw) = (bt * (road - rail), bt * (road - 40.0 - (i % 13) as f32) + p.asc_walk());
        s += p.rail_bounds(p.car_choice(util, uw, bt, 10.0, 0.4), fast_exp(uw - util), rail, road + 0.6 * 10.0) as f64;
    }
    out.push(("shipped f32", t.elapsed().as_secs_f64() * 1e9 / n as f64, s));
    for (k, ns, s) in out {
        eprintln!("{k}: {ns:.1} ns ({s:.1})");
    }
}
