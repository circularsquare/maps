//! T-918 (built): the ceiling tests, the slow line and the farming check on the shipped curve,
//! `Params::car_choice` as the game calls it. Run from a copy of the crate as
//! `everyday::t938_ceiling`:
//!
//!   T938_PACK=<pack .json> T938_OUT=out.json cargo test --release --features demand --lib t938 -- --ignored --nocapture

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
    ratio: f32,
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
const SCENS: [(&str, Scen); 13] = [
    ("A", sc(140.0, 3.0, 2.5, 0.0, 0.0)),
    ("B", sc(100.0, 3.0, 2.5, 1.5, 0.05)),
    ("T", sc(30.0, 6.0, 3.0, 1.5, 0.1)),
    ("S", sc(20.0, 10.0, 7.5, 1.5, 0.1)),
    ("R1.5", rs(1.5)),
    ("R2", rs(2.0)),
    ("R3", rs(3.0)),
    ("R3.5", rs(3.5)),
    ("R4", rs(4.0)),
    ("R5", rs(5.0)),
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

/// [rail, walk, car]
fn shares(p: &Params, sc: &Scen, d_m: f32, cw: f32, drive: f32, park: (f32, f32, f32), home: f32, rk: f32) -> [f64; 3] {
    let (fixed, paid, free) = park;
    let road = p.car_drive(drive, rk) + fixed + home;
    let car_mean = road + (1.0 - free) * paid;
    let rail = rail_cost(sc, p, d_m, car_mean);
    let bt = p.mode_beta(road);
    let util = bt * (road - rail) + p.asc_rail;
    let uw = bt * (road - cw) + p.asc_walk();
    let r = p.car_choice(util, uw, bt, paid, free, rail, road) as f64;
    let e = fast_exp(uw - util);
    let w0 = e / (1.0 + fast_exp(-util) + e);
    let w1 = e / (1.0 + fast_exp(-(util + bt * paid)) + e);
    let w = if paid < 1e-30 { w0 } else { w0 + (1.0 - free) * (w1 - w0) } as f64;
    [r, w, 1.0 - r - w]
}

struct Acc(Vec<[f64; 4]>);
impl Acc {
    fn add(&mut self, s: usize, g: usize, t: f64, sh: [f64; 3]) {
        for gg in [g, NG - 1] {
            let a = &mut self.0[s * NG + gg];
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
fn t938() {
    let pack_path = std::env::var("T938_PACK").expect("T938_PACK");
    let out_path = std::env::var("T938_OUT").expect("T938_OUT");
    let (h, b) = pack::load_raw(&pack_path).unwrap();
    let mut api = DemandApi::new(&h, &b).unwrap();
    if let Ok(s) = std::env::var("T938_SET") {
        for kv in s.split(',').filter(|x| !x.is_empty()) {
            let (k, x) = kv.split_once('=').unwrap();
            api.params_mut().set_extra(k, x.parse().unwrap()).unwrap();
        }
    }
    let ev = EverydayApi::beside(&api, &h, &b).unwrap();
    let n = api.city.len();
    let pop: Vec<f32> = pack::cell_weights(&h, &b, "pop", n).unwrap().unwrap();
    let mut acc = Acc(vec![[0.0; 4]; NS * NG]);
    {
        let (z, gr, p) = (&*api.z, &*api.gr, &api.base);
        let (_, dtab) = decay_table(z.w, z.h, z.zone_m, 1.0, 0.0);
        let place: Vec<(f32, f32)> = z.density().iter().map(|&x| p.place_min_per_km(x)).collect();
        let park: Vec<(f32, f32, f32)> = (0..z.n).map(|j| p.park_parts(z, j)).collect();
        let dres = zone_dens(z, &pop);
        let homes: Vec<f32> = dres.iter().map(|&d| p.home_park_at(d)).collect();
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
                let f = p.drive_factor(i, j);
                let drive = p.drive_min_f(d_km * 1000.0, f, place[i], place[j]);
                for (s, (_, sc)) in SCENS.iter().enumerate() {
                    acc.add(s, g, t as f64, shares(p, sc, d_km * 1000.0, cw, drive, park[j], homes[i], d_km * f));
                }
            }
        }
    }
    {
        let pz = &*ev.engine.z;
        let p = ev.ep.clone();
        let place: Vec<(f32, f32)> = pz.density().iter().map(|&x| p.place_min_per_km(x)).collect();
        let park: Vec<(f32, f32, f32)> = (0..pz.n).map(|j| p.park_parts(pz, j)).collect();
        let dres = zone_dens(pz, &pop);
        let homes: Vec<f32> = dres.iter().map(|&d| p.home_park_at(d)).collect();
        for k in 0..ev.destinations.len() {
            let Some((_, c, dd, w, _)) = ev.edge(k) else { continue };
            let dist = ev.distance(c, dd);
            let (hz, az) = (pz.cell_zone[c] as usize, pz.cell_zone[dd] as usize);
            let g = group(dres[hz]);
            let cw = p.walk_trip_cost(dist * ev.walk_factor(c, dd) / 1000.0);
            for (s, (_, sc)) in SCENS.iter().enumerate() {
                for (u, v2) in [(hz, az), (az, hz)] {
                    let f = p.drive_factor(u, v2);
                    let drive = p.drive_min_f(dist, f, place[u], place[v2]);
                    acc.add(s, g, w as f64, shares(&p, sc, dist, cw, drive, park[v2], homes[hz], dist * 0.001 * f));
                }
            }
        }
    }
    let out = json!({
        "pack": pack_path, "groups": ["<3k", "3-10k", "10-20k", ">=20k", "all"],
        "scens": SCENS.iter().map(|s| s.0).collect::<Vec<_>>(),
        "vars": ["built"],
        "acc": acc.0.iter().map(|a| a.to_vec()).collect::<Vec<_>>(),
    });
    std::fs::write(&out_path, serde_json::to_string(&out).unwrap()).unwrap();
}
