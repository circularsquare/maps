//! T-1277: Anita's ceilings (T-1056's harness, walking rule on) and a fare sweep on the ceiling's
//! B train, with the fare's weight, a per-layer weight and the ratio curve's power as knobs.
//!
//!   T1277_PACK=<pack> T1277_OUT=out.json [T1277_SET=k=v,..] [T1277_FW=1] [T1277_FWE=<FW>]
//!   [T1277_POW=3] cargo test --release --features demand --lib t1277 -- --ignored

use super::*;
use crate::demand::api::{DemandApi, FARE_MIN_PER_USD};
use crate::demand::pack;
use serde_json::json;

const CELL: f32 = 0.1053;
const MULTS: [f32; 16] = [0.333, 0.5, 0.667, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 13.0, 16.0, 20.0, 30.0];

#[derive(Clone, Copy)]
struct Scen {
    kmh: f32,
    walk_min: f32,
    wait_min: f32,
    base_usd: f32,
    km_usd: f32,
}

fn rail_cost(s: &Scen, p: &Params, d_m: f32, fw: f32) -> f32 {
    if 2.0 * s.walk_min > p.walk_min(d_m) {
        return 1e6;
    }
    let km = d_m / 1000.0;
    let walk = s.walk_min.min(p.walk_easy_min) * p.w_walk + (s.walk_min - p.walk_easy_min).max(0.0) * p.w_walk_far;
    km / s.kmh * 60.0 + 2.0 * walk + s.wait_min + fw * (s.base_usd + s.km_usd * km) * FARE_MIN_PER_USD
}

fn ratio(p: &Params, rail: f32, car: f32, pw: f32) -> f32 {
    let q = rail.max(0.01) / car;
    let s = 1.0 / (1.0 + q.powf(pw));
    let k = 1.0 / (p.ratio_fade_hi - p.ratio_fade_lo);
    let fr = ((p.ratio_fade_hi - q) * k).max(0.0).min(1.0);
    let fc = ((p.ratio_fade_hi - 1.0 / q) * k).max(0.0).min(1.0);
    1.0 - (1.0 - s * fr) * fc
}

/// [rail, walk, car]: `Params::car_choice` with the ratio's power `pw`
fn shares(p: &Params, rail: f32, d_m: f32, cw: f32, drive: f32, park: (f32, f32, f32), home: f32, rk: f32, pw: f32) -> [f64; 3] {
    let _ = d_m;
    let (fixed, paid, free) = park;
    let road = p.car_drive(drive, rk) + fixed + home;
    let bt = p.mode_beta(road);
    let util = bt * (road - rail) + p.asc_rail;
    let uw = bt * (road - cw) + p.asc_walk();
    let e = fast_exp(uw - util);
    let w0 = e / (1.0 + fast_exp(-util) + e);
    let s0 = (1.0 - w0) * ratio(p, rail, road, pw);
    let (r, w) = if paid < 1e-30 {
        (s0, w0)
    } else {
        let w1 = e / (1.0 + fast_exp(-(util + bt * paid)) + e);
        let s1 = (1.0 - w1) * ratio(p, rail, road + paid, pw);
        (s0 + (1.0 - free) * (s1 - s0), w0 + (1.0 - free) * (w1 - w0))
    };
    [r as f64, w as f64, (1.0 - r - w) as f64]
}

fn zone_dens(z: &Zones, pop: &[f32]) -> Vec<f32> {
    let mut pp = vec![0f64; z.n];
    for c in 0..pop.len() {
        pp[z.cell_zone[c] as usize] += pop[c] as f64;
    }
    let land = |i: usize| (z.ncell[i].max(1) as f32 * CELL).min((z.zone_m / 1000.0).powi(2));
    (0..z.n).map(|i| pp[i] as f32 / land(i)).collect()
}

/// per scenario and layer: trips, rail, walk, car, fares in US$
struct Acc(Vec<[f64; 5]>);
impl Acc {
    fn add(&mut self, s: usize, layer: usize, t: f64, sh: [f64; 3], fare: f64) {
        let a = &mut self.0[2 * s + layer];
        a[0] += t;
        for k in 0..3 {
            a[k + 1] += t * sh[k];
        }
        a[4] += t * sh[0] * fare;
    }
}

fn envf(k: &str, d: f32) -> f32 {
    std::env::var(k).ok().map_or(d, |v| v.parse().unwrap())
}

#[test]
#[ignore]
fn t1277() {
    let pack_path = std::env::var("T1277_PACK").expect("T1277_PACK");
    let out_path = std::env::var("T1277_OUT").expect("T1277_OUT");
    let fw = envf("T1277_FW", 1.0);
    let fwe = envf("T1277_FWE", fw);
    let pw = envf("T1277_POW", 3.0);
    let (h, b) = pack::load_raw(&pack_path).unwrap();
    let mut api = DemandApi::new(&h, &b).unwrap();
    if let Ok(s) = std::env::var("T1277_SET") {
        for kv in s.split(',').filter(|x| !x.is_empty()) {
            let (k, x) = kv.split_once('=').unwrap();
            api.params_mut().set_extra(k, x.parse().unwrap()).unwrap();
        }
        api.refresh_walk();
    }
    let ev = EverydayApi::beside(&api, &h, &b).unwrap();
    let n = api.city.len();
    let pop: Vec<f32> = pack::cell_weights(&h, &b, "pop", n).unwrap().unwrap();
    // A, B, then the sweep on B's train at m x the game's starting fares ($1.50 + $0.10 a km)
    let mut scens: Vec<(String, Scen)> = vec![
        ("A".into(), Scen { kmh: 140.0, walk_min: 3.0, wait_min: 2.5, base_usd: 0.0, km_usd: 0.0 }),
        ("B".into(), Scen { kmh: 100.0, walk_min: 3.0, wait_min: 2.5, base_usd: 1.5, km_usd: 0.05 }),
    ];
    for m in MULTS {
        scens.push((format!("m{m}"), Scen { kmh: 100.0, walk_min: 3.0, wait_min: 2.5, base_usd: 1.5 * m, km_usd: 0.10 * m }));
    }
    let ns = scens.len();
    let mut acc = Acc(vec![[0.0; 5]; 2 * ns]);
    {
        let (z, gr, p) = (&*api.z, &*api.gr, &api.base);
        let (_, dtab) = decay_tables(z, Curve { km: 1.0, pow: 0.0, flat_km: 0.0 });
        let place: Vec<(f32, f32)> = z.density().iter().map(|&x| p.place_min_per_km(x)).collect();
        let park: Vec<(f32, f32, f32)> = (0..z.n).map(|j| p.park_parts(z, j)).collect();
        let dres = zone_dens(z, &pop);
        let homes: Vec<f32> = dres.iter().map(|&d| p.home_park_at(d)).collect();
        for i in 0..z.n {
            if gr.row[i] == 0.0 {
                continue;
            }
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
                for (s, (_, sc)) in scens.iter().enumerate() {
                    let rail = rail_cost(sc, p, d_km * 1000.0, fw);
                    let fare = (sc.base_usd + sc.km_usd * d_km) as f64;
                    acc.add(s, 0, t as f64, shares(p, rail, d_km * 1000.0, cw, drive, park[j], homes[i], d_km * f, pw), fare);
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
            let cw = p.walk_trip_cost(dist * ev.walk_factor(c, dd) / 1000.0);
            for (s, (_, sc)) in scens.iter().enumerate() {
                let rail = rail_cost(sc, &p, dist, fwe);
                let fare = (sc.base_usd + sc.km_usd * dist / 1000.0) as f64;
                for (u, v2) in [(hz, az), (az, hz)] {
                    let f = p.drive_factor(u, v2);
                    let drive = p.drive_min_f(dist, f, place[u], place[v2]);
                    acc.add(s, 1, w as f64, shares(&p, rail, dist, cw, drive, park[v2], homes[hz], dist * 0.001 * f, pw), fare);
                }
            }
        }
    }
    let out = json!({
        "pack": pack_path, "fw": fw, "fwe": fwe, "pow": pw,
        "scens": scens.iter().map(|s| s.0.clone()).collect::<Vec<_>>(),
        "mults": MULTS,
        "acc": acc.0.iter().map(|a| a.to_vec()).collect::<Vec<_>>(),
    });
    std::fs::write(&out_path, serde_json::to_string(&out).unwrap()).unwrap();
}
