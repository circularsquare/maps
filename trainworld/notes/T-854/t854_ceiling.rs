//! T-854: Anita's ceiling tests (A, B), two made-up services (T a typical metro, S a slow line)
//! and rail at fixed multiples of the car's cost (R1.5, R2, R3), by home density, for the old
//! car-free split, home-end parking, and floors and ceilings inside the choice curve. Built only in
//! a private copy of the crate (as T-636's test), as `everyday::t854_ceiling`; the shared crate
//! never sees it.
//!
//!   T854_PACK=<pack .json> T854_OUT=out.json cargo test --release --features demand --lib t854 -- --ignored --nocapture
//!
//! Every choice uses the model's own `Params` expressions (`drive_min_f`, `car_drive`,
//! `park_parts`, `home_park_at`, `mode_beta`, the walk costs, `other_min_km`); new are only the
//! made-up trains' costs and the variants' rules, all here.

use super::*;
use crate::demand::api::{DemandApi, FARE_MIN_PER_USD};
use crate::demand::pack;
use serde_json::json;

const CELL: f32 = 0.1053;
/// Home density groups, people per km2 of the 2 km zone's land: <3k, 3-10k, 10-20k, >=20k.
const GROUP_EDGES: [f32; 3] = [3000.0, 10000.0, 20000.0];
const NG: usize = 5; // the four groups and all
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
}
const fn sc(kmh: f32, walk_min: f32, wait_min: f32, base_usd: f32, km_usd: f32) -> Scen {
    Scen { kmh, walk_min, wait_min, base_usd, km_usd, ratio: 0.0 }
}
const fn rs(ratio: f32) -> Scen {
    Scen { kmh: 0.0, walk_min: 0.0, wait_min: 0.0, base_usd: 0.0, km_usd: 0.0, ratio }
}
const SCENS: [(&str, Scen); 8] = [
    ("A", sc(140.0, 3.0, 2.5, 0.0, 0.0)),
    ("B", sc(100.0, 3.0, 2.5, 1.5, 0.05)),
    ("T", sc(30.0, 6.0, 3.0, 1.5, 0.1)),
    ("S", sc(20.0, 10.0, 7.5, 1.5, 0.1)),
    ("R1.2", rs(1.2)),
    ("R1.5", rs(1.5)),
    ("R2", rs(2.0)),
    ("R3", rs(3.0)),
];
const NS: usize = SCENS.len();

fn rail_cost(s: &Scen, p: &Params, d_m: f32, car_mean: f32) -> f32 {
    if s.ratio > 0.0 {
        return s.ratio * car_mean;
    }
    let km = d_m / 1000.0;
    let walk = s.walk_min.min(p.walk_easy_min) * p.w_walk + (s.walk_min - p.walk_easy_min).max(0.0) * p.w_walk_far;
    km / s.kmh * 60.0 + 2.0 * walk + s.wait_min + (s.base_usd + s.km_usd * km) * FARE_MIN_PER_USD
}

#[derive(Clone, Copy)]
struct Var {
    /// 0 the pack's car-free shares, 1 nobody car-free
    nocar: u8,
    home_f: f32,
    /// the curve's bounds: `a` of rail + car always rail where rail costs at most the car's, fading
    /// to none at `rmax` times; with `ceil`, the same share never rail, mirrored on car / rail
    a: f32,
    rmax: f32,
    ceil: bool,
}
const fn v(nocar: u8, home_f: f32, a: f32, rmax: f32, ceil: bool) -> Var {
    Var { nocar, home_f, a, rmax, ceil }
}
const VARS: [(&str, Var); 19] = [
    ("old", v(0, 0.0, 0.0, 0.0, false)),
    ("home0.4", v(1, 0.4, 0.0, 0.0, false)),
    ("home0.5", v(1, 0.5, 0.0, 0.0, false)),
    ("b2/1.5", v(1, 0.4, 0.02, 1.5, true)),
    ("b2/2", v(1, 0.4, 0.02, 2.0, true)),
    ("b2/3", v(1, 0.4, 0.02, 3.0, true)),
    ("b2/4", v(1, 0.4, 0.02, 4.0, true)),
    ("b5/1.5", v(1, 0.4, 0.05, 1.5, true)),
    ("b5/2", v(1, 0.4, 0.05, 2.0, true)),
    ("b5/3", v(1, 0.4, 0.05, 3.0, true)),
    ("b5/4", v(1, 0.4, 0.05, 4.0, true)),
    ("b10/1.5", v(1, 0.4, 0.10, 1.5, true)),
    ("b10/2", v(1, 0.4, 0.10, 2.0, true)),
    ("b10/3", v(1, 0.4, 0.10, 3.0, true)),
    ("b10/4", v(1, 0.4, 0.10, 4.0, true)),
    ("f5/3", v(1, 0.4, 0.05, 3.0, false)),
    ("f10/3", v(1, 0.4, 0.10, 3.0, false)),
    ("f5/2", v(1, 0.4, 0.05, 2.0, false)),
    ("b5/2.5", v(1, 0.4, 0.05, 2.5, true)),
];
const NV: usize = VARS.len();

/// Drivers' shares [rail, walk, car]: free parkers and payers apart, at scale `bt`.
fn drivers(p: &Params, bt: f32, road: f32, paid: f32, free: f32, rail: f32, cw: f32) -> [f64; 3] {
    let util = (bt * (road - rail) + p.asc_rail) as f64;
    let walk = cw < NO_WALK;
    let uw = if walk { (bt * (road - cw) + bt * p.walk_bonus_min) as f64 } else { 0.0 };
    let three = |sh: f64| {
        let er = if rail < 1e5 { (util + sh).exp() } else { 0.0 };
        let ew = if walk { (uw + sh).exp() } else { 0.0 };
        let d = 1.0 + er + ew;
        [er / d, ew / d, 1.0 / d]
    };
    let (a, b) = (three(0.0), three((bt * paid) as f64));
    let f = free as f64;
    [f * a[0] + (1.0 - f) * b[0], f * a[1] + (1.0 - f) * b[1], f * a[2] + (1.0 - f) * b[2]]
}

/// The bounds inside the curve, on rail of rail + car; walking keeps its share.
fn bounds(m: [f64; 3], v: &Var, rail: f32, car: f32) -> [f64; 3] {
    if v.a <= 0.0 || rail >= 1e5 {
        return m;
    }
    let rho = (rail / car.max(1e-6)) as f64;
    let g = |x: f64| ((v.rmax as f64 - x) / (v.rmax as f64 - 1.0)).clamp(0.0, 1.0);
    let a = v.a as f64;
    let lo = a * g(rho);
    let hi = if v.ceil { a * g(1.0 / rho.max(1e-6)) } else { 0.0 };
    let n = 1.0 - m[1];
    if n <= 0.0 {
        return m;
    }
    let r = lo + (1.0 - lo - hi) * m[0] / n;
    [r * n, m[1], (1.0 - r) * n]
}

/// One trip's shares [rail, walk, car, other].
#[allow(clippy::too_many_arguments)]
fn shares(p: &Params, v: &Var, sc: &Scen, d_m: f32, cw: f32, drive: f32, park: (f32, f32, f32), home: f32, cu: f32, rk: f32) -> [f64; 4] {
    let (fixed, paid, free) = park;
    let road = p.car_drive(drive, rk) + fixed + home;
    let car_mean = road + (1.0 - free) * paid;
    let rail = rail_cost(sc, p, d_m, car_mean);
    let bt = p.mode_beta(road);
    let m = bounds(drivers(p, bt, road, paid, free, rail, cw), v, rail, car_mean);
    if cu <= 0.0 {
        return [m[0], m[1], m[2], 0.0];
    }
    let c = cu as f64;
    let oth = rk * p.other_min_km() + p.other_wait();
    let bn = p.mode_beta(oth);
    let oa = oth + p.other_asc;
    let walk = cw < NO_WALK;
    let ur = (bn * (oa - rail) + p.asc_rail) as f64;
    let uwn = if walk { (bn * (oa - cw) + p.asc_walk()) as f64 } else { 0.0 };
    let (er, ew) = (if rail < 1e5 { ur.exp() } else { 0.0 }, if walk { uwn.exp() } else { 0.0 });
    let d = 1.0 + er + ew;
    [(1.0 - c) * m[0] + c * er / d, (1.0 - c) * m[1] + c * ew / d, (1.0 - c) * m[2], c / d]
}

/// [variant][scenario][group][trips, rail, walk, car, other]
struct Acc(Vec<[f64; 5]>);
impl Acc {
    fn add(&mut self, vi: usize, s: usize, g: usize, t: f64, sh: [f64; 4]) {
        for gg in [g, NG - 1] {
            let a = &mut self.0[(vi * NS + s) * NG + gg];
            a[0] += t;
            for k in 0..4 {
                a[k + 1] += t * sh[k];
            }
        }
    }
}

/// Residential density per zone (people per km2 of its cells' land) and the zone's mean share
/// without a car, from the cells.
fn zone_homes(z: &Zones, pop: &[f32], nocar: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let mut pp = vec![0f64; z.n];
    let mut nc = vec![0f64; z.n];
    for c in 0..pop.len() {
        let i = z.cell_zone[c] as usize;
        pp[i] += pop[c] as f64;
        nc[i] += pop[c] as f64 * nocar[c] as f64;
    }
    let land = |i: usize| (z.ncell[i].max(1) as f32 * CELL).min((z.zone_m / 1000.0).powi(2));
    ((0..z.n).map(|i| pp[i] as f32 / land(i)).collect(), (0..z.n).map(|i| if pp[i] > 0.0 { (nc[i] / pp[i]) as f32 } else { 0.0 }).collect())
}

#[test]
#[ignore]
fn t854() {
    let pack_path = std::env::var("T854_PACK").expect("T854_PACK");
    let out_path = std::env::var("T854_OUT").expect("T854_OUT");
    let (h, b) = pack::load_raw(&pack_path).unwrap();
    let api = DemandApi::new(&h, &b).unwrap();
    let ev = EverydayApi::beside(&api, &h, &b).unwrap();
    let n = api.city.len();
    // people, not the commute's home weights (`City::pop`)
    let pop: Vec<f32> = pack::cell_weights(&h, &b, "pop", n).unwrap().unwrap();
    // the pack's car-free shares, as the model read them before T-854
    let nocar: Vec<f32> = match pack::cell_weights(&h, &b, "nocar", n).unwrap() {
        Some(nc) => nc.iter().zip(&pop).map(|(&x, &p)| if p > 0.0 { (x / p).min(1.0) } else { 0.0 }).collect(),
        None => vec![0.0; n],
    };
    let mut acc = Acc(vec![[0.0; 5]; NV * NS * NG]);
    let hp = |p: &Params, d: f32, f: f32| Params { home_park_f: f, ..p.clone() }.home_park_at(d);
    // ---- commutes, by zone pair
    {
        let (z, gr, p) = (&*api.z, &*api.gr, &api.base);
        let (_, dtab) = decay_table(z.w, z.h, z.zone_m, 1.0, 0.0);
        let place: Vec<(f32, f32)> = z.density().iter().map(|&x| p.place_min_per_km(x)).collect();
        let park: Vec<(f32, f32, f32)> = (0..z.n).map(|j| p.park_parts(z, j)).collect();
        let (dres, cz) = zone_homes(z, &pop, &nocar);
        let homes: Vec<[f32; NV]> = dres.iter().map(|&d| std::array::from_fn(|vi| hp(p, d, VARS[vi].1.home_f))).collect();
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
                    for (vi, (_, v)) in VARS.iter().enumerate() {
                        let cu = if v.nocar == 0 { cz[i] } else { 0.0 };
                        let sh = shares(p, v, sc, d_km * 1000.0, cw, drive, park[j], homes[i][vi], cu, d_km * f);
                        acc.add(vi, s, g, t, sh);
                    }
                }
            }
        }
    }
    eprintln!("commutes done");
    // ---- everyday samples, both directions; the home end's parking on both trips
    {
        let pz = &*ev.engine.z;
        let p = ev.ep.clone();
        let place: Vec<(f32, f32)> = pz.density().iter().map(|&x| p.place_min_per_km(x)).collect();
        let park: Vec<(f32, f32, f32)> = (0..pz.n).map(|j| p.park_parts(pz, j)).collect();
        let (dres, _) = zone_homes(pz, &pop, &nocar);
        let homes: Vec<[f32; NV]> = dres.iter().map(|&d| std::array::from_fn(|vi| hp(&p, d, VARS[vi].1.home_f))).collect();
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
                    for (vi, (_, v)) in VARS.iter().enumerate() {
                        let cu = if v.nocar == 0 { nocar[c] } else { 0.0 };
                        let sh = shares(&p, v, sc, dist, cw, drive, park[v2], homes[hz][vi], cu, dist * 0.001 * f);
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
    });
    std::fs::write(&out_path, serde_json::to_string(&out).unwrap()).unwrap();
}
