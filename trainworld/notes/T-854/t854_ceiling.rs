//! T-854: Anita's ceiling tests (A, B) and two made-up services (T a typical metro, S a slow
//! line), with the car-free split replaced by other rules, by home density. Built only in a
//! private copy of the crate (as T-636's test); the shared crate never sees it.
//!
//!   T854_PACK=<pack .json> T854_OUT=out.json cargo test --release --features demand --lib t854 -- --ignored --nocapture
//!
//! Every choice uses the model's own `Params` expressions (`drive_min_f`, `car_drive`,
//! `park_parts`, `mode_beta`, the walk costs, `other_min_km`); new are only the made-up trains'
//! costs, the home-end parking and the tails, all here.

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
}
const SCENS: [(&str, Scen); 4] = [
    ("A", Scen { kmh: 140.0, walk_min: 3.0, wait_min: 2.5, base_usd: 0.0, km_usd: 0.0 }),
    ("B", Scen { kmh: 100.0, walk_min: 3.0, wait_min: 2.5, base_usd: 1.5, km_usd: 0.05 }),
    // a typical metro reached on foot: 30 km/h door to platform average, 6 minutes' walk each end,
    // a train every 6 minutes, the game's starting fares
    ("T", Scen { kmh: 30.0, walk_min: 6.0, wait_min: 3.0, base_usd: 1.5, km_usd: 0.1 }),
    // a slow, sparse line: 20 km/h, 10 minutes' walk each end, every 15 minutes
    ("S", Scen { kmh: 20.0, walk_min: 10.0, wait_min: 7.5, base_usd: 1.5, km_usd: 0.1 }),
];
const NS: usize = SCENS.len();

fn rail_cost(s: &Scen, p: &Params, d_m: f32) -> f32 {
    let km = d_m / 1000.0;
    let walk = s.walk_min.min(p.walk_easy_min) * p.w_walk + (s.walk_min - p.walk_easy_min).max(0.0) * p.w_walk_far;
    km / s.kmh * 60.0 + 2.0 * walk + s.wait_min + (s.base_usd + s.km_usd * km) * FARE_MIN_PER_USD
}

/// How a variant treats homes without a car, the home end's parking and the choice's tail.
#[derive(Clone, Copy)]
struct Var {
    /// 0 the pack's, 1 nobody, 2 one density curve, 3 the pack's mean everywhere
    nocar: u8,
    /// home-end parking: this times the destination's paid-space rule on residential density
    home_f: f32,
    /// 0 none; 1 blend: a share `a` at scale `b`; 2 classes: `a` rail-leaning (rail cost x `b`),
    /// `a` never rail; 3 floor: rail at least `a` and at most 1 - `a` where it exists;
    /// 4 ratio: a share `a` choosing on log cost at `b`
    tail: u8,
    a: f32,
    b: f32,
}
const fn v(nocar: u8, home_f: f32, tail: u8, a: f32, b: f32) -> Var {
    Var { nocar, home_f, tail, a, b }
}
const VARS: [(&str, Var); 20] = [
    ("now", v(0, 0.0, 0, 0.0, 0.0)),
    ("nocar0", v(1, 0.0, 0, 0.0, 0.0)),
    ("flat", v(3, 0.0, 0, 0.0, 0.0)),
    ("curve", v(2, 0.0, 0, 0.0, 0.0)),
    ("home0.25", v(1, 0.25, 0, 0.0, 0.0)),
    ("home0.5", v(1, 0.5, 0, 0.0, 0.0)),
    ("home0.75", v(1, 0.75, 0, 0.0, 0.0)),
    ("home1", v(1, 1.0, 0, 0.0, 0.0)),
    ("home1.5", v(1, 1.5, 0, 0.0, 0.0)),
    ("home2", v(1, 2.0, 0, 0.0, 0.0)),
    ("h.5+blend5/.025", v(1, 0.5, 1, 0.05, 0.025)),
    ("h.5+blend10/.025", v(1, 0.5, 1, 0.10, 0.025)),
    ("h.5+blend10/.02", v(1, 0.5, 1, 0.10, 0.02)),
    ("h.5+classes5/.5", v(1, 0.5, 2, 0.05, 0.5)),
    ("h.5+classes10/.5", v(1, 0.5, 2, 0.10, 0.5)),
    ("h.5+floor2", v(1, 0.5, 3, 0.02, 0.0)),
    ("h.5+floor5", v(1, 0.5, 3, 0.05, 0.0)),
    ("h.5+ratio10/2", v(1, 0.5, 4, 0.10, 2.0)),
    ("h.5+ratio10/3", v(1, 0.5, 4, 0.10, 3.0)),
    ("h1+blend10/.025", v(1, 1.0, 1, 0.10, 0.025)),
];
const NV: usize = VARS.len();

fn curve(d: f32) -> f32 {
    0.75 / (1.0 + (11500.0 / d.max(1.0)).powf(1.9))
}

/// Home-end parking, perceived minutes a car trip: the destination's paid-space rule
/// (`park_parts`) on people per km2, times `f`.
fn home_park(p: &Params, d: f32, f: f32) -> f32 {
    if f == 0.0 || d <= p.park_paid_d0 {
        return 0.0;
    }
    f * (p.park_paid_min * (d / p.park_paid_d0).ln()).min(p.park_paid_cap_min)
}

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

/// One trip's shares [rail, walk, car, other] under variant `v`. `home` the home-end parking,
/// `cu` the share without a car.
#[allow(clippy::too_many_arguments)]
fn shares(p: &Params, v: &Var, rail: f32, cw: f32, drive: f32, park: (f32, f32, f32), home: f32, cu: f32, rk: f32) -> [f64; 4] {
    let (fixed, paid, free) = park;
    let road = p.car_drive(drive, rk) + fixed + home;
    let bt = p.mode_beta(road);
    let base = drivers(p, bt, road, paid, free, rail, cw);
    let m: [f64; 3] = match v.tail {
        1 => {
            let t = drivers(p, bt * v.b / p.beta, road, paid, free, rail, cw);
            let a = v.a as f64;
            std::array::from_fn(|k| (1.0 - a) * base[k] + a * t[k])
        }
        2 => {
            let lean = drivers(p, bt, road, paid, free, rail * v.b, cw);
            let never = drivers(p, bt, road, paid, free, 1e6, cw);
            let a = v.a as f64;
            std::array::from_fn(|k| (1.0 - 2.0 * a) * base[k] + a * lean[k] + a * never[k])
        }
        3 if rail < 1e5 => {
            let e = v.a as f64;
            let r = e + (1.0 - 2.0 * e) * base[0];
            let s = if base[0] < 1.0 { (1.0 - r) / (1.0 - base[0]) } else { 0.0 };
            [r, base[1] * s, base[2] * s]
        }
        4 => {
            let g = v.b as f64;
            let car = (road + (1.0 - free) * paid) as f64;
            let er = if rail < 1e5 { (car / rail.max(1.0) as f64).powf(g) } else { 0.0 };
            let ew = if cw < NO_WALK { (car / (cw - p.walk_bonus_min).max(1.0) as f64).powf(g) } else { 0.0 };
            let d = 1.0 + er + ew;
            let a = v.a as f64;
            let t = [er / d, ew / d, 1.0 / d];
            std::array::from_fn(|k| (1.0 - a) * base[k] + a * t[k])
        }
        _ => base,
    };
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
fn zone_homes(z: &Zones, pop: &[f32], nocar: Option<&Vec<f32>>) -> (Vec<f32>, Vec<f32>) {
    let mut pp = vec![0f64; z.n];
    let mut nc = vec![0f64; z.n];
    for c in 0..pop.len() {
        let i = z.cell_zone[c] as usize;
        pp[i] += pop[c] as f64;
        if let Some(n) = nocar {
            nc[i] += pop[c] as f64 * n[c] as f64;
        }
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
    // people, not the commute's home weights (`City::pop`)
    let pop: Vec<f32> = pack::cell_weights(&h, &b, "pop", api.city.len()).unwrap().unwrap();
    let mut acc = Acc(vec![[0.0; 5]; NV * NS * NG]);
    let mut mean_nc = 0.0f64;
    let mut pop_all = 0.0f64;
    if let Some(n) = &api.base.nocar {
        for c in 0..pop.len() {
            mean_nc += pop[c] as f64 * n[c] as f64;
            pop_all += pop[c] as f64;
        }
    }
    let flat = if pop_all > 0.0 { (mean_nc / pop_all) as f32 } else { 0.0 };
    let cu_of = |v: &Var, now: f32, dens: f32| match v.nocar {
        0 => now,
        2 => curve(dens),
        3 => flat,
        _ => 0.0,
    };
    // ---- commutes, by zone pair
    {
        let (z, gr, p) = (&*api.z, &*api.gr, &api.base);
        let (_, dtab) = decay_table(z.w, z.h, z.zone_m, 1.0, 0.0);
        let place: Vec<(f32, f32)> = z.density().iter().map(|&x| p.place_min_per_km(x)).collect();
        let park: Vec<(f32, f32, f32)> = (0..z.n).map(|j| p.park_parts(z, j)).collect();
        let (dres, cz) = zone_homes(z, &pop, p.nocar.as_deref());
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
                    let rail = rail_cost(sc, p, d_km * 1000.0);
                    for (vi, (_, v)) in VARS.iter().enumerate() {
                        let sh = shares(p, v, rail, cw, drive, park[j], home_park(p, dres[i], v.home_f), cu_of(v, cz[i], dres[i]), d_km * f);
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
        let p = ev.ep.clone();
        let place: Vec<(f32, f32)> = pz.density().iter().map(|&x| p.place_min_per_km(x)).collect();
        let park: Vec<(f32, f32, f32)> = (0..pz.n).map(|j| p.park_parts(pz, j)).collect();
        let (dres, _) = zone_homes(pz, &pop, None);
        let nocar = p.nocar.clone();
        for k in 0..ev.destinations.len() {
            let Some((_, c, dd, w, _)) = ev.edge(k) else { continue };
            let dist = ev.distance(c, dd);
            let (hz, az) = (pz.cell_zone[c] as usize, pz.cell_zone[dd] as usize);
            let cu_now = nocar.as_ref().map_or(0.0, |n| n[c]);
            let g = group(dres[hz]);
            let cw = p.walk_trip_cost(dist * ev.walk_factor(c, dd) / 1000.0);
            for (s, (_, sc)) in SCENS.iter().enumerate() {
                let rail = rail_cost(sc, &p, dist);
                for (u, v2) in [(hz, az), (az, hz)] {
                    let f = p.drive_factor(u, v2);
                    let drive = p.drive_min_f(dist, f, place[u], place[v2]);
                    for (vi, (_, v)) in VARS.iter().enumerate() {
                        let sh = shares(&p, v, rail, cw, drive, park[v2], home_park(&p, dres[hz], v.home_f), cu_of(v, cu_now, dres[hz]), dist * 0.001 * f);
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
        "flat": flat,
        "acc": acc.0.iter().map(|a| a.to_vec()).collect::<Vec<_>>(),
    });
    std::fs::write(&out_path, serde_json::to_string(&out).unwrap()).unwrap();
}
