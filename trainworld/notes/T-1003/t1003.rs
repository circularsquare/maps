//! T-1003: walking with no rail, broken down, and the ceiling tests A, B, T, per pack. Run from a
//! copy of the crate as `everyday::t1003`:
//!
//!   T1003_PACK=<pack .json> T1003_OUT=out.json [T1003_SET=k=v,..] <test exe> t1003::t1003 --ignored
//!
//! Output `acc[layer][band][group]` = [trips, walk trips (no rail, the model's own split),
//! sum d_km, sum walk cost, sum drive real min, sum car_drive, sum park fixed, sum paid x (1 -
//! free), sum home park, then per scenario A, B, T: rail, walk, car]. Layers commute, everyday;
//! bands of straight km; groups by home zone residents per km² of land.

use super::*;
use crate::demand::api::{DemandApi, FARE_MIN_PER_USD};
use crate::demand::pack;
use serde_json::json;

const BANDS: [f32; 6] = [1.0, 2.0, 3.0, 5.0, 10.0, 1e9];
const NB: usize = 6;
const GROUP_EDGES: [f32; 4] = [3000.0, 10000.0, 20000.0, 40000.0];
const NG: usize = 6; // 5 groups + all
const NV: usize = 9 + 9 + 1;

/// T1003_OTHER="base,per_km": an "other" way (bus, bike) at base + per_km x straight km perceived
/// minutes with no parking; the car side becomes the cheaper of it and the car, per driver kind.
fn other() -> Option<(f32, f32)> {
    static O: std::sync::OnceLock<Option<(f32, f32)>> = std::sync::OnceLock::new();
    *O.get_or_init(|| {
        let s = std::env::var("T1003_OTHER").ok()?;
        let (a, b) = s.split_once(',')?;
        Some((a.parse().ok()?, b.parse().ok()?))
    })
}

fn band(d_km: f32) -> usize {
    BANDS.iter().position(|&e| d_km < e).unwrap()
}
fn group(d: f32) -> usize {
    GROUP_EDGES.iter().position(|&e| d < e).unwrap_or(4)
}

struct Scen {
    kmh: f32,
    walk_min: f32,
    wait_min: f32,
    base_usd: f32,
    km_usd: f32,
}
const SCENS: [Scen; 3] = [
    Scen { kmh: 140.0, walk_min: 3.0, wait_min: 2.5, base_usd: 0.0, km_usd: 0.0 },
    Scen { kmh: 100.0, walk_min: 3.0, wait_min: 2.5, base_usd: 1.5, km_usd: 0.05 },
    Scen { kmh: 30.0, walk_min: 6.0, wait_min: 3.0, base_usd: 1.5, km_usd: 0.1 },
];

fn rail_cost(s: &Scen, p: &Params, d_m: f32) -> f32 {
    let km = d_m / 1000.0;
    let walk = s.walk_min.min(p.walk_easy_min) * p.w_walk + (s.walk_min - p.walk_easy_min).max(0.0) * p.w_walk_far;
    km / s.kmh * 60.0 + 2.0 * walk + s.wait_min + (s.base_usd + s.km_usd * km) * FARE_MIN_PER_USD
}

/// [rail, walk, car] under scenario s; `road` the free parker's car, walk cost cw
fn shares(p: &Params, s: &Scen, d_m: f32, cw: f32, road: f32, paid: f32, free: f32) -> [f64; 3] {
    let rail = rail_cost(s, p, d_m);
    let bt = p.mode_beta(road);
    let util = bt * (road - rail) + p.asc_rail;
    if cw >= NO_WALK {
        let r = p.car_choice(util, -1e9, bt, paid, free, rail, road) as f64;
        return [r, 0.0, 1.0 - r];
    }
    let uw = bt * (road - cw) + p.asc_walk();
    let r = p.car_choice(util, uw, bt, paid, free, rail, road) as f64;
    let e = fast_exp(uw - util);
    let w0 = e / (1.0 + fast_exp(-util) + e);
    let w1 = e / (1.0 + fast_exp(-(util + bt * paid)) + e);
    let w = if paid < 1e-30 { w0 } else { w0 + (1.0 - free) * (w1 - w0) } as f64;
    [r, w, 1.0 - r - w]
}

struct Acc(Vec<[f64; NV]>);
impl Acc {
    fn at(layer: usize, b: usize, g: usize) -> usize {
        (layer * NB + b) * NG + g
    }
    fn add(&mut self, layer: usize, b: usize, g: usize, v: &[f64; NV]) {
        for gg in [g, NG - 1] {
            let a = &mut self.0[Self::at(layer, b, gg)];
            for k in 0..NV {
                a[k] += v[k];
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn row(p: &Params, t: f64, w: f64, d_m: f32, cw: f32, drive_min: f32, cdrive: f32, park: (f32, f32, f32), home: f32) -> [f64; NV] {
    let (fixed, paid0, free) = park;
    let mut road = cdrive + fixed + home;
    let mut paid = paid0;
    if let Some((ob, ok)) = other() {
        let o = ob + ok * d_m / 1000.0;
        let r1 = road.min(o);
        paid = (road + paid0).min(o) - r1;
        road = r1;
    }
    let mut v = [0f64; NV];
    v[18] = t * p.walk_no_rail(road, paid, free, cw) as f64;
    v[0] = t;
    v[1] = t * w;
    v[2] = t * (d_m / 1000.0) as f64;
    v[3] = t * cw.min(500.0) as f64;
    v[4] = t * drive_min as f64;
    v[5] = t * cdrive as f64;
    v[6] = t * fixed as f64;
    v[7] = t * ((1.0 - free) * paid0) as f64;
    v[8] = t * home as f64;
    for (s, sc) in SCENS.iter().enumerate() {
        let sh = shares(p, sc, d_m, cw, road, paid, free);
        for k in 0..3 {
            v[9 + 3 * s + k] = t * sh[k];
        }
    }
    v
}

#[test]
#[ignore]
fn t1003() {
    let pack_path = std::env::var("T1003_PACK").expect("T1003_PACK");
    let out_path = std::env::var("T1003_OUT").expect("T1003_OUT");
    let (h, b) = pack::load_raw(&pack_path).unwrap();
    let mut api = DemandApi::new(&h, &b).unwrap();
    let mut set_any = false;
    if let Ok(s) = std::env::var("T1003_SET") {
        for kv in s.split(',').filter(|x| !x.is_empty()) {
            let (k, x) = kv.split_once('=').unwrap();
            api.params_mut().set_extra(k, x.parse().unwrap()).unwrap();
            set_any = true;
        }
    }
    if set_any {
        api.refresh_walk();
    }
    let ev = EverydayApi::beside(&api, &h, &b).unwrap();
    let n = api.city.len();
    let pop: Vec<f32> = pack::cell_weights(&h, &b, "pop", n).unwrap().unwrap();
    let mut acc = Acc(vec![[0.0; NV]; 2 * NB * NG]);
    // residents per km² of land by zone (as home parking reads them) and the zones' commuter density
    let zone_res = |z: &Zones| -> Vec<f32> {
        let mut pp = vec![0f64; z.n];
        for c in 0..pop.len() {
            pp[z.cell_zone[c] as usize] += pop[c] as f64;
        }
        (0..z.n).map(|i| pp[i] as f32 / (z.ncell[i].max(1) as f32 * CELL_KM2).min(z.square_km2(i))).collect()
    };
    let mut zinfo = vec![];
    {
        let (z, gr, p, nr) = (&*api.z, &*api.gr, &api.base, &*api.nr);
        let place: Vec<(f32, f32)> = z.density().iter().map(|&x| p.place_min_per_km(x)).collect();
        let park: Vec<(f32, f32, f32)> = (0..z.n).map(|j| p.park_parts(z, j)).collect();
        let dres = zone_res(z);
        let mut home = vec![0f32; z.n];
        for (c, &zi) in z.cell_zone.iter().enumerate() {
            home[zi as usize] = p.home_park_cell(c);
        }
        let dens = z.density();
        for i in 0..z.n {
            zinfo.push(json!([z.pop[i], z.jobs[i], dres[i], dens[i], park[i].1, park[i].2, home[i]]));
        }
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
                let d_km = gr.dist_km(k);
                let cw = gr.walk_cost_at(i, j, k);
                let w = if cw >= NO_WALK { 0.0 } else { nr.split(gr, p, i, j, k) as f64 };
                let f = p.drive_factor(i, j);
                let dm = p.drive_min_f(d_km * 1000.0, f, place[i], place[j]);
                let cd = p.car_drive(dm, d_km * f);
                let v = row(p, t, w, d_km * 1000.0, cw, dm, cd, park[j], home[i]);
                acc.add(0, band(d_km), g, &v);
            }
        }
    }
    {
        let pz = &*ev.engine.z;
        let p = ev.ep.clone();
        let dres = zone_res(pz);
        for k in 0..ev.destinations.len() {
            let Some((_, c, dd, w, pw)) = ev.edge(k) else { continue };
            let dist = ev.distance(c, dd);
            let (hz, az) = (pz.cell_zone[c] as usize, pz.cell_zone[dd] as usize);
            let g = group(dres[hz]);
            let cw = p.walk_trip_cost(dist * ev.walk_factor(c, dd) / 1000.0);
            let hp = p.home_park_cell(c);
            for (u, v2) in [(hz, az), (az, hz)] {
                let f = p.drive_factor(u, v2);
                let dm = p.drive_min_f(dist, f, ev.wplace[u], ev.wplace[v2]);
                let cd = p.car_drive(dm, dist * 0.001 * f);
                let v = row(&p, w as f64, pw as f64, dist, cw, dm, cd, ev.wpark[v2], hp);
                acc.add(1, band(dist / 1000.0), g, &v);
            }
        }
    }
    let out = json!({
        "pack": pack_path,
        "bands": BANDS, "groups": ["<3k", "3-10k", "10-20k", "20-40k", ">=40k", "all"],
        "vars": ["trips", "walk", "km", "walk_cost", "drive_min", "car_drive", "park_fixed", "park_paid", "home_park",
                 "A_rail", "A_walk", "A_car", "B_rail", "B_walk", "B_car", "T_rail", "T_walk", "T_car", "walk_own"],
        "acc": acc.0.iter().map(|a| a.to_vec()).collect::<Vec<_>>(),
        "zones": zinfo,
    });
    std::fs::write(&out_path, serde_json::to_string(&out).unwrap()).unwrap();
}
