//! T-551: the ceiling test, the real network's result by distance, the car as modelled, and a few
//! example trips (notes/T-551.md). Compiled only in the private copy `t551_copy.py` makes, as a
//! test module inside `everyday` (it reads the layer's samples); the shared crate never sees it.
//!
//!   T551_CITY=nyc T551_OUT=out.json cargo test --release --features demand --lib t551 -- --ignored --nocapture
//!
//! Every choice below goes through the model's own `Params` expressions (`drive_min_f`,
//! `park_parts`, `mode_beta`, `rail_share_parked`, `nocar_choice`, the walk shares); only the
//! rail cost of the two ceiling scenarios is made up, as the task asks.

use super::*;
use crate::demand::api::{DemandApi, FARE_MIN_PER_USD};
use crate::demand::network::PERIODS;
use crate::demand::pack;
use serde_json::{json, Value};
use std::cell::RefCell;

thread_local! {
    /// (zones, rail trips by home zone x work zone) of the commute pass running now.
    static HOOK: RefCell<(usize, Vec<f64>)> = RefCell::new((0, vec![]));
}

/// Called from the copy's `kernel::mode_choice` for each pair lane's rail trips.
pub(crate) fn hook(i: usize, j: usize, x: f32) {
    HOOK.with(|h| {
        let mut h = h.borrow_mut();
        let n = h.0;
        if n > 0 {
            h.1[i * n + j] += x as f64;
        }
    });
}
fn hook_start(n: usize) {
    HOOK.with(|h| {
        let mut h = h.borrow_mut();
        h.0 = n;
        h.1.clear();
        h.1.resize(n * n, 0.0);
    });
}
fn hook_take() -> Vec<f64> {
    HOOK.with(|h| {
        let mut h = h.borrow_mut();
        h.0 = 0;
        std::mem::take(&mut h.1)
    })
}

const BANDS_KM: [f32; 5] = [2.0, 5.0, 10.0, 20.0, 50.0];
const NB: usize = 6;
/// Commutes, then the everyday purposes in `KINDS` order.
const NL: usize = 7;
fn band(d_km: f32) -> usize {
    BANDS_KM.iter().position(|&b| d_km < b).unwrap_or(NB - 1)
}

/// Day trips, reserved walks and rail trips by layer and band.
#[derive(Clone)]
struct Acc(Vec<[f64; 3]>);
impl Acc {
    fn new() -> Acc {
        Acc(vec![[0.0; 3]; NL * NB])
    }
    fn add(&mut self, l: usize, b: usize, t: f64, w: f64, r: f64) {
        let a = &mut self.0[l * NB + b];
        a[0] += t;
        a[1] += w;
        a[2] += r;
    }
    fn json(&self) -> Value {
        json!((0..NL).map(|l| (0..NB).map(|b| self.0[l * NB + b].to_vec()).collect::<Vec<_>>()).collect::<Vec<_>>())
    }
}

/// A made-up direct train door to door: average speed, a walk at each end (real minutes), a
/// headway (0: no wait), a fare.
#[derive(Clone, Copy)]
struct Scen {
    kmh: f32,
    walk_min: f32,
    headway: f32,
    base_usd: f32,
    km_usd: f32,
}
const SCENS: [(&str, Scen); 2] = [
    ("a", Scen { kmh: 140.0, walk_min: 0.0, headway: 0.0, base_usd: 0.0, km_usd: 0.0 }),
    ("b", Scen { kmh: 100.0, walk_min: 4.0, headway: 5.0, base_usd: 1.0, km_usd: 0.02 }),
];

/// Its perceived minutes over `d_m` straight metres, weighed as the model weighs each part: the
/// walk as `access_cost`'s, the wait as `board_cost`'s first boarding (no transfer penalty, as
/// `skims` takes it off), the fare at `FARE_MIN_PER_USD`.
fn rail_cost(s: &Scen, p: &Params, d_m: f32) -> f32 {
    let km = d_m / 1000.0;
    let walk = s.walk_min.min(p.walk_easy_min) * p.w_walk + (s.walk_min - p.walk_easy_min).max(0.0) * p.w_walk_far;
    let wait = if s.headway > 0.0 { board_cost(s.headway, p) - p.transfer_min } else { 0.0 };
    km / s.kmh * 60.0 + 2.0 * walk + wait + (s.base_usd + s.km_usd * km) * FARE_MIN_PER_USD
}

/// Rail's share of the trips that do not walk, as `mode_choice`'s lanes and `pass` compute it:
/// the car's perceived minutes `drive_min x w_drive + fixed` (+ `car_km` a road km, a lever the
/// model does not have), the scale damped on them, the parked mix, and those without a car.
fn rail_share(p: &Params, rail: f32, drive_min: f32, park: (f32, f32, f32), cu: f32, rk: f32, car_km: f32) -> f32 {
    let (fixed, paid, free) = park;
    let road = drive_min * p.w_drive + fixed + car_km * rk;
    let bt = p.mode_beta(road);
    let r_car = p.rail_share_parked(bt * (road - rail) + p.asc_rail, bt, paid, free);
    if cu > 0.0 {
        let (r_n, _) = p.nocar_choice(rk * p.walk_trip_min(), rk * p.other_min_km() + p.other_wait(), Some(rail));
        r_car + cu * (r_n - r_car)
    } else {
        r_car
    }
}

/// The car as modelled, by band over trips that do not walk: [trips, drive minutes, road km,
/// straight km, parking perceived (mean over drivers), of which paid space, cordon].
#[derive(Clone)]
struct Car(Vec<[f64; 7]>);
impl Car {
    fn add(&mut self, l: usize, b: usize, t: f64, v: [f64; 6]) {
        let a = &mut self.0[l * NB + b];
        a[0] += t;
        for k in 0..6 {
            a[k + 1] += t * v[k];
        }
    }
}

struct Lever {
    name: &'static str,
    set: &'static [(&'static str, f32)],
    /// Car cost a road km, perceived minutes (ceiling only: the model has no such term).
    car_km: f32,
}
const LEVERS: [Lever; 9] = [
    Lever { name: "base", set: &[], car_km: 0.0 },
    // $10 a day of car ownership over a day's two commute trips: $5 a trip at 3 minutes a dollar
    Lever { name: "own5", set: &[("park_min", 18.0)], car_km: 0.0 },
    // the same over three trips a day
    Lever { name: "own3", set: &[("park_min", 13.0)], car_km: 0.0 },
    // a running cost of $0.15 a road km (ceiling only)
    Lever { name: "km15", set: &[], car_km: 0.45 },
    // every road speed 25% faster
    Lever { name: "fast25", set: &[("road_city_kmh", 18.75), ("road_town_kmh", 50.0), ("road_far_kmh", 100.0), ("road_open_kmh", 125.0)], car_km: 0.0 },
    // no paid parking anywhere (every driver still pays the 3 minutes and any cordon)
    Lever { name: "nopaid", set: &[("park_paid_min", 0.0)], car_km: 0.0 },
    // paid parking from lower densities: from 1,500 jobs a km2, half free at 3,000
    Lever { name: "parkwide", set: &[("park_paid_d0", 1500.0), ("park_free_d", 3000.0)], car_km: 0.0 },
    // walks to stations weigh 1.0 a minute instead of 1.5
    Lever { name: "walk10", set: &[("w_walk", 1.0), ("w_walk_far", 1.0)], car_km: 0.0 },
    // rail propensity +0.5 (commutes and everyday alike)
    Lever { name: "asc05", set: &[("asc_rail", 0.9)], car_km: 0.0 },
];

fn arr_f32(v: &Value, k: &str) -> Vec<f32> {
    v[k].as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect()
}
fn arr_u32(v: &Value, k: &str) -> Vec<u32> {
    v[k].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect()
}

/// Example trips for the breakdown (lon, lat home; lon, lat destination).
fn examples(city: &str) -> Vec<(&'static str, [f64; 4])> {
    match city {
        "nyc" => vec![
            ("Hempstead to Midtown", [-73.6187, 40.7062, -73.9772, 40.7527]),
            ("Mineola to Hicksville", [-73.6402, 40.7470, -73.5287, 40.7676]),
            ("Morristown to Newark", [-74.4815, 40.7968, -74.1644, 40.7347]),
            ("White Plains to Stamford", [-73.7629, 41.0340, -73.5422, 41.0466]),
            ("Jamaica to Downtown Brooklyn", [-73.8070, 40.7020, -73.9903, 40.6928]),
            ("Paramus to Midtown", [-74.0754, 40.9445, -73.9772, 40.7527]),
        ],
        "lon" => vec![
            ("Croydon to Bank", [-0.0982, 51.3762, -0.0886, 51.5133]),
            ("Watford to Oxford Circus", [-0.3903, 51.6565, -0.1419, 51.5152]),
            ("Romford to Stratford", [0.1832, 51.5751, -0.0034, 51.5416]),
            ("Kingston to Croydon", [-0.3007, 51.4123, -0.0982, 51.3762]),
            ("Bromley to Canary Wharf", [0.0145, 51.4060, -0.0235, 51.5054]),
        ],
        _ => vec![],
    }
}

#[test]
#[ignore]
fn t551() {
    let city = std::env::var("T551_CITY").unwrap_or("nyc".into());
    let out_path = std::env::var("T551_OUT").expect("T551_OUT");
    let only: Option<String> = std::env::var("T551_LEVERS").ok();
    let rounds = 3;
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/..");
    // the copy lives outside the tree: the data root comes in by name
    let data = std::env::var("T551_DATA").unwrap_or(format!("{root}/data"));
    let net_path = format!("{data}/work/realnet/{city}_real.json");
    let net: Value = serde_json::from_str(&std::fs::read_to_string(&net_path).unwrap()).unwrap();
    let pack_path = format!("{data}/packs/{}.json", net["pack"].as_str().unwrap());
    let (h, b) = pack::load_raw(&pack_path).unwrap();
    let hdr: Value = serde_json::from_str(&h).unwrap();
    let (lon0, lat0) = (hdr["origin"]["lon"].as_f64().unwrap(), hdr["origin"]["lat"].as_f64().unwrap());
    let d = &net["demand"];
    let (st_xy, line_n, stops, times, tph, cars) = (arr_f32(d, "st_xy"), arr_u32(d, "line_n"), arr_u32(d, "stops"), arr_f32(d, "times"), arr_f32(d, "tph"), arr_f32(d, "cars"));
    let mut hop_km = vec![];
    let mut off = 0usize;
    for &n in &line_n {
        let n = n as usize;
        for dir in 0..2 {
            for i in 0..n {
                let k = if dir == 0 { off + i } else { off + n - 1 - i };
                hop_km.push(if i + 1 < n {
                    let nx = if dir == 0 { k + 1 } else { k - 1 };
                    let (s0, s1) = (stops[k] as usize, stops[nx] as usize);
                    (st_xy[2 * s0] - st_xy[2 * s1]).hypot(st_xy[2 * s0 + 1] - st_xy[2 * s1 + 1]) / 1000.0
                } else {
                    0.0
                });
            }
        }
        off += n;
    }

    let mut out = serde_json::Map::new();
    let mut checks = vec![];
    for lever in LEVERS.iter() {
        if let Some(o) = &only {
            if !o.split(',').any(|x| x == lever.name) {
                continue;
            }
        }
        let t0 = std::time::Instant::now();
        let mut api = DemandApi::new(&h, &b).unwrap();
        for &(k, v) in lever.set {
            api.params_mut().set_extra(k, v).unwrap();
        }
        if !lever.set.is_empty() {
            api.refresh_walk();
        }
        let mut ev = EverydayApi::beside(&api, &h, &b).unwrap();

        // ---- the two ceilings, and the car as modelled (no network needed)
        let mut ceil = vec![Acc::new(), Acc::new()];
        let mut car = Car(vec![[0.0; 7]; NL * NB]);
        {
            let (z, gr, p) = (&*api.z, &*api.gr, &api.base);
            let (_, dtab) = decay_table(z.w, z.h, z.zone_m, 1.0, 0.0);
            let place: Vec<(f32, f32)> = z.density().iter().map(|&x| p.place_min_per_km(x)).collect();
            let park: Vec<(f32, f32, f32)> = (0..z.n).map(|j| p.park_parts(z, j)).collect();
            let mut cz = vec![0f32; z.n];
            if let Some(nc) = &p.nocar {
                for c in 0..api.city.len() {
                    let i = z.cell_zone[c] as usize;
                    if z.pop[i] > 0.0 {
                        cz[i] += (api.city.pop[c] as f64 * nc[c] as f64 / z.pop[i]) as f32;
                    }
                }
            }
            let cord = |j: usize| p.cordon.as_ref().and_then(|c| c.get(j)).map_or(0.0, |&c| c);
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
                    let d_km = dtab[(base + gr.key(z, j)) as usize];
                    let pw = gr.walk_share_pair(z, i, j) as f64;
                    let f = p.drive_factor(i, j);
                    let drive = p.drive_min_f(d_km * 1000.0, f, place[i], place[j]);
                    let rk = d_km * f;
                    let bd = band(d_km);
                    for (s, (_, sc)) in SCENS.iter().enumerate() {
                        let r = rail_share(p, rail_cost(sc, p, d_km * 1000.0), drive, park[j], cz[i], rk, lever.car_km) as f64;
                        ceil[s].add(0, bd, t, t * pw, t * (1.0 - pw) * r);
                    }
                    let (fx, paid, free) = park[j];
                    car.add(0, bd, t * (1.0 - pw), [drive as f64, rk as f64, d_km as f64, (fx + (1.0 - free) * paid) as f64, ((1.0 - free) * paid) as f64, cord(j) as f64]);
                }
            }
            // the everyday samples, both directions, on the layer's own zones and constant
            let eng = &ev.engine;
            let pz = &*eng.z;
            let mut p = eng.base.clone();
            p.asc_rail += p.everyday_asc_rail;
            let place: Vec<(f32, f32)> = pz.density().iter().map(|&x| p.place_min_per_km(x)).collect();
            let park: Vec<(f32, f32, f32)> = (0..pz.n).map(|j| p.park_parts(pz, j)).collect();
            let cz = &pz.cell_zone;
            let nocar = p.nocar.clone();
            for k in 0..ev.destinations.len() {
                let Some((kind, c, dd, w, pw)) = ev.edge(k) else { continue };
                let dist = ev.distance(c, dd);
                let (hz, az) = (cz[c] as usize, cz[dd] as usize);
                let cu = nocar.as_ref().map_or(0.0, |n| n[c]);
                let (w, pw) = (w as f64, pw as f64);
                let bd = band(dist / 1000.0);
                for (s, (_, sc)) in SCENS.iter().enumerate() {
                    let rail = rail_cost(sc, &p, dist);
                    let mut r = 0.0;
                    for (u, v) in [(hz, az), (az, hz)] {
                        let f = p.drive_factor(u, v);
                        let drive = p.drive_min_f(dist, f, place[u], place[v]);
                        r += rail_share(&p, rail, drive, park[v], cu, dist * 0.001 * f, lever.car_km) as f64;
                    }
                    // each direction carries the sample's trips once a day (OUTWARD, RETURN sum to 1)
                    ceil[s].add(1 + kind, bd, 2.0 * w, 2.0 * w * pw, w * (1.0 - pw) * r);
                }
                for (u, v) in [(hz, az), (az, hz)] {
                    let f = p.drive_factor(u, v);
                    let drive = p.drive_min_f(dist, f, place[u], place[v]);
                    let (fx, paid, free) = park[v];
                    let cordon = p.cordon.as_ref().and_then(|c| c.get(v)).map_or(0.0, |&c| c);
                    car.add(1 + kind, bd, w * (1.0 - pw), [drive as f64, (dist * 0.001 * f) as f64, (dist * 0.001) as f64, (fx + (1.0 - free) * paid) as f64, ((1.0 - free) * paid) as f64, cordon as f64]);
                }
            }
        }
        let t_ceil = t0.elapsed().as_secs_f64();

        // ---- the real network (as `bin/calib.rs`: a weekday, 3 crowding rounds, starting fares)
        let mut real = Acc::new();
        let mut ex_out = vec![];
        let skip_real = lever.car_km != 0.0;
        if !skip_real {
            api.set_network(&st_xy, &line_n, &stops, &times, &tph, &cars).unwrap();
            ev.set_network(&st_xy, &line_n, &stops, &times, &tph, &cars).unwrap();
            api.set_fares(1.5, 0.1, &hop_km).unwrap();
            ev.set_fares(1.5, 0.1, &hop_km).unwrap();
            let n = api.z.n;
            let mut c_rail = vec![0f64; n * n];
            let mut e_rail: Vec<f64> = vec![0.0; ev.destinations.len()];
            let (mut sum_c, mut sum_e) = ([0f64; 2], [0f64; 2]);
            for q in 0..PERIODS {
                hook_start(n);
                api.solve(q);
                let mut avg = hook_take();
                ev.solve(q);
                for k in 1..=rounds {
                    let nr = api.seg(q).len();
                    let mut total = vec![0f32; 3 * nr];
                    let mut put = |v: Vec<f32>, at: usize| {
                        for (t, x) in total[at * nr..(at + 1) * nr].iter_mut().zip(v) {
                            *t += x;
                        }
                    };
                    put(api.seg(q), 0);
                    put(api.board(q), 1);
                    put(api.alight(q), 2);
                    put(ev.seg(q), 0);
                    put(ev.board(q), 1);
                    put(ev.alight(q), 2);
                    hook_start(n);
                    assert!(api.crowd_with_external(q, &total));
                    let m = hook_take();
                    // a skipped round (settled) leaves the average as it was (`crowd_with_external`)
                    if m.iter().any(|&x| x != 0.0) {
                        let w = 1.0 / (k as f64 + 1.0);
                        for (a, x) in avg.iter_mut().zip(&m) {
                            *a += (x - *a) * w;
                        }
                    }
                    assert!(ev.crowd_with_external(q, &total));
                }
                for (a, x) in c_rail.iter_mut().zip(&avg) {
                    *a += x;
                }
                sum_c[0] += avg.iter().sum::<f64>();
                sum_c[1] += api.summary(q)[2];
                // everyday: each sample's rail, as `cell_modes` disaggregates a pair's
                let st = ev.per[q].as_ref().unwrap();
                for k in 0..ev.destinations.len() {
                    let Some((kind, _, _, w, pw)) = ev.edge(k) else { continue };
                    let pi = if ev.edge_pair.is_empty() { NONE } else { ev.edge_pair[k] };
                    if pi == NONE {
                        continue;
                    }
                    let pr = st.probability[pi as usize];
                    let sc = ev.edge_motorized_scale(pi, pw) as f64;
                    e_rail[k] += w as f64 * (OUTWARD[kind][q] as f64 * pr[0] as f64 + RETURN[kind][q] as f64 * pr[1] as f64) * sc;
                }
                sum_e[1] += ev.summary(q)[2];
            }
            sum_e[0] = e_rail.iter().sum();
            checks.push(json!({"lever": lever.name, "commute_rail": sum_c, "everyday_rail": sum_e}));
            eprintln!("{} {}: commute rail by pair {:.0} against the summary's {:.0}; everyday {:.0} against {:.0}", city, lever.name, sum_c[0], sum_c[1], sum_e[0], sum_e[1]);
            {
                let (z, gr) = (&*api.z, &*api.gr);
                let (_, dtab) = decay_table(z.w, z.h, z.zone_m, 1.0, 0.0);
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
                        let d_km = dtab[(base + gr.key(z, j)) as usize];
                        let pw = gr.walk_share_pair(z, i, j) as f64;
                        real.add(0, band(d_km), t, t * pw, c_rail[i * n + j]);
                    }
                }
            }
            for k in 0..ev.destinations.len() {
                let Some((kind, c, dd, w, pw)) = ev.edge(k) else { continue };
                let (w, pw) = (w as f64, pw as f64);
                real.add(1 + kind, band(ev.distance(c, dd) / 1000.0), 2.0 * w, 2.0 * w * pw, e_rail[k]);
            }
            // ---- example trips, as the trip explorer gives them (a commute, morning peak)
            if lever.name == "base" {
                const R: f64 = 6371008.8;
                let kx = 180.0 / std::f64::consts::PI / R / lat0.to_radians().cos();
                let ky = 180.0 / std::f64::consts::PI / R;
                let xy = |lon: f64, lat: f64| (((lon - lon0) / kx) as f32, ((lat - lat0) / ky) as f32);
                for (name, [a, bb, c, dd]) in examples(&city) {
                    let (ox, oy) = xy(a, bb);
                    let (dx, dy) = xy(c, dd);
                    let v = api.trip(0, ox, oy, dx, dy);
                    // the destination zone's parking parts and jobs density
                    let cell = api.nearest_cell(dx, dy).map(|c| api.z.cell_zone[c] as usize);
                    let pk = cell.map(|j| {
                        let land = (api.z.ncell[j].max(1) as f32 * CELL_KM2).min((api.z.zone_m / 1000.0).powi(2));
                        let pp = api.base.park_parts(&api.z, j);
                        vec![pp.0 as f64, pp.1 as f64, pp.2 as f64, (api.z.jobs[j] as f32 / land) as f64]
                    });
                    // the check: this file's `rail_share` on the explorer's own rail cost and drive
                    // gives the explorer's (the model's) share
                    let mine = match (api.nearest_cell(ox, oy), api.nearest_cell(dx, dy), api.sz.as_ref()) {
                        (Some(o), Some(dc), Some(sz)) if v[13] == 1.0 => {
                            let (i, j) = (api.z.cell_zone[o] as usize, api.z.cell_zone[dc] as usize);
                            let p = &api.period(0).unwrap().p;
                            let rk = v[7] as f32 * 0.001 * p.drive_factor(i, j);
                            let cu = sz.nocar[sz.cell[o] as usize];
                            rail_share(p, v[14] as f32, v[9] as f32, p.park_parts(&api.z, j), cu, rk, 0.0) as f64
                        }
                        _ => -1.0,
                    };
                    ex_out.push(json!({"name": name, "trip": v, "park": pk, "check_r": mine}));
                }
            }
        }
        eprintln!("{} {}: ceiling {:.1} s, all {:.1} s", city, lever.name, t_ceil, t0.elapsed().as_secs_f64());
        out.insert(
            lever.name.into(),
            json!({
                "ceiling_a": ceil[0].json(),
                "ceiling_b": ceil[1].json(),
                "real": if skip_real { Value::Null } else { real.json() },
                "car": json!((0..NL).map(|l| (0..NB).map(|b| car.0[l * NB + b].to_vec()).collect::<Vec<_>>()).collect::<Vec<_>>()),
                "examples": ex_out,
                "asc": [api.base.asc_rail, api.base.asc_rail + api.base.everyday_asc_rail],
            }),
        );
        std::fs::write(&out_path, serde_json::to_string(&json!({"city": city, "bands_km": BANDS_KM, "layers": ["commute", "errands", "social", "school", "university", "airport", "medical"], "levers": out, "checks": checks})).unwrap()).unwrap();
    }
}
