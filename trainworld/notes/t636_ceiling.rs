//! T-636: Anita's two "maximum ridership" tests (A, B) and the real network (C), with walking a
//! mode in the choice (T-558), by purpose layer and distance band. Compiled only in the private
//! copy `t636_copy.py` makes, as a test module inside `everyday`; the shared crate never sees it.
//!
//!   T636_CITY=nyc T636_OUT=out.json T636_SET=k=v,k=v T636_DATA=<data> cargo test --release \
//!       --features demand --lib t636 -- --ignored --nocapture
//!
//! Every choice goes through the model's own `Params` expressions (`drive_min_f`, `car_drive`,
//! `park_parts`, `mode_beta`, the walk costs, `other_min_km`); only the made-up train's cost is
//! new. Shares are of all trips, walking included: rail, walk, car, other.

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

/// Day trips and the trips by rail, walking, car and other, by layer and band.
#[derive(Clone)]
struct Acc(Vec<[f64; 5]>);
impl Acc {
    fn new() -> Acc {
        Acc(vec![[0.0; 5]; NL * NB])
    }
    /// `t` trips with the four modes' shares.
    fn add(&mut self, l: usize, b: usize, t: f64, s: [f64; 4]) {
        let a = &mut self.0[l * NB + b];
        a[0] += t;
        for k in 0..4 {
            a[k + 1] += t * s[k];
        }
    }
    fn json(&self) -> Value {
        json!((0..NL).map(|l| (0..NB).map(|b| self.0[l * NB + b].to_vec()).collect::<Vec<_>>()).collect::<Vec<_>>())
    }
}

/// A made-up direct train door to door: average speed, a walk at each end (real minutes), a wait
/// (plain minutes), a fare.
#[derive(Clone, Copy)]
struct Scen {
    kmh: f32,
    walk_min: f32,
    wait_min: f32,
    base_usd: f32,
    km_usd: f32,
}
const SCENS: [(&str, Scen); 3] = [
    // Anita's A: 140 km/h, free
    ("A", Scen { kmh: 140.0, walk_min: 3.0, wait_min: 2.5, base_usd: 0.0, km_usd: 0.0 }),
    // her B: 100 km/h, $1.50 + $0.05 a km
    ("B", Scen { kmh: 100.0, walk_min: 3.0, wait_min: 2.5, base_usd: 1.5, km_usd: 0.05 }),
    // no rail at all, to check the walk and other shares against the model's own
    ("none", Scen { kmh: 0.0, walk_min: 0.0, wait_min: 0.0, base_usd: 0.0, km_usd: 0.0 }),
];

/// Its perceived minutes over `d_m` straight metres: the walks as `access_cost`'s, the wait as
/// `board_cost`'s first boarding (a plain half headway, no transfer penalty), the fare at
/// `FARE_MIN_PER_USD`.
fn rail_cost(s: &Scen, p: &Params, d_m: f32) -> f32 {
    if s.kmh == 0.0 {
        return 1e6;
    }
    let km = d_m / 1000.0;
    let walk = s.walk_min.min(p.walk_easy_min) * p.w_walk + (s.walk_min - p.walk_easy_min).max(0.0) * p.w_walk_far;
    km / s.kmh * 60.0 + 2.0 * walk + s.wait_min + (s.base_usd + s.km_usd * km) * FARE_MIN_PER_USD
}

/// One trip's mode shares [rail, walk, car, other], as the model's lanes compute them: drivers
/// (parking free, or paying for a space) choose among rail, walking the whole way and the car;
/// the share `cu` without a car among rail, walking and other. `rk` the road km, `cw` the walk's
/// cost (`NO_WALK`: not offered).
fn shares(p: &Params, rail: f32, cw: f32, drive: f32, park: (f32, f32, f32), cu: f32, rk: f32) -> [f64; 4] {
    let (fixed, paid, free) = park;
    let road = p.car_drive(drive, rk) + fixed;
    let bt = p.mode_beta(road);
    let util = (bt * (road - rail) + p.asc_rail) as f64;
    let walk = cw < NO_WALK;
    let uw = if walk { (bt * (road - cw) + p.asc_walk()) as f64 } else { 0.0 };
    let three = |sh: f64| {
        let er = (util + sh).exp();
        let ew = if walk { (uw + sh).exp() } else { 0.0 };
        let d = 1.0 + er + ew;
        [er / d, ew / d, 1.0 / d]
    };
    let (a, b) = (three(0.0), three((bt * paid) as f64));
    let f = free as f64;
    let m: Vec<f64> = (0..3).map(|k| f * a[k] + (1.0 - f) * b[k]).collect();
    let c = cu as f64;
    if cu <= 0.0 {
        return [m[0], m[1], m[2], 0.0];
    }
    let oth = rk * p.other_min_km() + p.other_wait();
    let bn = p.mode_beta(oth);
    let oa = oth + p.other_asc;
    let ur = (bn * (oa - rail) + p.asc_rail) as f64;
    let uwn = if walk { (bn * (oa - cw) + p.asc_walk()) as f64 } else { 0.0 };
    let (er, ew) = (ur.exp(), if walk { uwn.exp() } else { 0.0 });
    let d = 1.0 + er + ew;
    [(1.0 - c) * m[0] + c * er / d, (1.0 - c) * m[1] + c * ew / d, (1.0 - c) * m[2], c / d]
}

fn arr_f32(v: &Value, k: &str) -> Vec<f32> {
    v[k].as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect()
}
fn arr_u32(v: &Value, k: &str) -> Vec<u32> {
    v[k].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect()
}

#[test]
#[ignore]
fn t636() {
    let city = std::env::var("T636_CITY").unwrap_or("nyc".into());
    let out_path = std::env::var("T636_OUT").expect("T636_OUT");
    let sets = std::env::var("T636_SET").unwrap_or_default();
    let skip_real = std::env::var("T636_NOREAL").is_ok();
    let rounds = 3;
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/..");
    let data = std::env::var("T636_DATA").unwrap_or(format!("{root}/data"));
    let net_path = format!("{data}/work/realnet/{city}_real.json");
    let net: Value = serde_json::from_str(&std::fs::read_to_string(&net_path).unwrap()).unwrap();
    let pack_path = format!("{data}/packs/{}.json", net["pack"].as_str().unwrap());
    let (h, b) = pack::load_raw(&pack_path).unwrap();
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

    let mut api = DemandApi::new(&h, &b).unwrap();
    for kv in sets.split(',').filter(|x| !x.is_empty()) {
        let (k, v) = kv.split_once('=').unwrap();
        api.params_mut().set_extra(k, v.parse().unwrap()).unwrap();
    }
    if !sets.is_empty() {
        api.refresh_walk();
    }
    let mut ev = EverydayApi::beside(&api, &h, &b).unwrap();
    let no_rail_walk = api.info()[3];

    // ---- the ceilings, and the no-rail check (no network needed)
    let mut ceil = vec![Acc::new(), Acc::new(), Acc::new()];
    // walking with no rail at all, per commute zone pair and per everyday sample (summed over the two directions): rail then takes its share of it (T-558)
    let mut w0_pair: Vec<f32> = vec![];
    let mut w0_edge: Vec<f64> = vec![0.0; ev.destinations.len()];
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
        w0_pair.resize(z.n * z.n, 0.0);
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
                let bd = band(d_km);
                for (s, (_, sc)) in SCENS.iter().enumerate() {
                    let sh = shares(p, rail_cost(sc, p, d_km * 1000.0), cw, drive, park[j], cz[i], d_km * f);
                    ceil[s].add(0, bd, t, sh);
                    if s == 2 {
                        w0_pair[i * z.n + j] = sh[1] as f32;
                    }
                }
            }
        }
        // the everyday samples, both directions, on the layer's own zones and constants
        let pz = &*ev.engine.z;
        let p = ev.ep.clone();
        let place: Vec<(f32, f32)> = pz.density().iter().map(|&x| p.place_min_per_km(x)).collect();
        let park: Vec<(f32, f32, f32)> = (0..pz.n).map(|j| p.park_parts(pz, j)).collect();
        let cell_zone = &pz.cell_zone;
        let nocar = p.nocar.clone();
        for k in 0..ev.destinations.len() {
            let Some((kind, c, dd, w, _)) = ev.edge(k) else { continue };
            let dist = ev.distance(c, dd);
            let (hz, az) = (cell_zone[c] as usize, cell_zone[dd] as usize);
            let cu = nocar.as_ref().map_or(0.0, |n| n[c]);
            let w = w as f64;
            let bd = band(dist / 1000.0);
            let cw = p.walk_trip_cost(dist * ev.walk_factor(hz, az) / 1000.0);
            for (s, (_, sc)) in SCENS.iter().enumerate() {
                let rail = rail_cost(sc, &p, dist);
                for (u, v) in [(hz, az), (az, hz)] {
                    let f = p.drive_factor(u, v);
                    let drive = p.drive_min_f(dist, f, place[u], place[v]);
                    let sh = shares(&p, rail, cw, drive, park[v], cu, dist * 0.001 * f);
                    ceil[s].add(1 + kind, bd, w, sh);
                    if s == 2 {
                        w0_edge[k] += sh[1];
                    }
                }
            }
        }
    }
    eprintln!("{city} {sets}: ceilings done; model's no-rail commute walk share {:.4}", no_rail_walk);

    // ---- the real network (as `bin/calib.rs`: a weekday, 3 crowding rounds, starting fares)
    let mut real = Acc::new();
    let (mut day_c, mut day_e) = ([0f64; 4], [0f64; 4]);
    let mut chk = vec![];
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
            let s = api.summary(q);
            sum_c[1] += s[2];
            let es = ev.summary(q);
            for (i, k) in [1, 2, 3, 10].into_iter().enumerate() {
                day_c[i] += s[k];
                day_e[i] += es[k];
            }
            // everyday: each sample's rail, from its pair's probability
            let st = ev.per[q].as_ref().unwrap();
            for k in 0..ev.destinations.len() {
                let Some((kind, _, _, w, _)) = ev.edge(k) else { continue };
                let pi = if ev.edge_pair.is_empty() { NONE } else { ev.edge_pair[k] };
                if pi == NONE {
                    continue;
                }
                let pr = st.probability[pi as usize];
                e_rail[k] += w as f64 * (OUTWARD[kind][q] as f64 * pr[0] as f64 + RETURN[kind][q] as f64 * pr[1] as f64);
            }
            sum_e[1] += es[2];
        }
        sum_e[0] = e_rail.iter().sum();
        chk.push(json!({"commute_rail": sum_c, "everyday_rail": sum_e}));
        eprintln!("{city}: commute rail by pair {:.0} against the summary's {:.0}; everyday {:.0} against {:.0}", sum_c[0], sum_c[1], sum_e[0], sum_e[1]);
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
                let r = c_rail[i * n + j] / t;
                real.add(0, band(d_km), t, [r, w0_pair[i * n + j] as f64 * (1.0 - r), 0.0, 0.0]);
            }
        }
        for k in 0..ev.destinations.len() {
            let Some((kind, c, dd, w, _)) = ev.edge(k) else { continue };
            let r = e_rail[k] / (2.0 * w as f64);
            real.add(1 + kind, band(ev.distance(c, dd) / 1000.0), 2.0 * w as f64, [r, w0_edge[k] * w as f64 / (2.0 * w as f64) * (1.0 - r), 0.0, 0.0]);
        }
    }
    let pj = |p: &Params| format!("{:?}", (p.beta, p.asc_rail, p.country_asc_rail, p.asc_walk(), p.everyday_asc_rail, p.walk_bonus_min, p.other_asc, p.car_own_usd_trip, p.car_own_usd_km));
    let out = json!({
        "city": city, "set": sets, "bands_km": BANDS_KM,
        "layers": ["commute", "errands", "social", "school", "university", "airport", "medical"],
        "A": ceil[0].json(), "B": ceil[1].json(), "none": ceil[2].json(), "C": real.json(),
        "day_c": day_c, "day_e": day_e, "no_rail_walk_model": no_rail_walk, "checks": chk,
        "params": pj(&api.base), "params_everyday": pj(&ev.ep),
    });
    std::fs::write(&out_path, serde_json::to_string(&out).unwrap()).unwrap();
}
