//! T-558: walking as a third mode, measured with the model's own costs (notes/T-558.md).
//! Compiled only in the private copy `t558_copy.py` makes, as a test module inside `everyday`.
//!
//! Each commute lane (subzone pair offered rail) and each everyday pair hands its car and rail
//! costs to this module on the free-flow solve; the module evaluates today's choice (the walk
//! reserved by distance) and the proposals (walking in the choice) on the same costs.

use super::*;
use crate::demand::api::DemandApi;
use crate::demand::network::PERIODS;
use crate::demand::pack;
use serde_json::{json, Value};
use std::cell::RefCell;

const EDGES: [f32; 11] = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 10.0, 20.0, 50.0];
const NB: usize = 12;
const NL: usize = 9;
/// trips, walk, rail, car, other
const NM: usize = 5;
fn band(d_km: f32) -> usize {
    EDGES.iter().position(|&b| d_km < b).unwrap_or(NB - 1)
}

/// A form of the choice. `today`: the walk reserved by distance, as the model has it. Otherwise
/// walking competes: perceived walk minutes `w1` a real minute up to `t0`, `w2` after; its constant
/// `ac` (commutes) / `ae` (everyday) against the car (or other); nest parameter `lambda` (1: a
/// plain three-way logit; towards 0 walking ignores rail).
#[derive(Clone, Copy, Debug)]
struct Var {
    today: bool,
    lambda: f64,
    w1: f32,
    t0: f32,
    w2: f32,
    ac: f32,
    ae: f32,
}
impl Var {
    fn from_json(v: &Value) -> Var {
        let g = |k: &str, d: f64| v.get(k).and_then(|x| x.as_f64()).unwrap_or(d);
        Var { today: false, lambda: g("l", 1.0), w1: g("w1", 1.5) as f32, t0: g("t0", 10.0) as f32, w2: g("w2", 4.0) as f32, ac: g("ac", 2.0) as f32, ae: g("ae", 2.0) as f32 }
    }
    fn json(&self) -> Value {
        json!({"today": self.today, "l": self.lambda, "w1": self.w1, "t0": self.t0, "w2": self.w2, "ac": self.ac, "ae": self.ae})
    }
}
const TODAY: Var = Var { today: true, lambda: 0.0, w1: 0.0, t0: 0.0, w2: 0.0, ac: 0.0, ae: 0.0 };

/// The walk distance (straight km times the walk factor round water) from the reserved share.
fn walk_km(pw: f32, d0: f32, s: f32) -> f32 {
    if pw <= 0.0 {
        return f32::INFINITY;
    }
    d0 + s * ((1.0 - pw) / pw).ln()
}

/// Shares [walk, rail, car, other] of one trip.
#[allow(clippy::too_many_arguments)]
fn choose(p: &Params, v: &Var, everyday: bool, pw: f32, road: f32, paid: f32, free: f32, bt: f32, rail: Option<f32>, rail_n: Option<f32>, cu: f32, rd: f32) -> [f64; 4] {
    let (kw, ko, kow) = (p.walk_trip_min(), p.other_min_km(), p.other_wait());
    if v.today {
        let m = (1.0 - pw) as f64;
        let r_car = rail.map_or(0.0, |r| p.rail_share_parked(bt * (road - r) + p.asc_rail, bt, paid, free)) as f64;
        let (rn, q) = if cu > 0.0 { p.nocar_choice(rd * kw, rd * ko + kow, rail_n) } else { (0.0, 0.0) };
        let (rn, q, cu) = (rn as f64, q as f64, cu as f64);
        return [pw as f64 + m * cu * (1.0 - rn) * (1.0 - q), m * ((1.0 - cu) * r_car + cu * rn), m * (1.0 - cu) * (1.0 - r_car), m * cu * (1.0 - rn) * q];
    }
    let d0 = if everyday { p.everyday_walk_d0_km } else { p.walk_d0_km };
    let dw = walk_km(pw, d0, p.walk_s_km);
    let t = dw * p.walk_detour * 1000.0 / p.walk_mps / 60.0;
    let cw = if t.is_finite() { (v.w1 * t.min(v.t0) + v.w2 * (t - v.t0).max(0.0)) as f64 } else { f64::INFINITY };
    let asc_w = (if everyday { v.ae } else { v.ac }) as f64;
    let nest = |ur: Option<f64>, uw: f64| -> (f64, f64, f64) {
        let er = ur.map_or(0.0, |u| u.min(60.0).exp());
        let s = 1.0 + er;
        let ew = if uw.is_finite() { uw.min(60.0).exp() } else { 0.0 };
        let sl = if v.lambda == 1.0 { s } else { s.powf(v.lambda) };
        let pw = ew / (ew + sl);
        (pw, (1.0 - pw) * er / s, (1.0 - pw) / s)
    };
    let mut o = [0.0f64; 4];
    let cu = cu as f64;
    if cu < 1.0 {
        let bt = bt as f64;
        let cls: [(f64, f64); 2] = if paid < 1e-30 { [(road as f64, 1.0), (0.0, 0.0)] } else { [(road as f64, free as f64), ((road + paid) as f64, 1.0 - free as f64)] };
        for (c, wt) in cls {
            if wt <= 0.0 {
                continue;
            }
            let ur = rail.map(|r| bt * (c - r as f64) + p.asc_rail as f64);
            let (a, b, k) = nest(ur, bt * (c - cw) + asc_w);
            o[0] += (1.0 - cu) * wt * a;
            o[1] += (1.0 - cu) * wt * b;
            o[2] += (1.0 - cu) * wt * k;
        }
    }
    if cu > 0.0 {
        let oth = rd * ko + kow;
        let oa = (oth + p.other_asc) as f64;
        let btn = p.mode_beta(oth) as f64;
        let ur = rail_n.map(|r| btn * (oa - r as f64) + p.asc_rail as f64);
        let (a, b, k) = nest(ur, btn * (oa - cw) + asc_w);
        o[0] += cu * a;
        o[1] += cu * b;
        o[3] += cu * k;
    }
    o
}

#[derive(Clone)]
struct Acc {
    nv: usize,
    a: Vec<f64>,
}
impl Acc {
    fn new(nv: usize) -> Acc {
        Acc { nv, a: vec![0.0; nv * NL * NB * NM] }
    }
    #[inline]
    fn add(&mut self, var: usize, l: usize, b: usize, t: f64, s: &[f64; 4], sign: f64) {
        let i = ((var * NL + l) * NB + b) * NM;
        self.a[i] += sign * t;
        for m in 0..4 {
            self.a[i + 1 + m] += sign * t * s[m];
        }
    }
    /// `add`, and again in layer 7 (commutes) or 8 (everyday) where the destination has paid
    /// parking.
    #[inline]
    fn add2(&mut self, var: usize, l: usize, b: usize, t: f64, s: &[f64; 4], sign: f64, paid: f32) {
        self.add(var, l, b, t, s, sign);
        if paid > 1e-30 {
            self.add(var, if l == 0 { 7 } else { 8 }, b, t, s, sign);
        }
    }
    fn plus(&mut self, o: &Acc, sign: f64) {
        for (a, b) in self.a.iter_mut().zip(&o.a) {
            *a += sign * b;
        }
    }
    fn json(&self) -> Value {
        json!((0..self.nv).map(|v| (0..NL).map(|l| (0..NB).map(|b| { let i = ((v * NL + l) * NB + b) * NM; self.a[i..i + NM].to_vec() }).collect::<Vec<_>>()).collect::<Vec<_>>()).collect::<Vec<_>>())
    }
}

/// What the commute hook needs about the zones.
struct Ctx {
    p: Params,
    vars: Vec<Var>,
    place: Vec<(f32, f32)>,
    park: Vec<(f32, f32, f32)>,
    cz: Vec<f32>,
    dtab: Vec<f32>,
    base: Vec<i32>,
    key: Vec<i32>,
    /// covered lanes with rail; the same trips at the zone pair's no-rail choice
    cov: Acc,
    cov0: Acc,
    lanes: u64,
    xa: f64,
    mine: f64,
}

thread_local! {
    static CTX: RefCell<Option<Ctx>> = const { RefCell::new(None) };
    /// everyday: per pair and direction, rail cost of car owners and of those without a car
    static EV: RefCell<(bool, Vec<[f32; 4]>)> = const { RefCell::new((false, vec![])) };
}

/// One commute lane offered rail: [trips, reserved walk share, car perceived (free parker), paid
/// space, share parking free, rail, rail without a car (inf: none), road km, scale, share
/// without a car, the kernel's rail trips].
pub(crate) fn hook_c(i: usize, j: usize, v: [f32; 11]) {
    CTX.with(|c| {
        let mut c = c.borrow_mut();
        let Some(c) = c.as_mut() else { return };
        let [t, pw, road, paid, free, rail, rail_n, rd, bt, cu, xa] = v;
        let p = &c.p;
        let d_km = c.dtab[(c.base[i] + c.key[j]) as usize];
        let b = band(d_km);
        let f = p.drive_factor(i, j);
        let road_z = p.drive_min_f(d_km * 1000.0, f, c.place[i], c.place[j]) * p.w_drive + c.park[j].0;
        let (paid_z, free_z) = (c.park[j].1, c.park[j].2);
        let bt_z = p.mode_beta(road_z);
        let rn = rail_n.is_finite().then_some(rail_n);
        let td = t as f64;
        for (k, var) in c.vars.iter().enumerate() {
            let s = choose(p, var, false, pw, road, paid, free, bt, Some(rail), rn, cu, rd);
            if k == 0 {
                c.mine += td * s[1];
            }
            c.cov.add2(k, 0, b, td, &s, 1.0, paid);
            let s0 = choose(p, var, false, pw, road_z, paid_z, free_z, bt_z, None, None, c.cz[i], d_km * f);
            c.cov0.add2(k, 0, b, td, &s0, 1.0, paid_z);
        }
        c.lanes += 1;
        c.xa += xa as f64;
    });
}

pub(crate) fn hook_ev_start(n: usize) {
    EV.with(|e| {
        let mut e = e.borrow_mut();
        if e.0 {
            e.1.clear();
            e.1.resize(n, [f32::INFINITY; 4]);
        }
    });
}
pub(crate) fn hook_ev(i: usize, dir: usize, rail: f32, rail_n: Option<f32>) {
    EV.with(|e| {
        let mut e = e.borrow_mut();
        if e.0 {
            e.1[i][2 * dir] = rail;
            e.1[i][2 * dir + 1] = rail_n.unwrap_or(f32::INFINITY);
        }
    });
}

fn arr_f32(v: &Value, k: &str) -> Vec<f32> {
    v[k].as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect()
}
fn arr_u32(v: &Value, k: &str) -> Vec<u32> {
    v[k].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect()
}

struct Zonal {
    place: Vec<(f32, f32)>,
    park: Vec<(f32, f32, f32)>,
    cz: Vec<f32>,
    dtab: Vec<f32>,
}
fn zonal(api: &DemandApi) -> Zonal {
    let (z, p) = (&*api.z, &api.base);
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
    Zonal { place, park, cz, dtab }
}

/// Every commute zone pair and every everyday sample with no rail: trips per day.
fn no_rail(api: &DemandApi, ev: &EverydayApi, zn: &Zonal, vars: &[Var], skip_far: bool) -> Acc {
    let mut acc = Acc::new(vars.len());
    let (z, gr, p) = (&*api.z, &*api.gr, &api.base);
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
            let d_km = zn.dtab[(base + gr.key(z, j)) as usize];
            let pw = gr.walk_share_pair(z, i, j);
            if skip_far && pw < 1e-11 {
                continue;
            }
            let f = p.drive_factor(i, j);
            let road = p.drive_min_f(d_km * 1000.0, f, zn.place[i], zn.place[j]) * p.w_drive + zn.park[j].0;
            let bt = p.mode_beta(road);
            for (k, v) in vars.iter().enumerate() {
                let s = choose(p, v, false, pw, road, zn.park[j].1, zn.park[j].2, bt, None, None, zn.cz[i], d_km * f);
                acc.add2(k, 0, band(d_km), t, &s, 1.0, zn.park[j].1);
            }
        }
    }
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
        if skip_far && pw < 1e-11 {
            continue;
        }
        let dist = ev.distance(c, dd);
        let (hz, az) = (cz[c] as usize, cz[dd] as usize);
        let cu = nocar.as_ref().map_or(0.0, |n| n[c]);
        let b = band(dist / 1000.0);
        for (u, v) in [(hz, az), (az, hz)] {
            let f = p.drive_factor(u, v);
            let road = p.drive_min_f(dist, f, place[u], place[v]) * p.w_drive + park[v].0;
            let bt = p.mode_beta(road);
            for (vi, var) in vars.iter().enumerate() {
                let s = choose(&p, var, true, pw, road, park[v].1, park[v].2, bt, None, None, cu, dist * 0.001 * f);
                acc.add2(vi, 1 + kind, b, w as f64, &s, 1.0, park[v].1);
            }
        }
    }
    acc
}

fn load(city: &str) -> (String, Vec<u8>, Value) {
    let data = std::env::var("T558_DATA").expect("T558_DATA");
    let net_path = format!("{data}/work/realnet/{city}_real.json");
    let net: Value = serde_json::from_str(&std::fs::read_to_string(&net_path).unwrap()).unwrap();
    let pack_path = format!("{data}/packs/{}.json", net["pack"].as_str().unwrap());
    let (h, b) = pack::load_raw(&pack_path).unwrap();
    (h, b, net)
}

fn open(h: &str, b: &[u8]) -> (DemandApi, EverydayApi) {
    let mut api = DemandApi::new(h, b).unwrap();
    let xfer: f32 = std::env::var("T558_XFER").ok().and_then(|x| x.parse().ok()).unwrap_or(0.0);
    api.params_mut().set_extra("transfer_min", xfer).unwrap();
    let ev = EverydayApi::beside(&api, h, b).unwrap();
    (api, ev)
}

/// The no-rail fit: for a grid of walk costs and constants, walking by layer and band.
#[test]
#[ignore]
fn t558_fit() {
    let city = std::env::var("T558_CITY").unwrap_or("nyc".into());
    let out_path = std::env::var("T558_OUT").expect("T558_OUT");
    let (h, b, _) = load(&city);
    let (api, ev) = open(&h, &b);
    let zn = zonal(&api);
    let mut vars = vec![TODAY];
    let t0s: Vec<f32> = std::env::var("T558_T0").map(|s| s.split(',').map(|x| x.parse().unwrap()).collect()).unwrap_or(vec![10.0]);
    let w2s: Vec<f32> = std::env::var("T558_W2").map(|s| s.split(',').map(|x| x.parse().unwrap()).collect()).unwrap_or(vec![1.5, 3.0, 4.5]);
    let ascs: Vec<f32> = std::env::var("T558_ASC").map(|s| s.split(',').map(|x| x.parse().unwrap()).collect()).unwrap_or((0..=24).map(|k| k as f32 * 0.25).collect());
    for &t0 in &t0s {
        for &w2 in &w2s {
            for &a in &ascs {
                vars.push(Var { today: false, lambda: 1.0, w1: 1.5, t0, w2, ac: a, ae: a });
            }
        }
    }
    let t = std::time::Instant::now();
    // walking only matters where the reserved share is not negligible; the rest is counted once
    let near = no_rail(&api, &ev, &zn, &vars, true);
    let all = no_rail(&api, &ev, &zn, &vars[..1], false);
    eprintln!("fit: {} variants in {:.1} s; {} samples", vars.len(), t.elapsed().as_secs_f64(), ev.destinations.len());
    std::fs::write(&out_path, serde_json::to_string(&json!({"city": city, "edges": EDGES, "vars": vars.iter().map(|v| v.json()).collect::<Vec<_>>(), "near": near.json(), "all": all.json()})).unwrap()).unwrap();
}

/// Today and the proposals on the real network and with no rail.
#[test]
#[ignore]
fn t558() {
    let city = std::env::var("T558_CITY").unwrap_or("nyc".into());
    let out_path = std::env::var("T558_OUT").expect("T558_OUT");
    let mut vars = vec![TODAY];
    if let Ok(s) = std::env::var("T558_VARS") {
        let v: Value = serde_json::from_str(&s).unwrap();
        vars.extend(v.as_array().unwrap().iter().map(Var::from_json));
    }
    let (h, b, net) = load(&city);
    let (mut api, mut ev) = open(&h, &b);
    let zn = zonal(&api);
    let t_all = std::time::Instant::now();
    let none = no_rail(&api, &ev, &zn, &vars, false);
    eprintln!("no rail: {:.1} s", t_all.elapsed().as_secs_f64());

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
    api.set_network(&st_xy, &line_n, &stops, &times, &tph, &cars).unwrap();
    ev.set_network(&st_xy, &line_n, &stops, &times, &tph, &cars).unwrap();
    api.set_fares(1.5, 0.1, &hop_km).unwrap();
    ev.set_fares(1.5, 0.1, &hop_km).unwrap();

    {
        let (z, gr) = (&*api.z, &*api.gr);
        CTX.with(|c| {
            *c.borrow_mut() = Some(Ctx {
                p: api.base.clone(),
                vars: vars.clone(),
                place: zn.place.clone(),
                park: zn.park.clone(),
                cz: zn.cz.clone(),
                dtab: zn.dtab.clone(),
                base: (0..z.n).map(|i| gr.base(z, i)).collect(),
                key: (0..z.n).map(|j| gr.key(z, j)).collect(),
                cov: Acc::new(vars.len()),
                cov0: Acc::new(vars.len()),
                lanes: 0,
                xa: 0.0,
                mine: 0.0,
            })
        });
    }
    // everyday: real = no rail + the change on each sample offered rail
    let mut ev_delta = Acc::new(vars.len());
    let mut summ = vec![];
    let mut ev_mine = 0.0f64;
    let rounds = 3;
    for q in 0..PERIODS {
        let t0 = std::time::Instant::now();
        EV.with(|e| e.borrow_mut().0 = true);
        api.solve(q);
        ev.solve(q);
        EV.with(|e| e.borrow_mut().0 = false);
        let free_c = api.summary(q);
        let free_e = ev.summary(q);
        let t_solve = t0.elapsed().as_secs_f64();
        {
            let rail = EV.with(|e| std::mem::take(&mut e.borrow_mut().1));
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
                let pi = if ev.edge_pair.is_empty() { NONE } else { ev.edge_pair[k] };
                if pi == NONE || rail.is_empty() {
                    continue;
                }
                let rc = rail[pi as usize];
                let dist = ev.distance(c, dd);
                let (hz, az) = (cz[c] as usize, cz[dd] as usize);
                let cu = nocar.as_ref().map_or(0.0, |n| n[c]);
                let b = band(dist / 1000.0);
                for (dir, (u, v)) in [(hz, az), (az, hz)].into_iter().enumerate() {
                    let r = rc[2 * dir];
                    if !r.is_finite() {
                        continue;
                    }
                    let rn = rc[2 * dir + 1];
                    let rn = rn.is_finite().then_some(rn);
                    let share = if dir == 0 { OUTWARD[kind][q] } else { RETURN[kind][q] };
                    let t = (w * share) as f64;
                    let f = p.drive_factor(u, v);
                    let road = p.drive_min_f(dist, f, place[u], place[v]) * p.w_drive + park[v].0;
                    let bt = p.mode_beta(road);
                    let rd = dist * 0.001 * f;
                    for (vi, var) in vars.iter().enumerate() {
                        let s = choose(&p, var, true, pw, road, park[v].1, park[v].2, bt, Some(r), rn, cu, rd);
                        let s0 = choose(&p, var, true, pw, road, park[v].1, park[v].2, bt, None, None, cu, rd);
                        if vi == 0 {
                            ev_mine += t * s[1];
                        }
                        ev_delta.add2(vi, 1 + kind, b, t, &s, 1.0, park[v].1);
                        ev_delta.add2(vi, 1 + kind, b, t, &s0, -1.0, park[v].1);
                    }
                }
            }
        }
        let t_ev = t0.elapsed().as_secs_f64() - t_solve;
        // the crowding rounds, for the model's own final figures (no hooks)
        for _ in 1..=rounds {
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
            let saved = CTX.with(|c| c.borrow_mut().take());
            assert!(api.crowd_with_external(q, &total));
            CTX.with(|c| *c.borrow_mut() = saved);
            assert!(ev.crowd_with_external(q, &total));
        }
        summ.push(json!({"q": q, "free_commute": free_c, "free_everyday": free_e, "final_commute": api.summary(q), "final_everyday": ev.summary(q)}));
        eprintln!("{city} period {q}: solve {t_solve:.1} s, everyday samples {t_ev:.1} s, all {:.1} s", t0.elapsed().as_secs_f64());
    }
    let c = CTX.with(|c| c.borrow_mut().take()).unwrap();
    let mut real = none.clone();
    real.plus(&c.cov0, -1.0);
    real.plus(&c.cov, 1.0);
    real.plus(&ev_delta, 1.0);
    let free_rail_e: f64 = summ.iter().map(|s| s["free_everyday"][2].as_f64().unwrap_or(0.0)).sum();
    eprintln!("{city}: {} lanes; commute rail kernel {:.0} against mine {:.0}; everyday free-flow rail {:.0} against mine {:.0}; all {:.1} s", c.lanes, c.xa, c.mine, free_rail_e, ev_mine, t_all.elapsed().as_secs_f64());
    std::fs::write(&out_path, serde_json::to_string(&json!({
        "city": city, "edges": EDGES, "vars": vars.iter().map(|v| v.json()).collect::<Vec<_>>(),
        "none": none.json(), "real": real.json(), "summary": summ,
        "checks": {"lanes": c.lanes, "kernel_rail": c.xa, "mine_rail": c.mine, "ev_free_rail": free_rail_e, "ev_mine_rail": ev_mine},
    })).unwrap()).unwrap();
}
