//! T-1163 (T-1175): everyday outings across a zone line, worked out offline for one solve unit
//! (`bin/cross_everyday.rs`, docs/zones.md 3).
//!
//! Each origin 2 km zone of a solved pack draws its errands, social outings and medical visits (and
//! each school's and university's site its students' homes) as the everyday layer does, zone
//! attraction times exp(-d / scale), a fifth of errands and social to dense centres at the 6 km
//! scale, but over its own pack's zones and every other loaded pack's within `REACH_KM` together.
//! Two outputs: per origin zone and purpose the share of the draw that stays in its own pack (the
//! pack's `everyday_kept`, which the everyday layer multiplies its budget by), and the rest as
//! outings a day by pair of 6 km squares of each pack's own frame and purpose, with each pair's
//! share walking the whole way with no rail, worked out between sampled cells as the everyday
//! layer's `no_rail` does (straight walks: no pack's water routes span two packs). The regional
//! layer carries those pairs.
use super::{
    everyday::{local_share, CENTRAL_DECAY_M, CENTRAL_SHARE, DECAY, KINDS},
    kernel::{zones_from_pack, Params, Zones, CELL_KM2},
    pack,
};
use crate::track::frame::Frame;
use serde_json::{json, Value};
use std::collections::HashMap;

/// The everyday purposes that cross a line, by `everyday` kind: errands, social, school,
/// university, medical. Airports and attractions keep each site's own total, so none of theirs
/// moves.
pub const CROSS: [usize; 5] = [0, 1, 2, 3, 5];
/// Pairs this far apart, centre to centre, take part: where the everyday layer's `local_share`
/// reaches nothing.
pub const REACH_KM: f64 = 80.0;
/// Zone pairs within this many km walk between sampled cells (`WALK_REACH_KM` and a zone's width
/// at each end); past it nobody walks.
const WALK_PAIR_KM: f64 = 13.0;
/// Cells sampled at each end of a zone pair for its walking share, at even steps of their weight.
const WALK_CELLS: usize = 6;
/// 6 km pairs under this many outings a day are dropped, their outings spread over their draw
/// origin square's kept pairs of the purpose (as `joint_gravity`'s `keep_pairs`, at its 1 a day).
pub const MIN_TRIPS: f64 = 1.0;
/// A zone's kept share is written only where a purpose keeps less than this.
const KEPT_WRITE: f64 = 0.9995;
const SQ_M: f64 = 6000.0;

fn site_anchored(slot: usize) -> bool {
    matches!(CROSS[slot], 2 | 3)
}
fn has_central(slot: usize) -> bool {
    CROSS[slot] < 2
}

/// One pack as the draw reads it.
pub struct CrossPack {
    pub id: String,
    pub country: String,
    pub solved: bool,
    origin: (f64, f64),
    p: Params,
    z: Zones,
    x: Vec<f32>,
    y: Vec<f32>,
    /// per `CROSS` slot and cell: outings a day sent (residents x rate, or a site's enrollment x
    /// rate), and the attraction drawn on
    budget: Vec<Vec<f32>>,
    attract: Vec<Vec<f32>>,
    /// per slot and zone: the sums of those, and the central draw's mass (attraction x attraction
    /// per km² of land)
    zb: Vec<Vec<f64>>,
    za: Vec<Vec<f64>>,
    zc: Vec<Vec<f64>>,
    /// per zone: centre in the pack's frame and in the unit's, streets' and main roads' minutes a
    /// km, parking; per cell: residents per km² for parking at home
    px: Vec<f64>,
    py: Vec<f64>,
    wx: Vec<f64>,
    wy: Vec<f64>,
    place: Vec<(f32, f32)>,
    park: Vec<(f32, f32, f32)>,
    home_dens: Vec<f32>,
    /// per cell, its position in the unit's frame and its 6 km square's index in the unit's list
    cwx: Vec<f64>,
    cwy: Vec<f64>,
    csq: Vec<u32>,
    /// per zone, its cells
    members: Vec<Vec<u32>>,
}

impl CrossPack {
    /// A pack from its header and .bin (`solved`: its zones are draw origins; else it is only
    /// drawn on).
    pub fn new(id: &str, header: &str, bin: &[u8], solved: bool) -> Result<CrossPack, String> {
        let city = pack::parse(header, bin)?;
        let pg = city.gravity.clone().ok_or(format!("{id} has no gravity"))?;
        let mut z = zones_from_pack(&city, &pg);
        let n = city.len();
        let h: Value = serde_json::from_str(header).map_err(|e| e.to_string())?;
        let w = |name: &str| pack::cell_weights(header, bin, name, n);
        let pop = w("pop")?.ok_or(format!("{id}: no resident population"))?;
        let jobs = w("jobs")?.ok_or(format!("{id}: no jobs"))?;
        // the everyday layer's zones carry residents and jobs (`DemandApi::with_weights`)
        z.pop.fill(0.0);
        z.jobs.fill(0.0);
        for c in 0..n {
            let zi = z.cell_zone[c] as usize;
            z.pop[zi] += pop[c] as f64;
            z.jobs[zi] += jobs[c] as f64;
        }
        let mut p = Params::default();
        for (kind, v) in h["everyday"]["rate"].as_object().into_iter().flatten() {
            if let Some(x) = v.as_f64() {
                p.set_extra(&format!("rate_{kind}"), x as f32)?;
            }
        }
        let sum = |v: &[f32]| v.iter().map(|&x| x as f64).sum::<f64>();
        let zeros = vec![0f32; n];
        let or0 = |v: Option<Vec<f32>>| v.filter(|v| sum(v) > 0.0).unwrap_or_else(|| zeros.clone());
        let retail = or0(w("errand_jobs")?);
        let leisure = or0(w("leisure_jobs")?);
        let parks = or0(w("park_draw")?);
        // social as `EverydayApi::beside_with` weighs it, each part a share of the pack's people
        // rather than of 1, so packs compare (T-1175)
        let (tp, tl, tk) = (sum(&pop), sum(&leisure), sum(&parks));
        let ps = if tk > 0.0 { 0.08 } else { 0.0 };
        let (hp, hl) = if tl > 0.0 { (0.5 * (1.0 - ps), 0.5 * (1.0 - ps)) } else { (1.0 - ps, 0.0) };
        let social: Vec<f32> = (0..n)
            .map(|c| {
                let l = if tl > 0.0 { hl * leisure[c] as f64 / tl } else { 0.0 };
                let k = if tk > 0.0 { ps * parks[c] as f64 / tk } else { 0.0 };
                ((hp * pop[c] as f64 / tp.max(1e-9) + l + k) * tp) as f32
            })
            .collect();
        let pair = |a: &str, b: &str| -> Result<(Vec<f32>, Vec<f32>), String> {
            match (w(a)?, w(b)?) {
                (Some(a), Some(b)) if sum(&a) > 0.0 && sum(&b) > 0.0 => Ok((a, b)),
                _ => Ok((zeros.clone(), zeros.clone())),
            }
        };
        let (children, school) = pair("school_children", "school_enrollment")?;
        let (adults, university) = pair("university_adults", "university_enrollment")?;
        let medical = or0(w("medical_visits")?);
        let rate = |kind: usize| p.everyday_rate[kind];
        let scaled = |v: &[f32], kind: usize| v.iter().map(|&x| x * rate(kind)).collect::<Vec<f32>>();
        let budget = vec![scaled(&pop, 0), scaled(&pop, 1), scaled(&school, 2), scaled(&university, 3), scaled(&pop, 5)];
        let attract = vec![retail, social, children, adults, medical];
        let nz = z.n;
        let by_zone = |v: &[f32]| {
            let mut s = vec![0f64; nz];
            for c in 0..n {
                s[z.cell_zone[c] as usize] += v[c] as f64;
            }
            s
        };
        let zb: Vec<Vec<f64>> = budget.iter().map(|v| by_zone(v)).collect();
        let za: Vec<Vec<f64>> = attract.iter().map(|v| by_zone(v)).collect();
        let land = |k: usize| (z.ncell[k].max(1) as f32 * CELL_KM2).min(z.square_km2(k)) as f64;
        let zc: Vec<Vec<f64>> = (0..CROSS.len()).map(|s| if has_central(s) { (0..nz).map(|k| za[s][k] * za[s][k] / land(k)).collect() } else { vec![] }).collect();
        let px: Vec<f64> = (0..nz).map(|i| z.x0 as f64 + z.hx(i) as f64 * 0.5 * z.zone_m as f64).collect();
        let py: Vec<f64> = (0..nz).map(|i| z.y0 as f64 + z.hy(i) as f64 * 0.5 * z.zone_m as f64).collect();
        let ep = p.everyday();
        let place = z.density().iter().map(|&d| ep.place_min_per_km(d)).collect();
        let park = (0..nz).map(|j| ep.park_parts(&z, j)).collect();
        let home_dens = z.home_density(&pop);
        let mut members = vec![vec![]; nz];
        for c in 0..n {
            members[z.cell_zone[c] as usize].push(c as u32);
        }
        let origin = (h["origin"]["lon"].as_f64().ok_or("no origin")?, h["origin"]["lat"].as_f64().ok_or("no origin")?);
        let country = h["country"].as_str().unwrap_or("US").to_uppercase();
        Ok(CrossPack {
            id: id.to_string(),
            country,
            solved,
            origin,
            p: ep,
            x: city.x.clone(),
            y: city.y.clone(),
            z,
            budget,
            attract,
            zb,
            za,
            zc,
            px,
            py,
            wx: vec![],
            wy: vec![],
            place,
            park,
            home_dens,
            cwx: vec![],
            cwy: vec![],
            csq: vec![],
            members,
        })
    }

    /// The share walking the whole way with no rail between home cell `h` (this pack) and activity
    /// cell `a` (pack `q`), `d` metres apart: `EverydayApi::no_rail` with no water routes.
    fn no_rail(&self, h: usize, q: &CrossPack, a: usize, d: f32) -> f32 {
        let p = &self.p;
        let cw = p.walk_trip_cost(d / 1000.0);
        let hp = p.home_park_at(self.home_dens[h]);
        let (hz, az) = (self.z.cell_zone[h] as usize, q.z.cell_zone[a] as usize);
        let f = p.road_detour;
        let mut out = 0.0;
        for ((pu, u), (pv, v)) in [((self, hz), (q, az)), ((q, az), (self, hz))] {
            let (fixed, paid, free) = pv.park[v];
            let road = p.car_drive(p.drive_min_f(d, f, pu.place[u], pv.place[v]), d * 0.001 * f) + fixed + hp;
            out += 0.5 * p.walk_no_rail(road, paid, free, cw);
        }
        out
    }

    /// Up to `WALK_CELLS` cells of zone `k`, at even steps of weight `v`.
    fn sample_cells(&self, k: usize, v: &[f32]) -> Vec<u32> {
        let m = &self.members[k];
        let total: f64 = m.iter().map(|&c| v[c as usize] as f64).sum();
        if total <= 0.0 {
            return vec![];
        }
        let mut out = Vec::with_capacity(WALK_CELLS);
        let (mut acc, mut i) = (0.0, 0);
        for s in 0..WALK_CELLS {
            let target = (s as f64 + 0.5) / WALK_CELLS as f64 * total;
            while i + 1 < m.len() && acc + v[m[i] as usize] as f64 <= target {
                acc += v[m[i] as usize] as f64;
                i += 1;
            }
            out.push(m[i]);
        }
        out
    }
}

/// One 6 km pair of one purpose: outings a day and outings walking with no rail.
#[derive(Default, Clone, Copy)]
struct Flow {
    trips: f64,
    walk: f64,
}

/// What a unit's run gives: per solved pack its zones' kept shares (zone, a share per `CROSS`
/// slot), and the pairs.
pub struct Unit {
    pub kept: Vec<Vec<(u32, [f64; 5])>>,
    /// (draw origin square, other square, slot) -> flow; squares index `squares`
    flows: HashMap<(u32, u32, u8), Flow>,
    /// (pack, x, y) of each 6 km square
    pub squares: Vec<(u32, i32, i32)>,
    /// outings a day moved out of each pack's zones, by slot (before dropping small pairs)
    pub moved: Vec<[f64; 5]>,
    pub outings: Vec<[f64; 5]>,
}

/// The border factor between two packs' countries: 1 inside a country, `borders.json`'s national
/// factor across a national border (as the commutes take it).
pub fn border(borders: &Value, a: &str, b: &str) -> f64 {
    if a == b {
        return 1.0;
    }
    let key = if a < b { format!("{a}|{b}") } else { format!("{b}|{a}") };
    borders["national"][key.as_str()].as_f64().or(borders["national_default"].as_f64()).unwrap_or(1.0)
}

/// The unit's draw: every solved pack's zones over every pack's within `REACH_KM`, on `threads`
/// threads.
pub fn run(packs: &mut [CrossPack], borders: &Value, threads: usize) -> Unit {
    let n = packs.len() as f64;
    let world = Frame::lcc(packs.iter().map(|p| p.origin.0).sum::<f64>() / n, packs.iter().map(|p| p.origin.1).sum::<f64>() / n);
    let mut squares: Vec<(u32, i32, i32)> = vec![];
    let mut sq_index: HashMap<(u32, i32, i32), u32> = HashMap::new();
    for (pi, p) in packs.iter_mut().enumerate() {
        let fr = Frame::equirect(p.origin.0, p.origin.1);
        let to_world = |x: f64, y: f64| {
            let (lon, lat) = fr.to_lonlat(x, y);
            world.from_lonlat(lon, lat)
        };
        (p.wx, p.wy) = (0..p.z.n).map(|i| to_world(p.px[i], p.py[i])).unzip();
        (p.cwx, p.cwy) = (0..p.x.len()).map(|c| to_world(p.x[c] as f64, p.y[c] as f64)).unzip();
        p.csq = (0..p.x.len())
            .map(|c| {
                let k = (pi as u32, (p.x[c] as f64 / SQ_M).floor() as i32, (p.y[c] as f64 / SQ_M).floor() as i32);
                *sq_index.entry(k).or_insert_with(|| {
                    squares.push(k);
                    (squares.len() - 1) as u32
                })
            })
            .collect();
    }
    let packs: &[CrossPack] = packs;
    let factor: Vec<Vec<f64>> = packs.iter().map(|a| packs.iter().map(|b| border(borders, &a.country, &b.country)).collect()).collect();
    // every zone in 10 km buckets of the unit's frame
    let bucket_m = 10_000.0;
    let bucket = |x: f64| (x / bucket_m).floor() as i32;
    let mut grid: HashMap<(i32, i32), Vec<(u32, u32)>> = HashMap::new();
    for (pi, p) in packs.iter().enumerate() {
        for k in 0..p.z.n {
            grid.entry((bucket(p.wx[k]), bucket(p.wy[k]))).or_default().push((pi as u32, k as u32));
        }
    }
    let reach = (REACH_KM * 1000.0 / bucket_m).ceil() as i32;
    let reach2 = (REACH_KM * 1000.0).powi(2);
    // the origins: every zone of a solved pack sending any outings
    let origins: Vec<(usize, usize)> = packs
        .iter()
        .enumerate()
        .filter(|(_, p)| p.solved)
        .flat_map(|(pi, p)| (0..p.z.n).filter(move |&k| (0..CROSS.len()).any(|s| p.zb[s][k] > 0.0)).map(move |k| (pi, k)))
        .collect();
    let chunk = origins.len().div_ceil(threads.max(1)).max(1);
    let decay: Vec<f64> = CROSS.iter().map(|&k| DECAY[k] as f64).collect();
    let q = CENTRAL_SHARE as f64;
    let parts: Vec<(Vec<(usize, usize, [f64; 5])>, HashMap<(u32, u32, u8), Flow>, Vec<[f64; 5]>, Vec<[f64; 5]>)> = std::thread::scope(|sc| {
        let handles: Vec<_> = origins
            .chunks(chunk)
            .map(|part| {
                let grid = &grid;
                let factor = &factor;
                let decay = &decay;
                sc.spawn(move || {
                    let mut kept = vec![];
                    let mut flows: HashMap<(u32, u32, u8), Flow> = HashMap::new();
                    let mut moved = vec![[0f64; 5]; packs.len()];
                    let mut outings = vec![[0f64; 5]; packs.len()];
                    // the other packs' zones in reach of the origin: (pack, zone, metres)
                    let mut near: Vec<(u32, u32, f64)> = vec![];
                    let mut cells_o: Vec<Vec<u32>> = vec![vec![]; CROSS.len()];
                    for &(pi, o) in part {
                        let p = &packs[pi];
                        let (ox, oy) = (p.wx[o], p.wy[o]);
                        let (bx, by) = (bucket(ox), bucket(oy));
                        // the draw's sums over the own pack (its own frame, as `DecayTable`) and
                        // the others, local then central
                        let (mut lo, mut ln, mut co, mut cn) = ([0f64; 5], [0f64; 5], [0f64; 5], [0f64; 5]);
                        near.clear();
                        for gx in bx - reach..=bx + reach {
                            for gy in by - reach..=by + reach {
                                let Some(list) = grid.get(&(gx, gy)) else { continue };
                                for &(qi, j) in list {
                                    let (qi, j) = (qi as usize, j as usize);
                                    let qp = &packs[qi];
                                    let own = qi == pi;
                                    let d2 = if own { (p.px[j] - p.px[o]).powi(2) + (p.py[j] - p.py[o]).powi(2) } else { (qp.wx[j] - ox).powi(2) + (qp.wy[j] - oy).powi(2) };
                                    if d2 > reach2 {
                                        continue;
                                    }
                                    let floor = 0.52 * p.z.zone_m as f64 * p.z.side(o) as f64;
                                    let d = if own && j == o { floor } else { d2.sqrt().max(if own { 0.0 } else { floor }) };
                                    let b = factor[pi][qi];
                                    let ce = (-d / CENTRAL_DECAY_M as f64).exp();
                                    for s in 0..CROSS.len() {
                                        let m = qp.za[s][j];
                                        if m <= 0.0 || p.zb[s][o] <= 0.0 {
                                            continue;
                                        }
                                        let l = b * m * (-d / decay[s]).exp();
                                        let c = if has_central(s) { b * qp.zc[s][j] * ce } else { 0.0 };
                                        if own {
                                            lo[s] += l;
                                            co[s] += c;
                                        } else {
                                            ln[s] += l;
                                            cn[s] += c;
                                        }
                                    }
                                    if !own {
                                        near.push((qi as u32, j as u32, d));
                                    }
                                }
                            }
                        }
                        // per slot, the weights of the local and central draws (`mix_central`)
                        let mut wl = [0f64; 5];
                        let mut wc = [0f64; 5];
                        let mut k = [1f64; 5];
                        for s in 0..CROSS.len() {
                            let qs = if has_central(s) { q } else { 0.0 };
                            let (la, ca) = (lo[s] + ln[s], co[s] + cn[s]);
                            (wl[s], wc[s]) = match (la > 0.0, ca > 0.0 && qs > 0.0) {
                                (true, true) => ((1.0 - qs) / la, qs / ca),
                                (true, false) => (1.0 / la, 0.0),
                                (false, true) => (0.0, 1.0 / ca),
                                (false, false) => (0.0, 0.0),
                            };
                            if la > 0.0 || ca > 0.0 {
                                k[s] = lo[s] * wl[s] + co[s] * wc[s];
                            }
                            outings[pi][s] += p.zb[s][o];
                        }
                        if k.iter().any(|&v| v < KEPT_WRITE) {
                            kept.push((pi, o, k));
                        }
                        if near.is_empty() {
                            continue;
                        }
                        // the origin's cells for walks, per slot
                        for s in 0..CROSS.len() {
                            cells_o[s] = if p.zb[s][o] > 0.0 { p.sample_cells(o, &p.budget[s]) } else { vec![] };
                        }
                        for &(qi, j, d) in &near {
                            let (qi, j) = (qi as usize, j as usize);
                            let qp = &packs[qi];
                            let b = factor[pi][qi];
                            let ls = local_share(d as f32) as f64;
                            if ls <= 0.0 {
                                continue;
                            }
                            let ce = (-d / CENTRAL_DECAY_M as f64).exp();
                            for s in 0..CROSS.len() {
                                let m = qp.za[s][j];
                                if m <= 0.0 || p.zb[s][o] <= 0.0 {
                                    continue;
                                }
                                let draw = wl[s] * b * m * (-d / decay[s]).exp() + if has_central(s) { wc[s] * b * qp.zc[s][j] * ce } else { 0.0 };
                                let t = p.zb[s][o] * draw;
                                moved[pi][s] += t;
                                let t = t * ls;
                                if t < 1e-6 {
                                    continue;
                                }
                                // the walking share at cell grain, home end first
                                let mut walk = 0.0;
                                if d <= WALK_PAIR_KM * 1000.0 {
                                    let cells_j = qp.sample_cells(j, &qp.attract[s]);
                                    let mut nw = 0;
                                    for &a in &cells_o[s] {
                                        for &c in &cells_j {
                                            let (a, c) = (a as usize, c as usize);
                                            let dm = ((p.cwx[a] - qp.cwx[c]).powi(2) + (p.cwy[a] - qp.cwy[c]).powi(2)).sqrt() as f32;
                                            walk += if site_anchored(s) { qp.no_rail(c, p, a, dm) } else { p.no_rail(a, qp, c, dm) } as f64;
                                            nw += 1;
                                        }
                                    }
                                    walk /= nw.max(1) as f64;
                                }
                                // spread over the 6 km squares by the cells' weights at each end
                                let spread = |pk: &CrossPack, z: usize, v: &[f32]| -> Vec<(u32, f64)> {
                                    let mut out: Vec<(u32, f64)> = vec![];
                                    let tot: f64 = pk.members[z].iter().map(|&c| v[c as usize] as f64).sum();
                                    for &c in &pk.members[z] {
                                        let w = v[c as usize] as f64;
                                        if w <= 0.0 {
                                            continue;
                                        }
                                        let sq = pk.csq[c as usize];
                                        match out.iter_mut().find(|e| e.0 == sq) {
                                            Some(e) => e.1 += w / tot,
                                            None => out.push((sq, w / tot)),
                                        }
                                    }
                                    out
                                };
                                for (so, fo) in spread(p, o, &p.budget[s]) {
                                    for (sj, fj) in spread(qp, j, &qp.attract[s]) {
                                        let f = flows.entry((so, sj, s as u8)).or_default();
                                        f.trips += t * fo * fj;
                                        f.walk += t * walk * fo * fj;
                                    }
                                }
                            }
                        }
                    }
                    (kept, flows, moved, outings)
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let mut unit = Unit { kept: vec![vec![]; packs.len()], flows: HashMap::new(), squares, moved: vec![[0f64; 5]; packs.len()], outings: vec![[0f64; 5]; packs.len()] };
    for (kept, flows, moved, outings) in parts {
        for (pi, o, k) in kept {
            unit.kept[pi].push((o as u32, k));
        }
        for (key, f) in flows {
            let e = unit.flows.entry(key).or_default();
            e.trips += f.trips;
            e.walk += f.walk;
        }
        for pi in 0..packs.len() {
            for s in 0..CROSS.len() {
                unit.moved[pi][s] += moved[pi][s];
                unit.outings[pi][s] += outings[pi][s];
            }
        }
    }
    for k in unit.kept.iter_mut() {
        k.sort_by_key(|e| e.0);
    }
    unit
}

impl Unit {
    /// The pairs of at least `MIN_TRIPS` a day, the rest's outings spread over their origin
    /// square's kept pairs of the purpose: (home square, activity square, slot, outings, walking
    /// share), the home end first whichever end drew.
    pub fn pairs(&self) -> Vec<(u32, u32, u8, f64, f64)> {
        let mut all: HashMap<(u32, u8), f64> = HashMap::new();
        let mut kept: HashMap<(u32, u8), f64> = HashMap::new();
        for (&(o, _, s), f) in &self.flows {
            *all.entry((o, s)).or_default() += f.trips;
            if f.trips >= MIN_TRIPS {
                *kept.entry((o, s)).or_default() += f.trips;
            }
        }
        let mut out: Vec<(u32, u32, u8, f64, f64)> = self
            .flows
            .iter()
            .filter(|(_, f)| f.trips >= MIN_TRIPS)
            .map(|(&(o, j, s), f)| {
                let up = all[&(o, s)] / kept[&(o, s)];
                let (h, a) = if site_anchored(s as usize) { (j, o) } else { (o, j) };
                (h, a, s, f.trips * up, f.walk / f.trips)
            })
            .collect();
        out.sort_by(|a, b| (a.0, a.1, a.2).cmp(&(b.0, b.1, b.2)));
        out
    }

    /// The unit's file (`data/work/<name>.cross_everyday.json`): the pairs as pack and square of
    /// each pack's own frame, and every solved pack's kept shares.
    pub fn to_json(&self, packs: &[CrossPack]) -> Value {
        let pairs = self.pairs();
        let sq = |i: u32| self.squares[i as usize];
        let r = |v: f64, k: f64| (v * k).round() / k;
        let kept: serde_json::Map<String, Value> = packs
            .iter()
            .enumerate()
            .filter(|(_, p)| p.solved)
            .map(|(pi, p)| {
                let k = &self.kept[pi];
                (p.id.clone(), json!({ "zones": p.z.n, "zone": k.iter().map(|e| e.0).collect::<Vec<_>>(), "kept": k.iter().map(|e| e.1.iter().map(|&v| r(v, 1e4)).collect::<Vec<_>>()).collect::<Vec<_>>() }))
            })
            .collect();
        json!({
            "task": "T-1175",
            "packs": packs.iter().map(|p| p.id.clone()).collect::<Vec<_>>(),
            "solved": packs.iter().filter(|p| p.solved).map(|p| p.id.clone()).collect::<Vec<_>>(),
            "solved_at": std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0.0, |d| d.as_secs_f64()),
            "zone_m": SQ_M,
            "frame": "each pack's own equirectangular frame at its origin; a zone is [floor(x / 6000), floor(y / 6000)]",
            "kinds": CROSS.iter().map(|&k| KINDS[k]).collect::<Vec<_>>(),
            "source": format!("bin/cross_everyday.rs (T-1163, T-1175): everyday outings drawn over every pack within {REACH_KM} km, those leaving the drawing pack; pairs under {MIN_TRIPS} a day spread over their square's others"),
            "home_pack": pairs.iter().map(|x| sq(x.0).0).collect::<Vec<_>>(),
            "home_zone": pairs.iter().map(|x| [sq(x.0).1, sq(x.0).2]).collect::<Vec<_>>(),
            "act_pack": pairs.iter().map(|x| sq(x.1).0).collect::<Vec<_>>(),
            "act_zone": pairs.iter().map(|x| [sq(x.1).1, sq(x.1).2]).collect::<Vec<_>>(),
            "kind": pairs.iter().map(|x| CROSS[x.2 as usize]).collect::<Vec<_>>(),
            "trips": pairs.iter().map(|x| r(x.3, 1e3)).collect::<Vec<_>>(),
            "walk": pairs.iter().map(|x| r(x.4, 1e4)).collect::<Vec<_>>(),
            "kept": kept,
        })
    }
}
