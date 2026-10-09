//! The demand kernel stages (SPEC 4.4). Flat arrays throughout; no per-cell structs.
//!
//! Scheme (see notes/T-005.md, T-019.md, T-020.md for why):
//! 1. Gravity, doubly constrained, between square zones of `zone_m` (2 km): homes -> jobs by
//!    distance only, so it does not depend on the network. The pipeline solves it once
//!    (bin/pack_gravity.rs) and ships the factors in the pack; `city_gravity` rebuilds it.
//! 2. Per cell, up to K=6 stations within reach of the fast walk (the one access mode, T-078).
//! 3. Access subzones: the cells of one zone that share their nearest station (and walk band, if
//!    bands are on). They carry the pop/jobs-weighted walk time to each of their stations, used
//!    at both ends of a trip. Cells with no station within reach are in no subzone: no rail.
//! 4. Station-to-station generalised times: one Dijkstra per station over station + route nodes.
//! 5. Mode choice per (origin subzone, destination subzone) pair: zone trips spread by pop/jobs
//!    weight, rail cost = logsum over access options and egress stations, logit against road.
//! 6. Rail trips by station pair, loaded onto line segments along the stored shortest-path trees.
//! 7. Crowding: penalty per segment from the loads, repeat 4-6, average loads (MSA).

use super::network::Network;
use super::pack::{City, PackGravity};
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};

/// Stations kept per cell and per subzone.
pub const K: usize = 6;
const NONE: u32 = u32::MAX;

#[derive(Clone, Debug)]
pub struct Params {
    /// The one way to reach a station: a fast walk at about bicycle speed and willingness (SPEC
    /// 4.3, T-078), at both ends. The path is `walk_detour` x the straight line, covered at
    /// `walk_mps`; its first `walk_easy_min` minutes weigh `w_walk` each and the rest
    /// `w_walk_far` (T-090: a long way to the station puts people off more per minute).
    /// `walk_cutoff_m` is only a computational bound (straight line, metres), set so far out that
    /// few rail trips come from near it: a far station loses riders because the whole trip gets
    /// worse than driving, not at an edge.
    pub walk_cutoff_m: f32,
    pub walk_detour: f32,
    pub walk_mps: f32,
    pub walk_easy_min: f32,
    pub w_walk_far: f32,
    /// Gravity zone size, metres.
    pub zone_m: f32,
    /// Walk band width for subzones, minutes. 0 = no bands; < 0 = one subzone per cell (exact).
    pub band_min: f32,
    /// Gravity distance decay: f(d) = d^-decay_pow x exp(-d / decay_km), d in km (T-006: fitted
    /// to LODES OD for New York; decay_pow 0 is the plain exponential).
    pub decay_km: f32,
    pub decay_pow: f32,
    pub w_walk: f32,
    pub w_wait: f32,
    pub w_home_wait: f32,
    /// Waiting beyond this many minutes (half headway) counts at `w_home_wait`.
    pub platform_wait_cap_min: f32,
    pub transfer_min: f32,
    pub fare_min: f32,
    /// Logit scale per perceived minute, and the per-city rail propensity.
    pub beta: f32,
    pub asc_rail: f32,
    /// Driving door to door (T-090), the alternative for the whole trip: the road is
    /// `road_detour` x the straight line; its first and last `road_local_km` (at most half the
    /// road each) are local streets, at `road_town_kmh` in open country, and the rest is main
    /// roads at `road_far_kmh`; both slow towards `road_city_kmh` where it is dense (half way at
    /// `road_dens50` commuters living and working per km² of land in the zone;
    /// `place_min_per_km`). So short trips and trips in the city crawl and long suburban ones
    /// are quick. No road network (T-072).
    pub road_detour: f32,
    pub road_local_km: f32,
    pub road_city_kmh: f32,
    pub road_town_kmh: f32,
    pub road_dens50: f32,
    pub road_far_kmh: f32,
    pub w_drive: f32,
    pub park_min: f32,
    pub park_cost_cap_min: f32,
    /// Parking cost: one perceived minute per this many jobs per km² in the destination zone.
    pub park_jobs_per_km2_per_min: f32,
    pub period: usize,
    /// Share of all home->work trips in this period.
    pub period_share: f32,
    pub period_hours: f32,
    pub peak_hour_factor: f32,
    pub crowd_alpha: f32,
    /// Zone pairs carrying fewer trips than this are skipped in mode choice.
    pub prune_trips: f32,
    /// Subzone pairs whose cheapest walks at the two ends together cost more than this many
    /// perceived minutes are not offered rail (T-090: a computational bound like
    /// `walk_cutoff_m`, set where little rail is left).
    pub pair_walk_max: f32,
    /// Logit scale of the choice among access and egress stations (and access modes), per
    /// perceived minute. notes/T-020.md. The pair loop keeps exp(-theta x station-to-station
    /// minutes) in f32, so theta x the longest such trip must stay under ~85 (0.2 allows 7 h);
    /// it cannot be pushed towards all-or-nothing.
    pub station_theta: f32,
    /// Share of the trips not taken by rail that walk, by zone-centre distance d:
    /// 1 / (1 + exp((d - walk_d0_km) / walk_s_km)). The rest drive (or take the bus). Judgement
    /// (notes/T-026.md): ~84% within a 2 km zone, 32% to the next zone, 6% two zones away.
    pub walk_d0_km: f32,
    pub walk_s_km: f32,
    /// Walking transfers (SPEC 6.1): a rider leaving a train may walk to another station within
    /// this straight-line distance and board there, at the walk's perceived time plus the usual
    /// boarding cost. Only after a ride, so walking between two stations is never a rail trip.
    /// 0 turns it off. T-007: crossing lines at different levels cannot share a station node.
    pub xfer_walk_m: f32,
    /// The transfer walk is on foot: speed and perceived weight (Subway Builder's 1.39).
    pub xfer_mps: f32,
    pub w_xfer: f32,
    /// Stations of the snapshot this close together are one demand station (`api::merge_stations`).
    pub station_merge_m: f32,}

impl Default for Params {
    fn default() -> Self {
        Params {
            // T-078, T-090: each km to the station costs about 10 perceived minutes for the
            // first 10 minutes (1.9 km), four times that beyond, so a station's draw fades out
            // between about 2 and 3.5 km with no edge the player sees. The bound is past where
            // that leaves anything (notes/T-090.md)
            walk_cutoff_m: 4000.0,
            walk_detour: 1.3,
            walk_mps: 4.17,
            walk_easy_min: 10.0,
            w_walk_far: 8.0,
            zone_m: 2000.0,
            band_min: 0.0,
            decay_km: 29.0,
            decay_pow: 1.0,
            w_walk: 2.0,
            w_wait: 1.37,
            w_home_wait: 0.4,
            platform_wait_cap_min: 5.0,
            transfer_min: 5.0,
            fare_min: 6.0,
            beta: 0.05,
            asc_rail: 0.0,
            road_detour: 1.3,
            road_local_km: 2.0,
            road_city_kmh: 15.0,
            road_town_kmh: 40.0,
            road_dens50: 5000.0,
            road_far_kmh: 80.0,
            w_drive: 1.33,
            park_min: 3.0,
            park_cost_cap_min: 40.0,
            park_jobs_per_km2_per_min: 2500.0,
            period: 0,
            period_share: 0.45,
            period_hours: 3.0,
            peak_hour_factor: 1.25,
            crowd_alpha: 2.0,
            prune_trips: 0.0,
            // T-090: holds the pairs at what the 2.5 km edge had; on the real New York network
            // 0.3% of rail trips have walks of 70-80 perceived minutes together and 0.2% were
            // beyond it (a lone line in Manhattan, the worst case: 2.3%, and the bounds cost it
            // 4% of its riders)
            pair_walk_max: 80.0,
            station_theta: 0.2,
            walk_d0_km: 1.7,
            walk_s_km: 0.4,
            xfer_walk_m: 300.0,
            xfer_mps: 1.2,
            w_xfer: 1.39,
            station_merge_m: 100.0,
        }
    }
}

impl Params {
    /// Set a parameter by name, for the checking tools (`realnet --set`); not every field.
    pub fn set_extra(&mut self, k: &str, v: f32) -> Result<(), String> {
        match k {
            "station_theta" => self.station_theta = v,
            "asc_rail" => self.asc_rail = v,
            "walk_mps" => self.walk_mps = v,
            "walk_cutoff_m" => self.walk_cutoff_m = v,
            "w_wait" => self.w_wait = v,
            "w_walk" => self.w_walk = v,
            "walk_easy_min" => self.walk_easy_min = v,
            "w_walk_far" => self.w_walk_far = v,
            "pair_walk_max" => self.pair_walk_max = v,
            "road_detour" => self.road_detour = v,
            "road_local_km" => self.road_local_km = v,
            "road_city_kmh" => self.road_city_kmh = v,
            "road_town_kmh" => self.road_town_kmh = v,
            "road_dens50" => self.road_dens50 = v,
            "road_far_kmh" => self.road_far_kmh = v,
            // the drive before T-090: one speed everywhere
            "road_kmh" => {
                self.road_city_kmh = v;
                self.road_town_kmh = v;
                self.road_far_kmh = v;
            }
            _ => return Err(format!("unknown parameter {k}")),
        }
        Ok(())
    }
    /// Real minutes of the fast walk to a station `d_m` metres away in a straight line.
    pub fn walk_min(&self, d_m: f32) -> f32 {
        d_m * self.walk_detour / self.walk_mps / 60.0
    }
    /// Perceived minutes of that walk: the first `walk_easy_min` at `w_walk`, the rest at
    /// `w_walk_far`.
    pub fn access_cost(&self, d_m: f32) -> f32 {
        let t = self.walk_min(d_m);
        t.min(self.walk_easy_min) * self.w_walk + (t - self.walk_easy_min).max(0.0) * self.w_walk_far
    }
    /// Minutes a km at a place with `dens` homes and jobs per km² of land: on its local streets,
    /// and on the main roads near it. Both fall from the open-country speeds (`road_town_kmh`,
    /// `road_far_kmh`) towards `road_city_kmh` as it gets dense, half way at `road_dens50`.
    pub fn place_min_per_km(&self, dens: f32) -> (f32, f32) {
        let f = 1.0 / (1.0 + dens / self.road_dens50.max(1e-6));
        let c = self.road_city_kmh;
        (60.0 / (c + (self.road_town_kmh - c) * f), 60.0 / (c + (self.road_far_kmh - c) * f))
    }
    /// Real minutes to drive `d_m` metres (straight line) between places a and b
    /// (`place_min_per_km` of each): the first and last `road_local_km` of road (at most half
    /// each) on each end's local streets, the middle on main roads at the mean of the two ends'
    /// minutes a km.
    #[inline]
    pub fn drive_min(&self, d_m: f32, a: (f32, f32), b: (f32, f32)) -> f32 {
        let road = d_m * 0.001 * self.road_detour;
        let e = (0.5 * road).min(self.road_local_km);
        e * (a.0 + b.0) + (road - 2.0 * e) * 0.5 * (a.1 + b.1)
    }
    /// Real minutes of the walk between two stations on a transfer, `d_m` metres straight line.
    pub fn xfer_min(&self, d_m: f32) -> f32 {
        d_m * self.walk_detour / self.xfer_mps / 60.0
    }
    /// Share of non-rail trips over `d_km` (zone-centre distance) that walk.
    pub fn walk_share(&self, d_km: f32) -> f32 {
        1.0 / (1.0 + ((d_km - self.walk_d0_km) / self.walk_s_km).exp())
    }
}

// ---------------------------------------------------------------------------------------------
// 1. Zones and gravity

pub struct Zones {
    pub zone_m: f32,
    pub n: usize,
    pub cell_zone: Vec<u32>,
    pub gx: Vec<i32>,
    pub gy: Vec<i32>,
    /// Grid size in zones, and its lower-left corner in metres.
    pub w: i32,
    pub h: i32,
    pub x0: f32,
    pub y0: f32,
    pub pop: Vec<f64>,
    pub jobs: Vec<f64>,
    /// Cells in each zone (its land, at `CELL_KM2` a cell).
    pub ncell: Vec<u32>,
}

/// Mean area of an H3 resolution 9 cell, km².
pub const CELL_KM2: f32 = 0.1053;

impl Zones {
    /// Homes and jobs per km² of land, per zone (the drive's local street speed).
    pub fn density(&self) -> Vec<f32> {
        (0..self.n).map(|i| ((self.pop[i] + self.jobs[i]) / (self.ncell[i].max(1) as f64 * CELL_KM2 as f64)) as f32).collect()
    }
}

pub fn build_zones(city: &City, zone_m: f32) -> Zones {
    let minx = city.x.iter().cloned().fold(f32::INFINITY, f32::min);
    let miny = city.y.iter().cloned().fold(f32::INFINITY, f32::min);
    let maxx = city.x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let maxy = city.y.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let w = ((maxx - minx) / zone_m) as i32 + 1;
    let h = ((maxy - miny) / zone_m) as i32 + 1;
    let mut grid = vec![NONE; (w * h) as usize];
    let mut z = Zones { zone_m, n: 0, cell_zone: vec![0; city.len()], gx: vec![], gy: vec![], w, h, x0: minx, y0: miny, pop: vec![], jobs: vec![], ncell: vec![] };
    for c in 0..city.len() {
        let gx = ((city.x[c] - minx) / zone_m) as i32;
        let gy = ((city.y[c] - miny) / zone_m) as i32;
        let g = (gx * h + gy) as usize;
        if grid[g] == NONE {
            grid[g] = z.n as u32;
            z.n += 1;
            z.gx.push(gx);
            z.gy.push(gy);
            z.pop.push(0.0);
            z.jobs.push(0.0);
            z.ncell.push(0);
        }
        let zi = grid[g] as usize;
        z.cell_zone[c] = zi as u32;
        z.ncell[zi] += 1;
        z.pop[zi] += city.pop[c] as f64;
        z.jobs[zi] += city.jobs[c] as f64;
    }
    z
}

pub struct Gravity {
    /// T_IJ = row[I] * col[J] * f(I, J), with row = A_I O_I and col = B_J D_J.
    pub row: Vec<f64>,
    pub col: Vec<f64>,
    /// f by grid offset: index (dx + offx) * th + (dy + offy).
    pub ftab: Vec<f32>,
    /// Walk share of non-rail trips by grid offset, same index as `ftab` (`Params::walk_share`);
    /// empty until `set_walk` fills it.
    pub wtab: Vec<f32>,
    pub th: i32,
    pub offx: i32,
    pub offy: i32,
    pub iters: usize,
    pub max_err: f64,
    pub mean_km: f64,
    pub total: f64,
}

impl Gravity {
    /// Index offset to add to `key(J)` for origin zone I.
    #[inline]
    pub fn base(&self, z: &Zones, i: usize) -> i32 {
        (self.offx - z.gx[i]) * self.th + self.offy - z.gy[i]
    }
    #[inline]
    pub fn key(&self, z: &Zones, j: usize) -> i32 {
        z.gx[j] * self.th + z.gy[j]
    }
    pub fn t(&self, z: &Zones, i: usize, j: usize) -> f64 {
        self.row[i] * self.col[j] * self.ftab[(self.base(z, i) + self.key(z, j)) as usize] as f64
    }

    /// Fill `wtab` from the zone-centre distances (the same ones `decay_table` uses).
    pub fn set_walk(&mut self, z: &Zones, p: &Params) {
        let (_, dtab) = decay_table(z.w, z.h, z.zone_m, 1.0, 0.0);
        self.wtab = dtab.iter().map(|&d| p.walk_share(d)).collect();
    }

    /// All-day trips that would walk with no rail at all: sum over zone pairs of T_IJ times the
    /// walk share at their distance.
    pub fn walk_trips(&self, z: &Zones) -> f64 {
        self.walk_trips_by_zone(z).0.iter().sum()
    }

    /// The same by home zone and by work zone. Only offsets where the share is not negligible
    /// are visited.
    pub fn walk_trips_by_zone(&self, z: &Zones) -> (Vec<f64>, Vec<f64>) {
        let (mut home, mut work) = (vec![0.0f64; z.n], vec![0.0f64; z.n]);
        if self.wtab.is_empty() {
            return (home, work);
        }
        let mut at = vec![NONE; (z.w * z.h) as usize];
        for j in 0..z.n {
            at[(z.gx[j] * z.h + z.gy[j]) as usize] = j as u32;
        }
        // reach: the largest offset (in zones) along either axis whose share is above 1e-6
        let mut r = 0;
        while r < z.w.max(z.h) && self.wtab[((r + self.offx) * self.th + self.offy) as usize] > 1e-6 {
            r += 1;
        }
        for i in 0..z.n {
            if self.row[i] == 0.0 {
                continue;
            }
            let base = self.base(z, i);
            for gx in (z.gx[i] - r).max(0)..=(z.gx[i] + r).min(z.w - 1) {
                for gy in (z.gy[i] - r).max(0)..=(z.gy[i] + r).min(z.h - 1) {
                    let j = at[(gx * z.h + gy) as usize];
                    if j == NONE {
                        continue;
                    }
                    let k = (base + gx * self.th + gy) as usize;
                    let t = self.row[i] * self.col[j as usize] * (self.ftab[k] * self.wtab[k]) as f64;
                    home[i] += t;
                    work[j as usize] += t;
                }
            }
        }
        (home, work)
    }
}

/// Decay f and distance d (km) by zone grid offset, index (dx + w - 1) * (2h - 1) + (dy + h - 1).
/// Zone-centre distance; within a zone 0.52 x the zone side (mean distance between two random
/// points of a square). The one definition of the decay: the pipeline's solve and the pack
/// loader both use it, so shipped factors and rebuilt trips agree exactly.
pub fn decay_table(w: i32, h: i32, zone_m: f32, decay_km: f32, decay_pow: f32) -> (Vec<f32>, Vec<f32>) {
    let (tw, th) = (2 * w - 1, 2 * h - 1);
    let (offx, offy) = (w - 1, h - 1);
    let zk = zone_m / 1000.0;
    let mut ftab = vec![0f32; (tw * th) as usize];
    let mut dtab = vec![0f32; (tw * th) as usize];
    for dx in -offx..=offx {
        for dy in -offy..=offy {
            let d = if dx == 0 && dy == 0 { 0.52 * zk } else { zk * ((dx * dx + dy * dy) as f32).sqrt() };
            let i = ((dx + offx) * th + dy + offy) as usize;
            ftab[i] = (-d / decay_km).exp() * if decay_pow != 0.0 { d.powf(-decay_pow) } else { 1.0 };
            dtab[i] = d;
        }
    }
    (ftab, dtab)
}

/// Zones and gravity for a city: from the pack when it ships them for these parameters (city
/// open, a few ms), otherwise solved here (seconds). Returns whether the pack's were used.
pub fn city_gravity(city: &City, p: &Params) -> (Zones, Gravity, bool) {
    if let Some(pg) = &city.gravity {
        if (pg.zone_m - p.zone_m).abs() < 0.5 && (pg.decay_km - p.decay_km).abs() < 1e-6 && (pg.decay_pow - p.decay_pow).abs() < 1e-6 {
            let z = zones_from_pack(city, pg);
            let mut g = gravity_from_pack(&z, pg);
            g.set_walk(&z, p);
            return (z, g, true);
        }
    }
    let z = build_zones(city, p.zone_m);
    let mut g = gravity(&z, p, 100, 1e-3);
    g.set_walk(&z, p);
    (z, g, false)
}

pub fn zones_from_pack(city: &City, pg: &PackGravity) -> Zones {
    let n = pg.gx.len();
    let mut z = Zones {
        zone_m: pg.zone_m,
        n,
        cell_zone: pg.cell_zone.clone(),
        gx: pg.gx.clone(),
        gy: pg.gy.clone(),
        w: pg.grid_w,
        h: pg.grid_h,
        x0: pg.grid_x0,
        y0: pg.grid_y0,
        pop: vec![0.0; n],
        jobs: vec![0.0; n],
        ncell: vec![0; n],
    };
    for c in 0..city.len() {
        let zi = z.cell_zone[c] as usize;
        z.ncell[zi] += 1;
        z.pop[zi] += city.pop[c] as f64;
        z.jobs[zi] += city.jobs[c] as f64;
    }
    z
}

pub fn gravity_from_pack(z: &Zones, pg: &PackGravity) -> Gravity {
    let (ftab, _) = decay_table(z.w, z.h, z.zone_m, pg.decay_km, pg.decay_pow);
    Gravity {
        row: pg.row.clone(),
        col: pg.col.clone(),
        ftab,
        wtab: vec![],
        th: 2 * z.h - 1,
        offx: z.w - 1,
        offy: z.h - 1,
        iters: pg.iters,
        max_err: pg.max_err,
        mean_km: pg.mean_km,
        total: pg.trips,
    }
}

/// What the pipeline ships (pack format 1).
pub fn to_pack_gravity(z: &Zones, g: &Gravity, p: &Params) -> PackGravity {
    PackGravity {
        zone_m: z.zone_m,
        decay_km: p.decay_km,
        decay_pow: p.decay_pow,
        grid_x0: z.x0,
        grid_y0: z.y0,
        grid_w: z.w,
        grid_h: z.h,
        cell_zone: z.cell_zone.clone(),
        gx: z.gx.clone(),
        gy: z.gy.clone(),
        row: g.row.clone(),
        col: g.col.clone(),
        iters: g.iters,
        max_err: g.max_err,
        mean_km: g.mean_km,
        trips: g.total,
    }
}

/// Doubly constrained gravity by Furness balancing. Workers = population scaled to total jobs.
/// The decay depends on zone-centre distance only, so f is a small table by grid offset.
pub fn gravity(z: &Zones, p: &Params, max_iters: usize, tol: f64) -> Gravity {
    let (w, h) = (z.w, z.h);
    let th = 2 * h - 1;
    let (offx, offy) = (w - 1, h - 1);
    let (ftab, dtab) = decay_table(w, h, z.zone_m, p.decay_km, p.decay_pow);
    let tp: f64 = z.pop.iter().sum();
    let tj: f64 = z.jobs.iter().sum();
    let o: Vec<f64> = z.pop.iter().map(|&v| v * tj / tp).collect();
    let rows: Vec<usize> = (0..z.n).filter(|&i| o[i] > 0.0).collect();
    let cols: Vec<usize> = (0..z.n).filter(|&j| z.jobs[j] > 0.0).collect();
    let ckey: Vec<i32> = cols.iter().map(|&j| z.gx[j] * th + z.gy[j]).collect();
    let mut g = Gravity { row: vec![0.0; z.n], col: z.jobs.clone(), ftab, wtab: vec![], th, offx, offy, iters: 0, max_err: 0.0, mean_km: 0.0, total: 0.0 };
    let mut colc: Vec<f64> = cols.iter().map(|&j| g.col[j]).collect();
    let mut acc = vec![0f64; cols.len()];
    for it in 0..max_iters {
        // rows: A_I = 1 / sum_J col_J f_IJ
        let mut err: f64 = 0.0;
        for &i in &rows {
            let base = g.base(z, i);
            let mut s = 0.0f64;
            for (k, &key) in ckey.iter().enumerate() {
                s += colc[k] * g.ftab[(base + key) as usize] as f64;
            }
            if it > 0 {
                err = err.max((g.row[i] * s / o[i] - 1.0).abs());
            }
            g.row[i] = o[i] / s;
        }
        g.iters = it + 1;
        g.max_err = err;
        if it > 0 && err < tol {
            break;
        }
        // columns: B_J = 1 / sum_I row_I f_IJ
        acc.iter_mut().for_each(|v| *v = 0.0);
        for &i in &rows {
            let base = g.base(z, i);
            let r = g.row[i];
            for (k, &key) in ckey.iter().enumerate() {
                acc[k] += r * g.ftab[(base + key) as usize] as f64;
            }
        }
        for (k, &j) in cols.iter().enumerate() {
            colc[k] = z.jobs[j] / acc[k];
            g.col[j] = colc[k];
        }
    }
    // mean trip length
    let (mut sum, mut sumd) = (0.0f64, 0.0f64);
    for &i in &rows {
        let base = g.base(z, i);
        for (k, &key) in ckey.iter().enumerate() {
            let idx = (base + key) as usize;
            let t = g.row[i] * colc[k] * g.ftab[idx] as f64;
            sum += t;
            sumd += t * dtab[idx] as f64;
        }
    }
    g.total = sum;
    g.mean_km = sumd / sum;
    g
}

// ---------------------------------------------------------------------------------------------
// 2. Station access per cell

pub struct Access {
    pub n: Vec<u8>,
    /// K per cell, nearest first.
    pub st: Vec<u32>,
    /// Real walk minutes, K per cell.
    pub walk: Vec<f32>,
    pub cells_with_access: usize,
}

/// Station lookup grid with buckets the size of the walk cutoff.
struct StationGrid {
    minx: f32,
    miny: f32,
    size: f32,
    w: i32,
    h: i32,
    off: Vec<u32>,
    items: Vec<u32>,
}

impl StationGrid {
    /// Buckets of `size` metres over the stations' extent plus `pad` metres all round.
    fn new(net: &Network, size: f32, pad: f32) -> Self {
        let minx = net.st_x.iter().cloned().fold(f32::INFINITY, f32::min) - pad;
        let miny = net.st_y.iter().cloned().fold(f32::INFINITY, f32::min) - pad;
        let maxx = net.st_x.iter().cloned().fold(f32::NEG_INFINITY, f32::max) + pad;
        let maxy = net.st_y.iter().cloned().fold(f32::NEG_INFINITY, f32::max) + pad;
        let w = ((maxx - minx) / size) as i32 + 1;
        let h = ((maxy - miny) / size) as i32 + 1;
        let b = |s: usize| (((net.st_x[s] - minx) / size) as i32 * h + ((net.st_y[s] - miny) / size) as i32) as usize;
        let mut off = vec![0u32; (w * h + 1) as usize];
        for s in 0..net.n_stations() {
            off[b(s) + 1] += 1;
        }
        for i in 0..(w * h) as usize {
            off[i + 1] += off[i];
        }
        let mut fill = off.clone();
        let mut items = vec![0u32; net.n_stations()];
        for s in 0..net.n_stations() {
            let k = b(s);
            items[fill[k] as usize] = s as u32;
            fill[k] += 1;
        }
        StationGrid { minx, miny, size, w, h, off, items }
    }
    fn near(&self, x: f32, y: f32, mut f: impl FnMut(u32)) {
        let bx = ((x - self.minx) / self.size).floor() as i32;
        let by = ((y - self.miny) / self.size).floor() as i32;
        for gx in (bx - 1).max(0)..=(bx + 1).min(self.w - 1) {
            for gy in (by - 1).max(0)..=(by + 1).min(self.h - 1) {
                let k = (gx * self.h + gy) as usize;
                for &s in &self.items[self.off[k] as usize..self.off[k + 1] as usize] {
                    f(s);
                }
            }
        }
    }
}

/// Bucket size of the cell grid the access search scatters stations over.
const ACCESS_BUCKET_M: f32 = 1000.0;

/// K nearest stations per cell within `walk_cutoff_m`, by scattering each station over the
/// cells near it (a station touches the few hundred cells within the bound, so a far bound
/// costs little; searching outwards from every cell cost 5-15x as much where stations are
/// sparse).
pub fn build_access(city: &City, net: &Network, p: &Params) -> Access {
    let n = city.len();
    let mut a = Access { n: vec![0; n], st: vec![NONE; n * K], walk: vec![f32::INFINITY; n * K], cells_with_access: 0 };
    if net.n_stations() == 0 || n == 0 {
        return a;
    }
    // cells bucketed on a grid (CSR)
    let size = ACCESS_BUCKET_M;
    let minx = city.x.iter().cloned().fold(f32::INFINITY, f32::min);
    let miny = city.y.iter().cloned().fold(f32::INFINITY, f32::min);
    let maxx = city.x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let maxy = city.y.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let w = ((maxx - minx) / size) as i32 + 1;
    let h = ((maxy - miny) / size) as i32 + 1;
    let bucket = |x: f32, y: f32| ((((x - minx) / size) as i32).clamp(0, w - 1) * h + (((y - miny) / size) as i32).clamp(0, h - 1)) as usize;
    let mut off = vec![0u32; (w * h + 1) as usize];
    for c in 0..n {
        off[bucket(city.x[c], city.y[c]) + 1] += 1;
    }
    for i in 0..(w * h) as usize {
        off[i + 1] += off[i];
    }
    let mut fill = off.clone();
    let mut cells = vec![0u32; n];
    for c in 0..n {
        let b = bucket(city.x[c], city.y[c]);
        cells[fill[b] as usize] = c as u32;
        fill[b] += 1;
    }
    drop(fill);
    // each station into the sorted K lists (distance squared for now) of the cells near it
    let cut = p.walk_cutoff_m;
    let cut2 = cut * cut;
    for s in 0..net.n_stations() {
        let (sx, sy) = (net.st_x[s], net.st_y[s]);
        let gx0 = (((sx - cut - minx) / size).floor() as i32).max(0);
        let gx1 = (((sx + cut - minx) / size).floor() as i32).min(w - 1);
        let gy0 = (((sy - cut - miny) / size).floor() as i32).max(0);
        let gy1 = (((sy + cut - miny) / size).floor() as i32).min(h - 1);
        for gx in gx0..=gx1 {
            for gy in gy0..=gy1 {
                let k = (gx * h + gy) as usize;
                for &c in &cells[off[k] as usize..off[k + 1] as usize] {
                    let c = c as usize;
                    let d2 = (city.x[c] - sx).powi(2) + (city.y[c] - sy).powi(2);
                    let cnt = a.n[c] as usize;
                    let wk = &mut a.walk[c * K..c * K + K];
                    if d2 > cut2 || (cnt == K && d2 >= wk[K - 1]) {
                        continue;
                    }
                    let st = &mut a.st[c * K..c * K + K];
                    let mut i = if cnt < K { cnt } else { K - 1 };
                    while i > 0 && wk[i - 1] > d2 {
                        wk[i] = wk[i - 1];
                        st[i] = st[i - 1];
                        i -= 1;
                    }
                    wk[i] = d2;
                    st[i] = s as u32;
                    if cnt < K {
                        a.n[c] += 1;
                    }
                }
            }
        }
    }
    for c in 0..n {
        let cnt = a.n[c] as usize;
        for wv in a.walk[c * K..c * K + cnt].iter_mut() {
            *wv = p.walk_min(wv.sqrt());
        }
        if cnt > 0 {
            a.cells_with_access += 1;
        }
    }
    a
}

// ---------------------------------------------------------------------------------------------
// 3. Access subzones

pub struct Subzones {
    pub n: usize,
    pub zone: Vec<u32>,
    pub pop: Vec<f32>,
    pub jobs: Vec<f32>,
    pub cx: Vec<f32>,
    pub cy: Vec<f32>,
    /// CSR access list (the fast walk, SPEC 4.3): station, perceived minutes (the members'
    /// `Params::access_cost`, pop/jobs-weighted mean), and the weighted mean straight-line
    /// distance in metres. Used at both ends of a trip.
    pub acc_off: Vec<u32>,
    pub acc_st: Vec<u32>,
    pub acc_w: Vec<f32>,
    pub acc_d: Vec<f32>,
    /// Per zone, subzones with commuters living there (origins) and with jobs (destinations).
    pub zo_off: Vec<u32>,
    pub zo: Vec<u32>,
    pub zd_off: Vec<u32>,
    pub zd: Vec<u32>,
    /// Subzone of each cell (NONE for cells with no station within reach).
    pub cell: Vec<u32>,
    /// CSR: the cells of each subzone (`mem[mem_off[u]..mem_off[u + 1]]`).
    pub mem_off: Vec<u32>,
    pub mem: Vec<u32>,
}

pub fn build_subzones(city: &City, zones: &Zones, acc: &Access, net: &Network, p: &Params) -> Subzones {
    let n = city.len();
    let mut cell_sub = vec![NONE; n];
    let mut map: HashMap<u64, u32> = HashMap::new();
    let mut nsub = 0u32;
    for c in 0..n {
        if acc.n[c] == 0 || city.pop[c] + city.jobs[c] <= 0.0 {
            continue; // no station within reach, or nobody commutes from or to here
        }
        let key = if p.band_min < 0.0 {
            c as u64
        } else {
            let band = if p.band_min > 0.0 { (acc.walk[c * K] / p.band_min) as u64 } else { 0 };
            ((zones.cell_zone[c] as u64) << 32) | ((acc.st[c * K] as u64) << 8) | band.min(255)
        };
        let id = *map.entry(key).or_insert_with(|| {
            nsub += 1;
            nsub - 1
        });
        cell_sub[c] = id;
    }
    drop(map);
    let ns = nsub as usize;
    // members CSR
    let mut moff = vec![0u32; ns + 1];
    for c in 0..n {
        if cell_sub[c] != NONE {
            moff[cell_sub[c] as usize + 1] += 1;
        }
    }
    for i in 0..ns {
        moff[i + 1] += moff[i];
    }
    let mut fill = moff.clone();
    let mut mem = vec![0u32; moff[ns] as usize];
    for c in 0..n {
        let s = cell_sub[c];
        if s != NONE {
            mem[fill[s as usize] as usize] = c as u32;
            fill[s as usize] += 1;
        }
    }
    let mut sz = Subzones {
        n: ns,
        zone: vec![0; ns],
        pop: vec![0.0; ns],
        jobs: vec![0.0; ns],
        cx: vec![0.0; ns],
        cy: vec![0.0; ns],
        acc_off: vec![0],
        acc_st: vec![],
        acc_w: vec![],
        acc_d: vec![],
        zo_off: vec![],
        zo: vec![],
        zd_off: vec![],
        zd: vec![],
        cell: vec![],
        mem_off: vec![],
        mem: vec![],
    };
    // candidates: (mean perceived minutes, station, mean straight-line metres)
    let mut cand: Vec<(f32, u32, f32)> = Vec::with_capacity(32);
    for s in 0..ns {
        let m = &mem[moff[s] as usize..moff[s + 1] as usize];
        let (mut wsum, mut x, mut y) = (0.0f64, 0.0f64, 0.0f64);
        for &c in m {
            let c = c as usize;
            let wt = (city.pop[c] + city.jobs[c]).max(1e-6) as f64;
            wsum += wt;
            x += wt * city.x[c] as f64;
            y += wt * city.y[c] as f64;
            sz.pop[s] += city.pop[c];
            sz.jobs[s] += city.jobs[c];
        }
        sz.zone[s] = zones.cell_zone[m[0] as usize];
        sz.cx[s] = (x / wsum) as f32;
        sz.cy[s] = (y / wsum) as f32;
        // candidates: union of the members' lists; distance = weighted mean over members
        cand.clear();
        for &c in m {
            let c = c as usize;
            for k in 0..acc.n[c] as usize {
                let st = acc.st[c * K + k];
                if !cand.iter().any(|&(_, t, _)| t == st) {
                    cand.push((0.0, st, 0.0));
                }
            }
        }
        for e in cand.iter_mut() {
            let (sx, sy) = (net.st_x[e.1 as usize], net.st_y[e.1 as usize]);
            let (mut wd, mut wc) = (0.0f64, 0.0f64);
            for &c in m {
                let c = c as usize;
                let wt = (city.pop[c] + city.jobs[c]).max(1e-6) as f64;
                let d = ((city.x[c] - sx).powi(2) + (city.y[c] - sy).powi(2)).sqrt();
                wd += wt * d as f64;
                wc += wt * (-(p.beta * p.access_cost(d)) as f64).exp();
            }
            e.2 = (wd / wsum) as f32;
            // the members' perceived walks averaged as the mode choice would weigh them (a
            // logsum at its scale): the cost grows faster than the distance, and the near
            // members carry most of the rail (T-090: a plain mean put rail 3.3% under one
            // subzone per cell on the hand-made network, this 0.1% over)
            e.0 = (-(wc / wsum).ln() / p.beta as f64) as f32;
        }
        cand.retain(|e| e.2 <= p.walk_cutoff_m);
        if cand.is_empty() {
            // every member is within reach of its own nearest station, so the mean distance to
            // that station is too; this only guards against rounding
            let c = m[0] as usize;
            let d = acc.walk[c * K] * p.walk_mps * 60.0 / p.walk_detour;
            cand.push((p.access_cost(d), acc.st[c * K], d));
        }
        cand.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        for &(wm, st, d) in cand.iter().take(K) {
            sz.acc_st.push(st);
            sz.acc_w.push(wm);
            sz.acc_d.push(d);
        }
        sz.acc_off.push(sz.acc_st.len() as u32);
    }
    // zone -> subzone lists
    let csr = |want: &dyn Fn(usize) -> bool| {
        let mut off = vec![0u32; zones.n + 1];
        for s in 0..ns {
            if want(s) {
                off[sz.zone[s] as usize + 1] += 1;
            }
        }
        for i in 0..zones.n {
            off[i + 1] += off[i];
        }
        let mut f = off.clone();
        let mut l = vec![0u32; off[zones.n] as usize];
        for s in 0..ns {
            if want(s) {
                let z = sz.zone[s] as usize;
                l[f[z] as usize] = s as u32;
                f[z] += 1;
            }
        }
        (off, l)
    };
    let (zo_off, zo) = csr(&|s| sz.pop[s] > 0.0);
    let (zd_off, mut zd) = csr(&|s| sz.jobs[s] > 0.0);
    // destinations in each zone by their cheapest walk, so mode choice can stop at
    // `Params::pair_walk_max`
    for j in 0..zones.n {
        zd[zd_off[j] as usize..zd_off[j + 1] as usize].sort_by(|&a, &b| sz.acc_w[sz.acc_off[a as usize] as usize].partial_cmp(&sz.acc_w[sz.acc_off[b as usize] as usize]).unwrap());
    }
    sz.zo_off = zo_off;
    sz.zo = zo;
    sz.zd_off = zd_off;
    sz.zd = zd;
    sz.cell = cell_sub;
    sz.mem_off = moff;
    sz.mem = mem;
    sz
}

// ---------------------------------------------------------------------------------------------
// 4. Network graph and station-to-station search

pub struct Graph {
    pub n_st: usize,
    pub n: usize,
    /// Route nodes (index node - n_st): one per stop per direction per line.
    pub rn_station: Vec<u32>,
    pub rn_line: Vec<u32>,
    /// Real ride minutes to the next stop in this direction (0 at the end).
    pub rn_ride: Vec<f32>,
    pub adj_off: Vec<u32>,
    pub adj_to: Vec<u32>,
    pub adj_cost: Vec<f32>,
    /// For ride edges, the route node the segment starts at; NONE otherwise.
    pub adj_seg: Vec<u32>,
}

pub fn board_cost(headway_min: f32, p: &Params) -> f32 {
    let half = headway_min / 2.0;
    half.min(p.platform_wait_cap_min) * p.w_wait + (half - p.platform_wait_cap_min).max(0.0) * p.w_home_wait + p.transfer_min
}

pub fn build_graph(net: &Network, p: &Params) -> Graph {
    let s = net.n_stations();
    let mut rn_station = vec![];
    let mut rn_line = vec![];
    let mut rn_ride = vec![];
    let mut rn_next = vec![];
    for l in 0..net.n_lines() {
        let (a, b) = (net.line_off[l] as usize, net.line_off[l + 1] as usize);
        for dir in 0..2 {
            let base = s + rn_station.len();
            let len = b - a;
            for i in 0..len {
                let k = if dir == 0 { a + i } else { b - 1 - i };
                rn_station.push(net.line_stops[k]);
                rn_line.push(l as u32);
                let ride = if i + 1 < len {
                    if dir == 0 {
                        net.hop_min[k]
                    } else if net.hop_back.is_empty() {
                        net.hop_min[k - 1]
                    } else {
                        net.hop_back[k]
                    }
                } else {
                    0.0
                };
                rn_ride.push(ride);
                rn_next.push(if i + 1 < len { (base + i + 1) as u32 } else { NONE });
            }
        }
    }
    let n = s + rn_station.len();
    let mut edges: Vec<(u32, u32, f32, u32)> = vec![];
    for (r, &st) in rn_station.iter().enumerate() {
        let node = (s + r) as u32;
        let l = rn_line[r] as usize;
        if rn_next[r] != NONE {
            edges.push((st, node, board_cost(net.line_headway_min[l][p.period], p), NONE));
            edges.push((node, rn_next[r], rn_ride[r], node));
        }
        edges.push((node, st, 0.0, NONE));
    }
    // walking transfers: from each route node (having ridden) to the stations nearby
    if p.xfer_walk_m > 0.0 && s > 1 {
        let grid = StationGrid::new(net, p.xfer_walk_m, p.xfer_walk_m);
        let cut2 = p.xfer_walk_m * p.xfer_walk_m;
        let mut near: Vec<Vec<(u32, f32)>> = vec![vec![]; s];
        for a in 0..s {
            let (x, y) = (net.st_x[a], net.st_y[a]);
            grid.near(x, y, |b| {
                let d2 = (net.st_x[b as usize] - x).powi(2) + (net.st_y[b as usize] - y).powi(2);
                if b as usize != a && d2 <= cut2 {
                    near[a].push((b, p.xfer_min(d2.sqrt()) * p.w_xfer));
                }
            });
        }
        for (r, &st) in rn_station.iter().enumerate() {
            for &(b, c) in &near[st as usize] {
                edges.push(((s + r) as u32, b, c, NONE));
            }
        }
    }
    edges.sort_by_key(|e| e.0);
    let mut adj_off = vec![0u32; n + 1];
    for e in &edges {
        adj_off[e.0 as usize + 1] += 1;
    }
    for i in 0..n {
        adj_off[i + 1] += adj_off[i];
    }
    Graph {
        n_st: s,
        n,
        rn_station,
        rn_line,
        rn_ride,
        adj_off,
        adj_to: edges.iter().map(|e| e.1).collect(),
        adj_cost: edges.iter().map(|e| e.2).collect(),
        adj_seg: edges.iter().map(|e| e.3).collect(),
    }
}

/// Ride edges cost real minutes times a crowding factor per segment (indexed by route node).
pub fn set_crowding(g: &mut Graph, factor: &[f32]) {
    for e in 0..g.adj_to.len() {
        let seg = g.adj_seg[e];
        if seg != NONE {
            let r = seg as usize - g.n_st;
            g.adj_cost[e] = g.rn_ride[r] * factor[r];
        }
    }
}

pub struct Skims {
    pub s: usize,
    pub n: usize,
    /// Generalised minutes station to station (wait + ride + transfers), S x S.
    pub r: Vec<f32>,
    /// Shortest-path tree per source: predecessor node, S x N.
    pub pred: Vec<u32>,
    /// Settle order per source, S x N (first `n_order[s]` valid).
    pub order: Vec<u32>,
    pub n_order: Vec<u32>,
}

pub fn skims(g: &Graph, p: &Params) -> Skims {
    let (s, n) = (g.n_st, g.n);
    let mut sk = Skims { s, n, r: vec![f32::INFINITY; s * s], pred: vec![NONE; s * n], order: vec![0; s * n], n_order: vec![0; s] };
    let mut dist = vec![f32::INFINITY; n];
    let mut done = vec![false; n];
    let mut heap: BinaryHeap<Reverse<(u32, u32)>> = BinaryHeap::with_capacity(n);
    for src in 0..s {
        dist.iter_mut().for_each(|d| *d = f32::INFINITY);
        done.iter_mut().for_each(|d| *d = false);
        let pred = &mut sk.pred[src * n..(src + 1) * n];
        let order = &mut sk.order[src * n..(src + 1) * n];
        let mut no = 0usize;
        dist[src] = 0.0;
        heap.push(Reverse((0f32.to_bits(), src as u32)));
        while let Some(Reverse((db, v))) = heap.pop() {
            let v = v as usize;
            if done[v] {
                continue;
            }
            done[v] = true;
            order[no] = v as u32;
            no += 1;
            let d = f32::from_bits(db);
            for e in g.adj_off[v] as usize..g.adj_off[v + 1] as usize {
                let to = g.adj_to[e] as usize;
                let nd = d + g.adj_cost[e];
                if nd < dist[to] {
                    dist[to] = nd;
                    pred[to] = v as u32;
                    // non-negative floats order like their bit patterns
                    heap.push(Reverse((nd.to_bits(), to as u32)));
                }
            }
        }
        sk.n_order[src] = no as u32;
        let row = &mut sk.r[src * s..(src + 1) * s];
        for t in 0..s {
            // every path boards at least once; the first boarding is not a transfer
            row[t] = if t == src { 0.0 } else { dist[t] - p.transfer_min };
        }
    }
    sk
}

// ---------------------------------------------------------------------------------------------
// 5. Mode choice

/// exp(x) to ~2e-4 relative: 2^int by exponent bits, 2^frac by a degree-5 polynomial. wasm has
/// no exp instruction, and libm's exp was the larger part of the mode choice loop there.
#[inline]
pub fn fast_exp(x: f32) -> f32 {
    let t = x.clamp(-80.0, 80.0) * std::f32::consts::LOG2_E;
    let i = t.floor();
    let f = t - i;
    let p = 1.0 + f * (0.693_147_2 + f * (0.240_226_5 + f * (0.055_504_11 + f * (0.009_618_129 + f * 0.001_333_355_8))));
    f32::from_bits(((i as i32 + 127) << 23) as u32) * p
}

/// ln(x) for normal x > 0, to 3e-6 absolute: exponent bits, then a degree-6 polynomial in
/// t = m - 1 with the mantissa m in [0.71, 1.41) (least-squares fit, notes/T-020.md). Same reason
/// as `fast_exp`: wasm has no log instruction. Branch- and division-free: shifting the bits by
/// those of 1/sqrt(2) folds the mantissa without a data-dependent branch, which in the pair loop
/// mispredicted half the time. In a logsum cost, 3e-6 is 1.5e-5 perceived minutes.
#[inline]
pub fn fast_ln(x: f32) -> f32 {
    const HALF_SQRT2: u32 = 0x3f35_04f3; // 0.70710677
    let b = x.to_bits().wrapping_sub(HALF_SQRT2);
    let e = (b as i32) >> 23;
    let t = f32::from_bits((b & 0x007f_ffff).wrapping_add(HALF_SQRT2)) - 1.0;
    let q = 1.000_004_68 + t * (-0.499_903_616 + t * (0.332_567_778 + t * (-0.253_978_318 + t * (0.220_702_996 + t * -0.143_383_126))));
    e as f32 * std::f32::consts::LN_2 + t * q
}

/// exp(-theta x station-to-station cost), S x S, with a zero diagonal: boarding and alighting at
/// the same station is not a rail trip. The access logsum per origin subzone is built from it.
pub fn station_exp(sk: &Skims, p: &Params) -> Vec<f32> {
    let s = sk.s;
    let mut er: Vec<f32> = sk.r.iter().map(|&r| if r.is_finite() { fast_exp(-p.station_theta * r) } else { 0.0 }).collect();
    for a in 0..s {
        er[a * s + a] = 0.0;
    }
    er
}

/// Distance bands (km, upper edges) for rail trips by how far home is from the boarding station.
pub const DIST_BANDS_KM: [f32; N_BANDS] = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 8.0, f32::INFINITY];
pub const N_BANDS: usize = 13;
pub const N_WALKSUM: usize = 16;
pub const WALKSUM_BAND_MIN: f32 = 10.0;

/// The band of `DIST_BANDS_KM` a straight-line distance in metres falls in.
pub fn dist_band(d_m: f32) -> usize {
    DIST_BANDS_KM.iter().position(|&e| d_m < e * 1000.0).unwrap_or(N_BANDS - 1)
}

pub struct ModeOut {
    /// Rail trips by (access station, egress station), S x S, in the home -> work direction.
    pub od: Vec<f32>,
    pub rail: f64,
    /// Rail trips by straight-line distance from home (subzone centroid) to the boarding station,
    /// bands `DIST_BANDS_KM`.
    pub rail_band: [f64; N_BANDS],
    /// Rail trips by the perceived minutes of the cheapest walks at both ends together, in
    /// bands of `WALKSUM_BAND_MIN` (the last one open-ended): how much rail sits near
    /// `Params::pair_walk_max`.
    pub rail_walksum: [f64; N_WALKSUM],
    /// Trips between subzones offered rail: both ends within the walk's bound of a station, and
    /// at least one pair of the two zones under `Params::pair_walk_max`.
    pub trips_access: f64,
    /// All home->work trips in the period.
    pub trips_total: f64,
    /// Rail trips times the walk share at their zone distance (`Gravity::wtab`): what the walked
    /// trips lose to rail. Walked trips = share x `Gravity::walk_trips` - this.
    pub rail_walk: f64,
    pub zone_pairs: u64,
    pub pairs: u64,
    pub pruned_trips: f64,
    /// Rail trips by home (origin) subzone and by work (destination) subzone.
    pub rail_by_sub: Vec<f32>,
    pub rail_to_sub: Vec<f32>,
    /// `rail_walk` by home subzone and by work subzone.
    pub walk_lost_by_sub: Vec<f32>,
    pub walk_lost_to_sub: Vec<f32>,
    /// Rail trips per access entry (`Subzones::acc_st`): at the home end, boarding at that
    /// station from that subzone; at the work end, leaving the train there for that subzone.
    pub acc_home: Vec<f32>,
    pub acc_work: Vec<f32>,
}

/// Mode choice per (origin subzone, destination subzone) pair, with a logit over access and
/// egress stations (SPEC 4.3, 4.4; notes/T-020.md, T-078.md).
///
/// For a pair (u, v) the rail alternatives are every (access station a of u, egress station t of
/// v); each costs access_a + R[a, t] + egress_t. Their logsum separates:
///     S_uv = sum_t A_u[t] E_v[t],  A_u[t] = sum_a exp(-theta (access_a - m_u)) exp(-theta R[a, t]),
///     E_v[t] = exp(-theta (egress_t - m_v)),  rail cost = m_u + m_v + fare - ln(S_uv) / theta.
/// A_u is one row of S per origin subzone, built once per subzone (access options x S
/// multiply-adds), so the pair loop costs one product per egress station plus one ln and one
/// exp. Rail trips are kept by (u, egress station) as Y_u[t] = sum_v x_uv E_v[t] / S_uv and split
/// back onto access stations once per subzone: od[a, t] += Y_u[t] w_a ER[a, t].
pub fn mode_choice(z: &Zones, gr: &Gravity, sz: &Subzones, er: &[f32], s: usize, p: &Params) -> ModeOut {
    let share = p.period_share as f64;
    let th = p.station_theta;
    // per subzone: its part of the zone's row / column factor
    let ou: Vec<f32> = (0..sz.n)
        .map(|u| {
            let zi = sz.zone[u] as usize;
            if z.pop[zi] > 0.0 { (gr.row[zi] * sz.pop[u] as f64 / z.pop[zi] * share) as f32 } else { 0.0 }
        })
        .collect();
    let zone_area = (z.zone_m / 1000.0).powi(2);
    let dv: Vec<f32> = (0..sz.n)
        .map(|v| {
            let zj = sz.zone[v] as usize;
            if z.jobs[zj] > 0.0 { (gr.col[zj] * sz.jobs[v] as f64 / z.jobs[zj]) as f32 } else { 0.0 }
        })
        .collect();
    let park: Vec<f32> = (0..sz.n)
        .map(|v| {
            let dens = z.jobs[sz.zone[v] as usize] as f32 / zone_area;
            p.park_min + (dens / p.park_jobs_per_km2_per_min).min(p.park_cost_cap_min)
        })
        .collect();
    // per subzone and access entry: offset m and weights exp(-theta (cost - m)), the same at
    // both ends (one access mode)
    let mut acc_m = vec![0f32; sz.n];
    let mut acc_e = vec![0f32; sz.acc_st.len()];
    for v in 0..sz.n {
        let r = sz.acc_off[v] as usize..sz.acc_off[v + 1] as usize;
        if r.is_empty() {
            continue;
        }
        let m = sz.acc_w[r.clone()].iter().cloned().fold(f32::INFINITY, f32::min);
        acc_m[v] = m;
        for k in r {
            acc_e[k] = fast_exp(-th * (sz.acc_w[k] - m));
        }
    }
    // per subzone: minutes a km on its local streets and main roads (by its zone's density)
    let dens = z.density();
    let local: Vec<(f32, f32)> = (0..sz.n).map(|v| p.place_min_per_km(dens[sz.zone[v] as usize])).collect();
    let mut out = ModeOut {
        od: vec![0.0; s * s],
        rail: 0.0,
        rail_band: [0.0; N_BANDS],
        rail_walksum: [0.0; N_WALKSUM],
        trips_access: 0.0,
        trips_total: gr.total * share,
        rail_walk: 0.0,
        zone_pairs: 0,
        pairs: 0,
        pruned_trips: 0.0,
        rail_by_sub: vec![0.0; sz.n],
        rail_to_sub: vec![0.0; sz.n],
        walk_lost_by_sub: vec![0.0; sz.n],
        walk_lost_to_sub: vec![0.0; sz.n],
        acc_home: vec![0.0; sz.acc_st.len()],
        acc_work: vec![0.0; sz.acc_st.len()],
    };
    let ozones: Vec<usize> = (0..z.n).filter(|&i| sz.zo_off[i + 1] > sz.zo_off[i]).collect();
    // Destination zones by their cheapest walk (their first subzone: `build_subzones` sorts
    // them), so an origin visits only the prefix within `pair_walk_max` of its own walk: in
    // 10-minute steps of that walk, in zone order within a step (zone order keeps the decay
    // table lookups near each other; sorting by the walk alone was 20% slower).
    let mut dzones: Vec<usize> = (0..z.n).filter(|&j| sz.zd_off[j + 1] > sz.zd_off[j]).collect();
    let zmin = |j: usize| acc_m[sz.zd[sz.zd_off[j] as usize] as usize];
    dzones.sort_by_key(|&j| ((zmin(j) * 0.1) as u32, j));
    let dstep: Vec<u32> = dzones.iter().map(|&j| (zmin(j) * 0.1) as u32).collect();
    // Everything the pair loop reads about a destination, packed in visiting order (T-036's
    // "packed per-destination layout"): per zone, then per subzone, then its egress stations.
    struct DZone {
        key: i32,
        min: f32,
        dv: f32,
        q0: u32,
        q1: u32,
    }
    #[derive(Clone, Copy)]
    struct Dst {
        x: f32,
        y: f32,
        loc: (f32, f32),
        park: f32,
        dv: f32,
        m: f32,
        v: u32,
        e0: u32,
        e1: u32,
    }
    let mut dz: Vec<DZone> = Vec::with_capacity(dzones.len());
    let mut dst: Vec<Dst> = Vec::with_capacity(sz.zd.len());
    // egress stations and their weights; `eg_k` is each one's index in `Subzones::acc_st`
    let (mut eg_st, mut eg_e, mut eg_k): (Vec<u32>, Vec<f32>, Vec<u32>) = (vec![], vec![], vec![]);
    for &j in &dzones {
        let q0 = dst.len() as u32;
        let mut dvz = 0.0f32;
        for &v in &sz.zd[sz.zd_off[j] as usize..sz.zd_off[j + 1] as usize] {
            let v = v as usize;
            let e0 = eg_st.len() as u32;
            for k in sz.acc_off[v] as usize..sz.acc_off[v + 1] as usize {
                eg_st.push(sz.acc_st[k]);
                eg_e.push(acc_e[k]);
                eg_k.push(k as u32);
            }
            dvz += dv[v];
            dst.push(Dst { x: sz.cx[v], y: sz.cy[v], loc: local[v], park: park[v], dv: dv[v], m: acc_m[v], v: v as u32, e0, e1: eg_st.len() as u32 });
        }
        dz.push(DZone { key: gr.key(z, j), min: zmin(j), dv: dvz, q0, q1: dst.len() as u32 });
    }
    // the decay and the walk share by zone offset side by side: one lookup per zone pair
    let fw: Vec<[f32; 2]> = gr.ftab.iter().enumerate().map(|(k, &f)| [f, if gr.wtab.is_empty() { 0.0 } else { gr.wtab[k] }]).collect();
    let mut fz = vec![[0f32; 2]; dz.len()];
    let mut a_row = vec![0f32; s];
    let mut y_row = vec![0f32; s];
    // per destination (position in `dst`): rail trips over the logsum, for the origin at hand
    let mut gq = vec![0f32; dst.len()];
    // destinations given rail trips by the current origin
    let mut touched: Vec<u32> = Vec::with_capacity(dst.len());
    let mut walksum = [0f64; N_WALKSUM];
    for &i in &ozones {
        let base = gr.base(z, i);
        let us = &sz.zo[sz.zo_off[i] as usize..sz.zo_off[i + 1] as usize];
        let ou_zone: f32 = us.iter().map(|&u| ou[u as usize]).sum();
        // the decay and walk share to every destination zone, once for all of this zone's
        // origins
        for (o, zj) in fz.iter_mut().zip(&dz) {
            *o = fw[(base + zj.key) as usize];
        }
        for (ui, &u) in us.iter().enumerate() {
            let u = u as usize;
            let opts = sz.acc_off[u] as usize..sz.acc_off[u + 1] as usize;
            let m_u = acc_m[u];
            a_row.iter_mut().for_each(|v| *v = 0.0);
            for k in opts.clone() {
                let (a, w) = (sz.acc_st[k] as usize, acc_e[k]);
                for (av, &e) in a_row.iter_mut().zip(&er[a * s..(a + 1) * s]) {
                    *av += w * e;
                }
            }
            y_row.iter_mut().for_each(|v| *v = 0.0);
            let (ux, uy, lu) = (sz.cx[u], sz.cy[u], local[u]);
            let mut rail_sum = 0.0f32;
            let mut all_sum = 0.0f32;
            let mut rail_walk = 0.0f32;
            let mut n_pairs = 0usize;
            // past this, a destination's walk takes the pair over `pair_walk_max`
            let wmax = p.pair_walk_max - m_u;
            let nz = dstep.partition_point(|&m| m as f32 * 10.0 <= wmax);
            for (zj, &[f, pw]) in dz[..nz].iter().zip(&fz) {
                if zj.min > wmax {
                    continue;
                }
                let rail_before = rail_sum;
                if p.prune_trips > 0.0 && ou_zone * zj.dv * f < p.prune_trips {
                    out.pruned_trips += (ou[u] * zj.dv * f) as f64;
                    continue;
                }
                if ui == 0 {
                    out.zone_pairs += 1;
                }
                let tu = ou[u] * f;
                all_sum += tu * zj.dv;
                let walks = pw > 1e-6;
                // cheapest walk first: past the bound, every later one is too
                let ds = &dst[zj.q0 as usize..zj.q1 as usize];
                let nv = if ds[ds.len() - 1].m <= wmax { ds.len() } else { ds.partition_point(|d| d.m <= wmax) };
                n_pairs += nv;
                for (qi, dd) in ds[..nv].iter().enumerate() {
                    let mut ssum = 0.0f32;
                    for k in dd.e0 as usize..dd.e1 as usize {
                        ssum += a_row[eg_st[k] as usize] * eg_e[k];
                    }
                    if ssum < 1e-30 {
                        continue; // no rail route (or only the same station at both ends)
                    }
                    let d = ((ux - dd.x).powi(2) + (uy - dd.y).powi(2)).sqrt();
                    let road = p.drive_min(d, lu, dd.loc) * p.w_drive + dd.park;
                    let rail = m_u + dd.m + p.fare_min - fast_ln(ssum) / th;
                    let util = p.beta * (road - rail) + p.asc_rail;
                    let x = tu * dd.dv / (1.0 + fast_exp(-util));
                    rail_sum += x;
                    // kept per destination and spread onto egress stations after the loop:
                    // scattering here chains each pair to the last through y_row
                    let q = zj.q0 as usize + qi;
                    gq[q] = x / ssum;
                    touched.push(q as u32);
                    if walks {
                        out.walk_lost_to_sub[dd.v as usize] += x * pw;
                    }
                }
                if walks {
                    rail_walk += (rail_sum - rail_before) * pw;
                }
            }
            out.pairs += n_pairs as u64;
            out.rail_walk += rail_walk as f64;
            out.walk_lost_by_sub[u] = rail_walk;
            for &q in &touched {
                let dd = &dst[q as usize];
                let g = gq[q as usize];
                let mut x = 0.0f32;
                for k in dd.e0 as usize..dd.e1 as usize {
                    let t = eg_st[k] as usize;
                    let y = g * eg_e[k];
                    y_row[t] += y;
                    let yw = y * a_row[t];
                    out.acc_work[eg_k[k] as usize] += yw;
                    x += yw;
                }
                walksum[(((m_u + dd.m) * (1.0 / WALKSUM_BAND_MIN)) as usize).min(N_WALKSUM - 1)] += x as f64;
            }
            touched.clear();
            out.trips_access += all_sum as f64;
            if rail_sum == 0.0 {
                continue;
            }
            out.rail += rail_sum as f64;
            out.rail_by_sub[u] = rail_sum;
            // split this subzone's rail trips back onto its access stations
            for k in opts {
                let (a, w) = (sz.acc_st[k] as usize, acc_e[k]);
                let row = &mut out.od[a * s..(a + 1) * s];
                let mut tot = 0.0f32;
                for ((o, &y), &e) in row.iter_mut().zip(&y_row).zip(&er[a * s..(a + 1) * s]) {
                    let f = y * w * e;
                    *o += f;
                    tot += f;
                }
                out.acc_home[k] += tot;
                out.rail_band[dist_band(sz.acc_d[k])] += tot as f64;
            }
        }
    }
    // rail trips by work subzone: what its egress stations took
    for v in 0..sz.n {
        out.rail_to_sub[v] = out.acc_work[sz.acc_off[v] as usize..sz.acc_off[v + 1] as usize].iter().sum();
    }
    out.rail_walksum = walksum;
    out
}

// ---------------------------------------------------------------------------------------------
// 6. Assignment

#[derive(Clone)]
pub struct Loads {
    /// Riders per period on the segment starting at each route node.
    pub seg: Vec<f32>,
    /// Boardings and alightings per station, transfers included.
    pub board: Vec<f32>,
    pub alight: Vec<f32>,
    /// The same per route node: riders boarding or leaving that line, direction and stop.
    pub rn_board: Vec<f32>,
    pub rn_alight: Vec<f32>,
    /// Rail trips starting (first boarding) and ending (last alighting) at each station: entries
    /// and exits, transfers not counted.
    pub entry: Vec<f32>,
    pub exit: Vec<f32>,
    pub pax_min: f64,
}

impl Loads {
    pub fn zero(n_st: usize, n_rn: usize) -> Loads {
        Loads { seg: vec![0.0; n_rn], board: vec![0.0; n_st], alight: vec![0.0; n_st], rn_board: vec![0.0; n_rn], rn_alight: vec![0.0; n_rn], entry: vec![0.0; n_st], exit: vec![0.0; n_st], pax_min: 0.0 }
    }
}

/// Each source's tree is walked once in reverse settle order, pushing flow to the predecessor:
/// O(nodes) per source, independent of how many destination pairs carry flow.
pub fn assign(g: &Graph, sk: &Skims, od: &[f32]) -> Loads {
    let (s, n) = (sk.s, sk.n);
    let mut l = Loads::zero(s, n - s);
    let mut flow = vec![0f32; n];
    for src in 0..s {
        let row = &od[src * s..(src + 1) * s];
        if row.iter().all(|&v| v == 0.0) {
            continue;
        }
        for (t, &v) in row.iter().enumerate() {
            l.entry[src] += v;
            l.exit[t] += v;
        }
        flow.iter_mut().for_each(|f| *f = 0.0);
        flow[..s].copy_from_slice(row);
        let pred = &sk.pred[src * n..(src + 1) * n];
        let order = &sk.order[src * n..src * n + sk.n_order[src] as usize];
        for &v in order.iter().skip(1).rev() {
            let v = v as usize;
            let f = flow[v];
            if f == 0.0 {
                continue;
            }
            let pv = pred[v] as usize;
            flow[pv] += f;
            match (pv < s, v < s) {
                (false, false) => {
                    l.seg[pv - s] += f;
                    l.pax_min += (f * g.rn_ride[pv - s]) as f64;
                }
                (true, false) => {
                    l.board[pv] += f;
                    l.rn_board[v - s] += f;
                }
                (false, true) => {
                    // the station of the route node: a walking transfer ends elsewhere
                    l.alight[g.rn_station[pv - s] as usize] += f;
                    l.rn_alight[pv - s] += f;
                }
                (true, true) => {}
            }
        }
    }
    l
}

// ---------------------------------------------------------------------------------------------
// 7. Crowding

pub struct CrowdStats {
    pub segs_over_seats: usize,
    pub segs: usize,
    pub max_load_of_crush: f32,
    /// Rider-weighted percentiles of the penalty factor minus one.
    pub p50: f32,
    pub p90: f32,
}

/// Penalty per segment: 1 + alpha x(1+x)/2, x = max(load - seats, 0) / (crush - seats), per hour
/// at the peak hour. Starts where seats run out, never saturates (memory note on crowding).
pub fn crowd_factors(g: &Graph, net: &Network, loads: &Loads, p: &Params) -> (Vec<f32>, CrowdStats) {
    let nr = g.rn_line.len();
    let mut fac = vec![1.0f32; nr];
    let mut st = CrowdStats { segs_over_seats: 0, segs: 0, max_load_of_crush: 0.0, p50: 0.0, p90: 0.0 };
    let mut wv: Vec<(f32, f32)> = vec![];
    for r in 0..nr {
        if g.rn_ride[r] == 0.0 {
            continue;
        }
        st.segs += 1;
        let l = g.rn_line[r] as usize;
        let tph = 60.0 / net.line_headway_min[l][p.period];
        let seats = tph * net.line_seats[l];
        let crush = tph * net.line_crush[l];
        let hour = loads.seg[r] / p.period_hours * p.peak_hour_factor;
        let x = ((hour - seats) / (crush - seats)).max(0.0);
        fac[r] = 1.0 + p.crowd_alpha * x * (1.0 + x) / 2.0;
        if hour > seats {
            st.segs_over_seats += 1;
        }
        st.max_load_of_crush = st.max_load_of_crush.max(hour / crush);
        if loads.seg[r] > 0.0 {
            wv.push((fac[r] - 1.0, loads.seg[r] * g.rn_ride[r]));
        }
    }
    wv.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let tot: f32 = wv.iter().map(|e| e.1).sum();
    let (mut acc, mut p50, mut p90) = (0.0f32, None, None);
    for &(v, w) in &wv {
        acc += w;
        if p50.is_none() && acc >= 0.5 * tot {
            p50 = Some(v);
        }
        if p90.is_none() && acc >= 0.9 * tot {
            p90 = Some(v);
        }
    }
    st.p50 = p50.unwrap_or(0.0);
    st.p90 = p90.unwrap_or(0.0);
    (fac, st)
}

/// Method of successive averages: `avg += (new - avg) / (round + 1)`.
pub fn msa(avg: &mut [f32], new: &[f32], round: usize) {
    let w = 1.0 / (round as f32 + 1.0);
    for (a, &n) in avg.iter_mut().zip(new) {
        *a += (n - *a) * w;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::demand::{network, synth};

    #[test]
    fn fast_exp_is_close() {
        let mut x = -30.0f32;
        while x < 30.0 {
            let (a, b) = (fast_exp(x), x.exp());
            assert!(((a - b) / b).abs() < 3e-4, "x {x}: {a} vs {b}");
            x += 0.0137;
        }
    }

    /// Small end-to-end run: gravity conserves trips, rail trips are a share of them, every rail
    /// trip shows up as a boarding.
    #[test]
    fn kernel_runs_on_a_small_city() {
        let full = synth::synth_city(1);
        // keep the inner 15 km so the test is quick
        let keep: Vec<usize> = (0..full.len()).filter(|&c| full.x[c].hypot(full.y[c]) < 15_000.0).collect();
        let city = crate::demand::pack::City {
            name: "t".into(),
            h3: keep.iter().map(|&c| full.h3[c]).collect(),
            x: keep.iter().map(|&c| full.x[c]).collect(),
            y: keep.iter().map(|&c| full.y[c]).collect(),
            pop: keep.iter().map(|&c| full.pop[c]).collect(),
            jobs: keep.iter().map(|&c| full.jobs[c]).collect(),
            gravity: None,
        };
        let net = network::synth_network();
        let p = Params::default();
        let z = build_zones(&city, p.zone_m);
        let mut gr = gravity(&z, &p, 100, 1e-4);
        gr.set_walk(&z, &p);
        let (_, tj) = city.totals();
        assert!((gr.total / tj - 1.0).abs() < 1e-3);
        let acc = build_access(&city, &net, &p);
        let sz = build_subzones(&city, &z, &acc, &net, &p);
        let g = build_graph(&net, &p);
        let sk = skims(&g, &p);
        let er = station_exp(&sk, &p);
        let mo = mode_choice(&z, &gr, &sz, &er, g.n_st, &p);
        assert!(mo.rail > 0.0 && mo.rail < mo.trips_access && mo.trips_access <= mo.trips_total * 1.0001);
        let l = assign(&g, &sk, &mo.od);
        let od_sum: f64 = mo.od.iter().map(|&v| v as f64).sum();
        // the split back onto access stations conserves rail trips
        assert!((od_sum / mo.rail - 1.0).abs() < 1e-3, "od {od_sum} vs rail {}", mo.rail);
        let bands: f64 = mo.rail_band.iter().sum();
        assert!((bands / mo.rail - 1.0).abs() < 1e-3);
        // rail trips add up the same by home subzone, work subzone and access entry at each end
        let sum = |v: &[f32]| v.iter().map(|&x| x as f64).sum::<f64>();
        for (what, v) in [("by", &mo.rail_by_sub), ("to", &mo.rail_to_sub), ("home", &mo.acc_home), ("work", &mo.acc_work)] {
            assert!((sum(v) / mo.rail - 1.0).abs() < 1e-3, "{what}: {} vs {}", sum(v), mo.rail);
        }
        let (wb, wt) = (sum(&mo.walk_lost_by_sub), sum(&mo.walk_lost_to_sub));
        assert!(wb > 0.0 && (wb / mo.rail_walk - 1.0).abs() < 1e-3 && (wt / mo.rail_walk - 1.0).abs() < 1e-3);
        let boards: f64 = l.board.iter().map(|&v| v as f64).sum();
        assert!(boards >= od_sum * 0.999);
    }

    /// The ring search finds the same K nearest stations within the bound as a full scan.
    #[test]
    fn access_matches_a_full_scan() {
        let city = synth::synth_city(3);
        let net = network::synth_network();
        for cut in [800.0f32, 2500.0, 5000.0] {
            let p = Params { walk_cutoff_m: cut, ..Params::default() };
            let a = build_access(&city, &net, &p);
            for c in (0..city.len()).step_by(37) {
                let mut d: Vec<(f32, u32)> = (0..net.n_stations())
                    .map(|s| ((net.st_x[s] - city.x[c]).hypot(net.st_y[s] - city.y[c]), s as u32))
                    .filter(|e| e.0 <= cut)
                    .collect();
                d.sort_by(|x, y| x.partial_cmp(y).unwrap());
                d.truncate(K);
                assert_eq!(a.n[c] as usize, d.len(), "cell {c} cut {cut}");
                for (k, e) in d.iter().enumerate() {
                    assert!((a.walk[c * K + k] - p.walk_min(e.0)).abs() < 1e-3, "cell {c} k {k}");
                }
            }
        }
    }

    /// The walk's perceived cost: linear for the first `walk_easy_min`, steeper after; the drive
    /// is slower in dense places and on short trips.
    #[test]
    fn access_and_drive_costs() {
        let p = Params::default();
        let easy_m = p.walk_easy_min * 60.0 * p.walk_mps / p.walk_detour;
        assert!((p.access_cost(easy_m) - p.walk_easy_min * p.w_walk).abs() < 1e-3);
        let km = |d: f32| p.access_cost(d + 1000.0) - p.access_cost(d);
        assert!(km(easy_m + 100.0) > 1.9 * km(0.0));
        let (dense, open) = (p.place_min_per_km(50_000.0), p.place_min_per_km(100.0));
        assert!(dense.0 > open.0 && dense.1 > open.1);
        // 30 km between open places is much quicker a km than 3 km
        assert!(p.drive_min(30_000.0, open, open) / 30.0 < 0.6 * p.drive_min(3_000.0, open, open) / 3.0);
        assert!(p.drive_min(10_000.0, dense, dense) > 2.0 * p.drive_min(10_000.0, open, open));
    }

    #[test]
    fn fast_ln_is_close() {
        let mut x = 1e-30f32;
        while x < 1e30 {
            let (a, b) = (fast_ln(x), x.ln());
            assert!((a - b).abs() < 4e-6 * b.abs().max(1.0), "x {x}: {a} vs {b}");
            x *= 1.0137;
        }
    }
}
