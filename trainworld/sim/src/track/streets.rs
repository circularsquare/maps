//! Street speed by place (T-1577) and the rule for a bus's run time (T-2904). The car's figures are
//! demand's (`Params::place_min_per_km`, `drive_min_f`: local streets falling from 44 km/h in open
//! country towards 16.5 where homes and jobs are dense, main roads from 88 the same way, half way at
//! 5,000 a km²), repeated here because the track model is built without the demand module; a test
//! under the `demand` feature keeps the two equal.
//!
//! A pack gives a raster of its 2 km zones' speeds (`StreetGrid`), fetched by the clock worker with
//! the pack's water mask and kept in the network beside the terrain. A route outside every raster
//! runs at open-country car speeds.

use std::sync::Arc;

use super::frame::Frame;

/// The car's speed on local streets in open country, km/h (`Params::road_town_kmh`).
pub const CAR_TOWN_KMH: f64 = 44.0;
/// And where homes and jobs are dense (`Params::road_city_kmh`), on local streets and main roads alike.
pub const CAR_CITY_KMH: f64 = 16.5;
/// Homes and jobs a km² of land at which it is half way between them (`Params::road_dens50`).
pub const CAR_DENS50: f64 = 5000.0;
/// The car's speed on main roads in open country, km/h (`Params::road_far_kmh`).
pub const CAR_FAR_KMH: f64 = 88.0;
/// And on open highway, past each end's main roads (`Params::road_open_kmh`).
pub const CAR_OPEN_KMH: f64 = 110.0;
/// Road metres of local streets at each end of a car's trip, at most half of it (`Params::road_local_km`).
pub const CAR_LOCAL_M: f64 = 2000.0;
/// Road metres of main roads after them at each end, at most half the rest (`Params::road_end_km`).
pub const CAR_END_M: f64 = 8000.0;
/// Mean area of a resolution 9 H3 cell, km² (`demand::kernel::CELL_KM2`).
pub const CELL_KM2: f64 = 0.1053;

/// A bus's moving speed as a share of the car's.
pub const BUS_SHARE: f64 = 0.85;
/// Seconds each stop costs a bus: 20 standing, 8 speeding up and slowing down.
pub const BUS_STOP_S: f64 = 28.0;
/// The least a bus's moving speed falls to, km/h: its own lane, so the car's traffic does not hold it.
pub const BUS_LANE_KMH: f64 = 25.0;

/// The least a bus line's moving speed falls to, km/h: a BRT line's (T-3907) is the plain floor plus
/// `params::BRT_LANE_BONUS_KMH`.
pub fn lane_kmh(brt: bool) -> f64 {
    if brt { BUS_LANE_KMH + super::params::BRT_LANE_BONUS_KMH } else { BUS_LANE_KMH }
}
/// T-3907: whether a place where the car's local streets run at `street_kmh` is in town for BRT: a plain bus there
/// moves under BRT's floor, so the lanes make it faster. BRT's price and upkeep count only the road there (Anita,
/// 11-21: "can we make brt free / same as normal bus cost for portions that are out of town").
pub fn brt_in_town(street_kmh: f64) -> bool {
    BUS_SHARE * street_kmh < lane_kmh(true)
}
/// The car on a trunk road, whatever the density and however short the bus's hop, km/h.
pub const CAR_TRUNK_KMH: f64 = CAR_FAR_KMH;

/// The car's speed on local streets at a place with `dens` homes and jobs a km² of land, km/h.
pub fn car_street_kmh(dens: f64) -> f64 {
    let f = 1.0 / (1.0 + dens / CAR_DENS50);
    CAR_CITY_KMH + (CAR_TOWN_KMH - CAR_CITY_KMH) * f
}

/// The car's speed on main roads at a place where its local streets run at `street_kmh`.
pub fn car_main_kmh(street_kmh: f64) -> f64 {
    let f = (street_kmh - CAR_CITY_KMH) / (CAR_TOWN_KMH - CAR_CITY_KMH);
    CAR_CITY_KMH + (CAR_FAR_KMH - CAR_CITY_KMH) * f
}

/// A bus's moving speed where the car runs at `car_kmh`.
pub fn bus_kmh(car_kmh: f64) -> f64 {
    bus_kmh_at(car_kmh, BUS_LANE_KMH)
}

/// A bus's moving speed where the car runs at `car_kmh`, on a line whose floor is `lane` (`lane_kmh`).
pub fn bus_kmh_at(car_kmh: f64, lane: f64) -> f64 {
    (BUS_SHARE * car_kmh).max(lane)
}

/// The most a trunk road or motorway lifts a bus's moving speed to, over the same line's on the local streets at
/// the same place (Anita, 11-21: trunk roads through city centres "75km/hing them is definitely concerning"; "lower
/// it to like 150%"; then "ok sure we can do 25 35 35 49": plain and BRT on the street, then on a trunk road).
pub const BUS_TRUNK_MAX_X: f64 = 1.4;

/// A bus's moving speed on a trunk road on a line whose floor is `lane`: `bus_fast_kmh` at `BUS_TRUNK_MAX_X`.
pub fn bus_trunk_kmh(road_kmh: f64, street_kmh: f64, lane: f64) -> f64 {
    bus_fast_kmh(road_kmh, street_kmh, lane, BUS_TRUNK_MAX_X)
}

/// A bus's moving speed on a trunk road where the road's car runs at `road_kmh` and the car's local streets at
/// `street_kmh`, on a line whose floor is `lane`: its share of the road's car, taken as at least `CAR_TRUNK_KMH`,
/// but no more than `max_x` times the line's own street speed there (a dense centre at the 25 floor: 35 plain, 49 BRT;
/// open country 52), and never under the same road's speed without the trunk (open country's main roads keep 75).
pub fn bus_fast_kmh(road_kmh: f64, street_kmh: f64, lane: f64, max_x: f64) -> f64 {
    (BUS_SHARE * road_kmh.max(CAR_TRUNK_KMH)).min(max_x * bus_kmh_at(street_kmh, lane)).max(bus_kmh_at(road_kmh, lane))
}

/// Quarter km/h in a byte (16.5 is 66, 44 is 176); 0 is no zone.
const Q: f64 = 4.0;

/// One pack's zones, as the car's street speed on a grid of its zone squares.
pub struct StreetGrid {
    pack: Frame,
    zone_m: f64,
    x0: f64,
    y0: f64,
    w: i32,
    h: i32,
    /// column after column from the south-west square: `gx * h + gy`
    q: Vec<u8>,
}

impl StreetGrid {
    /// From the pack's arrays: per cell its `pop`, `jobs` and `zone`; per zone its square
    /// (`gx`, `gy`), its side in squares (`size` 1 or 2, empty for all 1) and its land in resolution 9
    /// cells (`cells9`). The grid is `w` x `h` squares of `zone_m` from the corner (`x0`, `y0`), in the
    /// pack's frame about (`lon`, `lat`). None if the arrays do not fit each other.
    #[allow(clippy::too_many_arguments)]
    pub fn from_zones(zone_m: f64, x0: f64, y0: f64, w: u32, h: u32, pop: &[f32], jobs: &[f32], zone: &[u32], gx: &[u16], gy: &[u16], size: &[u8], cells9: &[u32], lon: f64, lat: f64) -> Option<StreetGrid> {
        let n = gx.len();
        if n == 0 || gy.len() != n || cells9.len() != n || pop.len() != zone.len() || jobs.len() != zone.len() || !(size.is_empty() || size.len() == n) {
            return None;
        }
        if !(zone_m > 0.0) || w == 0 || h == 0 || w as u64 * h as u64 > 4_000_000 || zone.iter().any(|&z| z as usize >= n) {
            return None;
        }
        let mut people = vec![0f64; n];
        for (c, &z) in zone.iter().enumerate() {
            people[z as usize] += pop[c] as f64 + jobs[c] as f64;
        }
        let (w, h) = (w as i32, h as i32);
        let mut q = vec![0u8; (w * h) as usize];
        for i in 0..n {
            let side = if size.get(i) == Some(&2) { 2 } else { 1 };
            let v = (car_street_kmh(people[i] / (cells9[i].max(1) as f64 * CELL_KM2)) * Q).round().clamp(1.0, 255.0) as u8;
            for dx in 0..side {
                for dy in 0..side {
                    let (sx, sy) = (gx[i] as i32 + dx, gy[i] as i32 + dy);
                    if sx < w && sy < h {
                        let at = &mut q[(sx * h + sy) as usize];
                        // two zones in a square (a region border): the denser, so the slower
                        *at = if *at == 0 { v } else { (*at).min(v) };
                    }
                }
            }
        }
        Some(StreetGrid { pack: Frame::equirect(lon, lat), zone_m, x0, y0, w, h, q })
    }

    /// T-4155: the longitudes and latitudes its squares cover, [west, south, east, north] (its frame is
    /// equirectangular, so the box is exact).
    pub fn lonlat_box(&self) -> [f64; 4] {
        let (w, s) = self.pack.to_lonlat(self.x0, self.y0);
        let (e, n) = self.pack.to_lonlat(self.x0 + self.w as f64 * self.zone_m, self.y0 + self.h as f64 * self.zone_m);
        [w, s, e, n]
    }

    /// The car's street speed at a place, km/h, or None outside the zones.
    pub fn at(&self, lon: f64, lat: f64) -> Option<f64> {
        let (x, y) = self.pack.from_lonlat(lon, lat);
        let (fx, fy) = ((x - self.x0) / self.zone_m, (y - self.y0) / self.zone_m);
        if fx < 0.0 || fy < 0.0 {
            return None;
        }
        let (sx, sy) = (fx as i32, fy as i32);
        if sx >= self.w || sy >= self.h {
            return None;
        }
        match self.q[(sx * self.h + sy) as usize] {
            0 => None,
            v => Some(v as f64 / Q),
        }
    }

    /// T-3913: the squares in town for BRT (`brt_in_town`), merged into polygons in longitude and latitude, for the
    /// bus tool's road hint: the polygon count, then per polygon its ring count, then per ring (its outline first, then
    /// its holes) its point count and the points' longitude, latitude, the first repeated last. A square whose centre
    /// a grid in `before` covers is left to that grid, as `Streets::car_kmh_at` does.
    pub fn town_outline(&self, before: &[Arc<StreetGrid>]) -> Vec<f64> {
        let (w, h) = (self.w, self.h);
        let mut town: Vec<bool> = self.q.iter().map(|&v| v != 0 && brt_in_town(v as f64 / Q)).collect();
        if !before.is_empty() {
            for sx in 0..w {
                for sy in 0..h {
                    let i = (sx * h + sy) as usize;
                    if town[i] {
                        let (lon, lat) = self.pack.to_lonlat(self.x0 + (sx as f64 + 0.5) * self.zone_m, self.y0 + (sy as f64 + 0.5) * self.zone_m);
                        town[i] = !before.iter().any(|g| g.at(lon, lat).is_some());
                    }
                }
            }
        }
        let inside = |sx: i32, sy: i32| sx >= 0 && sy >= 0 && sx < w && sy < h && town[(sx * h + sy) as usize];
        // the squares joined side to side, one label each
        let mut part = vec![u32::MAX; town.len()];
        let mut parts = 0u32;
        let mut stack = Vec::new();
        for i in 0..town.len() {
            if !town[i] || part[i] != u32::MAX {
                continue;
            }
            part[i] = parts;
            stack.push(i);
            while let Some(j) = stack.pop() {
                let (sx, sy) = ((j as i32) / h, (j as i32) % h);
                for (nx, ny) in [(sx + 1, sy), (sx - 1, sy), (sx, sy + 1), (sx, sy - 1)] {
                    let k = (nx * h + ny) as usize;
                    if inside(nx, ny) && part[k] == u32::MAX {
                        part[k] = parts;
                        stack.push(k);
                    }
                }
            }
            parts += 1;
        }
        // the edges between a town square and any other, the town on their left: start corner, direction (east,
        // north, west, south), the square's label
        const STEP: [(i32, i32); 4] = [(1, 0), (0, 1), (-1, 0), (0, -1)];
        let mut edges: Vec<(i32, i32, usize, u32)> = Vec::new();
        for sx in 0..w {
            for sy in 0..h {
                if !inside(sx, sy) {
                    continue;
                }
                let p = part[(sx * h + sy) as usize];
                if !inside(sx, sy - 1) { edges.push((sx, sy, 0, p)); }
                if !inside(sx + 1, sy) { edges.push((sx + 1, sy, 1, p)); }
                if !inside(sx, sy + 1) { edges.push((sx + 1, sy + 1, 2, p)); }
                if !inside(sx - 1, sy) { edges.push((sx, sy + 1, 3, p)); }
            }
        }
        let key = |x: i32, y: i32| x as i64 * (h as i64 + 1) + y as i64;
        let mut from: std::collections::HashMap<i64, [usize; 2]> = std::collections::HashMap::new();
        for (e, &(x, y, _, _)) in edges.iter().enumerate() {
            let slot = from.entry(key(x, y)).or_insert([usize::MAX; 2]);
            if slot[0] == usize::MAX { slot[0] = e } else { slot[1] = e }
        }
        // each edge's next: where two town squares meet only at a corner, the left turn, so a ring keeps to its
        // own squares
        let next = |e: usize| -> usize {
            let (x, y, d, _) = edges[e];
            let at = from[&key(x + STEP[d].0, y + STEP[d].1)];
            for turn in [1, 0, 3] {
                for &o in &at {
                    if o != usize::MAX && edges[o].2 == (d + turn) % 4 {
                        return o;
                    }
                }
            }
            unreachable!("a town edge with no edge after it")
        };
        let mut used = vec![false; edges.len()];
        // per label: its outline and its holes, in grid corners
        let mut rings: Vec<Vec<Vec<(i32, i32)>>> = vec![Vec::new(); parts as usize];
        for e0 in 0..edges.len() {
            if used[e0] {
                continue;
            }
            let mut ring = Vec::new();
            let mut e = e0;
            loop {
                used[e] = true;
                let n = next(e);
                if edges[n].2 != edges[e].2 {
                    ring.push((edges[n].0, edges[n].1));
                }
                e = n;
                if e == e0 {
                    break;
                }
            }
            ring.push(ring[0]);
            let twice_area: i64 = ring.windows(2).map(|s| s[0].0 as i64 * s[1].1 as i64 - s[1].0 as i64 * s[0].1 as i64).sum();
            let r = &mut rings[edges[e0].3 as usize];
            if twice_area > 0 { r.insert(0, ring) } else { r.push(ring) }
        }
        let mut out = vec![parts as f64];
        for p in &rings {
            out.push(p.len() as f64);
            for ring in p {
                out.push(ring.len() as f64);
                for &(x, y) in ring {
                    let (lon, lat) = self.pack.to_lonlat(self.x0 + x as f64 * self.zone_m, self.y0 + y as f64 * self.zone_m);
                    out.extend([lon, lat]);
                }
            }
        }
        out
    }
}

/// The street grids installed, one per pack near the track or the view. Cheap to clone: the grids
/// are shared.
#[derive(Clone, Default)]
pub struct Streets {
    pub grids: Vec<Arc<StreetGrid>>,
}

impl Streets {
    pub fn is_empty(&self) -> bool {
        self.grids.is_empty()
    }
    /// The car's street speed at a place, km/h, from the first grid with a zone there.
    pub fn car_kmh_at(&self, lon: f64, lat: f64) -> Option<f64> {
        self.grids.iter().find_map(|g| g.at(lon, lat))
    }
    /// T-3913: the last grid's in-town squares as polygons (`StreetGrid::town_outline`); empty with no grid.
    pub fn last_town_outline(&self) -> Vec<f64> {
        match self.grids.split_last() {
            Some((g, before)) => g.town_outline(before),
            None => Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests;
