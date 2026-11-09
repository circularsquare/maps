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
pub const BUS_SHARE: f64 = 0.9;
/// Seconds each stop costs a bus: 20 standing, 8 speeding up and slowing down.
pub const BUS_STOP_S: f64 = 28.0;
/// The least a bus's moving speed falls to, km/h: its own lane, so the car's traffic does not hold it.
pub const BUS_LANE_KMH: f64 = 30.0;
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
    (BUS_SHARE * car_kmh).max(BUS_LANE_KMH)
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
}

#[cfg(test)]
mod tests;
