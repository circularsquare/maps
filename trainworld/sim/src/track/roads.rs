//! Roads that track at ground level crosses (T-1929, notes/T-1929.md). The clock worker sends the
//! basemap's zoom 14 road lines tile by tile, each line with its weight (a motorway 3, others 1;
//! bridges and tunnels left out); each edge keeps the chainages where its alignment crosses one,
//! worked out again when a tile lands as terrain is, and its price adds `ROAD_CROSSING_COST` for
//! each that is at level 0 there.
//!
//! Positions are in zoom 14 tile units of the Web Mercator world (4096 a tile), where a road's
//! straight pieces are straight: no frame, so the tiles stay good across loads.

use super::geom;
use super::params::{LEVEL_H, ROAD_CROSSING_COST, ROAD_STEP};
use std::collections::HashMap;
use std::sync::Arc;

/// Tile units across a tile.
pub const EXTENT: f64 = 4096.0;
/// Tiles across the world at zoom 14.
pub const TILES: u32 = 1 << 14;
/// Cells across a tile in its index.
const CELLS: usize = 32;
const CELL: f64 = EXTENT / CELLS as f64;

/// One tile's road pieces, indexed by cell.
pub struct RoadTile {
    /// each piece's ends, tile units
    seg: Vec<[f32; 4]>,
    weight: Vec<u8>,
    /// by cell (row major), the pieces reaching into it: `items[off[c]..off[c + 1]]`
    off: Vec<u32>,
    items: Vec<u32>,
}

impl RoadTile {
    /// `pts` x, y pairs in units of `extent` across the tile; line `i` is points `starts[i]` to
    /// `starts[i + 1]` (the last runs to the end) with weight `weights[i]`. Pieces wholly outside
    /// the tile's square are dropped: the tile next door has them.
    pub fn new(extent: f64, pts: &[i32], starts: &[u32], weights: &[u8]) -> RoadTile {
        let k = EXTENT / extent.max(1.0);
        let n = pts.len() / 2;
        let mut seg = vec![];
        let mut weight = vec![];
        for (i, &s) in starts.iter().enumerate() {
            let end = starts.get(i + 1).map_or(n, |&e| (e as usize).min(n));
            let w = weights.get(i).copied().unwrap_or(1);
            if w == 0 {
                continue;
            }
            for j in (s as usize)..end.saturating_sub(1) {
                let p = [pts[2 * j] as f64 * k, pts[2 * j + 1] as f64 * k, pts[2 * j + 2] as f64 * k, pts[2 * j + 3] as f64 * k];
                if p[0].max(p[2]) < 0.0 || p[0].min(p[2]) >= EXTENT || p[1].max(p[3]) < 0.0 || p[1].min(p[3]) >= EXTENT {
                    continue;
                }
                seg.push(p.map(|v| v as f32));
                weight.push(w);
            }
        }
        let range = |a: f32, b: f32| {
            let lo = ((a.min(b) as f64 / CELL).floor().max(0.0) as usize).min(CELLS - 1);
            let hi = ((a.max(b) as f64 / CELL).floor().max(0.0) as usize).min(CELLS - 1);
            lo..=hi
        };
        let mut count = vec![0u32; CELLS * CELLS + 1];
        for p in &seg {
            for cy in range(p[1], p[3]) {
                for cx in range(p[0], p[2]) {
                    count[cy * CELLS + cx + 1] += 1;
                }
            }
        }
        for c in 1..count.len() {
            count[c] += count[c - 1];
        }
        let mut fill = count.clone();
        let mut items = vec![0u32; *count.last().unwrap() as usize];
        for (i, p) in seg.iter().enumerate() {
            for cy in range(p[1], p[3]) {
                for cx in range(p[0], p[2]) {
                    let c = cy * CELLS + cx;
                    items[fill[c] as usize] = i as u32;
                    fill[c] += 1;
                }
            }
        }
        RoadTile { seg, weight, off: count, items }
    }

    pub fn pieces(&self) -> usize {
        self.seg.len()
    }

    /// Where the piece `a`-`b` (tile units) crosses a road inside this tile's square: `f(t, weight)`
    /// with `t` in [0, 1) along it. Each crossing is found once: in the cell holding the point.
    fn cross(&self, a: (f64, f64), b: (f64, f64), mut f: impl FnMut(f64, u8)) {
        if a.0.max(b.0) < 0.0 || a.0.min(b.0) >= EXTENT || a.1.max(b.1) < 0.0 || a.1.min(b.1) >= EXTENT {
            return;
        }
        let cell = |v: f64| ((v / CELL).floor().max(0.0) as usize).min(CELLS - 1);
        let (rx, ry) = (b.0 - a.0, b.1 - a.1);
        for cy in cell(a.1.min(b.1))..=cell(a.1.max(b.1)) {
            for cx in cell(a.0.min(b.0))..=cell(a.0.max(b.0)) {
                let c = cy * CELLS + cx;
                for &i in &self.items[self.off[c] as usize..self.off[c + 1] as usize] {
                    let s = self.seg[i as usize];
                    let (cx0, cy0) = (s[0] as f64, s[1] as f64);
                    let (sx, sy) = (s[2] as f64 - cx0, s[3] as f64 - cy0);
                    let den = rx * sy - ry * sx;
                    if den == 0.0 {
                        continue;
                    }
                    let t = ((cx0 - a.0) * sy - (cy0 - a.1) * sx) / den;
                    let u = ((cx0 - a.0) * ry - (cy0 - a.1) * rx) / den;
                    if !(0.0..1.0).contains(&t) || !(0.0..1.0).contains(&u) {
                        continue;
                    }
                    let (px, py) = (a.0 + t * rx, a.1 + t * ry);
                    if !(0.0..EXTENT).contains(&px) || !(0.0..EXTENT).contains(&py) || cell(px) != cx || cell(py) != cy {
                        continue;
                    }
                    f(t, self.weight[i as usize]);
                }
            }
        }
    }
}

/// The road tiles installed, by `key`. Cheap to clone: the tiles are shared.
#[derive(Clone, Default)]
pub struct Roads {
    pub tiles: HashMap<u32, Arc<RoadTile>>,
}

impl Roads {
    pub fn key(tx: u32, ty: u32) -> u32 {
        tx * TILES + ty
    }
    pub fn is_empty(&self) -> bool {
        self.tiles.is_empty()
    }
}

/// Longitude and latitude in zoom 14 tile units.
pub fn merc(lon: f64, lat: f64) -> (f64, f64) {
    let w = TILES as f64 * EXTENT;
    let s = lat.to_radians().sin().clamp(-0.9999, 0.9999);
    ((lon + 180.0) / 360.0 * w, (0.5 - ((1.0 + s) / (1.0 - s)).ln() / (4.0 * std::f64::consts::PI)) * w)
}

/// A tile's corners' longitude and latitude: west, south, east, north.
pub fn tile_bounds(tx: u32, ty: u32) -> [f64; 4] {
    let n = TILES as f64;
    let lat = |y: f64| (std::f64::consts::PI * (1.0 - 2.0 * y / n)).sinh().atan().to_degrees();
    [tx as f64 / n * 360.0 - 180.0, lat(ty as f64 + 1.0), (tx as f64 + 1.0) / n * 360.0 - 180.0, lat(ty as f64)]
}

/// The chainages `[s, weight]` where an alignment of length `len` crosses an installed road, in
/// order; `at` gives longitude and latitude at a chainage. Sampled every `ROAD_STEP` m.
pub fn crossings(len: f64, at: impl Fn(f64) -> (f64, f64), roads: &Roads, out: &mut Vec<[f64; 2]>) {
    if roads.is_empty() || len <= 0.0 {
        return;
    }
    let n = (len / ROAD_STEP).ceil().max(1.0) as usize;
    let step = len / n as f64;
    let first = at(0.0);
    let mut a = merc(first.0, first.1);
    let start = out.len();
    for j in 0..n {
        let s0 = j as f64 * step;
        let p = at((j + 1) as f64 * step);
        let b = merc(p.0, p.1);
        let tile = |v: f64| (v / EXTENT).floor().max(0.0) as u32;
        for ty in tile(a.1.min(b.1))..=tile(a.1.max(b.1)) {
            for tx in tile(a.0.min(b.0))..=tile(a.0.max(b.0)) {
                let Some(t) = roads.tiles.get(&Roads::key(tx, ty)) else { continue };
                let (ox, oy) = (tx as f64 * EXTENT, ty as f64 * EXTENT);
                t.cross((a.0 - ox, a.1 - oy), (b.0 - ox, b.1 - oy), |t, w| out.push([s0 + t * step, w as f64]));
            }
        }
        a = b;
    }
    out[start..].sort_by(|x, y| x[0].total_cmp(&y[0]));
}

/// The crossings `crossings` finds in tile (`tx`, `ty`) alone, of the chords between chainages `lo`
/// and `hi` (T-2230: a tile landing adds only its own, as each crossing is found in one tile). The
/// chords are those a whole pass samples, so the chainages are the same to the bit.
#[allow(clippy::too_many_arguments)]
pub fn crossings_in_tile(len: f64, lo: f64, hi: f64, at: impl Fn(f64) -> (f64, f64), tile: &RoadTile, tx: u32, ty: u32, out: &mut Vec<[f64; 2]>) {
    if len <= 0.0 || lo > hi {
        return;
    }
    let n = (len / ROAD_STEP).ceil().max(1.0) as usize;
    let step = len / n as f64;
    let j0 = ((lo / step).floor().max(0.0) as usize).min(n - 1);
    let j1 = ((hi / step).ceil().max(0.0) as usize).clamp(j0 + 1, n);
    let p = at(j0 as f64 * step);
    let mut a = merc(p.0, p.1);
    let (ox, oy) = (tx as f64 * EXTENT, ty as f64 * EXTENT);
    let tile_of = |v: f64| (v / EXTENT).floor().max(0.0) as u32;
    let start = out.len();
    for j in j0..j1 {
        let s0 = j as f64 * step;
        let p = at((j + 1) as f64 * step);
        let b = merc(p.0, p.1);
        if (tile_of(a.0.min(b.0))..=tile_of(a.0.max(b.0))).contains(&tx) && (tile_of(a.1.min(b.1))..=tile_of(a.1.max(b.1))).contains(&ty) {
            tile.cross((a.0 - ox, a.1 - oy), (b.0 - ox, b.1 - oy), |t, w| out.push([s0 + t * step, w as f64]));
        }
        a = b;
    }
    out[start..].sort_by(|x, y| x[0].total_cmp(&y[0]));
}

/// Whether no crossing of track with this vertical profile can ever be charged (`charge`): every
/// breakpoint half a level or more off the ground, all on one side of it (heights are linear between).
/// Such track keeps no crossings (T-2230: most of a big city's track is underground or on viaducts).
pub fn off_ground(vert: &[[f64; 2]]) -> bool {
    !vert.is_empty() && (vert.iter().all(|v| v[1] >= LEVEL_H / 2.0) || vert.iter().all(|v| v[1] <= -LEVEL_H / 2.0))
}

/// The crossings charged, weighted, and their price, US$M: those where the track is at level 0
/// (under half a level from the ground, so the low half of a ramp too).
pub fn charge(vert: &[[f64; 2]], crossings: &[[f64; 2]]) -> (f64, f64) {
    let n: f64 = crossings.iter().filter(|c| geom::height_at(vert, c[0]).abs() < LEVEL_H / 2.0).map(|c| c[1]).sum();
    (n, n * ROAD_CROSSING_COST)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tile with a grid of roads every 100 m north-south across its middle.
    fn tile_at(lon: f64, lat: f64, every_m: f64) -> (u32, u32, RoadTile) {
        let (x, y) = merc(lon, lat);
        let (tx, ty) = ((x / EXTENT) as u32, (y / EXTENT) as u32);
        // tile units a metre at this latitude
        let upm = TILES as f64 * EXTENT / (40_075_016.7 * lat.to_radians().cos());
        let step = (every_m * upm) as i32;
        let (mut pts, mut starts, mut weights) = (vec![], vec![], vec![]);
        let mut gx = step / 2;
        while gx < EXTENT as i32 {
            starts.push((pts.len() / 2) as u32);
            pts.extend([gx, -64, gx, EXTENT as i32 + 64]);
            weights.push(1);
            gx += step;
        }
        (tx, ty, RoadTile::new(EXTENT, &pts, &starts, &weights))
    }

    #[test]
    fn counts_each_road_once_across_cells_and_steps() {
        let (lon, lat) = (-73.98, 40.75);
        let (tx, ty, t) = tile_at(lon, lat, 100.0);
        let mut roads = Roads::default();
        roads.tiles.insert(Roads::key(tx, ty), Arc::new(t));
        let [w, s, e, n] = tile_bounds(tx, ty);
        let mid = 0.5 * (s + n);
        // a line east-west across the whole tile, a little inside its edges
        let (lon0, lon1) = (w + 0.001, e - 0.001);
        let len = (lon1 - lon0).to_radians() * 6_371_000.0 * mid.to_radians().cos();
        let mut out = vec![];
        crossings(len, |c| (lon0 + (lon1 - lon0) * c / len, mid), &roads, &mut out);
        let expect = (len / 100.0).round();
        assert!((out.len() as f64 - expect).abs() <= 1.0, "{} crossings, about {expect} expected", out.len());
        assert!(out.windows(2).all(|p| p[1][0] > p[0][0] + 50.0), "each once, in order");
        // parallel to the roads: none
        let mut along = vec![];
        let x = w + 0.3 * (e - w) + 1e-7;
        crossings(1000.0, |c| (x, s + 0.001 + c / 111_000.0), &roads, &mut along);
        assert!(along.len() <= 1);
    }

    #[test]
    fn only_ground_level_is_charged() {
        let c = [[100.0, 1.0], [500.0, 3.0], [900.0, 1.0]];
        // level 0 to 300 m, a ramp to +1 at 700 m, +1 on
        let vert = [[0.0, 0.0], [300.0, 0.0], [700.0, LEVEL_H], [1000.0, LEVEL_H]];
        let (n, cost) = charge(&vert, &c);
        // at 500 m the ramp is at half a level: not under it
        assert_eq!(n, 1.0);
        assert_eq!(cost, ROAD_CROSSING_COST);
        let flat = [[0.0, 0.0], [1000.0, 0.0]];
        assert_eq!(charge(&flat, &c).0, 5.0);
    }
}
