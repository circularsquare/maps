//! Build costs (SPEC 6.4) and the water mask interface.

use super::frame::Frame;
use super::params::*;
use std::sync::Arc;

/// Where the water is. T-030 ships a mask in the city pack; anything that can answer "is this
/// point water?" in local metres works here.
pub trait WaterMask {
    fn is_water(&self, x: f64, y: f64) -> bool;
    /// True when there is no water anywhere (lets the model skip sampling).
    fn is_empty(&self) -> bool {
        false
    }
}

/// No water at all.
pub struct NoWater;
impl WaterMask for NoWater {
    fn is_water(&self, _: f64, _: f64) -> bool {
        false
    }
    fn is_empty(&self) -> bool {
        true
    }
}

/// A pack raster in a shared world's legacy equirectangular frame. Pack -> world is
/// `offset + scale * point`.
pub struct TransformedMask {
    pub mask: Arc<RunMask>,
    pub offset: [f64; 2],
    pub scale: [f64; 2],
}
impl WaterMask for TransformedMask {
    fn is_water(&self, x: f64, y: f64) -> bool {
        self.mask.is_water((x - self.offset[0]) / self.scale[0], (y - self.offset[1]) / self.scale[1])
    }
}

/// A pack raster seen from any world frame (T-125). The pack grid is laid out in the pack's own
/// equirectangular frame, so a world point goes to longitude and latitude and from there into the
/// pack frame. That is a few trigonometric calls, too slow to do for every sample of every edit, so
/// the map is tabulated on a `WARP_STEP` lattice over the raster's extent and interpolated
/// bilinearly: the map is smooth (its curvature is 1/R), which makes the interpolation off by
/// under 0.1 m, against 25 m raster cells. A point outside the lattice is land.
pub struct GeoMask {
    pub mask: Arc<RunMask>,
    /// `[xmin, ymin, xmax, ymax]` of the lattice in the world frame.
    pub bounds: [f64; 4],
    nx: usize,
    /// Pack coordinates at each lattice node, row after row from the south-west corner.
    warp: Vec<[f64; 2]>,
}

/// Lattice spacing of `GeoMask`, metres.
pub const WARP_STEP: f64 = 2000.0;

impl GeoMask {
    pub fn new(mask: Arc<RunMask>, world: Frame, pack_lon: f64, pack_lat: f64) -> GeoMask {
        let pack = Frame::equirect(pack_lon, pack_lat);
        let (x0, y0) = (mask.x0, mask.y0);
        let (x1, y1) = (x0 + mask.w as f64 * mask.cell, y0 + mask.h as f64 * mask.cell);
        let mut b = [f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY];
        const STEPS: usize = 32;
        for i in 0..=STEPS {
            let t = i as f64 / STEPS as f64;
            let (u, v) = (x0 + (x1 - x0) * t, y0 + (y1 - y0) * t);
            for (px, py) in [(u, y0), (u, y1), (x0, v), (x1, v)] {
                let (lon, lat) = pack.to_lonlat(px, py);
                let (wx, wy) = world.from_lonlat(lon, lat);
                b = [b[0].min(wx), b[1].min(wy), b[2].max(wx), b[3].max(wy)];
            }
        }
        let m = 2.0 * mask.cell + 0.002 * (b[2] - b[0]).max(b[3] - b[1]);
        let bounds = [b[0] - m, b[1] - m, b[2] + m, b[3] + m];
        let nx = ((bounds[2] - bounds[0]) / WARP_STEP).ceil() as usize + 2;
        let ny = ((bounds[3] - bounds[1]) / WARP_STEP).ceil() as usize + 2;
        let mut warp = Vec::with_capacity(nx * ny);
        for j in 0..ny {
            for i in 0..nx {
                let (lon, lat) = world.to_lonlat(bounds[0] + i as f64 * WARP_STEP, bounds[1] + j as f64 * WARP_STEP);
                let (px, py) = pack.from_lonlat(lon, lat);
                warp.push([px, py]);
            }
        }
        GeoMask { mask, bounds, nx, warp }
    }

    /// A world point in the pack's frame (bilinear), or `None` outside the lattice.
    pub fn to_pack(&self, x: f64, y: f64) -> Option<(f64, f64)> {
        if x < self.bounds[0] || y < self.bounds[1] {
            return None;
        }
        let (fx, fy) = ((x - self.bounds[0]) / WARP_STEP, (y - self.bounds[1]) / WARP_STEP);
        let (i, j) = (fx as usize, fy as usize);
        let ny = self.warp.len() / self.nx;
        if i + 1 >= self.nx || j + 1 >= ny {
            return None;
        }
        let (tx, ty) = (fx - i as f64, fy - j as f64);
        let at = |i: usize, j: usize| self.warp[j * self.nx + i];
        let (a, b, c, d) = (at(i, j), at(i + 1, j), at(i, j + 1), at(i + 1, j + 1));
        let lerp = |k: usize| (a[k] * (1.0 - tx) + b[k] * tx) * (1.0 - ty) + (c[k] * (1.0 - tx) + d[k] * tx) * ty;
        Some((lerp(0), lerp(1)))
    }
}
impl WaterMask for GeoMask {
    fn is_water(&self, x: f64, y: f64) -> bool {
        match self.to_pack(x, y) {
            Some((px, py)) => self.mask.is_water(px, py),
            None => false,
        }
    }
}

/// Keep overlapping city rasters as a union: an uncovered point in one is not evidence of land.
pub struct UnionMask(pub Box<dyn WaterMask>, pub Box<dyn WaterMask>);
impl WaterMask for UnionMask {
    fn is_water(&self, x: f64, y: f64) -> bool {
        self.0.is_water(x, y) || self.1.is_water(x, y)
    }
    fn is_empty(&self) -> bool { self.0.is_empty() && self.1.is_empty() }
}

/// Several rasters as one union (T-1039), each with its bounds in the network frame
/// (`[xmin, ymin, xmax, ymax]`), outside which it is land; a point is tested only against the
/// rasters around it.
pub struct MaskSet(pub Vec<(Arc<dyn WaterMask>, [f64; 4])>);
impl WaterMask for MaskSet {
    fn is_water(&self, x: f64, y: f64) -> bool {
        self.0.iter().any(|(m, b)| x >= b[0] && y >= b[1] && x <= b[2] && y <= b[3] && m.is_water(x, y))
    }
    fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl<F: Fn(f64, f64) -> bool> WaterMask for F {
    fn is_water(&self, x: f64, y: f64) -> bool {
        self(x, y)
    }
}

/// The city pack's water mask (notes/T-004.md "Water mask"): a grid of `cell` metre pixels from
/// the south-west corner (`x0`, `y0`), `w` x `h`, stored as row runs. Row r's entries are
/// `xs[row[r] .. row[r+1]]`, pairs (start, end): columns start..end-1 are water. Outside the grid
/// is land.
pub struct RunMask {
    pub x0: f64,
    pub y0: f64,
    pub cell: f64,
    pub w: u32,
    pub h: u32,
    pub row: Vec<u32>,
    pub xs: Vec<u16>,
}

impl RunMask {
    /// From the pack header's `water` block and the two arrays' bytes (little-endian).
    pub fn from_pack(cell: f64, origin: [f64; 2], size: [u32; 2], row_bytes: &[u8], x_bytes: &[u8]) -> RunMask {
        let row = row_bytes.chunks_exact(4).map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
        let xs = x_bytes.chunks_exact(2).map(|b| u16::from_le_bytes([b[0], b[1]])).collect();
        RunMask { x0: origin[0], y0: origin[1], cell, w: size[0], h: size[1], row, xs }
    }
}

impl WaterMask for RunMask {
    fn is_water(&self, x: f64, y: f64) -> bool {
        let c = ((x - self.x0) / self.cell).floor();
        let r = ((y - self.y0) / self.cell).floor();
        if c < 0.0 || r < 0.0 || c >= self.w as f64 || r >= self.h as f64 {
            return false;
        }
        let r = r as usize;
        let row = &self.xs[self.row[r] as usize..self.row[r + 1] as usize];
        row.partition_point(|&v| (v as f64) <= c) % 2 == 1
    }
}

/// Chainage spans `[s0, s1]` over water along an alignment, sampled every `WATER_STEP` m.
pub fn water_spans(len: f64, at: impl Fn(f64) -> (f64, f64), mask: &dyn WaterMask, out: &mut Vec<[f64; 2]>) {
    if mask.is_empty() || len <= 0.0 {
        return;
    }
    let n = (len / WATER_STEP).ceil().max(1.0) as usize;
    let step = len / n as f64;
    let mut open: Option<f64> = None;
    for j in 0..n {
        let (x, y) = at((j as f64 + 0.5) * step);
        let w = mask.is_water(x, y);
        match (w, open) {
            (true, None) => open = Some(j as f64 * step),
            (false, Some(s0)) => {
                out.push([s0, j as f64 * step]);
                open = None;
            }
            _ => {}
        }
    }
    if let Some(s0) = open {
        out.push([s0, len]);
    }
}

/// Terrain classes on a grid of whole fractions of a degree (T-1270, `pipeline/zone_terrain.py`):
/// 0 flat, 1 rolling, 2 hilly, 3 mountain, row after row from the south-west corner. Every zone's
/// grid sits on one world lattice, so overlapping grids agree cell for cell.
pub struct ClassGrid {
    /// lattice cells a degree
    pub per_deg: f64,
    /// lattice column and row of the south-west cell, counted from 180 W and 90 S
    pub col0: i64,
    pub row0: i64,
    pub w: u32,
    pub h: u32,
    pub cls: Vec<u8>,
}

/// A terrain file's magic: `TWT1`, then u32 w, h, i32 col0, row0, u32 cells a degree, then
/// (class u8, run length as LEB128) runs covering the grid row-major.
pub const TERRAIN_MAGIC: &[u8; 4] = b"TWT1";

impl ClassGrid {
    pub fn decode(b: &[u8]) -> Option<ClassGrid> {
        if b.len() < 24 || &b[0..4] != TERRAIN_MAGIC {
            return None;
        }
        let u = |i: usize| u32::from_le_bytes([b[i], b[i + 1], b[i + 2], b[i + 3]]);
        let (w, h) = (u(4), u(8));
        let (col0, row0) = (u(12) as i32 as i64, u(16) as i32 as i64);
        let per_deg = u(20) as f64;
        let n = w as usize * h as usize;
        if per_deg <= 0.0 || n == 0 || n > 64 << 20 {
            return None;
        }
        let mut cls = Vec::with_capacity(n);
        let mut i = 24;
        while i < b.len() && cls.len() < n {
            let c = b[i].min(3);
            i += 1;
            let (mut run, mut shift) = (0usize, 0);
            loop {
                let x = *b.get(i)?;
                i += 1;
                run |= ((x & 0x7f) as usize) << shift;
                if x & 0x80 == 0 { break; }
                shift += 7;
                if shift > 35 { return None; }
            }
            cls.extend(std::iter::repeat(c).take(run.min(n - cls.len())));
        }
        (cls.len() == n).then_some(ClassGrid { per_deg, col0, row0, w, h, cls })
    }

    /// `[west, south, east, north]` in degrees.
    pub fn bounds(&self) -> [f64; 4] {
        let d = 1.0 / self.per_deg;
        let (w, s) = (self.col0 as f64 * d - 180.0, self.row0 as f64 * d - 90.0);
        [w, s, w + self.w as f64 * d, s + self.h as f64 * d]
    }

    /// The class at a point, or `None` outside the grid.
    pub fn at(&self, lon: f64, lat: f64) -> Option<u8> {
        let c = ((lon + 180.0) * self.per_deg).floor() as i64 - self.col0;
        let r = ((lat + 90.0) * self.per_deg).floor() as i64 - self.row0;
        if c < 0 || r < 0 || c >= self.w as i64 || r >= self.h as i64 {
            return None;
        }
        Some(self.cls[r as usize * self.w as usize + c as usize])
    }
}

/// The terrain grids installed (one per zone near the view or the track). A point no grid covers is
/// flat. Cheap to clone: the grids are shared.
#[derive(Clone, Default)]
pub struct Terrain {
    pub grids: Vec<Arc<ClassGrid>>,
}

impl Terrain {
    pub fn class_at(&self, lon: f64, lat: f64) -> u8 {
        self.grids.iter().find_map(|g| g.at(lon, lat)).unwrap_or(0)
    }
    pub fn is_empty(&self) -> bool {
        self.grids.is_empty()
    }
}

/// Chainage spans `[s0, s1, class]` of terrain that is not flat along an alignment, sampled every
/// `TERRAIN_STEP` m; `at` gives longitude and latitude at a chainage.
pub fn terrain_spans(len: f64, at: impl Fn(f64) -> (f64, f64), t: &Terrain, out: &mut Vec<[f64; 3]>) {
    if t.is_empty() || len <= 0.0 {
        return;
    }
    let n = (len / TERRAIN_STEP).ceil().max(1.0) as usize;
    let step = len / n as f64;
    let mut open: Option<(f64, u8)> = None;
    for j in 0..n {
        let (lon, lat) = at((j as f64 + 0.5) * step);
        let c = t.class_at(lon, lat);
        let s = j as f64 * step;
        match open {
            Some((_, k)) if k == c => {}
            Some((s0, k)) => {
                out.push([s0, s, k as f64]);
                open = (c > 0).then_some((s, c));
            }
            None if c > 0 => open = Some((s, c)),
            None => {}
        }
    }
    if let Some((s0, k)) = open {
        out.push([s0, len, k as f64]);
    }
}

/// The terrain factor at a level: none for bored tunnels.
pub fn terrain_factor(level: i8, class: u8) -> f64 {
    if level >= TERRAIN_MIN_LEVEL { TERRAIN_FACTOR[class.min(3) as usize] } else { 1.0 }
}

/// Level of a height, rounded.
pub fn level_of(z: f64) -> i8 {
    (z / LEVEL_H).round() as i8
}

/// Cost of one edge's track on flat ground, US$M (`track_cost_t` with no terrain).
pub fn track_cost(len: f64, vert: &[[f64; 2]], water: &[[f64; 2]], tracks: u8, hsr: bool) -> f64 {
    track_cost_t(len, vert, water, &[], tracks, hsr)
}

/// Cost of one edge's track, US$M: per sub-interval, the level's multiplier (a ramp at the dearer
/// of its two levels), 2x over water, the terrain's factor (`terrain_factor`), `track_factor` for
/// the track type.
pub fn track_cost_t(len: f64, vert: &[[f64; 2]], water: &[[f64; 2]], terrain: &[[f64; 3]], tracks: u8, hsr: bool) -> f64 {
    let mut c = 0.0;
    track_cost_parts_t(len, vert, water, terrain, tracks, hsr, |_, _, _, _, _, cost| c += cost);
    c
}

/// A track type's price over double track: 0.6x single, 1.5x high speed (always double).
pub fn track_factor(tracks: u8, hsr: bool) -> f64 {
    if tracks == 1 { SINGLE_TRACK_COST } else if hsr { HSR_COST } else { 1.0 }
}

/// Visit each priced interval on flat ground: charged level, water, ramp, metres, US$M.
pub fn track_cost_parts(len: f64, vert: &[[f64; 2]], water: &[[f64; 2]], tracks: u8, hsr: bool, mut part: impl FnMut(i8, bool, bool, f64, f64)) {
    track_cost_parts_t(len, vert, water, &[], tracks, hsr, |l, wet, ramp, _, m, c| part(l, wet, ramp, m, c));
}

/// Visit each priced interval: charged level, water, ramp, terrain class, metres, US$M.
/// Shared by the price and its explanation, including the dearer end of a ramp (terrain counted).
pub fn track_cost_parts_t(len: f64, vert: &[[f64; 2]], water: &[[f64; 2]], terrain: &[[f64; 3]], tracks: u8, hsr: bool, mut part: impl FnMut(i8, bool, bool, u8, f64, f64)) {
    let mut cuts: Vec<f64> = Vec::with_capacity(vert.len() + 2 * water.len() + 2 * terrain.len() + 2);
    cuts.push(0.0);
    cuts.push(len);
    cuts.extend(vert.iter().map(|p| p[0]));
    for w in water {
        cuts.push(w[0]);
        cuts.push(w[1]);
    }
    for t in terrain {
        cuts.push(t[0]);
        cuts.push(t[1]);
    }
    cuts.sort_by(|a, b| a.total_cmp(b));
    cuts.dedup_by(|a, b| (*a - *b).abs() < 1e-9);
    let track_f = track_factor(tracks, hsr);
    for w in cuts.windows(2) {
        let (u, v) = (w[0].max(0.0), w[1].min(len));
        if v <= u {
            continue;
        }
        let m = 0.5 * (u + v);
        let wet = water.iter().any(|s| m >= s[0] && m < s[1]);
        // over water the water's 2x stands for the crossing; the ground's class does not add to it
        let class = if wet { 0 } else { terrain.iter().find(|s| m >= s[0] && m < s[1]).map_or(0, |s| s[2] as u8) };
        let dear = |l: i8| level_mult(l) * terrain_factor(l, class);
        let i = vert.partition_point(|p| p[0] <= m).clamp(1, vert.len().max(2) - 1);
        let (level, ramp) = if vert.len() < 2 {
            (level_of(vert.first().map_or(0.0, |p| p[1])), false)
        } else {
            let a = level_of(vert[i - 1][1]);
            let b = level_of(vert[i][1]);
            (if dear(a) >= dear(b) { a } else { b }, (vert[i - 1][1] - vert[i][1]).abs() > 1e-9)
        };
        let c = (v - u) / 1000.0 * BASE_COST_PER_KM * dear(level) * if wet { WATER_COST } else { 1.0 } * track_f;
        part(level, wet, ramp, class, v - u, c);
    }
}

/// Station cost, US$M: the 200 m price for its level (a fixed part and a part by level) scaled by
/// `0.4 + 0.6 x length / 200`.
pub fn station_cost(level: i8, platform: u16, tracks: u8, wet: bool) -> f64 {
    let base = STATION_FIXED_200 + STATION_LEVEL_200[(level.clamp(MIN_LEVEL, MAX_LEVEL) + 3) as usize];
    base * (0.4 + 0.6 * platform as f64 / 200.0)
        * if tracks == 1 { SINGLE_TRACK_STATION } else { 1.0 }
        * if wet { WATER_COST } else { 1.0 }
}

/// Junction switches: 0.25 km of track at its level; a flying junction adds 0.6 km one level away
/// from the ground (toward it at +3 and -3).
pub fn junction_cost(level: i8, flying: bool, wet: bool) -> f64 {
    let w = if wet { WATER_COST } else { 1.0 };
    let mut c = JUNCTION_KM * BASE_COST_PER_KM * level_mult(level);
    if flying {
        let away = match level {
            MAX_LEVEL => MAX_LEVEL - 1,
            MIN_LEVEL => MIN_LEVEL + 1,
            l if l >= 0 => l + 1,
            l => l - 1,
        };
        c += FLYOVER_KM * BASE_COST_PER_KM * level_mult(away);
    }
    c * w
}

/// A flat crossing: 0.1 km of track at its level.
pub fn crossing_cost(level: i8, wet: bool) -> f64 {
    FLAT_CROSSING_KM * BASE_COST_PER_KM * level_mult(level) * if wet { WATER_COST } else { 1.0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transformed_water_union_keeps_cell_edges_and_uncovered_land() {
        let pack = RunMask { x0: -10.0, y0: -20.0, cell: 5.0, w: 3, h: 2,
            row: vec![0, 2, 4], xs: vec![1, 3, 0, 1] };
        let transformed = TransformedMask { mask: Arc::new(pack), offset: [-100000.0, -90000.0], scale: [0.5, 1.0] };
        let at = |x, y| transformed.is_water(-100000.0 + 0.5 * x, -90000.0 + y);
        assert!(!at(-7.5, -17.5));
        assert!(at(-2.5, -17.5));
        assert!(at(-7.5, -12.5));
        assert!(!at(-2.5, -12.5));
        assert!(!at(5.0, -17.5));
        assert!(!at(-7.5, -10.0));
        let union = UnionMask(Box::new(transformed), Box::new(|x: f64, y: f64| x.abs() < 1.0 && y.abs() < 1.0));
        assert!(union.is_water(0.0, 0.0));
        assert!(union.is_water(-100001.25, -90017.5));
        assert!(!union.is_water(1000.0, 1000.0));
    }

    #[test]
    fn spec_worked_example_20km_at_minus2_with_18_stations() {
        // a 20 km line at -2 with 18 stations is about 4.7B (T-1245)
        let vert = [[0.0, -16.0], [20000.0, -16.0]];
        let track = track_cost(20000.0, &vert, &[], 2, false);
        assert!((track - 1980.0).abs() < 1e-9);
        let stations = 18.0 * station_cost(-2, 200, 2, false);
        assert!((stations - 2700.0).abs() < 1e-9);
        let total = track + stations;
        assert!((total - 4680.0).abs() < 1e-9, "{total}");
    }

    #[test]
    fn multipliers_ramps_water_single() {
        let flat0 = [[0.0, 0.0], [1000.0, 0.0]];
        assert!((track_cost(1000.0, &flat0, &[], 2, false) - 45.0).abs() < 1e-9);
        assert!((track_cost(1000.0, &flat0, &[], 1, false) - 27.0).abs() < 1e-9);
        // +1 viaduct over 500 m of water: 90 x 0.5 + 180 x 0.5
        let v1 = [[0.0, 8.0], [1000.0, 8.0]];
        assert!((track_cost(1000.0, &v1, &[[250.0, 750.0]], 2, false) - 135.0).abs() < 1e-9);
        // 0 -> -2 ramp over 400 m in a 1000 m edge: 300 m at 0.5, 400 m at 1.1 (dearer), 300 m at 1.1.
        let ramp = [[0.0, 0.0], [300.0, 0.0], [700.0, -16.0], [1000.0, -16.0]];
        let want = 90.0 * (0.3 * 0.5 + 0.4 * 1.1 + 0.3 * 1.1);
        assert!((track_cost(1000.0, &ramp, &[], 2, false) - want).abs() < 1e-9);
        // Ramp +1 -> -1 is priced at the dearer end level (-1: 1.0), not at 0.
        let r2 = [[0.0, 8.0], [400.0, -8.0]];
        assert!((track_cost(400.0, &r2, &[], 2, false) - 36.0).abs() < 1e-9);
    }

    #[test]
    fn high_speed_track_is_half_again_double() {
        // T-817: 1.5x double track at its level, water and ramps priced as for double
        let flat0 = [[0.0, 0.0], [1000.0, 0.0]];
        assert!((track_cost(1000.0, &flat0, &[], 2, true) - 67.5).abs() < 1e-9);
        let v1 = [[0.0, 8.0], [1000.0, 8.0]];
        assert!((track_cost(1000.0, &v1, &[[250.0, 750.0]], 2, true) - 202.5).abs() < 1e-9);
        // single track has no high-speed form: the flag is ignored there
        assert!((track_cost(1000.0, &flat0, &[], 1, true) - 27.0).abs() < 1e-9);
    }

    #[test]
    fn stations_junctions_crossings() {
        // T-1267: a fixed 50 plus the level's part
        assert!((station_cost(0, 200, 2, false) - 75.0).abs() < 1e-12);
        assert!((station_cost(1, 200, 2, false) - 90.0).abs() < 1e-12);
        assert!((station_cost(-3, 200, 2, false) - 175.0).abs() < 1e-12);
        assert!((station_cost(-2, 100, 2, false) - 105.0).abs() < 1e-9);
        assert!((station_cost(-2, 400, 2, false) - 240.0).abs() < 1e-9);
        assert!((junction_cost(0, false, false) - 0.25 * 45.0).abs() < 1e-9);
        assert!((junction_cost(0, true, false) - (11.25 + 0.6 * 90.0)).abs() < 1e-9);
        assert!((junction_cost(-2, true, false) - (0.25 * 99.0 + 0.6 * 108.0)).abs() < 1e-9);
        assert!((crossing_cost(0, false) - 4.5).abs() < 1e-9);
    }

    #[test]
    fn breakdown_splits_water_ramps_and_single_track() {
        let vert = [[0.0, 0.0], [300.0, 0.0], [700.0, -16.0], [1000.0, -16.0]];
        let mut rows = vec![];
        track_cost_parts(1000.0, &vert, &[[500.0, 900.0]], 1, false, |l, wet, ramp, m, c| rows.push((l, wet, ramp, m, c)));
        assert_eq!(rows.iter().map(|r| r.3).sum::<f64>(), 1000.0);
        assert_eq!(rows.iter().filter(|r| r.1).map(|r| r.3).sum::<f64>(), 400.0);
        assert_eq!(rows.iter().filter(|r| r.2).map(|r| r.3).sum::<f64>(), 400.0);
        assert!(rows.iter().filter(|r| r.2).all(|r| r.0 == -2));
        assert!((rows.iter().map(|r| r.4).sum::<f64>() - 73.44).abs() < 1e-9);
    }

    /// A terrain file of `w` x `h` cells at 240 a degree from its south-west lattice cell, by runs.
    fn terrain_file(w: u32, h: u32, col0: i32, row0: i32, runs: &[(u8, usize)]) -> Vec<u8> {
        let mut b = TERRAIN_MAGIC.to_vec();
        for v in [w, h, col0 as u32, row0 as u32, 240] {
            b.extend(v.to_le_bytes());
        }
        for &(c, mut n) in runs {
            b.push(c);
            loop {
                let x = (n & 0x7f) as u8;
                n >>= 7;
                if n == 0 { b.push(x); break; }
                b.push(x | 0x80);
            }
        }
        b
    }

    #[test]
    fn terrain_grid_decodes_and_finds_cells() {
        // 3 x 2 cells at 0..3/240 E, 45..45+2/240 N: south row flat, hilly, mountain; north row 200
        // cells long in all by a run that overruns (clipped)
        let b = terrain_file(3, 2, 180 * 240, 135 * 240, &[(0, 1), (2, 1), (3, 1), (1, 300)]);
        let g = ClassGrid::decode(&b).unwrap();
        let d = 1.0 / 240.0;
        assert_eq!(g.at(0.5 * d, 45.0 + 0.5 * d), Some(0));
        assert_eq!(g.at(1.5 * d, 45.0 + 0.5 * d), Some(2));
        assert_eq!(g.at(2.5 * d, 45.0 + 0.5 * d), Some(3));
        assert_eq!(g.at(1.5 * d, 45.0 + 1.5 * d), Some(1));
        assert_eq!(g.at(3.5 * d, 45.0 + 0.5 * d), None);
        assert_eq!(g.at(-0.5 * d, 45.0), None);
        let bb = g.bounds();
        assert!((bb[0] - 0.0).abs() < 1e-9 && (bb[1] - 45.0).abs() < 1e-9 && (bb[2] - 3.0 * d).abs() < 1e-9);
        assert!(ClassGrid::decode(&b[..b.len() - 2]).is_none(), "a short file is refused");
        assert!(ClassGrid::decode(b"TWT0").is_none());
    }

    #[test]
    fn terrain_prices_surface_track_not_bored_tunnels() {
        // T-1270: 1 km at each level in the mountain class (2.5x), the tunnels at their flat price
        let mtn = [[0.0, 1000.0, 3.0]];
        let at = |l: f64| [[0.0, l * 8.0], [1000.0, l * 8.0]];
        assert!((track_cost_t(1000.0, &at(0.0), &[], &mtn, 2, false) - 112.5).abs() < 1e-9);
        assert!((track_cost_t(1000.0, &at(1.0), &[], &mtn, 2, false) - 225.0).abs() < 1e-9);
        assert!((track_cost_t(1000.0, &at(-1.0), &[], &mtn, 2, false) - 225.0).abs() < 1e-9);
        assert!((track_cost_t(1000.0, &at(-2.0), &[], &mtn, 2, false) - 99.0).abs() < 1e-9);
        assert!((track_cost_t(1000.0, &at(-3.0), &[], &mtn, 2, false) - 108.0).abs() < 1e-9);
        // so in mountains the tunnel is the cheaper way, in hilly ground the surface still is
        let hilly = [[0.0, 1000.0, 2.0]];
        assert!((track_cost_t(1000.0, &at(0.0), &[], &hilly, 2, false) - 72.0).abs() < 1e-9);
        // half the edge rolling, over water nothing on top of the water's 2x
        let half = [[500.0, 1000.0, 1.0]];
        assert!((track_cost_t(1000.0, &at(0.0), &[], &half, 2, false) - (22.5 + 22.5 * 1.25)).abs() < 1e-9);
        assert!((track_cost_t(1000.0, &at(1.0), &[[0.0, 1000.0]], &mtn, 2, false) - 180.0).abs() < 1e-9);
        // a ramp from ground down to -2 in mountains is charged at the dearer end: ground's 2.5x
        let ramp = [[0.0, 0.0], [400.0, -16.0]];
        assert!((track_cost_t(400.0, &ramp, &[], &mtn, 2, false) - 0.4 * 112.5).abs() < 1e-9);
        let mut classes = vec![];
        track_cost_parts_t(1000.0, &at(0.0), &[], &half, 2, false, |_, _, _, c, m, _| classes.push((c, m)));
        assert_eq!(classes, vec![(0, 500.0), (1, 500.0)]);
    }

    #[test]
    fn terrain_spans_merge_runs_and_skip_flat() {
        let b = terrain_file(4, 1, 180 * 240, 135 * 240, &[(0, 1), (2, 2), (0, 1)]);
        let t = Terrain { grids: vec![Arc::new(ClassGrid::decode(&b).unwrap())] };
        let d = 1.0 / 240.0;
        let mut out = vec![];
        // 4 cells along a parallel, about 1.3 km at 45 N
        terrain_spans(4000.0, |s| (4.0 * d * s / 4000.0, 45.0 + 0.5 * d), &t, &mut out);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0][2], 2.0);
        assert!((out[0][0] - 1000.0).abs() <= 100.0 && (out[0][1] - 3000.0).abs() <= 100.0, "{out:?}");
        assert_eq!(t.class_at(10.0, 10.0), 0, "outside every grid is flat");
    }

    #[test]
    fn water_spans_sampled() {
        let mask = |x: f64, _y: f64| (400.0..600.0).contains(&x);
        let mut out = vec![];
        water_spans(1000.0, |s| (s, 0.0), &mask, &mut out);
        assert_eq!(out.len(), 1);
        assert!((out[0][0] - 400.0).abs() <= 25.0 && (out[0][1] - 600.0).abs() <= 25.0);
        // Row 0: columns 1..3 water; row 1: none.
        let r = RunMask { x0: 0.0, y0: 0.0, cell: 50.0, w: 4, h: 2, row: vec![0, 2, 2], xs: vec![1, 3] };
        assert!(!r.is_water(10.0, 10.0) && r.is_water(60.0, 10.0) && r.is_water(149.0, 10.0));
        assert!(!r.is_water(151.0, 10.0) && !r.is_water(60.0, 60.0) && !r.is_water(-1.0, 0.0));
    }
}
