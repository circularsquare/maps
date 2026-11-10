//! Build costs (SPEC 6.4) and the water mask interface.

use super::frame::Frame;
use super::params::*;
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

/// Where the water is. T-030 ships a mask in the city pack; anything that can answer "is this
/// point water?" in local metres works here.
pub trait WaterMask {
    fn is_water(&self, x: f64, y: f64) -> bool;
    /// True where this mask's raster says land or water, so a coarser one does not decide (T-1285).
    fn covers(&self, _x: f64, _y: f64) -> bool {
        false
    }
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
    fn covers(&self, x: f64, y: f64) -> bool {
        self.mask.covers((x - self.offset[0]) / self.scale[0], (y - self.offset[1]) / self.scale[1])
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
    fn covers(&self, x: f64, y: f64) -> bool {
        self.to_pack(x, y).is_some_and(|(px, py)| self.mask.covers(px, py))
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

/// The packs' rasters as one union (T-1039), each with its bounds in the network frame
/// (`[xmin, ymin, xmax, ymax]`); a point is tested only against the rasters around it. A point no
/// raster covers is what the world's coarse mask says (T-1285), land without one.
pub struct MaskSet {
    pub rasters: Vec<(Arc<dyn WaterMask>, [f64; 4])>,
    pub world: Option<Arc<dyn WaterMask>>,
}
impl WaterMask for MaskSet {
    fn is_water(&self, x: f64, y: f64) -> bool {
        let mut covered = false;
        for (m, b) in &self.rasters {
            if x >= b[0] && y >= b[1] && x <= b[2] && y <= b[3] {
                if m.is_water(x, y) {
                    return true;
                }
                covered = covered || m.covers(x, y);
            }
        }
        !covered && self.world.as_ref().is_some_and(|w| w.is_water(x, y))
    }
    fn is_empty(&self) -> bool {
        self.rasters.is_empty() && self.world.is_none()
    }
}

/// The world's coarse water (T-1285, `pipeline/world_water.py`): a `RunMask` whose cells are
/// fractions of a degree from 180 W and 90 S, read through the network's frame.
pub struct WorldWater {
    pub mask: Arc<RunMask>,
    pub frame: Frame,
}

/// A world water file's magic: `TWW1`, then u32 width, height, cells a degree, then per row from the
/// south a LEB128 count of boundaries and the boundaries as LEB128 gaps (water starts and ends in
/// turn).
pub const WORLD_WATER_MAGIC: &[u8; 4] = b"TWW1";

impl WorldWater {
    pub fn decode(b: &[u8]) -> Option<RunMask> {
        if b.len() < 16 || &b[0..4] != WORLD_WATER_MAGIC {
            return None;
        }
        let u = |i: usize| u32::from_le_bytes([b[i], b[i + 1], b[i + 2], b[i + 3]]);
        let (w, h, per) = (u(4), u(8), u(12));
        if per == 0 || w != 360 * per || h != 180 * per || w > u16::MAX as u32 {
            return None;
        }
        let mut i = 16;
        let mut leb = || -> Option<u32> {
            let (mut v, mut shift) = (0u32, 0);
            loop {
                let x = *b.get(i)?;
                i += 1;
                v |= ((x & 0x7f) as u32) << shift;
                if x & 0x80 == 0 { return Some(v); }
                shift += 7;
                if shift > 28 { return None; }
            }
        };
        let mut row = Vec::with_capacity(h as usize + 1);
        let mut xs: Vec<u16> = vec![];
        row.push(0);
        for _ in 0..h {
            let n = leb()?;
            if n % 2 == 1 || n > w + 1 {
                return None;
            }
            let mut at = 0u32;
            for _ in 0..n {
                at = at.checked_add(leb()?)?;
                if at > w {
                    return None;
                }
                xs.push(at as u16);
            }
            row.push(xs.len() as u32);
        }
        Some(RunMask { x0: -180.0, y0: -90.0, cell: 1.0 / per as f64, w, h, row, xs })
    }
}

impl WaterMask for WorldWater {
    fn is_water(&self, x: f64, y: f64) -> bool {
        let (lon, lat) = self.frame.to_lonlat(x, y);
        self.mask.is_water((lon + 180.0).rem_euclid(360.0) - 180.0, lat)
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
    fn covers(&self, x: f64, y: f64) -> bool {
        let (c, r) = ((x - self.x0) / self.cell, (y - self.y0) / self.cell);
        c >= 0.0 && r >= 0.0 && c < self.w as f64 && r < self.h as f64
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

/// Terrain classes on a grid of whole fractions of a degree (T-1270, T-2901, `pipeline/world_terrain.py`):
/// 0 flat, 1 rolling, 2 hilly, 3 mountain, row after row from the south-west corner. The world's land is
/// one such grid per degree block, 960 x 960 cells. Two bits a cell (a block is 230 KB).
pub struct ClassGrid {
    /// lattice cells a degree
    pub per_deg: f64,
    /// lattice column and row of the south-west cell, counted from 180 W and 90 S
    pub col0: i64,
    pub row0: i64,
    pub w: u32,
    pub h: u32,
    /// four cells a byte, the first in the low bits
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
        if per_deg <= 0.0 || n == 0 || n > 256 << 20 {
            return None;
        }
        let mut cls = vec![0u8; n.div_ceil(4)];
        let (mut i, mut k) = (24, 0usize);
        while i < b.len() && k < n {
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
            let end = k + run.min(n - k);
            if c > 0 {
                // whole bytes of a run at once: a big zone is tens of millions of cells
                let mut j = k;
                while j < end && j & 3 != 0 {
                    cls[j >> 2] |= c << ((j & 3) * 2);
                    j += 1;
                }
                let whole = end & !3;
                if j < whole {
                    cls[j >> 2..whole >> 2].fill(c * 0x55);
                    j = whole;
                }
                while j < end {
                    cls[j >> 2] |= c << ((j & 3) * 2);
                    j += 1;
                }
            }
            k = end;
        }
        (k == n).then_some(ClassGrid { per_deg, col0, row0, w, h, cls })
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
        let j = r as usize * self.w as usize + c as usize;
        Some((self.cls[j >> 2] >> ((j & 3) * 2)) & 3)
    }
}

/// A degree block as one number: `(la + 90) * 360 + (lo + 180)`, `la` and `lo` the block's south-west
/// corner in whole degrees. Must match the app's `blockKey` (`game/terrainBlocks.ts`).
pub fn block_key(lon: f64, lat: f64) -> i32 {
    let lo = (lon + 180.0).rem_euclid(360.0).floor() as i32;
    let la = (lat + 90.0).floor().clamp(0.0, 179.0) as i32;
    la * 360 + lo.min(359)
}

/// Which degree blocks have a terrain file: `index.bin` from `pipeline/world_terrain.py`. A block it does
/// not list has no land to price (open sea, or Antarctica): known flat, nothing to fetch.
pub struct BlockIndex {
    south: i32,
    rows: i32,
    west: i32,
    cols: i32,
    bits: Vec<u8>,
}

impl BlockIndex {
    /// `TWI1`, i16 south, u16 rows, i16 west, u16 columns, u32 blocks listed, then a bit a block
    /// row-major from the south-west, the first in the low bit of each byte.
    pub fn decode(b: &[u8]) -> Option<BlockIndex> {
        if b.len() < 16 || &b[0..4] != b"TWI1" {
            return None;
        }
        let i16_at = |i: usize| i16::from_le_bytes([b[i], b[i + 1]]) as i32;
        let u16_at = |i: usize| u16::from_le_bytes([b[i], b[i + 1]]) as i32;
        let (south, rows, west, cols) = (i16_at(4), u16_at(6), i16_at(8), u16_at(10));
        let bits = b[16..].to_vec();
        (rows > 0 && cols > 0 && ((rows * cols) as usize).div_ceil(8) == bits.len()).then_some(BlockIndex { south, rows, west, cols, bits })
    }

    pub fn has(&self, key: i32) -> bool {
        let (la, lo) = (key.div_euclid(360) - 90 - self.south, key.rem_euclid(360) - 180 - self.west);
        if la < 0 || lo < 0 || la >= self.rows || lo >= self.cols {
            return false;
        }
        let i = (la * self.cols + lo) as usize;
        self.bits[i >> 3] >> (i & 7) & 1 == 1
    }
}

/// The terrain blocks installed, by `block_key` (T-2901). The track model loads the blocks under what it
/// prices and the view reaches; a point in a block not installed reads flat, and the block goes on the
/// `wanted` list for the worker to fetch. Cheap to clone: the grids are shared, and so is the list, so a
/// preview's model reports what it wanted to the same place.
#[derive(Clone, Default)]
pub struct Terrain {
    blocks: HashMap<i32, Arc<ClassGrid>>,
    index: Option<Arc<BlockIndex>>,
    wanted: Arc<Mutex<HashSet<i32>>>,
}

/// Longitude and latitude of the centre of the terrain hex a point is in (T-2890): pointy-topped
/// hexes, regular on the mercator plane, so the map draws them as regular hexes. Must match the
/// app's `hexCentre` (`game/terrain.ts`), which draws each hex in the class of its centre.
pub fn hex_centre(lon: f64, lat: f64) -> (f64, f64) {
    use std::f64::consts::PI;
    let row = TERRAIN_HEX * 3f64.sqrt() / 2.0;
    let x = (lon + 180.0) / 360.0 - TERRAIN_HEX_SHIFT;
    let y = 0.5 - (PI / 4.0 + lat.to_radians() / 2.0).tan().ln() / (2.0 * PI);
    let rf = y / row;
    let qf = x / TERRAIN_HEX - rf / 2.0;
    let sf = -qf - rf;
    let (mut q, mut r, s) = (qf.round(), rf.round(), sf.round());
    let (dq, dr, ds) = ((q - qf).abs(), (r - rf).abs(), (s - sf).abs());
    if dq > dr && dq > ds {
        q = -r - s;
    } else if dr > ds {
        r = -q - s;
    }
    let (cx, cy) = (TERRAIN_HEX * (q + r / 2.0) + TERRAIN_HEX_SHIFT, row * r);
    (cx * 360.0 - 180.0, (PI * (1.0 - 2.0 * cy)).sinh().atan().to_degrees())
}

impl Terrain {
    /// A block's grid: its key from its south-west cell.
    pub fn add(&mut self, grid: Arc<ClassGrid>) {
        let per = grid.per_deg as i64;
        self.blocks.insert((grid.row0.div_euclid(per) * 360 + grid.col0.div_euclid(per)) as i32, grid);
    }

    pub fn remove(&mut self, key: i32) {
        self.blocks.remove(&key);
    }

    pub fn set_index(&mut self, index: BlockIndex) {
        self.index = Some(Arc::new(index));
    }

    /// Whether a block should have a file; every block while the index is not in.
    pub fn listed(&self, key: i32) -> bool {
        self.index.as_ref().map_or(true, |i| i.has(key))
    }

    /// The grid of a block, or `None` with the block marked wanted when it should have one.
    fn grid(&self, key: i32) -> Option<&ClassGrid> {
        if let Some(g) = self.blocks.get(&key) {
            return Some(g);
        }
        if self.listed(key) {
            self.wanted.lock().unwrap().insert(key);
        }
        None
    }

    /// The class charged at a point: that of its hex's centre (T-2890).
    pub fn class_at(&self, lon: f64, lat: f64) -> u8 {
        let (lon, lat) = hex_centre(lon, lat);
        self.grid(block_key(lon, lat)).and_then(|g| g.at(wrap_lon(lon), lat)).unwrap_or(0)
    }

    /// Blocks asked for and not installed since the last call (listed ones, not listed ones being
    /// flat). Nothing while the index is not in, the list then kept for when it is.
    pub fn take_wanted(&self) -> Vec<i32> {
        if self.index.is_none() {
            return vec![];
        }
        let mut w = self.wanted.lock().unwrap();
        let out = w.iter().copied().filter(|k| !self.blocks.contains_key(k) && self.listed(*k)).collect();
        w.clear();
        out
    }
}

/// A longitude into -180..180, as hex centres can stray just past the antimeridian.
fn wrap_lon(lon: f64) -> f64 {
    (lon + 180.0).rem_euclid(360.0) - 180.0
}

/// How many terrain samples an alignment of length `len` has and their spacing; the n-th (from 0) is
/// at chainage `(n + 0.5) * step`.
fn terrain_samples(len: f64) -> (usize, f64) {
    let n = (len / TERRAIN_STEP).ceil().max(1.0) as usize;
    (n, len / n as f64)
}

/// Chainage spans `[s0, s1, class]` of terrain that is not flat along an alignment, sampled every
/// `TERRAIN_STEP` m; `at` gives longitude and latitude at a chainage.
pub fn terrain_spans(len: f64, at: impl Fn(f64) -> (f64, f64), t: &Terrain, out: &mut Vec<[f64; 3]>) {
    if len <= 0.0 {
        return;
    }
    let (n, step) = terrain_samples(len);
    let mut open: Option<(f64, u8)> = None;
    let (mut key, mut grid) = (-1, None);
    for j in 0..n {
        let (lon, lat) = at((j as f64 + 0.5) * step);
        let (lon, lat) = hex_centre(lon, lat);
        let k = block_key(lon, lat);
        if k != key {
            key = k;
            grid = t.grid(k);
        }
        let c = grid.and_then(|g| g.at(wrap_lon(lon), lat)).unwrap_or(0);
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

/// Every block the terrain samples of an alignment fall in, as `terrain_spans` reads them.
pub fn terrain_blocks(len: f64, at: impl Fn(f64) -> (f64, f64), out: &mut HashSet<i32>) {
    if len <= 0.0 {
        return;
    }
    let (n, step) = terrain_samples(len);
    let mut last = -1;
    for j in 0..n {
        let (lon, lat) = at((j as f64 + 0.5) * step);
        let (lon, lat) = hex_centre(lon, lat);
        let k = block_key(lon, lat);
        if k != last {
            last = k;
            out.insert(k);
        }
    }
}

/// What the terrain adds to a level's multiplier: nothing for bored tunnels.
pub fn terrain_surcharge(level: i8, class: u8) -> f64 {
    if level >= TERRAIN_MIN_LEVEL { TERRAIN_SURCHARGE[class.min(3) as usize] } else { 0.0 }
}

/// Upkeep rate a route-km of double track, real US$, for an edge whose levels run from `lo` to `hi`:
/// the dearest structure in that range (bored -2 and -3, then cut and cover -1, then viaduct, then
/// ground) (T-1931).
pub fn track_upkeep_km(lo: i32, hi: i32) -> f64 {
    if lo <= -2 { UPKEEP_BORED_KM } else if lo == -1 { UPKEEP_CUT_KM } else if hi > 0 { UPKEEP_VIADUCT_KM } else { UPKEEP_GROUND_KM }
}

/// A station's price on track of `kind`: the rail formula, or a tram stop's flat price at ground with the
/// rail formula's step for its level (T-2921 section 8).
pub fn station_cost_k(kind: TrackKind, level: i8, platform: u16, tracks: u8, wet: bool) -> f64 {
    match kind.spec().station_flat {
        None => station_cost(level, platform, tracks, wet),
        Some(flat) => {
            let at = |l: i8| STATION_FIXED_200 + STATION_LEVEL_200[(l.clamp(MIN_LEVEL, MAX_LEVEL) + 3) as usize];
            flat * at(level) / at(0) * if wet { WATER_COST } else { 1.0 }
        }
    }
}

/// Upkeep rate of a station at a level, real US$ a day (T-2619): underground (-1 to -3) or not.
pub fn station_upkeep(level: i8) -> f64 {
    if level < 0 { UPKEEP_STATION_UNDER } else { UPKEEP_STATION }
}

/// Level of a height, rounded.
pub fn level_of(z: f64) -> i8 {
    (z / LEVEL_H).round() as i8
}

/// Cost of one edge's track on flat ground, US$M (`track_cost_t` with no terrain).
pub fn track_cost(len: f64, vert: &[[f64; 2]], water: &[[f64; 2]], tracks: u8, kind: TrackKind) -> f64 {
    track_cost_t(len, vert, water, &[], tracks, kind)
}

/// Cost of one edge's track, US$M: per sub-interval, the level's multiplier (a ramp at the dearer
/// of its two levels) plus the terrain's surcharge (`terrain_surcharge`) or water's, times
/// `track_factor` for the track type.
pub fn track_cost_t(len: f64, vert: &[[f64; 2]], water: &[[f64; 2]], terrain: &[[f64; 3]], tracks: u8, kind: TrackKind) -> f64 {
    let mut c = 0.0;
    track_cost_parts_t(len, vert, water, terrain, tracks, kind, |_, _, _, _, _, cost| c += cost);
    c
}

/// A track type's upkeep over double ground rail: 0.6x single, then the kind's `upkeep` (1.5x high speed,
/// which is always double).
pub fn track_factor(tracks: u8, kind: TrackKind) -> f64 {
    if tracks == 1 { SINGLE_TRACK_COST } else { kind.spec().upkeep }
}

/// What the preview's multiplier is measured against: single track's 0.6, and the kind's own price scale
/// (high speed's 1.5), so the multiplier is the level's figure from the kind's table.
pub fn price_scale(tracks: u8, kind: TrackKind) -> f64 {
    if tracks == 1 { SINGLE_TRACK_COST } else { kind.spec().price_scale }
}

/// Visit each priced interval on flat ground: charged level, water, ramp, metres, US$M.
pub fn track_cost_parts(len: f64, vert: &[[f64; 2]], water: &[[f64; 2]], tracks: u8, kind: TrackKind, mut part: impl FnMut(i8, bool, bool, f64, f64)) {
    track_cost_parts_t(len, vert, water, &[], tracks, kind, |l, wet, ramp, _, m, c| part(l, wet, ramp, m, c));
}

/// Visit each priced interval: charged level, water, ramp, terrain class, metres, US$M.
/// Shared by the price and its explanation, including the dearer end of a ramp (terrain counted).
pub fn track_cost_parts_t(len: f64, vert: &[[f64; 2]], water: &[[f64; 2]], terrain: &[[f64; 3]], tracks: u8, kind: TrackKind, mut part: impl FnMut(i8, bool, bool, u8, f64, f64)) {
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
    // single track has no special kind: the flag is ignored there
    let spec = if tracks == 1 { TrackKind::Regular.spec() } else { kind.spec() };
    let track_f = if tracks == 1 { SINGLE_TRACK_COST } else { 1.0 };
    for w in cuts.windows(2) {
        let (u, v) = (w[0].max(0.0), w[1].min(len));
        if v <= u {
            continue;
        }
        let m = 0.5 * (u + v);
        let wet = water.iter().any(|s| m >= s[0] && m < s[1]);
        // over water the water's surcharge stands for the crossing; the ground's class adds nothing
        let class = if wet { 0 } else { terrain.iter().find(|s| m >= s[0] && m < s[1]).map_or(0, |s| s[2] as u8) };
        let dear = |l: i8| spec.level_mult[(l.clamp(MIN_LEVEL, MAX_LEVEL) + 3) as usize] + terrain_surcharge(l, class) * spec.surcharge_mult;
        let i = vert.partition_point(|p| p[0] <= m).clamp(1, vert.len().max(2) - 1);
        let (level, ramp) = if vert.len() < 2 {
            (level_of(vert.first().map_or(0.0, |p| p[1])), false)
        } else {
            let a = level_of(vert[i - 1][1]);
            let b = level_of(vert[i][1]);
            (if dear(a) >= dear(b) { a } else { b }, (vert[i - 1][1] - vert[i][1]).abs() > 1e-9)
        };
        let c = (v - u) / 1000.0 * BASE_COST_PER_KM * (dear(level) + if wet { WATER_SURCHARGE * spec.surcharge_mult } else { 0.0 }) * track_f;
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
        // a 20 km line at -2 with 18 stations is about 5.1B (T-1245, T-1928)
        let vert = [[0.0, -16.0], [20000.0, -16.0]];
        let track = track_cost(20000.0, &vert, &[], 2, TrackKind::Regular);
        assert!((track - 2400.0).abs() < 1e-9);
        let stations = 18.0 * station_cost(-2, 200, 2, false);
        assert!((stations - 2700.0).abs() < 1e-9);
        let total = track + stations;
        assert!((total - 5100.0).abs() < 1e-9, "{total}");
    }

    #[test]
    fn multipliers_ramps_water_single() {
        let flat0 = [[0.0, 0.0], [1000.0, 0.0]];
        assert!((track_cost(1000.0, &flat0, &[], 2, TrackKind::Regular) - 37.5).abs() < 1e-9);
        assert!((track_cost(1000.0, &flat0, &[], 1, TrackKind::Regular) - 22.5).abs() < 1e-9);
        // +1 viaduct over 500 m of water: 75 x 0.5 x 1 + 75 x 0.5 x (1 + 2.5) (T-1890, T-1928)
        let v1 = [[0.0, 8.0], [1000.0, 8.0]];
        assert!((track_cost(1000.0, &v1, &[[250.0, 750.0]], 2, TrackKind::Regular) - 168.75).abs() < 1e-9);
        // 0 -> -2 ramp over 400 m in a 1000 m edge: 300 m at 0.5, 400 m at 1.6 (dearer), 300 m at 1.6.
        let ramp = [[0.0, 0.0], [300.0, 0.0], [700.0, -16.0], [1000.0, -16.0]];
        let want = 75.0 * (0.3 * 0.5 + 0.4 * 1.6 + 0.3 * 1.6);
        assert!((track_cost(1000.0, &ramp, &[], 2, TrackKind::Regular) - want).abs() < 1e-9);
        // Ramp +1 -> -1 is priced at the dearer end level (both 1.0), not at 0.
        let r2 = [[0.0, 8.0], [400.0, -8.0]];
        assert!((track_cost(400.0, &r2, &[], 2, TrackKind::Regular) - 30.0).abs() < 1e-9);
        // every level's price for 1 km, double track (T-1928)
        for (l, want) in [(-3, 150.0), (-2, 120.0), (-1, 75.0), (0, 37.5), (1, 75.0), (2, 112.5), (3, 150.0)] {
            let v = [[0.0, l as f64 * 8.0], [1000.0, l as f64 * 8.0]];
            assert!((track_cost(1000.0, &v, &[], 2, TrackKind::Regular) - want).abs() < 1e-9, "level {l}");
        }
    }

    #[test]
    fn high_speed_track_is_half_again_double() {
        // T-817: 1.5x double track at its level, water and ramps priced as for double
        let flat0 = [[0.0, 0.0], [1000.0, 0.0]];
        assert!((track_cost(1000.0, &flat0, &[], 2, TrackKind::HighSpeed) - 56.25).abs() < 1e-9);
        let v1 = [[0.0, 8.0], [1000.0, 8.0]];
        assert!((track_cost(1000.0, &v1, &[[250.0, 750.0]], 2, TrackKind::HighSpeed) - 253.125).abs() < 1e-9);
        // single track has no high-speed form: the flag is ignored there
        assert!((track_cost(1000.0, &flat0, &[], 1, TrackKind::HighSpeed) - 22.5).abs() < 1e-9);
    }

    #[test]
    fn stations_junctions_crossings() {
        // T-1267: a fixed 50 plus the level's part
        assert!((station_cost(0, 200, 2, false) - 75.0).abs() < 1e-12);
        assert!((station_cost(1, 200, 2, false) - 90.0).abs() < 1e-12);
        assert!((station_cost(-3, 200, 2, false) - 175.0).abs() < 1e-12);
        assert!((station_cost(-2, 100, 2, false) - 105.0).abs() < 1e-9);
        assert!((station_cost(-2, 400, 2, false) - 240.0).abs() < 1e-9);
        // junction and crossing tracks follow the level table (T-1928)
        assert!((junction_cost(0, false, false) - 0.25 * 37.5).abs() < 1e-9);
        assert!((junction_cost(0, true, false) - (0.25 * 37.5 + 0.6 * 75.0)).abs() < 1e-9);
        assert!((junction_cost(-2, true, false) - (0.25 * 120.0 + 0.6 * 150.0)).abs() < 1e-9);
        assert!((crossing_cost(0, false) - 3.75).abs() < 1e-9);
    }

    #[test]
    fn breakdown_splits_water_ramps_and_single_track() {
        let vert = [[0.0, 0.0], [300.0, 0.0], [700.0, -16.0], [1000.0, -16.0]];
        let mut rows = vec![];
        track_cost_parts(1000.0, &vert, &[[500.0, 900.0]], 1, TrackKind::Regular, |l, wet, ramp, m, c| rows.push((l, wet, ramp, m, c)));
        assert_eq!(rows.iter().map(|r| r.3).sum::<f64>(), 1000.0);
        assert_eq!(rows.iter().filter(|r| r.1).map(|r| r.3).sum::<f64>(), 400.0);
        assert_eq!(rows.iter().filter(|r| r.2).map(|r| r.3).sum::<f64>(), 400.0);
        assert!(rows.iter().filter(|r| r.2).all(|r| r.0 == -2));
        assert!((rows.iter().map(|r| r.4).sum::<f64>() - 102.15).abs() < 1e-9);
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
        // T-1270, T-1890, T-1928: 1 km at each level in the mountain class (+3), the tunnels at
        // their flat price
        let mtn = [[0.0, 1000.0, 3.0]];
        let at = |l: f64| [[0.0, l * 8.0], [1000.0, l * 8.0]];
        assert!((track_cost_t(1000.0, &at(0.0), &[], &mtn, 2, TrackKind::Regular) - 262.5).abs() < 1e-9);
        assert!((track_cost_t(1000.0, &at(1.0), &[], &mtn, 2, TrackKind::Regular) - 300.0).abs() < 1e-9);
        assert!((track_cost_t(1000.0, &at(-1.0), &[], &mtn, 2, TrackKind::Regular) - 300.0).abs() < 1e-9);
        assert!((track_cost_t(1000.0, &at(-2.0), &[], &mtn, 2, TrackKind::Regular) - 120.0).abs() < 1e-9);
        assert!((track_cost_t(1000.0, &at(-3.0), &[], &mtn, 2, TrackKind::Regular) - 150.0).abs() < 1e-9);
        let hilly = [[0.0, 1000.0, 2.0]];
        assert!((track_cost_t(1000.0, &at(0.0), &[], &hilly, 2, TrackKind::Regular) - 127.5).abs() < 1e-9);
        // half the edge low hills; over water only the water's +2.5
        let half = [[500.0, 1000.0, 1.0]];
        assert!((track_cost_t(1000.0, &at(0.0), &[], &half, 2, TrackKind::Regular) - (18.75 + 37.5)).abs() < 1e-9);
        assert!((track_cost_t(1000.0, &at(1.0), &[[0.0, 1000.0]], &mtn, 2, TrackKind::Regular) - 262.5).abs() < 1e-9);
        // a ramp from ground down to -2 in mountains is charged at the dearer end: the ground's
        let ramp = [[0.0, 0.0], [400.0, -16.0]];
        assert!((track_cost_t(400.0, &ramp, &[], &mtn, 2, TrackKind::Regular) - 0.4 * 262.5).abs() < 1e-9);
        let mut classes = vec![];
        track_cost_parts_t(1000.0, &at(0.0), &[], &half, 2, TrackKind::Regular, |_, _, _, c, m, _| classes.push((c, m)));
        assert_eq!(classes, vec![(0, 500.0), (1, 500.0)]);
    }

    #[test]
    fn a_terrain_hex_has_one_centre() {
        // T-2890: every point of a hex finds the same centre, and that centre is its own, within a
        // hex's reach of the point
        let hex_deg = TERRAIN_HEX * 360.0;
        for i in 0..4000 {
            let (lon, lat) = (-170.0 + i as f64 * 0.0817, -60.0 + i as f64 * 0.0291);
            let (cl, ca) = hex_centre(lon, lat);
            assert_eq!((cl, ca), hex_centre(cl, ca), "at {lon}, {lat}");
            assert!((cl - lon).abs() <= hex_deg && (ca - lat).abs() <= hex_deg, "at {lon}, {lat}: {cl}, {ca}");
        }
    }

    #[test]
    fn terrain_spans_merge_runs_and_skip_flat() {
        let b = terrain_file(4, 1, 180 * 240, 135 * 240, &[(0, 1), (2, 2), (0, 1)]);
        let mut t = Terrain::default();
        t.add(Arc::new(ClassGrid::decode(&b).unwrap()));
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
    fn blocks_are_found_by_key_and_listed_by_the_index() {
        // T-2901: a hilly block at 10-11 E, 45-46 N
        assert_eq!(block_key(10.5, 45.5), 135 * 360 + 190);
        assert_eq!(block_key(10.0, 45.0), 135 * 360 + 190);
        assert_eq!(block_key(-180.0, 45.5), 135 * 360);
        assert_eq!(block_key(180.0, 45.5), 135 * 360, "the antimeridian wraps");
        assert_eq!(block_key(179.9999, 45.5), 135 * 360 + 359);
        let b = terrain_file(240, 240, 190 * 240, 135 * 240, &[(2, 240 * 240)]);
        let mut t = Terrain::default();
        t.add(Arc::new(ClassGrid::decode(&b).unwrap()));
        assert_eq!(t.class_at(10.5, 45.5), 2);
        assert_eq!(t.class_at(11.5, 45.5), 0, "not installed reads flat");
        // nothing is handed over until the index is in
        assert!(t.take_wanted().is_empty());
        // one row at 45 N, three columns from 10 E: the first and the third listed
        let mut idx = b"TWI1".to_vec();
        idx.extend(45i16.to_le_bytes());
        idx.extend(1u16.to_le_bytes());
        idx.extend(10i16.to_le_bytes());
        idx.extend(3u16.to_le_bytes());
        idx.extend(2u32.to_le_bytes());
        idx.push(0b101);
        t.set_index(BlockIndex::decode(&idx).unwrap());
        assert!(t.listed(135 * 360 + 190) && !t.listed(135 * 360 + 191) && t.listed(135 * 360 + 192));
        assert!(!t.listed(134 * 360 + 190), "outside the index is not listed");
        t.class_at(11.5, 45.5); // not listed: not wanted
        t.class_at(12.5, 45.5); // listed: wanted
        assert_eq!(t.take_wanted(), vec![135 * 360 + 192]);
        assert!(t.take_wanted().is_empty(), "taken once");
        assert!(BlockIndex::decode(&idx[..idx.len() - 1]).is_none());
        // a grid removed reads flat again and is wanted again
        t.remove(135 * 360 + 190);
        assert_eq!(t.class_at(10.5, 45.5), 0);
        assert_eq!(t.take_wanted(), vec![135 * 360 + 190]);
    }

    #[test]
    fn spans_read_across_a_block_edge_and_terrain_blocks_lists_them() {
        // T-2901: 2 km along a parallel across 11 E at 45.5 N, a hilly block each side
        let mut t = Terrain::default();
        for lo in [190, 191] {
            let b = terrain_file(240, 240, lo * 240, 135 * 240, &[(2, 240 * 240)]);
            t.add(Arc::new(ClassGrid::decode(&b).unwrap()));
        }
        let m_per_deg = 111_320.0 * 45.5f64.to_radians().cos();
        let at = |s: f64| (11.0 + (s - 1000.0) / m_per_deg, 45.5);
        let mut out = vec![];
        terrain_spans(2000.0, at, &t, &mut out);
        assert_eq!(out.len(), 1, "{out:?}");
        assert!(out[0][0].abs() < 1e-9 && (out[0][1] - 2000.0).abs() < 1e-9 && out[0][2] == 2.0, "{out:?}");
        let mut blocks = HashSet::new();
        terrain_blocks(2000.0, at, &mut blocks);
        assert_eq!(blocks.len(), 2, "{blocks:?}");
        assert!(blocks.contains(&(135 * 360 + 190)) && blocks.contains(&(135 * 360 + 191)));
    }

    /// A world water file at one cell a degree: `runs` per row index, (first water column, first land
    /// column after it).
    fn world_file(runs: &[(u32, &[(u32, u32)])]) -> Vec<u8> {
        let leb = |mut n: u32, b: &mut Vec<u8>| loop {
            let x = (n & 0x7f) as u8;
            n >>= 7;
            if n == 0 { b.push(x); break; }
            b.push(x | 0x80);
        };
        let mut b = WORLD_WATER_MAGIC.to_vec();
        for v in [360u32, 180, 1] {
            b.extend(v.to_le_bytes());
        }
        for r in 0..180 {
            let row = runs.iter().find(|x| x.0 == r).map_or(&[][..], |x| x.1);
            leb(2 * row.len() as u32, &mut b);
            let mut last = 0;
            for &(a, z) in row {
                leb(a - last, &mut b);
                leb(z - a, &mut b);
                last = z;
            }
        }
        b
    }

    #[test]
    fn world_water_decodes_and_wraps() {
        // 40-41 N: water from 70 W to 10 W, and from 175 E to 180
        let b = world_file(&[(130, &[(110, 170), (355, 360)])]);
        let m = Arc::new(WorldWater::decode(&b).unwrap());
        let frame = Frame::equirect(-40.0, 40.0);
        let w = WorldWater { mask: m.clone(), frame };
        let at = |lon: f64, lat: f64| { let (x, y) = frame.from_lonlat(lon, lat); w.is_water(x, y) };
        assert!(at(-40.0, 40.5) && at(-69.9, 40.5) && at(-10.1, 40.5));
        assert!(!at(-9.9, 40.5) && !at(-40.0, 41.1) && !at(-40.0, 39.9));
        assert!(m.is_water(179.5, 40.5) && m.is_water(-180.0 + 359.5, 40.5));
        assert!(WorldWater::decode(&b[..b.len() - 1]).is_none(), "a short file is refused");
        let mut odd = b.clone();
        odd[12] = 2;
        assert!(WorldWater::decode(&odd).is_none(), "the size must be the whole world at its cells a degree");
    }

    #[test]
    fn a_pack_raster_decides_inside_itself_and_the_world_outside() {
        // T-1285: the world says water everywhere; a 100 m x 100 m pack raster at the origin says
        // land but for its east half
        struct Sea;
        impl WaterMask for Sea {
            fn is_water(&self, _: f64, _: f64) -> bool { true }
        }
        let pack = RunMask { x0: 0.0, y0: 0.0, cell: 50.0, w: 2, h: 2, row: vec![0, 2, 4], xs: vec![1, 2, 1, 2] };
        let set = MaskSet { rasters: vec![(Arc::new(pack), [0.0, 0.0, 100.0, 100.0])], world: Some(Arc::new(Sea)) };
        assert!(!set.is_water(25.0, 25.0), "the pack's land wins over the world's water");
        assert!(set.is_water(75.0, 25.0));
        assert!(set.is_water(150.0, 25.0) && set.is_water(-1.0, 50.0), "outside the pack the world decides");
        assert!(!set.is_empty());
        let none = MaskSet { rasters: vec![], world: None };
        assert!(none.is_empty() && !none.is_water(0.0, 0.0));
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
