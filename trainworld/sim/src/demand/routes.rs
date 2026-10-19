//! T-275: routed walking and driving (docs/demand.md 4.3, "Routed walking and driving").
//!
//! What `pipeline/routes.py` adds to a pack, read here:
//! - the walking barrier raster (`walk_row`, `walk_x`: water, less OSM bridges a person may walk
//!   on), on which the walk to each station is searched when a network is set (stations are the
//!   player's, so this cannot be done offline). A station's search is kept for the thread
//!   (`station_walks`), so an edit searches only the stations it moved or added, and the
//!   commute and everyday populations of one city share it.
//! - per part of a zone (its cells split by bank, `walk_part`, T-700), the walk's detour factor
//!   to the parts of the zones within 2 squares (`walk_zone`), for walking the whole way, and the drive's routed-over-straight factor to the zones within 6
//!   squares (`drive_near`) and between supercells beyond (`drive_far`, `drive_super`).
//!
//! A pack without them, or whose zones are not the ones they were made for, has no `Routes`:
//! every walk and drive is the straight line x 1.3, as before.

use serde_json::Value;
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

use super::kernel::Zones;

/// Factor = exp(q / Q_SCALE) for a stored byte q.
pub const Q_SCALE: f32 = 64.0;
/// Search costs of a step to a side neighbour and to a corner neighbour (99 / 70 = 1.41429).
const ORTH: u32 = 70;
const DIAG: u32 = 99;
const INF: u32 = u32::MAX;
/// Buckets of the search's queue (more than the largest spread of queued costs).
const RING: u32 = 256;
/// The octile metric over-measures a straight line by at most 8.24%: the search reaches that much
/// further than the straight bound.
const OCTILE_MAX: f32 = 1.0824;
/// A station or cell on a barrier pixel (a pier, a cell centre just off the shore) joins the
/// walkable pixels within this many pixels (500 m at 50 m: a res-9 cell's centre can lie that far
/// out when the land it holds is a strip of shore).
const SNAP_PX: i32 = 10;

/// The walking barrier: row runs of barrier pixels, row 0 south, as the pack's water mask.
pub struct WalkGrid {
    pub x0: f32,
    pub y0: f32,
    pub cell: f32,
    pub w: i32,
    pub h: i32,
    row_off: Vec<u32>,
    xs: Vec<u16>,
}

impl WalkGrid {
    /// Barrier bytes (1 barrier) over columns c0..c0+w, rows r0..r0+h; outside the grid is land.
    fn window(&self, c0: i32, r0: i32, w: i32, h: i32) -> (Vec<u8>, bool) {
        let mut bar = vec![0u8; (w * h) as usize];
        let mut any = false;
        for r in r0.max(0)..(r0 + h).min(self.h) {
            let runs = &self.xs[self.row_off[r as usize] as usize..self.row_off[r as usize + 1] as usize];
            let row = &mut bar[((r - r0) * w) as usize..((r - r0 + 1) * w) as usize];
            for p in runs.chunks_exact(2) {
                let (s, e) = ((p[0] as i32).max(c0), (p[1] as i32).min(c0 + w));
                if s < e {
                    row[(s - c0) as usize..(e - c0) as usize].fill(1);
                    any = true;
                }
            }
        }
        (bar, any)
    }

    /// Whether a pixel is a barrier (outside the grid: no).
    pub fn is_barrier(&self, x: f32, y: f32) -> bool {
        let c = ((x - self.x0) / self.cell).floor() as i32;
        let r = ((y - self.y0) / self.cell).floor() as i32;
        if c < 0 || r < 0 || c >= self.w || r >= self.h {
            return false;
        }
        let runs = &self.xs[self.row_off[r as usize] as usize..self.row_off[r as usize + 1] as usize];
        runs.partition_point(|&v| v as i32 <= c) % 2 == 1
    }

    /// The walk from (x, y) out to about `max_m` straight, around barriers: a search over the
    /// window's pixels, eight neighbours, never across a barrier's corner. None if the window
    /// holds no barrier pixel (every walk is the straight line).
    pub fn field(&self, x: f32, y: f32, max_m: f32) -> Option<Field> {
        let rad = (max_m * OCTILE_MAX / self.cell).ceil() as i32 + 3;
        let sc = ((x - self.x0) / self.cell).floor() as i32;
        let sr = ((y - self.y0) / self.cell).floor() as i32;
        let (c0, r0, w) = (sc - rad, sr - rad, 2 * rad + 1);
        let (bar, any) = self.window(c0, r0, w, w);
        if !any {
            return None;
        }
        let n = (w * w) as usize;
        let mut d = vec![INF; n];
        let limit = ((max_m * OCTILE_MAX / self.cell) * ORTH as f32) as u32 + 2 * DIAG;
        let mut buckets: Vec<Vec<u32>> = (0..RING).map(|_| Vec::new()).collect();
        let mut pending = 0usize;
        let (lc, lr) = (rad, rad);
        // the start: the station's pixel, or the nearest walkable pixels if it is a barrier (none
        // within reach: a station out on the water, no telling; the straight line)
        let start = nearest_walkable(&bar, w, lc, lr);
        if start.is_empty() {
            return None;
        }
        // Dial's bucket queue: costs are small integers, and every queued cost is within RING of
        // the one being settled (seeds span at most 2 x ORTH, a step adds at most DIAG)
        let mut cur = start.iter().map(|e| e.1).min().unwrap_or(0);
        for (i, v) in start {
            if v < d[i] {
                d[i] = v;
                buckets[(v % RING) as usize].push(i as u32);
                pending += 1;
            }
        }
        while pending > 0 && cur <= limit {
            let b = (cur % RING) as usize;
            while let Some(i) = buckets[b].pop() {
                pending -= 1;
                let i = i as usize;
                if d[i] != cur {
                    continue;
                }
                let (c, r) = ((i as i32) % w, (i as i32) / w);
                for (dc, dr) in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)] {
                    let (nc, nr) = (c + dc, r + dr);
                    if nc < 0 || nr < 0 || nc >= w || nr >= w {
                        continue;
                    }
                    let j = (nr * w + nc) as usize;
                    if bar[j] != 0 {
                        continue;
                    }
                    let step = if dc != 0 && dr != 0 {
                        // no cutting a barrier's corner
                        if bar[(r * w + nc) as usize] != 0 || bar[(nr * w + c) as usize] != 0 {
                            continue;
                        }
                        DIAG
                    } else {
                        ORTH
                    };
                    let v = cur + step;
                    if v < d[j] {
                        d[j] = v;
                        buckets[(v % RING) as usize].push(j as u32);
                        pending += 1;
                    }
                }
            }
            cur += 1;
        }
        Some(Field { x0: self.x0 + c0 as f32 * self.cell, y0: self.y0 + r0 as f32 * self.cell, cell: self.cell, w, sc: lc, sr: lr, d, bar })
    }
}

/// Pixel (c, r) itself if walkable, else the walkable pixels within SNAP_PX that are at most two
/// steps further than the nearest one (one bank, not the far bank of a narrow river), each with
/// its octile cost from (c, r).
fn nearest_walkable(bar: &[u8], w: i32, c: i32, r: i32) -> Vec<(usize, u32)> {
    let at = |c: i32, r: i32| (r * w + c) as usize;
    if bar[at(c, r)] == 0 {
        return vec![(at(c, r), 0)];
    }
    let mut v = vec![];
    for dr in -SNAP_PX..=SNAP_PX {
        for dc in -SNAP_PX..=SNAP_PX {
            let (nc, nr) = (c + dc, r + dr);
            if nc < 0 || nr < 0 || nc >= w || nr >= w || bar[at(nc, nr)] != 0 {
                continue;
            }
            let (a, b) = (dc.unsigned_abs().min(dr.unsigned_abs()), dc.unsigned_abs().max(dr.unsigned_abs()));
            v.push((at(nc, nr), DIAG * a + ORTH * (b - a)));
        }
    }
    let m = v.iter().map(|e| e.1).min().unwrap_or(0);
    v.retain(|e| e.1 <= m + 2 * ORTH);
    v
}

/// One station's walk search (`WalkGrid::field`).
pub struct Field {
    x0: f32,
    y0: f32,
    cell: f32,
    w: i32,
    sc: i32,
    sr: i32,
    d: Vec<u32>,
    bar: Vec<u8>,
}

impl Field {
    /// Walking metres to (x, y), `straight` metres away in a straight line: the straight line
    /// times how much longer the search's path is than its free path between the same two pixels
    /// (so exactly the straight line wherever nothing is in the way). None if the search did not
    /// reach it (cut off by water, or beyond its bound).
    pub fn eff(&self, x: f32, y: f32, straight: f32) -> Option<f32> {
        let c = ((x - self.x0) / self.cell).floor() as i32;
        let r = ((y - self.y0) / self.cell).floor() as i32;
        if c < 1 || r < 1 || c >= self.w - 1 || r >= self.w - 1 {
            return None;
        }
        let i = (r * self.w + c) as usize;
        let mut g = if self.bar[i] == 0 { self.d[i] } else { INF };
        if self.bar[i] != 0 {
            // a point on a barrier pixel (a pier, a cell centre just off the shore): from the
            // nearest walkable pixels; with none within reach (a cell centred in a marsh), no
            // telling: the straight line
            let near = nearest_walkable(&self.bar, self.w, c, r);
            if near.is_empty() {
                return Some(straight);
            }
            for (j, v) in near {
                if self.d[j] != INF {
                    g = g.min(self.d[j] + v);
                }
            }
        }
        if g == INF {
            return None;
        }
        let (a, b) = ((c - self.sc).unsigned_abs(), (r - self.sr).unsigned_abs());
        let free = DIAG * a.min(b) + ORTH * (a.max(b) - a.min(b));
        if free == 0 || g <= free {
            return Some(straight);
        }
        Some(straight * g as f32 / free as f32)
    }
}

/// A walk that is this much longer than the straight line is kept as a station's exception.
const EXCEPTION: f32 = 1.0005;

thread_local! {
    /// Per (pack, station position, bound): the cells whose walk to the station is not the
    /// straight line, with their walking metres (infinite: cut off). Shared by the populations of
    /// one city on this thread (its commutes and its everyday purposes).
    static STATION_WALKS: RefCell<HashMap<(u64, u32, u32, u32), Arc<Vec<(u32, f32)>>>> = RefCell::new(HashMap::new());
}

/// Forget every station's walk search (tests).
pub fn clear_station_walks() {
    STATION_WALKS.with(|c| c.borrow_mut().clear());
}

/// Read one array of a pack as raw little-endian bytes.
fn array<'a>(arrays: &[Value], bin: &'a [u8], name: &str, size: usize) -> Option<(&'a [u8], usize)> {
    let a = arrays.iter().find(|a| a["name"] == name)?;
    let off = a["offset"].as_u64()? as usize;
    let count = a["count"].as_u64()? as usize;
    Some((bin.get(off..off + count * size)?, count))
}

/// The routed walks and drives of one pack (module doc).
pub struct Routes {
    /// Identifies the pack's routes in the thread's station cache.
    pub id: u64,
    pub walk: Option<WalkGrid>,
    nreg: usize,
    gx: Vec<i32>,
    gy: Vec<i32>,
    reg: Vec<u8>,
    drive_win: i32,
    /// Dense: (zone * slots + slot) * nreg + region of the other zone.
    drive_near: Vec<u8>,
    sup: Vec<u16>,
    nsup: usize,
    drive_far: Vec<u8>,
    walk_win: i32,
    /// Dense: ((unit * slots + slot) * nreg + region of the other zone) * pmax + its part, a unit
    /// being one part of a zone (`walk_part`, T-700), `unit_off[zone] + part`.
    walk_zone: Vec<u8>,
    /// Each cell's part of its zone (empty: every cell part 0, one unit a zone).
    part: Vec<u8>,
    unit_off: Vec<u32>,
    pmax: usize,
    exp: [f32; 256],
}

impl std::fmt::Debug for Routes {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Routes({} zones, walk grid {})", self.gx.len(), self.walk.is_some())
    }
}

/// A pack's zones, as the pipeline checks them: zones, then a sum over them of their position
/// and region (the zone ids the tables are laid out by must be the same).
pub fn zones_check(z: &Zones) -> u64 {
    let mut h: u64 = 0;
    for i in 0..z.n {
        let v = (z.gx[i] as u64) * 1_000_003 + (z.gy[i] as u64) * 1_009 + z.region[i] as u64;
        h = h.wrapping_add((i as u64 + 1).wrapping_mul(v));
    }
    h % (1u64 << 52)
}

impl Routes {
    /// No walking barrier and no tables yet, for zones `z`.
    pub fn bare(z: &Zones) -> Routes {
        Routes {
            id: 0,
            walk: None,
            nreg: z.region.iter().copied().max().map_or(1, |r| r as usize + 1),
            gx: z.gx.clone(),
            gy: z.gy.clone(),
            reg: z.region.clone(),
            drive_win: 0,
            drive_near: vec![],
            sup: vec![],
            nsup: 0,
            drive_far: vec![],
            walk_win: 0,
            walk_zone: vec![],
            part: vec![],
            unit_off: vec![],
            pmax: 1,
            exp: std::array::from_fn(|q| (q as f32 / Q_SCALE).exp()),
        }
    }

    /// The pack's routes for zones `z`, or None (with the reason) if it has none or they do not
    /// fit these zones.
    pub fn from_pack(header: &str, bin: &[u8], z: &Zones) -> Result<Option<Routes>, String> {
        let v: Value = serde_json::from_str(header).map_err(|e| format!("header: {e}"))?;
        let rb = &v["routes"];
        if !rb.is_object() {
            return Ok(None);
        }
        let arrays = v["arrays"].as_array().ok_or("header has no arrays")?;
        let check = rb["zones_check"].as_u64().unwrap_or(0);
        if rb["zones"].as_u64() != Some(z.n as u64) || check != zones_check(z) {
            // made for other zones (the gravity re-solved since): not used, every walk and drive
            // straight x 1.3 until pipeline/routes.py is rerun (routes.py --check says so)
            return Ok(None);
        }
        let mut r = Routes::bare(z);
        // the walking barrier
        let wb = &rb["walk"];
        if let (Some((rows, nr)), Some((xs, nx))) = (array(arrays, bin, "walk_row", 4), array(arrays, bin, "walk_x", 2)) {
            let size = wb["size"].as_array().ok_or("routes.walk lacks size")?;
            let origin = wb["origin_m"].as_array().ok_or("routes.walk lacks origin_m")?;
            let (w, h) = (size[0].as_i64().unwrap_or(0) as i32, size[1].as_i64().unwrap_or(0) as i32);
            if nr != h as usize + 1 {
                return Err("walk_row does not have one entry per row plus one".into());
            }
            let row_off: Vec<u32> = rows.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
            if row_off[h as usize] as usize != nx {
                return Err("walk_row does not index walk_x".into());
            }
            r.walk = Some(WalkGrid {
                x0: origin[0].as_f64().unwrap_or(0.0) as f32,
                y0: origin[1].as_f64().unwrap_or(0.0) as f32,
                cell: wb["cell_m"].as_f64().ok_or("routes.walk lacks cell_m")? as f32,
                w,
                h,
                row_off,
                xs: xs.chunks_exact(2).map(|c| u16::from_le_bytes(c.try_into().unwrap())).collect(),
            });
            r.id = (nx as u64) << 32 ^ (z.n as u64) << 8 ^ check;
        }
        // zone tables: canonical windows in the pack, dense here
        if let Some((near, _)) = array(arrays, bin, "drive_near", 1) {
            let win = rb["drive"]["window"].as_i64().ok_or("routes.drive lacks window")? as i32;
            r.drive_near = r.expand(z, near, win)?;
            r.drive_win = win;
            let (sup, ns) = array(arrays, bin, "drive_super", 2).ok_or("routes lack drive_super")?;
            if ns != z.n {
                return Err("drive_super does not have one entry per zone".into());
            }
            r.sup = sup.chunks_exact(2).map(|c| u16::from_le_bytes(c.try_into().unwrap())).collect();
            r.nsup = r.sup.iter().map(|&s| s as usize + 1).max().unwrap_or(0);
            let (far, nf) = array(arrays, bin, "drive_far", 1).ok_or("routes lack drive_far")?;
            if nf != r.nsup * r.nsup {
                return Err(format!("drive_far has {nf} entries for {} supercells", r.nsup));
            }
            r.drive_far = far.to_vec();
        }
        if let Some((wz, _)) = array(arrays, bin, "walk_zone", 1) {
            let win = rb["walk_zone"]["window"].as_i64().ok_or("routes.walk_zone lacks window")? as i32;
            let part = match array(arrays, bin, "walk_part", 1) {
                Some((p, np)) if rb["walk_zone"]["parts"] == true => {
                    if np != z.cell_zone.len() {
                        return Err("walk_part does not have one entry per cell".into());
                    }
                    p.to_vec()
                }
                _ => vec![],
            };
            r.set_walk_zone(z, wz, win, part)?;
        }
        Ok(Some(r))
    }

    /// The walk factors (`walk_zone`, canonical bytes of window `win`) and each cell's part
    /// (`walk_part`; empty for a pack made before T-700: one part a zone).
    pub fn set_walk_zone(&mut self, z: &Zones, bytes: &[u8], win: i32, part: Vec<u8>) -> Result<(), String> {
        let mut np = vec![1u32; z.n];
        for (c, &p) in part.iter().enumerate() {
            let zi = z.cell_zone[c] as usize;
            np[zi] = np[zi].max(p as u32 + 1);
        }
        self.pmax = np.iter().copied().max().unwrap_or(1) as usize;
        self.unit_off = std::iter::once(0).chain(np.iter().scan(0u32, |s, &v| {
            *s += v;
            Some(*s)
        })).collect();
        self.walk_zone = self.expand_walk(z, bytes, win, &np)?;
        self.walk_win = win;
        self.part = part;
        Ok(())
    }

    /// `expand` for the walk table, whose window entries are each zone's parts (T-700): units
    /// (each zone's parts in order) outer, then the window's zones as `expand`, then their parts.
    fn expand_walk(&self, z: &Zones, bytes: &[u8], win: i32, np: &[u32]) -> Result<Vec<u8>, String> {
        let nreg = self.nreg;
        let pmax = self.pmax;
        let mut grid = vec![u32::MAX; (z.w * z.h) as usize * nreg];
        for i in 0..z.n {
            grid[((z.gx[i] * z.h + z.gy[i]) as usize) * nreg + z.region[i] as usize] = i as u32;
        }
        let slots = ((2 * win + 1) * (2 * win + 1)) as usize;
        let units = *self.unit_off.last().unwrap_or(&0) as usize;
        let mut out = vec![0u8; units * slots * nreg * pmax];
        let mut k = 0usize;
        for i in 0..z.n {
            for pi in 0..np[i] as usize {
                let u = self.unit_off[i] as usize + pi;
                for dx in -win..=win {
                    let gx = z.gx[i] + dx;
                    if gx < 0 || gx >= z.w {
                        continue;
                    }
                    for dy in -win..=win {
                        let gy = z.gy[i] + dy;
                        if gy < 0 || gy >= z.h {
                            continue;
                        }
                        for reg in 0..nreg {
                            let j = grid[((gx * z.h + gy) as usize) * nreg + reg];
                            if j == u32::MAX {
                                continue;
                            }
                            for pj in 0..np[j as usize] as usize {
                                let b = *bytes.get(k).ok_or("walk_zone is shorter than its units' windows")?;
                                out[((u * slots + Self::slot(dx, dy, win)) * nreg + reg) * pmax + pj] = b;
                                k += 1;
                            }
                        }
                    }
                }
            }
        }
        if k != bytes.len() {
            return Err(format!("walk_zone has {} bytes for {k} window entries", bytes.len()));
        }
        Ok(out)
    }

    /// The window's slot of square offset (dx, dy).
    #[inline]
    fn slot(dx: i32, dy: i32, win: i32) -> usize {
        ((dx + win) * (2 * win + 1) + dy + win) as usize
    }

    /// Canonical bytes (each zone's window, the zones that exist, in slot then region order) to
    /// the dense layout.
    fn expand(&self, z: &Zones, bytes: &[u8], win: i32) -> Result<Vec<u8>, String> {
        let nreg = self.nreg;
        let mut exists = vec![false; (z.w * z.h) as usize * nreg];
        for i in 0..z.n {
            exists[((z.gx[i] * z.h + z.gy[i]) as usize) * nreg + z.region[i] as usize] = true;
        }
        let slots = ((2 * win + 1) * (2 * win + 1)) as usize;
        let mut out = vec![0u8; z.n * slots * nreg];
        let mut k = 0usize;
        for i in 0..z.n {
            for dx in -win..=win {
                let gx = z.gx[i] + dx;
                if gx < 0 || gx >= z.w {
                    continue;
                }
                for dy in -win..=win {
                    let gy = z.gy[i] + dy;
                    if gy < 0 || gy >= z.h {
                        continue;
                    }
                    for reg in 0..nreg {
                        if exists[((gx * z.h + gy) as usize) * nreg + reg] {
                            let b = *bytes.get(k).ok_or("a routes table is shorter than its zones' windows")?;
                            out[(i * slots + Self::slot(dx, dy, win)) * nreg + reg] = b;
                            k += 1;
                        }
                    }
                }
            }
        }
        if k != bytes.len() {
            return Err(format!("a routes table has {} bytes for {k} window entries", bytes.len()));
        }
        Ok(out)
    }

    pub fn has_drive(&self) -> bool {
        !self.drive_near.is_empty()
    }

    /// Road over straight distance between zones i and j (`drive_near` within the window, the
    /// supercells' `drive_far` beyond); the game's 1.3 without tables.
    #[inline]
    pub fn drive_factor(&self, i: usize, j: usize) -> f32 {
        if self.drive_near.is_empty() {
            return 1.3;
        }
        let (dx, dy) = (self.gx[j] - self.gx[i], self.gy[j] - self.gy[i]);
        let w = self.drive_win;
        if dx.abs() <= w && dy.abs() <= w {
            let slots = ((2 * w + 1) * (2 * w + 1)) as usize;
            self.exp[self.drive_near[(i * slots + Self::slot(dx, dy, w)) * self.nreg + self.reg[j] as usize] as usize]
        } else {
            self.exp[self.drive_far[self.sup[i] as usize * self.nsup + self.sup[j] as usize] as usize]
        }
    }

    /// `drive_factor` from zone j's square less zone i's (dx, dy), j's region and supercell
    /// (`sup_of`), with no branch between the tables (T-439). Needs `has_drive`.
    #[inline]
    pub fn drive_factor_by(&self, i: usize, dx: i32, dy: i32, reg_j: u8, sup_j: u16) -> f32 {
        let w = self.drive_win;
        let near = dx.abs() <= w && dy.abs() <= w;
        let slots = ((2 * w + 1) * (2 * w + 1)) as usize;
        let a = self.drive_near[(i * slots + Self::slot(dx.clamp(-w, w), dy.clamp(-w, w), w)) * self.nreg + reg_j as usize];
        let b = self.drive_far[self.sup[i] as usize * self.nsup + sup_j as usize];
        self.exp[if near { a } else { b } as usize]
    }

    pub fn sup_of(&self, j: usize) -> u16 {
        self.sup.get(j).copied().unwrap_or(0)
    }

    /// The walk's detour factor between part `pi` of zone i and part `pj` of zone j (1 beyond the
    /// window or without tables).
    #[inline]
    pub fn walk_factor_parts(&self, i: usize, pi: u8, j: usize, pj: u8) -> f32 {
        if self.walk_zone.is_empty() {
            return 1.0;
        }
        let (dx, dy) = (self.gx[j] - self.gx[i], self.gy[j] - self.gy[i]);
        let w = self.walk_win;
        if dx.abs() > w || dy.abs() > w {
            return 1.0;
        }
        let slots = ((2 * w + 1) * (2 * w + 1)) as usize;
        let u = self.unit_off[i] as usize + pi as usize;
        self.exp[self.walk_zone[((u * slots + Self::slot(dx, dy, w)) * self.nreg + self.reg[j] as usize) * self.pmax + pj as usize] as usize]
    }

    /// The walk's detour factor between cells a and b (their zones' parts they are in).
    #[inline]
    pub fn walk_factor_cells(&self, z: &Zones, a: usize, b: usize) -> f32 {
        self.walk_factor_parts(z.cell_zone[a] as usize, self.part(a), z.cell_zone[b] as usize, self.part(b))
    }

    /// The most parts any zone has (1 for a pack made before T-700).
    pub fn parts_max(&self) -> usize {
        self.pmax
    }

    /// Cell c's part of its zone (T-700).
    #[inline]
    pub fn part(&self, c: usize) -> u8 {
        self.part.get(c).copied().unwrap_or(0)
    }

    /// The walk's detour factor between zones i and j from their first parts (where no cell is
    /// at hand: walks between zone centres).
    #[inline]
    pub fn walk_factor(&self, i: usize, j: usize) -> f32 {
        self.walk_factor_parts(i, 0, j, 0)
    }

    /// The walk window (squares), 0 without tables.
    pub fn walk_window(&self) -> i32 {
        if self.walk_zone.is_empty() { 0 } else { self.walk_win }
    }

    /// The cells whose walk to a station at (sx, sy) is not the straight line, sorted by cell,
    /// with their walking metres (infinite where water cuts them off within the bound). `visit`
    /// calls its argument with every cell (id, x, y) that might be within `cut` metres. None
    /// without a walk grid. Kept for the thread, so a station is searched once.
    pub fn station_walks(&self, sx: f32, sy: f32, cut: f32, visit: impl FnOnce(&mut dyn FnMut(u32, f32, f32))) -> Option<Arc<Vec<(u32, f32)>>> {
        let grid = self.walk.as_ref()?;
        let key = (self.id, sx.to_bits(), sy.to_bits(), cut.to_bits());
        if let Some(v) = STATION_WALKS.with(|c| c.borrow().get(&key).cloned()) {
            return Some(v);
        }
        let mut v = vec![];
        if let Some(f) = grid.field(sx, sy, cut) {
            let cut2 = cut * cut;
            visit(&mut |c, x, y| {
                let d2 = (x - sx).powi(2) + (y - sy).powi(2);
                if d2 > cut2 {
                    return;
                }
                let s = d2.sqrt();
                match f.eff(x, y, s) {
                    None => v.push((c, f32::INFINITY)),
                    Some(e) if e > s * EXCEPTION => v.push((c, e)),
                    _ => {}
                }
            });
            v.sort_unstable_by_key(|e| e.0);
        }
        let v = Arc::new(v);
        STATION_WALKS.with(|c| {
            let mut c = c.borrow_mut();
            if c.len() > 50_000 {
                c.clear();
            }
            c.insert(key, v.clone());
        });
        Some(v)
    }
}

/// The walking metres from a station to cell `c`, `straight` metres away, given the station's
/// exceptions (`Routes::station_walks`).
#[inline]
pub fn walk_m(ex: Option<&Arc<Vec<(u32, f32)>>>, c: u32, straight: f32) -> f32 {
    match ex {
        Some(v) if !v.is_empty() => match v.binary_search_by_key(&c, |e| e.0) {
            Ok(k) => v[k].1,
            Err(_) => straight,
        },
        _ => straight,
    }
}

/// Most parts a zone's cells split into by bank (T-700).
pub const WALK_PARTS: usize = 4;
/// A cell is on its part's side of the water when the walk from the part's representative is at
/// most `PART_DETOUR` times the straight line plus `PART_SLACK_M`.
const PART_DETOUR: f32 = 1.5;
const PART_SLACK_M: f32 = 500.0;

/// What `walk_tables` makes: the canonical `walk_zone` bytes, each cell's part, and counts.
pub struct WalkTables {
    pub bytes: Vec<u8>,
    pub part: Vec<u8>,
    pub units: usize,
    pub split_zones: usize,
    pub searched: usize,
    pub pairs: usize,
    pub over: usize,
    pub cut: usize,
}

/// The walk factors of `pipeline/routes.py` (run by `pack_walk`). Each zone's cells (positions
/// `x`, `y`, weights `w`) split into at most `pmax` parts, one per bank: part 0 is the cells the
/// representative (the cell nearest the weighted centre, off the barrier if any is) reaches by a
/// short walk, the next part's representative is the same for the cells left, and so on; cells
/// left after `pmax` parts join the part whose representative walks to them shortest. From each
/// part's representative the walk is searched, and its factor to each part of the window's zones
/// is the cells' walking metres over their straight metres weighted by `w` (a cell cut off counts
/// at the cap); the two directions are averaged. A factor is never a blend of two banks, which
/// put the cap's 53 on walks that cross no water (T-700).
pub fn walk_tables(grid: &WalkGrid, z: &Zones, x: &[f32], y: &[f32], w: &[f64], win: i32, pmax: usize) -> WalkTables {
    let n = z.n;
    let nreg = z.region.iter().copied().max().map_or(1, |v| v as usize + 1);
    let dist = |a: usize, b: usize| ((x[a] - x[b]).powi(2) + (y[a] - y[b]).powi(2)).sqrt();
    let mut members: Vec<Vec<u32>> = vec![vec![]; n];
    for c in 0..x.len() {
        members[z.cell_zone[c] as usize].push(c as u32);
    }
    let rep_of = |cells: &[u32]| -> usize {
        let (mut sx, mut sy, mut sw) = (0f64, 0f64, 0f64);
        for &c in cells {
            let c = c as usize;
            sx += w[c] * x[c] as f64;
            sy += w[c] * y[c] as f64;
            sw += w[c];
        }
        let (mx, my) = (sx / sw, sy / sw);
        let mut best = (true, f64::INFINITY, cells[0] as usize);
        for &c in cells {
            let c = c as usize;
            let wet = grid.is_barrier(x[c], y[c]);
            let d = ((x[c] as f64 - mx).powi(2) + (y[c] as f64 - my).powi(2)).sqrt();
            if (wet, d) < (best.0, best.1) {
                best = (wet, d, c);
            }
        }
        best.2
    };
    // the parts, searched as far as a walk across the zone and back
    let mut part = vec![0u8; x.len()];
    let mut reps: Vec<Vec<usize>> = vec![vec![]; n];
    let part_reach = z.zone_m * 2.5;
    let mut split_zones = 0;
    for zi in 0..n {
        if members[zi].is_empty() {
            continue;
        }
        let mut left = members[zi].clone();
        let mut fields = vec![];
        while !left.is_empty() && reps[zi].len() < pmax {
            let r = rep_of(&left);
            let k = reps[zi].len() as u8;
            reps[zi].push(r);
            let f = grid.field(x[r], y[r], part_reach);
            left.retain(|&c| {
                let c = c as usize;
                let s = dist(r, c);
                let near = c == r || f.as_ref().map_or(true, |f| f.eff(x[c], y[c], s).is_some_and(|e| e <= PART_DETOUR * s + PART_SLACK_M));
                if near {
                    part[c] = k;
                }
                !near
            });
            fields.push(f);
        }
        for &c in &left {
            let c = c as usize;
            let mut best = (f32::INFINITY, 0u8);
            for (k, f) in fields.iter().enumerate() {
                let s = dist(reps[zi][k], c);
                if let Some(e) = f.as_ref().and_then(|f| f.eff(x[c], y[c], s)) {
                    if e < best.0 {
                        best = (e, k as u8);
                    }
                }
            }
            part[c] = best.1;
        }
        split_zones += (reps[zi].len() > 1) as usize;
    }
    let np: Vec<usize> = (0..n).map(|zi| reps[zi].len().max(1)).collect();
    let mut unit_off = vec![0usize; n + 1];
    for zi in 0..n {
        unit_off[zi + 1] = unit_off[zi] + np[zi];
    }
    let mut cells: Vec<Vec<u32>> = vec![vec![]; unit_off[n]];
    for c in 0..x.len() {
        let zi = z.cell_zone[c] as usize;
        cells[unit_off[zi] + part[c] as usize].push(c as u32);
    }
    let mut grid_zone = vec![u32::MAX; (z.w * z.h) as usize * nreg];
    for zi in 0..n {
        grid_zone[((z.gx[zi] * z.h + z.gy[zi]) as usize) * nreg + z.region[zi] as usize] = zi as u32;
    }
    // each zone's window as units, canonical order
    let window = |zi: usize| {
        let mut v = vec![];
        for dx in -win..=win {
            let gx = z.gx[zi] + dx;
            if gx < 0 || gx >= z.w {
                continue;
            }
            for dy in -win..=win {
                let gy = z.gy[zi] + dy;
                if gy < 0 || gy >= z.h {
                    continue;
                }
                for reg in 0..nreg {
                    let j = grid_zone[((gx * z.h + gy) as usize) * nreg + reg];
                    if j != u32::MAX {
                        v.extend(unit_off[j as usize]..unit_off[j as usize + 1]);
                    }
                }
            }
        }
        v
    };
    // the search reaches the window's farthest cells: (W + 1) squares on each axis
    let reach = z.zone_m * (win as f32 + 1.0) * std::f32::consts::SQRT_2;
    let fmax = (255.0 / Q_SCALE).exp();
    let mut f: Vec<Vec<f32>> = Vec::with_capacity(unit_off[n]);
    let mut wins: Vec<Vec<usize>> = Vec::with_capacity(unit_off[n]);
    let mut searched = 0;
    for zi in 0..n {
        let js = window(zi);
        for pi in 0..np[zi] {
            let mut row = vec![1.0f32; js.len()];
            if let Some(&r) = reps[zi].get(pi) {
                if let Some(fl) = grid.field(x[r], y[r], reach) {
                    searched += 1;
                    for (k, &u) in js.iter().enumerate() {
                        let (mut num, mut den) = (0f64, 0f64);
                        for &c in &cells[u] {
                            let c = c as usize;
                            let s = dist(r, c);
                            if s < 1.0 {
                                continue;
                            }
                            let e = fl.eff(x[c], y[c], s).unwrap_or(s * fmax);
                            num += w[c] * e as f64;
                            den += w[c] * s as f64;
                        }
                        if den > 0.0 {
                            row[k] = (num / den) as f32;
                        }
                    }
                }
            }
            f.push(row);
            wins.push(js.clone());
        }
    }
    // both directions averaged
    let mut bytes = Vec::new();
    let (mut over, mut cut, mut pairs) = (0, 0, 0);
    for u in 0..unit_off[n] {
        for (k, &v) in wins[u].iter().enumerate() {
            let back = wins[v].iter().position(|&x| x == u).map_or(f[u][k], |b| f[v][b]);
            let m = 0.5 * (f[u][k] + back);
            pairs += 1;
            over += (m > 1.2) as usize;
            cut += (m > 5.0) as usize;
            bytes.push((Q_SCALE * m.max(1.0).ln()).round().clamp(0.0, 255.0) as u8);
        }
    }
    WalkTables { bytes, part, units: unit_off[n], split_zones, searched, pairs, over, cut }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A grid of `w` x `h` pixels of 50 m from a barrier test, origin (0, 0).
    fn grid(w: i32, h: i32, bar: impl Fn(i32, i32) -> bool) -> WalkGrid {
        let (mut row_off, mut xs) = (vec![0u32], vec![]);
        for r in 0..h {
            let mut c = 0;
            while c < w {
                if bar(c, r) {
                    let s = c;
                    while c < w && bar(c, r) {
                        c += 1;
                    }
                    xs.extend([s as u16, c as u16]);
                } else {
                    c += 1;
                }
            }
            row_off.push(xs.len() as u32);
        }
        WalkGrid { x0: 0.0, y0: 0.0, cell: 50.0, w, h, row_off, xs }
    }

    fn at(px: f32) -> f32 {
        px * 50.0 + 25.0
    }

    #[test]
    fn walks_go_round_water_and_over_bridges() {
        // a river 3 pixels wide down columns 100-102, a bridge across it at row 70
        let g = grid(200, 200, |c, r| (100..=102).contains(&c) && r != 70);
        let f = g.field(at(90.0), at(50.0), 4000.0).expect("water within reach");
        // the same bank: the straight line
        let s = 50.0 * ((5.0f32).powi(2) + 20.0f32.powi(2)).sqrt();
        assert_eq!(f.eff(at(95.0), at(70.0), s), Some(s));
        // the far bank, 1 km across: up to the bridge 20 rows away and back down
        let e = f.eff(at(110.0), at(50.0), 1000.0).unwrap();
        assert!(e > 2000.0 && e < 2600.0, "{e}");
        // no bridge: cut off within the bound
        let g2 = grid(200, 200, |c, _| (100..=102).contains(&c));
        let f2 = g2.field(at(90.0), at(50.0), 4000.0).unwrap();
        assert_eq!(f2.eff(at(110.0), at(50.0), 1000.0), None);
        // nothing in the way: no search at all
        assert!(grid(200, 200, |_, _| false).field(at(90.0), at(50.0), 4000.0).is_none());
        // a station on a pier (a barrier pixel) starts from the nearest bank
        let f3 = g.field(at(101.0), at(30.0), 4000.0).unwrap();
        let near = f3.eff(at(95.0), at(30.0), 300.0).unwrap();
        let far = f3.eff(at(107.0), at(30.0), 300.0).unwrap();
        assert!(near < 400.0 || far < 400.0, "one bank is near: {near} {far}");
    }

    #[test]
    fn a_diagonal_line_of_water_blocks() {
        // water along the diagonal c == r: no corner cutting across it
        let g = grid(120, 120, |c, r| c == r);
        let f = g.field(at(40.0), at(60.0), 3000.0).unwrap();
        assert_eq!(f.eff(at(60.0), at(40.0), 1414.0), None);
    }

    /// T-700: a zone astride a river nobody can walk across. Walks on one bank, in or out of it,
    /// go straight; walks across are cut off. Before, the zone's one factor blended both banks
    /// (Shanghai's Bund: x48 on a 300 m walk); `pmax` 1 is that blend.
    #[test]
    fn a_zone_astride_a_river_keeps_each_bank_apart() {
        // 10 km square, the river down columns 100-102 (x 5,000-5,150 m), no bridge; 2 km zones,
        // so the zones of column 2 (x 4,000-6,000 m) hold both banks
        let g = grid(200, 200, |c, _| (100..=102).contains(&c));
        let (zw, zh) = (5, 5);
        let (mut x, mut y, mut w, mut cz) = (vec![], vec![], vec![], vec![]);
        for i in 0..40 {
            for j in 0..40 {
                let (px, py) = (125.0 + 250.0 * i as f32, 125.0 + 250.0 * j as f32);
                if g.is_barrier(px, py) {
                    continue;
                }
                x.push(px);
                y.push(py);
                // more people on the east bank: the astride zones' representatives are there
                w.push(if px > 5000.0 { 3.0 } else { 1.0 });
                cz.push(((px / 2000.0) as i32 * zh + (py / 2000.0) as i32) as u32);
            }
        }
        let n = (zw * zh) as usize;
        let z = Zones {
            zone_m: 2000.0,
            n,
            cell_zone: cz,
            gx: (0..n as i32).map(|i| i / zh).collect(),
            gy: (0..n as i32).map(|i| i % zh).collect(),
            region: vec![0; n],
            w: zw,
            h: zh,
            x0: 0.0,
            y0: 0.0,
            pop: vec![1.0; n],
            jobs: vec![1.0; n],
            ncell: vec![1; n],
        };
        let cell = |px: f32, py: f32| x.iter().zip(&y).position(|(&a, &b)| a == px && b == py).unwrap();
        // west bank: 4,625 in the astride zone, 4,375 too, 3,875 in the zone west of it
        let (a, b, c) = (cell(4625.0, 5125.0), cell(4375.0, 5125.0), cell(3875.0, 5125.0));
        // east bank, in the same astride zone
        let e = cell(5375.0, 5125.0);
        let routes = |pmax: usize| {
            let t = walk_tables(&g, &z, &x, &y, &w, 2, pmax);
            let mut r = Routes::bare(&z);
            r.set_walk_zone(&z, &t.bytes, 2, if pmax > 1 { t.part } else { vec![] }).unwrap();
            r
        };
        let r = routes(WALK_PARTS);
        for (p, q) in [(a, b), (a, c), (c, a), (b, c)] {
            let f = r.walk_factor_cells(&z, p, q);
            assert!(f < 1.1, "one bank, cells {p} and {q}: x{f}");
        }
        assert!(r.walk_factor_cells(&z, a, e) > 5.0, "across the river: x{}", r.walk_factor_cells(&z, a, e));
        assert!(r.walk_factor_cells(&z, e, e) < 1.1);
        // today's blend: one bank's walks counted many times over
        let old = routes(1);
        assert!(old.walk_factor_cells(&z, a, c) > 5.0, "the blend: x{}", old.walk_factor_cells(&z, a, c));
    }
}
