//! T-275: routed walking and driving (docs/demand.md 4.3, "Routed walking and driving").
//!
//! What `pipeline/routes.py` adds to a pack, read here:
//! - the walking barrier raster (`walk_row`, `walk_x`: water, less OSM bridges a person may walk
//!   on), on which the walk to each station is searched when a network is set (stations are the
//!   player's, so this cannot be done offline). A station's search is kept for the thread
//!   (`station_walks`), so an edit searches only the stations it moved or added, and the
//!   commute and everyday populations of one city share it.
//! - per zone, the walk's detour factor to the zones within 2 squares (`walk_zone`), for the
//!   walk share between zones, and the drive's routed-over-straight factor to the zones within 6
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
    walk_zone: Vec<u8>,
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
        let nreg = z.region.iter().copied().max().map_or(1, |r| r as usize + 1);
        let mut r = Routes {
            id: 0,
            walk: None,
            nreg,
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
            exp: std::array::from_fn(|q| (q as f32 / Q_SCALE).exp()),
        };
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
            r.walk_zone = r.expand(z, wz, win)?;
            r.walk_win = win;
        }
        Ok(Some(r))
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

    /// The walk's detour factor between zones i and j (1 beyond the window or without tables).
    #[inline]
    pub fn walk_factor(&self, i: usize, j: usize) -> f32 {
        if self.walk_zone.is_empty() {
            return 1.0;
        }
        let (dx, dy) = (self.gx[j] - self.gx[i], self.gy[j] - self.gy[i]);
        let w = self.walk_win;
        if dx.abs() > w || dy.abs() > w {
            return 1.0;
        }
        let slots = ((2 * w + 1) * (2 * w + 1)) as usize;
        self.exp[self.walk_zone[(i * slots + Self::slot(dx, dy, w)) * self.nreg + self.reg[j] as usize] as usize]
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
}
