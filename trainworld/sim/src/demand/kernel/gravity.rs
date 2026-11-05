use super::*;

// ---------------------------------------------------------------------------------------------
// 1. Zones and gravity

#[derive(Clone)]
pub struct Zones {
    pub zone_m: f32,
    pub n: usize,
    pub cell_zone: Vec<u32>,
    pub gx: Vec<i32>,
    pub gy: Vec<i32>,
    /// First-level region of each zone (T-162). A square cut by a region border is one zone per
    /// region, all at the square's position: two zones can share `gx`, `gy`.
    pub region: Vec<u8>,
    /// Grid size in zones, and its lower-left corner in metres.
    pub w: i32,
    pub h: i32,
    pub x0: f32,
    pub y0: f32,
    pub pop: Vec<f64>,
    pub jobs: Vec<f64>,
    /// Resolution 9 cells in each zone (its land, at `CELL_KM2` a cell; `res9_cells`).
    pub ncell: Vec<u32>,
    /// T-852: each zone's side in squares, 1 or 2 (a 2x2 block, `gx`, `gy` its lower-left
    /// square, in one region); empty: every zone one square.
    pub size: Vec<u8>,
}

/// Mean area of an H3 resolution 9 cell, km².
pub const CELL_KM2: f32 = 0.1053;

impl Zones {
    /// Whether the pack has blocks (T-852).
    #[inline]
    pub fn mixed(&self) -> bool {
        !self.size.is_empty()
    }
    /// Whether zone i is a 2x2 block.
    #[inline]
    pub fn big(&self, i: usize) -> bool {
        self.size.get(i).is_some_and(|&s| s == 2)
    }
    /// Zone i's side in squares.
    #[inline]
    pub fn side(&self, i: usize) -> i32 {
        1 + self.big(i) as i32
    }
    /// Zone i's centre in half squares from the grid's corner.
    #[inline]
    pub fn hx(&self, i: usize) -> i32 {
        2 * self.gx[i] + self.side(i)
    }
    #[inline]
    pub fn hy(&self, i: usize) -> i32 {
        2 * self.gy[i] + self.side(i)
    }
    /// Squared metres between the centres of zones i and j (for one-square zones exactly the
    /// squares' offset squared x `zone_m`²).
    #[inline]
    pub fn d2_m(&self, i: usize, j: usize) -> f32 {
        let (fx, fy) = ((self.hx(j) - self.hx(i)) as f32 * 0.5, (self.hy(j) - self.hy(i)) as f32 * 0.5);
        (fx * fx + fy * fy) * self.zone_m * self.zone_m
    }
    /// Zone i's whole square in km², the cap on its land.
    #[inline]
    pub fn square_km2(&self, i: usize) -> f32 {
        (self.side(i) as f32 * self.zone_m / 1000.0).powi(2)
    }
    /// Every zone with a square within `r` squares (each way) of one of zone i's, once, in the
    /// order of their lower-left squares (`by_square`'s index `sq`).
    #[inline]
    pub fn near(&self, sq: &(Vec<u32>, Vec<u32>), i: usize, r: i32, mut f: impl FnMut(usize)) {
        let (off, at) = sq;
        let (m, e) = (self.mixed() as i32, self.side(i) - 1);
        for gx in (self.gx[i] - r - m).max(0)..=(self.gx[i] + e + r).min(self.w - 1) {
            for gy in (self.gy[i] - r - m).max(0)..=(self.gy[i] + e + r).min(self.h - 1) {
                let s = (gx * self.h + gy) as usize;
                for &j in &at[off[s] as usize..off[s + 1] as usize] {
                    let j = j as usize;
                    if m == 1 {
                        let ej = self.side(j) - 1;
                        let gap = |a: i32, b: i32| (b - a - e).max(a - b - ej).max(0);
                        if gap(self.gx[i], gx) > r || gap(self.gy[i], gy) > r {
                            continue;
                        }
                    }
                    f(j);
                }
            }
        }
    }
    /// The zone holding square (gx, gy) (`by_square`'s index `sq`): the first one at that square,
    /// else a block covering it.
    pub fn zone_at(&self, sq: &(Vec<u32>, Vec<u32>), gx: i32, gy: i32) -> Option<usize> {
        let (off, at) = sq;
        let first = |gx: i32, gy: i32, block: bool| {
            if gx < 0 || gy < 0 || gx >= self.w || gy >= self.h {
                return None;
            }
            let s = (gx * self.h + gy) as usize;
            at[off[s] as usize..off[s + 1] as usize].iter().map(|&j| j as usize).find(|&j| !block || self.big(j))
        };
        first(gx, gy, false).or_else(|| if self.mixed() { first(gx - 1, gy, true).or_else(|| first(gx, gy - 1, true)).or_else(|| first(gx - 1, gy - 1, true)) } else { None })
    }
    /// Homes and jobs per km² of land, per zone (the drive's local street speed).
    pub fn density(&self) -> Vec<f32> {
        (0..self.n).map(|i| ((self.pop[i] + self.jobs[i]) / (self.ncell[i].max(1) as f64 * CELL_KM2 as f64)) as f32).collect()
    }
    /// Per cell, its zone's residents (`pop`, people per cell) per km² of land, measured as
    /// `Params::park_parts` measures jobs (`Params::home_dens`, T-854).
    pub fn home_density(&self, pop: &[f32]) -> Vec<f32> {
        let mut p = vec![0f64; self.n];
        for (c, &zi) in self.cell_zone.iter().enumerate() {
            p[zi as usize] += pop.get(c).copied().unwrap_or(0.0) as f64;
        }
        let d: Vec<f32> = (0..self.n).map(|i| (p[i] / (self.ncell[i].max(1) as f32 * CELL_KM2).min(self.square_km2(i)) as f64) as f32).collect();
        self.cell_zone.iter().map(|&zi| d[zi as usize]).collect()
    }
    /// The zones of each grid square, CSR by square index gx * h + gy (usually one; two or more
    /// where a region border cuts the square); a block under its lower-left square only
    /// (`zone_at` finds it from the others).
    pub fn by_square(&self) -> (Vec<u32>, Vec<u32>) {
        let ns = (self.w * self.h) as usize;
        let mut off = vec![0u32; ns + 1];
        for i in 0..self.n {
            off[(self.gx[i] * self.h + self.gy[i]) as usize + 1] += 1;
        }
        for s in 0..ns {
            off[s + 1] += off[s];
        }
        let mut fill = off.clone();
        let mut items = vec![0u32; self.n];
        for i in 0..self.n {
            let s = (self.gx[i] * self.h + self.gy[i]) as usize;
            items[fill[s] as usize] = i as u32;
            fill[s] += 1;
        }
        (off, items)
    }
}

/// Congestion charges at a destination (T-372): the charging area (lon, lat ring, a rough outline
/// at 2 km zones' scale) and its charge in perceived minutes a one-way trip, half the day's charge
/// at 3 minutes a US dollar.
pub const CORDONS: &[(&str, f32, &[(f32, f32)])] = &[
    // Manhattan south of 60th St, $9 at peak (2025)
    ("Manhattan", 13.5, &[(-74.020, 40.700), (-74.016, 40.715), (-74.012, 40.730), (-74.009, 40.745), (-74.002, 40.760), (-73.994, 40.774), (-73.959, 40.759), (-73.968, 40.748), (-73.972, 40.738), (-73.974, 40.726), (-73.978, 40.711), (-73.999, 40.707), (-74.012, 40.701)]),
    // the London congestion charge, £18 from 2026 at $1.30
    ("London", 35.1, &[(-0.159, 51.513), (-0.163, 51.520), (-0.126, 51.528), (-0.105, 51.530), (-0.088, 51.526), (-0.077, 51.524), (-0.072, 51.515), (-0.075, 51.5055), (-0.100, 51.495), (-0.125, 51.487), (-0.144, 51.495), (-0.152, 51.503)]),
];

pub(super) fn in_ring(ring: &[(f32, f32)], x: f32, y: f32) -> bool {
    let mut inside = false;
    let mut j = ring.len() - 1;
    for i in 0..ring.len() {
        let ((xi, yi), (xj, yj)) = (ring[i], ring[j]);
        if (yi > y) != (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi {
            inside = !inside;
        }
        j = i;
    }
    inside
}

/// Each zone's cordon charge (`CORDONS`), as the share of its jobs inside the charging area times
/// the charge; None where no charging area holds any jobs. `origin` is the pack's (lon, lat), its
/// cells' metres equirectangular about it.
pub fn cordon_by_zone(city: &City, z: &Zones, origin: (f64, f64)) -> Option<Vec<f32>> {
    const R_EARTH: f64 = 6371008.8;
    let (lon0, lat0) = origin;
    let kx = 180.0 / std::f64::consts::PI / R_EARTH / lat0.to_radians().cos();
    let ky = 180.0 / std::f64::consts::PI / R_EARTH;
    let mut charged = vec![0f64; z.n];
    let mut any = false;
    for c in 0..city.len() {
        if city.jobs[c] <= 0.0 {
            continue;
        }
        let (lon, lat) = ((lon0 + city.x[c] as f64 * kx) as f32, (lat0 + city.y[c] as f64 * ky) as f32);
        if let Some(&(_, min, _)) = CORDONS.iter().find(|(_, _, ring)| in_ring(ring, lon, lat)) {
            charged[z.cell_zone[c] as usize] += (city.jobs[c] * min) as f64;
            any = true;
        }
    }
    any.then(|| (0..z.n).map(|j| if z.jobs[j] > 0.0 { (charged[j] / z.jobs[j]) as f32 } else { 0.0 }).collect())
}

/// Zones are the cells' 2 km squares, one per region in a square a region border cuts (T-162).
/// Zone ids in order of each zone's first cell, so a pack with one region gets the zones it had
/// before regions.
pub fn build_zones(city: &City, zone_m: f32) -> Zones {
    if city.len() == 0 {
        // the extent folds below would be infinite and w * h overflow (T-449)
        return Zones { zone_m, n: 0, cell_zone: vec![], gx: vec![], gy: vec![], region: vec![], w: 0, h: 0, x0: 0., y0: 0., pop: vec![], jobs: vec![], ncell: vec![], size: vec![] };
    }
    let minx = city.x.iter().cloned().fold(f32::INFINITY, f32::min);
    let miny = city.y.iter().cloned().fold(f32::INFINITY, f32::min);
    let maxx = city.x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let maxy = city.y.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let w = ((maxx - minx) / zone_m) as i32 + 1;
    let h = ((maxy - miny) / zone_m) as i32 + 1;
    let nreg = city.region.iter().copied().max().map_or(1, |r| r as usize + 1);
    let mut grid = vec![NONE; (w * h) as usize * nreg];
    let mut z = Zones { zone_m, n: 0, cell_zone: vec![0; city.len()], gx: vec![], gy: vec![], region: vec![], w, h, x0: minx, y0: miny, pop: vec![], jobs: vec![], ncell: vec![], size: vec![] };
    for c in 0..city.len() {
        let gx = ((city.x[c] - minx) / zone_m) as i32;
        let gy = ((city.y[c] - miny) / zone_m) as i32;
        let r = city.region.get(c).copied().unwrap_or(0);
        let g = (gx * h + gy) as usize * nreg + r as usize;
        if grid[g] == NONE {
            grid[g] = z.n as u32;
            z.n += 1;
            z.gx.push(gx);
            z.gy.push(gy);
            z.region.push(r);
            z.pop.push(0.0);
            z.jobs.push(0.0);
            z.ncell.push(0);
        }
        let zi = grid[g] as usize;
        z.cell_zone[c] = zi as u32;
        z.ncell[zi] += res9_cells(city, c);
        z.pop[zi] += city.pop[c] as f64;
        z.jobs[zi] += city.jobs[c] as f64;
    }
    z
}

/// T-852: one-square zones `z` with each 2x2 block of squares (aligned to the grid, even `gx`
/// and `gy` at its lower-left) as one zone where its zones of one region are at least two and
/// each makes under `under` trip ends a weekday (`ends`, per zone). Zone ids in order of each
/// zone's first cell, as `build_zones`; the grid grows to an even size so every block is in it.
pub fn coarsen(z: &Zones, ends: &[f64], under: f64) -> Zones {
    assert!(!z.mixed() && ends.len() == z.n);
    let key = |i: usize| ((z.gx[i] / 2) as u64) << 40 | ((z.gy[i] / 2) as u64) << 8 | z.region[i] as u64;
    let mut blocks: HashMap<u64, (u32, bool)> = HashMap::new();
    for i in 0..z.n {
        let e = blocks.entry(key(i)).or_insert((0, true));
        e.0 += 1;
        e.1 &= ends[i] < under;
    }
    let coarse = |i: usize| blocks[&key(i)].0 >= 2 && blocks[&key(i)].1;
    if !(0..z.n).any(coarse) {
        return z.clone();
    }
    let mut out = Zones { zone_m: z.zone_m, n: 0, cell_zone: vec![0; z.cell_zone.len()], gx: vec![], gy: vec![], region: vec![], w: z.w + z.w % 2, h: z.h + z.h % 2, x0: z.x0, y0: z.y0, pop: vec![], jobs: vec![], ncell: vec![], size: vec![] };
    let mut new_of = vec![NONE; z.n];
    let mut block_id: HashMap<u64, u32> = HashMap::new();
    for c in 0..z.cell_zone.len() {
        let i = z.cell_zone[c] as usize;
        if new_of[i] == NONE {
            let big = coarse(i);
            let id = if big { block_id.get(&key(i)).copied() } else { None };
            new_of[i] = id.unwrap_or_else(|| {
                out.gx.push(if big { z.gx[i] & !1 } else { z.gx[i] });
                out.gy.push(if big { z.gy[i] & !1 } else { z.gy[i] });
                out.region.push(z.region[i]);
                out.size.push(1 + big as u8);
                out.n += 1;
                if big {
                    block_id.insert(key(i), out.n as u32 - 1);
                }
                out.n as u32 - 1
            });
        }
        out.cell_zone[c] = new_of[i];
    }
    out.pop = vec![0.0; out.n];
    out.jobs = vec![0.0; out.n];
    out.ncell = vec![0; out.n];
    for i in 0..z.n {
        if let Some(&k) = new_of.get(i).filter(|&&k| k != NONE) {
            out.pop[k as usize] += z.pop[i];
            out.jobs[k as usize] += z.jobs[i];
            out.ncell[k as usize] += z.ncell[i];
        }
    }
    out
}

#[derive(Clone)]
pub struct Gravity {
    /// T_IJ = row[I] * col[J] * f(I, J) * k(I, J), with row = A_I O_I and col = B_J D_J, and k
    /// = `border` when zones I and J are in different regions, else 1 (`Gravity::k`, T-162).
    pub row: Vec<f64>,
    pub col: Vec<f64>,
    /// The decay's factor across a region border.
    pub border: f32,
    /// f by grid offset: index (dx + offx) * th + (dy + offy).
    pub ftab: Vec<f32>,
    /// The squares' side in km, for the zone-centre distance of a table index (`dist_km`), and
    /// whether `set_walk` has set the walk's cost (`walk_c`) and (T-275) the routes whose walk
    /// factors lengthen the walk between nearby zones around water (`walk_cost_at`).
    pub zk: f32,
    pub walk_set: bool,
    pub wroutes: Option<std::sync::Arc<super::super::routes::Routes>>,
    pub wcost: WalkCost,
    /// Walks between cells (T-587, `set_walk_fine`), per home zone CSR over the zones within
    /// `WALK_FINE_R` squares, ascending: zone, the walk's cost for the choice with rail, and the
    /// share walking with no rail. Empty: zone centres only.
    pub wf_off: Vec<u32>,
    pub wf_j: Vec<u32>,
    pub wf_c: Vec<f32>,
    pub wf_s: Vec<f32>,
    pub th: i32,
    pub offx: i32,
    pub offy: i32,
    /// A block's key shift (T-852, `table_layout`); 0 without blocks.
    pub kb: i32,
    pub iters: usize,
    pub max_err: f64,
    pub mean_km: f64,
    pub total: f64,
    /// T-1000, from `gravity`: the band `col / jobs` was held in, the column sums over jobs of the
    /// columns inside it, and the share of jobs in columns at its edges.
    pub pull: Option<[f64; 4]>,
}

impl Gravity {
    /// Index offset to add to `key(J)` for origin zone I.
    #[inline]
    pub fn base(&self, z: &Zones, i: usize) -> i32 {
        (self.offx - z.gx[i]) * self.th + self.offy - z.gy[i] - 2 * self.kb * z.big(i) as i32
    }
    #[inline]
    pub fn key(&self, z: &Zones, j: usize) -> i32 {
        z.gx[j] * self.th + z.gy[j] + self.kb * z.big(j) as i32
    }
    /// The region factor between zones i and j: 1 in one region, `border` across.
    #[inline]
    pub fn k(&self, z: &Zones, i: usize, j: usize) -> f32 {
        if z.region[i] == z.region[j] { 1.0 } else { self.border }
    }
    pub fn t(&self, z: &Zones, i: usize, j: usize) -> f64 {
        self.row[i] * self.col[j] * (self.ftab[(self.base(z, i) + self.key(z, j)) as usize] * self.k(z, i, j)) as f64
    }

    /// The zone-centre distance (km) of table index `k`, as `decay_table` and `decay_tables` give
    /// it (within a zone 0.52 of its side).
    #[inline]
    pub fn dist_km(&self, k: usize) -> f32 {
        let (r, y) = (k as i32 / self.th - self.offx, k as i32 % self.th);
        let h = self.kb / 2;
        let (dx, dy, own) = if self.kb == 0 || y < 2 * h - 1 {
            (r as f32, (y - self.offy) as f32, 1.0)
        } else if y < 4 * h - 1 {
            (r as f32 + 0.5, (y - 3 * h + 1) as f32 + 0.5, 1.0)
        } else {
            ((r + 1) as f32, (y - 5 * h + 2) as f32, 2.0)
        };
        if dx == 0.0 && dy == 0.0 { 0.52 * self.zk * own } else { self.zk * (dx * dx + dy * dy).sqrt() }
    }
    /// T-2365: table index `k`'s pair as `dist_km` reads it: the centres' offset in half squares,
    /// and the two zones' sides in squares (2 for a block).
    pub fn half_offset(&self, k: usize) -> (i32, i32, i32, i32) {
        let (r, y) = (k as i32 / self.th - self.offx, k as i32 % self.th);
        let h = self.kb / 2;
        if self.kb == 0 || y < 2 * h - 1 {
            (2 * r, 2 * (y - self.offy), 1, 1)
        } else if y < 4 * h - 1 {
            (2 * r + 1, 2 * (y - 3 * h + 1) + 1, 1, 2)
        } else {
            (2 * (r + 1), 2 * (y - 5 * h + 2), 2, 2)
        }
    }
    /// The cost of walking the whole way between zone centres at table index `k`
    /// (`Params::walk_trip_cost`; `NO_WALK` before `set_walk`).
    #[inline]
    pub fn walk_c(&self, k: usize) -> f32 {
        if self.walk_set { self.wcost.at(self.dist_km(k)) } else { NO_WALK }
    }

    /// Set the walk's cost between zone centres (`walk_c`); the walks between cells go
    /// (`set_walk_fine` after).
    pub fn set_walk(&mut self, z: &Zones, p: &Params) {
        self.wcost = p.walk_cost();
        self.zk = z.zone_m / 1000.0;
        self.walk_set = true;
        self.wroutes = p.routes.clone().filter(|r| r.walk_window() > 0);
        self.wf_off.clear();
        self.wf_j.clear();
        self.wf_c.clear();
        self.wf_s.clear();
    }

    /// `wf_*` (T-587): each near zone pair's commuters as the pairs of its zones' points
    /// (`walk_points`), each walking `cell_walk_m` times the zones' walk factor round water. The
    /// pair's share walking with no rail is its cell pairs' mean (`Params::walk_no_rail`); its cost for the choice with rail is the one walk whose share
    /// against a driver parking free is that mean.
    pub fn set_walk_fine(&mut self, city: &City, z: &Zones, p: &Params, nr: &NoRail) {
        let ns = p.walk_samples;
        if ns == 0 || !self.walk_set || city.len() != z.cell_zone.len() {
            return;
        }
        let (home, work) = walk_points(city, z, ns);
        let sq = z.by_square();
        // T-2214: only the cell pairs on one land part walk
        let pp = p.parts.as_deref();
        let mut row: Vec<(u32, f32, f32)> = vec![];
        let mut c = vec![0f32; ns * ns];
        for i in 0..z.n {
            self.wf_off.push(self.wf_j.len() as u32);
            let hi = &home[i * ns..(i + 1) * ns];
            if hi[0] == NONE {
                continue;
            }
            row.clear();
            z.near(&sq, i, WALK_FINE_R, |ju| {
                let wj = &work[ju * ns..(ju + 1) * ns];
                if wj[0] == NONE {
                    return;
                }
                let k = (self.base(z, i) + self.key(z, ju)) as usize;
                let mut cnt = 0usize;
                for (n, &a) in hi.iter().enumerate() {
                    for (m, &b) in wj.iter().enumerate() {
                        if pp.is_some_and(|pp| pp.cell[a as usize] != pp.cell[b as usize]) {
                            c[n * ns + m] = f32::NAN;
                            continue;
                        }
                        cnt += 1;
                        let f = self.walk_factor_cells(z, a as usize, b as usize) * 0.001;
                        c[n * ns + m] = self.wcost.at(cell_walk_m(city, a, b) * f);
                    }
                }
                let w2 = if cnt > 0 { 1.0 / cnt as f32 } else { 0.0 };
                let (road, paid, free) = nr.road(self, p, i, ju, k);
                let bt = p.mode_beta(road);
                let (mut m0, mut sw) = (0f32, 0f32);
                for &x in &c {
                    if x.is_nan() {
                        continue;
                    }
                    sw += p.walk_no_rail(road, paid, free, x);
                    if x < NO_WALK {
                        m0 += 1.0 / (1.0 + fast_exp(-(bt * (road - x) + p.asc_walk())));
                    }
                }
                let cw = if m0 > 0.0 {
                    let m0 = (m0 * w2).clamp(1e-7, 1.0 - 1e-6);
                    road - ((m0 / (1.0 - m0)).ln() - p.asc_walk()) / bt
                } else {
                    NO_WALK
                };
                row.push((ju as u32, cw, sw * w2));
            });
            row.sort_unstable_by_key(|x| x.0);
            self.wf_j.extend(row.iter().map(|x| x.0));
            self.wf_c.extend(row.iter().map(|x| x.1));
            self.wf_s.extend(row.iter().map(|x| x.2));
        }
        self.wf_off.push(self.wf_j.len() as u32);
        // grown a zone's row at a time, so up to twice their length until shrunk (T-1611)
        self.wf_off.shrink_to_fit();
        self.wf_j.shrink_to_fit();
        self.wf_c.shrink_to_fit();
        self.wf_s.shrink_to_fit();
    }

    /// The entry of `wf_*` for zones i and j at table index `k`, if they are a near pair.
    #[inline]
    pub fn fine_at(&self, i: usize, j: usize, k: usize) -> Option<usize> {
        if self.wf_off.is_empty() {
            return None;
        }
        let (dx, dy) = self.square_offset(k);
        // a block's pairs reach a square further (`Zones::near`)
        let r = WALK_FINE_R + (self.kb != 0) as i32;
        if dx.abs() > r || dy.abs() > r {
            return None;
        }
        let a = self.wf_off[i] as usize;
        self.wf_j[a..self.wf_off[i + 1] as usize].binary_search(&(j as u32)).ok().map(|n| a + n)
    }

    /// The offset in squares of table index `k`'s pair (with blocks, T-852, within a square of
    /// their centres' offset).
    #[inline]
    pub(super) fn square_offset(&self, k: usize) -> (i32, i32) {
        let (r, y) = (k as i32 / self.th - self.offx, k as i32 % self.th);
        let h = self.kb / 2;
        if self.kb == 0 || y < 2 * h - 1 {
            (r, y - self.offy)
        } else if y < 4 * h - 1 {
            (r, y - 3 * h + 1)
        } else {
            (r + 1, y - 5 * h + 2)
        }
    }

    /// The walk's factor round water between zones i and j (T-275), at least 1.
    #[inline]
    pub fn walk_factor(&self, i: usize, j: usize) -> f32 {
        self.wroutes.as_ref().map_or(1.0, |r| r.walk_factor(i, j)).max(1.0)
    }

    /// The same between cells a and b, from the parts of their zones they are in (T-700).
    #[inline]
    pub fn walk_factor_cells(&self, z: &Zones, a: usize, b: usize) -> f32 {
        self.wroutes.as_ref().map_or(1.0, |r| r.walk_factor_cells(z, a, b)).max(1.0)
    }

    /// The cost of walking the whole way between zones i and j at table index `k` (`base(i) +
    /// key(j)`): between cells for near pairs (`wf_*`, T-587), else `walk_c`, or with the pack's
    /// routes (T-275) the walk around water between them, the zone-centre distance times their
    /// walk factor.
    #[inline]
    pub fn walk_cost_at(&self, i: usize, j: usize, k: usize) -> f32 {
        if let Some(n) = self.fine_at(i, j, k) {
            return self.wf_c[n];
        }
        if let Some(r) = &self.wroutes {
            let f = r.walk_factor(i, j);
            if f > 1.0 {
                return self.wcost.at(self.dist_km(k) * f);
            }
        }
        self.walk_c(k)
    }

    /// The same for any zone pair (`NO_WALK` before `set_walk`).
    pub fn walk_cost_pair(&self, z: &Zones, i: usize, j: usize) -> f32 {
        if !self.walk_set {
            return NO_WALK;
        }
        self.walk_cost_at(i, j, (self.base(z, i) + self.key(z, j)) as usize)
    }

    /// T-558: all-day trips with no rail at all that walk, by home zone and by work zone: over the
    /// zone pairs within `WALK_REACH_KM`, T_IJ times `NoRail::split`.
    pub fn no_rail_by_zone(&self, z: &Zones, p: &Params, nr: &NoRail) -> [Vec<f64>; 2] {
        let (mut wh, mut ww) = (vec![0.0f64; z.n], vec![0.0f64; z.n]);
        if !self.walk_set {
            return [wh, ww];
        }
        // the walk factor round water is at most this (`routes`' byte tables), so past this many
        // squares no walk is offered
        let r = (WALK_REACH_KM * 1000.0 / z.zone_m).ceil() as i32 + 1;
        let sq = z.by_square();
        for i in 0..z.n {
            if self.row[i] == 0.0 {
                continue;
            }
            let base = self.base(z, i);
            z.near(&sq, i, r, |j| {
                let k = (base + self.key(z, j)) as usize;
                if self.walk_c(k) >= NO_WALK {
                    return;
                }
                let w = nr.split(self, p, i, j, k);
                if w <= 0.0 {
                    return;
                }
                let t = self.row[i] * self.col[j] * (self.ftab[k] * self.k(z, i, j)) as f64;
                wh[i] += t * w as f64;
                ww[j] += t * w as f64;
            });
        }
        [wh, ww]
    }

    /// T-2214: `no_rail_by_zone` by slot of the pack's land parts (`Params::parts`; by zone with
    /// none), and the commutes no way serves. A zone pair's trips split over its zones' slots by
    /// their homes and jobs; those between slots of one part walk as the pair does
    /// (`NoRail::split`), those between two parts neither walk nor drive.
    pub fn no_rail_by_slot(&self, z: &Zones, p: &Params, nr: &NoRail) -> NoRailSlots {
        let Some(pp) = p.parts.as_deref() else {
            let [walk_home, walk_work] = self.no_rail_by_zone(z, p, nr);
            return NoRailSlots { walk_home, walk_work, unr_home: vec![], unr_work: vec![] };
        };
        let ns = pp.s_part.len();
        let hs = |i: usize, a: usize| if z.pop[i] > 0.0 { pp.s_pop[a] / z.pop[i] } else { 0.0 };
        let js = |j: usize, b: usize| if z.jobs[j] > 0.0 { pp.s_jobs[b] / z.jobs[j] } else { 0.0 };
        let (mut wh, mut ww) = (vec![0.0f64; ns], vec![0.0f64; ns]);
        if self.walk_set {
            let r = (WALK_REACH_KM * 1000.0 / z.zone_m).ceil() as i32 + 1;
            let sq = z.by_square();
            for i in 0..z.n {
                if self.row[i] == 0.0 {
                    continue;
                }
                let base = self.base(z, i);
                z.near(&sq, i, r, |j| {
                    let k = (base + self.key(z, j)) as usize;
                    if self.walk_c(k) >= NO_WALK {
                        return;
                    }
                    let w = nr.split(self, p, i, j, k);
                    if w <= 0.0 {
                        return;
                    }
                    let t = self.row[i] * self.col[j] * (self.ftab[k] * self.k(z, i, j)) as f64 * w as f64;
                    for a in pp.slots(i) {
                        for b in pp.slots(j) {
                            if pp.s_part[a] == pp.s_part[b] {
                                let x = t * hs(i, a) * js(j, b);
                                wh[a] += x;
                                ww[b] += x;
                            }
                        }
                    }
                });
            }
        }
        // Between parts: a zone wholly on the main part sends nothing off it to another such, so
        // only pairs with a mixed zone (`Parts::mixed`) are summed, and a slot off the main part
        // takes its zone's whole row (or column) less what it sends to (takes from) mixed zones.
        let (mut uh, mut uw) = (vec![0.0f64; ns], vec![0.0f64; ns]);
        let is_mixed = {
            let mut v = vec![false; z.n];
            pp.mixed.iter().for_each(|&m| v[m as usize] = true);
            v
        };
        let (mut rows, mut cols) = (vec![0.0f64; z.n], vec![0.0f64; z.n]);
        for &m in &pp.mixed {
            let m = m as usize;
            for j in 0..z.n {
                rows[m] += self.t(z, m, j);
                cols[m] += self.t(z, j, m);
            }
        }
        for i in 0..z.n {
            let (mut to_m, mut from_m) = (0.0f64, 0.0f64);
            for &m in &pp.mixed {
                let m = m as usize;
                let (out, inn) = (self.t(z, i, m), self.t(z, m, i));
                to_m += out;
                from_m += inn;
                for a in pp.slots(i) {
                    let q = pp.s_part[a];
                    uh[a] += hs(i, a) * out * (1.0 - pp.job_share(z, m, q));
                    uw[a] += js(i, a) * inn * (1.0 - pp.home_share(z, m, q));
                }
            }
            if is_mixed[i] {
                for a in pp.slots(i).filter(|&a| pp.s_part[a] != pp.main) {
                    uh[a] += hs(i, a) * (rows[i] - to_m).max(0.0);
                    uw[a] += js(i, a) * (cols[i] - from_m).max(0.0);
                }
            }
        }
        NoRailSlots { walk_home: wh, walk_work: ww, unr_home: uh, unr_work: uw }
    }
}

/// Decay f and distance d (km) by zone grid offset, index (dx + w - 1) * (2h - 1) + (dy + h - 1).
/// Zone-centre distance; within a zone 0.52 x the zone side (mean distance between two random
/// points of a square). The one definition of the decay: the pipeline's solve and the pack
/// loader both use it, so shipped factors and rebuilt trips agree exactly.
/// Each zone's homes and jobs as `ns` cells (`Params::walk_samples`, T-587): the cells where the
/// cumulative commute weight, in pack order, crosses (s + 0.5) / ns of the zone's; NONE for a
/// zone with no weight. Zone by zone, `ns` each.
pub fn walk_points(city: &City, z: &Zones, ns: usize) -> (Vec<u32>, Vec<u32>) {
    let mut off = vec![0u32; z.n + 1];
    for &zi in &z.cell_zone {
        off[zi as usize + 1] += 1;
    }
    for i in 0..z.n {
        off[i + 1] += off[i];
    }
    let mut fill = off.clone();
    let mut cells = vec![0u32; z.cell_zone.len()];
    for (c, &zi) in z.cell_zone.iter().enumerate() {
        cells[fill[zi as usize] as usize] = c as u32;
        fill[zi as usize] += 1;
    }
    let points = |w: &[f32]| {
        let mut out = vec![NONE; z.n * ns];
        for i in 0..z.n {
            let cs = &cells[off[i] as usize..off[i + 1] as usize];
            let tot: f64 = cs.iter().map(|&c| w[c as usize] as f64).sum();
            if tot <= 0.0 {
                continue;
            }
            let (mut acc, mut s) = (0.0f64, 0usize);
            for &c in cs {
                acc += w[c as usize] as f64;
                while s < ns && acc >= (s as f64 + 0.5) / ns as f64 * tot {
                    out[i * ns + s] = c;
                    s += 1;
                }
            }
            // rounding can leave the last on no cell: the last cell with weight
            let last = *cs.iter().rev().find(|&&c| w[c as usize] > 0.0).unwrap();
            out[i * ns + s..(i + 1) * ns].fill(last);
        }
        out
    };
    (points(&city.pop), points(&city.jobs))
}

/// Straight-line metres between cells a and b; a cell to itself 0.52 of its square's side, the
/// mean distance within a square.
#[inline]
pub fn cell_walk_m(city: &City, a: u32, b: u32) -> f32 {
    let (a, b) = (a as usize, b as usize);
    if a == b {
        0.52 * (res9_cells(city, a) as f32 * CELL_KM2).sqrt() * 1000.0
    } else {
        (city.x[a] - city.x[b]).hypot(city.y[a] - city.y[b])
    }
}

/// The US packs' commute curve (T-198), fitted on LODES block pairs: `Params`' default.
pub const US_DECAY_KM: f32 = 30.0;
pub const US_DECAY_POW: f32 = 0.8;
/// T-1091 (docs/demand-kernel.md 1): a curve fitted on units coarser than the 2 km squares
/// (every curve but the US one) is held flat inside this many km, so a job in the home square
/// weighs as one in the next square.
pub const FLAT_NEAR_KM: f32 = 2.0;

/// The flat near end a pack solved now takes for its curve (T-1091): none on the US curve, which
/// was fitted at block level, `FLAT_NEAR_KM` on any other.
pub fn flat_near_km(decay_km: f32, decay_pow: f32) -> f32 {
    if (decay_km - US_DECAY_KM).abs() < 1e-6 && (decay_pow - US_DECAY_POW).abs() < 1e-6 {
        0.0
    } else {
        FLAT_NEAR_KM
    }
}

/// The commute decay f(d) = d^-pow x exp(-d / km), d in km, held at f(flat_km) for d under
/// flat_km (T-1091; 0: none).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Curve {
    pub km: f32,
    pub pow: f32,
    pub flat_km: f32,
}

impl Curve {
    #[inline]
    pub fn at(&self, d: f32) -> f32 {
        let d = d.max(self.flat_km);
        (-d / self.km).exp() * if self.pow != 0.0 { d.powf(-self.pow) } else { 1.0 }
    }
}

impl Params {
    pub fn curve(&self) -> Curve {
        Curve { km: self.decay_km, pow: self.decay_pow, flat_km: self.decay_flat_km }
    }
}

impl PackGravity {
    pub fn curve(&self) -> Curve {
        Curve { km: self.decay_km, pow: self.decay_pow, flat_km: self.flat_km }
    }
}

pub fn decay_table(w: i32, h: i32, zone_m: f32, c: Curve) -> (Vec<f32>, Vec<f32>) {
    let (tw, th) = (2 * w - 1, 2 * h - 1);
    let (offx, offy) = (w - 1, h - 1);
    let zk = zone_m / 1000.0;
    let mut ftab = vec![0f32; (tw * th) as usize];
    let mut dtab = vec![0f32; (tw * th) as usize];
    for dx in -offx..=offx {
        for dy in -offy..=offy {
            let d = if dx == 0 && dy == 0 { 0.52 * zk } else { zk * ((dx * dx + dy * dy) as f32).sqrt() };
            let i = ((dx + offx) * th + dy + offy) as usize;
            ftab[i] = c.at(d);
            dtab[i] = d;
        }
    }
    (ftab, dtab)
}

/// The pair tables' layout for zones `z` (`Gravity`'s `th`, `offx`, `offy`, `kb`), and its
/// rows. One-square zones: `decay_table`'s. With blocks (T-852) each row holds three runs,
/// one per kind of pair: two one-square zones by their squares' offset, a square and a block
/// by the half-square offset of their centres, two blocks by their blocks' offset; a block's key
/// is `kb` on and its base `2 kb` off, which lands each pair in its run (`Gravity::base`).
pub fn table_layout(z: &Zones) -> (i32, i32, i32, i32, i32) {
    let (w, h) = (z.w.max(1), z.h.max(1));
    if z.mixed() {
        (6 * h - 1, w - 1, h - 1, 2 * h, 2 * w - 1)
    } else {
        (2 * h - 1, w - 1, h - 1, 0, 2 * w - 1)
    }
}

/// `Gravity::base` and `key` alone, for other tables of `table_layout`'s.
#[derive(Clone, Copy, Debug)]
pub struct PairKey {
    pub(super) th: i32,
    pub(super) offx: i32,
    pub(super) offy: i32,
    pub(super) kb: i32,
}

impl PairKey {
    pub fn of(z: &Zones) -> PairKey {
        let (th, offx, offy, kb, _) = table_layout(z);
        PairKey { th, offx, offy, kb }
    }
    #[inline]
    pub fn base(&self, z: &Zones, i: usize) -> i32 {
        (self.offx - z.gx[i]) * self.th + self.offy - z.gy[i] - 2 * self.kb * z.big(i) as i32
    }
    #[inline]
    pub fn key(&self, z: &Zones, j: usize) -> i32 {
        z.gx[j] * self.th + z.gy[j] + self.kb * z.big(j) as i32
    }
}

/// Each entry of a mixed layout (`table_layout`): `f` gets its index, the centres' offset in
/// squares and its run (0 squares, 1 a square and a block, 2 blocks).
pub(super) fn mixed_entries(z: &Zones, mut f: impl FnMut(usize, f32, f32, u8)) {
    let (th, offx, offy, _, tw) = table_layout(z);
    let h = z.h.max(1);
    for r in 0..tw {
        let dx = r - offx;
        for y in 0..th {
            let k = (r * th + y) as usize;
            if y < 2 * h - 1 {
                f(k, dx as f32, (y - offy) as f32, 0);
            } else if y < 4 * h - 1 {
                f(k, dx as f32 + 0.5, (y - 3 * h + 1) as f32 + 0.5, 1);
            } else {
                f(k, (dx + 1) as f32, (y - 5 * h + 2) as f32, 2);
            }
        }
    }
}

/// A pair table over zones `z` of `decay(d_km)`: by `decay_table`'s rule for one-square zones;
/// with blocks (T-852), a pair with a block takes the mean of the decay between its squares (each
/// square's centre, one square's own 0.52 of a side), so a block draws as its four squares would
/// with its homes and jobs spread evenly over them.
pub fn pair_decay(z: &Zones, decay: impl Fn(f32) -> f32) -> Vec<f32> {
    let zk = z.zone_m / 1000.0;
    let (th, _, _, _, tw) = table_layout(z);
    // by absolute square offset, one more each way than the grid for a block's far squares
    let (qw, qh) = (z.w.max(1) as usize + 2, z.h.max(1) as usize + 2);
    let mut q = vec![0f32; qw * qh];
    for ax in 0..qw {
        for ay in 0..qh {
            let d = if ax == 0 && ay == 0 { 0.52 * zk } else { zk * ((ax * ax + ay * ay) as f32).sqrt() };
            q[ax * qh + ay] = decay(d);
        }
    }
    let fq = |dx: i32, dy: i32| q[(dx.unsigned_abs() as usize).min(qw - 1) * qh + (dy.unsigned_abs() as usize).min(qh - 1)];
    let mut out = vec![0f32; (tw * th) as usize];
    mixed_entries(z, |k, dx, dy, kind| {
        out[k] = match kind {
            0 => fq(dx as i32, dy as i32),
            1 => {
                let (x, y) = ((dx - 0.5) as i32, (dy - 0.5) as i32);
                0.25 * (fq(x, y) + fq(x + 1, y) + fq(x, y + 1) + fq(x + 1, y + 1))
            }
            _ => {
                let (x, y) = (dx as i32, dy as i32);
                let wt = [0.25f32, 0.5, 0.25];
                let mut s = 0.0;
                for (a, wa) in wt.iter().enumerate() {
                    for (b, wb) in wt.iter().enumerate() {
                        s += wa * wb * fq(x + a as i32 - 1, y + b as i32 - 1);
                    }
                }
                s
            }
        };
    });
    out
}

/// `decay_tables`' decay alone.
pub fn decay_ftab(z: &Zones, c: Curve) -> Vec<f32> {
    if !z.mixed() {
        return decay_table(z.w, z.h, z.zone_m, c).0;
    }
    pair_decay(z, |d| c.at(d))
}

/// The pair tables of `decay_table` for zones `z`: decay and centre distance (km). With blocks
/// (T-852) the decay is `pair_decay`'s and a block's own pair is 0.52 of its side apart.
pub fn decay_tables(z: &Zones, c: Curve) -> (Vec<f32>, Vec<f32>) {
    if !z.mixed() {
        return decay_table(z.w, z.h, z.zone_m, c);
    }
    let ftab = pair_decay(z, |d| c.at(d));
    let zk = z.zone_m / 1000.0;
    let mut dtab = vec![0f32; ftab.len()];
    mixed_entries(z, |k, dx, dy, kind| {
        dtab[k] = if dx == 0.0 && dy == 0.0 { 0.52 * zk * if kind == 2 { 2.0 } else { 1.0 } } else { zk * (dx * dx + dy * dy).sqrt() };
    });
    (ftab, dtab)
}

/// Zones and gravity for a city: from the pack when it ships them for these parameters (city
/// open, a few ms), otherwise solved here (seconds). Returns whether the pack's were used.
pub fn city_gravity(city: &City, p: &Params) -> (Zones, Gravity, bool) {
    if let Some(pg) = &city.gravity {
        if (pg.zone_m - p.zone_m).abs() < 0.5 && (pg.decay_km - p.decay_km).abs() < 1e-6 && (pg.decay_pow - p.decay_pow).abs() < 1e-6 && pg.flat_km == p.decay_flat_km {
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
        region: if pg.zone_region.len() == n { pg.zone_region.clone() } else { vec![0; n] },
        w: pg.grid_w,
        h: pg.grid_h,
        x0: pg.grid_x0,
        y0: pg.grid_y0,
        pop: vec![0.0; n],
        jobs: vec![0.0; n],
        ncell: vec![0; n],
        size: if pg.size.len() == n && pg.size.iter().any(|&s| s == 2) { pg.size.clone() } else { vec![] },
    };
    for c in 0..city.len() {
        let zi = z.cell_zone[c] as usize;
        z.ncell[zi] += res9_cells(city, c);
        z.pop[zi] += city.pop[c] as f64;
        z.jobs[zi] += city.jobs[c] as f64;
    }
    if pg.cells9.len() == n {
        z.ncell.clone_from(&pg.cells9);
    }
    z
}

/// Cell `c`'s land in resolution 9 cells: 7 for a merged res-8 cell (T-044), else 1 (synthetic
/// cities' ids are not H3).
pub(super) fn res9_cells(city: &City, c: usize) -> u32 {
    match city.h3.get(c).map(|&h| (h >> 52) & 0xf) {
        Some(8) => 7,
        _ => 1,
    }
}

pub fn gravity_from_pack(z: &Zones, pg: &PackGravity) -> Gravity {
    let ftab = decay_ftab(z, pg.curve());
    let (th, offx, offy, kb, _) = table_layout(z);
    Gravity {
        row: pg.row.clone(),
        col: pg.col.clone(),
        border: pg.border_factor,
        ftab,
        zk: z.zone_m / 1000.0,
        walk_set: false,
        wroutes: None,
        wcost: WalkCost::default(),
        wf_off: vec![],
        wf_j: vec![],
        wf_c: vec![], wf_s: vec![],
        th,
        offx,
        offy,
        kb,
        iters: pg.iters,
        max_err: pg.max_err,
        mean_km: pg.mean_km,
        total: pg.trips,
        pull: None,
    }
}

/// What the pipeline ships (pack format 1).
pub fn to_pack_gravity(z: &Zones, g: &Gravity, p: &Params) -> PackGravity {
    PackGravity {
        zone_m: z.zone_m,
        decay_km: p.decay_km,
        decay_pow: p.decay_pow,
        flat_km: p.decay_flat_km,
        border_factor: g.border,
        grid_x0: z.x0,
        grid_y0: z.y0,
        grid_w: z.w,
        grid_h: z.h,
        cell_zone: z.cell_zone.clone(),
        gx: z.gx.clone(),
        gy: z.gy.clone(),
        // one region: nothing to ship (and the pack reads as before T-162)
        zone_region: if z.region.iter().any(|&r| r != z.region[0]) { z.region.clone() } else { vec![] },
        row: g.row.clone(),
        col: g.col.clone(),
        iters: g.iters,
        max_err: g.max_err,
        mean_km: g.mean_km,
        trips: g.total,
        // the pack's own gravity: every commute stays in it (T-276's joint solve sets these)
        home: vec![],
        work: vec![],
        joint: None,
        pull: g.pull.map(|[lo, hi, fill, held]| serde_json::json!({"cap": PULL_CAP, "band": [lo, hi], "fill": fill, "held": held})),
        cells9: vec![],
        size: z.size.clone(),
    }
}

/// T-1000 (docs/demand-kernel.md 1): the balance's pull cap. A zone's jobs pull workers at most this
/// many times harder, or more weakly, than their count and distance alone, measured against the
/// pack's job-weighted geometric mean pull; past it they stay part unfilled, or overfill.
pub const PULL_CAP: f64 = 20.0;

/// Gravity by Furness balancing, rows exact and columns to their jobs within `PULL_CAP`. Workers =
/// population scaled to total jobs.
/// The decay depends on zone-centre distance only, so f is a small table by grid offset; it is
/// multiplied by `p.border_factor` between zones of different regions (T-162).
pub fn gravity(z: &Zones, p: &Params, max_iters: usize, tol: f64) -> Gravity {
    let (th, offx, offy, kb, _) = table_layout(z);
    let (ftab, dtab) = decay_tables(z, p.curve());
    let tp: f64 = z.pop.iter().sum();
    let tj: f64 = z.jobs.iter().sum();
    let o: Vec<f64> = z.pop.iter().map(|&v| v * tj / tp).collect();
    let rows: Vec<usize> = (0..z.n).filter(|&i| o[i] > 0.0).collect();
    let cols: Vec<usize> = (0..z.n).filter(|&j| z.jobs[j] > 0.0).collect();
    let ckey: Vec<i32> = cols.iter().map(|&j| z.gx[j] * th + z.gy[j] + kb * z.big(j) as i32).collect();
    let creg: Vec<u8> = cols.iter().map(|&j| z.region[j]).collect();
    // the factor by "regions differ" (index 0 same, 1 different)
    let kf = [1.0f64, p.border_factor as f64];
    let mut g = Gravity { row: vec![0.0; z.n], col: z.jobs.clone(), border: p.border_factor, ftab, zk: z.zone_m / 1000.0, walk_set: false, wroutes: None, wcost: WalkCost::default(), wf_off: vec![], wf_j: vec![], wf_c: vec![], wf_s: vec![], th, offx, offy, kb, iters: 0, max_err: 0.0, mean_km: 0.0, total: 0.0, pull: None };
    let mut colc: Vec<f64> = cols.iter().map(|&j| g.col[j]).collect();
    let mut acc = vec![0f64; cols.len()];
    for it in 0..max_iters {
        // rows: A_I = 1 / sum_J col_J f_IJ k_IJ
        let mut err: f64 = 0.0;
        for &i in &rows {
            let base = g.base(z, i);
            let ri = z.region[i];
            let mut s = 0.0f64;
            for (k, &key) in ckey.iter().enumerate() {
                s += colc[k] * g.ftab[(base + key) as usize] as f64 * kf[(creg[k] != ri) as usize];
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
        // columns: B_J = 1 / sum_I row_I f_IJ k_IJ
        acc.iter_mut().for_each(|v| *v = 0.0);
        for &i in &rows {
            let base = g.base(z, i);
            let ri = z.region[i];
            let r = g.row[i];
            for (k, &key) in ckey.iter().enumerate() {
                acc[k] += r * g.ftab[(base + key) as usize] as f64 * kf[(creg[k] != ri) as usize];
            }
        }
        // T-1000: a column's pull 1 / acc is held within PULL_CAP of the job-weighted geometric
        // mean. The columns inside share what the held ones leave or take over (all divided by
        // `scale`, exactly 1 when none is held), so the rows' total still meets the jobs' and the
        // balance has a fixed point.
        let (mut sl, mut sw) = (0.0f64, 0.0f64);
        for (k, &j) in cols.iter().enumerate() {
            if acc[k] > 0.0 {
                sl -= z.jobs[j] * acc[k].ln();
                sw += z.jobs[j];
            }
        }
        let (lo, hi) = if sw > 0.0 { ((sl / sw).exp() / PULL_CAP, (sl / sw).exp() * PULL_CAP) } else { (0.0, f64::INFINITY) };
        let (mut all, mut met, mut held) = (0.0f64, 0.0f64, 0.0f64);
        for (k, &j) in cols.iter().enumerate() {
            let b = 1.0 / acc[k];
            all += z.jobs[j];
            met += if b < lo || b > hi {
                held += z.jobs[j];
                z.jobs[j] * b.clamp(lo, hi) * acc[k]
            } else {
                z.jobs[j]
            };
        }
        let scale = if all > 0.0 { met / all } else { 1.0 };
        for (k, &j) in cols.iter().enumerate() {
            let b = 1.0 / acc[k];
            colc[k] = (if b < lo { z.jobs[j] * lo } else if b > hi { z.jobs[j] * hi } else { z.jobs[j] / acc[k] }) / scale;
            g.col[j] = colc[k];
        }
        g.pull = Some([lo / scale, hi / scale, 1.0 / scale, held / all.max(1e-300)]);
    }
    // mean trip length
    let (mut sum, mut sumd) = (0.0f64, 0.0f64);
    for &i in &rows {
        let base = g.base(z, i);
        let ri = z.region[i];
        for (k, &key) in ckey.iter().enumerate() {
            let idx = (base + key) as usize;
            let t = g.row[i] * colc[k] * g.ftab[idx] as f64 * kf[(creg[k] != ri) as usize];
            sum += t;
            sumd += t * dtab[idx] as f64;
        }
    }
    g.total = sum;
    g.mean_km = sumd / sum;
    g
}
