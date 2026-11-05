use super::*;

// ---------------------------------------------------------------------------------------------
// 2. Station access per cell

pub struct Access {
    pub n: Vec<u8>,
    /// `w` per cell: up to K train stations nearest first, then up to `BUS_SLOTS` bus stops nearest
    /// first (T-1464); `n` of them used.
    pub st: Vec<u32>,
    /// Real walk minutes, `w` per cell.
    pub walk: Vec<f32>,
    /// Entries per cell: K, or K + `BUS_SLOTS` with bus corridors.
    pub w: usize,
    pub cells_with_access: usize,
    /// Per station, the cells whose walk to it goes around water (T-275,
    /// `routes::Routes::station_walks`), with their walking metres; None without routes.
    pub ex: Vec<Option<std::sync::Arc<Vec<(u32, f32)>>>>,
    /// T-2214: each station's land part, its nearest cell's (`Params::parts`); empty with none.
    pub st_part: Vec<u16>,
}

/// Station lookup grid with buckets the size of the walk cutoff; only occupied buckets are stored
/// (a dense grid over a continent-wide network is gigabytes, T-1558).
pub(super) struct StationGrid {
    pub(super) minx: f32,
    pub(super) miny: f32,
    pub(super) size: f32,
    pub(super) w: i32,
    pub(super) h: i32,
    /// Occupied buckets' ids (`gx * h + gy`), sorted; bucket `i` holds `items[off[i]..off[i + 1]]`.
    pub(super) keys: Vec<u64>,
    pub(super) off: Vec<u32>,
    pub(super) items: Vec<u32>,
}

impl StationGrid {
    /// Buckets of `size` metres over the stations' extent plus `pad` metres all round.
    pub(super) fn new(net: &Network, size: f32, pad: f32) -> Self {
        let minx = net.st_x.iter().cloned().fold(f32::INFINITY, f32::min) - pad;
        let miny = net.st_y.iter().cloned().fold(f32::INFINITY, f32::min) - pad;
        let maxx = net.st_x.iter().cloned().fold(f32::NEG_INFINITY, f32::max) + pad;
        let maxy = net.st_y.iter().cloned().fold(f32::NEG_INFINITY, f32::max) + pad;
        let w = ((maxx - minx) / size) as i32 + 1;
        let h = ((maxy - miny) / size) as i32 + 1;
        let b = |s: usize| (((net.st_x[s] - minx) / size) as i64 * h as i64 + ((net.st_y[s] - miny) / size) as i64) as u64;
        let mut by_bucket: Vec<(u64, u32)> = (0..net.n_stations()).map(|s| (b(s), s as u32)).collect();
        by_bucket.sort_unstable();
        let mut keys: Vec<u64> = vec![];
        let mut off: Vec<u32> = vec![];
        for (i, &(k, _)) in by_bucket.iter().enumerate() {
            if keys.last() != Some(&k) {
                keys.push(k);
                off.push(i as u32);
            }
        }
        off.push(by_bucket.len() as u32);
        let items = by_bucket.iter().map(|&(_, s)| s).collect();
        StationGrid { minx, miny, size, w, h, keys, off, items }
    }
    pub(super) fn near(&self, x: f32, y: f32, mut f: impl FnMut(u32)) {
        let bx = ((x - self.minx) / self.size).floor() as i32;
        let by = ((y - self.miny) / self.size).floor() as i32;
        for gx in (bx - 1).max(0)..=(bx + 1).min(self.w - 1) {
            for gy in (by - 1).max(0)..=(by + 1).min(self.h - 1) {
                let k = (gx as i64 * self.h as i64 + gy as i64) as u64;
                if let Ok(i) = self.keys.binary_search(&k) {
                    for &s in &self.items[self.off[i] as usize..self.off[i + 1] as usize] {
                        f(s);
                    }
                }
            }
        }
    }
}

impl Access {
    /// Cell `c`'s entries (indices into `st` and `walk`).
    #[inline]
    pub fn of(&self, c: usize) -> std::ops::Range<usize> {
        c * self.w..c * self.w + self.n[c] as usize
    }
}

/// Bucket size of the cell grid the access search scatters stations over.
pub(super) const ACCESS_BUCKET_M: f32 = 1000.0;
/// T-1464: a home this near a bus route (ground m) may walk to it and ride.
pub const BUS_REACH_M: f32 = 300.0;
/// T-1576: perceived minutes added to every boarding of a bus line, a change onto it included
/// (`Graph::rn_bus`): a bus is less dependable and less comfortable than a train.
pub const BUS_BOARD_MIN: f32 = 6.0;
/// The walk along the route to the nearest street stop, on average (street stops every 400 m).
pub const BUS_WALK_EXTRA_M: f32 = 100.0;
/// The bus stops a cell keeps beside its K train stations.
pub const BUS_SLOTS: usize = 2;

/// Distance from (x, y) to a polyline, m.
pub fn polyline_dist(pts: &[[f32; 2]], x: f32, y: f32) -> f32 {
    let mut best = f32::INFINITY;
    for w in pts.windows(2) {
        let ([ax, ay], [bx, by]) = (w[0], w[1]);
        let (dx, dy) = (bx - ax, by - ay);
        let l2 = dx * dx + dy * dy;
        let t = if l2 > 0.0 { (((x - ax) * dx + (y - ay) * dy) / l2).clamp(0.0, 1.0) } else { 0.0 };
        best = best.min((x - ax - t * dx).hypot(y - ay - t * dy));
    }
    best
}

/// T-1464: the bus stops whose stretch of route passes within `BUS_REACH_M` of (x, y), each with
/// the walk to it (the distance to the route plus `BUS_WALK_EXTRA_M`), nearest first, at most
/// `BUS_SLOTS`.
pub fn bus_walks(catches: &[super::super::network::Catch], x: f32, y: f32) -> Vec<(u32, f32)> {
    let mut out: Vec<(u32, f32)> = vec![];
    for c in catches {
        let d = polyline_dist(&c.pts, x, y);
        if d <= BUS_REACH_M {
            bus_insert(&mut out, c.st, d + BUS_WALK_EXTRA_M);
        }
    }
    out
}

/// Put a bus stop into a list of at most `BUS_SLOTS`, nearest first, each stop once.
pub(super) fn bus_insert(list: &mut Vec<(u32, f32)>, st: u32, m: f32) {
    if let Some(i) = list.iter().position(|e| e.0 == st) {
        if list[i].1 <= m {
            return;
        }
        list.remove(i);
    }
    let i = list.partition_point(|e| e.1 <= m);
    if i < BUS_SLOTS {
        list.insert(i, (st, m));
        list.truncate(BUS_SLOTS);
    }
}

/// K nearest stations per cell within `walk_cutoff_m`, by scattering each station over the
/// cells near it (a station touches the few hundred cells within the bound, so a far bound
/// costs little; searching outwards from every cell cost 5-15x as much where stations are
/// sparse).
pub fn build_access(city: &City, net: &Network, p: &Params) -> Access {
    let n = city.len();
    let mut a = Access { n: vec![0; n], st: vec![NONE; n * K], walk: vec![f32::INFINITY; n * K], w: K, cells_with_access: 0, ex: vec![None; net.n_stations()], st_part: vec![] };
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
    // T-2214: a station's land part is its nearest cell's (a ferry port's, its own shore's)
    let pp = p.parts.as_deref();
    if let Some(pp) = pp {
        a.st_part = (0..net.n_stations())
            .map(|s| {
                let (sx, sy) = (net.st_x[s], net.st_y[s]);
                // a station off the cells' grid searches from the grid's edge
                let (bx, by) = ((((sx - minx) / size).floor() as i32).clamp(0, w - 1), (((sy - miny) / size).floor() as i32).clamp(0, h - 1));
                let mut best = (f32::INFINITY, pp.main);
                // rings of buckets until one holds a cell, then one more (a nearer cell can lie
                // in the next ring)
                let mut found = None;
                for r in 0..(w.max(h) + 1) {
                    for gx in (bx - r).max(0)..=(bx + r).min(w - 1) {
                        for gy in (by - r).max(0)..=(by + r).min(h - 1) {
                            if (gx - bx).abs() != r && (gy - by).abs() != r {
                                continue;
                            }
                            let k = (gx * h + gy) as usize;
                            for &c in &cells[off[k] as usize..off[k + 1] as usize] {
                                let d2 = (city.x[c as usize] - sx).powi(2) + (city.y[c as usize] - sy).powi(2);
                                if d2 < best.0 {
                                    best = (d2, pp.cell[c as usize]);
                                }
                            }
                        }
                    }
                    if best.0.is_finite() && found.is_none() {
                        found = Some(r);
                    }
                    if found.is_some_and(|f| r > f) {
                        break;
                    }
                }
                best.1
            })
            .collect();
    }
    // each station into the sorted K lists (distance squared for now) of the cells near it
    let cut = p.walk_cutoff_m;
    let cut2 = cut * cut;
    for s in 0..net.n_stations() {
        // a bus stop is reached through its corridor only (T-1464)
        if net.bus_only(s) {
            continue;
        }
        let (sx, sy) = (net.st_x[s], net.st_y[s]);
        let gx0 = (((sx - cut - minx) / size).floor() as i32).max(0);
        let gx1 = (((sx + cut - minx) / size).floor() as i32).min(w - 1);
        let gy0 = (((sy - cut - miny) / size).floor() as i32).max(0);
        let gy1 = (((sy + cut - miny) / size).floor() as i32).min(h - 1);
        // T-275: the cells whose walk to this station goes around water (searched once a station
        // position, kept for the thread)
        let ex = p.routes.as_ref().and_then(|r| {
            r.station_walks(sx, sy, cut, |f| {
                for gx in gx0..=gx1 {
                    for gy in gy0..=gy1 {
                        let k = (gx * h + gy) as usize;
                        for &c in &cells[off[k] as usize..off[k + 1] as usize] {
                            f(c, city.x[c as usize], city.y[c as usize]);
                        }
                    }
                }
            })
        });
        a.ex[s] = ex.clone();
        let exl: &[(u32, f32)] = ex.as_deref().map_or(&[], |v| &v[..]);
        let sp = a.st_part.get(s).copied();
        for gx in gx0..=gx1 {
            for gy in gy0..=gy1 {
                let k = (gx * h + gy) as usize;
                for &c in &cells[off[k] as usize..off[k + 1] as usize] {
                    if let (Some(pp), Some(sp)) = (pp, sp) {
                        if pp.cell[c as usize] != sp {
                            continue; // T-2214: no walk to another land part
                        }
                    }
                    let mut d2 = (city.x[c as usize] - sx).powi(2) + (city.y[c as usize] - sy).powi(2);
                    if !exl.is_empty() && d2 <= cut2 {
                        if let Ok(e) = exl.binary_search_by_key(&c, |e| e.0) {
                            let m = exl[e].1;
                            if m > cut {
                                continue; // around the water it is beyond the walk's bound
                            }
                            d2 = m * m;
                        }
                    }
                    let c = c as usize;
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
    if !net.catches.is_empty() {
        add_bus_access(&mut a, city, net, &cells, &off, (minx, miny, w, h), pp);
    }
    let wd = a.w;
    for c in 0..n {
        let cnt = a.n[c] as usize;
        for wv in a.walk[c * wd..c * wd + cnt].iter_mut() {
            *wv = p.walk_min(wv.sqrt());
        }
        if cnt > 0 {
            a.cells_with_access += 1;
        }
    }
    a
}

/// T-1464: each cell's bus stops (`bus_walks`, squared metres for now) after its train stations,
/// the entries laid out `K + BUS_SLOTS` a cell. `grid` is `build_access`'s cell buckets.
pub(super) fn add_bus_access(a: &mut Access, city: &City, net: &Network, cells: &[u32], off: &[u32], grid: (f32, f32, i32, i32), pp: Option<&Parts>) {
    let (minx, miny, w, h) = grid;
    let size = ACCESS_BUCKET_M;
    let n = city.len();
    let mut bus: Vec<Vec<(u32, f32)>> = vec![vec![]; n];
    for c in &net.catches {
        let (mut x0, mut y0, mut x1, mut y1) = (f32::INFINITY, f32::INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY);
        for p in &c.pts {
            (x0, y0, x1, y1) = (x0.min(p[0]), y0.min(p[1]), x1.max(p[0]), y1.max(p[1]));
        }
        let gx0 = (((x0 - BUS_REACH_M - minx) / size).floor() as i32).max(0);
        let gx1 = (((x1 + BUS_REACH_M - minx) / size).floor() as i32).min(w - 1);
        let gy0 = (((y0 - BUS_REACH_M - miny) / size).floor() as i32).max(0);
        let gy1 = (((y1 + BUS_REACH_M - miny) / size).floor() as i32).min(h - 1);
        for gx in gx0..=gx1 {
            for gy in gy0..=gy1 {
                let k = (gx * h + gy) as usize;
                for &cell in &cells[off[k] as usize..off[k + 1] as usize] {
                    let cell = cell as usize;
                    if pp.is_some_and(|pp| a.st_part.get(c.st as usize).is_some_and(|&sp| pp.cell[cell] != sp)) {
                        continue;
                    }
                    let d = polyline_dist(&c.pts, city.x[cell], city.y[cell]);
                    if d <= BUS_REACH_M {
                        bus_insert(&mut bus[cell], c.st, d + BUS_WALK_EXTRA_M);
                    }
                }
            }
        }
    }
    let wd = K + BUS_SLOTS;
    let (mut st, mut walk) = (vec![NONE; n * wd], vec![f32::INFINITY; n * wd]);
    for c in 0..n {
        let t = a.n[c] as usize;
        st[c * wd..c * wd + t].copy_from_slice(&a.st[c * K..c * K + t]);
        walk[c * wd..c * wd + t].copy_from_slice(&a.walk[c * K..c * K + t]);
        for (j, &(s, m)) in bus[c].iter().enumerate() {
            st[c * wd + t + j] = s;
            walk[c * wd + t + j] = m * m;
        }
        a.n[c] += bus[c].len() as u8;
    }
    (a.st, a.walk, a.w) = (st, walk, wd);
}
