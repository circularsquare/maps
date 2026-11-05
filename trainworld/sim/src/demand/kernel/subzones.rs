use super::*;

// ---------------------------------------------------------------------------------------------
// 3. Access subzones

pub struct Subzones {
    pub n: usize,
    pub zone: Vec<u32>,
    pub pop: Vec<f32>,
    pub jobs: Vec<f32>,
    pub cx: Vec<f32>,
    pub cy: Vec<f32>,
    /// Parking at the home end of its homes' car trips (`Params::home_park_cell`, home-weighted;
    /// T-854), perceived minutes. Its ride entries include it.
    pub home_park: Vec<f32>,
    /// CSR access list, per subzone its walk entries (`walk`) then its ride entries (`ride`,
    /// T-371): station, perceived minutes, the distance in metres and the real minutes. A walk
    /// entry's minutes are the members' `Params::access_cost` (a pop/jobs-weighted logsum) and
    /// its distance their weighted mean (straight, or around water with the pack's routes,
    /// T-275); a ride entry's are its zone's (`zone_rides`). Walk entries serve both ends of a
    /// trip, ride entries only the home end.
    pub acc_off: Vec<u32>,
    pub acc_mid: Vec<u32>,
    pub acc_st: Vec<u32>,
    pub acc_w: Vec<f32>,
    pub acc_d: Vec<f32>,
    pub acc_t: Vec<f32>,
    /// CSR: a ride-only subzone's homes by zone (zone, homes); empty for the others, whose homes
    /// are all in `zone`.
    pub home_off: Vec<u32>,
    pub home_zones: Vec<(u32, f32)>,
    /// Per zone, subzones with commuters living there (origins) and with jobs (destinations).
    pub zo_off: Vec<u32>,
    pub zo: Vec<u32>,
    pub zd_off: Vec<u32>,
    pub zd: Vec<u32>,
    /// Subzone of each cell (NONE for cells with no station within reach). The cells with homes
    /// beyond every walk but within a ride are one subzone per `RIDE_GROUP` square and region,
    /// with no walk entries and no jobs: origins only, in their most populous zone (`zone`).
    pub cell: Vec<u32>,
    /// CSR: the cells of each subzone (`mem[mem_off[u]..mem_off[u + 1]]`).
    pub mem_off: Vec<u32>,
    pub mem: Vec<u32>,
    /// T-1920: each cell's cheapest walk to a station, perceived minutes as the trip explorer's
    /// own cell has it (infinite with none; empty where not built).
    pub walk_c: Vec<f32>,
    /// T-2214: each subzone's land part (every member's; `Params::parts`), empty with none; and
    /// each station's (`Access::st_part`).
    pub part: Vec<u16>,
    pub st_part: Vec<u16>,
}

impl Subzones {
    /// Whether subzones u and v are on two land parts (T-2214).
    #[inline]
    pub fn apart(&self, u: usize, v: usize) -> bool {
        !self.part.is_empty() && self.part[u] != self.part[v]
    }
    /// Subzone `u`'s walk entries (`acc_st` indices).
    #[inline]
    pub fn walk(&self, u: usize) -> std::ops::Range<usize> {
        self.acc_off[u] as usize..self.acc_mid[u] as usize
    }
    /// Its ride entries, home end only (T-371).
    #[inline]
    pub fn ride(&self, u: usize) -> std::ops::Range<usize> {
        self.acc_mid[u] as usize..self.acc_off[u + 1] as usize
    }
    /// Subzone `u`'s cheapest walk or ride as `Params::pair_walk_max` sees it: a ride's drive at
    /// `w_drive`, not `ride_w` (T-916: the heavier drive prices long drives without cutting them
    /// off where streets are slow).
    pub fn bound_min(&self, p: &Params, u: usize) -> f32 {
        let walk = self.walk(u).map(|k| self.acc_w[k]).fold(f32::INFINITY, f32::min);
        walk.min(self.ride_min(p, u))
    }
    /// Its cheapest ride as `bound_min` prices it (infinite with none).
    pub fn ride_min(&self, p: &Params, u: usize) -> f32 {
        self.ride(u).map(|k| p.ride_bound_cost(self.acc_w[k], self.acc_t[k])).fold(f32::INFINITY, f32::min)
    }
    /// Subzone `u`'s part of the home -> work trips of the period (`share` of the day): its
    /// homes' share of their zones' gravity row factors. A ride-only subzone's homes keep their
    /// own zones' totals, and its trips take its `zone`'s distance decay.
    pub fn origin_weight(&self, u: usize, z: &Zones, gr: &Gravity, share: f64) -> f32 {
        let h = &self.home_zones[self.home_off[u] as usize..self.home_off[u + 1] as usize];
        if h.is_empty() {
            let zi = self.zone[u] as usize;
            return if z.pop[zi] > 0.0 { (gr.row[zi] * self.pop[u] as f64 / z.pop[zi] * share) as f32 } else { 0.0 };
        }
        h.iter().map(|&(zi, pop)| if z.pop[zi as usize] > 0.0 { gr.row[zi as usize] * pop as f64 / z.pop[zi as usize] * share } else { 0.0 }).sum::<f64>() as f32
    }
}

/// Whether a commute between zones `i` and `j` is long enough to be offered the ride (T-371).
#[inline]
pub fn ride_offered(z: &Zones, p: &Params, i: usize, j: usize) -> bool {
    z.d2_m(i, j) >= (p.ride_min_km * 1000.0).powi(2)
}

/// A ride entry (T-371): station, perceived minutes, straight metres, real drive minutes.
pub type Ride = (u32, f32, f32, f32);

/// The end of a trip a ride serves (T-373): at home the rider may park at the station or be
/// dropped off; at the far end there is no car to park, only a drop-off (`d_drop`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum End {
    Home,
    Far,
}

/// A ride's fixed cost to or from station `s` (`p_ride`), less `PARK_RIDE_MIN` where it has a park
/// and ride (T-1281).
pub fn ride_fixed(p: &Params, net: &Network, s: usize) -> f32 {
    if net.ups(s) & upgrades::PARK_RIDE != 0 { (p.p_ride - upgrades::PARK_RIDE_MIN).max(0.0) } else { p.p_ride }
}

/// How much of its minutes a wait at station `s` counts in mode choice (T-1281): less in a
/// waiting room, which a station building is for its members (T-2276).
pub fn wait_factor(net: &Network, s: usize) -> f32 {
    if net.ups(s) & (upgrades::WAITING_ROOM | upgrades::IN_BUILDING) != 0 { upgrades::WAITING_ROOM_WAIT } else { 1.0 }
}

/// Past the `ride_lines` cheapest, a place's rides take up to this many more (T-1511): the
/// cheapest line of each track component none of those is on.
pub const RIDE_TRACK_MAX: usize = 2;

/// The ride options from each place (its zone, x, y), CSR, for every layer (local commutes and
/// everyday trips, the regional market, T-362's long trips): walks come from `build_access` per
/// cell, rides from here. For each line, its station with the cheapest ride within `ride_max_m`;
/// then the stations of the `ride_lines` cheapest lines, and with `track` (each station's track
/// component and whether a train line stops there) up to `RIDE_TRACK_MAX` stations on track the
/// cheapest lines miss. The drive is the whole-trip drive's (`drive_min_f` with the zones' routed
/// factor) from the place, in its zone, to the station's zone; at the station, the cheaper of
/// parking there and `d_drop` at the home end, `d_drop` at the far end.
pub fn access_options(zones: &Zones, net: &Network, p: &Params, places: &[(usize, f32, f32)], end: End, track: Option<(&[u32], &[bool])>) -> (Vec<u32>, Vec<Ride>) {
    let mut off = vec![0u32; places.len() + 1];
    let mut out = vec![];
    if p.ride_lines == 0 || net.n_stations() == 0 {
        return (off, out);
    }
    let dens = zones.density();
    let place = |z: Option<usize>| p.place_min_per_km(z.map_or(0.0, |z| dens[z]));
    let station_end = |z: Option<usize>| p.ride_station_end(zones, z, end).0;
    let own = |road_km: f32| p.ride_own(road_km, end);
    let sq = zones.by_square();
    let st_zone: Vec<Option<usize>> = (0..net.n_stations())
        .map(|s| {
            let gx = ((net.st_x[s] - zones.x0) / zones.zone_m).floor() as i32;
            let gy = ((net.st_y[s] - zones.y0) / zones.zone_m).floor() as i32;
            zones.zone_at(&sq, gx, gy)
        })
        .collect();
    let mut st_lines: Vec<Vec<u32>> = vec![vec![]; net.n_stations()];
    for l in 0..net.n_lines() {
        for &s in net.stops(l) {
            if st_lines[s as usize].last() != Some(&(l as u32)) {
                st_lines[s as usize].push(l as u32);
            }
        }
    }
    let grid = StationGrid::new(net, p.ride_max_m, p.ride_max_m);
    let max2 = p.ride_max_m * p.ride_max_m;
    // per line: (cost, station, metres, minutes), or none yet from this place
    let mut best: Vec<(f32, u32, f32, f32)> = vec![(f32::INFINITY, NONE, 0.0, 0.0); net.n_lines()];
    let mut seen: Vec<u32> = vec![];
    for (pi, &(i, x, y)) in places.iter().enumerate() {
        {
            let a = place(Some(i));
            grid.near(x, y, |s| {
                let s = s as usize;
                let d2 = (net.st_x[s] - x).powi(2) + (net.st_y[s] - y).powi(2);
                if d2 > max2 || st_lines[s].is_empty() {
                    return;
                }
                let d = d2.sqrt();
                let j = st_zone[s];
                let f = j.map_or(p.road_detour, |j| p.drive_factor(i, j));
                let t = p.drive_min_f(d, f, a, place(j));
                let cost = ride_fixed(p, net, s) + p.ride_w() * t + own(d * 0.001 * f) + station_end(j);
                for &l in &st_lines[s] {
                    let b = &mut best[l as usize];
                    if b.1 == NONE {
                        seen.push(l);
                    }
                    if cost < b.0 {
                        *b = (cost, s as u32, d, t);
                    }
                }
            });
            seen.sort_by(|&a, &b| best[a as usize].0.total_cmp(&best[b as usize].0));
            let start = out.len();
            for &l in seen.iter().take(p.ride_lines) {
                let (cost, s, d, t) = best[l as usize];
                if !out[start..].iter().any(|e: &Ride| e.0 == s) {
                    out.push((s, cost, d, t));
                }
            }
            if let Some((comp, on_train)) = track {
                let mut comps: Vec<u32> = out[start..].iter().map(|e| comp[e.0 as usize]).collect();
                let top = comps.len();
                for &l in seen.iter().skip(p.ride_lines) {
                    if comps.len() >= top + RIDE_TRACK_MAX {
                        break;
                    }
                    let (cost, s, d, t) = best[l as usize];
                    if on_train[s as usize] && !comps.contains(&comp[s as usize]) {
                        comps.push(comp[s as usize]);
                        out.push((s, cost, d, t));
                    }
                }
            }
            for &l in &seen {
                best[l as usize] = (f32::INFINITY, NONE, 0.0, 0.0);
            }
            seen.clear();
        }
        off[pi + 1] = out.len() as u32;
    }
    (off, out)
}

/// Whether each station has a train line: one with trains in some period that is no bus line.
pub fn on_train(net: &Network) -> Vec<bool> {
    let mut out = vec![false; net.n_stations()];
    for l in 0..net.n_lines() {
        let stops = net.stops(l);
        if net.bus_line(l) || net.line_headway_min[l].iter().all(|h| !h.is_finite()) {
            continue;
        }
        for &s in stops {
            out[s as usize] = true;
        }
    }
    out
}

/// Each station's component of the graph (edges taken both ways), named by its lowest station.
pub fn station_components(g: &Graph) -> Vec<u32> {
    let mut parent: Vec<u32> = (0..g.n as u32).collect();
    fn find(p: &mut [u32], mut i: u32) -> u32 {
        while p[i as usize] != i {
            p[i as usize] = p[p[i as usize] as usize];
            i = p[i as usize];
        }
        i
    }
    for a in 0..g.n {
        for e in g.adj_off[a] as usize..g.adj_off[a + 1] as usize {
            let (ra, rb) = (find(&mut parent, a as u32), find(&mut parent, g.adj_to[e]));
            if ra != rb {
                parent[ra.max(rb) as usize] = ra.min(rb);
            }
        }
    }
    (0..g.n_st as u32).map(|s| find(&mut parent, s)).collect()
}

/// Ride-only subzones group homes beyond every walk by squares of this many zones a side
/// (T-371: 6 km, as the market's sites; one per 2 km zone cost the real New York day a third
/// more solve time).
pub(super) const RIDE_GROUP: i32 = 3;

pub fn build_subzones(city: &City, zones: &Zones, acc: &Access, net: &Network, p: &Params) -> Subzones {
    build_subzones_on(city, &city.pop, &city.jobs, zones, acc, net, p)
}

/// `build_subzones` on home and work weights other than the city's own (`zones` summing them).
pub fn build_subzones_on(city: &City, home_w: &[f32], work_w: &[f32], zones: &Zones, acc: &Access, net: &Network, p: &Params) -> Subzones {
    let n = city.len();
    let aw = acc.w;
    let mut cell_sub = vec![NONE; n];
    // (zone, nearest station and band; its nearest bus stop if any, T-1464)
    let mut map: HashMap<(u64, u32), u32> = HashMap::new();
    let mut nsub = 0u32;
    // homes beyond every walk, by group square and region (T-371): per group its homes, their
    // centre, and its homes by zone
    let mut group_of: HashMap<(u64, u16), u32> = HashMap::new();
    let mut cell_group = vec![NONE; n];
    let mut gsum: Vec<(f64, f64, f64)> = vec![];
    // T-2214: a group per land part too, and every subzone's rides on its own part
    let pp = p.parts.as_deref();
    let cpart = |c: usize| pp.map_or(0, |pp| pp.cell[c]);
    let mut gpart: Vec<u16> = vec![];
    let mut gzone: std::collections::BTreeMap<(u32, u32), f64> = Default::default();
    let (mut zx, mut zy) = (vec![0f64; zones.n], vec![0f64; zones.n]);
    for c in 0..n {
        let zi = zones.cell_zone[c] as usize;
        let pop = home_w[c] as f64;
        zx[zi] += pop * city.x[c] as f64;
        zy[zi] += pop * city.y[c] as f64;
        if acc.n[c] == 0 {
            if pop > 0.0 && p.ride_lines > 0 {
                let key = ((zones.gx[zi] / RIDE_GROUP) as u64) << 40 | ((zones.gy[zi] / RIDE_GROUP) as u64) << 8 | zones.region[zi] as u64;
                let g = *group_of.entry((key, cpart(c))).or_insert_with(|| {
                    gsum.push((0.0, 0.0, 0.0));
                    gpart.push(cpart(c));
                    gsum.len() as u32 - 1
                });
                cell_group[c] = g;
                let e = &mut gsum[g as usize];
                e.0 += pop;
                e.1 += pop * city.x[c] as f64;
                e.2 += pop * city.y[c] as f64;
                *gzone.entry((g, zi as u32)).or_default() += pop;
            }
            continue;
        }
        if home_w[c] + work_w[c] <= 0.0 {
            continue; // nobody commutes from or to here
        }
        let key = if p.band_min < 0.0 {
            (c as u64, NONE)
        } else {
            let band = if p.band_min > 0.0 { (acc.walk[c * aw] / p.band_min) as u64 } else { 0 };
            let bus = acc.of(c).map(|i| acc.st[i]).find(|&s| net.bus_only(s as usize)).unwrap_or(NONE);
            (((zones.cell_zone[c] as u64) << 32) | ((acc.st[c * aw] as u64) << 8) | band.min(255), bus)
        };
        let id = *map.entry(key).or_insert_with(|| {
            nsub += 1;
            nsub - 1
        });
        cell_sub[c] = id;
    }
    drop(map);
    let n_walk = nsub as usize;
    // rides from each zone's home-weighted centre (for its walk subzones), then from each group's
    let mut places: Vec<(usize, f32, f32)> = (0..zones.n)
        .map(|i| {
            let w = zones.pop[i].max(1e-9);
            (i, (zx[i] / w) as f32, (zy[i] / w) as f32)
        })
        .collect();
    // a group's zone is its most populous one, its homes by zone `gz`
    let mut g_rep = vec![(NONE, 0f64); gsum.len()];
    let mut gz_off = vec![0u32; gsum.len() + 1];
    let mut gz: Vec<(u32, f32)> = vec![];
    for (&(g, zi), &pop) in &gzone {
        let r = &mut g_rep[g as usize];
        if pop > r.1 {
            *r = (zi, pop);
        }
        gz_off[g as usize + 1] += 1;
        gz.push((zi, pop as f32));
    }
    for g in 0..gsum.len() {
        gz_off[g + 1] += gz_off[g];
    }
    for (g, &(pop, x, y)) in gsum.iter().enumerate() {
        places.push((g_rep[g].0 as usize, (x / pop) as f32, (y / pop) as f32));
    }
    let (ride_off, rides) = access_options(zones, net, p, &places, End::Home, None);
    let place_ride = |pl: usize| &rides[ride_off[pl] as usize..ride_off[pl + 1] as usize];
    let ride_ok = |st: u32, q: u16| pp.is_none() || acc.st_part.get(st as usize).map_or(true, |&sp| sp == q);
    // a group with rides is a subzone
    let mut group_sub = vec![NONE; gsum.len()];
    for g in 0..gsum.len() {
        if place_ride(zones.n + g).iter().any(|e| ride_ok(e.0, gpart[g])) {
            group_sub[g] = nsub;
            nsub += 1;
        }
    }
    for c in 0..n {
        if cell_group[c] != NONE {
            cell_sub[c] = group_sub[cell_group[c] as usize];
        }
    }
    let ns = nsub as usize;
    let mut sub_group = vec![NONE; ns - n_walk];
    for (g, &u) in group_sub.iter().enumerate() {
        if u != NONE {
            sub_group[u as usize - n_walk] = g as u32;
        }
    }
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
        home_park: vec![0.0; ns],
        acc_off: vec![0],
        acc_mid: vec![],
        acc_st: vec![],
        acc_w: vec![],
        acc_d: vec![],
        acc_t: vec![],
        home_off: vec![0],
        home_zones: vec![],
        zo_off: vec![],
        zo: vec![],
        zd_off: vec![],
        zd: vec![],
        cell: vec![],
        mem_off: vec![],
        mem: vec![],
        walk_c: (0..n).map(|c| acc.of(c).map(|i| p.access_cost_min(acc.walk[i])).fold(f32::INFINITY, f32::min)).collect(),
        part: if pp.is_some() { (0..ns).map(|s| cpart(mem[moff[s] as usize] as usize)).collect() } else { vec![] },
        st_part: acc.st_part.clone(),
    };
    // candidates: (mean perceived minutes, station, mean straight-line metres)
    let mut cand: Vec<(f32, u32, f32)> = Vec::with_capacity(32);
    for s in 0..ns {
        let m = &mem[moff[s] as usize..moff[s + 1] as usize];
        let walks = acc.n[m[0] as usize] > 0;
        let (mut wsum, mut x, mut y) = (0.0f64, 0.0f64, 0.0f64);
        for &c in m {
            let c = c as usize;
            let jobs = if walks { work_w[c] } else { 0.0 };
            let wt = (home_w[c] + jobs).max(1e-6) as f64;
            wsum += wt;
            x += wt * city.x[c] as f64;
            y += wt * city.y[c] as f64;
            sz.pop[s] += home_w[c];
            sz.jobs[s] += jobs;
            sz.home_park[s] += home_w[c] * p.home_park_cell(c);
        }
        sz.home_park[s] = if sz.pop[s] > 0.0 { sz.home_park[s] / sz.pop[s] } else { 0.0 };
        let hp = sz.home_park[s];
        sz.cx[s] = (x / wsum) as f32;
        sz.cy[s] = (y / wsum) as f32;
        if !walks {
            let g = sub_group[s - n_walk] as usize;
            sz.zone[s] = g_rep[g].0;
            sz.acc_mid.push(sz.acc_st.len() as u32);
            for &(st, w, d, t) in place_ride(zones.n + g).iter().filter(|e| ride_ok(e.0, gpart[g])) {
                sz.acc_st.push(st);
                sz.acc_w.push(w + hp);
                sz.acc_d.push(d);
                sz.acc_t.push(t);
            }
            sz.acc_off.push(sz.acc_st.len() as u32);
            sz.home_zones.extend_from_slice(&gz[gz_off[g] as usize..gz_off[g + 1] as usize]);
            sz.home_off.push(sz.home_zones.len() as u32);
            continue;
        }
        let zi = zones.cell_zone[m[0] as usize];
        sz.zone[s] = zi;
        sz.home_off.push(sz.home_zones.len() as u32);
        let q = cpart(m[0] as usize);
        let zone_rides = || place_ride(zi as usize).iter().filter(move |e| ride_ok(e.0, q));
        // candidates: union of the members' lists; distance = weighted mean over members
        cand.clear();
        for &c in m {
            for i in acc.of(c as usize) {
                let st = acc.st[i];
                if !cand.iter().any(|&(_, t, _)| t == st) {
                    cand.push((0.0, st, 0.0));
                }
            }
        }
        for e in cand.iter_mut() {
            let (sx, sy) = (net.st_x[e.1 as usize], net.st_y[e.1 as usize]);
            let ex = acc.ex.get(e.1 as usize).and_then(|v| v.as_ref());
            let bus = net.bus_only(e.1 as usize);
            let (mut wd, mut wc) = (0.0f64, 0.0f64);
            for &c in m {
                let wt = (home_w[c as usize] + work_w[c as usize]).max(1e-6) as f64;
                // T-1464: a bus stop is the member's walk to its route, beyond it cut off
                let d = if bus {
                    acc.of(c as usize).find(|&i| acc.st[i] == e.1).map_or(f32::INFINITY, |i| acc.walk[i] * p.walk_mps * 60.0 / p.walk_detour)
                } else {
                    ((city.x[c as usize] - sx).powi(2) + (city.y[c as usize] - sy).powi(2)).sqrt()
                };
                // T-275: around water where the walk must go round (cut off: no weight)
                let d = super::super::routes::walk_m(ex, c, d);
                if !d.is_finite() {
                    wd += wt * p.walk_cutoff_m as f64;
                    continue;
                }
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
        cand.retain(|e| e.2 <= p.walk_cutoff_m && e.0.is_finite());
        if cand.is_empty() {
            // every member is within reach of its own nearest station, so the mean distance to
            // that station is too; this only guards against rounding
            let c = m[0] as usize;
            let d = acc.walk[c * aw] * p.walk_mps * 60.0 / p.walk_detour;
            cand.push((p.access_cost(d), acc.st[c * aw], d));
        }
        cand.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        // K train stations and at most `BUS_SLOTS` bus stops: a bus never hides a station (T-1464)
        let (mut trains, mut buses) = (0, 0);
        for &(wm, st, d) in cand.iter() {
            if net.bus_only(st as usize) {
                if buses == BUS_SLOTS {
                    continue;
                }
                buses += 1;
            } else {
                if trains == K {
                    continue;
                }
                trains += 1;
            }
            sz.acc_st.push(st);
            sz.acc_w.push(wm);
            sz.acc_d.push(d);
            sz.acc_t.push(p.walk_min(d));
        }
        sz.acc_mid.push(sz.acc_st.len() as u32);
        if sz.pop[s] > 0.0 {
            for &(st, w, d, t) in zone_rides() {
                sz.acc_st.push(st);
                sz.acc_w.push(w + hp);
                sz.acc_d.push(d);
                sz.acc_t.push(t);
            }
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
