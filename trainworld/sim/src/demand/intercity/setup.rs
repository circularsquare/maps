use super::*;

impl IntercityApi {
    /// T-1286: each station pair's expected fare less the base (`FareX::dist`) on the strategies towards the
    /// stations `want`ed that `strategy` has; empty while no one needs them (`need_km`).
    pub(super) fn pair_km<'s>(&self, g: &Graph, want: &[bool], strategy: impl Fn(usize) -> Option<Strategy<'s>>) -> Vec<f32> {
        if !self.need_km() {
            return vec![];
        }
        let s = g.n_st;
        let mut km = vec![0f32; s * s];
        let (mut col, mut kw, mut dep) = (vec![], vec![], vec![]);
        for dst in (0..s).filter(|&d| want[d]) {
            if let Some(st) = strategy(dst) {
                self.fare_col(g, st, dst, &mut col, &mut kw, &mut dep);
                for src in 0..s {
                    km[src * s + dst] = col[src];
                }
            }
        }
        km
    }
    /// The fares' weighing (`FareX`) on `km` (`pair_km` on `g`), none while it is empty.
    pub(super) fn fare_x<'x>(&self, p: &Params, km: &'x [f32]) -> Option<FareX<'x>> {
        (!km.is_empty()).then(|| FareX { dist: km, base: p.fare_min })
    }
    /// What `rail_logsum_x` takes for trips from `zone` by riders weighing money at `f` times its
    /// country's (T-1431, an income group's): none where the factor comes to 1.
    pub(super) fn fx_of<'x>(&self, fx: &'x Option<FareX<'x>>, zone: usize, f: f32) -> Option<(&'x FareX<'x>, f32)> {
        let d = self.fare_d(zone, f);
        fx.as_ref().filter(|_| d != 0.).map(|x| (x, d))
    }
    /// The extra weight `d` on the fares of trips from `zone` (they weigh `1 + d` times the market's).
    pub(super) fn fare_d(&self, zone: usize, f: f32) -> f32 {
        (1. + self.zone_fx.get(zone).copied().unwrap_or(0.)) * f - 1.
    }
    /// Whether the stations' expected fare km are needed (`pair_km`): a fare factor set somewhere,
    /// or a place's riders in income groups (T-1431).
    pub(super) fn need_km(&self) -> bool {
        !self.zone_fx.is_empty() || self.m.grouped
    }
    /// `pair_km`'s column towards `dst`, into `col` by node: each one's expected fare there less the
    /// base, perceived minutes: its fare km at the rate plus the curve's extra, times the trip's
    /// km-weighted multiplier (T-1626). `kw`: scratch.
    pub(super) fn fare_col(&self, g: &Graph, st: Strategy, dst: usize, col: &mut Vec<f32>, kw: &mut Vec<f32>, dep: &mut Vec<(u32, f32)>) {
        trip_km_on(g, st, dst, &self.fare_km, col, dep);
        let weighed = !g.rn_kmw.is_empty();
        if weighed {
            trip_km_on(g, st, dst, &g.rn_kmw, kw, dep);
        }
        for (i, v) in col.iter_mut().enumerate() {
            let k = *v;
            *v = (self.fare_per_km * k + g.fare_curve.extra(k)) * if weighed { fare_mul_of(k, kw[i]) } else { 1. };
        }
    }
    /// T-1229: each served everyday pair's kind, home place, rail legs in period `q` and the
    /// walking legs those riders were taken from (the period's average over crowding rounds).
    pub(super) fn each_e_rail(&self, s: &Period, q: usize, mut f: impl FnMut(usize, usize, f64, f64)) {
        for (ei, ep) in self.epairs.iter().enumerate() {
            let pair = &self.m.everyday[ep.e as usize];
            let k = pair.kind as usize;
            let rail = pair.trips as f64 * (OUTWARD[k][q] + RETURN[k][q]) as f64 * s.e_frac.get(ei).copied().unwrap_or(0.) as f64;
            f(k, self.m.zones[pair.h as usize].place, rail, rail * pair.walk as f64);
        }
    }
    /// T-1175: an everyday pair's home zone and activity zone (open zones).
    pub(super) fn epair_zones(&self, ep: &EPairOn) -> (usize, usize) {
        let pair = &self.m.everyday[ep.e as usize];
        (pair.h as usize, self.m.comp_of_gid[pair.a as usize] as usize)
    }
    pub(super) fn per(&self, q: usize) -> Option<&Period> {
        self.per.get(q).and_then(|v| v.as_ref())
    }
    /// T-1373: the zone whose people reach (merged) station `s` with the most weight.
    pub(super) fn station_zone(&self, s: usize) -> Option<usize> {
        let mut best: Option<(f32, usize)> = None;
        for (z, list) in self.access.iter().enumerate() {
            for &(st, w) in list {
                if st == s && best.map_or(true, |b| w > b.0) {
                    best = Some((w, z));
                }
            }
        }
        best.map(|b| b.1)
    }
    /// The index in `pairs` of the pair from zone a to zone b, `u32::MAX` if none.
    pub(super) fn pair_at(&self, a: usize, b: usize) -> u32 {
        let key = (a.min(b) as u32, a.max(b) as u32);
        let at = |m: usize| (self.pairs[2 * m].a, self.pairs[2 * m].b);
        let n = self.pairs.len() / 2;
        let (mut k, mut hi) = (0, n);
        while k < hi {
            let m = (k + hi) / 2;
            if at(m) < key { k = m + 1 } else { hi = m }
        }
        if a != b && k < n && at(k) == key { (2 * k + (a > b) as usize) as u32 } else { u32::MAX }
    }
    /// Whether zones a and b could make a pair (`solved_pairs` with every pair solved).
    pub(super) fn solvable(&self, a: usize, b: usize) -> bool {
        let m = &self.m;
        let at = |z: usize| self.served_at.get(z).copied().unwrap_or(u32::MAX);
        let (i, k) = (at(a), at(b));
        let (za, zb) = (&m.zones[a], &m.zones[b]);
        a != b && i != u32::MAX && k != u32::MAX && m.pr(za.place, zb.place).has_alt() && m.eachway(za, zb) >= m.rule.min_trips && self.served_comp[i as usize].iter().any(|c| self.served_comp[k as usize].binary_search(c).is_ok())
    }
    /// The pairs between the `served` zones (ascending) joined by track, with a car or a plane to
    /// compete with, in the whole market's order (T-174, T-409, T-377). `comp` is `served_comp`.
    pub(super) fn set_served(&mut self, served: Vec<usize>, comp: Vec<Vec<u32>>) {
        let m = self.m.clone();
        // T-1735: with a window, only the pairs with an end in it
        let win = std::mem::take(&mut self.win);
        let inw = |z: usize| win.is_empty() || win[z];
        let joined = |i: usize, k: usize| (inw(served[i]) || inw(served[k])) && comp[i].iter().any(|c| comp[k].binary_search(c).is_ok());
        self.pairs = vec![];
        let (any, each) = solved_pairs(&m, &served, &joined, self.chunk, self.far_chunks);
        let mut pairs = Vec::with_capacity(2 * each.len());
        let (mut bit, mut next) = (0, each.into_iter());
        for (i, &a) in served.iter().enumerate() {
            let za = &m.zones[a];
            for &b in &served[i + 1..] {
                let zb = &m.zones[b];
                bit += 1;
                if any[(bit - 1) / 64] >> ((bit - 1) % 64) & 1 == 0 {
                    continue;
                }
                let (r, back) = (m.pr(za.place, zb.place), m.pr(zb.place, za.place));
                let trips = next.next().unwrap();
                let d = m.zone_km(za, zb) as f32;
                let corridor = m.open_corridor(za.place, zb.place) as u32;
                let (a, b) = (a as u32, b as u32);
                let band = self.base.band_of_d2(d * d * 1e6) as u8;
                let zr = m.zone_ratios(&r, za, zb);
                pairs.push(Pair { a, b, trips, corridor, car: r.zone_car(d, zb.park, zr), band });
                pairs.push(Pair { a: b, b: a, trips, corridor, car: back.zone_car(d, za.park, zr), band });
            }
        }
        self.pairs = pairs;
        let mut at = vec![u32::MAX; m.zones.len()];
        for (k, &z) in served.iter().enumerate() {
            at[z] = k as u32;
        }
        let pc = commute_params();
        self.cpairs.clear();
        for (c, &(h, wg, _)) in m.commutes.iter().enumerate() {
            let w = m.comp_of_gid[wg as usize];
            if w == u32::MAX {
                continue;
            }
            let (h, w) = (h as usize, w as usize);
            if at[h] == u32::MAX || at[w] == u32::MAX || !joined(at[h] as usize, at[w] as usize) {
                continue;
            }
            let (zh, zw) = (&m.zones[h], &m.zones[w]);
            let km = m.zone_km(zh, zw) as f32;
            let r = m.pr(zh.place, zw.place);
            let (rt, rk) = m.zone_ratios(&r, zh, zw);
            let car = drive_min(km) * rt * pc.w_drive + zw.park + pc.car_own_min() + km * ROAD_DETOUR * rk * pc.car_own_km_min() + r.cross.at(FARE_MIN_PER_USD);
            let corridor = m.open_corridor(zh.place, zw.place);
            let band = pc.band_of_d2(km * km * 1e6) as u8;
            self.cpairs.push(CPair { c: c as u32, car, corridor, ride: km >= pc.ride_min_km, band });
        }
        self.cpairs.shrink_to_fit();
        // T-1175: the everyday pairs, as the commutes
        self.epairs.clear();
        for (e, ep) in m.everyday.iter().enumerate() {
            let a = m.comp_of_gid[ep.a as usize];
            if a == u32::MAX {
                continue;
            }
            let (h, a) = (ep.h as usize, a as usize);
            if at[h] == u32::MAX || at[a] == u32::MAX || !joined(at[h] as usize, at[a] as usize) {
                continue;
            }
            let (zh, za) = (&m.zones[h], &m.zones[a]);
            let km = m.zone_km(zh, za) as f32;
            let r = m.pr(zh.place, za.place);
            let (rt, rk) = m.zone_ratios(&r, zh, za);
            let car = drive_min(km) * rt * pc.w_drive + za.park + pc.car_own_min() + km * ROAD_DETOUR * rk * pc.car_own_km_min() + r.cross.at(FARE_MIN_PER_USD);
            let w = ep.walk.min(0.999);
            self.epairs.push(EPairOn { e: e as u32, car, walk_e: w / (1. - w), ride: km >= pc.ride_min_km, band: pc.band_of_d2(km * km * 1e6) as u8 });
        }
        self.epairs.shrink_to_fit();
        self.served_at = at;
        self.served = served;
        self.served_comp = comp;
        self.win = win;
    }
    /// T-1735: the window from `win_in` on `svc`'s stations: every zone with one of the stations in
    /// an access list, and the zones given; then the zones by their nearest station's minutes by
    /// rail from those, nearest first, within the minutes and the share given. True if it
    /// changed (the pairs must be set up again).
    pub(super) fn refresh_window(&mut self, svc: &Service) -> bool {
        let Some((mask, zones, reach, share, floor)) = &self.win_in else { return false };
        let (nzz, ns) = (self.m.zones.len(), svc.st_x.len());
        let mut on = vec![false; ns];
        for (i, &v) in mask.iter().enumerate() {
            if v != 0 && i < svc.st_map.len() {
                on[svc.st_map[i] as usize] = true;
            }
        }
        let lists = [&self.access, &self.access_work, &self.access_c, &self.access_cw, &self.access_c_walk, &self.access_ew];
        let stations_of = |z: usize| lists.iter().filter_map(move |l| l.get(z)).flatten().map(|e| e.0);
        let mut win = vec![false; nzz];
        for &z in zones {
            if (z as usize) < nzz {
                win[z as usize] = true;
            }
        }
        for (z, w) in win.iter_mut().enumerate() {
            *w |= stations_of(z).any(|st| on[st]);
        }
        if *reach > 0. && on.iter().any(|&v| v) {
            let p = Params { period: WINDOW_PERIOD, period_share: SHARES[WINDOW_PERIOD], period_hours: (PERIOD_HOURS[WINDOW_PERIOD].1 - PERIOD_HOURS[WINDOW_PERIOD].0) as f32, peak_hour_factor: PEAK_HOUR_FACTOR[WINDOW_PERIOD], ..self.base.clone() };
            let g = build_graph(&DemandApi::period_net(svc, WINDOW_PERIOD), &p);
            let d = rail_minutes(&g, &on);
            let mut near: Vec<(f32, usize)> = (0..nzz).filter(|&z| !win[z]).map(|z| (stations_of(z).map(|st| d[st]).fold(f32::INFINITY, f32::min), z)).filter(|e| e.0 <= *reach).collect();
            near.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
            let budget = ((*share as f64 * self.want.iter().filter(|&&w| w).count() as f64) as usize).max(*floor as usize);
            let mut held = vec![false; ns];
            let mut n = 0;
            let mut hold = |z: usize, held: &mut [bool]| {
                for st in stations_of(z) {
                    if !held[st] {
                        held[st] = true;
                        n += 1;
                    }
                }
                n
            };
            for z in (0..nzz).filter(|&z| win[z]) {
                hold(z, &mut held);
            }
            for (_, z) in near {
                // (a zone's own stations counted before it is taken: the last may pass the budget)
                if hold(z, &mut held) > budget {
                    break;
                }
                win[z] = true;
            }
        }
        if win == self.win {
            return false;
        }
        self.win = win;
        self.per.iter_mut().for_each(|p| *p = None);
        true
    }
    /// The index in `cpairs` of the market's commute `c`, `u32::MAX` if none (`cpairs` are in
    /// commute order: a search, not a table the size of every commute a cluster's part would hold).
    pub(crate) fn cpair_of(&self, c: usize) -> u32 {
        self.cpairs.binary_search_by_key(&(c as u32), |p| p.c).map_or(u32::MAX, |i| i as u32)
    }
    /// The same for the market's everyday pairs in `epairs`.
    pub(crate) fn epair_of(&self, e: usize) -> u32 {
        self.epairs.binary_search_by_key(&(e as u32), |p| p.e).map_or(u32::MAX, |i| i as u32)
    }
    /// Every pair of the market, as the served pairs would be with every zone served.
    #[cfg(test)]
    pub(super) fn all_pairs(&self) -> Vec<Pair> {
        let mut a = IntercityApi::on(self.m.clone());
        let all: Vec<usize> = (0..self.m.zones.len()).collect();
        let comp = vec![vec![0u32]; all.len()];
        a.set_served(all, comp);
        a.pairs
    }
    /// Station access by zone and the zone pairs a station serves, for `svc`'s layout. Each group
    /// of sites sees the stations in its own frame.
    pub(super) fn set_access(&mut self, svc: &Service) {
        let m = self.m.clone();
        let net0 = DemandApi::period_net(svc, 0);
        let nzz = m.zones.len();
        let mut weights: Vec<BTreeMap<usize, f32>> = (0..nzz).map(|_| BTreeMap::new()).collect();
        let mut work = weights.clone();
        // T-1175: the everyday pairs' home ends walk and ride as the commutes' do
        let commuting = !m.commutes.is_empty() || !m.everyday.is_empty();
        let outings = !m.everyday.is_empty();
        let (mut cw_home, mut cw_work, mut cw_walk) = (weights.clone(), weights.clone(), weights.clone());
        let mut ew = weights.clone();
        let beta_c = commute_params().beta;
        // T-718: walks against rides to a station at the station choice's scale of the longest
        // band, where nearly every trip between zones falls
        let theta = self.base.band_unsoftened(self.base.n_bands() - 1);
        let mut walks: Vec<(usize, f32)> = Vec::with_capacity(K);
        let comp = station_components(&build_graph(&net0, &self.base));
        let on_train = on_train(&net0);
        for g in &m.groups {
            // the stations in this group's frame
            let mut net = net0.clone();
            if let (Some(a), Some(f)) = (&m.anchor, &g.frame) {
                if a != f {
                    for s in 0..net.st_x.len() {
                        let (lon, lat) = a.to_lonlat(net.st_x[s] as f64, net.st_y[s] as f64);
                        let (x, y) = f.from_lonlat(lon, lat);
                        net.st_x[s] = x as f32;
                        net.st_y[s] = y as f32;
                    }
                    for c in &mut net.catches {
                        for p in &mut c.pts {
                            let (lon, lat) = a.to_lonlat(p[0] as f64, p[1] as f64);
                            let (x, y) = f.from_lonlat(lon, lat);
                            *p = [x as f32, y as f32];
                        }
                    }
                }
            }
            let acc = build_access(&g.city, &net, &self.base);
            // T-373: rides from each site's 2 km zone
            // T-1511: and a ride to each track component the cheapest lines miss
            let track = Some((&comp[..], &on_train[..]));
            let (h_off, h_rides) = access_options(&g.kz, &net, &self.base, &g.kplace, End::Home, track);
            let (f_off, f_rides) = access_options(&g.kz, &net, &self.base, &g.kplace, End::Far, track);
            for c in 0..g.city.len() {
                let z = m.cell_zone[g.lo + c];
                let zz = &m.zones[z];
                let w = g.city.pop[c] / zz.pop.max(1e-12) as f32;
                let jw = g.city.jobs[c] / zz.w_attr.max(1e-12) as f32;
                let cjw = ((g.city.jobs[c] - g.city.pop[c]) / 2.).max(0.) / zz.jobs.max(1e-12) as f32;
                let kz = g.kz.cell_zone[c] as usize;
                let home = &h_rides[h_off[kz] as usize..h_off[kz + 1] as usize];
                let far = &f_rides[f_off[kz] as usize..f_off[kz + 1] as usize];
                walks.clear();
                for i in acc.of(c) {
                    let walk = acc.walk[i];
                    let cost = walk.min(self.base.walk_easy_min) * self.base.w_walk + (walk - self.base.walk_easy_min).max(0.) * self.base.w_walk_far;
                    walks.push((acc.st[i] as usize, cost));
                }
                add_access(&mut weights[z], &walks, home, self.base.beta, theta, w);
                add_access(&mut work[z], &walks, far, self.base.beta, theta, jw);
                if commuting {
                    add_access(&mut cw_home[z], &walks, home, beta_c, theta, w);
                    add_access(&mut cw_walk[z], &walks, &[], beta_c, theta, w);
                    add_access(&mut cw_work[z], &walks, &[], beta_c, theta, cjw);
                }
                if outings {
                    add_access(&mut ew[z], &walks, &[], beta_c, theta, jw);
                }
            }
        }
        let lists = |maps: Vec<BTreeMap<usize, f32>>, positive: bool| -> Vec<Vec<(usize, f32)>> { maps.into_iter().map(|mm| access_list(mm, positive, &comp, &on_train)).collect() };
        self.access = lists(weights, false);
        self.access_work = lists(work, false);
        self.access_c = if commuting { lists(cw_home, true) } else { vec![] };
        self.access_cw = if commuting { lists(cw_work, true) } else { vec![] };
        self.access_c_walk = if commuting { lists(cw_walk, true) } else { vec![] };
        self.access_ew = if outings { lists(ew, true) } else { vec![] };
        self.want = vec![false; svc.st_x.len()];
        for list in self.access_work.iter().chain(&self.access_cw).chain(&self.access_c).chain(&self.access_c_walk).chain(&self.access_ew) {
            for &(st, _) in list {
                self.want[st] = true;
            }
        }
        // T-174: only pairs between zones a station serves can carry a rail trip
        let served: Vec<usize> = (0..nzz).filter(|&z| !self.access[z].is_empty() || !self.access_work[z].is_empty()).collect();
        // T-409: and only pairs whose zones reach one track component
        let comp: Vec<Vec<u32>> = served
            .iter()
            .map(|&z| {
                let mut v: Vec<u32> = [&self.access, &self.access_work, &self.access_c, &self.access_cw, &self.access_c_walk, &self.access_ew].iter().filter_map(|l| l.get(z)).flatten().map(|&(st, _)| comp[st]).collect();
                v.sort_unstable();
                v.dedup();
                v
            })
            .collect();
        let moved = self.refresh_window(svc);
        if moved || served != self.served || comp != self.served_comp {
            self.set_served(served, comp);
        }
        self.set_st_pattern(svc.st_x.len());
    }
    /// The far-zone table's pattern (`StPattern`) for the access lists and pairs as they stand:
    /// every entry `demand` can add to.
    pub(super) fn set_st_pattern(&mut self, s: usize) {
        let nz = self.served.len();
        let words = (2 * nz).div_ceil(64);
        let mut bits = vec![0u64; s * words];
        let mut set = |st: usize, c: usize| bits[st * words + c / 64] |= 1 << (c % 64);
        let col = &self.served_at;
        for pair in &self.pairs {
            let (ca, cb) = (nz + col[pair.a as usize] as usize, col[pair.b as usize] as usize);
            for &(sa, _) in &self.access[pair.a as usize] {
                set(sa, cb);
            }
            for &(sb, _) in &self.access_work[pair.b as usize] {
                set(sb, ca);
            }
        }
        for cp in &self.cpairs {
            let (h, wg, _) = self.m.commutes[cp.c as usize];
            let (h, wz) = (h as usize, self.m.comp_of_gid[wg as usize] as usize);
            let home = if cp.ride { &self.access_c[h] } else { &self.access_c_walk[h] };
            let (hc, wc) = (col[h] as usize, col[wz] as usize);
            for &(sa, _) in home {
                set(sa, wc);
                set(sa, nz + wc);
            }
            for &(sb, _) in &self.access_cw[wz] {
                set(sb, nz + hc);
                set(sb, hc);
            }
        }
        for ep in &self.epairs {
            let (h, az) = self.epair_zones(ep);
            let home = if ep.ride { &self.access_c[h] } else { &self.access_c_walk[h] };
            let (hc, ac) = (col[h] as usize, col[az] as usize);
            for &(sa, _) in home {
                set(sa, ac);
                set(sa, nz + ac);
            }
            for &(sb, _) in &self.access_ew[az] {
                set(sb, nz + hc);
                set(sb, hc);
            }
        }
        let rows = || bits.chunks_exact(words.max(1)).take(s);
        let mut off = Vec::with_capacity(s + 1);
        off.push(0u32);
        for row in rows() {
            off.push(off.last().unwrap() + row.iter().map(|b| b.count_ones()).sum::<u32>());
        }
        off.resize(s + 1, *off.last().unwrap());
        let cols = rows().flat_map(|row| {
            row.iter().enumerate().flat_map(|(k, &b)| {
                let mut b = b;
                std::iter::from_fn(move || {
                    (b != 0).then(|| {
                        let c = 64 * k as u32 + b.trailing_zeros();
                        b &= b - 1;
                        c
                    })
                })
            })
        });
        let n = *off.last().unwrap() as usize;
        let col = if 2 * nz <= u16::MAX as usize {
            let mut v = Vec::with_capacity(n);
            v.extend(cols.map(|c| c as u16));
            Cols::Short(v)
        } else {
            let mut v = Vec::with_capacity(n);
            v.extend(cols);
            Cols::Long(v)
        };
        self.st_pat = StPattern::new(off, col, nz);
    }
    pub(super) fn set_network_in(&mut self, st_xy: &[f32], line_n: &[u32], stops: &[u32], times: &[f32], tph: &[f32], cars: &[f32]) -> Result<f64, String> {
        let t = now_ms();
        let mut off = vec![0];
        for &n in line_n {
            off.push(off.last().unwrap() + n)
        }
        if st_xy.len() % 2 != 0 || stops.len() != *off.last().unwrap() as usize || times.len() != 6 * stops.len() || tph.len() != 3 * cars.len() || cars.len() != line_n.len() || stops.iter().any(|&s| s as usize >= st_xy.len() / 2) {
            return Err("intercity network arrays disagree".into());
        }
        let (st_x, st_y, st_map) = merge_stations(st_xy, &off, stops, self.base.station_merge_m);
        let group = demand_groups(&st_map, st_x.len(), &self.complex_in);
        let ups = demand_ups(&st_map, st_x.len(), &self.ups_in);
        let (catches, st_bus) = demand_catches(&st_map, st_x.len(), stops, &self.bus_in.0, &self.bus_in.1);
        let svc = Service {
            catches,
            st_bus,
            st_x,
            st_y,
            st_map: st_map.clone(),
            line_off: off,
            stops: stops.iter().map(|&s| st_map[s as usize]).collect(),
            stop_xy: stops.iter().flat_map(|&s| [st_xy[2 * s as usize], st_xy[2 * s as usize + 1]]).collect(),
            times: times.to_vec(),
            tph: tph.chunks_exact(3).map(|v| [v[0], v[1], v[2]]).collect(),
            cars: cars.to_vec(),
            group,
            span: span_for(&self.span_in, stops.len()),
            ups,
            passages: demand_passages(&st_map, &self.passages_in),
        };
        // T-124: as `DemandApi::set_network`, access is kept with the layout and a period with
        // its level's inputs
        let old = self.svc.as_ref();
        let same = old.map_or(0, |o| svc.same_periods(o));
        if !old.is_some_and(|o| svc.same_layout(o)) {
            self.set_access(&svc);
        } else if self.refresh_window(&svc) {
            self.set_served(self.served.clone(), self.served_comp.clone());
            self.set_st_pattern(svc.st_x.len());
        }
        for (q, p) in self.per.iter_mut().enumerate() {
            match p {
                Some(st) if same >> q & 1 == 1 => st.net = DemandApi::period_net(&svc, q),
                _ => *p = None,
            }
        }
        self.svc = Some(svc);
        Ok(now_ms() - t)
    }
}
