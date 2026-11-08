use super::*;

impl Market {
    /// T-378: corridor (a, b)'s shares of its longer trips not by rail that fly and that drive
    /// (`IntercityApi::corridor_modes`). A pair of open places from the table; the others' modes
    /// alone worked out (T-1204), what the two ways share once (`pair_ways`, T-1849), the direct
    /// flights from `directs`. T-1277: a plane's fare weighs the departing country's, so the two ways
    /// can differ; the split is their mean.
    pub(super) fn corridor_mode(&self, directs: &Directs, a: usize, b: usize) -> [f32; 2] {
        let (car, x, y) = if self.rules.slot[a] != u32::MAX && self.rules.slot[b] != u32::MAX {
            let r = self.pr(a, b);
            (r.car, r.air, self.pr(b, a).air)
        } else {
            let (p, q) = (&self.places[a], &self.places[b]);
            let km = if a == b {
                0.
            } else if self.ground {
                chord_km(&p.u, &q.u)
            } else {
                ((p.x - q.x).hypot(p.y - q.y) / 1000.) as f64
            };
            let near = km < self.rule.near_km;
            let direct = |x: usize, y: usize| directs.has(a, x, b) || directs.has(b, y, a);
            let r = pair_ways(&self.places, &self.countries, &self.drives, a, b, near, km, PairRule { near, km, ..PairRule::default() }, direct);
            (r.car, r.flown(p, q, &self.mp), r.flown(q, p, &self.mp))
        };
        let air = 0.5 * (x + y);
        [air, if car { 1. - air } else { 0. }]
    }
    /// T-1373: the lean of trips from place `i` to place `j` on a going-out day: positive from the
    /// larger to the smaller (`Place::lean_size`), the opposite of `j` to `i`'s; their trips are
    /// times `1 + lean` going out and `1 - lean` coming back, so a pair's trips in all are kept.
    #[inline]
    pub(super) fn lean(&self, i: usize, j: usize) -> f32 {
        let (a, b) = (self.places[i].lean_size(), self.places[j].lean_size());
        if a + b > 0. { LEAN * ((a - b) / (a + b)) as f32 } else { 0. }
    }
    /// T-1374: the rush's whole lean of trips from place `i`, touristy `ti`, to place `j`, touristy
    /// `tj`: the places' (`lean`) and towards the more touristy end; still the opposite of the way
    /// back's, so a pair's trips in all are kept.
    #[inline]
    pub(super) fn rush_lean(&self, i: usize, ti: f32, j: usize, tj: f32) -> f32 {
        (self.lean(i, j) + TOUR_LEAN * (tj - ti)).clamp(-LEAN_MAX, LEAN_MAX)
    }
    /// `rush_lean` from zone `a` to zone `b`.
    #[inline]
    pub(super) fn zone_lean(&self, a: &Zone, b: &Zone) -> f32 {
        self.rush_lean(a.place, a.tour, b.place, b.tour)
    }
    /// `rush_lean` from zone `a` to place `p` (a place with no pack open counts as not touristy).
    #[inline]
    pub(super) fn zone_place_lean(&self, a: &Zone, p: usize) -> f32 {
        self.rush_lean(a.place, a.tour, p, self.places[p].tour)
    }
    /// T-1374: each open zone's and place's touristiness from the attractions (longitude, latitude,
    /// visitors a year): an attraction's visitors go to the zone of the nearest site within
    /// `TOUR_REACH` squares of it, of any open place; one with none is in no open place.
    pub(super) fn set_tourism(&mut self, attractions: &[[f64; 3]]) {
        if attractions.is_empty() || !self.ground {
            return;
        }
        let pad = (TOUR_REACH as f64 + 1.) * SITE_M as f64;
        // the nearest site so far of each attraction: squared metres, global site
        let mut best = vec![(f32::INFINITY, u32::MAX); attractions.len()];
        for g in &self.groups {
            let (Some(f), Some(&s0)) = (&g.frame, self.cell_zone.get(g.lo)) else { continue };
            if g.city.x.is_empty() {
                continue;
            }
            let place = self.zones[s0].place;
            let lo = |v: &[f32]| v.iter().fold(f32::INFINITY, |a, &b| a.min(b)) as f64 - pad;
            let hi = |v: &[f32]| v.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)) as f64 + pad;
            let (x0, x1, y0, y1) = (lo(&g.city.x), hi(&g.city.x), lo(&g.city.y), hi(&g.city.y));
            for (k, a) in attractions.iter().enumerate() {
                let (x, y) = f.from_lonlat(a[0], a[1]);
                if !(x >= x0 && x <= x1 && y >= y0 && y <= y1) {
                    continue;
                }
                let (cx, cy) = ((x / SITE_M as f64).floor() as i32, (y / SITE_M as f64).floor() as i32);
                for dx in -TOUR_REACH..=TOUR_REACH {
                    for dy in -TOUR_REACH..=TOUR_REACH {
                        let Some(s) = self.site_at((cx + dx, cy + dy, place)) else { continue };
                        let (sg, i) = self.site(s);
                        let d = ((sg.city.x[i] as f64 - x).powi(2) + (sg.city.y[i] as f64 - y).powi(2)) as f32;
                        if d < best[k].0 {
                            best[k] = (d, s as u32);
                        }
                    }
                }
            }
        }
        let mut visits = vec![0f64; self.zones.len()];
        for (k, &(_, s)) in best.iter().enumerate() {
            if s != u32::MAX && attractions[k][2].is_finite() && attractions[k][2] > 0. {
                visits[self.cell_zone[s as usize]] += attractions[k][2];
            }
        }
        let mut by_place: HashMap<usize, (f64, f64)> = HashMap::new();
        for (z, zz) in self.zones.iter_mut().enumerate() {
            zz.tour = touristy(visits[z], zz.pop);
            let e = by_place.entry(zz.place).or_default();
            e.0 += visits[z];
            e.1 += zz.pop;
        }
        for (p, (v, pop)) in by_place {
            self.places[p].tour = touristy(v, pop);
        }
    }
    #[inline(always)]
    pub(super) fn zone_km(&self, a: &Zone, b: &Zone) -> f64 {
        if self.ground {
            chord_km(&a.u, &b.u)
        } else {
            ((a.x - b.x).hypot(a.y - b.y) / 1000.) as f64
        }
    }
    /// T-2424: zone pair (a, b)'s routed ratios (minutes, road km): its places' (`r`, their pair
    /// rule), times a's own detour towards b's place and b's towards a's, so zones round a lake's
    /// end or across its widest part drive their own road.
    pub(super) fn zone_ratios(&self, r: &PairRule, a: &Zone, b: &Zone) -> (f32, f32) {
        if !r.car || a.place == b.place {
            return (r.rt, r.rk);
        }
        let k = |z: &Zone| z.gid - self.places[z.place].off;
        let (ta, ka) = self.drives.zone(a.place, k(a), b.place);
        let (tb, kb) = self.drives.zone(b.place, k(b), a.place);
        (r.rt * ta * tb, r.rk * ka * kb)
    }
    /// The ordered pair (i, j)'s rule: from the table when both places are open, else worked out
    /// (T-1204, T-1558).
    #[inline(always)]
    pub(super) fn pr(&self, i: usize, j: usize) -> PairRule {
        let r = &self.rules;
        let (si, sj) = (r.slot[i], r.slot[j]);
        if si != u32::MAX && sj != u32::MAX {
            r.out[si as usize * r.open.len() + sj as usize]
        } else {
            pair_rule(&self.places, &self.countries, &self.rule, self.ground, &self.mp, &self.drives, i, j)
        }
    }
    /// Trips a day from zone-like sums `(g, t)` at place `i` to `(a, ta)` at place `j`, `d` km.
    #[inline(always)]
    pub(super) fn directed(&self, i: usize, j: usize, g: f64, t: f64, a: f64, ta: f64, d: f64) -> f64 {
        let r = self.pr(i, j);
        self.directed_by(r.b, r.rel, g, t, a, ta, d)
    }
    /// `directed` with the pair's border and income factors given.
    #[inline(always)]
    pub(super) fn directed_by(&self, b: f64, rel: f64, g: f64, t: f64, a: f64, ta: f64, d: f64) -> f64 {
        let ground = if g > 0. { g * self.rule.ground(d) * a } else { 0. };
        let tail = if t > 0. { t * self.rule.tail(d) * ta * rel } else { 0. };
        b * (ground + tail)
    }
    /// A 6 km zone pair's trips a day each way: the mean of the two directions. Near places' zones
    /// at their own distance, far places' at their centres' (so their pairs add up to the places').
    #[inline(always)]
    pub(super) fn eachway(&self, a: &Zone, b: &Zone) -> f64 {
        let r = self.pr(a.place, b.place);
        let d = if r.near { self.zone_km(a, b) } else { r.km };
        (self.directed(a.place, b.place, a.g, a.t, b.a, b.ta, d) + self.directed(b.place, a.place, b.g, b.t, a.a, a.ta, d)) * 0.5
    }
    /// `eachway` with the curves at the pair's distance given (`curves`), so a run of far zones works
    /// them out once (T-780): the same figure to the bit.
    #[inline(always)]
    pub(super) fn eachway_with(&self, a: &Zone, b: &Zone, (gc, tc): (f64, f64)) -> f64 {
        let one = |r: PairRule, g: f64, t: f64, at: f64, ta: f64| r.b * ((if g > 0. { g * gc * at } else { 0. }) + if t > 0. { t * tc * ta * r.rel } else { 0. });
        (one(self.pr(a.place, b.place), a.g, a.t, b.a, b.ta) + one(self.pr(b.place, a.place), b.g, b.t, a.a, a.ta)) * 0.5
    }
    /// The ground and tail curves of zones a and b (`eachway_with`).
    #[inline(always)]
    pub(super) fn curves(&self, a: &Zone, b: &Zone) -> (f64, f64) {
        let r = self.pr(a.place, b.place);
        let d = if r.near { self.zone_km(a, b) } else { r.km };
        (self.rule.ground(d), self.rule.tail(d))
    }
    /// Two places' trips a day each way: near ones from the rows (every zone pair added up),
    /// far ones from their centres.
    pub(super) fn place_eachway(&self, i: usize, j: usize) -> f64 {
        let r = self.pr(i, j);
        if r.near {
            return self.near.get(&(i.min(j) as u32, i.max(j) as u32)).copied().unwrap_or(0.);
        }
        let (p, q) = (&self.places[i], &self.places[j]);
        (self.directed(i, j, p.g, p.t, q.a, q.ta, r.km) + self.directed(j, i, q.g, q.t, p.a, p.ta, r.km)) * 0.5
    }
    /// A zone's trips a day each way with a place that is not open: far ones exactly; near ones
    /// the places' trips by the zone's share of its place's departures and attraction (judgement,
    /// T-377: the place's zones are not known here).
    pub(super) fn zone_place_eachway(&self, z: &Zone, j: usize) -> f64 {
        let r = self.pr(z.place, j);
        if r.near {
            let p = &self.places[z.place];
            let share = 0.5 * (if p.g > 0. { z.g / p.g } else { 0. } + if p.a > 0. { z.a / p.a } else { 0. });
            return self.place_eachway(z.place, j) * share;
        }
        let (q, back) = (&self.places[j], self.pr(j, z.place));
        (self.directed_by(r.b, r.rel, z.g, z.t, q.a, q.ta, r.km) + self.directed_by(back.b, back.rel, q.g, q.t, z.a, z.ta, r.km)) * 0.5
    }
    /// The corridor of places i and j (either order): pairs a <= b numbered a then b.
    pub(super) fn corridor(&self, i: usize, j: usize) -> usize {
        tri(self.places.len(), i.min(j), i.max(j))
    }
    pub(super) fn n_corridors(&self) -> usize {
        let np = self.places.len();
        np * (np + 1) / 2
    }
    /// T-1204: the corridor of two open places among the open places' corridors (the periods'
    /// rail figures are kept for those alone), in the same order as `corridor`.
    #[inline]
    pub(super) fn open_corridor(&self, i: usize, j: usize) -> usize {
        let (a, b) = (self.rules.slot[i] as usize, self.rules.slot[j] as usize);
        tri(self.rules.open.len(), a.min(b), a.max(b))
    }
    pub(super) fn n_open_corridors(&self) -> usize {
        let k = self.rules.open.len();
        k * (k + 1) / 2
    }
    /// The open places' corridors: (index among them, corridor), in corridor order.
    pub(super) fn open_corridors(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        let (np, open) = (self.places.len(), &self.rules.open);
        (0..open.len()).flat_map(move |a| (a..open.len()).map(move |b| tri(np, open[a], open[b]))).enumerate()
    }
    /// Every corridor's two places, in corridor order.
    pub(super) fn each_corridor(&self) -> impl Iterator<Item = (usize, usize)> {
        let np = self.places.len();
        (0..np).flat_map(move |a| (a..np).map(move |b| (a, b)))
    }
    /// Site `s` (global): its group's city and its index there.
    #[inline(always)]
    pub(super) fn site(&self, s: usize) -> (&Group, usize) {
        let g = match self.groups.binary_search_by(|g| g.lo.cmp(&s)) {
            Ok(i) => i,
            Err(i) => i - 1,
        };
        (&self.groups[g], s - self.groups[g].lo)
    }
    pub(super) fn n_sites(&self) -> usize {
        self.cell_zone.len()
    }
    /// Site `s`'s 600 m square and place (`site_at`'s key).
    pub(super) fn site_key_of(&self, s: usize) -> (i32, i32, usize) {
        let (g, i) = self.site(s);
        ((g.city.x[i] / SITE_M).floor() as i32, (g.city.y[i] / SITE_M).floor() as i32, self.zones[self.cell_zone[s]].place)
    }
    /// The first site in a 600 m square (x, y) of a place.
    pub(super) fn site_at(&self, key: (i32, i32, usize)) -> Option<usize> {
        let v = &self.site_key.0;
        let i = v.partition_point(|&s| self.site_key_of(s as usize) < key);
        v.get(i).map(|&s| s as usize).filter(|&s| self.site_key_of(s) == key)
    }
    /// T-1204: the corridors' figures (`Corridors`): each corridor's longer trips a day (as
    /// `place_eachway` gives them, its pair factors worked out here rather than kept), the
    /// commutes on it from the rows (`commute_rows`, by place) or else from the composition's.
    pub(super) fn corridors_of(&self, commute_rows: Option<&[Vec<(u32, f64)>]>) -> Corridors {
        let np = self.places.len();
        let mut commute: BTreeMap<u32, f64> = BTreeMap::new();
        let mut commuters = 0.;
        match commute_rows {
            Some(rows) => {
                for (p, list) in rows.iter().enumerate() {
                    for &(j, n) in list {
                        if (j as usize) < np && n.is_finite() && n >= 0. {
                            *commute.entry(self.corridor(p, j as usize) as u32).or_insert(0.) += 2. * n;
                            commuters += n;
                        }
                    }
                }
            }
            None => {
                for &(h, w, n) in &self.commutes {
                    let c = self.corridor(self.zones[h as usize].place, self.gid_place[w as usize] as usize);
                    *commute.entry(c as u32).or_insert(0.) += 2. * n as f64;
                    commuters += n as f64;
                }
            }
        }
        // what `pair_trip_rule` reads of each place, once a place: the border factor's general
        // part, and whether its country is in the open area (country ids name one country each)
        let rule = &self.rule;
        let general: Vec<f64> = self.places.iter().map(|p| rule.border * p.inc.powf(rule.border_eps)).collect();
        let open_area: Vec<bool> = self.countries.iter().map(|c| OPEN_AREA.contains(&c.as_str())).collect();
        let factors = |i: usize, j: usize| {
            let (p, q) = (&self.places[i], &self.places[j]);
            let b = if p.country == q.country {
                1.
            } else if open_area[p.country as usize] && open_area[q.country as usize] {
                OPEN_AREA_BORDER.max(general[i])
            } else {
                general[i]
            };
            // a ratio of 1 or more is held at 1, and 1 to any power is 1
            let rel = if p.inc > 0. {
                let x = (q.inc / p.inc).min(1.);
                if x == 1. { 1. } else { x.powf(rule.eps_t) }
            } else {
                1.
            };
            (b, rel)
        };
        // (corridor order is this loop's order, so the lists come sorted; sized for every pair of places
        // with zones, nearly all of which have trips, so they never grow by doubling)
        let has = self.places.iter().filter(|p| p.has).count();
        let (mut at, mut ct) = (Vec::with_capacity(has * (has + 1) / 2), Vec::with_capacity(has * (has + 1) / 2));
        let mut to_places = vec![0f64; np];
        let mut pairs = 0usize;
        for i in 0..np {
            let p = &self.places[i];
            if !p.has {
                continue;
            }
            for j in i..np {
                let q = &self.places[j];
                if !q.has {
                    continue;
                }
                let km = if i == j {
                    0.
                } else if self.ground {
                    chord_km(&p.u, &q.u)
                } else {
                    ((p.x - q.x).hypot(p.y - q.y) / 1000.) as f64
                };
                let t = if km < rule.near_km {
                    self.near.get(&(i as u32, j as u32)).copied().unwrap_or(0.)
                } else {
                    let ((bi, ri), (bj, rj)) = (factors(i, j), factors(j, i));
                    (self.directed_by(bi, ri, p.g, p.t, q.a, q.ta, km) + self.directed_by(bj, rj, q.g, q.t, p.a, p.ta, km)) * 0.5
                };
                if t > 0. {
                    at.push(tri(np, i, j) as u32);
                    ct.push(2. * t);
                    // a place's own pairs both start there
                    to_places[i] += t;
                    to_places[j] += t;
                    pairs += if i == j { 1 } else { 2 };
                }
            }
        }
        let leaving_by: Vec<f64> = (0..np).map(|p| if self.places[p].has { (self.places[p].dep - to_places[p]).max(0.) } else { 0. }).collect();
        at.shrink_to_fit();
        ct.shrink_to_fit();
        Corridors { id: next_corridors_id(), daily: ct.iter().fold(0., |s, &t| s + t), leaving: leaving_by.iter().sum(), trips_at: at, trips: ct, commute: commute.into_iter().collect(), commuters, leaving_by, place_pairs: pairs, rows: Default::default() }
    }
    /// Each open place's run of zones (zones are laid out place by place): (place, first, end).
    pub(super) fn place_spans(&self) -> Vec<(usize, usize, usize)> {
        let mut spans: Vec<(usize, usize, usize)> = vec![];
        for (z, zz) in self.zones.iter().enumerate() {
            match spans.last_mut() {
                Some(s) if s.0 == zz.place => s.2 = z + 1,
                _ => spans.push((zz.place, z, z + 1)),
            }
        }
        spans
    }
    /// Every global zone's position (x, y in the output frame) and place: open ones their own,
    /// others their place's centre.
    pub(super) fn write_zone_xy(&self, out: &mut [f64]) {
        for g in 0..self.n_gid {
            out[3 * g..3 * g + 3].copy_from_slice(&self.gid_xy(g));
        }
    }
    /// `write_zone_xy` of one global zone.
    pub(super) fn gid_xy(&self, g: usize) -> [f64; 3] {
        let c = self.comp_of_gid[g];
        let (x, y) = if c != u32::MAX {
            let z = &self.zones[c as usize];
            (z.x, z.y)
        } else {
            let p = &self.places[self.gid_place[g] as usize];
            (p.x, p.y)
        };
        [x as f64, y as f64, self.gid_place[g] as f64]
    }
    /// Zone `z`'s pairs with every other open zone (trips each way), partner ascending.
    #[inline(never)]
    pub(super) fn zone_row(&self, z: usize) -> Vec<(usize, f64)> {
        let (a, min) = (&self.zones[z], self.rule.min_trips);
        let mut out = vec![];
        for (b, zb) in self.zones.iter().enumerate() {
            if b == z {
                continue;
            }
            let t = self.eachway(a, zb);
            if t >= min {
                out.push((b, t));
            }
        }
        out
    }
}
