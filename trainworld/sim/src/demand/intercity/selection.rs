use super::*;

impl IntercityApi {
    pub fn station_map(&self) -> Vec<u32> {
        self.svc.as_ref().map_or(vec![], |s| s.st_map.clone())
    }
    /// Per corridor (place pair): its two places, longer trips in period `q` and their rail trips.
    pub fn corridor_summary(&self, q: usize) -> Vec<f64> {
        self.per(q).map_or(vec![], |s| {
            let m = &self.m;
            let mut out: Vec<f64> = m.each_corridor().flat_map(|(a, b)| [a as f64, b as f64, 0., 0.]).collect();
            for (&c, &t) in m.cw.trips_at.iter().zip(&m.cw.trips) {
                out[4 * c as usize + 2] = t * SHARES[q] as f64;
            }
            for (k, c) in m.open_corridors() {
                out[4 * c + 3] = s.corridors.get(k);
            }
            out
        })
    }
    /// T-1008: the corridors' two places each, in `corridor_summary`'s order. With
    /// `corridor_trips`, `commute_base` and `corridor_modes` the part of the corridors no network or
    /// period changes, which the main thread is sent once.
    pub fn corridor_pairs(&self) -> Vec<u32> {
        self.m.each_corridor().flat_map(|(a, b)| [a as u32, b as u32]).collect()
    }
    /// T-1008: per corridor, the commutes between zones that `commute_corridors` counts, before the
    /// period's legs (`corridor_shares(q)[1]`); empty where it gives none.
    pub fn commute_base(&self) -> Vec<f64> {
        let m = &self.m;
        if m.cw.commuters <= 0. {
            return vec![];
        }
        let mut out = vec![0f64; m.n_corridors()];
        for &(c, n) in &m.cw.commute {
            out[c as usize] = n;
        }
        out
    }
    pub fn exits(&self, q: usize) -> Vec<f32> {
        self.per(q).map_or(vec![], |s| s.avg.exit.clone())
    }
    /// `selection_flows` on a kind of day (T-260, `days.rs`): that day's trips and period shares.
    pub fn selection_flows_day(&self, market: u32, xy: &[f32], share: &[f32], day: u32) -> Vec<f64> {
        self.selection(market, xy, share, day, self.solved_mask(), true)
    }
}
impl IntercityApi {
    /// How many `table_rows` there are (each column is sized to them exactly, T-2122).
    pub(super) fn n_rows(&self) -> usize {
        *self.m.cw.rows.get_or_init(|| self.table_rows().count())
    }
    /// T-2122: the corridors `corridor_trips` or `commute_base` gives more than 0, as (corridor, its
    /// two places, its trips), in corridor order: `trips_at` and `commute` merged (each sorted by
    /// corridor), not every corridor of the world walked (T-1849).
    pub(super) fn table_rows(&self) -> impl Iterator<Item = (usize, usize, usize, f64)> + '_ {
        let cw = &*self.m.cw;
        let np = self.m.places.len();
        let commute: &[(u32, f64)] = if cw.commuters > 0. { &cw.commute } else { &[] };
        // (a: the first place of the corridor last given)
        let (mut kt, mut kc, mut a) = (0, 0, 0);
        std::iter::from_fn(move || loop {
            let ti = cw.trips_at.get(kt).map_or(usize::MAX, |&i| i as usize);
            let ci = commute.get(kc).map_or(usize::MAX, |e| e.0 as usize);
            let i = ti.min(ci);
            if i == usize::MAX {
                return None;
            }
            let t = if ti == i { kt += 1; cw.trips[kt - 1] } else { 0. };
            let c = ci == i && { kc += 1; commute[kc - 1].1 > 0. };
            if t > 0. || c {
                while a + 1 < np && tri(np, a + 1, a + 1) <= i {
                    a += 1;
                }
                return Some((i, a, a + i - tri(np, a, a), t));
            }
        })
    }
    pub(super) fn selection(&self, market: u32, xy: &[f32], share: &[f32], day: u32, mask: u32, with_rail: bool) -> Vec<f64> {
        let m = &self.m;
        let day = day as usize;
        let (np, ng) = (m.places.len(), m.n_gid);
        let head = 1 + 8 * np;
        let etail = head + 7 * ng + 4 * np + 4 * ng;
        let mut out = vec![0.; etail + E_CROSS.len() * 6 * ng + 2];
        let away = out.len() - 2;
        m.write_zone_xy(&mut out[head + 4 * ng..head + 7 * ng]);
        if xy.len() != 2 * share.len() || market as usize >= np {
            return out;
        }
        let nz = m.zones.len();
        let (mut pop_w, mut attr_w, mut job_w) = (vec![0f64; nz], vec![0f64; nz], vec![0f64; nz]);
        let mut matched = 0.;
        for (i, &f) in share.iter().enumerate() {
            let key = ((xy[2 * i] / SITE_M).floor() as i32, (xy[2 * i + 1] / SITE_M).floor() as i32, market as usize);
            let Some(c) = m.site_at(key) else { continue };
            let f = (f as f64).clamp(0., 1.);
            matched += 1.;
            let z = m.cell_zone[c];
            let (g, ci) = m.site(c);
            let zz = &m.zones[z];
            pop_w[z] += f * g.city.pop[ci] as f64 / zz.pop.max(1e-12);
            attr_w[z] += f * g.city.jobs[ci] as f64 / zz.w_attr.max(1e-12);
            if zz.jobs > 0. {
                job_w[z] += f * ((g.city.jobs[ci] - g.city.pop[ci]) as f64 / 2.).max(0.) / zz.jobs;
            }
        }
        out[0] = if share.is_empty() { 1. } else { matched / share.len() as f64 };
        if mask == 0 {
            return out;
        }
        let per_of = |q: usize| if mask >> q & 1 == 0 { None } else { Some(if with_rail { self.per(q) } else { None }) };
        // each selected zone's pairs with every other open zone, and with each place not open
        const NONE: u32 = u32::MAX;
        let closed: Vec<usize> = (0..np).filter(|&p| m.places[p].has && !m.places[p].open).collect();
        let mut rows: HashMap<usize, (Vec<(u32, u32, u32, f64)>, Vec<(usize, f64)>)> = HashMap::new();
        // T-378: each zone's trips each way with places that have no zone: its departures less
        // its trips with every place, as `Market::leaving_by` is a place's
        let mut away_z: HashMap<usize, f64> = HashMap::new();
        for z in (0..nz).filter(|&z| pop_w[z] > 0. || attr_w[z] > 0.) {
            let at = |a: usize, b: usize| self.pair_at(a, b);
            let row: Vec<_> = m.zone_row(z).into_iter().map(|(b, trips)| (b as u32, at(z, b), at(b, z), trips)).collect();
            let far: Vec<_> = closed.iter().map(|&p| (p, m.zone_place_eachway(&m.zones[z], p))).filter(|(_, t)| *t > 0.).collect();
            let known = row.iter().map(|r| r.3).sum::<f64>() + far.iter().map(|f| f.1).sum::<f64>();
            away_z.insert(z, (m.zones[z].dep - known).max(0.));
            rows.insert(z, (row, far));
        }
        let rail = |per: Option<&Period>, i: u32| match per {
            Some(p) if i != NONE => p.pair_rail[i as usize],
            _ => 0.,
        };
        // T-780: with sampled pairs a zone's rail with a place is its sampled pairs', spread over the
        // place's zones it could ride to by their trips, so no far zone shows more rail than trips
        let mut spread = vec![(0f64, 0f64); if self.chunk > 0. { np } else { 0 }];
        // T-1373, T-1374: a rush day's trips lean by their places and zones, from here (dir 0) or to
        // here (dir 1); the far end a zone (`Some`) or a place with no pack open
        let sign = lean_sign(day);
        let lean = |dir: usize, here: &Zone, there: Option<&Zone>, p: usize| -> f64 {
            if sign == 0. {
                return 1.;
            }
            let l = match there {
                Some(b) => m.zone_lean(here, b),
                None => m.zone_place_lean(here, p),
            };
            1. + (sign * if dir == 0 { l } else { -l }) as f64
        };
        for q in 0..PERIODS {
            let Some(per) = per_of(q) else { continue };
            let sh = share_out(day, LONGER, q) as f64;
            let f = load_factor(day, LONGER, q) as f64;
            for (dir, weights) in [(0usize, &pop_w), (1usize, &attr_w)] {
                for (z, &w) in weights.iter().enumerate().filter(|(_, w)| **w > 0.) {
                    let (row, far) = &rows[&z];
                    out[away + dir] += w * away_z[&z] * sh;
                    if !spread.is_empty() {
                        spread.fill((0., 0.));
                        for &(b, out_i, in_i, trips) in row {
                            let e = &mut spread[m.zones[b as usize].place];
                            e.0 += rail(per, if dir == 0 { out_i } else { in_i }) as f64;
                            e.1 += if self.solvable(z, b as usize) { trips } else { 0. };
                        }
                    }
                    for &(b, out_i, in_i, trips) in row {
                        let zb = &m.zones[b as usize];
                        let k = lean(dir, &m.zones[z], Some(zb), zb.place);
                        let all = w * trips * sh * k;
                        let pair_r = match spread.get(zb.place) {
                            Some(&(r, t)) if t > 0. && self.solvable(z, b as usize) => (r * trips / t) as f32,
                            Some(_) => 0.,
                            None => rail(per, if dir == 0 { out_i } else { in_i }),
                        };
                        let r = w * pair_r as f64 * f * k;
                        let o = 1 + 8 * zb.place + 4 * dir;
                        out[o] += all;
                        out[o + 1] += r;
                        out[o + 2] += all * zb.x as f64;
                        out[o + 3] += all * zb.y as f64;
                        out[head + 4 * zb.gid as usize + 2 * dir] += all;
                        out[head + 4 * zb.gid as usize + 2 * dir + 1] += r;
                    }
                    for &(p, trips) in far {
                        let all = w * trips * sh * lean(dir, &m.zones[z], None, p);
                        let o = 1 + 8 * p + 4 * dir;
                        out[o] += all;
                        out[o + 2] += all * m.places[p].x as f64;
                        out[o + 3] += all * m.places[p].y as f64;
                    }
                }
            }
        }
        // T-276: the commutes between zones, per far place, then per far zone: legs a day of the
        // commuters living in the selection, by where they work, and rail legs; then the same for
        // the commuters working in the selection, by where they live
        let tail = head + 7 * ng;
        let mut add = |far_gid: usize, side: usize, all: f64, rail: f64| {
            let p = m.gid_place[far_gid] as usize;
            out[tail + 4 * p + 2 * side] += all;
            out[tail + 4 * p + 2 * side + 1] += rail;
            let o = tail + 4 * np + 4 * far_gid + 2 * side;
            out[o] += all;
            out[o + 1] += rail;
        };
        for q in 0..PERIODS {
            let Some(per) = per_of(q) else { continue };
            let legs = (share_out(day, WORK, q) + share_ret(day, WORK, q)) as f64;
            let rail_legs = (TO_WORK[q] + TO_HOME[q]) as f64 * load_factor(day, WORK, q) as f64;
            let frac = |c: u32| match (per, self.cpair_of(c as usize)) {
                (Some(per), i) if i != u32::MAX => per.c_frac.get(i as usize).copied().unwrap_or(0.) as f64,
                _ => 0.,
            };
            for (end, weights) in [(0usize, &pop_w), (1usize, &job_w)] {
                let (off, items) = if end == 0 { &m.c_home } else { &m.c_work };
                for (z, &w) in weights.iter().enumerate().filter(|(_, w)| **w > 0.) {
                    for &c in &items[off[z] as usize..off[z + 1] as usize] {
                        let (h, wg, n) = m.commutes[c as usize];
                        let far = if end == 0 { wg as usize } else { m.zones[h as usize].gid as usize };
                        let n = w * n as f64;
                        add(far, end, n * legs, n * rail_legs * frac(c));
                    }
                }
            }
        }
        // T-1229: the everyday outings across a zone line, by purpose and far zone (home end:
        // outings of the selection's residents by where they go; activity end: by where they live):
        // legs a day, rail legs and walking legs. Without rail (`with_rail` off) the walking legs are
        // the pairs' own, and with it the walkers rail takes, negative.
        for q in 0..PERIODS {
            let Some(per) = per_of(q) else { continue };
            for (end, weights) in [(0usize, &pop_w), (1usize, &attr_w)] {
                let (off, items) = if end == 0 { &m.e_home } else { &m.e_act };
                for (z, &w) in weights.iter().enumerate().filter(|(_, w)| **w > 0.) {
                    for &i in &items[off[z] as usize..off[z + 1] as usize] {
                        let pair = &m.everyday[i as usize];
                        let k = pair.kind as usize;
                        let Some(ki) = E_CROSS.iter().position(|&c| c == k) else { continue };
                        let far = if end == 0 { pair.a as usize } else { m.zones[pair.h as usize].gid as usize };
                        let n = w * pair.trips as f64;
                        let legs = n * (share_out(day, everyday_purpose(k), q) + share_ret(day, everyday_purpose(k), q)) as f64;
                        let o = etail + ki * 6 * ng + 6 * far + 3 * end;
                        out[o] += legs;
                        match per {
                            None => out[o + 2] += legs * pair.walk as f64,
                            Some(per) => {
                                let ei = self.epair_of(i as usize);
                                let frac = if ei == u32::MAX { 0. } else { per.e_frac.get(ei as usize).copied().unwrap_or(0.) as f64 };
                                let rail = n * (OUTWARD[k][q] + RETURN[k][q]) as f64 * load_factor(day, everyday_purpose(k), q) as f64 * frac;
                                out[o + 1] += rail;
                                out[o + 2] -= rail * pair.walk as f64;
                            }
                        }
                    }
                }
            }
        }
        out
    }
}
#[wasm_bindgen]
impl IntercityApi {
    /// T-409: `selection_flows_day` with only its rail numbers, the rest zero: what a market beside
    /// another adds to that one's answer.
    pub fn selection_rail_day(&self, market: u32, xy: &[f32], share: &[f32], day: u32) -> Vec<f64> {
        let mut out = self.selection_flows_day(market, xy, share, day);
        let (np, ng) = (self.m.places.len(), self.m.n_gid);
        let (head, tail) = (1 + 8 * np, 1 + 8 * np + 7 * ng);
        let etail = tail + 4 * np + 4 * ng;
        let away = out.len() - 2;
        // T-1229: the everyday pairs' blocks hold (legs, rail legs, walkers rail takes): the last
        // two are rail's
        let rail = |i: usize| {
            if i == 0 || i >= away {
                false
            } else if i < head {
                (i - 1) % 4 == 1
            } else if i < head + 4 * ng {
                (i - head) % 2 == 1
            } else if i < tail {
                false
            } else if i < etail {
                (i - tail) % 2 == 1
            } else {
                (i - etail) % 3 != 0
            }
        };
        for (i, v) in out.iter_mut().enumerate() {
            if !rail(i) {
                *v = 0.;
            }
        }
        out
    }
    /// `station_flows` on a kind of day (T-260): each period weighed by the market's load factor.
    pub fn station_flows_day(&self, station: u32, day: u32) -> Vec<f64> {
        let Some(svc) = &self.svc else { return vec![] };
        let Some(&s) = svc.st_map.get(station as usize) else {
            return vec![];
        };
        let m = &self.m;
        let (np, ng) = (m.places.len(), m.n_gid);
        let head = 6 * np;
        let mut out = vec![0.; head + 5 * ng];
        let ns = self.served.len();
        // T-1373, T-1374: a rush day's riders lean by their places and zones, the station's the
        // zone's that reaches it most
        let sign = lean_sign(day as usize);
        let here = if sign == 0. { None } else { self.station_zone(s as usize).map(|z| &m.zones[z]) };
        for q in 0..PERIODS {
            let Some(per) = self.per(q) else { continue };
            let f = load_factor(day as usize, LONGER, q) as f64;
            let pat = &self.st_pat;
            if (s as usize) + 1 < pat.off.len() && per.st.len() == pat.col.len() {
                for i in pat.row(s as usize) {
                    let k = pat.col.get(i);
                    let (end, k) = if k < ns { (0, k) } else { (ng, k - ns) };
                    let zone = &m.zones[self.served[k]];
                    let lean = here.map_or(0., |h| sign * if end == 0 { m.zone_lean(h, zone) } else { m.zone_lean(zone, h) });
                    out[head + end + zone.gid as usize] += per.st[i] as f64 * f * (1. + lean as f64);
                }
            }
        }
        for d in 0..2 {
            for &z in &self.served {
                let zone = &m.zones[z];
                let v = out[head + d * ng + zone.gid as usize];
                out[d * np + zone.place] += v;
                out[(2 + 2 * d) * np + zone.place] += v * zone.x as f64;
                out[(3 + 2 * d) * np + zone.place] += v * zone.y as f64;
            }
        }
        m.write_zone_xy(&mut out[head + 2 * ng..]);
        out
    }
    /// `station_flows` of a station this network does not hold but the whole one does (T-326).
    pub fn no_station_flows(&self) -> Vec<f64> {
        let (np, ng) = (self.m.places.len(), self.m.n_gid);
        let head = 6 * np;
        let mut out = vec![0.; head + 5 * ng];
        self.m.write_zone_xy(&mut out[head + 2 * ng..]);
        out
    }
    pub fn load_of_crush(&self, q: usize) -> Vec<f32> {
        self.per(q).map_or(vec![], |s| {
            s.avg
                .seg
                .iter()
                .enumerate()
                .map(|(r, &v)| {
                    let l = s.g.rn_line[r] as usize;
                    let cap = 60. / s.net.line_headway_min[l][q] * s.net.line_crush[l];
                    if cap > 0. {
                        v / s.p.period_hours * s.p.peak_hour_factor / cap
                    } else {
                        0.
                    }
                })
                .collect()
        })
    }
}
