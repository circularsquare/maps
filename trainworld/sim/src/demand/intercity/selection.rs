use super::*;

/// T-2747: a far zone's numbers in a selection's answer: (all, rail) of the longer trips starting
/// in the selection and of those ending there, the same of the commutes, then per `E_CROSS` kind
/// (legs, rail legs, walking legs) at the home end and at the activity end.
pub(super) const SEL_Z: usize = 8 + 6 * E_CROSS.len();

/// T-2747: numbers by global zone, only the zones an answer reaches.
pub(super) struct ByGid<const W: usize> {
    slot: HashMap<u32, u32>,
    gid: Vec<u32>,
    v: Vec<[f64; W]>,
}
impl<const W: usize> ByGid<W> {
    fn new() -> Self {
        ByGid { slot: HashMap::new(), gid: vec![], v: vec![] }
    }
    fn at(&mut self, g: usize) -> usize {
        let n = self.v.len() as u32;
        let s = *self.slot.entry(g as u32).or_insert(n);
        if s == n {
            self.gid.push(g as u32);
            self.v.push([0.; W]);
        }
        s as usize
    }
    fn get(&self, g: usize) -> Option<&[f64; W]> {
        self.slot.get(&(g as u32)).map(|&s| &self.v[s as usize])
    }
    /// The zone count, then per zone with any number not zero, by global zone ascending: its
    /// global zone, x, y and place (`Market::gid_xy`), its `W` numbers.
    fn pack(&self, m: &Market, out: &mut Vec<f64>) {
        let mut keep: Vec<usize> = (0..self.v.len()).filter(|&i| self.v[i].iter().any(|&x| x != 0.)).collect();
        keep.sort_unstable_by_key(|&i| self.gid[i]);
        out.reserve(1 + keep.len() * (4 + W));
        out.push(keep.len() as f64);
        for i in keep {
            let g = self.gid[i] as usize;
            out.push(g as f64);
            out.extend(m.gid_xy(g));
            out.extend(self.v[i]);
        }
    }
    #[cfg(test)]
    fn unpack(v: &[f64]) -> Self {
        let mut z = Self::new();
        for r in v[1..].chunks_exact(4 + W).take(v[0] as usize) {
            let s = z.at(r[0] as usize);
            z.v[s].copy_from_slice(&r[4..]);
        }
        z
    }
}

/// T-2747: a selection's longer trips, commutes and everyday outings: `head` holds the matched
/// share, 8 numbers per place (as `selection_flows`), the commutes 4 per place, then the trips
/// with places that have no zone, starting and ending; `zones` the far zones (`SEL_Z`).
pub(super) struct Selection {
    head: Vec<f64>,
    zones: ByGid<SEL_Z>,
}
impl Selection {
    fn empty(np: usize) -> Self {
        Selection { head: vec![0.; 3 + 12 * np], zones: ByGid::new() }
    }
    /// What the app is sent: `head`, then the zones as `ByGid::pack`.
    pub(super) fn pack(&self, m: &Market) -> Vec<f64> {
        let mut out = self.head.clone();
        self.zones.pack(m, &mut out);
        out
    }
    /// `selection_flows`'s layout: every global zone, each with its x, y and place.
    fn dense(&self, m: &Market) -> Vec<f64> {
        let (np, ng) = (m.places.len(), m.n_gid);
        let head = 1 + 8 * np;
        let tail = head + 7 * ng;
        let etail = tail + 4 * np + 4 * ng;
        let mut out = vec![0.; etail + E_CROSS.len() * 6 * ng + 2];
        let n = out.len();
        out[..head].copy_from_slice(&self.head[..head]);
        out[tail..tail + 4 * np].copy_from_slice(&self.head[head..head + 4 * np]);
        out[n - 2..].copy_from_slice(&self.head[head + 4 * np..]);
        m.write_zone_xy(&mut out[head + 4 * ng..tail]);
        for (&g, v) in self.zones.gid.iter().zip(&self.zones.v) {
            let g = g as usize;
            out[head + 4 * g..head + 4 * g + 4].copy_from_slice(&v[..4]);
            out[tail + 4 * np + 4 * g..tail + 4 * np + 4 * g + 4].copy_from_slice(&v[4..8]);
            for ki in 0..E_CROSS.len() {
                let o = etail + ki * 6 * ng + 6 * g;
                out[o..o + 6].copy_from_slice(&v[8 + 6 * ki..14 + 6 * ki]);
            }
        }
        out
    }
    /// T-409: its rail numbers alone, the rest zero. The everyday blocks hold (legs, rail legs,
    /// walkers rail takes): the last two are rail's (T-1229).
    fn rail_only(&mut self, np: usize) {
        let c = 1 + 8 * np;
        for (i, v) in self.head.iter_mut().enumerate() {
            let rail = if i == 0 {
                false
            } else if i < c {
                (i - 1) % 4 == 1
            } else if i < c + 4 * np {
                (i - c) % 2 == 1
            } else {
                false
            };
            if !rail {
                *v = 0.;
            }
        }
        for z in &mut self.zones.v {
            for (j, v) in z.iter_mut().enumerate() {
                if if j < 8 { j % 2 == 0 } else { (j - 8) % 3 == 0 } {
                    *v = 0.;
                }
            }
        }
    }
}

/// T-2747: a station's longer-trip rail riders: `head` per place (6 numbers, as `station_flows`),
/// `zones` per far zone the riders starting, then ending.
pub(super) struct StationSel {
    head: Vec<f64>,
    zones: ByGid<2>,
}
impl StationSel {
    pub(super) fn pack(&self, m: &Market) -> Vec<f64> {
        let mut out = self.head.clone();
        self.zones.pack(m, &mut out);
        out
    }
    fn dense(&self, m: &Market) -> Vec<f64> {
        let (np, ng) = (m.places.len(), m.n_gid);
        let head = 6 * np;
        let mut out = vec![0.; head + 5 * ng];
        out[..head].copy_from_slice(&self.head);
        for (&g, v) in self.zones.gid.iter().zip(&self.zones.v) {
            out[head + g as usize] = v[0];
            out[head + ng + g as usize] = v[1];
        }
        m.write_zone_xy(&mut out[head + 2 * ng..]);
        out
    }
}

impl IntercityApi {
    pub fn station_map(&self) -> Vec<u32> {
        self.svc.as_ref().map_or(vec![], |s| s.st_map.clone())
    }
    /// Per corridor (place pair): its two places, longer trips in period `q` and their rail trips.
    pub fn corridor_summary(&self, q: usize) -> Vec<f64> {
        self.per(q).map_or(vec![], |s| {
            let m = &self.m;
            let mut out: Vec<f64> = m.each_corridor().flat_map(|(a, b)| [a as f64, b as f64, 0., 0.]).collect();
            for (c, t) in m.table().with_trips() {
                out[4 * c + 2] = t * SHARES[q] as f64;
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
        self.selection(market, xy, share, &[], day, self.solved_mask(), true).dense(&self.m)
    }
    /// `station_flows` on a kind of day, every global zone listed (the app's answer is `station_flows_day`).
    pub fn station_flows_dense(&self, station: u32, day: u32) -> Vec<f64> {
        self.station_sel(station, day).map_or(vec![], |s| s.dense(&self.m))
    }
    /// T-2747: a `selection_all_day` or `selection_rail_day` answer in `selection_flows`'s layout.
    #[cfg(test)]
    pub(super) fn unpack_selection(&self, v: &[f64]) -> Vec<f64> {
        let h = 3 + 12 * self.m.places.len();
        Selection { head: v[..h].to_vec(), zones: ByGid::unpack(&v[h..]) }.dense(&self.m)
    }
    /// `unpack_selection` of a `selection_rail_day` answer, its zones' x, y and place zero (they
    /// are not rail's, so an answer and a part's rail add up to the whole).
    #[cfg(test)]
    pub(super) fn unpack_rail(&self, v: &[f64]) -> Vec<f64> {
        let (np, ng) = (self.m.places.len(), self.m.n_gid);
        let mut out = self.unpack_selection(v);
        out[1 + 8 * np + 4 * ng..1 + 8 * np + 7 * ng].fill(0.);
        out
    }
    /// T-2747: a `station_flows_day` answer in `station_flows`'s layout.
    #[cfg(test)]
    pub(super) fn unpack_station(&self, v: &[f64]) -> Vec<f64> {
        if v.is_empty() {
            return vec![];
        }
        let h = 6 * self.m.places.len();
        StationSel { head: v[..h].to_vec(), zones: ByGid::unpack(&v[h..]) }.dense(&self.m)
    }
}
impl IntercityApi {
    /// How many `table_rows` there are (each column is sized to them exactly, T-2122).
    pub(super) fn n_rows(&self) -> usize {
        *self.m.cw.rows.get_or_init(|| self.table_rows().count())
    }
    /// T-2122: the corridors `corridor_trips` or `commute_base` gives more than 0, as (corridor, its
    /// two places, its trips), in corridor order: `with_trips` and `commute` merged (each sorted by
    /// corridor), not every corridor of the world walked (T-1849).
    pub(super) fn table_rows(&self) -> impl Iterator<Item = (usize, usize, usize, f64)> + '_ {
        let cw = &*self.m.cw;
        let np = self.m.places.len();
        let commute: &[(u32, f64)] = if cw.commuters > 0. { &cw.commute } else { &[] };
        // (a: the first place of the corridor last given)
        let (mut trips, mut kc, mut a) = (self.m.table().with_trips().peekable(), 0, 0);
        std::iter::from_fn(move || loop {
            let ti = trips.peek().map_or(usize::MAX, |e| e.0);
            let ci = commute.get(kc).map_or(usize::MAX, |e| e.0 as usize);
            let i = ti.min(ci);
            if i == usize::MAX {
                return None;
            }
            let t = if ti == i { trips.next().unwrap().1 } else { 0. };
            let c = ci == i && { kc += 1; commute[kc - 1].1 > 0. };
            if t > 0. || c {
                while a + 1 < np && tri(np, a + 1, a + 1) <= i {
                    a += 1;
                }
                return Some((i, a, a + i - tri(np, a, a), t));
            }
        })
    }
    /// `table_rows` with each row's commute legs a day (`Corridors::commute_of`'s): `f(corridor, a, b, trips,
    /// commute)`, in the same order. A plain loop over the two sorted lists, for the passes over every row of a
    /// load (`table_summary`, millions of rows), where the iterator and a search a row cost a tenth (T-4182).
    pub(super) fn each_table_row(&self, mut f: impl FnMut(usize, usize, usize, f64, f64)) {
        let cw = &*self.m.cw;
        let np = self.m.places.len();
        let commute: &[(u32, f64)] = if cw.commuters > 0. { &cw.commute } else { &[] };
        // (a: the first place of the corridor last given)
        let table = self.m.table();
        let (mut kc, mut kt, mut a) = (0usize, 0usize, 0usize);
        let mut row = |i: usize, t: f64, c: f64| {
            if t > 0. || c > 0. {
                while a + 1 < np && tri(np, a + 1, a + 1) <= i {
                    a += 1;
                }
                f(i, a, a + i - tri(np, a, a), t, c);
            }
        };
        for (k, &w) in table.has_trips.iter().enumerate() {
            let mut w = w;
            while w != 0 {
                let i = 64 * k + w.trailing_zeros() as usize;
                w &= w - 1;
                let t = table.trips[kt] as f64;
                kt += 1;
                while kc < commute.len() && (commute[kc].0 as usize) < i {
                    row(commute[kc].0 as usize, 0., commute[kc].1);
                    kc += 1;
                }
                let c = if kc < commute.len() && commute[kc].0 as usize == i {
                    kc += 1;
                    commute[kc - 1].1
                } else {
                    0.
                };
                row(i, t, c);
            }
        }
        for &(i, c) in &commute[kc..] {
            row(i as usize, 0., c);
        }
    }
    /// `air`: four numbers per 600 m square holding airport terminals (T-3500): its x, y, its
    /// terminals' weight and the selected part of that weight. Airport trips from across a zone line
    /// end at the terminals, so a zone's share of them is its terminals' share selected, not its
    /// attraction's; with none given the selection holds none of them at that end.
    pub(super) fn selection(&self, market: u32, xy: &[f32], share: &[f32], air: &[f32], day: u32, mask: u32, with_rail: bool) -> Selection {
        let m = &self.m;
        let day = day as usize;
        let np = m.places.len();
        let mut sel = Selection::empty(np);
        let (head, zones) = (&mut sel.head, &mut sel.zones);
        // the commutes per place, then the trips with places that have no zone
        let (tail, away) = (1 + 8 * np, 1 + 12 * np);
        if xy.len() != 2 * share.len() || market as usize >= np {
            return sel;
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
        head[0] = if share.is_empty() { 1. } else { matched / share.len() as f64 };
        if mask == 0 {
            return sel;
        }
        let mut air_w: Vec<(usize, f64)> = {
            let mut by: HashMap<usize, (f64, f64)> = HashMap::new();
            for a in air.chunks_exact(4) {
                let key = ((a[0] / SITE_M).floor() as i32, (a[1] / SITE_M).floor() as i32, market as usize);
                let Some(c) = m.site_at(key) else { continue };
                let e = by.entry(m.cell_zone[c]).or_default();
                e.0 += a[2].max(0.) as f64;
                e.1 += a[3].max(0.) as f64;
            }
            by.into_iter().filter(|(_, (t, s))| *t > 0. && *s > 0.).map(|(z, (t, s))| (z, (s / t).min(1.))).collect()
        };
        air_w.sort_unstable_by_key(|e| e.0);
        let per_of = |q: usize| if mask >> q & 1 == 0 { None } else { Some(if with_rail { self.per(q) } else { None }) };
        // each selected zone's pairs with every other open zone (and the far zone's slot in
        // `zones`), and with each place not open
        const NONE: u32 = u32::MAX;
        let closed: Vec<usize> = (0..np).filter(|&p| m.places[p].has && !m.places[p].open).collect();
        let mut rows: HashMap<usize, (Vec<(u32, u32, u32, f64, usize)>, Vec<(usize, f64)>)> = HashMap::new();
        // T-378: each zone's trips each way with places that have no zone: its departures less
        // its trips with every place, as `Market::leaving_by` is a place's
        let mut away_z: HashMap<usize, f64> = HashMap::new();
        for z in (0..nz).filter(|&z| pop_w[z] > 0. || attr_w[z] > 0.) {
            let at = |a: usize, b: usize| self.pair_at(a, b);
            let row: Vec<_> = m.zone_row(z).into_iter().map(|(b, trips)| (b as u32, at(z, b), at(b, z), trips, zones.at(m.zones[b].gid as usize))).collect();
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
                    head[away + dir] += w * away_z[&z] * sh;
                    if !spread.is_empty() {
                        spread.fill((0., 0.));
                        for &(b, out_i, in_i, trips, _) in row {
                            let e = &mut spread[m.zones[b as usize].place];
                            e.0 += rail(per, if dir == 0 { out_i } else { in_i }) as f64;
                            e.1 += if self.solvable(z, b as usize) { trips } else { 0. };
                        }
                    }
                    for &(b, out_i, in_i, trips, s) in row {
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
                        head[o] += all;
                        head[o + 1] += r;
                        head[o + 2] += all * zb.x as f64;
                        head[o + 3] += all * zb.y as f64;
                        zones.v[s][2 * dir] += all;
                        zones.v[s][2 * dir + 1] += r;
                    }
                    for &(p, trips) in far {
                        let all = w * trips * sh * lean(dir, &m.zones[z], None, p);
                        let o = 1 + 8 * p + 4 * dir;
                        head[o] += all;
                        head[o + 2] += all * m.places[p].x as f64;
                        head[o + 3] += all * m.places[p].y as f64;
                    }
                }
            }
        }
        // T-276: the commutes between zones, per far place, then per far zone: legs a day of the
        // commuters living in the selection, by where they work, and rail legs; then the same for
        // the commuters working in the selection, by where they live
        let mut add = |far_gid: usize, side: usize, all: f64, rail: f64| {
            let p = m.gid_place[far_gid] as usize;
            head[tail + 4 * p + 2 * side] += all;
            head[tail + 4 * p + 2 * side + 1] += rail;
            let s = zones.at(far_gid);
            zones.v[s][4 + 2 * side] += all;
            zones.v[s][4 + 2 * side + 1] += rail;
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
                        if end == 1 && k == E_AIRPORT {
                            continue;
                        }
                        let far = if end == 0 { pair.a as usize } else { m.zones[pair.h as usize].gid as usize };
                        let v = self.e_legs(i as usize, w, day, q, per);
                        let s = zones.at(far);
                        let e = &mut zones.v[s][8 + 6 * ki + 3 * end..8 + 6 * ki + 3 * end + 3];
                        e[0] += v[0];
                        e[1] += v[1];
                        e[2] += v[2];
                    }
                }
            }
            // T-3500: airport trips at the airports' end, by the terminals selected
            let ki = E_CROSS.iter().position(|&c| c == E_AIRPORT).unwrap();
            let (off, items) = &m.e_act;
            for &(z, w) in &air_w {
                for &i in &items[off[z] as usize..off[z + 1] as usize] {
                    let pair = &m.everyday[i as usize];
                    if pair.kind as usize != E_AIRPORT {
                        continue;
                    }
                    let v = self.e_legs(i as usize, w, day, q, per);
                    let s = zones.at(m.zones[pair.h as usize].gid as usize);
                    let e = &mut zones.v[s][8 + 6 * ki + 3..8 + 6 * ki + 6];
                    e[0] += v[0];
                    e[1] += v[1];
                    e[2] += v[2];
                }
            }
        }
        sel
    }
    /// T-1229: everyday pair `i`'s legs in period `q` of a kind of day at weight `w`: legs, rail
    /// legs and walking legs. Without `per` (all modes) the walking legs are the pair's own with no
    /// rail; with it the rail is that network's and the walking legs the walkers rail takes, negative.
    fn e_legs(&self, i: usize, w: f64, day: usize, q: usize, per: Option<&Period>) -> [f64; 3] {
        let pair = &self.m.everyday[i];
        let k = pair.kind as usize;
        let n = w * pair.trips as f64;
        let legs = n * (share_out(day, everyday_purpose(k), q) + share_ret(day, everyday_purpose(k), q)) as f64;
        match per {
            None => [legs, 0., legs * pair.walk as f64],
            Some(per) => {
                let ei = self.epair_of(i);
                let frac = if ei == u32::MAX { 0. } else { per.e_frac.get(ei as usize).copied().unwrap_or(0.) as f64 };
                let rail = n * (OUTWARD[k][q] + RETURN[k][q]) as f64 * load_factor(day, everyday_purpose(k), q) as f64 * frac;
                [legs, rail, -(rail * pair.walk as f64)]
            }
        }
    }
    /// T-3500: the airport trips across zone lines at points `xy` (market metres) of place
    /// `market`, at end `end` (0 the homes, 1 the airports), on a kind of day over the periods of
    /// `mask`: each point's zone's legs, rail legs and walking legs (as `selection`'s blocks: all
    /// modes without `rail`, that network's rail with it, its legs then zero) times the point's
    /// weight `w` over its zone's points'. Three numbers a point.
    pub(super) fn airport_legs(&self, market: u32, end: u32, xy: &[f32], w: &[f32], day: u32, mask: u32, rail: bool) -> Vec<f64> {
        let m = &self.m;
        let (n, day) = (w.len(), day as usize);
        let mut out = vec![0.; 3 * n];
        if xy.len() != 2 * n || market as usize >= m.places.len() || end > 1 {
            return out;
        }
        let zone: Vec<Option<usize>> = (0..n)
            .map(|i| {
                let key = ((xy[2 * i] / SITE_M).floor() as i32, (xy[2 * i + 1] / SITE_M).floor() as i32, market as usize);
                m.site_at(key).map(|c| m.cell_zone[c]).filter(|_| w[i] > 0.)
            })
            .collect();
        let mut by: HashMap<usize, (f64, [f64; 3])> = HashMap::new();
        for (i, z) in zone.iter().enumerate() {
            if let Some(z) = z {
                by.entry(*z).or_default().0 += w[i] as f64;
            }
        }
        let (off, items) = if end == 0 { &m.e_home } else { &m.e_act };
        for (&z, (_, e)) in by.iter_mut() {
            for q in (0..PERIODS).filter(|&q| mask >> q & 1 == 1) {
                let per = match (rail, self.per(q)) {
                    (false, _) => None,
                    (true, Some(p)) => Some(p),
                    (true, None) => continue,
                };
                for &i in &items[off[z] as usize..off[z + 1] as usize] {
                    if m.everyday[i as usize].kind as usize != E_AIRPORT {
                        continue;
                    }
                    let v = self.e_legs(i as usize, 1., day, q, per);
                    e[0] += v[0];
                    e[1] += v[1];
                    e[2] += v[2];
                }
            }
        }
        for (i, z) in zone.iter().enumerate() {
            let Some(z) = z else { continue };
            let (t, e) = by[z];
            let f = w[i] as f64 / t;
            out[3 * i] = if rail { 0. } else { e[0] * f };
            out[3 * i + 1] = e[1] * f;
            out[3 * i + 2] = e[2] * f;
        }
        out
    }
    /// T-3503: the regional layer's legs at 600 m squares `xy` (market metres) of place `market` on a
    /// kind of day over the periods of `mask`, as the zone figures count them: with other places, and
    /// (T-3509) the longer trips and long commutes between the place's own zones; places with no zone
    /// left out. Six numbers a square: at
    /// the home end (legs leaving: longer trips starting there, commutes and everyday outings of the
    /// people living there) legs, rail legs and walking legs, then the same at the destination end
    /// (longer trips arriving, commutes of the people working there, outings coming there; airports'
    /// left out, spread by their terminals instead, `airport_legs`). A square takes its zone's legs by
    /// its site's share of the zone's people, attraction or jobs, as `selection` weighs them. `kinds`:
    /// bit 0 longer trips, bit 1 commutes, bit 2 + k everyday kind k. Without `rail` all modes and
    /// the walking with no rail; with it this network's rail legs and the walkers rail takes
    /// (negative), the legs zero.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn zone_legs(&self, market: u32, xy: &[f32], day: u32, mask: u32, kinds: u32, rail: bool) -> Vec<f64> {
        let m = &self.m;
        let (n, day) = (xy.len() / 2, day as usize);
        let mut out = vec![0f64; 6 * n];
        if market as usize >= m.places.len() {
            return out;
        }
        let nz = m.zones.len();
        // each square's zone and its shares of the zone's people, attraction and jobs
        let sq: Vec<Option<(usize, [f64; 3])>> = (0..n)
            .map(|i| {
                let key = ((xy[2 * i] / SITE_M).floor() as i32, (xy[2 * i + 1] / SITE_M).floor() as i32, market as usize);
                let c = m.site_at(key)?;
                let z = m.cell_zone[c];
                let (g, ci) = m.site(c);
                let zz = &m.zones[z];
                let jobs = if zz.jobs > 0. { ((g.city.jobs[ci] - g.city.pop[ci]) as f64 / 2.).max(0.) / zz.jobs } else { 0. };
                Some((z, [g.city.pop[ci] as f64 / zz.pop.max(1e-12), g.city.jobs[ci] as f64 / zz.w_attr.max(1e-12), jobs]))
            })
            .collect();
        let mut need = vec![false; nz];
        for &(z, _) in sq.iter().flatten() {
            need[z] = true;
        }
        // per zone: longer leaving, longer arriving, commutes of residents, of workers, outings of
        // residents, to here; three numbers each
        let mut zl = vec![[0f64; 18]; nz];
        let add = |zl: &mut Vec<[f64; 18]>, z: usize, cat: usize, v: [f64; 3]| {
            let e = &mut zl[z][3 * cat..3 * cat + 3];
            e[0] += v[0];
            e[1] += v[1];
            e[2] += v[2];
        };
        let sign = lean_sign(day);
        let lean = |dir: usize, l: f32| if sign == 0. { 1. } else { 1. + (sign * if dir == 0 { l } else { -l }) as f64 };
        let periods: Vec<usize> = (0..PERIODS).filter(|&q| mask >> q & 1 == 1 && (!rail || self.per(q).is_some())).collect();
        if kinds & 1 != 0 {
            if !rail {
                let sh: f64 = periods.iter().map(|&q| share_out(day, LONGER, q) as f64).sum();
                self.zone_longer_all(&need, sh, |z, dir, v| add(&mut zl, z, dir, [v, 0., 0.]), &lean);
            } else {
                for &q in &periods {
                    let per = self.per(q).unwrap();
                    let f = load_factor(day, LONGER, q) as f64;
                    for p in 0..self.pairs.len() {
                        let r = per.pair_rail.get(p).copied().unwrap_or(0.) as f64;
                        if r <= 0. {
                            continue;
                        }
                        let base = &self.pairs[p & !1];
                        let (a, b) = if p & 1 == 0 { (base.a as usize, base.b as usize) } else { (base.b as usize, base.a as usize) };
                        let (za, zb) = (&m.zones[a], &m.zones[b]);
                        if need[a] {
                            add(&mut zl, a, 0, [0., r * f * lean(0, m.zone_lean(za, zb)), 0.]);
                        }
                        if need[b] {
                            add(&mut zl, b, 1, [0., r * f * lean(1, m.zone_lean(zb, za)), 0.]);
                        }
                    }
                }
            }
        }
        if kinds & 2 != 0 {
            for &q in &periods {
                let legs = (share_out(day, WORK, q) + share_ret(day, WORK, q)) as f64;
                let rail_legs = (TO_WORK[q] + TO_HOME[q]) as f64 * load_factor(day, WORK, q) as f64;
                if rail {
                    let per = self.per(q).unwrap();
                    for (i, cp) in self.cpairs.iter().enumerate() {
                        let (h, wg, nc) = m.commutes[cp.c as usize];
                        let (h, w) = (h as usize, m.comp_of_gid[wg as usize]);
                        if w == u32::MAX {
                            continue;
                        }
                        let r = nc as f64 * rail_legs * per.c_frac.get(i).copied().unwrap_or(0.) as f64;
                        if need[h] {
                            add(&mut zl, h, 2, [0., r, 0.]);
                        }
                        if need[w as usize] {
                            add(&mut zl, w as usize, 3, [0., r, 0.]);
                        }
                    }
                } else {
                    for (end, (off, items)) in [(0usize, &m.c_home), (1, &m.c_work)] {
                        for z in (0..nz).filter(|&z| need[z]) {
                            for &c in &items[off[z] as usize..off[z + 1] as usize] {
                                let nc = m.commutes[c as usize].2;
                                add(&mut zl, z, 2 + end, [nc as f64 * legs, 0., 0.]);
                            }
                        }
                    }
                }
            }
        }
        if kinds >> 2 != 0 {
            let want = |k: usize| kinds >> (2 + k) & 1 == 1;
            let same = |pair: &EPair| m.gid_place[pair.a as usize] as usize == m.zones[pair.h as usize].place;
            for &q in &periods {
                if rail {
                    let per = self.per(q).unwrap();
                    for ep in &self.epairs {
                        let pair = &m.everyday[ep.e as usize];
                        let k = pair.kind as usize;
                        if !want(k) || same(pair) {
                            continue;
                        }
                        let v = self.e_legs(ep.e as usize, 1., day, q, Some(per));
                        let (h, a) = (pair.h as usize, m.comp_of_gid[pair.a as usize]);
                        if need[h] {
                            add(&mut zl, h, 4, [0., v[1], v[2]]);
                        }
                        if a != u32::MAX && need[a as usize] && k != E_AIRPORT {
                            add(&mut zl, a as usize, 5, [0., v[1], v[2]]);
                        }
                    }
                } else {
                    for (end, (off, items)) in [(0usize, &m.e_home), (1, &m.e_act)] {
                        for z in (0..nz).filter(|&z| need[z]) {
                            for &i in &items[off[z] as usize..off[z + 1] as usize] {
                                let pair = &m.everyday[i as usize];
                                let k = pair.kind as usize;
                                if !want(k) || same(pair) || (end == 1 && k == E_AIRPORT) {
                                    continue;
                                }
                                add(&mut zl, z, 4 + end, self.e_legs(i as usize, 1., day, q, None));
                            }
                        }
                    }
                }
            }
        }
        for (i, s) in sq.iter().enumerate() {
            let Some((z, [pop, attr, jobs])) = *s else { continue };
            let v = &zl[z];
            for j in 0..3 {
                out[6 * i + j] = pop * (v[j] + v[6 + j] + v[12 + j]);
                out[6 * i + 3 + j] = attr * (v[3 + j] + v[15 + j]) + jobs * v[9 + j];
            }
            if rail {
                out[6 * i] = 0.;
                out[6 * i + 3] = 0.;
            }
        }
        out
    }
    /// T-3503: every zone in `need`'s longer trips a day with its own place's other zones (T-3509)
    /// and other places' zones, all modes, at
    /// `sh` (the periods' outward shares summed): `f(zone, 0, leaving)` and `f(zone, 1, arriving)`.
    /// Its own and near places' open zones pair zone to zone as `zone_row`; every other place by the zone's share
    /// (`zone_place_eachway`), which for far places is linear in the zone's normalisers, so off a
    /// rush day their sum is worked out once for the zone's place.
    fn zone_longer_all(&self, need: &[bool], sh: f64, mut f: impl FnMut(usize, usize, f64), lean: &dyn Fn(usize, f32) -> f64) {
        let m = &self.m;
        let np = m.places.len();
        let spans = m.place_spans();
        let mut span_of = vec![None; np];
        for &(p, a, b) in &spans {
            span_of[p] = Some((a, b));
        }
        let rush = lean(0, 1.) != 1.;
        // per own place, once: its near places, its far places' rules both ways, and their sums
        // (the coefficients of a zone's g, t, a and ta)
        struct Own {
            near: Vec<usize>,
            far: Vec<(usize, PairRule, PairRule)>,
            sums: [f64; 4],
        }
        let mut own: HashMap<usize, Own> = HashMap::new();
        for (z, zz) in m.zones.iter().enumerate().filter(|(z, _)| need[*z]) {
            let i = zz.place;
            let o = own.entry(i).or_insert_with(|| {
                let mut o = Own { near: vec![], far: vec![], sums: [0.; 4] };
                for j in (0..np).filter(|&j| j != i && m.places[j].has) {
                    let r = m.pr(i, j);
                    if r.near {
                        o.near.push(j);
                        continue;
                    }
                    let (q, bk) = (&m.places[j], m.pr(j, i));
                    let (g, t) = (m.rule.ground(r.km), m.rule.tail(r.km));
                    let w = r.b * m.pull_w(r.pull, r.pull_near, r.km);
                    let wb = bk.b * m.pull_w(bk.pull, bk.pull_near, r.km);
                    o.sums[0] += w * g * q.a;
                    o.sums[1] += w * t * q.ta * r.rel;
                    o.sums[2] += wb * g * q.g;
                    o.sums[3] += wb * t * q.t * bk.rel;
                    if rush {
                        o.far.push((j, r, bk));
                    }
                }
                o
            });
            let (mut out, mut back) = (0f64, 0f64);
            // T-3509: the longer trips between the place's own zones count in its figures too
            if let Some((a, b)) = span_of[i] {
                for zb in (a..b).filter(|&w| w != z).map(|w| &m.zones[w]) {
                    let t = m.eachway(zz, zb);
                    if t >= m.rule.min_trips {
                        let l = m.zone_lean(zz, zb);
                        out += t * lean(0, l);
                        back += t * lean(1, l);
                    }
                }
            }
            for &j in &o.near {
                if let Some((a, b)) = span_of[j] {
                    for zb in &m.zones[a..b] {
                        let t = m.eachway(zz, zb);
                        if t >= m.rule.min_trips {
                            let l = m.zone_lean(zz, zb);
                            out += t * lean(0, l);
                            back += t * lean(1, l);
                        }
                    }
                } else {
                    let (t, l) = (m.zone_place_eachway(zz, j), m.zone_place_lean(zz, j));
                    out += t * lean(0, l);
                    back += t * lean(1, l);
                }
            }
            if rush {
                for (j, r, bk) in &o.far {
                    let q = &m.places[*j];
                    let t = 0.5
                        * (m.directed_by(r.b, r.rel, r.pull, r.pull_near, zz.g, zz.t, q.a, q.ta, r.km)
                            + m.directed_by(bk.b, bk.rel, bk.pull, bk.pull_near, q.g, q.t, zz.a, zz.ta, r.km));
                    let l = m.zone_place_lean(zz, *j);
                    out += t * lean(0, l);
                    back += t * lean(1, l);
                }
            } else {
                let s = &o.sums;
                let t = 0.5 * (zz.g * s[0] + zz.t * s[1] + zz.a * s[2] + zz.ta * s[3]);
                out += t;
                back += t;
            }
            f(z, 0, out * sh);
            f(z, 1, back * sh);
        }
    }
    /// T-2747: `station_flows`'s numbers, by far zone only where the station reaches; None with no
    /// such station.
    fn station_sel(&self, station: u32, day: u32) -> Option<StationSel> {
        let svc = self.svc.as_ref()?;
        let s = *svc.st_map.get(station as usize)?;
        let m = &self.m;
        let np = m.places.len();
        let mut head = vec![0.; 6 * np];
        let mut zones = ByGid::<2>::new();
        let ns = self.served.len();
        // T-1373, T-1374: a rush day's riders lean by their places and zones, the station's the
        // zone's that reaches it most
        let sign = lean_sign(day as usize);
        let here = if sign == 0. { None } else { self.station_zone(s as usize).map(|z| &m.zones[z]) };
        // T-4094: a stop of a merged group answers for its share of the group's riders
        let share = svc.fold.as_ref().map_or(1.0, |f| f.share(station as usize)) as f64;
        for q in 0..PERIODS {
            let Some(per) = self.per(q) else { continue };
            let f = load_factor(day as usize, LONGER, q) as f64 * share;
            let pat = &self.st_pat;
            if (s as usize) + 1 < pat.off.len() && per.st.len() == pat.col.len() {
                for i in pat.row(s as usize) {
                    let k = pat.col.get(i);
                    let (end, k) = if k < ns { (0, k) } else { (1, k - ns) };
                    let zone = &m.zones[self.served[k]];
                    let lean = here.map_or(0., |h| sign * if end == 0 { m.zone_lean(h, zone) } else { m.zone_lean(zone, h) });
                    let z = zones.at(zone.gid as usize);
                    zones.v[z][end] += per.st[i] as f64 * f * (1. + lean as f64);
                }
            }
        }
        for d in 0..2 {
            for &z in &self.served {
                let zone = &m.zones[z];
                let Some(v) = zones.get(zone.gid as usize) else { continue };
                let v = v[d];
                head[d * np + zone.place] += v;
                head[(2 + 2 * d) * np + zone.place] += v * zone.x as f64;
                head[(3 + 2 * d) * np + zone.place] += v * zone.y as f64;
            }
        }
        Some(StationSel { head, zones })
    }
}
#[wasm_bindgen]
impl IntercityApi {
    /// T-409: `selection_flows_day` with only its rail numbers, the rest zero: what a market beside
    /// another adds to that one's answer. Packed as `selection_all_day`.
    pub fn selection_rail_day(&self, market: u32, xy: &[f32], share: &[f32], air: &[f32], day: u32) -> Vec<f64> {
        let mut sel = self.selection(market, xy, share, air, day, self.solved_mask(), true);
        sel.rail_only(self.m.places.len());
        sel.pack(&self.m)
    }
    /// T-3500: `airport_legs` (the airport trips across zone lines at points of a place, one end).
    pub fn airport_legs_day(&self, market: u32, end: u32, xy: &[f32], w: &[f32], day: u32, mask: u32, rail: bool) -> Vec<f64> {
        self.airport_legs(market, end, xy, w, day, mask, rail)
    }
    /// T-3503: `zone_legs` (the regional layer's legs at a place's squares, both ends).
    pub fn zone_legs_day(&self, market: u32, xy: &[f32], day: u32, mask: u32, kinds: u32, rail: bool) -> Vec<f64> {
        self.zone_legs(market, xy, day, mask, kinds, rail)
    }
    /// T-3503: the everyday outings across a zone line with another place, by place, in period `q`
    /// (a weekday): for each place with any, its index, then the legs with an end there (both ends
    /// counted, all modes, the same in every part of the market), this network's rail legs, the
    /// walking legs (the walkers this network's rail takes out), the walking legs with no rail and (T-2216) the
    /// legs between two landmasses or two zones' lands (no car, no walking) and this network's rail legs of them.
    /// Empty before the period is solved.
    pub fn everyday_place_legs(&self, q: usize) -> Vec<f64> {
        let mut by = vec![[0f64; 6]; self.m.places.len()];
        if !self.each_everyday(q, |h, a, d| {
            for p in [h, a] {
                for (x, y) in by[p].iter_mut().zip(d) {
                    *x += y;
                }
            }
        }) {
            return vec![];
        }
        Self::pack_everyday(by.iter().enumerate())
    }
    /// T-4610: `everyday_place_legs` for the outings of one place alone, by the place at their other end
    /// (the same seven numbers, `place` being the partner), so a zone's bubbles can count each pair's.
    pub fn everyday_partner_legs(&self, q: usize, place: u32) -> Vec<f64> {
        let mut by = std::collections::BTreeMap::<usize, [f64; 6]>::new();
        if !self.each_everyday(q, |h, a, d| {
            let other = if h == place as usize { a } else if a == place as usize { h } else { return };
            for (x, y) in by.entry(other).or_default().iter_mut().zip(d) {
                *x += y;
            }
        }) {
            return vec![];
        }
        Self::pack_everyday(by.iter().map(|(p, v)| (*p, v)))
    }
}
impl IntercityApi {
    /// Each everyday outing across a zone line in period `q` (places `h`, `a`) as what it adds to the numbers of
    /// both its places (`everyday_place_legs`' six, in its order): once for its legs, then once for the rail's
    /// share of them. False before the period is solved.
    fn each_everyday(&self, q: usize, mut add: impl FnMut(usize, usize, [f64; 6])) -> bool {
        let Some(s) = self.per(q) else { return false };
        let m = &self.m;
        let ends = |pair: &EPair| (m.zones[pair.h as usize].place, m.gid_place[pair.a as usize] as usize);
        for pair in &m.everyday {
            let (h, a) = ends(pair);
            if h == a {
                continue;
            }
            let k = pair.kind as usize;
            let legs = pair.trips as f64 * (OUTWARD[k][q] + RETURN[k][q]) as f64;
            let apart = m.everyday_apart(pair.h, pair.a);
            let walk = legs * pair.walk as f64;
            add(h, a, [legs, 0., walk, walk, if apart { legs } else { 0. }, 0.]);
        }
        for (ei, ep) in self.epairs.iter().enumerate() {
            let pair = &m.everyday[ep.e as usize];
            let (h, a) = ends(pair);
            if h == a {
                continue;
            }
            let k = pair.kind as usize;
            let rail = pair.trips as f64 * (OUTWARD[k][q] + RETURN[k][q]) as f64 * s.e_frac.get(ei).copied().unwrap_or(0.) as f64;
            let apart = m.everyday_apart(pair.h, pair.a);
            add(h, a, [0., rail, -rail * pair.walk as f64, 0., 0., if apart { rail } else { 0. }]);
        }
        true
    }
    fn pack_everyday<'a>(by: impl Iterator<Item = (usize, &'a [f64; 6])>) -> Vec<f64> {
        let mut out = vec![];
        for (p, v) in by.filter(|(_, v)| v[0] > 0.) {
            out.extend([p as f64, v[0], v[1], v[2].max(0.), v[3], v[4], v[5]]);
        }
        out
    }
}
#[wasm_bindgen]
impl IntercityApi {
    /// `station_flows` on a kind of day (T-260): each period weighed by the market's load factor.
    /// T-2747: packed, the 6 numbers per place, then the far zones the station reaches as
    /// `selection_all_day`'s, each with the riders starting, then ending; empty with no such station.
    pub fn station_flows_day(&self, station: u32, day: u32) -> Vec<f64> {
        self.station_sel(station, day).map_or(vec![], |s| s.pack(&self.m))
    }
    /// `station_flows_day` of a station this network does not hold but the whole one does (T-326).
    pub fn no_station_flows(&self) -> Vec<f64> {
        StationSel { head: vec![0.; 6 * self.m.places.len()], zones: ByGid::new() }.pack(&self.m)
    }
    pub fn load_of_crush(&self, q: usize) -> Vec<f32> {
        self.per(q).map_or(vec![], |s| {
            let v: Vec<f32> = s.avg
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
                .collect();
            self.svc().out_load(v)
        })
    }
    /// The network solved on; a period's result implies one.
    pub(crate) fn svc(&self) -> &crate::demand::api::Service {
        self.svc.as_ref().expect("a solved period has a network")
    }
}
