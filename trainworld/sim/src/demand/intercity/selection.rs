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
        self.selection(market, xy, share, day, self.solved_mask(), true).dense(&self.m)
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
    pub(super) fn selection(&self, market: u32, xy: &[f32], share: &[f32], day: u32, mask: u32, with_rail: bool) -> Selection {
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
                        let far = if end == 0 { pair.a as usize } else { m.zones[pair.h as usize].gid as usize };
                        let n = w * pair.trips as f64;
                        let legs = n * (share_out(day, everyday_purpose(k), q) + share_ret(day, everyday_purpose(k), q)) as f64;
                        let s = zones.at(far);
                        let e = &mut zones.v[s][8 + 6 * ki + 3 * end..8 + 6 * ki + 3 * end + 3];
                        e[0] += legs;
                        match per {
                            None => e[2] += legs * pair.walk as f64,
                            Some(per) => {
                                let ei = self.epair_of(i as usize);
                                let frac = if ei == u32::MAX { 0. } else { per.e_frac.get(ei as usize).copied().unwrap_or(0.) as f64 };
                                let rail = n * (OUTWARD[k][q] + RETURN[k][q]) as f64 * load_factor(day, everyday_purpose(k), q) as f64 * frac;
                                e[1] += rail;
                                e[2] -= rail * pair.walk as f64;
                            }
                        }
                    }
                }
            }
        }
        sel
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
        for q in 0..PERIODS {
            let Some(per) = self.per(q) else { continue };
            let f = load_factor(day as usize, LONGER, q) as f64;
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
    pub fn selection_rail_day(&self, market: u32, xy: &[f32], share: &[f32], day: u32) -> Vec<f64> {
        let mut sel = self.selection(market, xy, share, day, self.solved_mask(), true);
        sel.rail_only(self.m.places.len());
        sel.pack(&self.m)
    }
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
