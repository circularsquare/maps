use super::*;

/// T-2653: tests turn the far-zone table's by-column filling off, to compare a pass without it.
pub(super) static ST_BY_ZONE_OFF: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

impl IntercityApi {
    pub(super) fn on(m: Rc<Market>) -> Self {
        IntercityApi {
            m,
            pairs: vec![],
            served: vec![],
            served_at: vec![],
            base: market_params(),
            svc: None,
            access: vec![],
            access_work: vec![],
            access_c: vec![],
            access_cw: vec![],
            access_c_walk: vec![],
            cpairs: vec![],
            access_ew: vec![],
            epairs: vec![],
            want: vec![],
            per: (0..PERIODS).map(|_| None).collect(),
            fare_km: vec![],
            fare_mul: vec![],
            fare_per_km: 0.,
            fare_curve: vec![],
            zone_fx: vec![],
            complex_in: vec![],
            ups_in: vec![],
            passages_in: vec![],
            span_in: vec![],
            bus_in: (vec![], vec![]),
            chunk: 0.,
            far_chunks: true,
            served_comp: vec![],
            st_pat: StPattern::default(),
            win: vec![],
            win_in: None,
        }
    }
    pub(super) fn pass(&self, g: &Graph, p: &Params, q: usize, ix: &XferIndex) -> Pass {
        if !self.win.is_empty() {
            return self.pass_searched(g, p, q, ix);
        }
        let todo = self.want.iter().filter(|&&w| w).count();
        if todo * strategy_bytes(g) <= MARKET_STRATEGY_BYTES || skims_cached(g, p) {
            // the strategies are shared with the local populations on the same service (T-119)
            return with_skims(g, p, &self.want, |sk| self.pass_on(g, p, q, sk, ix));
        }
        self.pass_searched(g, p, q, ix)
    }
    pub(super) fn pass_on(&self, g: &Graph, p: &Params, q: usize, sk: &Skims, ix: &XferIndex) -> Pass {
        self.pass_window(g, p, q, ix, Columns::Skims(sk), &self.win)
    }
    /// `pass_on` with each destination searched when the window first needs it (T-761, T-1545), so
    /// no table of every destination's strategy is held.
    pub(super) fn pass_searched(&self, g: &Graph, p: &Params, q: usize, ix: &XferIndex) -> Pass {
        self.pass_searched_in(g, p, q, ix, &self.win)
    }
    /// `pass_searched` in a window of zones `win` (T-1735; empty: none).
    pub(super) fn pass_searched_in(&self, g: &Graph, p: &Params, q: usize, ix: &XferIndex, win: &[bool]) -> Pass {
        // the market's part solves last in its cluster, so the tables the cities searched for this
        // period and round have no further use, and a search for the market shares none of them
        clear_skim_cache();
        self.pass_window(g, p, q, ix, Columns::Search(DestSearch::new(g, p)), win)
    }
    /// The market's rail trips in period `q` and their loads, a window of destination zones at a
    /// time (T-1545). A zone's pairs (the longer trips' ending there, the commutes' working there,
    /// the everyday outings' going there) read the route costs towards its own stations alone and
    /// fill those stations' columns of trips, and the return legs the home stations'. The zones go
    /// in the order of their centres' x; a station's columns are held from the first zone that
    /// touches it to the last, then its trips are assigned along its strategy and it is dropped. So
    /// no station x station table is held, and a destination is searched once, or twice when its
    /// strategy was not kept (`WINDOW_STRATEGY_BYTES`). The longer trips
    /// cost their first boarding at the planned wait (`Skims::entry_r`, T-1430); the commutes and
    /// everyday outings as the local layer. Each destination's loads are the column's sums, added
    /// in another order than station order: the same trips, to rounding.
    pub(super) fn pass_window<'a>(&self, g: &'a Graph, p: &Params, q: usize, ix: &XferIndex, mut src: Columns<'a>, win: &[bool]) -> Pass {
        let s = g.n_st;
        let m = &self.m;
        let nzz = m.zones.len();
        // T-1735: with a window (`set_window`) only the pairs with an end in it, each under that end
        // (a commute or outing with its home there and its other end outside `swapped`), and every
        // leg ending outside taken as the leg the other way turned round (`mirror_into`)
        let windowed = !win.is_empty();
        let inw = |z: usize| !windowed || win[z];
        let c_ends = |ci: usize| {
            let c = &m.commutes[self.cpairs[ci].c as usize];
            (c.0 as usize, m.comp_of_gid[c.1 as usize] as usize)
        };
        let under = |(h, w): (usize, usize)| if inw(w) { w } else if inw(h) { h } else { usize::MAX };
        // each zone's pairs ending there, in their own order
        let (l_off, l_ix) = by_zone(nzz, self.pairs.len(), |i| under((self.pairs[i].a as usize, self.pairs[i].b as usize)));
        let (c_off, c_ix) = by_zone(nzz, self.cpairs.len(), |i| under(c_ends(i)));
        let (e_off, e_ix) = by_zone(nzz, self.epairs.len(), |i| under(self.epair_zones(&self.epairs[i])));
        let ls = |z: usize| &l_ix[l_off[z] as usize..l_off[z + 1] as usize];
        let cs = |z: usize| &c_ix[c_off[z] as usize..c_off[z + 1] as usize];
        let es = |z: usize| &e_ix[e_off[z] as usize..e_off[z + 1] as usize];
        let l_swapped = |pi: usize| !inw(self.pairs[pi].b as usize);
        let c_swapped = |ci: usize| !inw(c_ends(ci).1);
        let e_swapped = |ei: usize| !inw(self.epair_zones(&self.epairs[ei]).1);
        let home_of = |h: usize, ride: bool| if ride { &self.access_c[h] } else { &self.access_c_walk[h] };
        let c_home = |ci: usize| {
            let cp = &self.cpairs[ci];
            let h = m.commutes[cp.c as usize].0 as usize;
            (h, home_of(h, cp.ride))
        };
        let e_home = |ei: usize| {
            let ep = &self.epairs[ei];
            let h = self.epair_zones(ep).0;
            (h, home_of(h, ep.ride))
        };
        // the stations whose route costs a zone's pairs read: its own access list for each kind of
        // pair ending there (a swapped pair's home list)
        let reads = |z: usize, out: &mut Vec<usize>| {
            out.clear();
            let (mut far, mut near) = (false, false);
            for &pi in ls(z) {
                if l_swapped(pi as usize) {
                    near = true;
                } else {
                    far = true;
                }
            }
            if far {
                out.extend(self.access_work[z].iter().map(|e| e.0));
            }
            if near {
                out.extend(self.access[z].iter().map(|e| e.0));
            }
            let (mut work, mut act, mut ride, mut walk) = (false, false, false, false);
            for &ci in cs(z) {
                let ci = ci as usize;
                if c_swapped(ci) {
                    let r = self.cpairs[ci].ride;
                    ride |= r;
                    walk |= !r;
                } else {
                    work = true;
                }
            }
            for &ei in es(z) {
                let ei = ei as usize;
                if e_swapped(ei) {
                    let r = self.epairs[ei].ride;
                    ride |= r;
                    walk |= !r;
                } else {
                    act = true;
                }
            }
            for (on, list) in [(work, &self.access_cw), (act, &self.access_ew), (ride, &self.access_c), (walk, &self.access_c_walk)] {
                if on {
                    out.extend(list[z].iter().map(|e| e.0));
                }
            }
        };
        // and whose trips they fill: those, and the home stations the trips home go to (in the
        // window: unless the home is outside it, the trip home then turned round)
        let touched = |z: usize, out: &mut Vec<usize>| {
            reads(z, out);
            for &ci in cs(z) {
                let (h, list) = c_home(ci as usize);
                if !c_swapped(ci as usize) && inw(h) {
                    out.extend(list.iter().map(|e| e.0));
                }
            }
            for &ei in es(z) {
                let (h, list) = e_home(ei as usize);
                if !e_swapped(ei as usize) && inw(h) {
                    out.extend(list.iter().map(|e| e.0));
                }
            }
        };
        let mut zones: Vec<usize> = (0..nzz).filter(|&z| !(ls(z).is_empty() && cs(z).is_empty() && es(z).is_empty())).collect();
        zones.sort_by(|&a, &b| m.zones[a].x.total_cmp(&m.zones[b].x).then(a.cmp(&b)));
        let (mut tl, mut rl) = (vec![], vec![]);
        let (mut last, mut last_read) = (vec![u32::MAX; s], vec![u32::MAX; s]);
        for (u, &z) in zones.iter().enumerate() {
            touched(z, &mut tl);
            for &st in &tl {
                last[st] = u as u32;
            }
            reads(z, &mut rl);
            for &st in &rl {
                last_read[st] = u as u32;
            }
        }

        // A station's route costs from every station (`r`), the same with the planned first wait
        // where any differs (`re`) and its fare km (`km`, T-1286) are held from the first zone that
        // reads them to the last (a read slot); its trips (the longer trips', with their lean,
        // T-1373, none where the dense table was too large, `LEAN_OD_BYTES`; the commutes'; each
        // everyday purpose's) from the first zone that touches it to the last (a trip slot). The
        // trips home make the second span far the longer (on all of Japan at most about 200
        // stations read at once, about 2,000 touched), so a strategy, searched at the first read,
        // is kept to the station's assignment only within `WINDOW_STRATEGY_BYTES`, those wanted
        // soonest first, and the others are searched again then.
        let lean_on = s * s * 4 <= LEAN_OD_BYTES;
        let need_km = self.need_km();
        // the stations whose fare km are read: with no fare factor set, only the longer trips'
        // (their income groups', T-1520) ends
        let mut km_on = vec![need_km && !self.zone_fx.is_empty(); s];
        if need_km && self.zone_fx.is_empty() {
            for &z in &zones {
                for &pi in ls(z) {
                    let list = if l_swapped(pi as usize) { &self.access[z] } else { &self.access_work[z] };
                    for &(x, _) in list {
                        km_on[x] = true;
                    }
                }
            }
        }
        let has_c = !self.cpairs.is_empty();
        let mut kinds = vec![false; if self.epairs.is_empty() { 0 } else { KIND_COUNT }];
        for ep in &self.epairs {
            kinds[m.everyday[ep.e as usize].kind as usize] = true;
        }
        let col = |on: bool| if on { vec![0f32; s].into_boxed_slice() } else { Box::default() };
        let (mut rs, mut ws) = (vec![u32::MAX; s], vec![u32::MAX; s]);
        let (mut rfree, mut wfree): (Vec<usize>, Vec<usize>) = (vec![], vec![]);
        let (mut rc, mut rec, mut has_re, mut kmc): (Vec<Box<[f32]>>, Vec<Box<[f32]>>, Vec<bool>, Vec<Box<[f32]>>) = (vec![], vec![], vec![], vec![]);
        let (mut odc, mut odrc, mut cc): (Vec<Box<[f32]>>, Vec<Box<[f32]>>, Vec<Box<[f32]>>) = (vec![], vec![], vec![]);
        let mut ec: Vec<Vec<Box<[f32]>>> = kinds.iter().map(|_| vec![]).collect();
        // T-1735: the same for the legs to be turned round (a window's)
        let (mut mo, mut mr, mut mcc): (Vec<Box<[f32]>>, Vec<Box<[f32]>>, Vec<Box<[f32]>>) = (vec![], vec![], vec![]);
        let mut mec: Vec<Vec<Box<[f32]>>> = kinds.iter().map(|_| vec![]).collect();
        let mut held: Vec<Option<Held<'a>>> = vec![];
        let cap = if matches!(src, Columns::Skims(_)) { usize::MAX } else { WINDOW_STRATEGY_BYTES.with(|c| c.get()) };
        let mut kept: std::collections::BinaryHeap<(u32, u32)> = Default::default();
        let mut held_bytes = 0usize;
        let (mut kcol, mut kwcol, mut dep) = (vec![], vec![], vec![]);
        let (mut live_r, mut live_w) = (0usize, 0usize);
        let mut stats = [0usize; 5];

        let fx = need_km.then(|| FareX { dist: &[], base: p.fare_min });
        let fx_of = |zone: usize| self.fx_of(&fx, zone, 1.);
        // T-1300: the riders' fare sensitivity: each trip's rail share with its fare dearer by
        // `FARE_D_MIN` perceived minutes of the market's (times its fare factor), less its share
        let mut rail_d = FareSense::default();
        let zkm = |a: usize, b: usize| m.zone_km(&m.zones[a], &m.zones[b]);
        let add = |s: &mut FareSense, dd: f64, rail: f64, km: f64| {
            s.d += dd / FARE_D_MIN as f64;
            s.d_km += dd / FARE_D_MIN as f64 * km;
            s.km += rail * km;
        };
        let step = |zone: usize, f: f32| FARE_D_MIN * (1. + self.zone_fx.get(zone).copied().unwrap_or(0.)) * f;
        let mut rail = 0.;
        let mut cor = vec![0.; m.n_open_corridors()];
        let mut pair_rail = vec![0f32; self.pairs.len()];
        let nz = self.served.len();
        let zcol = &self.served_at;
        let pat = &self.st_pat;
        // T-2653: the table's start half (trips starting at a station, by the zone they go to) is
        // filled by column while the pass runs, in `st` after the table's own `ne` entries: a zone's
        // longer trips and commutes add into one short column, not a row per start station. Each
        // entry gets the same adds in the same order, and is copied to its place in the rows at the end.
        let ne = pat.col.len();
        let by_zone = !windowed && pat.zoff.len() == nz + 1 && !ST_BY_ZONE_OFF.load(std::sync::atomic::Ordering::Relaxed);
        let mut st = vec![0f32; ne + if by_zone { pat.zoff[nz] as usize } else { 0 }];
        // z's own column: per station, where it is in the column
        let mut zmap = vec![0u32; if by_zone { s } else { 0 }];
        // per station, where its row was last looked up, for starts and for ends
        let mut cur = vec![0u32; 2 * s];
        let at = |cur: &mut [u32], x: usize, c: usize| pat.at(x, c, &mut cur[2 * x + (c >= nz) as usize]);
        // a start entry's place in `st` while the pass runs: z's own column (`cz`) through `zmap`,
        // another column's through the entry's place in the rows
        let zslot = |cur: &mut [u32], zm: &[u32], cz: usize, x: usize, c: usize| if c == cz { zm[x] as usize } else { ne + pat.zpos(x, at(cur, x, c)) };
        // (a window's far-zone table is never read: its loads are what it is for)
        // a commute's or outing's entry: a start entry in another zone's column is looked up once a
        // station for the pair (`memo`: the stamp of the pair it was found for, and its place)
        let put_z = |st: &mut [f32], cur: &mut [u32], zm: &[u32], cz: usize, memo: &mut [(u32, u32)], stamp: u32, x: usize, c: usize, v: f32| {
            if windowed {
                return;
            }
            let i = if by_zone && c < nz {
                if c == cz {
                    zm[x] as usize
                } else {
                    let m = &mut memo[x];
                    if m.0 != stamp {
                        *m = (stamp, zslot(cur, zm, cz, x, c) as u32);
                    }
                    m.1 as usize
                }
            } else {
                at(cur, x, c)
            };
            st[i] += v;
        };
        let (mut memo_a, mut memo_b) = (vec![(0u32, 0u32); if by_zone { s } else { 0 }], vec![(0u32, 0u32); if by_zone { s } else { 0 }]);
        let mut stamp = 0u32;
        // a longer trip's entries in that table, its start stations' in the run of its station pairs
        // from one and its end stations' by place in its access list, each looked up once a pair
        let mut ends: Vec<usize> = vec![];
        let mut terms: Vec<(u32, u32, f64)> = vec![];
        let mut fares: Vec<f32> = vec![];
        let pc = commute_params();
        let (mut c_rail, mut c_frac) = (0f64, vec![0f32; self.cpairs.len()]);
        let mut c_cor = vec![0f64; m.n_open_corridors()];
        let pe = commute_params().everyday();
        let (mut e_rail, mut e_walk_lost, mut e_frac) = (0f64, 0f64, vec![0f32; self.epairs.len()]);
        let mut e_used = vec![false; kinds.len()];

        // the loads: each destination's column of every part together (crowding reads these), of
        // the commutes alone, of the longer trips' lean, of each everyday purpose
        let counting = !(cfg!(test) && XFER_OFF.load(std::sync::atomic::Ordering::Relaxed));
        let (mut all, mut com) = (DestAssign::new(g, 0, counting), DestAssign::new(g, 0, false));
        com.fares = false;
        let quiet = || {
            let mut a = DestAssign::new(g, 0, false);
            a.fares = false;
            a
        };
        let mut lean = lean_on.then(quiet);
        let mut ev: Vec<Option<DestAssign>> = kinds.iter().map(|&on| on.then(quiet)).collect();
        let mut took = vec![false; ev.len()];
        // T-1735: the legs to be turned round, as the loads above
        let mut mall = windowed.then(|| DestAssign::new(g, 0, false));
        let mut mcom = (windowed && has_c).then(quiet);
        let mut mlean = (windowed && lean_on).then(quiet);
        let mut mev: Vec<Option<DestAssign>> = kinds.iter().map(|&on| (windowed && on).then(quiet)).collect();
        let mut mtook = vec![false; mev.len()];
        let mut mecol = vec![0f32; if windowed && !kinds.is_empty() { s } else { 0 }];
        let mut xfer = vec![0.0; ix.len];
        let zero = vec![0f32; s];
        let mut ecol = vec![0f32; if kinds.is_empty() { 0 } else { s }];

        for (u, &z) in zones.iter().enumerate() {
            touched(z, &mut tl);
            for &x in &tl {
                if ws[x] != u32::MAX {
                    continue;
                }
                let k = match wfree.pop() {
                    Some(k) => {
                        for c in [&mut odc[k], &mut odrc[k], &mut cc[k], &mut mo[k], &mut mr[k], &mut mcc[k]] {
                            c.fill(0.);
                        }
                        for e in ec.iter_mut().chain(mec.iter_mut()) {
                            e[k].fill(0.);
                        }
                        k
                    }
                    None => {
                        odc.push(col(true));
                        odrc.push(col(lean_on));
                        cc.push(col(has_c));
                        mo.push(col(windowed));
                        mr.push(col(windowed && lean_on));
                        mcc.push(col(windowed && has_c));
                        for (e, &on) in ec.iter_mut().zip(&kinds) {
                            e.push(col(on));
                        }
                        for (e, &on) in mec.iter_mut().zip(&kinds) {
                            e.push(col(windowed && on));
                        }
                        held.push(None);
                        odc.len() - 1
                    }
                };
                ws[x] = k as u32;
                live_w += 1;
                stats[1] = stats[1].max(live_w);
            }
            reads(z, &mut rl);
            for &x in &rl {
                if rs[x] != u32::MAX {
                    continue;
                }
                let k = rfree.pop().unwrap_or_else(|| {
                    rc.push(col(true));
                    rec.push(col(true));
                    has_re.push(false);
                    kmc.push(col(need_km));
                    rc.len() - 1
                });
                rs[x] = k as u32;
                live_r += 1;
                stats[0] = stats[0].max(live_r);
                let h = src.fill(x, &mut rc[k], &mut rec[k], &mut has_re[k]);
                stats[3] += 1;
                if km_on[x] {
                    match &h {
                        Some(h) => {
                            self.fare_col(g, h.view(), x, &mut kcol, &mut kwcol, &mut dep);
                            kmc[k].copy_from_slice(&kcol[..s]);
                        }
                        None => kmc[k].fill(0.),
                    }
                }
                // kept for the assignment while the strategies wanted sooner leave room
                let Some(h) = h else { continue };
                let b = h.bytes();
                while held_bytes + b > cap {
                    let Some(&(l, y)) = kept.peek() else { break };
                    let y = y as usize;
                    let w = ws[y];
                    if w == u32::MAX || held[w as usize].is_none() {
                        kept.pop();
                    } else if l > last[x] {
                        kept.pop();
                        held_bytes -= held[w as usize].take().map_or(0, |h| h.bytes());
                    } else {
                        break;
                    }
                }
                if held_bytes + b <= cap {
                    held_bytes += b;
                    stats[2] = stats[2].max(held_bytes);
                    kept.push((last[x], x as u32));
                    held[ws[x] as usize] = Some(h);
                }
            }
            // each destination station's route costs and fare km, as the unit's access list has them
            let cols = |list: &[(usize, f32)], planned: bool| -> Vec<(&[f32], &[f32])> {
                list.iter()
                    .map(|&(x, _)| {
                        let k = rs[x] as usize;
                        (if planned && has_re[k] { &rec[k][..] } else { &rc[k][..] }, &kmc[k][..])
                    })
                    .collect()
            };

            // (z's own column of the start half: where each of its stations is)
            let cz = if by_zone && (zcol[z] as usize) < nz { zcol[z] as usize } else { usize::MAX };
            if cz != usize::MAX {
                let (a, b) = (pat.zoff[cz] as usize, pat.zoff[cz + 1] as usize);
                for k in a..b {
                    zmap[pat.zst.get(k)] = (ne + k) as u32;
                }
            }
            // the longer trips ending in z (T-1735, a window's swapped ones: starting there)
            if !ls(z).is_empty() {
                let (mut bc, mut hc) = (None, None);
                for &pi in ls(z) {
                    let pi = pi as usize;
                    let pair = &self.pairs[pi];
                    let (za, zb) = (pair.a as usize, pair.b as usize);
                    let swap = l_swapped(pi);
                    let (pa, pb) = (m.zones[za].place, m.zones[zb].place);
                    let (from, to) = (&m.places[pa], &m.places[pb]);
                    let rule = m.pr(pa, pb);
                    let split = rule.split_car(from, to);
                    let (a, b) = (&self.access[za], &self.access_work[zb]);
                    let (ca, cb) = (nz + zcol[za] as usize, zcol[zb] as usize);
                    let ln = if lean_on { m.zone_lean(&m.zones[za], &m.zones[zb]) } else { 0. };
                    let mut r_all = 0f64;
                    // T-1431: each income group of the place the trip starts from, every mode's
                    // money at its riders' minutes a dollar; T-1520: rail for all of them from one
                    // logsum at their mean
                    let gs = from.groups();
                    let th0 = p.band_unsoftened(pair.band as usize);
                    let fm: f32 = gs.iter().map(|g| g.0 * g.1).sum();
                    let alt_m = rule.alt(pair.car, split, from.w() * fm, &m.mp);
                    let (th, d) = (p.soften(th0, alt_m.car), self.fare_d(za, fm));
                    // (swapped: priced on the route costs the other way, from its far stations to its home ones)
                    let mix = if swap {
                        rail_mix_cols(hc.get_or_insert_with(|| cols(a, true)), p.fare_min, b, a, p.beta, th, fx.as_ref(), d, &mut terms, &mut fares)
                    } else {
                        rail_mix_cols(bc.get_or_insert_with(|| cols(b, true)), p.fare_min, a, b, p.beta, th, fx.as_ref(), d, &mut terms, &mut fares)
                    };
                    let sum = mix.sum;
                    for &(share, f) in gs {
                        let trips = pair.trips * SHARES[q] as f64 * share as f64;
                        let alt = if gs.len() == 1 { alt_m } else { rule.alt(pair.car, split, from.w() * f, &m.mp) };
                        let rail_c = mix.at(self.fare_d(za, f), p.soften(th0, alt.car));
                        let bt = p.mode_beta(alt.car);
                        let v_rail = if sum > 0. { ((-bt * (rail_c - alt.drive)).clamp(-80., 80.) as f64).exp() } else { 0. };
                        let fraction = v_rail / (alt.rest as f64 + v_rail);
                        let r = trips * fraction;
                        r_all += r;
                        if sum > 0. {
                            let v = ((-bt * (rail_c + step(za, f) - alt.drive)).clamp(-80., 80.) as f64).exp();
                            add(&mut rail_d, trips * (v / (alt.rest as f64 + v) - fraction), r, zkm(za, zb));
                        }
                    }
                    if sum > 0. && r_all > 0. {
                        ends.clear();
                        ends.resize(b.len(), usize::MAX);
                        let (mut last_sa, mut ia, mut jb) = (u32::MAX, 0, 0);
                        for &(sa, sb, v) in &terms {
                            let rr = (r_all * v / sum) as f32;
                            if swap {
                                // from far station sa to home station sb: the trip the other way
                                let (sa, sb) = (sa as usize, sb as usize);
                                let k = ws[sb] as usize;
                                mo[k][sa] += rr;
                                if ln != 0. {
                                    mr[k][sa] += rr * ln;
                                }
                                continue;
                            }
                            if !windowed {
                                if sa != last_sa {
                                    (last_sa, ia) = (sa, if by_zone { zslot(&mut cur, &zmap, cz, sa as usize, cb) } else { at(&mut cur, sa as usize, cb) });
                                }
                                while b[jb].0 != sb as usize {
                                    jb = (jb + 1) % b.len();
                                }
                                if ends[jb] == usize::MAX {
                                    ends[jb] = at(&mut cur, sb as usize, ca);
                                }
                            }
                            let (sa, sb) = (sa as usize, sb as usize);
                            let k = ws[sb] as usize;
                            odc[k][sa] += rr;
                            if ln != 0. {
                                odrc[k][sa] += rr * ln;
                            }
                            if !windowed {
                                st[ia] += rr;
                                st[ends[jb]] += rr;
                            }
                        }
                    }
                    rail += r_all;
                    cor[pair.corridor as usize] += r_all;
                    pair_rail[pi] = r_all as f32;
                }
            }
            // T-276: the commutes working in z, chosen as the local layer chooses: once per
            // commuter on this period's service from home to work, the trip home on the same
            // station pair reversed
            if !cs(z).is_empty() {
                // T-1735: a swapped pair (`under`) searched towards its home stations from the work
                // stations, the trips there taken as the trips home turned round
                let (mut bc, mut hc_ride, mut hc_walk) = (None, None, None);
                for &ci in cs(z) {
                    let ci = ci as usize;
                    stamp += 1;
                    let cp = &self.cpairs[ci];
                    let n = m.commutes[cp.c as usize].2;
                    let (h, home) = c_home(ci);
                    let w = c_ends(ci).1;
                    let wc = zcol[w] as usize;
                    let swap = c_swapped(ci);
                    let (out, ret) = (n as f64 * TO_WORK[q] as f64, n as f64 * TO_HOME[q] as f64);
                    let th = p.soften(pc.band_unsoftened(cp.band as usize), cp.car);
                    let (rail_c, sum) = if swap {
                        let hcols = if cp.ride { hc_ride.get_or_insert_with(|| cols(home, false)) } else { hc_walk.get_or_insert_with(|| cols(home, false)) };
                        rail_logsum_cols(hcols, p.fare_min, &self.access_cw[w], home, pc.beta, th, fx_of(h), &mut terms)
                    } else {
                        let b = &self.access_cw[z];
                        rail_logsum_cols(bc.get_or_insert_with(|| cols(b, false)), p.fare_min, home, b, pc.beta, th, fx_of(h), &mut terms)
                    };
                    if sum <= 0. {
                        continue;
                    }
                    // T-918: rail against the car on the cost ratio, as the local commutes
                    let frac = pc.ratio_share(rail_c, cp.car) as f64;
                    c_frac[ci] = frac as f32;
                    add(&mut rail_d, (out + ret) * (pc.ratio_share(rail_c + step(h, 1.), cp.car) as f64 - frac), (out + ret) * frac, zkm(h, w));
                    let (ro, rr_) = (out * frac, ret * frac);
                    c_rail += ro + rr_;
                    c_cor[cp.corridor] += ro + rr_;
                    let hc = zcol[h] as usize;
                    // (the trips home turned round where the home is outside the window)
                    let back = !swap && !inw(h);
                    for &(sa, sb, v) in &terms {
                        let (a, bk) = ((ro * v / sum) as f32, (rr_ * v / sum) as f32);
                        // home and work stations; swapped, the terms run from work to home
                        let (sa, sb) = if swap { (sb as usize, sa as usize) } else { (sa as usize, sb as usize) };
                        if swap {
                            cc[ws[sa] as usize][sb] += bk;
                            mcc[ws[sa] as usize][sb] += a;
                        } else {
                            cc[ws[sb] as usize][sa] += a;
                            if back {
                                mcc[ws[sb] as usize][sa] += bk;
                            } else {
                                cc[ws[sa] as usize][sb] += bk;
                            }
                        }
                        put_z(&mut st, &mut cur, &zmap, cz, &mut memo_a, stamp, sa, wc, a);
                        put_z(&mut st, &mut cur, &zmap, cz, &mut memo_a, stamp, sb, nz + hc, a);
                        put_z(&mut st, &mut cur, &zmap, cz, &mut memo_b, stamp, sb, hc, bk);
                        put_z(&mut st, &mut cur, &zmap, cz, &mut memo_b, stamp, sa, nz + wc, bk);
                    }
                }
            }
            // T-1175: the everyday outings to z across a zone line, chosen as the everyday layer
            // chooses: the pair's walkers with no rail first (their weight against the car from
            // that share), rail and the car splitting the rest on their cost ratio, rail taking
            // its share of the walkers; out at the purpose's outward timing, home on the same
            // station pairs reversed
            if !es(z).is_empty() {
                // (swapped as the commutes)
                let (mut bc, mut hc_ride, mut hc_walk) = (None, None, None);
                for &ei in es(z) {
                    let ei = ei as usize;
                    stamp += 1;
                    let ep = &self.epairs[ei];
                    let pair = m.everyday[ep.e as usize];
                    let k = pair.kind as usize;
                    let (h, home) = e_home(ei);
                    let az = self.epair_zones(ep).1;
                    let ac = zcol[az] as usize;
                    let swap = e_swapped(ei);
                    let (out, ret) = (pair.trips as f64 * OUTWARD[k][q] as f64, pair.trips as f64 * RETURN[k][q] as f64);
                    let th = p.soften(pe.band_unsoftened(ep.band as usize), ep.car);
                    let (rail_c, sum) = if swap {
                        let hcols = if ep.ride { hc_ride.get_or_insert_with(|| cols(home, false)) } else { hc_walk.get_or_insert_with(|| cols(home, false)) };
                        rail_logsum_cols(hcols, p.fare_min, &self.access_ew[az], home, pe.beta, th, fx_of(h), &mut terms)
                    } else {
                        let b = &self.access_ew[z];
                        rail_logsum_cols(bc.get_or_insert_with(|| cols(b, false)), p.fare_min, home, b, pe.beta, th, fx_of(h), &mut terms)
                    };
                    if sum <= 0. {
                        continue;
                    }
                    let util = (pe.mode_beta(ep.car) * (ep.car - rail_c) + pe.asc_rail).clamp(-80., 80.) as f64;
                    let w0 = ep.walk_e as f64 / (util.exp() + 1. + ep.walk_e as f64);
                    let frac = (1. - w0) * pe.ratio_share(rail_c, ep.car) as f64;
                    e_frac[ei] = frac as f32;
                    let rd = rail_c + step(h, 1.);
                    let ud = (pe.mode_beta(ep.car) * (ep.car - rd) + pe.asc_rail).clamp(-80., 80.) as f64;
                    add(&mut rail_d, (out + ret) * ((1. - ep.walk_e as f64 / (ud.exp() + 1. + ep.walk_e as f64)) * pe.ratio_share(rd, ep.car) as f64 - frac), (out + ret) * frac, zkm(h, az));
                    let (ro, rr_) = (out * frac, ret * frac);
                    e_rail += ro + rr_;
                    e_walk_lost += (ro + rr_) * pair.walk as f64;
                    let hc = zcol[h] as usize;
                    e_used[k] |= !terms.is_empty();
                    let back = !swap && !inw(h);
                    for &(sa, sb, v) in &terms {
                        let (a, bk) = ((ro * v / sum) as f32, (rr_ * v / sum) as f32);
                        let (sa, sb) = if swap { (sb as usize, sa as usize) } else { (sa as usize, sb as usize) };
                        if swap {
                            ec[k][ws[sa] as usize][sb] += bk;
                            mec[k][ws[sa] as usize][sb] += a;
                        } else {
                            ec[k][ws[sb] as usize][sa] += a;
                            if back {
                                mec[k][ws[sb] as usize][sa] += bk;
                            } else {
                                ec[k][ws[sa] as usize][sb] += bk;
                            }
                        }
                        put_z(&mut st, &mut cur, &zmap, cz, &mut memo_a, stamp, sa, ac, a);
                        put_z(&mut st, &mut cur, &zmap, cz, &mut memo_a, stamp, sb, nz + hc, a);
                        put_z(&mut st, &mut cur, &zmap, cz, &mut memo_b, stamp, sb, hc, bk);
                        put_z(&mut st, &mut cur, &zmap, cz, &mut memo_b, stamp, sa, nz + ac, bk);
                    }
                }
            }

            // the stations whose last read z is: their read slots freed
            for &x in &rl {
                if last_read[x] as usize == u && rs[x] != u32::MAX {
                    rfree.push(rs[x] as usize);
                    rs[x] = u32::MAX;
                    live_r -= 1;
                }
            }
            // the stations z is the last zone of: their columns assigned, their slots freed
            for &dst in &tl {
                if last[dst] as usize != u || ws[dst] == u32::MAX {
                    continue;
                }
                let k = ws[dst] as usize;
                ws[dst] = u32::MAX;
                wfree.push(k);
                live_w -= 1;
                let kept_h = held[k].take();
                if let Some(h) = &kept_h {
                    held_bytes -= h.bytes();
                }
                let ccol: &[f32] = if has_c { &cc[k] } else { &zero };
                if !kinds.is_empty() {
                    ecol.fill(0.);
                    for (e, &on) in ec.iter().zip(&kinds) {
                        if on {
                            for (x, &v) in ecol.iter_mut().zip(e[k].iter()) {
                                *x += v;
                            }
                        }
                    }
                }
                let ecol: &[f32] = if kinds.is_empty() { &zero } else { &ecol };
                let od = &odc[k];
                let c = com.take(dst, |x| ccol[x]);
                let a = all.take(dst, |x| od[x] + ccol[x] + ecol[x]);
                let r = lean.as_mut().is_some_and(|l| l.take(dst, |x| odrc[k][x]));
                for (i, ea) in ev.iter_mut().enumerate() {
                    took[i] = ea.as_mut().is_some_and(|ea| ea.take(dst, |x| ec[i][k][x]));
                }
                // T-1735: the legs to be turned round
                let mut mirrored = false;
                if windowed {
                    if !kinds.is_empty() {
                        mecol.fill(0.);
                        for (e, &on) in mec.iter().zip(&kinds) {
                            if on {
                                for (x, &v) in mecol.iter_mut().zip(e[k].iter()) {
                                    *x += v;
                                }
                            }
                        }
                    }
                    let (mo, mcc, mecol) = (&mo[k], &mcc[k], &mecol);
                    let at = |v: &[f32], x: usize| v.get(x).copied().unwrap_or(0.);
                    mirrored |= mall.as_mut().is_some_and(|l| l.take(dst, |x| mo[x] + at(mcc, x) + at(mecol, x)));
                    mirrored |= mcom.as_mut().is_some_and(|l| l.take(dst, |x| mcc[x]));
                    mirrored |= mlean.as_mut().is_some_and(|l| l.take(dst, |x| mr[k][x]));
                    for (i, ea) in mev.iter_mut().enumerate() {
                        mtook[i] = ea.as_mut().is_some_and(|ea| ea.take(dst, |x| mec[i][k][x]));
                        mirrored |= mtook[i];
                    }
                }
                if !(a || c || r || mirrored || took.contains(&true)) {
                    continue;
                }
                let h = match kept_h {
                    Some(h) => h,
                    None => {
                        stats[4] += 1;
                        match src.strategy(dst) {
                            Some(h) => h,
                            None => continue,
                        }
                    }
                };
                let sv = h.view();
                if c {
                    com.walk(g, dst, sv, None);
                }
                if let (true, Some(l)) = (r, lean.as_mut()) {
                    l.walk(g, dst, sv, None);
                }
                for (i, ea) in ev.iter_mut().enumerate() {
                    if let (true, Some(ea)) = (took[i], ea.as_mut()) {
                        ea.walk(g, dst, sv, None);
                    }
                }
                if a {
                    all.walk(g, dst, sv, counting.then_some((ix, &mut xfer[..])));
                }
                if mirrored {
                    // (each walks a column of zeros harmlessly where it took none)
                    for l in [&mut mall, &mut mcom, &mut mlean].into_iter().chain(mev.iter_mut()).flatten() {
                        l.walk(g, dst, sv, None);
                    }
                }
            }
        }
        if by_zone {
            // T-2653: the start half back in row order (each entry is still 0 there)
            let mut fill: Vec<u32> = pat.zoff[..nz].to_vec();
            for x in 0..pat.off.len() - 1 {
                for i in pat.row(x) {
                    let c = pat.col.get(i);
                    if c < nz {
                        st[i] = st[ne + fill[c] as usize];
                        fill[c] += 1;
                    }
                }
            }
            st.truncate(ne);
            st.shrink_to_fit();
        }
        // T-1735: the window's legs turned round
        if let Some(l) = &mall {
            mirror_into(g, &l.l, &mut all.l);
        }
        if let Some(l) = &mcom {
            mirror_into(g, &l.l, &mut com.l);
        }
        if let (Some(l), Some(out)) = (&mlean, lean.as_mut()) {
            mirror_into(g, &l.l, &mut out.l);
        }
        for (i, l) in mev.iter().enumerate() {
            if let (Some(l), Some(out)) = (l, ev[i].as_mut()) {
                mirror_into(g, &l.l, &mut out.l);
                e_used[i] = true;
            }
        }
        WINDOW_STATS.with(|w| {
            let mut v = w.get();
            for i in 0..3 {
                v[i] = v[i].max(stats[i]);
            }
            v[3] += stats[3];
            v[4] += stats[4];
            w.set(v);
        });
        let e_loads = ev.into_iter().zip(e_used).map(|(a, u)| a.filter(|_| u).map(|a| a.l)).collect();
        Pass { loads: all.l, c_loads: com.l, r_loads: lean.map(|l| l.l), rail, c_rail, c_frac, cor, c_cor, pair_rail, st, xfer, e_loads, e_rail, e_walk_lost, e_frac, rail_d }
    }
}
