use super::*;

#[wasm_bindgen]
impl IntercityApi {
    /// A market from a world composition (`{"world": registry rule and rows, "zones": [place
    /// packs]}`) or an old single pack (`sites`: a closed world).
    #[wasm_bindgen(constructor)]
    pub fn new(header: &str) -> Result<IntercityApi, String> {
        Ok(IntercityApi::on(Rc::new(parse(header)?)))
    }
    /// T-409: the same market with no network yet, sharing this one's data, for a network of its
    /// own: one per cluster the market reaches.
    pub fn beside(&self) -> IntercityApi {
        IntercityApi::on(self.m.clone())
    }
    pub fn set_complexes(&mut self, complex: &[u32]) {
        self.complex_in = complex.to_vec();
    }
    /// `DemandApi::set_upgrades` (T-1281).
    pub fn set_upgrades(&mut self, ups: &[u8]) {
        self.ups_in = ups.to_vec();
    }
    /// `DemandApi::set_passages` (T-1281).
    pub fn set_passages(&mut self, pairs: &[u32]) {
        self.passages_in = pairs.to_vec();
    }
    /// `DemandApi::set_span` (T-842): riders between cities read the same service span.
    pub fn set_span(&mut self, span: &[f32]) {
        self.span_in = span.to_vec();
    }
    /// `DemandApi::set_bus` (T-1464), the points as longitude and latitude, as `set_network_ll`
    /// takes the stations.
    pub fn set_bus_ll(&mut self, catch_n: &[u32], catch_ll: &[f64]) {
        let xy: Vec<f32> = match &self.m.anchor {
            Some(a) => catch_ll.chunks_exact(2).flat_map(|p| {
                let (x, y) = a.from_lonlat(p[0], p[1]);
                [x as f32, y as f32]
            }).collect(),
            None => catch_ll.iter().map(|&v| v as f32).collect(),
        };
        self.bus_in = (catch_n.to_vec(), xy);
    }
    /// T-780: zone pairs sampled in chunks of `MARKET_CHUNK` trips (`solved_pairs`), or every pair
    /// solved; the next network sets its pairs up again.
    pub fn set_sampled(&mut self, on: bool) {
        self.set_chunk(if on { MARKET_CHUNK } else { 0. });
    }
    /// T-2641: with sampled pairs, the chunk grows with the trip's distance and far place pairs are
    /// dealt over both places' zones (`?perfOff=farChunks` turns it off); the next network sets
    /// its pairs up again.
    pub fn set_far_chunks(&mut self, on: bool) {
        if on != self.far_chunks {
            self.far_chunks = on;
            self.svc = None;
            self.served.clear();
        }
    }
    /// T-1735: from the next `set_network` on, only the pairs with an end in a window of zones:
    /// those with a station marked in `st_mask` (one byte per station that call takes) in an access
    /// list, and `zones` (indices as `window_zones` gives them). Each pair is searched towards its
    /// end in the window alone, its leg the other way taken as that leg turned round.
    /// So are the zones with a station within `minutes` (perceived, by rail at free flow) of a
    /// marked one, nearest first, while the window's stations are at most `share` of those the
    /// market searches (its pass costs about that share of a whole pass), or `floor` stations.
    pub fn set_window(&mut self, st_mask: &[u8], zones: &[u32], minutes: f32, share: f32, floor: u32) {
        self.win_in = Some((st_mask.to_vec(), zones.to_vec(), minutes, share, floor));
    }
    /// T-1735: the window's zones (empty: none).
    pub fn window_zones(&self) -> Vec<u32> {
        self.win.iter().enumerate().filter(|e| *e.1).map(|e| e.0 as u32).collect()
    }
    /// T-1735: the zones (indices as `window_zones` gives them) with a station marked in `st_mask`
    /// (one byte per station of the network this holds, as `set_network` took them) in an access list.
    pub fn zones_at(&self, st_mask: &[u8]) -> Vec<u32> {
        let Some(svc) = &self.svc else { return vec![] };
        let mut on = vec![false; svc.st_x.len()];
        for (i, &v) in st_mask.iter().enumerate() {
            if v != 0 && i < svc.st_map.len() {
                on[svc.st_map[i] as usize] = true;
            }
        }
        let mut win = vec![false; self.m.zones.len()];
        for lists in [&self.access, &self.access_work, &self.access_c, &self.access_cw, &self.access_c_walk, &self.access_ew] {
            for (z, list) in lists.iter().enumerate() {
                win[z] |= list.iter().any(|&(st, _)| on[st]);
            }
        }
        win.iter().enumerate().filter(|e| *e.1).map(|e| e.0 as u32).collect()
    }
    /// T-1735: period `q`'s loads of the pairs with an end in `zones` alone, as the window's pass
    /// takes them, on the network and crowding this holds solved (one pass): blocks of segment
    /// loads, boardings and alightings per route node, each three after another: every pair's, the
    /// commutes', then each everyday purpose's (as `everyday_loads`); empty if `q` is not solved.
    pub fn window_loads(&self, q: usize, zones: &[u32]) -> Vec<f32> {
        let Some(st) = self.per(q) else { return vec![] };
        let mut win = vec![false; self.m.zones.len()];
        for &z in zones {
            if let Some(w) = win.get_mut(z as usize) {
                *w = true;
            }
        }
        let nr = st.g.rn_line.len();
        if zones.is_empty() {
            return vec![0.; 3 * nr];
        }
        let pass = self.pass_searched_in(&st.g, &st.p, q, &st.xi, &win);
        let mut out = Vec::with_capacity(3 * nr * (2 + pass.e_loads.len()));
        for l in std::iter::once(Some(&pass.loads)).chain(std::iter::once(Some(&pass.c_loads))).chain(pass.e_loads.iter().map(|e| e.as_ref())) {
            match l {
                Some(l) => {
                    out.extend_from_slice(&l.seg);
                    out.extend_from_slice(&l.rn_board);
                    out.extend_from_slice(&l.rn_alight);
                }
                None => out.resize(out.len() + 3 * nr, 0.),
            }
        }
        out
    }
}
impl IntercityApi {
    /// `solve`, at free flow or (T-1735) at the crowding the loads `crowd` give.
    pub(super) fn solve_at(&mut self, q: usize, crowd: Option<&[f32]>) -> bool {
        let Some(svc) = &self.svc else { return false };
        if q >= PERIODS {
            return false;
        }
        let t = now_ms();
        let p = Params {
            period: q,
            period_share: SHARES[q],
            period_hours: (PERIOD_HOURS[q].1 - PERIOD_HOURS[q].0) as f32,
            peak_hour_factor: PEAK_HOUR_FACTOR[q],
            ..self.base.clone()
        };
        let net = DemandApi::period_net(svc, q);
        let mut g = build_graph(&net, &p);
        put_fares(&mut g, &self.fare_curve, &self.fare_km, &self.fare_mul, self.base.fare_min, self.fare_per_km, self.base.fare_w);
        let no_crowding = vec![1.0; g.rn_ride.len()];
        set_crowding(&mut g, &no_crowding);
        let (mut fac, mut uniform) = (no_crowding, true);
        if let Some(total) = crowd {
            let Some((f, left)) = round_costs(&g, &net, &p, total) else { return false };
            set_crowding(&mut g, &f);
            g.rn_left = left;
            (fac, uniform) = (f, false);
        }
        let xi = xfer_index(&g);
        let pass = self.pass(&g, &p, q, &xi);
        let (e_trips, e_walk_base) = self.m.everyday.iter().fold((0f64, 0f64), |(t, w), e| {
            let legs = e.trips as f64 * (OUTWARD[e.kind as usize][q] + RETURN[e.kind as usize][q]) as f64;
            (t + legs, w + legs * e.walk as f64)
        });
        self.per[q] = Some(Period {
            e_avg: pass.e_loads,
            e_trips,
            e_rail: pass.e_rail,
            e_walk: (e_walk_base - pass.e_walk_lost).max(0.),
            e_walk_base,
            e_frac: pass.e_frac,
            net,
            p,
            g,
            avg: pass.loads,
            c_avg: pass.c_loads,
            r_avg: pass.r_loads,
            rail: pass.rail,
            rail_d: pass.rail_d,
            trips: self.m.cw.daily * SHARES[q] as f64,
            c_rail: pass.c_rail,
            c_trips: self.m.cw.commuters * (TO_WORK[q] + TO_HOME[q]) as f64,
            c_frac: pass.c_frac,
            corridors: Sparse::of(&pass.cor),
            c_corridors: Sparse::of(&pass.c_cor),
            pair_rail: pass.pair_rail,
            st: pass.st,
            round: 0,
            ms: now_ms() - t,
            fac,
            uniform,
            seen: vec![],
            xi,
            xfer: pass.xfer,
        });
        true
    }
    pub(super) fn set_chunk(&mut self, chunk: f64) {
        if chunk != self.chunk {
            self.chunk = chunk;
            self.svc = None;
            self.served.clear();
        }
    }
}
#[wasm_bindgen]
impl IntercityApi {
    /// Sites, open zones, the day's trips between places, 1 if the places' rows came from the
    /// world (0: added up at opening, a closed world or the pipeline), the day's departures to
    /// places with no zone.
    pub fn info(&self) -> Vec<f64> {
        vec![self.m.n_sites() as f64, self.m.zones.len() as f64, self.m.cw.daily, self.m.from_rows as u8 as f64, self.m.cw.leaving]
    }
    /// Per open 6 km zone: its global id, then the box its sites lie in (west, south, east, north;
    /// degrees in a world composition, else metres in the pack's plane). A station far from a
    /// zone's box is in no access list of that zone (T-409).
    pub fn zone_boxes(&self) -> Vec<f64> {
        let m = &self.m;
        let mut out = vec![f64::INFINITY; 5 * m.zones.len()];
        for (z, zz) in m.zones.iter().enumerate() {
            out[5 * z] = zz.gid as f64;
            out[5 * z + 3] = f64::NEG_INFINITY;
            out[5 * z + 4] = f64::NEG_INFINITY;
        }
        for s in 0..m.n_sites() {
            let (g, i) = m.site(s);
            let (x, y) = (g.city.x[i] as f64, g.city.y[i] as f64);
            let (x, y) = match (&g.frame, m.world) {
                (Some(f), true) => f.to_lonlat(x, y),
                _ => (x, y),
            };
            let b = &mut out[5 * m.cell_zone[s] + 1..5 * m.cell_zone[s] + 5];
            b[0] = b[0].min(x);
            b[1] = b[1].min(y);
            b[2] = b[2].max(x);
            b[3] = b[3].max(y);
        }
        out
    }
    /// T-377: the network with stations as longitude and latitude (a world composition): put in
    /// the frame of its first group of sites for the graph, and in each group's own for access.
    pub fn set_network_ll(&mut self, st_ll: &[f64], line_n: &[u32], stops: &[u32], times: &[f32], tph: &[f32], cars: &[f32]) -> Result<f64, String> {
        if st_ll.len() % 2 != 0 || st_ll.iter().any(|v| !v.is_finite()) {
            return Err("intercity station positions".into());
        }
        let xy: Vec<f32> = match &self.m.anchor {
            Some(a) => st_ll.chunks_exact(2).flat_map(|p| {
                let (x, y) = a.from_lonlat(p[0], p[1]);
                [x as f32, y as f32]
            }).collect(),
            None => st_ll.iter().map(|&v| v as f32).collect(),
        };
        self.set_network_in(&xy, line_n, stops, times, tph, cars)
    }
    pub fn set_fares(&mut self, base: f32, per_km: f32, km: &[f32]) -> Result<(), String> {
        if !base.is_finite() || base < 0. || !per_km.is_finite() || per_km < 0. || km.iter().any(|x| !x.is_finite() || *x < 0.) || km.len() != self.svc.as_ref().map_or(0, |s| 2 * s.stops.len()) {
            return Err("invalid intercity fare".into());
        }
        let w = self.base.fare_w;
        let (fare_min, per_km) = (base * w, per_km * w);
        if fare_min.to_bits() != self.base.fare_min.to_bits() || per_km.to_bits() != self.fare_per_km.to_bits() || !same_bits(km, &self.fare_km) {
            self.base.fare_min = fare_min;
            self.fare_per_km = per_km;
            self.fare_km = km.to_vec();
            self.per.iter_mut().for_each(|p| *p = None);
        }
        Ok(())
    }
    /// T-1626: `DemandApi::set_fare_mul`.
    pub fn set_fare_mul(&mut self, mul: &[f32]) -> Result<(), String> {
        check_fare_mul(mul, self.svc.as_ref().map_or(0, |s| 2 * s.stops.len()))?;
        let mul = if mul.iter().all(|&v| v == 1.) { &[][..] } else { mul };
        if !same_bits(mul, &self.fare_mul) {
            self.fare_mul = mul.to_vec();
            self.per.iter_mut().for_each(|p| *p = None);
        }
        Ok(())
    }
    /// T-1277: perceived minutes a dollar of the fares the market is given (`DemandApi::
    /// set_fare_weight`); before `set_fares`. The plane's fare weighs each departing country's
    /// own (`WorldIn::fare_w`).
    pub fn set_fare_weight(&mut self, w: f32) -> Result<(), String> {
        if !(w.is_finite() && w > 0.) {
            return Err("invalid fare weight".into());
        }
        if w.to_bits() != self.base.fare_w.to_bits() {
            self.base.fare_w = w;
            self.per.iter_mut().for_each(|p| *p = None);
        }
        Ok(())
    }
    pub fn set_fare_curve(&mut self, knots: &[f32]) -> Result<(), String> {
        check_fare_curve(knots)?;
        if !same_bits(knots, &self.fare_curve) {
            self.fare_curve = knots.to_vec();
            self.per.iter_mut().for_each(|p| *p = None);
        }
        Ok(())
    }
    /// T-1286: the factor each country's riders weigh the market's fares at, as JSON
    /// `{"US": 1.25, ...}` (countries left out: 1): the player's fare scale there over its income's.
    /// Each trip takes its origin's (a commute or outing its home's).
    pub fn set_fare_factors(&mut self, json: &str) -> Result<(), String> {
        let by: HashMap<String, f32> = serde_json::from_str(json).map_err(|e| e.to_string())?;
        if by.values().any(|f| !(f.is_finite() && *f > 0.)) {
            return Err("invalid fare factor".into());
        }
        let m = &self.m;
        let of_country: Vec<f32> = m.countries.iter().map(|c| by.get(c).copied().unwrap_or(1.) - 1.).collect();
        let fx: Vec<f32> = if of_country.iter().all(|&d| d == 0.) {
            vec![]
        } else {
            m.zones.iter().map(|z| of_country[m.places[z.place].country as usize]).collect()
        };
        if !same_bits(&fx, &self.zone_fx) {
            self.zone_fx = fx;
            self.per.iter_mut().for_each(|p| *p = None);
        }
        Ok(())
    }
    pub fn solve(&mut self, q: usize) -> bool {
        self.solve_at(q, None)
    }
    /// T-1735: period `q` solved in one pass at the crowding the loads `total` give (as
    /// `crowd_with_external` takes them), for a window's pairs solved beside a market frozen at its
    /// last round's crowding. False with no network.
    pub fn solve_crowded(&mut self, q: usize, total: &[f32]) -> bool {
        self.solve_at(q, Some(total))
    }
    /// T-1607: drop period `q`'s result, which another worker holds (`DemandApi::forget_period`).
    pub fn forget_period(&mut self, q: usize) {
        if let Some(p) = self.per.get_mut(q) {
            *p = None;
        }
    }
    pub fn crowd_with_external(&mut self, q: usize, total: &[f32]) -> bool {
        let Some(st) = self.per(q) else { return false };
        let Some((fac, left)) = round_costs(&st.g, &st.net, &st.p, total) else {
            return false;
        };
        let t = now_ms();
        let mut st = self.per[q].take().unwrap();
        if settled(&mut st.seen, &total[..st.g.rn_line.len()]) || (st.uniform && fac == st.fac && left == st.g.rn_left) {
            st.round += 1;
            st.ms = now_ms() - t;
            self.per[q] = Some(st);
            return true;
        }
        set_crowding(&mut st.g, &fac);
        st.g.rn_left = left;
        st.uniform = false;
        st.fac = fac;
        let Pass { loads: l, c_loads, r_loads, rail, c_rail, c_frac, cor, c_cor, pair_rail, st: st_new, xfer, e_loads, e_rail, e_walk_lost, e_frac, rail_d } = self.pass(&st.g, &st.p, q, &st.xi);
        st.round += 1;
        let d = st.round as f64 + 1.;
        let (df, w) = (d as f32, 1. / d);
        let wf = 1. / df;
        if let (Some(a), Some(b)) = (st.r_avg.as_mut(), r_loads) {
            msa_w(&mut a.seg, &b.seg, wf);
            msa_w(&mut a.rn_board, &b.rn_board, wf);
            msa_w(&mut a.rn_alight, &b.rn_alight, wf);
        }
        msa_w(&mut st.xfer, &xfer, wf);
        msa_w(&mut st.pair_rail, &pair_rail, wf);
        msa_w(&mut st.c_frac, &c_frac, wf);
        msa_w(&mut st.c_avg.seg, &c_loads.seg, wf);
        msa_w(&mut st.c_avg.rn_board, &c_loads.rn_board, wf);
        msa_w(&mut st.c_avg.rn_alight, &c_loads.rn_alight, wf);
        st.c_rail += (c_rail - st.c_rail) / d;
        msa_w(&mut st.e_frac, &e_frac, wf);
        for (a, b) in st.e_avg.iter_mut().zip(e_loads) {
            if let (Some(a), Some(b)) = (a.as_mut(), b) {
                msa_w(&mut a.seg, &b.seg, wf);
                msa_w(&mut a.rn_board, &b.rn_board, wf);
                msa_w(&mut a.rn_alight, &b.rn_alight, wf);
            }
        }
        st.e_rail += (e_rail - st.e_rail) / d;
        let e_walk = (st.e_walk_base - e_walk_lost).max(0.);
        st.e_walk += (e_walk - st.e_walk) / d;
        for (a, b) in st.st.iter_mut().zip(st_new) {
            *a += (b - *a) / df;
        }
        msa_w(&mut st.avg.seg, &l.seg, wf);
        msa_w(&mut st.avg.board, &l.board, wf);
        msa_w(&mut st.avg.alight, &l.alight, wf);
        msa_w(&mut st.avg.rn_board, &l.rn_board, wf);
        msa_w(&mut st.avg.rn_alight, &l.rn_alight, wf);
        msa_w(&mut st.avg.entry, &l.entry, wf);
        msa_w(&mut st.avg.exit, &l.exit, wf);
        msa_w(&mut st.avg.rn_fare_x, &l.rn_fare_x, wf);
        st.rail += (rail - st.rail) * w;
        st.rail_d.toward(rail_d, w);
        st.corridors.toward(&cor, w);
        st.c_corridors.toward(&c_cor, w);
        st.ms = now_ms() - t;
        self.per[q] = Some(st);
        true
    }
    /// `DemandApi::cap_with_external` (T-1070). The longer trips' figures by pair, corridor and
    /// station fall by the share of all the market's rail trips that still fit.
    pub fn cap_with_external(&mut self, q: usize, total: &[f32]) -> bool {
        let Some(st) = self.per.get_mut(q).and_then(|p| p.as_mut()) else { return false };
        let Some(fit) = capacity_fit(&st.g, &st.net, &st.p, total) else { return false };
        if !fit.any {
            return true;
        }
        let (_, all) = cap_loads(&st.g, &fit, &st.xi, &mut st.avg, &mut st.xfer);
        let c = &mut st.c_avg;
        cap_part(&fit, &mut c.seg, &mut c.rn_board, &mut c.rn_alight);
        if let Some(c) = st.r_avg.as_mut() {
            cap_part(&fit, &mut c.seg, &mut c.rn_board, &mut c.rn_alight);
        }
        for e in st.e_avg.iter_mut().flatten() {
            cap_part(&fit, &mut e.seg, &mut e.rn_board, &mut e.rn_alight);
        }
        st.rail *= all;
        st.c_rail *= all;
        st.e_rail *= all;
        st.rail_d.scale(all);
        let f = all as f32;
        for v in st.pair_rail.iter_mut().chain(st.c_frac.iter_mut()).chain(st.e_frac.iter_mut()).chain(st.st.iter_mut()) {
            *v *= f;
        }
        st.corridors.scale(all);
        st.c_corridors.scale(all);
        true
    }
    pub fn summary(&self, q: usize) -> Vec<f64> {
        self.per(q).map_or(vec![], |s| {
            let (_, c) = crowd_factors(&s.g, &s.net, &s.avg, &s.p);
            vec![s.round as f64, s.trips + s.c_trips + s.e_trips, s.rail + s.c_rail + s.e_rail, s.e_walk, s.ms, self.m.cw.place_pairs as f64, c.segs_over_seats as f64, c.max_load_of_crush as f64, c.p50 as f64, c.p90 as f64, s.rail_d.d, s.rail_d.d_km, s.rail_d.km]
        })
    }
    /// The regional layer's parts in period `q` (T-276): trips and rail trips of the longer trips,
    /// then legs and rail legs of the commutes between zones, then (T-1175) legs, rail legs and
    /// walking legs of the everyday outings across a zone line, and their walking legs with no
    /// rail (the same on any network).
    ///
    /// Then (T-1229) the same four for each everyday purpose, `KIND_COUNT` blocks in `everyday`
    /// kind order (legs, rail legs, walking legs, walking legs with no rail), which add up to the
    /// first four.
    pub fn part_summary(&self, q: usize) -> Vec<f64> {
        let Some(s) = self.per(q) else { return vec![] };
        let mut out = vec![s.trips, s.rail, s.c_trips, s.c_rail, s.e_trips, s.e_rail, s.e_walk, s.e_walk_base];
        let by = self.everyday_by_place(q);
        for k in 0..KIND_COUNT {
            let mut f = [0f64; 4];
            for p in 0..self.m.places.len() {
                let b = &by[3 * (p * KIND_COUNT + k)..][..3];
                let base = self.m.e_base[p * KIND_COUNT + k].1 * (OUTWARD[k][q] + RETURN[k][q]) as f64;
                f[0] += b[0];
                f[1] += b[1];
                f[2] += b[2];
                f[3] += base;
            }
            out.extend(f);
        }
        out
    }
    /// T-1229: the everyday outings across a zone line by home place and purpose in period `q`:
    /// per place (market order), per `everyday` kind, legs, rail legs and walking legs (the
    /// walkers rail takes already out). Empty before the period is solved.
    pub(super) fn everyday_by_place(&self, q: usize) -> Vec<f64> {
        let Some(s) = self.per(q) else { return vec![] };
        let np = self.m.places.len();
        let mut out = vec![0f64; 3 * np * KIND_COUNT];
        for (i, &(n, w)) in self.m.e_base.iter().enumerate() {
            let k = i % KIND_COUNT;
            let share = (OUTWARD[k][q] + RETURN[k][q]) as f64;
            out[3 * i] = n * share;
            out[3 * i + 2] = w * share;
        }
        self.each_e_rail(s, q, |k, p, rail, lost| {
            out[3 * (p * KIND_COUNT + k) + 1] += rail;
            out[3 * (p * KIND_COUNT + k) + 2] -= lost;
        });
        for w in out.iter_mut().skip(2).step_by(3) {
            *w = w.max(0.);
        }
        out
    }
    /// T-276: the commutes' segment loads, boardings and alightings per route node in period `q`.
    pub fn commute_loads(&self, q: usize) -> Vec<f32> {
        self.per(q).filter(|_| !self.m.commutes.is_empty()).map_or(vec![], |s| [&s.c_avg.seg[..], &s.c_avg.rn_board[..], &s.c_avg.rn_alight[..]].concat())
    }
    /// T-1373: the longer trips' segment loads, boardings and alightings per route node in period
    /// `q`, each times its pair's lean (`Market::lean`): a going-out day's longer trips are theirs
    /// plus these, a coming-back day's theirs less these, before the day's factor. Empty without.
    pub fn lean_loads(&self, q: usize) -> Vec<f32> {
        self.per(q).and_then(|s| s.r_avg.as_ref()).map_or(vec![], |l| [&l.seg[..], &l.rn_board[..], &l.rn_alight[..]].concat())
    }
    /// T-1175: the everyday outings across a zone line, as `commute_loads` per everyday purpose
    /// in `everyday` kind order (`KIND_COUNT` blocks of three, zeros for a purpose with none);
    /// empty when the market has none.
    pub fn everyday_loads(&self, q: usize) -> Vec<f32> {
        let Some(s) = self.per(q).filter(|s| !s.e_avg.is_empty()) else { return vec![] };
        let m = s.c_avg.seg.len();
        let zero = vec![0f32; m];
        let mut out = Vec::with_capacity(3 * KIND_COUNT * m);
        for k in 0..KIND_COUNT {
            match s.e_avg.get(k).and_then(|l| l.as_ref()) {
                Some(l) => out.extend([&l.seg[..], &l.rn_board[..], &l.rn_alight[..]].concat()),
                None => (0..3).for_each(|_| out.extend_from_slice(&zero)),
            }
        }
        out
    }
    /// T-378: per corridor, the shares of its longer trips not by rail that fly and that drive
    /// (neither: near places on two landmasses). The same whatever the network.
    pub fn corridor_modes(&self) -> Vec<f32> {
        let m = &self.m;
        let mut out = Vec::with_capacity(2 * m.n_corridors());
        let directs = Directs::of(&m.places);
        for (a, b) in m.each_corridor() {
            out.extend(m.corridor_mode(&directs, a, b));
        }
        out
    }
    /// T-2122: the corridors with longer trips or commutes (the main thread's corridor table,
    /// T-1008), their corridor indices ascending; the `corridor_rows_*` columns come in this order.
    /// The four whole-world arrays they replace (`corridor_pairs` and the like) are a few hundred
    /// MB once the world has thousands of places, nearly all of it pairs with neither.
    pub fn corridor_rows(&self) -> Vec<u32> {
        let mut out = Vec::with_capacity(self.n_rows());
        out.extend(self.table_rows().map(|(i, ..)| i as u32));
        out
    }
    pub fn corridor_rows_pairs(&self) -> Vec<u32> {
        let mut out = Vec::with_capacity(2 * self.n_rows());
        out.extend(self.table_rows().flat_map(|(_, a, b, _)| [a as u32, b as u32]));
        out
    }
    pub fn corridor_rows_trips(&self) -> Vec<f64> {
        let mut out = Vec::with_capacity(self.n_rows());
        out.extend(self.table_rows().map(|(.., t)| t));
        out
    }
    /// As `commute_base` on the rows: empty where it is.
    pub fn corridor_rows_commutes(&self) -> Vec<f64> {
        let cw = &self.m.cw;
        if cw.commuters <= 0. {
            return vec![];
        }
        let mut k = 0;
        let mut out = Vec::with_capacity(self.n_rows());
        out.extend(self.table_rows().map(|(i, ..)| {
            while k < cw.commute.len() && (cw.commute[k].0 as usize) < i {
                k += 1;
            }
            if k < cw.commute.len() && cw.commute[k].0 as usize == i { cw.commute[k].1 } else { 0. }
        }));
        out
    }
    /// As `corridor_modes` on the rows.
    pub fn corridor_rows_modes(&self) -> Vec<f32> {
        let m = &self.m;
        let mut out = Vec::with_capacity(2 * self.n_rows());
        let directs = Directs::of(&m.places);
        out.extend(self.table_rows().flat_map(|(_, a, b, _)| m.corridor_mode(&directs, a, b)));
        out
    }
    /// T-378: per place, its longer trips in period `q` with places that have no zone, each way.
    pub fn elsewhere(&self, q: usize) -> Vec<f64> {
        let s = SHARES.get(q).copied().unwrap_or(0.) as f64;
        self.m.cw.leaving_by.iter().map(|v| v * s).collect()
    }
    /// T-1008: per corridor, its longer trips a day; period `q`'s are these times `corridor_shares(q)[0]`.
    pub fn corridor_trips(&self) -> Vec<f64> {
        let mut out = vec![0f64; self.m.n_corridors()];
        for (&c, &t) in self.m.cw.trips_at.iter().zip(&self.m.cw.trips) {
            out[c as usize] = t;
        }
        out
    }
    /// T-1008: period `q`'s factors of `corridor_trips` and `commute_base`; empty where the period
    /// is not solved here, as `corridor_summary` is.
    pub fn corridor_shares(&self, q: usize) -> Vec<f64> {
        self.per(q).map_or(vec![], |_| vec![SHARES[q] as f64, 0.5 * (TO_WORK[q] + TO_HOME[q]) as f64])
    }
    /// T-1008: period `q`'s corridors with rail, as (corridor, rail trips, commute rail legs): the
    /// pairs of `corridor_summary` and `commute_corridors` with rail, without the rest.
    pub fn corridor_rail(&self, q: usize) -> Vec<f64> {
        let Some(s) = self.per(q) else { return vec![] };
        let commutes = self.m.cw.commuters > 0.;
        let mut out = vec![];
        // (the open places' corridors come in corridor order)
        for (k, i) in self.m.open_corridors() {
            let r = s.corridors.get(k);
            let c = if commutes { s.c_corridors.get(k) } else { 0. };
            if r != 0. || c != 0. {
                out.extend([i as f64, r, c]);
            }
        }
        out
    }
    pub fn seg(&self, q: usize) -> Vec<f32> {
        self.per(q).map_or(vec![], |s| s.avg.seg.clone())
    }
    pub fn board(&self, q: usize) -> Vec<f32> {
        self.per(q).map_or(vec![], |s| s.avg.rn_board.clone())
    }
    pub fn alight(&self, q: usize) -> Vec<f32> {
        self.per(q).map_or(vec![], |s| s.avg.rn_alight.clone())
    }
    /// As `DemandApi::fare_extra`.
    pub fn fare_extra(&self, q: usize) -> Vec<f32> {
        self.per(q).map_or(vec![], |s| fare_x_usd(&s.avg.rn_fare_x, self.base.fare_w))
    }
    /// As `DemandApi::transfers_in`.
    pub fn transfers_in(&self, q: usize) -> Vec<f32> {
        self.per(q).map_or(vec![], |s| transfers_in(&s.g, &s.xi, &s.xfer))
    }
    /// `transfers` on a kind of day (T-260).
    pub fn transfers_day(&self, sel: &[u8], day: u32) -> Vec<f32> {
        let mut out = vec![];
        for (q, st) in self.per.iter().enumerate() {
            let Some(st) = st else { continue };
            let at = out.len();
            transfer_pairs(&st.g, &st.xi, &st.xfer, sel, &mut out);
            scale_triples(&mut out[at..], load_factor(day as usize, LONGER, q));
        }
        out
    }
    pub fn entries(&self, q: usize) -> Vec<f32> {
        self.per(q).map_or(vec![], |s| s.avg.entry.clone())
    }
    /// Bit q set for each period solved.
    pub fn solved_mask(&self) -> u32 {
        (0..PERIODS).filter(|&q| self.per(q).is_some()).fold(0, |m, q| m | 1 << q)
    }
    /// T-377: `selection_flows_day`'s all-mode numbers alone (rail zero) over the periods in
    /// `mask`, solved or not: a market composed of every open pack answers them, its parts add
    /// their rail (`selection_rail_day`).
    pub fn selection_all_day(&self, market: u32, xy: &[f32], share: &[f32], day: u32, mask: u32) -> Vec<f64> {
        self.selection(market, xy, share, day, mask, false)
    }
    /// T-1204: which corridors' figures this market reads; compositions of one world with rows
    /// (and their parts) share them, so `corridor_pairs`, `corridor_trips`, `commute_base` and
    /// `corridor_modes` are the same for every market with the same id.
    pub fn corridors_id(&self) -> u32 {
        self.m.cw.id
    }
    /// T-377: whether `other` is this market's composition (a part beside it, or itself).
    pub fn same_market(&self, other: &IntercityApi) -> bool {
        Rc::ptr_eq(&self.m, &other.m)
    }
}
