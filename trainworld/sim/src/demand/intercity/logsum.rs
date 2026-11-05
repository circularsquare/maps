use super::*;

/// T-718: rail between two zones as the local layers choose it (T-710): the station pairs from
/// access list `a` to `b` (weights exp(-`beta` x access), `add_access`) at the station choice's
/// scale `th`, each pair's weight into `terms`. Returns rail's perceived cost, their logsum (the
/// first boarding's, access and egress included, as the mode choice then damps it, T-495), and the
/// sum of the weights (0: no route).
#[cfg(test)]
pub(super) fn rail_logsum(r: &[f32], s: usize, fare: f32, a: &[(usize, f32)], b: &[(usize, f32)], beta: f32, th: f32, terms: &mut Vec<(u32, u32, f64)>) -> (f32, f64) {
    rail_logsum_x(r, s, fare, a, b, beta, th, None, terms)
}
/// T-1286: the fares of trips from one country weighed at a factor of the market's (the player's
/// fare scale there over its income's), on each station pair's strategy as searched: its expected
/// fare less the base (`IntercityApi::fare_col`, S x S as `Skims::r`, or a column at a time,
/// T-1545), perceived minutes.
pub(super) struct FareX<'a> {
    pub(super) dist: &'a [f32],
    pub(super) base: f32,
}
impl FareX<'_> {
    /// The perceived minutes of the fare from station `a` to `b` in the market's weight.
    #[inline]
    pub(super) fn fare(&self, a: usize, b: usize, s: usize) -> f32 {
        self.fare_of(self.dist[a * s + b])
    }
    /// The same for a trip whose fare less the base is `d`.
    #[inline]
    pub(super) fn fare_of(&self, d: f32) -> f32 {
        self.base + d
    }
}
/// `rail_logsum` with the fares weighed at `1 + x.1` (`FareX`).
#[allow(clippy::too_many_arguments)]
pub(super) fn rail_logsum_x(r: &[f32], s: usize, fare: f32, a: &[(usize, f32)], b: &[(usize, f32)], beta: f32, th: f32, x: Option<(&FareX, f32)>, terms: &mut Vec<(u32, u32, f64)>) -> (f32, f64) {
    rail_logsum_by(a, b, beta, th, terms, |sa, j| {
        let c = r[sa * s + b[j].0];
        match x {
            None => c + fare,
            Some((fx, d)) => c + fare + d * fx.fare(sa, b[j].0, s),
        }
    })
}
/// `rail_logsum_x` on the route costs and fare km towards `b`'s stations as columns: `cols[j]` for
/// `b[j]`, each by the station the trip starts from (T-1545).
#[allow(clippy::too_many_arguments)]
pub(super) fn rail_logsum_cols(cols: &[(&[f32], &[f32])], fare: f32, a: &[(usize, f32)], b: &[(usize, f32)], beta: f32, th: f32, x: Option<(&FareX, f32)>, terms: &mut Vec<(u32, u32, f64)>) -> (f32, f64) {
    rail_logsum_by(a, b, beta, th, terms, |sa, j| {
        let (r, km) = cols[j];
        match x {
            None => r[sa] + fare,
            Some((fx, d)) => r[sa] + fare + d * fx.fare_of(km[sa]),
        }
    })
}
/// T-1520: rail between two zones for riders of every income group from one logsum, taken at the
/// groups' mean fare factor `d` and scale `th`: each group's cost is that logsum moved to first
/// order by its own factor (the station pairs' expected fare, `fare`) and scale (`spread`, the
/// expected station pair's cost over the logsum: d logsum / d th = spread / th). The station pairs'
/// shares (`terms`) are the mean's for every group.
#[derive(Clone, Copy, Debug)]
pub(super) struct RailMix {
    pub(super) c: f32,
    pub(super) sum: f64,
    pub(super) fare: f32,
    pub(super) spread: f32,
    pub(super) d: f32,
    pub(super) th: f32,
}
impl RailMix {
    /// Rail's perceived cost for riders whose fares weigh `1 + d` and whose scale is `th`.
    #[inline]
    pub(super) fn at(&self, d: f32, th: f32) -> f32 {
        if self.sum <= 0. {
            return f32::INFINITY;
        }
        self.c + (d - self.d) * self.fare + (th - self.th) / self.th * self.spread
    }
}
/// `rail_logsum_cols` as a `RailMix` at fare factor `1 + d` (none weighed with no `fx`); `fares`:
/// scratch.
#[allow(clippy::too_many_arguments)]
pub(super) fn rail_mix_cols(cols: &[(&[f32], &[f32])], fare: f32, a: &[(usize, f32)], b: &[(usize, f32)], beta: f32, th: f32, fx: Option<&FareX>, d: f32, terms: &mut Vec<(u32, u32, f64)>, fares: &mut Vec<f32>) -> RailMix {
    rail_mix_by(a, b, beta, th, d, terms, fares, |sa, j| {
        let (r, km) = cols[j];
        let f = fx.map_or(0., |fx| fx.fare_of(km[sa]));
        (r[sa] + fare + d * f, f)
    })
}
/// `rail_logsum_x` as a `RailMix` (`rail_mix_cols` on a whole table).
#[allow(clippy::too_many_arguments)]
pub(super) fn rail_mix_x(r: &[f32], s: usize, fare: f32, a: &[(usize, f32)], b: &[(usize, f32)], beta: f32, th: f32, fx: Option<&FareX>, d: f32, terms: &mut Vec<(u32, u32, f64)>, fares: &mut Vec<f32>) -> RailMix {
    rail_mix_by(a, b, beta, th, d, terms, fares, |sa, j| {
        let f = fx.map_or(0., |fx| fx.fare(sa, b[j].0, s));
        (r[sa * s + b[j].0] + fare + d * f, f)
    })
}
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub(super) fn rail_mix_by(a: &[(usize, f32)], b: &[(usize, f32)], beta: f32, th: f32, d: f32, terms: &mut Vec<(u32, u32, f64)>, fares: &mut Vec<f32>, cost: impl Fn(usize, usize) -> (f32, f32)) -> RailMix {
    terms.clear();
    fares.clear();
    let k = (th / beta) as f64;
    let mut top = f64::NEG_INFINITY;
    let mut lb = [0f64; ACCESS_MAX];
    for (l, &(_, wb)) in lb.iter_mut().zip(b) {
        *l = k * (wb as f64).ln();
    }
    for &(sa, wa) in a {
        if wa <= 0. {
            continue;
        }
        let la = k * (wa as f64).ln();
        for (j, &(sb, wb)) in b.iter().enumerate() {
            let (cost, f) = cost(sa, j);
            if sa == sb || wb <= 0. || !cost.is_finite() {
                continue;
            }
            let lw = la + (if j < ACCESS_MAX { lb[j] } else { k * (wb as f64).ln() }) - (th * cost) as f64;
            top = top.max(lw);
            terms.push((sa as u32, sb as u32, lw));
            fares.push(f);
        }
    }
    if terms.is_empty() {
        return RailMix { c: f32::INFINITY, sum: 0., fare: 0., spread: 0., d, th };
    }
    let (mut sum, mut slw, mut sf) = (0f64, 0f64, 0f64);
    for (t, &f) in terms.iter_mut().zip(fares.iter()) {
        let lw = t.2;
        t.2 = (lw - top).exp();
        sum += t.2;
        slw += t.2 * (lw - top);
        sf += t.2 * f as f64;
    }
    let c = -(top + sum.ln()) / th as f64;
    // the expected -lw / th less the logsum: (ln sum - E[lw - top]) / th, never below 0
    let spread = ((sum.ln() - slw / sum) / th as f64).max(0.);
    RailMix { c: c as f32, sum, fare: (sf / sum) as f32, spread: spread as f32, d, th }
}
/// The logsum of `rail_logsum_x` with each station pair's rail cost, the base fare included, from
/// `cost(sa, j)` for `b[j]`.
#[inline(always)]
pub(super) fn rail_logsum_by(a: &[(usize, f32)], b: &[(usize, f32)], beta: f32, th: f32, terms: &mut Vec<(u32, u32, f64)>, cost: impl Fn(usize, usize) -> f32) -> (f32, f64) {
    terms.clear();
    let k = (th / beta) as f64;
    let mut top = f64::NEG_INFINITY;
    // the egress weights' logs, once a pair rather than once a station pair (T-761)
    let mut lb = [0f64; ACCESS_MAX];
    for (l, &(_, wb)) in lb.iter_mut().zip(b) {
        *l = k * (wb as f64).ln();
    }
    for &(sa, wa) in a {
        if wa <= 0. {
            continue;
        }
        let la = k * (wa as f64).ln();
        for (j, &(sb, wb)) in b.iter().enumerate() {
            // a rail journey must actually board a train
            let cost = cost(sa, j);
            if sa == sb || wb <= 0. || !cost.is_finite() {
                continue;
            }
            let lw = la + (if j < ACCESS_MAX { lb[j] } else { k * (wb as f64).ln() }) - (th * cost) as f64;
            top = top.max(lw);
            terms.push((sa as u32, sb as u32, lw));
        }
    }
    if terms.is_empty() {
        return (f32::INFINITY, 0.);
    }
    let mut sum = 0f64;
    for t in terms.iter_mut() {
        t.2 = (t.2 - top).exp();
        sum += t.2;
    }
    (-((top + sum.ln()) / th as f64) as f32, sum)
}
/// Items `0..n` by zone (`zone(i)`, of `nz` zones; `usize::MAX` leaves an item out), each zone's
/// in their own order: offsets and items.
pub(super) fn by_zone(nz: usize, n: usize, zone: impl Fn(usize) -> usize) -> (Vec<u32>, Vec<u32>) {
    let mut off = vec![0u32; nz + 1];
    for i in 0..n {
        let z = zone(i);
        if z != usize::MAX {
            off[z + 1] += 1;
        }
    }
    for z in 0..nz {
        off[z + 1] += off[z];
    }
    let mut fill = off.clone();
    let mut ix = vec![0u32; off[nz] as usize];
    for i in 0..n {
        let z = zone(i);
        if z != usize::MAX {
            ix[fill[z] as usize] = i as u32;
            fill[z] += 1;
        }
    }
    (off, ix)
}
/// T-1735: `l`'s figures with every journey turned round onto the same lines the other way, added
/// to `out`: a segment's riders onto the segment between the same stops the other way, boardings
/// onto alightings at the same stop and the reverse, entries onto exits.
pub(super) fn mirror_into(g: &Graph, l: &Loads, out: &mut Loads) {
    let (s, nr) = (g.n_st, g.n - g.n_st);
    let mut r = 0;
    while r < nr {
        let line = g.rn_line[r];
        let mut e = r;
        while e < nr && g.rn_line[e] == line {
            e += 1;
        }
        // a line's route nodes: its stops one way, then the other way
        let len = (e - r) / 2;
        for d in 0..2 {
            let (here, back) = (r + d * len, r + (1 - d) * len);
            for i in 0..len {
                let k = here + i;
                out.seg[back + (2 * len - 2 - i) % len] += l.seg[k];
                out.rn_board[back + len - 1 - i] += l.rn_alight[k];
                out.rn_alight[back + len - 1 - i] += l.rn_board[k];
                out.rn_fare_x[k] += l.rn_fare_x[k];
            }
        }
        r = e;
    }
    for x in 0..s {
        out.board[x] += l.alight[x];
        out.alight[x] += l.board[x];
        out.entry[x] += l.exit[x];
        out.exit[x] += l.entry[x];
    }
    out.pax_min += l.pax_min;
}
/// Where `IntercityApi::pass_window` gets a destination's route costs and strategy (T-1545): a
/// table of every destination's, or a search for it alone.
pub(super) enum Columns<'a> {
    Skims(&'a Skims),
    Search(DestSearch<'a>),
}
impl<'a> Columns<'a> {
    /// Destination `dst`'s route costs from each station into `r`, the same with the planned first
    /// wait into `re` where any differs (`has_re`, T-1430), and its strategy (none if not searched).
    pub(super) fn fill(&mut self, dst: usize, r: &mut [f32], re: &mut [f32], has_re: &mut bool) -> Option<Held<'a>> {
        let put = |r: &[f32], re: &mut [f32], has_re: &mut bool, entry: &[(u32, f32)]| {
            *has_re = !entry.is_empty();
            if *has_re {
                re.copy_from_slice(r);
                for &(src, c) in entry {
                    re[src as usize] = c;
                }
            }
        };
        match self {
            Columns::Skims(sk) => {
                let sk: &'a Skims = sk;
                let s = sk.s;
                for (src, v) in r.iter_mut().enumerate() {
                    *v = sk.r[src * s + dst];
                }
                let slot = sk.slot[dst];
                if slot == u32::MAX {
                    *has_re = false;
                    return None;
                }
                put(r, re, has_re, &sk.entry[slot as usize]);
                Some(Held::Ref(sk.strategy(slot as usize)))
            }
            Columns::Search(q) => {
                let mut d = q.run_col(dst, r);
                put(r, re, has_re, &std::mem::take(&mut d.entry));
                Some(Held::Own(d))
            }
        }
    }
    /// Destination `dst`'s strategy alone.
    pub(super) fn strategy(&mut self, dst: usize) -> Option<Held<'a>> {
        match self {
            Columns::Skims(sk) => {
                let sk: &'a Skims = sk;
                let slot = sk.slot[dst];
                (slot != u32::MAX).then(|| Held::Ref(sk.strategy(slot as usize)))
            }
            Columns::Search(q) => Some(Held::Own(q.run(dst, None))),
        }
    }
}
thread_local! {
    /// The bytes of searched strategies `IntercityApi::pass_window` keeps from a station's search to
    /// its assignment (T-1545); the rest are searched again for it. A cell so benches can vary it.
    pub(super) static WINDOW_STRATEGY_BYTES: std::cell::Cell<usize> = const { std::cell::Cell::new(192 << 20) };
    /// `pass_window`'s most read slots and trip slots at once, most bytes of strategies kept, and
    /// searches and searches again, over the passes since it was reset (benches).
    pub(super) static WINDOW_STATS: std::cell::Cell<[usize; 5]> = const { std::cell::Cell::new([0; 5]) };
}
/// T-1735: the period whose service a window's minutes are measured on (the midday's, the most of
/// the day).
pub(super) const WINDOW_PERIOD: usize = 1;
/// T-1735: per node of `g`, the perceived minutes from the nearest station marked in `from` along
/// its edges at free flow (a station's first, so `take(n_st)` gives the stations').
pub(super) fn rail_minutes(g: &Graph, from: &[bool]) -> Vec<f32> {
    let mut d = vec![f32::INFINITY; g.n];
    let mut heap = std::collections::BinaryHeap::new();
    for (s, _) in from.iter().enumerate().filter(|e| *e.1) {
        d[s] = 0.;
        heap.push((std::cmp::Reverse(0u32), s as u32));
    }
    while let Some((std::cmp::Reverse(k), v)) = heap.pop() {
        let v = v as usize;
        let dv = f32::from_bits(k);
        if dv > d[v] {
            continue;
        }
        for e in g.adj_off[v] as usize..g.adj_off[v + 1] as usize {
            let (to, c) = (g.adj_to[e] as usize, dv + g.adj_cost[e].max(0.));
            if c < d[to] {
                d[to] = c;
                // (non-negative floats order as their bits)
                heap.push((std::cmp::Reverse(c.to_bits()), to as u32));
            }
        }
    }
    d
}
pub(super) struct Period {
    pub(super) net: super::super::network::Network,
    pub(super) p: Params,
    pub(super) g: Graph,
    /// every part's loads together (crowding reads these), and the commutes' alone (T-276)
    pub(super) avg: Loads,
    pub(super) c_avg: Loads,
    pub(super) rail: f64,
    /// T-1300: `Demand::rail_d`, averaged like the rail trips
    pub(super) rail_d: FareSense,
    pub(super) trips: f64,
    pub(super) c_rail: f64,
    pub(super) c_trips: f64,
    pub(super) c_frac: Vec<f32>,
    pub(super) corridors: Sparse,
    pub(super) c_corridors: Sparse,
    /// T-1373: the longer trips' loads each times its pair's lean (`Market::lean`), averaged like
    /// the commutes'; none where the lean's table would be too large (`LEAN_OD_BYTES`)
    pub(super) r_avg: Option<Loads>,
    /// Rail trips of each directed pair in this period (T-118), averaged over rounds like the loads.
    pub(super) pair_rail: Vec<f32>,
    /// Per station, rail trips starting here by destination zone and ending here by origin zone
    /// (T-118, T-120), the entries of `IntercityApi::st_pat` (T-761).
    pub(super) st: Vec<f32>,
    pub(super) round: usize,
    pub(super) ms: f64,
    pub(super) fac: Vec<f32>,
    pub(super) uniform: bool,
    /// `kernel::settled` (T-434)
    pub(super) seen: Vec<f32>,
    pub(super) xi: XferIndex,
    pub(super) xfer: Vec<f32>,
    /// T-1175: the everyday pairs' loads by purpose, averaged like the commutes'; their legs, rail
    /// legs and walking legs in this period (every pair, served or not), and each served pair's
    /// rail share
    pub(super) e_avg: Vec<Option<Loads>>,
    pub(super) e_trips: f64,
    pub(super) e_rail: f64,
    pub(super) e_walk: f64,
    pub(super) e_walk_base: f64,
    pub(super) e_frac: Vec<f32>,
}
/// A period's figures by open corridor, its nonzero entries only (T-1611): a cluster's pairs touch
/// few of the world's corridors, and a dense table a period and part was most of a market part's
/// memory on all of China (250,000 corridors, 4 MB a period and part). Read as the dense table it
/// stands for, to the bit.
pub(super) struct Sparse {
    pub(super) k: Vec<u32>,
    pub(super) v: Vec<f64>,
}
impl Sparse {
    pub(super) fn of(dense: &[f64]) -> Sparse {
        let (mut k, mut v) = (vec![], vec![]);
        for (i, &x) in dense.iter().enumerate() {
            if x != 0. {
                k.push(i as u32);
                v.push(x);
            }
        }
        k.shrink_to_fit();
        v.shrink_to_fit();
        Sparse { k, v }
    }
    pub(super) fn get(&self, i: usize) -> f64 {
        self.k.binary_search(&(i as u32)).map_or(0., |j| self.v[j])
    }
    /// Each entry `a += (b - a) * w` towards `dense`, as on the dense table.
    pub(super) fn toward(&mut self, dense: &[f64], w: f64) {
        let (mut k, mut v) = (Vec::with_capacity(self.k.len()), Vec::with_capacity(self.k.len()));
        let mut j = 0;
        for (i, &b) in dense.iter().enumerate() {
            let held = j < self.k.len() && self.k[j] as usize == i;
            if held || b != 0. {
                let a = if held { self.v[j] } else { 0. };
                k.push(i as u32);
                v.push(a + (b - a) * w);
            }
            j += held as usize;
        }
        k.shrink_to_fit();
        v.shrink_to_fit();
        *self = Sparse { k, v };
    }
    pub(super) fn scale(&mut self, f: f64) {
        self.v.iter_mut().for_each(|x| *x *= f);
    }
}
