use super::*;

// ---------------------------------------------------------------------------------------------
// 6. Assignment

#[derive(Clone)]
pub struct Loads {
    /// Riders per period on the segment starting at each route node.
    pub seg: Vec<f32>,
    /// Boardings and alightings per station, transfers included.
    pub board: Vec<f32>,
    pub alight: Vec<f32>,
    /// The same per route node: riders boarding or leaving that line, direction and stop.
    pub rn_board: Vec<f32>,
    pub rn_alight: Vec<f32>,
    /// Rail trips starting (first boarding) and ending (last alighting) at each station: entries
    /// and exits, transfers not counted.
    pub entry: Vec<f32>,
    pub exit: Vec<f32>,
    pub pax_min: f64,
    /// Trips taken towards a station the strategy search never reached (`Skims::slot` is `NONE`): they are in
    /// `entry` and `exit` but walked nowhere, so no rail trip is made of them. Nonzero means a lost flow (T-3923).
    pub dropped: f64,
    /// The fare curve's extra on the km ridden from each route node, perceived minutes
    /// (`board_extra`; all 0 while the curve is a straight line).
    pub rn_fare_x: Vec<f32>,
    /// T-3578: the base's own multiplier past 1 on the trips, on the km ridden from each route node,
    /// perceived minutes (`board_base`; all 0 while no line has one). Kept apart from `rn_fare_x`: income shares
    /// it out with the base fares, not by line.
    pub rn_base_x: Vec<f32>,
}

impl Loads {
    pub fn zero(n_st: usize, n_rn: usize) -> Loads {
        Loads { seg: vec![0.0; n_rn], board: vec![0.0; n_st], alight: vec![0.0; n_st], rn_board: vec![0.0; n_rn], rn_alight: vec![0.0; n_rn], entry: vec![0.0; n_st], exit: vec![0.0; n_st], pax_min: 0.0, dropped: 0.0, rn_fare_x: vec![0.0; n_rn], rn_base_x: vec![0.0; n_rn] }
    }
    /// Every figure times `k` (T-1538's estimate of one period from another).
    pub fn scaled(&self, k: f32) -> Loads {
        let s = |v: &[f32]| v.iter().map(|x| x * k).collect::<Vec<f32>>();
        Loads { seg: s(&self.seg), board: s(&self.board), alight: s(&self.alight), rn_board: s(&self.rn_board), rn_alight: s(&self.rn_alight), entry: s(&self.entry), exit: s(&self.exit), pax_min: self.pax_min * k as f64, dropped: self.dropped * k as f64, rn_fare_x: s(&self.rn_fare_x), rn_base_x: s(&self.rn_base_x) }
    }
}

#[cfg(test)]
thread_local! {
    /// T-3923: a test thread may drop trips only after `KNOWN_DROPS` is set (the cases that do are listed in notes/T-3923.md).
    pub static KNOWN_DROPS: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// Every (station, trips) dropped on this thread.
    pub static DROP_LOG: std::cell::RefCell<Vec<(usize, f64)>> = const { std::cell::RefCell::new(vec![]) };
}

/// A destination's trips found no strategy: counted in `Loads::dropped`, and in a test build also logged, and a
/// panic unless the test is a known case, so every demand test is a check that nothing is dropped.
#[inline(never)]
pub fn note_dropped(dst: usize, trips: f64) {
    #[cfg(test)]
    {
        DROP_LOG.with(|l| l.borrow_mut().push((dst, trips)));
        if !KNOWN_DROPS.with(|k| k.get()) && trips > 0.0 {
            panic!("T-3923: {trips} trips taken towards station {dst}, which was not searched, were dropped");
        }
    }
    let _ = (dst, trips);
}

/// Each destination's strategy is walked from its farthest nodes towards it. Station flow
/// splits over useful departures; route flow follows its best ride/alight/walking-transfer edge.
/// Work is O(edges) per destination, independent of the number of OD pairs carrying flow.
pub fn assign(g: &Graph, sk: &Skims, od: &[f32]) -> Loads {
    assign_inner(g, sk, od, None)
}

/// `assign`, also adding the period's transfers into `xfer` (laid out by `ix`, T-176). The loads
/// are `assign`'s to the bit: the transfers are counted beside them, never mixed in.
pub fn assign_with_transfers(g: &Graph, sk: &Skims, od: &[f32], ix: &XferIndex, xfer: &mut [f32]) -> Loads {
    debug_assert_eq!(xfer.len(), ix.len);
    if cfg!(test) && XFER_OFF.load(std::sync::atomic::Ordering::Relaxed) {
        return assign(g, sk, od);
    }
    assign_inner(g, sk, od, Some((ix, xfer)))
}

/// Test switch (T-176's cost measurement): `assign_with_transfers` counts nothing.
pub static XFER_OFF: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Where transfers are counted (T-176): one slot per (arriving edge, departure) pair a rider can
/// change between. An arriving edge is a route node's edge to a station: alighting at its own
/// station, or a walking transfer to one nearby; the departures are that station's boarding
/// edges. So a slot is a pair of route nodes (line, direction and stop in, and out), which is
/// platform to platform. Depends only on the graph's edges, the same in every period and round.
pub struct XferIndex {
    /// Per edge: for an arriving edge, its first slot (its station's departures follow in edge
    /// order); `NONE` for every other edge.
    pub at: Vec<u32>,
    pub len: usize,
}

pub fn xfer_index(g: &Graph) -> XferIndex {
    let s = g.n_st;
    let mut at = vec![NONE; g.adj_to.len()];
    let mut len = 0usize;
    for v in s..g.n {
        for e in g.adj_off[v] as usize..g.adj_off[v + 1] as usize {
            let to = g.adj_to[e] as usize;
            if to < s {
                at[e] = len as u32;
                len += (g.adj_off[to + 1] - g.adj_off[to]) as usize;
            }
        }
    }
    XferIndex { at, len }
}

/// The transfers in `xfer` (laid out by `ix`) added up by the route node riders board at after
/// changing: `transfer_pairs` with every node marked, summed by its second node (T-1530).
pub fn transfers_in(g: &Graph, ix: &XferIndex, xfer: &[f32]) -> Vec<f32> {
    let s = g.n_st;
    let mut out = vec![0.0f32; g.n - s];
    if xfer.len() != ix.len {
        return out;
    }
    for v in s..g.n {
        for e in g.adj_off[v] as usize..g.adj_off[v + 1] as usize {
            let slot = ix.at[e];
            if slot == NONE {
                continue;
            }
            let to = g.adj_to[e] as usize;
            for (k, d) in (g.adj_off[to] as usize..g.adj_off[to + 1] as usize).enumerate() {
                let x = xfer[slot as usize + k];
                if x > 0.0 {
                    out[g.adj_to[d] as usize - s] += x;
                }
            }
        }
    }
    out
}

/// The transfers in `xfer` (laid out by `ix`) with either end at a route node marked in `sel`
/// (one byte per route node), as triples (route node in, route node out, riders), appended to
/// `out`. Route nodes are indices from 0 (node - n_st), the order of `Loads::seg`.
pub fn transfer_pairs(g: &Graph, ix: &XferIndex, xfer: &[f32], sel: &[u8], out: &mut Vec<f32>) {
    let s = g.n_st;
    if xfer.len() != ix.len || sel.len() != g.n - s {
        return;
    }
    for v in s..g.n {
        let a = sel[v - s] != 0;
        for e in g.adj_off[v] as usize..g.adj_off[v + 1] as usize {
            let slot = ix.at[e];
            if slot == NONE {
                continue;
            }
            let to = g.adj_to[e] as usize;
            for (k, d) in (g.adj_off[to] as usize..g.adj_off[to + 1] as usize).enumerate() {
                let x = xfer[slot as usize + k];
                let w = g.adj_to[d] as usize - s;
                if x > 0.0 && (a || sel[w] != 0) {
                    out.extend([(v - s) as f32, w as f32, x]);
                }
            }
        }
    }
}

pub(super) fn assign_inner(g: &Graph, sk: &Skims, od: &[f32], mut xf: Option<(&XferIndex, &mut [f32])>) -> Loads {
    let s = sk.s;
    let mut a = DestAssign::new(g, sk.most_change_nodes(), xf.is_some());
    for dst in 0..s {
        if !a.take(dst, |src| od[src * s + dst]) { continue; }
        let slot = sk.slot[dst] as usize;
        if slot == NONE as usize {
            a.l.dropped += a.taken;
            note_dropped(dst, a.taken);
            continue;
        }
        a.walk(g, dst, sk.strategy(slot), xf.as_mut().map(|(ix, x)| (*ix, &mut **x)));
    }
    a.l
}

/// `assign`'s work one destination at a time, each destination's strategy handed in (T-761):
/// destinations taken in station order give `assign`'s loads to the bit.
pub struct DestAssign {
    pub l: Loads,
    pub(super) flow: Vec<f32>,
    /// T-4041: per station, the riders of the column taken whose first boarding is an inter-city
    /// trip's (`take_entry`), boarded by `Strategy::entry_dep`; empty while none is taken.
    pub(super) eflow: Vec<f32>,
    // T-176: riders arriving at each station towards this destination, as a list per station
    // and change node (first record, then each record's next): the arriving edge's first slot and
    // the riders. A station is walked after every node that feeds it, so its list is whole by then.
    pub(super) head: Vec<u32>,
    pub(super) rec: Vec<(u32, f32, u32)>,
    pub(super) use_k: Vec<(u32, f32)>,
    pub(super) dep: Vec<(u32, f32)>,
    pub(super) counting: bool,
    /// The riders the last `take` took (all stations' columns together).
    pub(super) taken: f64,
    /// Whether the fare curve's extra is put on the loads (`Loads::rn_fare_x`); on by default.
    pub fares: bool,
    pub(super) km: Vec<f32>,
    pub(super) fx: Vec<f32>,
    pub(super) mb: Vec<f32>,
    /// T-3758: the fares under fare zones
    pub(super) zw: ZoneWork,
}

impl DestAssign {
    /// `change_nodes`: the most change nodes of a destination to come (more grows the tables);
    /// `counting`: transfers will be counted.
    pub fn new(g: &Graph, change_nodes: usize, counting: bool) -> DestAssign {
        let (s, n) = (g.n_st, g.n);
        DestAssign {
            l: Loads::zero(s, n - s),
            flow: vec![0f32; n + change_nodes],
            eflow: vec![],
            head: if counting { vec![NONE; s + change_nodes] } else { vec![] },
            rec: vec![],
            use_k: vec![],
            dep: vec![],
            counting,
            taken: 0.0,
            fares: true,
            km: vec![],
            fx: vec![],
            mb: vec![],
            zw: ZoneWork::default(),
        }
    }
    /// The riders the last `take` took.
    pub fn taken(&self) -> f64 {
        self.taken
    }
    /// Take destination `dst`'s column of the OD, `col(src)` riders from each station; false
    /// when nobody rides there (there is nothing to walk).
    #[inline(always)]
    pub fn take(&mut self, dst: usize, col: impl Fn(usize) -> f32) -> bool {
        let s = self.l.entry.len();
        self.flow.fill(0.0);
        self.eflow.clear();
        let mut total = 0.0;
        for src in 0..s {
            let v = col(src);
            self.flow[src] = v;
            self.l.entry[src] += v;
            total += v;
        }
        self.l.exit[dst] += total;
        self.taken = total as f64;
        total != 0.0
    }
    /// T-4041: add to the column `take` took towards `dst` the inter-city trips `col(src)`, whose
    /// first boarding is at the planned waits (`Strategy::entry_dep`); false when it adds none.
    pub fn take_entry(&mut self, dst: usize, col: impl Fn(usize) -> f32) -> bool {
        let s = self.l.entry.len();
        self.eflow.resize(s, 0.0);
        let mut total = 0.0;
        for src in 0..s {
            let v = col(src);
            self.eflow[src] += v;
            self.l.entry[src] += v;
            total += v;
        }
        self.l.exit[dst] += total;
        self.taken += total as f64;
        total != 0.0
    }
    /// Walk the column taken towards `dst` along its strategy.
    pub fn walk(&mut self, g: &Graph, dst: usize, st: Strategy, xf: Option<(&XferIndex, &mut [f32])>) {
        let (s, n) = (g.n_st, g.n);
        let np = st.change.len();
        if self.flow.len() < n + np {
            self.flow.resize(n + np, 0.0);
        }
        if self.counting {
            if self.head.len() < s + np {
                self.head.resize(s + np, NONE);
            }
            self.head.fill(NONE);
            self.rec.clear();
        }
        // T-3718: riders whose trip starts with a walk board at the station walked to (their entry
        // stays where the trip starts, as an exit after a walk is counted where it ends)
        st.fold_start(&mut self.flow);
        let entering = !self.eflow.is_empty();
        if entering {
            st.fold_start(&mut self.eflow);
            // (the fares below spread an inter-city start's along the plain strategy: a lie in where
            // income lands, not in what a trip pays)
            for (f, e) in self.flow.iter_mut().zip(self.eflow.iter_mut()) {
                *f += *e;
            }
        }
        let (bent, based) = (!g.fare_curve.flat(), !g.rn_base_mul.is_empty());
        if self.fares && (bent || based) {
            // the column taken is still in the stations' flows: the walk below adds the changes
            trip_km(g, st, dst, &mut self.km, &mut self.dep);
            let flow = &self.flow;
            // T-3758: under fare zones each trip's fare follows its rides
            if !g.fare_curve.zones.is_empty() {
                let hops = if g.rn_kmw.is_empty() { &g.rn_km } else { &g.rn_kmw };
                self.zw.run(g, st, dst, &g.rn_km, &g.rn_kmw, &mut self.dep);
                // T-3784: a change inside a zone run is refunded the transfer fare its counted transfer charges
                let xfer = if xf.is_some() { g.xfer_fare } else { 0.0 };
                self.zw.spread(g, st, dst, |src| flow[src], hops, xfer, &mut self.dep, &mut self.l.rn_fare_x);
            } else if bent {
                board_extra(g, st, dst, |src| flow[src], &self.km, &mut self.dep, &mut self.fx, &mut self.l.rn_fare_x);
            }
            // T-3578
            if based {
                trip_base_mul(g, st, dst, &mut self.mb, &mut self.dep);
                board_base(g, st, dst, |src| flow[src], &self.km, &self.mb, &mut self.dep, &mut self.fx, &mut self.l.rn_base_x);
            }
        }
        if entering {
            self.board_entries(g, st);
        }
        let mut w = DestWalk { g, s, n, dst, bits: st.board, cn: st.change, sh: st.share, flow: &mut self.flow, l: &mut self.l, head: &mut self.head, rec: &mut self.rec, use_k: &mut self.use_k, dep: &mut self.dep, xf };
        match (st.next, st.order) {
            (NodeIds::Short(next), NodeIds::Short(order)) => w.run(next, order),
            (NodeIds::Long(next), NodeIds::Long(order)) => w.run(next, order),
            _ => unreachable!("a destination's ids in two widths"),
        }
    }
}

impl DestAssign {
    /// T-4041: the inter-city starts in `eflow` (also in `flow`) boarded where the strategy's
    /// planned-wait departures differ from its own, as the walk boards a station's riders.
    fn board_entries(&mut self, g: &Graph, st: Strategy) {
        let s = g.n_st;
        let next = |v: usize| match st.next {
            NodeIds::Short(x) => x[v].ix(),
            NodeIds::Long(x) => x[v].ix(),
        };
        for &(v, _) in st.entry_dep.at.iter() {
            let v = v as usize;
            let f = std::mem::take(&mut self.eflow[v]);
            if f == 0.0 {
                continue;
            }
            self.flow[v] -= f;
            self.l.board[v] += f;
            for &(r, sh) in st.entry_dep.of(v).unwrap() {
                let (r, x) = (r as usize, f * sh);
                self.l.rn_board[r] += x;
                let nx = next(s + r);
                if nx >= s && nx < g.n {
                    self.flow[s + r] += x;
                } else {
                    // a boarding rides a stop first (T-1055)
                    self.flow[g.rn_next[r] as usize] += x;
                    self.l.seg[r] += x;
                    self.l.pax_min += (x * g.rn_ride[r]) as f64;
                }
            }
        }
        self.eflow.clear();
    }
}

/// One destination's part of `assign_inner`: its flows walked back along the settle order.
pub(super) struct DestWalk<'a> {
    pub(super) g: &'a Graph,
    pub(super) s: usize,
    pub(super) n: usize,
    pub(super) dst: usize,
    pub(super) bits: &'a [u64],
    pub(super) cn: &'a ChangeNodes,
    pub(super) sh: &'a Shares,
    pub(super) flow: &'a mut [f32],
    pub(super) l: &'a mut Loads,
    pub(super) head: &'a mut [u32],
    pub(super) rec: &'a mut Vec<(u32, f32, u32)>,
    pub(super) use_k: &'a mut Vec<(u32, f32)>,
    /// The departures a station or change node accepts, (edge, trains an hour), in edge order.
    pub(super) dep: &'a mut Vec<(u32, f32)>,
    pub(super) xf: Option<(&'a XferIndex, &'a mut [f32])>,
}

/// The departures station or change node `v` accepts towards a destination, (boarding edge, the
/// share of its riders that take it, T-4041) in edge order, into `dep`; returns the station. The
/// other edges' shares are 0 and add nothing, so these alone give the sums over every edge, to the
/// bit (T-581).
#[inline(always)]
pub(crate) fn departures(g: &Graph, bits: &[u64], cn: &ChangeNodes, sh: &Shares, v: usize, dep: &mut Vec<(u32, f32)>) -> usize {
    let s = g.n_st;
    if v < s {
        station_departures(g, bits, sh, v, dep);
        v
    } else {
        dep.clear();
        let k = v - g.n;
        let st = cn.st[k] as usize;
        let e0 = g.adj_off[st] as usize;
        for (w, &b) in cn.bits[cn.off[k] as usize..cn.off[k + 1] as usize].iter().enumerate() {
            let mut b = b;
            while b != 0 {
                let e = e0 + 64 * w + b.trailing_zeros() as usize;
                b &= b - 1;
                dep.push((e as u32, 1.0));
            }
        }
        put_shares(g, sh, v, dep);
        st
    }
}

/// `departures` at station `v`.
#[inline(always)]
pub(crate) fn station_departures(g: &Graph, bits: &[u64], sh: &Shares, v: usize, dep: &mut Vec<(u32, f32)>) {
    let s = g.n_st;
    dep.clear();
    for e in g.adj_off[v] as usize..g.adj_off[v + 1] as usize {
        if accepted(bits, g, g.adj_to[e] as usize - s) != 0.0 {
            dep.push((e as u32, 1.0));
        }
    }
    put_shares(g, sh, v, dep);
}

/// Node `v`'s departures in `dep` given their shares, where it accepts more than one; a share of 0
/// (a line never first) is left out.
#[inline(always)]
fn put_shares(g: &Graph, sh: &Shares, v: usize, dep: &mut Vec<(u32, f32)>) {
    if dep.len() < 2 {
        return;
    }
    match sh.find(v) {
        Some(i) => {
            for (j, x) in dep.iter_mut().enumerate() {
                x.1 = sh.val[i + 1 + j];
            }
            dep.retain(|x| x.1 > 0.0);
        }
        None => {
            // a strategy made with no shares: by trains an hour
            debug_assert!(sh.at.is_empty(), "T-4041: node {v} accepts {} departures and has no shares", dep.len());
            for x in dep.iter_mut() {
                x.1 = g.rn_tph[g.adj_to[x.0 as usize] as usize - g.n_st];
            }
        }
    }
}

impl DestWalk<'_> {
    #[inline(always)]
    pub(super) fn run<I: Id>(&mut self, next: &[I], order: &[I]) {
        let DestWalk { g, s, n, dst, bits, cn, sh, ref mut flow, ref mut l, ref mut head, ref mut rec, ref mut use_k, ref mut dep, ref mut xf } = *self;
        let end = n + cn.len();
        for &v in order.iter().skip(1).rev() {
            let v = v.ix();
            let f = flow[v];
            if f == 0.0 { continue; }
            if v < s || v >= n {
                // a station, or a change node boarding at its station (head slot s + k)
                let st = departures(g, bits, cn, sh, v, dep);
                let hx = if v < s { v } else { s + v - n };
                let sum: f32 = dep.iter().map(|x| x.1).sum();
                if sum == 0.0 { continue; }
                l.board[st] += f;
                for &(e, a) in dep.iter() {
                    let to = g.adj_to[e as usize] as usize;
                    let share = f * a / sum;
                    l.rn_board[to - s] += share;
                    let nx = next[to].ix();
                    if nx >= s && nx < n {
                        flow[to] += share;
                    } else {
                        // riders on board leave here, but a boarding rides a stop first (T-1055)
                        let r = to - s;
                        flow[g.rn_next[r] as usize] += share;
                        l.seg[r] += share;
                        l.pax_min += (share * g.rn_ride[r]) as f64;
                    }
                }
                if head.get(hx).is_some_and(|&h| h != NONE) {
                    // the riders who arrived here leave by the departures as everyone here does
                    // (only the useful ones: about half go the wrong way for this destination)
                    use_k.clear();
                    for &(e, b) in dep.iter() {
                        if b > 0.0 {
                            use_k.push((e - g.adj_off[st], b / sum));
                        }
                    }
                    let x = &mut xf.as_mut().unwrap().1;
                    let mut i = head[hx];
                    while i != NONE {
                        let (at, fin, nx) = rec[i as usize];
                        for &(k, share) in use_k.iter() {
                            x[(at + k) as usize] += fin * share;
                        }
                        i = nx;
                    }
                }
            } else {
                let to = next[v].ix();
                if to >= end { continue; }
                flow[to] += f;
                if to >= s && to < n {
                    l.seg[v - s] += f;
                    l.pax_min += (f * g.rn_ride[v - s]) as f64;
                } else {
                    // Walking transfers alight at the route node's station, not the walk's end.
                    l.alight[g.rn_station[v - s] as usize] += f;
                    l.rn_alight[v - s] += f;
                    let (st, hx) = if to < s { (to, to) } else { (cn.st[to - n] as usize, s + to - n) };
                    if st != dst {
                        if let Some((ix, _)) = xf.as_ref() {
                            let e = (g.adj_off[v] as usize..g.adj_off[v + 1] as usize).find(|&e| g.adj_to[e] as usize == st).unwrap();
                            rec.push((ix.at[e], f, head[hx]));
                            head[hx] = (rec.len() - 1) as u32;
                        }
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// 7. Crowding

pub struct CrowdStats {
    pub segs_over_seats: usize,
    pub segs: usize,
    pub max_load_of_crush: f32,
    /// Rider-weighted percentiles of the penalty factor minus one.
    pub p50: f32,
    pub p90: f32,
}

/// Penalty per segment: 1 + alpha x(1+x)/2, x = max(load - start, 0) / (rated - start), per hour
/// at the peak hour, start the larger of the seats and `crowd_start` x the rated load (a bus's start
/// is `network::BUS_CROWD_START`, over the span a train's has, T-2922) (T-710:
/// riders mind crowding only near full). The same curve runs on past the rated load to crush,
/// 1.5 times rated (T-2186): 3 at rated, about 10.8 at crush at the default start. Never saturates
/// (memory note on crowding).
pub fn crowd_factors(g: &Graph, net: &Network, loads: &Loads, p: &Params) -> (Vec<f32>, CrowdStats) {
    let nr = g.rn_line.len();
    let mut fac = vec![1.0f32; nr];
    let mut st = CrowdStats { segs_over_seats: 0, segs: 0, max_load_of_crush: 0.0, p50: 0.0, p90: 0.0 };
    let mut wv: Vec<(f32, f32)> = vec![];
    for r in 0..nr {
        if g.rn_ride[r] == 0.0 {
            continue;
        }
        st.segs += 1;
        let l = g.rn_line[r] as usize;
        let tph = 60.0 / net.line_headway_min[l][p.period];
        let seats = tph * net.line_seats[l];
        let crush = tph * net.line_crush[l];
        let rated = tph * net.rated(l);
        // a bus is minded from `BUS_CROWD_START` of its rated load, over the same span as a train's
        let (start, span) = if net.bus_line(l) {
            (seats.max(crate::demand::network::BUS_CROWD_START * rated), (1.0 - p.crowd_start) * rated)
        } else {
            let start = seats.max(p.crowd_start * rated);
            (start, rated - start)
        };
        let hour = loads.seg[r] / p.period_hours * p.peak_hour_factor;
        let x = ((hour - start) / span).max(0.0);
        fac[r] = 1.0 + p.crowd_alpha * x * (1.0 + x) / 2.0;
        if hour > seats {
            st.segs_over_seats += 1;
        }
        st.max_load_of_crush = st.max_load_of_crush.max(hour / crush);
        if loads.seg[r] > 0.0 {
            wv.push((fac[r] - 1.0, loads.seg[r] * g.rn_ride[r]));
        }
    }
    wv.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let tot: f32 = wv.iter().map(|e| e.1).sum();
    let (mut acc, mut p50, mut p90) = (0.0f32, None, None);
    for &(v, w) in &wv {
        acc += w;
        if p50.is_none() && acc >= 0.5 * tot {
            p50 = Some(v);
        }
        if p90.is_none() && acc >= 0.9 * tot {
            p90 = Some(v);
        }
    }
    st.p50 = p50.unwrap_or(0.0);
    st.p90 = p90.unwrap_or(0.0);
    (fac, st)
}

/// Method of successive averages: `avg += (new - avg) / (round + 1)`.
pub fn msa(avg: &mut [f32], new: &[f32], round: usize) {
    msa_w(avg, new, 1.0 / (round as f32 + 1.0));
}

/// `msa` at weight `w` for the new pass.
pub fn msa_w(avg: &mut [f32], new: &[f32], w: f32) {
    for (a, &n) in avg.iter_mut().zip(new) {
        *a += (n - *a) * w;
    }
}
