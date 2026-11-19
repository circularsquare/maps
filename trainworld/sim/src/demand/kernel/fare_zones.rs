use super::*;

// ---------------------------------------------------------------------------------------------
// T-3758: fare zones along a destination's strategy
//
// A ride is a zone ride when its line pays zone fares (`FareZones::dist` off) and the stations it
// boards and alights at are both in one fare area; every other ride is a distance ride. Zone rides
// one after another in one area (changes and walks between them allowed) pay one price, for the
// zones from the lowest to the highest of that area's stations they stop at or pass; the distance
// rides together pay the curve on their km. The base is paid once, so a trip pays
//
//     base + (curve(distance km) - base) + sum over its zone runs of (price - base)
//
// each part times the km-weighted multiplier of its own rides. `ZoneWork::run` works out each
// station's expected fare towards one destination, reading the strategy from the destination
// outward as `trip_km` does. A route node follows one edge, so the rest of a ride from it is fixed
// (`Ride`); riders split only at stations and change nodes, where the departures are averaged by
// frequency: exact on a single route; where riders split onto rides that open zone runs in one
// area, the runs' lowest and highest zones are averaged before pricing (a fraction of a zone takes
// the line between two prices), and where they open runs in different areas each run ends there.

/// The rest of a ride from a route node to the station it alights at.
#[derive(Clone, Copy, Default)]
struct Ride {
    /// the area (index + 1) of the station it alights at; 0 for none
    a: u32,
    /// the lowest and highest zone of that area it stops at or passes from here on
    lo: f32,
    hi: f32,
    km: f32,
    kmw: f32,
    /// the node after it alights (a station or change node), NONE where the strategy does not go on
    cont: u32,
}

/// A trip's fare from a station or change node on: its first zone run so far (open: the ride
/// before it may join it), and what follows.
#[derive(Clone, Copy, Default)]
struct ZState {
    /// the chance the trip from here starts with a zone ride
    p: f32,
    /// that run's area, its lowest and highest zone, km and weighted km (given it is open)
    a: u32,
    lo: f32,
    hi: f32,
    ok: f32,
    okw: f32,
    /// the runs after it: their fares over the base and rate, weighted km
    rest: f32,
    zk: f32,
    /// the distance rides: km and weighted km
    d: f32,
    dw: f32,
    /// T-3784: the expected changes between two zone rides of one run, which pay no transfer fare
    fr: f32,
}

/// One destination's fares under fare zones (`run`), and scratch for spreading them over the lines
/// (`spread`).
#[derive(Default)]
pub struct ZoneWork {
    ride: Vec<Ride>,
    zs: Vec<ZState>,
    /// Per station: the zone runs' fares over the base and rate the route search charges, and their
    /// weighted km; the distance rides' curve extra and their weighted km. Perceived minutes, each
    /// times its own rides' km-weighted multiplier.
    pub zx: Vec<f32>,
    pub zk: Vec<f32>,
    pub dx: Vec<f32>,
    pub dk: Vec<f32>,
    /// T-3784: per station, the trip's expected changes that pay no transfer fare
    pub fr: Vec<f32>,
    q: Vec<[f32; 2]>,
    qd: Vec<[f32; 2]>,
    /// T-3784, `spread` with a transfer fare: per node its riders; of those, on a route node the
    /// ones on a zone ride, at a station or change node the ones who left a zone ride the run after
    /// it continues
    nr: Vec<f32>,
    nz: Vec<f32>,
    br: Vec<(f32, ZState)>,
}

impl ZoneWork {
    /// A trip's fare over the base and rate from station `src` (after `run`).
    #[inline]
    pub fn extra(&self, src: usize) -> f32 {
        self.zx[src] + self.dx[src]
    }

    /// `extra` less the transfer fare `xfer` (perceived minutes) the search charged on the changes
    /// inside zone runs, which pay none (T-3784).
    #[inline]
    pub fn extra_less_free(&self, src: usize, xfer: f32) -> f32 {
        if xfer > 0.0 { self.extra(src) - xfer * self.fr[src] } else { self.extra(src) }
    }

    /// Each station's fare towards `dst` under strategy `st` (`zx`, `dx`), on `hop_km` a route node
    /// (`Graph::rn_km`'s layout) and `hop_kmw` those times their multipliers (empty: all 1).
    pub fn run(&mut self, g: &Graph, st: Strategy, dst: usize, hop_km: &[f32], hop_kmw: &[f32], dep: &mut Vec<(u32, f32)>) {
        match (st.next, st.order) {
            (NodeIds::Short(next), NodeIds::Short(order)) => self.run_in(g, st, dst, next, order, hop_km, hop_kmw, dep),
            (NodeIds::Long(next), NodeIds::Long(order)) => self.run_in(g, st, dst, next, order, hop_km, hop_kmw, dep),
            _ => unreachable!("a destination's ids in two widths"),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn run_in<I: Id>(&mut self, g: &Graph, st: Strategy, dst: usize, next: &[I], order: &[I], hop_km: &[f32], hop_kmw: &[f32], dep: &mut Vec<(u32, f32)>) {
        let (s, n) = (g.n_st, g.n);
        let end = n + st.change.len();
        let z = &g.fare_curve.zones;
        self.ride.clear();
        self.ride.resize(n - s, Ride { cont: NONE, ..Default::default() });
        self.zs.clear();
        self.zs.resize(end, ZState::default());
        let on_train = |v: usize| (s..n).contains(&next[v].ix());
        let hop = |r: usize| hop_km.get(r).copied().unwrap_or(0.0);
        let hopw = |r: usize| if hop_kmw.is_empty() { hop(r) } else { hop_kmw.get(r).copied().unwrap_or(0.0) };
        for &id in order {
            let id = id.ix();
            if id == dst {
                continue;
            }
            if id < s || id >= n {
                departures(g, st.board, st.change, id, dep);
                let (ride, zs) = (&self.ride, &self.zs);
                self.br.clear();
                let mut f = 0f32;
                for &(e, a) in dep.iter() {
                    let r = g.adj_to[e as usize] as usize - s;
                    // a boarding rides a stop first (T-1055)
                    let rd = if on_train(s + r) { ride[r] } else { hop_ride(z, g, r, ride[g.rn_next[r] as usize - s], hop(r), hopw(r)) };
                    let c = if rd.cont == NONE { ZState::default() } else { zs[rd.cont as usize] };
                    self.br.push((a, board(z, g, r, rd, c)));
                    f += a;
                }
                if f > 0.0 {
                    self.zs[id] = mix(z, g, &mut self.br, f);
                }
            } else {
                let nx = next[id].ix();
                if nx >= end {
                    continue;
                }
                let r = id - s;
                self.ride[r] = if on_train(id) {
                    hop_ride(z, g, r, self.ride[nx - s], hop(r), hopw(r))
                } else {
                    // it alights at its own station, whatever it walks to then
                    let c = z.code(g.rn_station[r] as usize);
                    let zn = (c & 255) as f32;
                    Ride { a: c >> 8, lo: zn, hi: zn, km: 0.0, kmw: 0.0, cont: nx as u32 }
                };
            }
        }
        self.zx.clear();
        self.zk.clear();
        self.dx.clear();
        self.dk.clear();
        self.fr.clear();
        let curve = &g.fare_curve;
        for v in 0..s {
            let t = &self.zs[v];
            self.zx.push(t.rest + t.p * run_fare(z, g, t));
            self.zk.push(t.zk + t.p * t.okw);
            self.dx.push(if t.d > 0.0 { curve.extra(t.d) * t.dw / t.d } else { 0.0 });
            self.dk.push(t.dw);
            self.fr.push(t.fr);
        }
    }

    /// Each trip's fare over the base and rate (after `run`, `col(src)` riders from each station)
    /// into `out` per route node, as `board_extra` puts the curve's: the zone runs' part by the
    /// weighted km of its trips' zone rides, the distance part by those of their distance rides.
    /// With a transfer fare `xfer` (perceived minutes, 0 none), less that fare on the route node
    /// boarded at each change inside a zone run, which pays none (T-3784): the transfers counted
    /// there charge it (`api::fare_extra_usd`).
    #[allow(clippy::too_many_arguments)]
    pub fn spread(&mut self, g: &Graph, st: Strategy, dst: usize, col: impl Fn(usize) -> f32, hop_kmw: &[f32], xfer: f32, dep: &mut Vec<(u32, f32)>, out: &mut [f32]) {
        match (st.next, st.order) {
            (NodeIds::Short(next), NodeIds::Short(order)) => self.spread_in(g, st, dst, next, order, col, hop_kmw, xfer, dep, out),
            (NodeIds::Long(next), NodeIds::Long(order)) => self.spread_in(g, st, dst, next, order, col, hop_kmw, xfer, dep, out),
            _ => unreachable!("a destination's ids in two widths"),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn spread_in<I: Id>(&mut self, g: &Graph, st: Strategy, dst: usize, next: &[I], order: &[I], col: impl Fn(usize) -> f32, hop_kmw: &[f32], xfer: f32, dep: &mut Vec<(u32, f32)>, out: &mut [f32]) {
        let (s, n) = (g.n_st, g.n);
        let end = n + st.change.len();
        let z = &g.fare_curve.zones;
        // per node, a rate a weighted km for the zone runs and one for the distance rides; on a
        // route node `q` carries the riders on a zone ride, `qd` those on a distance ride
        self.q.clear();
        self.q.resize(end, [0.0; 2]);
        self.qd.clear();
        self.qd.resize(n - s, [0.0; 2]);
        let free = xfer > 0.0;
        if free {
            self.nr.clear();
            self.nr.resize(end, 0.0);
            self.nz.clear();
            self.nz.resize(end, 0.0);
        }
        let mut any = false;
        for src in 0..s {
            let f = col(src);
            if f == 0.0 || src == dst {
                continue;
            }
            let rz = if self.zk[src] > 0.0 { f * self.zx[src] / self.zk[src] } else { 0.0 };
            let rd = if self.dk[src] > 0.0 { f * self.dx[src] / self.dk[src] } else { 0.0 };
            if rz != 0.0 || rd != 0.0 {
                self.q[src] = [rz, rd];
                any = true;
            }
            if free {
                self.nr[src] = f;
                any = true;
            }
        }
        if !any {
            return;
        }
        let on_train = |v: usize| (s..n).contains(&next[v].ix());
        let hopw = |r: usize| hop_kmw.get(r).copied().unwrap_or(0.0);
        for &id in order.iter().rev() {
            let id = id.ix();
            if id == dst {
                continue;
            }
            if id < s || id >= n {
                let v = self.q[id];
                let riders = if free { self.nr[id] } else { 0.0 };
                if v == [0.0; 2] && riders == 0.0 {
                    continue;
                }
                departures(g, st.board, st.change, id, dep);
                let sum: f32 = dep.iter().map(|d| d.1).sum();
                if !(sum > 0.0) {
                    continue;
                }
                // riders off a zone ride whose run goes on here ride on free (`board`'s join)
                let joining = if free && self.zs[id].p > 0.0 { self.nz[id] } else { 0.0 };
                for &(e, a) in dep.iter() {
                    let r = g.adj_to[e as usize] as usize - s;
                    let w = [v[0] * a / sum, v[1] * a / sum];
                    let (to, rd) = if on_train(s + r) { (s + r, self.ride[r]) } else { (g.rn_next[r] as usize, self.ride[g.rn_next[r] as usize - s]) };
                    let zone = is_zone_ride(z, g, r, rd.a);
                    if zone && joining > 0.0 {
                        out[r] -= xfer * joining * a / sum;
                    }
                    if !on_train(s + r) {
                        out[r] += w[if zone { 0 } else { 1 }] * hopw(r);
                        if to >= end {
                            continue;
                        }
                    }
                    let q = if zone { &mut self.q[to] } else { &mut self.qd[to - s] };
                    q[0] += w[0];
                    q[1] += w[1];
                    if free {
                        let k = riders * a / sum;
                        self.nr[to] += k;
                        if zone {
                            self.nz[to] += k;
                        }
                    }
                }
            } else {
                let (vz, vd) = (self.q[id], self.qd[id - s]);
                let (riders, on_zone) = if free { (self.nr[id], self.nz[id]) } else { (0.0, 0.0) };
                if vz == [0.0; 2] && vd == [0.0; 2] && riders == 0.0 {
                    continue;
                }
                let nx = next[id].ix();
                if nx >= end {
                    continue;
                }
                if on_train(id) {
                    out[id - s] += (vz[0] + vd[1]) * hopw(id - s);
                    let q = &mut self.q[nx];
                    q[0] += vz[0];
                    q[1] += vz[1];
                    let q = &mut self.qd[nx - s];
                    q[0] += vd[0];
                    q[1] += vd[1];
                    if free {
                        self.nr[nx] += riders;
                        self.nz[nx] += on_zone;
                    }
                } else {
                    let q = &mut self.q[nx];
                    q[0] += vz[0] + vd[0];
                    q[1] += vz[1] + vd[1];
                    if free {
                        self.nr[nx] += riders;
                        // the zone ride's area is its alighting station's (`run`'s `Ride::a`)
                        let a = z.code(g.rn_station[id - s] as usize) >> 8;
                        if on_zone > 0.0 && a != 0 && self.zs[nx].a == a {
                            self.nz[nx] += on_zone;
                        }
                    }
                }
            }
        }
    }
}

/// Whether a ride boarded at route node `r` and alighting in area `a` (`Ride::a`) is a zone ride.
#[inline]
fn is_zone_ride(z: &FareZones, g: &Graph, r: usize, a: u32) -> bool {
    a != 0 && !z.is_dist(r) && z.code(g.rn_station[r] as usize) >> 8 == a
}

/// The ride from route node `r` on: its station and those passed to the next stop, then `rest`.
#[inline]
fn hop_ride(z: &FareZones, g: &Graph, r: usize, rest: Ride, km: f32, kmw: f32) -> Ride {
    let mut out = Ride { km: rest.km + km, kmw: rest.kmw + kmw, ..rest };
    if rest.a != 0 {
        for c in z.passed(r).iter().copied().chain(std::iter::once(z.code(g.rn_station[r] as usize))) {
            if c >> 8 == rest.a {
                let zn = (c & 255) as f32;
                out.lo = out.lo.min(zn);
                out.hi = out.hi.max(zn);
            }
        }
    }
    out
}

/// A zone run's fare over the base and rate, times its rides' km-weighted multiplier.
#[inline]
fn run_fare(z: &FareZones, g: &Graph, t: &ZState) -> f32 {
    if t.p == 0.0 {
        return 0.0;
    }
    let c = &g.fare_curve;
    let Some(price) = z.price_of(t.a, t.lo, t.hi) else { return 0.0 };
    (price - c.base - c.rate * t.ok) * if t.ok > 0.0 { t.okw / t.ok } else { 1.0 }
}

/// `t` with its open run ended (paid as a run of its own).
#[inline]
fn close(z: &FareZones, g: &Graph, t: ZState) -> ZState {
    if t.p == 0.0 {
        return t;
    }
    ZState { p: 0.0, rest: t.rest + t.p * run_fare(z, g, &t), zk: t.zk + t.p * t.okw, ..t }
}

/// Boarding the ride `rd` at route node `r`, the trip going on from where it alights as `c`.
#[inline]
fn board(z: &FareZones, g: &Graph, r: usize, rd: Ride, c: ZState) -> ZState {
    if !is_zone_ride(z, g, r, rd.a) {
        let c = close(z, g, c);
        return ZState { d: c.d + rd.km, dw: c.dw + rd.kmw, ..c };
    }
    let zb = (z.code(g.rn_station[r] as usize) & 255) as f32;
    let (lo, hi) = (rd.lo.min(zb), rd.hi.max(zb));
    if c.p > 0.0 && c.a == rd.a {
        // the run after it continues this one, as often as it is taken
        let p = c.p;
        return ZState {
            p: 1.0,
            a: rd.a,
            lo: p * lo.min(c.lo) + (1.0 - p) * lo,
            hi: p * hi.max(c.hi) + (1.0 - p) * hi,
            ok: rd.km + p * c.ok,
            okw: rd.kmw + p * c.okw,
            fr: c.fr + p,
            ..c
        };
    }
    let c = close(z, g, c);
    ZState { p: 1.0, a: rd.a, lo, hi, ok: rd.km, okw: rd.kmw, ..c }
}

/// The departures `br` (frequency, state) averaged, `f` their total frequency. Runs opening in one
/// area stay open, averaged; runs opening in different areas each end here.
fn mix(z: &FareZones, g: &Graph, br: &mut [(f32, ZState)], f: f32) -> ZState {
    let a = br.iter().find(|b| b.1.p > 0.0).map_or(0, |b| b.1.a);
    if br.iter().any(|b| b.1.p > 0.0 && b.1.a != a) {
        for b in br.iter_mut() {
            b.1 = close(z, g, b.1);
        }
    }
    let mut out = ZState { a, ..Default::default() };
    for &(w, t) in br.iter() {
        let w = w / f;
        out.rest += w * t.rest;
        out.zk += w * t.zk;
        out.d += w * t.d;
        out.dw += w * t.dw;
        out.fr += w * t.fr;
        if t.p > 0.0 {
            let wp = w * t.p;
            out.p += wp;
            out.lo += wp * t.lo;
            out.hi += wp * t.hi;
            out.ok += wp * t.ok;
            out.okw += wp * t.okw;
        }
    }
    if out.p > 0.0 {
        out.lo /= out.p;
        out.hi /= out.p;
        out.ok /= out.p;
        out.okw /= out.p;
    }
    out
}

/// T-3758: the fare over the base and rate of one route of rides (`(route node boarded, route node
/// alighted, km, weighted km)` each, in order), as `ZoneWork` prices a single route: per ride its
/// share as `ZoneWork::spread` puts it, the zone runs' part by weighted km over the zone rides, the
/// distance part over the distance rides. Also per ride whether the change onto it is inside a zone
/// run, so pays no transfer fare (T-3784).
pub fn route_fares(g: &Graph, rides: &[(usize, usize, f32, f32)]) -> (Vec<f32>, Vec<bool>) {
    let z = &g.fare_curve.zones;
    let c = &g.fare_curve;
    let st = |r: usize| g.rn_station[r] as usize;
    let mut out = vec![0f32; rides.len()];
    let mut joins = vec![false; rides.len()];
    let (mut d, mut dw) = (0f32, 0f32);
    let mut zone = vec![false; rides.len()];
    // runs: (area, lo, hi, km, weighted km, first ride, last ride)
    let mut runs: Vec<(u32, u32, u32, f32, f32, usize, usize)> = vec![];
    for (i, &(from, to, km, kmw)) in rides.iter().enumerate() {
        let (cb, ca) = (z.code(st(from)), z.code(st(to)));
        let a = ca >> 8;
        if a == 0 || cb >> 8 != a || z.is_dist(from) {
            d += km;
            dw += kmw;
            continue;
        }
        zone[i] = true;
        let (mut lo, mut hi) = (u32::MAX, 0u32);
        let mut take = |c: u32| {
            if c >> 8 == a {
                lo = lo.min(c & 255);
                hi = hi.max(c & 255);
            }
        };
        let mut r = from;
        loop {
            take(z.code(st(r)));
            if r == to {
                break;
            }
            for &c in z.passed(r) {
                take(c);
            }
            let nx = g.rn_next[r];
            if nx == NONE || nx as usize - g.n_st == from {
                break;
            }
            r = nx as usize - g.n_st;
        }
        match runs.last_mut() {
            Some(x) if x.0 == a && x.6 + 1 == i => {
                x.1 = x.1.min(lo);
                x.2 = x.2.max(hi);
                x.3 += km;
                x.4 += kmw;
                x.6 = i;
                joins[i] = true;
            }
            _ => runs.push((a, lo, hi, km, kmw, i, i)),
        }
    }
    let (mut zx, mut zw) = (0f32, 0f32);
    for &(a, lo, hi, km, kmw, _, _) in &runs {
        let Some(price) = z.price_of(a, lo as f32, hi as f32) else { continue };
        zx += (price - c.base - c.rate * km) * if km > 0.0 { kmw / km } else { 1.0 };
        zw += kmw;
    }
    let dx = if d > 0.0 { c.extra(d) * dw / d } else { 0.0 };
    for (i, r) in rides.iter().enumerate() {
        let (x, w) = if zone[i] { (zx, zw) } else { (dx, dw) };
        if w > 0.0 {
            out[i] += x * r.3 / w;
        }
    }
    (out, joins)
}
