//! T-4094: a line's consecutive bus stops within `MERGE_M` of each other are one demand station, the line visiting
//! it once, for the search, mode choice and crowding only; every figure read out is split back over the real
//! stops (notes/T-4094.md). The network the player builds, draws and runs is untouched.
//!
//! The fold maps the snapshot's stops ("real") to the demand network's: a demand stop is a run of consecutive
//! real stops of one line, demand stop `j` covering real stops `first[j]..first[j + 1]`. Route nodes are laid out
//! as `build_graph` does on both sides: line after line, its stops in order, then reversed.

use super::api::TIMES_PER_STOP;
use super::network::PERIODS;
use std::collections::HashMap;

/// Stops of one line this close (between their places' centres, metres) may be one demand station.
pub(crate) const MERGE_M: f32 = 500.0;
/// A stop's catchment: the cells within this of it, nearer it than the other stops it is split with.
const CATCH_M: f32 = 400.0;
/// Every stop a figure is split over takes at least this share of their mean catchment, so none reads empty.
const FLOOR: f32 = 0.25;
const NONE: u32 = u32::MAX;

pub(crate) struct Fold {
    /// Real and demand route nodes.
    nr: usize,
    nd: usize,
    /// Per real route node, its demand route node and its stop's weight in its run (1 alone).
    rd: Vec<u32>,
    w: Vec<f32>,
    /// Per demand route node, its real route nodes in the order the vehicle passes them
    /// (`rr[rr_off[d]..rr_off[d + 1]]`), the demand route node before it on its run and the one after
    /// (a circle's last and first, T-798; NONE at a line's ends).
    rr_off: Vec<u32>,
    rr: Vec<u32>,
    prev: Vec<u32>,
    next: Vec<u32>,
    /// Per real route node, its real stop; per real stop, its demand stop; per demand stop, its first real
    /// stop, then the real stop count.
    rn_stop: Vec<u32>,
    of: Vec<u32>,
    first: Vec<u32>,
    /// Per real stop, its snapshot station; per snapshot station, its position, and its share of its
    /// demand station's figures (1 unless that is a group of several places).
    stop_st: Vec<u32>,
    st_xy: Vec<f32>,
    share: Vec<f32>,
    /// Per demand station, its snapshot stations when it is a group of several places (`m_off` empty
    /// range otherwise).
    m_off: Vec<u32>,
    members: Vec<u32>,
    /// The merged groups' walks, for every period's network.
    pub(crate) walks: Option<std::sync::Arc<Walks>>,
}

/// T-4094: walks to change at a merged group, measured between its real stops and the other station's
/// (a group's centre can sit 150 m from both its stops, T-4079's whole reach): per pair of stations with
/// a group at either end, the shortest straight line between their snapshot stations, up to
/// `network::BUS_WALK_M`. Between two stations of one place each, the kernel's own rule (their positions).
pub struct Walks {
    pub(crate) merged: Vec<bool>,
    pairs: Vec<(u32, u32, f32)>,
}

impl Walks {
    /// Whether demand station `a` is a merged group.
    pub(crate) fn merged(&self, a: usize) -> bool {
        self.merged.get(a).copied().unwrap_or(false)
    }
    /// The walk between stations `a` and `b`, metres, when either is a group and they are near.
    pub(crate) fn between(&self, a: u32, b: u32) -> Option<f32> {
        let k = (a.min(b), a.max(b));
        self.pairs.binary_search_by(|p| (p.0, p.1).cmp(&k)).ok().map(|i| self.pairs[i].2)
    }
    /// Every pair, each way.
    pub(crate) fn pairs(&self) -> impl Iterator<Item = (u32, u32)> + '_ {
        self.pairs.iter().flat_map(|p| [(p.0, p.1), (p.1, p.0)])
    }
}

fn walks_of(st_xy: &[f32], st_map: &[u32], merged: &[bool]) -> Walks {
    let reach = super::network::BUS_WALK_M;
    let ns = st_map.len();
    let key = |i: usize| ((st_xy[2 * i] / reach).floor() as i32, (st_xy[2 * i + 1] / reach).floor() as i32);
    let mut cells: HashMap<(i32, i32), Vec<u32>> = HashMap::new();
    for i in 0..ns {
        cells.entry(key(i)).or_default().push(i as u32);
    }
    let mut best: HashMap<(u32, u32), f32> = HashMap::new();
    for i in 0..ns {
        let a = st_map[i];
        if !merged[a as usize] {
            continue;
        }
        let (cx, cy) = key(i);
        for dx in -1..=1 {
            for dy in -1..=1 {
                for &j in cells.get(&(cx + dx, cy + dy)).map_or(&[][..], |v| &v[..]) {
                    let b = st_map[j as usize];
                    if b == a {
                        continue;
                    }
                    let d = (st_xy[2 * i] - st_xy[2 * j as usize]).hypot(st_xy[2 * i + 1] - st_xy[2 * j as usize + 1]);
                    if d <= reach {
                        let e = best.entry((a.min(b), a.max(b))).or_insert(f32::INFINITY);
                        *e = e.min(d);
                    }
                }
            }
        }
    }
    let mut pairs: Vec<(u32, u32, f32)> = best.into_iter().map(|((a, b), d)| (a, b, d)).collect();
    pairs.sort_by(|x, y| (x.0, x.1).cmp(&(y.0, y.1)));
    Walks { merged: merged.to_vec(), pairs }
}

/// The demand network `layout` gives: stations, the map to them, the lines' stops and run times, platforms.
pub(crate) struct Layout {
    pub(crate) st_x: Vec<f32>,
    pub(crate) st_y: Vec<f32>,
    pub(crate) st_map: Vec<u32>,
    pub(crate) line_off: Vec<u32>,
    pub(crate) stops: Vec<u32>,
    pub(crate) times: Vec<f32>,
    pub(crate) stop_xy: Vec<f32>,
    pub(crate) fold: Option<Fold>,
}

/// The real network the merge starts from: the snapshot's lines and the 100 m places
/// (`merge_stations`: `st_map` and the centres), and which places may merge (`places_ok`).
pub(crate) struct Real<'a> {
    pub st_xy: &'a [f32],
    pub off: &'a [u32],
    pub stops: &'a [u32],
    pub times: &'a [f32],
    pub st_map: &'a [u32],
    pub px: &'a [f32],
    pub py: &'a [f32],
    pub place_ok: &'a [bool],
}

/// Which places may merge: bus only (`bus`, from `demand_catches`), no tram stop or BRT station, and every
/// stop there its own reverse stop (T-3348) with no pass, board-only or alight-only mark in any period (T-4013).
#[allow(clippy::too_many_arguments)]
pub(crate) fn places_ok(st_map: &[u32], n: usize, off: &[u32], stops: &[u32], bus: &[bool], tram: &[bool], ups: &[u8], back: &[u32], span: &[f32]) -> Vec<bool> {
    if bus.len() != n {
        return vec![false; n];
    }
    let brt = |p: usize| ups.get(p).is_some_and(|&u| u & crate::track::upgrades::BRT_STATION != 0);
    let mut ok: Vec<bool> = (0..n).map(|p| bus[p] && !tram.get(p).copied().unwrap_or(false) && !brt(p)).collect();
    let ns = stops.len();
    let nr = 2 * ns;
    let marked = |r: usize| span.len() == PERIODS * nr && (0..PERIODS).any(|q| !(0.0..1.5).contains(&span[q * nr + r]));
    for l in 0..off.len() - 1 {
        let (a, b) = (off[l] as usize, off[l + 1] as usize);
        let len = b - a;
        for i in 0..len {
            let k = a + i;
            let p = st_map[stops[k] as usize] as usize;
            let other_back = back.len() == ns && st_map[back[k] as usize] != st_map[stops[k] as usize];
            // the stop's route nodes: run 0's i-th, run 1's (len - 1 - i)-th
            if other_back || marked(2 * a + i) || marked(2 * a + len + (len - 1 - i)) {
                ok[p] = false;
            }
        }
    }
    ok
}

fn same(r: &Real) -> Layout {
    Layout {
        st_x: r.px.to_vec(),
        st_y: r.py.to_vec(),
        st_map: r.st_map.to_vec(),
        line_off: r.off.to_vec(),
        stops: r.stops.iter().map(|&s| r.st_map[s as usize]).collect(),
        times: r.times.to_vec(),
        stop_xy: r.stops.iter().flat_map(|&s| [r.st_xy[2 * s as usize], r.st_xy[2 * s as usize + 1]]).collect(),
        fold: None,
    }
}

/// Runs of places along each line, a run's places all within `merge_m` of each other and served by the same
/// lines (lines in order; a run starts only at a place no run has taken, and never holds both ends of its
/// line). Same lines: every line through a group stops at each of its places, so parallel lines fold alike and
/// a change there is at a stop the lines share. Per place its group.
fn groups(r: &Real, merge_m: f32) -> (Vec<u32>, Vec<u32>) {
    let np = r.px.len();
    let mut lines: Vec<Vec<u32>> = vec![vec![]; np];
    for l in 0..r.off.len() - 1 {
        for k in r.off[l] as usize..r.off[l + 1] as usize {
            lines[r.st_map[r.stops[k] as usize] as usize].push(l as u32);
        }
    }
    for v in lines.iter_mut() {
        v.sort_unstable();
        v.dedup();
    }
    let mut group = vec![NONE; np];
    let mut size: Vec<u32> = vec![];
    let mut mem: Vec<Vec<usize>> = vec![];
    let d2 = |a: usize, b: usize| (r.px[a] - r.px[b]).powi(2) + (r.py[a] - r.py[b]).powi(2);
    let m2 = merge_m * merge_m;
    for l in 0..r.off.len() - 1 {
        let (a, b) = (r.off[l] as usize, r.off[l + 1] as usize);
        let len = b - a;
        let (mut cur, mut start) = (NONE, 0usize);
        for i in 0..len {
            let p = r.st_map[r.stops[a + i] as usize] as usize;
            if !r.place_ok[p] || group[p] != NONE {
                cur = NONE;
                continue;
            }
            let fits = cur != NONE && !(start == 0 && i + 1 == len) && lines[p] == lines[mem[cur as usize][0]] && mem[cur as usize].iter().all(|&m| d2(m, p) <= m2);
            if !fits {
                cur = mem.len() as u32;
                mem.push(vec![]);
                size.push(0);
                start = i;
            }
            group[p] = cur;
            mem[cur as usize].push(p);
            size[cur as usize] += 1;
        }
    }
    (group, size)
}

/// The demand network: `r`'s places, with each line's runs of bus places (`groups`) one station. A line
/// that would be left with one stop keeps its places apart.
pub(crate) fn layout(r: &Real, merge_m: f32) -> Layout {
    let ns = r.st_map.len();
    let nl = r.off.len().saturating_sub(1);
    if merge_m <= 0.0 || nl == 0 || !r.place_ok.iter().any(|&v| v) {
        return same(r);
    }
    let (mut group, size) = groups(r, merge_m);
    let mut alone: Vec<bool> = size.iter().map(|&s| s < 2).collect();
    // demand station of each place: its group's key, or its own
    let np = r.px.len();
    let key = |group: &[u32], alone: &[bool], p: usize| if group[p] != NONE && !alone[group[p] as usize] { np as u32 + group[p] } else { p as u32 };
    loop {
        let mut bad = false;
        for l in 0..nl {
            let (a, b) = (r.off[l] as usize, r.off[l + 1] as usize);
            if b - a < 2 {
                continue;
            }
            let ks: Vec<u32> = (a..b).map(|k| key(&group, &alone, r.st_map[r.stops[k] as usize] as usize)).collect();
            let m = 1 + ks.windows(2).filter(|w| w[0] != w[1]).count();
            if m < 2 {
                for k in a..b {
                    let p = r.st_map[r.stops[k] as usize] as usize;
                    if group[p] != NONE {
                        alone[group[p] as usize] = true;
                    }
                }
                bad = true;
            }
        }
        if !bad {
            break;
        }
    }
    if alone.iter().all(|&a| a) {
        return same(r);
    }
    for g in group.iter_mut() {
        if *g != NONE && alone[*g as usize] {
            *g = NONE;
        }
    }
    // stations numbered by their first snapshot station, at the mean of their snapshot stations, as
    // `merge_stations` numbers places
    let mut id: HashMap<u32, u32> = HashMap::new();
    let mut st_map = vec![0u32; ns];
    let (mut sx, mut sy, mut cnt): (Vec<f32>, Vec<f32>, Vec<f32>) = (vec![], vec![], vec![]);
    for i in 0..ns {
        let k = key(&group, &alone, r.st_map[i] as usize);
        let d = *id.entry(k).or_insert_with(|| {
            sx.push(0.0);
            sy.push(0.0);
            cnt.push(0.0);
            (sx.len() - 1) as u32
        }) as usize;
        st_map[i] = d as u32;
        sx[d] += r.st_xy[2 * i];
        sy[d] += r.st_xy[2 * i + 1];
        cnt[d] += 1.0;
    }
    for d in 0..sx.len() {
        sx[d] /= cnt[d];
        sy[d] /= cnt[d];
    }
    let nd_st = sx.len();
    let mut merged = vec![false; nd_st];
    for i in 0..ns {
        if group[r.st_map[i] as usize] != NONE {
            merged[st_map[i] as usize] = true;
        }
    }
    // the lines' demand stops: consecutive real stops at one station collapse
    let nstops = r.stops.len();
    let (mut of, mut first, mut dstops, mut d_off) = (vec![0u32; nstops], vec![], vec![], vec![0u32]);
    for l in 0..nl {
        let (a, b) = (r.off[l] as usize, r.off[l + 1] as usize);
        for k in a..b {
            let d = st_map[r.stops[k] as usize];
            // only a group's runs fold: two stops of one place in a row stay two, as the 100 m merge leaves them
            if k == a || dstops.last() != Some(&d) || !merged[d as usize] {
                first.push(k as u32);
                dstops.push(d);
            }
            of[k] = (dstops.len() - 1) as u32;
        }
        d_off.push(dstops.len() as u32);
    }
    first.push(nstops as u32);
    let ndst = dstops.len();
    let (nr, nd) = (2 * nstops, 2 * ndst);
    // route nodes both sides: per demand route node its real ones in passing order, before and after
    let mut rd = vec![0u32; nr];
    let mut rn_stop = vec![0u32; nr];
    let (mut rr_off, mut rr, mut prev, mut next) = (vec![0u32], Vec::with_capacity(nr), vec![NONE; nd], vec![NONE; nd]);
    for l in 0..nl {
        let (a, n) = (r.off[l] as usize, (r.off[l + 1] - r.off[l]) as usize);
        let (da, m) = (d_off[l] as usize, (d_off[l + 1] - d_off[l]) as usize);
        // a circle (T-798) only where the real line rides on round from its last stop in some period, as
        // `period_net` reads it; a line that just ends where it began is none
        let real_wrap = n >= 3 && r.times.len() == TIMES_PER_STOP * nstops && (0..PERIODS).any(|q| r.times[TIMES_PER_STOP * a + q * n + n - 1] > 0.0);
        let wrap = m >= 3 && dstops[da] == dstops[da + m - 1] && real_wrap;
        for dir in 0..2 {
            for i in 0..m {
                let dn = 2 * da + dir * m + i;
                let j = if dir == 0 { da + i } else { da + m - 1 - i };
                let (k0, k1) = (first[j] as usize, first[j + 1] as usize);
                let ks: Vec<usize> = if dir == 0 { (k0..k1).collect() } else { (k0..k1).rev().collect() };
                for k in ks {
                    let ri = if dir == 0 { k - a } else { n - 1 - (k - a) };
                    let rn = 2 * a + dir * n + ri;
                    rd[rn] = dn as u32;
                    rn_stop[rn] = k as u32;
                    rr.push(rn as u32);
                }
                rr_off.push(rr.len() as u32);
                let base = 2 * da + dir * m;
                if i > 0 {
                    prev[dn] = (dn - 1) as u32;
                } else if wrap {
                    prev[dn] = (base + m - 1) as u32;
                }
                if i + 1 < m {
                    next[dn] = (dn + 1) as u32;
                } else if wrap {
                    next[dn] = base as u32;
                }
            }
        }
    }
    let stop_st: Vec<u32> = r.stops.to_vec();
    let mut fold = Fold {
        nr,
        nd,
        rd,
        w: vec![1.0; nr],
        rn_stop,
        rr_off,
        rr,
        prev,
        next,
        of,
        first,
        stop_st,
        st_xy: r.st_xy.to_vec(),
        share: vec![1.0; ns],
        m_off: vec![],
        members: vec![],
        walks: Some(std::sync::Arc::new(walks_of(r.st_xy, &st_map, &merged))),
    };
    // members of the groups, numbered by station
    let mut m_off = vec![0u32; nd_st + 1];
    for i in 0..ns {
        if merged[st_map[i] as usize] {
            m_off[st_map[i] as usize + 1] += 1;
        }
    }
    for d in 0..nd_st {
        m_off[d + 1] += m_off[d];
    }
    let mut at = m_off.clone();
    let mut members = vec![0u32; m_off[nd_st] as usize];
    for i in 0..ns {
        let d = st_map[i] as usize;
        if merged[d] {
            members[at[d] as usize] = i as u32;
            at[d] += 1;
        }
    }
    fold.m_off = m_off;
    fold.members = members;
    fold.weigh(&[], &[], &[]);
    // the run times: `Service::times` per line, run and period, folded as the route nodes' hops
    let mut times = vec![0f32; TIMES_PER_STOP * ndst];
    if r.times.len() == TIMES_PER_STOP * nstops {
        let mut hop = vec![0f32; nr];
        for q in 0..PERIODS {
            for l in 0..nl {
                let (a, n) = (r.off[l] as usize, (r.off[l + 1] - r.off[l]) as usize);
                let o = TIMES_PER_STOP * a;
                for i in 0..n {
                    hop[2 * a + i] = r.times[o + q * n + i];
                    hop[2 * a + n + i] = r.times[o + (PERIODS + q) * n + (n - 1 - i)];
                }
            }
            let h = fold.in_hop(&hop);
            for l in 0..nl {
                let (a, n) = (r.off[l] as usize, (r.off[l + 1] - r.off[l]) as usize);
                let (da, m) = (d_off[l] as usize, (d_off[l + 1] - d_off[l]) as usize);
                let o = TIMES_PER_STOP * da;
                for i in 0..m {
                    times[o + q * m + i] = h[2 * da + i];
                    times[o + (PERIODS + q) * m + (m - 1 - i)] = h[2 * da + m + i];
                }
                // a circle's ride on round only in the periods the real one has it
                if hop[2 * a + n - 1] == 0.0 {
                    times[o + q * m + m - 1] = 0.0;
                }
                if hop[2 * a + 2 * n - 1] == 0.0 {
                    times[o + PERIODS * m + q * m] = 0.0;
                }
            }
        }
    }
    // a demand stop's platform (T-717): its real stop's, or at a group its centre, so a change there walks
    // nothing: the lines through a group all stop at its founding line's places, mostly at one of them together
    let mut stop_xy = vec![0f32; 2 * ndst];
    for j in 0..ndst {
        let d = dstops[j] as usize;
        if merged[d] {
            (stop_xy[2 * j], stop_xy[2 * j + 1]) = (sx[d], sy[d]);
            continue;
        }
        let (k0, k1) = (fold.first[j] as usize, fold.first[j + 1] as usize);
        for k in k0..k1 {
            let s = r.stops[k] as usize;
            stop_xy[2 * j] += r.st_xy[2 * s];
            stop_xy[2 * j + 1] += r.st_xy[2 * s + 1];
        }
        let c = (k1 - k0) as f32;
        stop_xy[2 * j] /= c;
        stop_xy[2 * j + 1] /= c;
    }
    Layout { st_x: sx, st_y: sy, st_map, line_off: d_off, stops: dstops, times, stop_xy, fold: Some(fold) }
}

impl Fold {
    fn run(&self, d: usize) -> &[u32] {
        &self.rr[self.rr_off[d] as usize..self.rr_off[d + 1] as usize]
    }

    /// Each run's stops weighed by their catchment: the population and jobs (`mass`) of the cells (`x`, `y`)
    /// within `CATCH_M` that are nearer that stop than the run's others, at least `FLOOR` of the run's mean;
    /// likewise each group's snapshot stations for `share`. With no cells, equal.
    pub(crate) fn weigh(&mut self, x: &[f32], y: &[f32], mass: &[f32]) {
        let mut cells: HashMap<(i32, i32), Vec<u32>> = HashMap::new();
        let key = |x: f32, y: f32| ((x / CATCH_M).floor() as i32, (y / CATCH_M).floor() as i32);
        if x.len() == y.len() && x.len() == mass.len() {
            for c in 0..x.len() {
                if mass[c] > 0.0 {
                    cells.entry(key(x[c], y[c])).or_default().push(c as u32);
                }
            }
        }
        let st_xy = &self.st_xy;
        // the weights of the snapshot stations `sts` (positions in `st_xy`), summing to 1
        let split = |sts: &[u32]| -> Vec<f32> {
            let n = sts.len();
            let mut m = vec![0f32; n];
            let mut seen: Vec<u32> = vec![];
            for &s in sts {
                let (sx, sy) = key(st_xy[2 * s as usize], st_xy[2 * s as usize + 1]);
                for dx in -1..=1 {
                    for dy in -1..=1 {
                        if let Some(v) = cells.get(&(sx + dx, sy + dy)) {
                            seen.extend_from_slice(v);
                        }
                    }
                }
            }
            seen.sort_unstable();
            seen.dedup();
            for &c in &seen {
                let (cx, cy) = (x[c as usize], y[c as usize]);
                let (mut best, mut bd) = (usize::MAX, CATCH_M * CATCH_M);
                for (i, &s) in sts.iter().enumerate() {
                    let d = (st_xy[2 * s as usize] - cx).powi(2) + (st_xy[2 * s as usize + 1] - cy).powi(2);
                    if d <= bd {
                        bd = d;
                        best = i;
                    }
                }
                if best != usize::MAX {
                    m[best] += mass[c as usize];
                }
            }
            let mean = m.iter().sum::<f32>() / n as f32;
            if mean <= 0.0 {
                return vec![1.0 / n as f32; n];
            }
            let v: Vec<f32> = m.iter().map(|&a| a + FLOOR * mean).collect();
            let t: f32 = v.iter().sum();
            v.iter().map(|&a| a / t).collect()
        };
        // runs: a demand stop's real stops, the same weights on both runs' route nodes
        let nst = self.first.len() - 1;
        let mut w_stop = vec![1f32; self.of.len()];
        for j in 0..nst {
            let (k0, k1) = (self.first[j] as usize, self.first[j + 1] as usize);
            if k1 - k0 < 2 {
                continue;
            }
            let sts: Vec<u32> = (k0..k1).map(|k| self.stop_st[k]).collect();
            for (i, v) in split(&sts).into_iter().enumerate() {
                w_stop[k0 + i] = v;
            }
        }
        self.w = self.rn_stop.iter().map(|&k| w_stop[k as usize]).collect();
        for d in 0..self.m_off.len().saturating_sub(1) {
            let (a, b) = (self.m_off[d] as usize, self.m_off[d + 1] as usize);
            if b > a {
                let sts: Vec<u32> = self.members[a..b].to_vec();
                for (i, v) in split(&sts).into_iter().enumerate() {
                    self.share[sts[i] as usize] = v;
                }
            }
        }
    }
}

impl Fold {
    /// The same merge and weights (T-124: a kept period must read out as a fresh one would).
    pub(crate) fn same(&self, o: &Fold) -> bool {
        self.first == o.first && self.rn_stop == o.rn_stop && self.stop_st == o.stop_st && super::api::same_bits(&self.w, &o.w) && super::api::same_bits(&self.share, &o.share)
    }
    /// Real route nodes, demand route nodes.
    pub(crate) fn sizes(&self) -> (usize, usize) {
        (self.nr, self.nd)
    }
    /// Snapshot station `s`'s share of its demand station's figures (1 for a station of one place).
    pub(crate) fn share(&self, s: usize) -> f32 {
        self.share.get(s).copied().unwrap_or(1.0)
    }
    /// The snapshot stations of group `d` (empty for a station of one place).
    pub(crate) fn members(&self, d: usize) -> &[u32] {
        match (self.m_off.get(d), self.m_off.get(d + 1)) {
            (Some(&a), Some(&b)) => &self.members[a as usize..b as usize],
            _ => &[],
        }
    }
    pub(crate) fn st_pos(&self, s: usize) -> (f32, f32) {
        (self.st_xy[2 * s], self.st_xy[2 * s + 1])
    }
    /// Snapshot station `s`'s part of a cell at (x, y) among its group's stations `m`: by the inverse
    /// square of the distance, from 50 m, so every station takes some and the nearest most.
    pub(crate) fn near_share(&self, m: &[u32], s: u32, x: f32, y: f32) -> f32 {
        let wt = |t: u32| {
            let (sx, sy) = self.st_pos(t as usize);
            1.0 / ((sx - x).hypot(sy - y) + 50.0).powi(2)
        };
        let all: f32 = m.iter().map(|&t| wt(t)).sum();
        if all > 0.0 { wt(s) / all } else { 1.0 / m.len() as f32 }
    }
    /// The real route node of demand route node `d`'s run nearest (x, y) (a trip's way on or off).
    pub(crate) fn nearest(&self, d: usize, x: f32, y: f32) -> u32 {
        let mut best = (f32::INFINITY, NONE);
        for &r in self.run(d) {
            let (sx, sy) = self.rn_pos(r as usize);
            let dd = (sx - x).powi(2) + (sy - y).powi(2);
            if dd < best.0 {
                best = (dd, r);
            }
        }
        best.1
    }
    /// A route's legs (`trip::Leg`) from demand route nodes to real ones: each ride boards at its run's stop
    /// nearest where the rider is (`from` at first) and leaves at the one nearest where they go next (the
    /// next boarding, or `to`); a walk starts where the ride left.
    pub(crate) fn map_legs(&self, legs: &mut [super::trip::Leg], from: (f32, f32), to: (f32, f32)) {
        let mut at = from;
        for i in 0..legs.len() {
            let next = legs[i + 1..].iter().find(|l| l.kind == 1 && l.rn_from != NONE).map(|l| self.rn_pos(self.nearest(l.rn_from as usize, at.0, at.1) as usize));
            let goal = next.unwrap_or(to);
            let l = &mut legs[i];
            if l.rn_from != NONE {
                l.rn_from = self.nearest(l.rn_from as usize, at.0, at.1);
            }
            if l.rn_to != NONE {
                let aim = if l.kind == 1 { goal } else { at };
                l.rn_to = self.nearest(l.rn_to as usize, aim.0, aim.1);
                at = self.rn_pos(l.rn_to as usize);
            }
        }
    }
    /// The position of real route node `r`'s stop.
    pub(crate) fn rn_pos(&self, r: usize) -> (f32, f32) {
        self.st_pos(self.stop_st[self.rn_stop[r] as usize] as usize)
    }

    // --- out: demand route nodes to real ones, block by block (any number of blocks of `nd`)

    /// Riders boarding, alighting, changing, fare extras: each run's figure split over its stops by weight.
    pub(crate) fn out_split(&self, v: &[f32]) -> Vec<f32> {
        self.blocks(v, |b, o| {
            for r in 0..self.nr {
                o[r] = b[self.rd[r] as usize] * self.w[r];
            }
        })
    }
    /// Segment loads (and the crush share, a load over the line's room): the segment leaving a run is its
    /// own; inside it the load moves from the one coming in by the weights, as the split boardings and
    /// alightings move it, so nothing jumps and no stretch reads flat.
    pub(crate) fn out_load(&self, v: &[f32]) -> Vec<f32> {
        self.blocks(v, |b, o| {
            for d in 0..self.nd {
                let run = self.run(d);
                let out = b[d];
                let inn = if self.prev[d] == NONE { 0.0 } else { b[self.prev[d] as usize] };
                let mut c = 0f32;
                for &r in run {
                    c += self.w[r as usize];
                    o[r as usize] = inn + c * (out - inn);
                }
                o[*run.last().unwrap() as usize] = out;
            }
        })
    }
    /// Segment loads, boardings and alightings one block after another, repeated (`part_loads`, the
    /// market's `*_loads`).
    pub(crate) fn out_loads3(&self, v: &[f32]) -> Vec<f32> {
        let nd = self.nd;
        if nd == 0 || v.len() % (3 * nd) != 0 {
            return v.to_vec();
        }
        let mut out = Vec::with_capacity(v.len() / nd * self.nr);
        for c in v.chunks_exact(3 * nd) {
            out.extend(self.out_load(&c[..nd]));
            out.extend(self.out_split(&c[nd..2 * nd]));
            out.extend(self.out_split(&c[2 * nd..]));
        }
        out
    }
    /// The same value on every real route node of a run (`route_nodes`).
    pub(crate) fn out_copy(&self, v: &[u32]) -> Vec<u32> {
        if v.len() != self.nd {
            return v.to_vec();
        }
        self.rd.iter().map(|&d| v[d as usize]).collect()
    }
    /// Transfer triples (route node in, route node out, riders): both ends split by the runs' weights, those
    /// with neither end in `sel` (one byte per real route node, as asked) left out.
    pub(crate) fn out_pairs_in(&self, v: &[f32], sel: &[u8]) -> Vec<f32> {
        let on = |r: u32| sel.get(r as usize).is_some_and(|&s| s != 0);
        self.out_pairs(v).chunks_exact(3).filter(|t| sel.len() != self.nr || on(t[0] as u32) || on(t[1] as u32)).flatten().copied().collect()
    }
    /// Transfer triples (route node in, route node out, riders): both ends split by the runs' weights.
    pub(crate) fn out_pairs(&self, v: &[f32]) -> Vec<f32> {
        let mut out = Vec::with_capacity(v.len());
        for t in v.chunks_exact(3) {
            let (a, b) = (t[0] as usize, t[1] as usize);
            if a >= self.nd || b >= self.nd {
                continue;
            }
            for &ra in self.run(a) {
                for &rb in self.run(b) {
                    let x = t[2] * self.w[ra as usize] * self.w[rb as usize];
                    if x > 0.0 {
                        out.extend([ra as f32, rb as f32, x]);
                    }
                }
            }
        }
        out
    }
    fn blocks(&self, v: &[f32], f: impl Fn(&[f32], &mut [f32])) -> Vec<f32> {
        let nd = self.nd;
        if nd == 0 || v.len() % nd != 0 {
            return v.to_vec();
        }
        let mut out = vec![0f32; v.len() / nd * self.nr];
        for (b, o) in v.chunks_exact(nd).zip(out.chunks_exact_mut(self.nr)) {
            f(b, o);
        }
        out
    }

    // --- in: real route nodes to demand ones

    /// Crowding totals as `crowd_with_external` and `cap_with_external` take them: the segment loads (the
    /// segment leaving each run), then any blocks of boardings or alightings (summed over the run).
    pub(crate) fn in_total(&self, v: &[f32]) -> Vec<f32> {
        let nr = self.nr;
        if nr == 0 || v.len() % nr != 0 {
            return v.to_vec();
        }
        let mut out = vec![0f32; v.len() / nr * self.nd];
        for (i, (b, o)) in v.chunks_exact(nr).zip(out.chunks_exact_mut(self.nd)).enumerate() {
            for d in 0..self.nd {
                let run = self.run(d);
                o[d] = if i == 0 { b[*run.last().unwrap() as usize] } else { run.iter().map(|&r| b[r as usize]).sum() };
            }
        }
        out
    }
    /// A value on the hop leaving each route node (run times, fare km): a run's own hops go half to the hop
    /// before it and half to the hop after (riders boarding at a run board on average at its middle); 0 at
    /// a line's last.
    pub(crate) fn in_hop(&self, v: &[f32]) -> Vec<f32> {
        self.in_blocks(v, |b, o| {
            let inner = |d: usize| -> f32 {
                let run = self.run(d);
                run[..run.len() - 1].iter().map(|&r| b[r as usize]).sum()
            };
            for d in 0..self.nd {
                let nx = self.next[d];
                o[d] = if nx == NONE { 0.0 } else { 0.5 * inner(d) + b[*self.run(d).last().unwrap() as usize] + 0.5 * inner(nx as usize) };
            }
        })
    }
    /// The largest over a run (a span share; runs hold no marked stop, `places_ok`).
    pub(crate) fn in_max(&self, v: &[f32]) -> Vec<f32> {
        self.in_blocks(v, |b, o| {
            for d in 0..self.nd {
                o[d] = self.run(d).iter().map(|&r| b[r as usize]).fold(f32::MIN, f32::max);
            }
        })
    }
    /// The run's first route node's (a line's fare multipliers, the same along it).
    pub(crate) fn in_first(&self, v: &[f32]) -> Vec<f32> {
        self.in_blocks(v, |b, o| {
            for d in 0..self.nd {
                o[d] = b[self.run(d)[0] as usize];
            }
        })
    }
    /// Any of the run's (a route node selection, a flag).
    pub(crate) fn in_any(&self, v: &[u8]) -> Vec<u8> {
        if v.len() != self.nr {
            return v.to_vec();
        }
        (0..self.nd).map(|d| self.run(d).iter().map(|&r| v[r as usize]).max().unwrap_or(0)).collect()
    }
    /// Per stop (parallel to the snapshot's stops), the run's first stop's.
    pub(crate) fn in_stops<T: Copy>(&self, v: &[T]) -> Vec<T> {
        if v.len() != self.of.len() {
            return v.to_vec();
        }
        self.first[..self.first.len() - 1].iter().map(|&k| v[k as usize]).collect()
    }
    /// Passed stations per route node (`pass_off`, `pass`; fare zones, T-3758): a run's together.
    pub(crate) fn in_pass(&self, off: &[u32], pass: &[u32]) -> (Vec<u32>, Vec<u32>) {
        if off.len() != self.nr + 1 {
            return (off.to_vec(), pass.to_vec());
        }
        let (mut o, mut p) = (vec![0u32], vec![]);
        for d in 0..self.nd {
            for &r in self.run(d) {
                p.extend_from_slice(&pass[off[r as usize] as usize..off[r as usize + 1] as usize]);
            }
            o.push(p.len() as u32);
        }
        (o, p)
    }
    fn in_blocks(&self, v: &[f32], f: impl Fn(&[f32], &mut [f32])) -> Vec<f32> {
        let nr = self.nr;
        if nr == 0 || v.len() % nr != 0 {
            return v.to_vec();
        }
        let mut out = vec![0f32; v.len() / nr * self.nd];
        for (b, o) in v.chunks_exact(nr).zip(out.chunks_exact_mut(self.nd)) {
            f(b, o);
        }
        out
    }
}

#[cfg(test)]
mod tests;
