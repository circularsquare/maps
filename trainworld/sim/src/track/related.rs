//! The lines related to a line (T-1168, notes/T-1168.md "Classifying the other lines"): where
//! each other line meets it (shared track, or a station complex both routes reach, stops and
//! stations passed alike), those meetings grouped into spans it runs beside the line, and lone
//! crossings. Pure: the routes come in as polylines with their nodes and stops
//! (`wasm/related.rs` builds them from the service's runs).

use std::collections::HashMap;

/// A line that leaves and comes back stays beside the line when its track strays at most
/// STRAY_FLOOR_M plus STRAY_NEAR of the first STRAY_NEAR_M of its own run between the two
/// meetings and STRAY_FAR of the rest, ground metres: 8.5% of a 2 km run, 5.7% of 5 km, 3.9% of
/// 9 km, 2.9% of 20 km (Anita 2026-10-24: a constant plus a share, the share lower on long runs;
/// DECISIONS 2026-10-26 has the cases it was set on).
pub const STRAY_FLOOR_M: f64 = 100.0;
pub const STRAY_NEAR: f64 = 0.035;
pub const STRAY_NEAR_M: f64 = 5000.0;
pub const STRAY_FAR: f64 = 0.02;
/// O's track is sampled this often for the stray, ground metres, besides its vertices.
const STRAY_STEP_M: f64 = 200.0;
const EPS_M: f64 = 1.0;
pub const NONE: u32 = u32::MAX;

pub fn stray_allowed(run: f64) -> f64 {
    let run = run.max(0.0);
    STRAY_FLOOR_M + STRAY_NEAR * run.min(STRAY_NEAR_M) + STRAY_FAR * (run - STRAY_NEAR_M).max(0.0)
}

/// One line's route (its first run): a polyline with ground offsets `ps` (increasing), the nodes
/// it passes with their offsets, its path entries (`edge << 1 | dir`, entry `k` from node `k` to
/// node `k + 1`), and per node the stop index there or -1.
#[derive(Clone, Debug, Default)]
pub struct Route {
    pub line: u32,
    pub pts: Vec<[f64; 2]>,
    pub ps: Vec<f64>,
    pub nodes: Vec<u32>,
    pub node_s: Vec<f64>,
    pub segs: Vec<u32>,
    pub stop_at: Vec<i32>,
}

impl Route {
    fn at(&self, s: f64) -> [f64; 2] {
        let k = self.ps.partition_point(|&a| a < s);
        if k == 0 {
            return self.pts[0];
        }
        if k >= self.ps.len() {
            return self.pts[self.pts.len() - 1];
        }
        let f = (s - self.ps[k - 1]) / (self.ps[k] - self.ps[k - 1]).max(1e-9);
        let (a, b) = (self.pts[k - 1], self.pts[k]);
        [a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1])]
    }
    /// Polyline segment indices covering offsets `s0..s1`.
    fn seg_range(&self, s0: f64, s1: f64) -> (usize, usize) {
        let n = self.pts.len();
        if n < 2 {
            return (0, 0);
        }
        let lo = self.ps.partition_point(|&a| a <= s0).saturating_sub(1).min(n - 2);
        let hi = self.ps.partition_point(|&a| a < s1).max(lo + 1).min(n - 1);
        (lo, hi)
    }
    /// Plane distance and offset of the nearest point to `p` on segment `k`.
    fn seg_near(&self, k: usize, p: [f64; 2]) -> (f64, f64) {
        let (a, b) = (self.pts[k], self.pts[k + 1]);
        let (vx, vy) = (b[0] - a[0], b[1] - a[1]);
        let ll = vx * vx + vy * vy;
        let t = if ll == 0.0 { 0.0 } else { (((p[0] - a[0]) * vx + (p[1] - a[1]) * vy) / ll).clamp(0.0, 1.0) };
        let d = (p[0] - a[0] - t * vx).hypot(p[1] - a[1] - t * vy);
        (d, self.ps[k] + t * (self.ps[k + 1] - self.ps[k]))
    }
    /// Plane distance and offset of the nearest point to `p` between offsets `s0..s1`.
    fn nearest(&self, p: [f64; 2], s0: f64, s1: f64) -> (f64, f64) {
        if self.pts.len() < 2 {
            return (self.pts.first().map_or(f64::INFINITY, |q| (p[0] - q[0]).hypot(p[1] - q[1])), 0.0);
        }
        let (lo, hi) = self.seg_range(s0, s1);
        (lo..hi).map(|k| self.seg_near(k, p)).fold((f64::INFINITY, s0), |b, x| if x.0 < b.0 { x } else { b })
    }
}

#[derive(Clone, Debug)]
struct Meet {
    /// Offsets on L and O where it starts and ends (a station: both the same).
    sl: f64,
    sl2: f64,
    so: f64,
    so2: f64,
    /// The complex, or NONE for shared track.
    cx: u32,
    /// L route node indices in this complex (a loop passes its first station twice); resolved to
    /// one while grouping.
    alts: Vec<usize>,
    /// O route node index.
    on: usize,
}

impl Meet {
    fn track(&self) -> bool {
        self.cx == NONE
    }
}

/// A gap between two consecutive meetings that O runs on its own track: ground metres.
#[derive(Clone, Copy, Debug)]
struct Gap {
    la: f64,
    lb: f64,
    stray: f64,
    run: f64,
    lstep: f64,
}

/// Every other line related to `l`, flattened per group (notes/T-1168.md, decoded by
/// `game/relatedLines.ts`):
/// `[line, kind (1 beside, 2 crossing), shared track 0/1, direction along L (1 or -1),
///   sL0, sL1, sO0, sO1, O stops before the span, O stops after it,
///   the gap that split it from the line's previous group: stray, run, L step (NaN when none;
///   stray NaN when O turned back along L rather than strayed),
///   meetings, shared-track stretches, own-track gaps, own stops (counts), then
///   per meeting [complex, sL, sO, L stop index or -1, O stop index or -1],
///   per stretch [sL from, sL to],
///   per gap [sL from, sL to, stray, O run, L step],
///   per own stop [O stop index, sL, complex]]`.
/// Offsets are ground metres along each line's first run; strays are ground metres.
/// `cx(node)` is the station's complex or NONE; `k(y)` plane metres per ground metre.
pub fn related(l: &Route, others: &[Route], cx: &dyn Fn(u32) -> u32, k: &dyn Fn(f64) -> f64) -> Vec<f64> {
    let mut lcx: HashMap<u32, Vec<usize>> = HashMap::new();
    for (i, &n) in l.nodes.iter().enumerate() {
        let c = cx(n);
        if c != NONE {
            lcx.entry(c).or_default().push(i);
        }
    }
    let mut ledge: HashMap<u32, Vec<usize>> = HashMap::new();
    for (j, &p) in l.segs.iter().enumerate() {
        ledge.entry(p >> 1).or_default().push(j);
    }
    let mut out = vec![];
    let mut chunks = Chunks::new(l);
    for o in others {
        if o.line == l.line || o.nodes.is_empty() {
            continue;
        }
        let ms = meetings(l, o, cx, &lcx, &ledge);
        if ms.is_empty() {
            continue;
        }
        // a line whose every stop is on L's route is always beside it (Anita, 2026-10-24)
        let all_shared = o.stop_at.iter().enumerate().filter(|x| *x.1 >= 0).all(|(i, _)| lcx.contains_key(&cx(o.nodes[i])));
        group_and_emit(l, o, ms, all_shared, cx, k, &mut chunks, &mut out);
    }
    out
}

fn meetings(l: &Route, o: &Route, cx: &dyn Fn(u32) -> u32, lcx: &HashMap<u32, Vec<usize>>, ledge: &HashMap<u32, Vec<usize>>) -> Vec<Meet> {
    let mut ms = vec![];
    for (i, &n) in o.nodes.iter().enumerate() {
        let c = cx(n);
        if c == NONE {
            continue;
        }
        if let Some(alts) = lcx.get(&c) {
            let s = o.node_s[i];
            ms.push(Meet { sl: l.node_s[alts[0]], sl2: l.node_s[alts[0]], so: s, so2: s, cx: c, alts: alts.clone(), on: i });
        }
    }
    for (j, &p) in o.segs.iter().enumerate() {
        let Some(ls) = ledge.get(&(p >> 1)) else { continue };
        let lj = ls[0];
        let (la, lb) = (l.node_s[lj], l.node_s[lj + 1]);
        let same = (l.segs[lj] & 1) == (p & 1);
        ms.push(Meet { sl: if same { la } else { lb }, sl2: if same { lb } else { la }, so: o.node_s[j], so2: o.node_s[j + 1], cx: NONE, alts: vec![], on: j });
    }
    ms.sort_by(|a, b| a.so.total_cmp(&b.so).then(a.track().cmp(&b.track())));
    ms
}

/// L's polyline segments in runs of CHUNK with their bounding boxes, for the nearest-point search
/// in `stray`: a run whose box is farther than the nearest segment so far is skipped.
struct Chunks {
    boxes: Vec<[f64; 4]>,
    hint: usize,
}
const CHUNK: usize = 32;

impl Chunks {
    fn new(l: &Route) -> Chunks {
        let n = l.pts.len().saturating_sub(1);
        let boxes = (0..n.div_ceil(CHUNK))
            .map(|c| {
                let pts = &l.pts[c * CHUNK..(c * CHUNK + CHUNK + 1).min(l.pts.len())];
                pts.iter().fold([f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY], |b, p| [b[0].min(p[0]), b[1].min(p[1]), b[2].max(p[0]), b[3].max(p[1])])
            })
            .collect();
        Chunks { boxes, hint: 0 }
    }
    /// Plane distance from `p` to the nearest of L's segments `lo..hi`, or to the first found
    /// within `enough`: the last search's nearest run of segments first, then the others nearer
    /// than the best so far.
    fn nearest(&mut self, l: &Route, p: [f64; 2], lo: usize, hi: usize, enough: f64) -> f64 {
        let (c0, c1) = (lo / CHUNK, (hi - 1) / CHUNK + 1);
        let mut best = f64::INFINITY;
        let start = self.hint.clamp(c0, c1 - 1);
        for t in 0..(c1 - c0) {
            let c = c0 + (start - c0 + t) % (c1 - c0);
            let b = self.boxes[c];
            let bd = (b[0] - p[0]).max(p[0] - b[2]).max(0.0).hypot((b[1] - p[1]).max(p[1] - b[3]).max(0.0));
            if bd >= best {
                continue;
            }
            for k in (c * CHUNK).max(lo)..(c * CHUNK + CHUNK).min(hi) {
                let d = l.seg_near(k, p).0;
                if d < best {
                    (best, self.hint) = (d, c);
                    if d <= enough {
                        return d;
                    }
                }
            }
        }
        best
    }
}
/// The farthest point of O between `oa..ob` from L between `la..lb`, plane metres, and where on
/// O it is (offset). Sampled at O's vertices and every STRAY_STEP_M.
fn stray(o: &Route, oa: f64, ob: f64, l: &Route, la: f64, lb: f64, chunks: &mut Chunks) -> (f64, f64) {
    let (la, lb) = if lb < la { (lb, la) } else { (la, lb) };
    let mut ss: Vec<f64> = vec![];
    let n = ((ob - oa) / STRAY_STEP_M).floor() as usize;
    ss.extend((0..=n).map(|i| oa + i as f64 * STRAY_STEP_M));
    let v0 = o.ps.partition_point(|&a| a <= oa);
    let v1 = o.ps.partition_point(|&a| a < ob);
    ss.extend_from_slice(&o.ps[v0..v1.max(v0)]);
    ss.push(ob);
    if l.pts.len() < 2 {
        let q = l.pts.first().copied().unwrap_or([0.0, 0.0]);
        return ss.iter().map(|&s| (o.at(s), s)).map(|(p, s)| ((p[0] - q[0]).hypot(p[1] - q[1]), s)).fold((0.0, oa), |b, x| if x.0 > b.0 { x } else { b });
    }
    let (lo, hi) = l.seg_range(la, lb);
    let mut best = (0.0f64, oa);
    for s in ss {
        // a sample nearer L than the farthest so far cannot raise it
        let p = o.at(s);
        let d = chunks.nearest(l, p, lo, hi, best.0);
        if d > best.0 {
            best = (d, s);
        }
    }
    best
}
fn group_and_emit(l: &Route, o: &Route, mut ms: Vec<Meet>, all_shared: bool, cx: &dyn Fn(u32) -> u32, k: &dyn Fn(f64) -> f64, chunks: &mut Chunks, out: &mut Vec<f64>) {
    // resolve each station meeting's L node: the one nearest the meeting before (the first: the
    // one nearest the meeting after)
    for i in 0..ms.len() {
        if ms[i].alts.len() < 2 {
            continue;
        }
        let near = if i > 0 { ms[i - 1].sl2 } else { ms.get(1).map_or(ms[i].sl, |m| m.sl) };
        let best = *ms[i].alts.iter().min_by(|&&a, &&b| (l.node_s[a] - near).abs().total_cmp(&(l.node_s[b] - near).abs())).unwrap();
        ms[i].sl = l.node_s[best];
        ms[i].sl2 = ms[i].sl;
        ms[i].alts = vec![best];
    }
    struct Group {
        meets: Vec<Meet>,
        gaps: Vec<Gap>,
        dir: f64,
        first_sl: f64,
        split: [f64; 3],
    }
    let new_group = |m: Meet, split: [f64; 3]| {
        let first_sl = m.sl;
        let mut g = Group { meets: vec![], gaps: vec![], dir: 0.0, first_sl, split };
        if m.sl2 - m.sl > EPS_M {
            g.dir = 1.0;
        } else if m.sl - m.sl2 > EPS_M {
            g.dir = -1.0;
        }
        g.meets.push(m);
        g
    };
    let mut groups: Vec<Group> = vec![];
    let mut it = ms.into_iter();
    groups.push(new_group(it.next().unwrap(), [f64::NAN; 3]));
    for m in it {
        let g = groups.last_mut().unwrap();
        let p = g.meets.last().unwrap();
        let step = m.sl - p.sl2;
        let inner = m.sl2 - m.sl;
        let mono = g.dir == 0.0 || (step * g.dir >= -EPS_M && inner * g.dir >= -EPS_M);
        let run = m.so - p.so2;
        let mut together = false;
        let mut gap = None;
        if !mono {
            gap = Some([f64::NAN, run.max(0.0), step.abs()]);
        } else if run <= EPS_M {
            together = true;
        } else {
            let (st, at) = stray(o, p.so2, m.so, l, p.sl2, m.sl, chunks);
            let st = st / k(o.at(at)[1]);
            together = all_shared || st <= stray_allowed(run);
            gap = Some([st, run, step.abs()]);
        }
        if together {
            if let Some([st, run, ls]) = gap {
                g.gaps.push(Gap { la: p.sl2, lb: m.sl, stray: st, run, lstep: ls });
            }
            if g.dir == 0.0 {
                let d = m.sl2.max(m.sl) - g.first_sl;
                let e = m.sl2.min(m.sl) - g.first_sl;
                if d > EPS_M && e > -EPS_M {
                    g.dir = 1.0;
                } else if e < -EPS_M && d < EPS_M {
                    g.dir = -1.0;
                }
            }
            g.meets.push(m);
        } else {
            groups.push(new_group(m, gap.unwrap_or([f64::NAN; 3])));
        }
    }
    // a line that only touches twice gets no short-run allowance (Anita: higher on short runs
    // that touch often): two stations on its own track with the gap past STRAY_FAR of the run
    // are two crossings (Broadway's N beside Lexington's 6X from Canal St to Union Sq, 7% of 2 km)
    if !all_shared {
        let mut split: Vec<Group> = vec![];
        for mut g in groups {
            let pair = g.meets.len() == 2 && g.meets.iter().all(|m| !m.track()) && g.gaps.len() == 1;
            if pair && g.gaps[0].stray > STRAY_FLOOR_M + STRAY_FAR * g.gaps[0].run {
                let gp = g.gaps.pop().unwrap();
                let second = g.meets.pop().unwrap();
                split.push(g);
                split.push(new_group(second, [gp.stray, gp.run, gp.lstep]));
            } else {
                split.push(g);
            }
        }
        groups = split;
    }
    let o_stops: Vec<(usize, i32)> = o.stop_at.iter().enumerate().filter(|x| *x.1 >= 0).map(|(i, &s)| (i, s)).collect();
    for g in groups {
        let sls = g.meets.iter().flat_map(|m| [m.sl, m.sl2]);
        let (sl0, sl1) = sls.fold((f64::INFINITY, f64::NEG_INFINITY), |a, s| (a.0.min(s), a.1.max(s)));
        let (so0, so1) = g.meets.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |a, m| (a.0.min(m.so), a.1.max(m.so2)));
        let stations: Vec<&Meet> = g.meets.iter().filter(|m| !m.track()).collect();
        let mut cxs: Vec<u32> = stations.iter().map(|m| m.cx).collect();
        cxs.sort_unstable();
        cxs.dedup();
        let shared = g.meets.iter().any(|m| m.track());
        let kind = if shared || cxs.len() >= 2 { 1.0 } else { 2.0 };
        // shared-track stretches, joined where one edge follows the next
        let mut stretches: Vec<[f64; 2]> = vec![];
        let mut last_so2 = f64::NAN;
        for m in g.meets.iter().filter(|m| m.track()) {
            match stretches.last_mut() {
                Some(s) if (m.so - last_so2).abs() <= EPS_M && (m.sl - s[1]).abs() <= EPS_M => s[1] = m.sl2,
                _ => stretches.push([m.sl, m.sl2]),
            }
            last_so2 = m.so2;
        }
        // O's own stops inside the span, placed at the nearest point of L between the meetings
        // on either side
        let mut own: Vec<(i32, f64, u32)> = vec![];
        for &(i, si) in &o_stops {
            let s = o.node_s[i];
            if s <= so0 + EPS_M || s >= so1 - EPS_M || cxs.binary_search(&cx(o.nodes[i])).is_ok() {
                continue;
            }
            let before = g.meets.iter().rev().find(|m| m.so2 <= s);
            let after = g.meets.iter().find(|m| m.so >= s);
            let (la, lb) = (before.map_or(sl0, |m| m.sl2), after.map_or(sl1, |m| m.sl));
            own.push((si, l.nearest(o.at(s), la.min(lb), la.max(lb)).1, cx(o.nodes[i])));
        }
        let before = o_stops.iter().filter(|x| o.node_s[x.0] < so0 - EPS_M).count();
        let after = o_stops.iter().filter(|x| o.node_s[x.0] > so1 + EPS_M).count();
        out.extend([o.line as f64, kind, shared as u8 as f64, if g.dir < 0.0 { -1.0 } else { 1.0 }, sl0, sl1, so0, so1, before as f64, after as f64]);
        out.extend(g.split);
        out.extend([stations.len() as f64, stretches.len() as f64, g.gaps.len() as f64, own.len() as f64]);

        for m in &stations {
            let lk = m.alts[0];
            out.extend([m.cx as f64, m.sl, m.so, l.stop_at[lk] as f64, o.stop_at[m.on] as f64]);
        }
        for s in &stretches {
            out.extend(s);
        }
        for gp in &g.gaps {
            out.extend([gp.la, gp.lb, gp.stray, gp.run, gp.lstep]);
        }
        for (si, sl, c) in own {
            out.extend([si as f64, sl, c as f64]);
        }
    }
}

#[cfg(test)]
#[path = "related_tests.rs"]
mod tests;
