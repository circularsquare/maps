//! T-3378: BRT lanes shared between the lines on one road. A BRT line's lanes are its route's road in town
//! (`Pace::town_m`), in the pace's steps. A step of one line shares another line's lane where the step's middle
//! lies within `MATCH_M` of that line's road in town and the two run within `MATCH_DEG` of each other, either
//! way (two lines on one road never have the same points, and a divided road has two carriageways). A line never
//! shares with itself: a route out and back along one road pays both ways, as before.

use super::{at_road, Pace, Poly};
use crate::track::net::Network;
use crate::track::streets::brt_in_town;
use std::collections::HashMap;

/// Ground metres from another line's road within which a step's middle shares its lane.
pub const MATCH_M: f64 = 20.0;
/// Degrees two lines' headings may differ by on a shared lane, either way.
pub const MATCH_DEG: f64 = 30.0;
/// The grid's square, ground metres: more than `MATCH_M`, so a step looks in its own square only.
const CELL_M: f64 = 50.0;

/// One pace step of a route in town: its middle, unit heading (frame), road metres, the frame's metres a
/// ground metre there (`Network::k_at`), and where it starts.
#[derive(Clone, Copy, Debug)]
pub struct Step {
    pub x: f64,
    pub y: f64,
    dx: f64,
    dy: f64,
    pub m: f64,
    k: f64,
    /// road metres along the route where it starts
    a: f64,
}

/// A route's road in town: its steps, and the stretches of road (frame) they cover.
pub fn town_steps(net: &Network, pts: &[[f64; 2]], kinds: &[u8]) -> (Vec<Step>, Vec<[[f64; 2]; 2]>) {
    let (mut steps, mut segs) = (vec![], vec![]);
    each_town_step(net, pts, kinds, |s, line| {
        steps.push(s);
        segs.extend(line.windows(2).map(|w| [w[0], w[1]]));
    });
    (steps, segs)
}

/// T-3944: a route's road in town as runs of steps of one kind (`kind` of each step), each `[kind, points, x, y,
/// ...]` flat, for the map (`Network::brt_marks`).
pub fn marks(net: &Network, pts: &[[f64; 2]], kinds: &[u8], kind: impl Fn(&Step) -> u8) -> Vec<f64> {
    let mut out: Vec<f64> = vec![];
    // where the open run starts in `out`, its kind, and the road metres it ends at
    let mut open: Option<(usize, u8, f64)> = None;
    each_town_step(net, pts, kinds, |s, line| {
        let k = kind(&s);
        let joins = open.is_some_and(|(_, ok, end)| ok == k && (s.a - end).abs() < 1e-6);
        if !joins {
            out.extend([k as f64, 0.0]);
            open = Some((out.len() - 2, k, 0.0));
        }
        let (at, _, _) = open.unwrap();
        // a joined run's first point is the last one's end
        for p in &line[if joins { 1 } else { 0 }..] {
            out.extend(p);
            out[at + 1] += 1.0;
        }
        open = Some((at, k, s.a + s.m));
    });
    out
}

/// Each step of a route's road in town, with its stretch of road as points (frame): its start, the road's own
/// points between, its end.
fn each_town_step(net: &Network, pts: &[[f64; 2]], kinds: &[u8], mut f: impl FnMut(Step, &[[f64; 2]])) {
    if pts.len() < 2 {
        return;
    }
    let poly = Poly::new(net, pts, kinds);
    let len = poly.road_len();
    let pace = Pace::along(net, pts, &poly.road, kinds, len);
    let at = |s: f64| {
        let (x, y) = at_road(pts, &poly.road, s);
        [x, y]
    };
    let mut line: Vec<[f64; 2]> = vec![];
    for (i, &c) in pace.car.iter().enumerate() {
        if !brt_in_town(c) {
            continue;
        }
        let (a, b) = (i as f64 * pace.step, ((i + 1) as f64 * pace.step).min(len));
        let (pa, pb, pm) = (at(a), at(b), at(0.5 * (a + b)));
        let (dx, dy) = (pb[0] - pa[0], pb[1] - pa[1]);
        let d = dx.hypot(dy);
        let (dx, dy) = if d > 0.0 { (dx / d, dy / d) } else { (0.0, 0.0) };
        // the road's own points between, so a bend is followed
        let (k0, k1) = (poly.road.partition_point(|&r| r <= a), poly.road.partition_point(|&r| r < b));
        line.clear();
        line.push(pa);
        line.extend(pts.iter().take(k1).skip(k0));
        line.push(pb);
        f(Step { x: pm[0], y: pm[1], dx, dy, m: pace.step, k: net.k_at(pm[1]), a }, &line);
    }
}

/// What a line's lanes come to beside the others' (`Lanes::split`), road metres.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Split {
    /// its road in town
    pub town: f64,
    /// not shared with a line before it in the order: what it adds to the lanes' built value
    pub first: f64,
    /// shared with no other line
    pub alone: f64,
    /// its part of the lanes' upkeep: each step split evenly between the lines on it
    pub share: f64,
}

/// The lanes of some BRT lines, in an order (`build`).
pub struct Lanes {
    lines: Vec<u32>,
    steps: Vec<Vec<Step>>,
    /// each stretch of road and the position in `lines` of the line it is of
    segs: Vec<([[f64; 2]; 2], u32)>,
    grid: HashMap<(i64, i64), Vec<u32>>,
    /// the grid's square, frame units
    cell: f64,
}

impl Lanes {
    /// The lanes of `lines` (bus lines; any other has none), in that order.
    pub fn build(net: &Network, lines: &[u32]) -> Lanes {
        let mut out = Lanes { lines: lines.to_vec(), steps: vec![], segs: vec![], grid: HashMap::new(), cell: 0.0 };
        for (i, &l) in lines.iter().enumerate() {
            let (steps, segs) = match net.line_bus.get(l as usize).and_then(|b| b.as_ref()) {
                Some(r) if net.line_ok(l) => town_steps(net, &r.points, &r.kinds),
                _ => (vec![], vec![]),
            };
            if out.cell == 0.0 {
                out.cell = steps.first().map_or(0.0, |s| CELL_M * s.k);
            }
            out.steps.push(steps);
            out.segs.extend(segs.into_iter().map(|s| (s, i as u32)));
        }
        if out.cell > 0.0 {
            for (j, (s, _)) in out.segs.iter().enumerate() {
                let pad = MATCH_M * net.k_at(0.5 * (s[0][1] + s[1][1])) * 1.05;
                let (x0, x1) = (s[0][0].min(s[1][0]) - pad, s[0][0].max(s[1][0]) + pad);
                let (y0, y1) = (s[0][1].min(s[1][1]) - pad, s[0][1].max(s[1][1]) + pad);
                let c = out.cell;
                for cx in (x0 / c).floor() as i64..=(x1 / c).floor() as i64 {
                    for cy in (y0 / c).floor() as i64..=(y1 / c).floor() as i64 {
                        out.grid.entry((cx, cy)).or_default().push(j as u32);
                    }
                }
            }
        }
        out
    }

    /// The positions in the order of the lines other than `own` whose lanes `s` shares, ascending.
    pub fn sharers(&self, s: &Step, own: Option<u32>) -> Vec<u32> {
        self.sharers_way(s, own).into_iter().map(|x| x.0).collect()
    }

    /// `sharers`, each with whether that line's route runs the way `s` does there (else the other way).
    pub fn sharers_way(&self, s: &Step, own: Option<u32>) -> Vec<(u32, bool)> {
        let mut out: Vec<(u32, bool)> = vec![];
        if self.cell <= 0.0 {
            return out;
        }
        let key = ((s.x / self.cell).floor() as i64, (s.y / self.cell).floor() as i64);
        let Some(cand) = self.grid.get(&key) else { return out };
        let cos = MATCH_DEG.to_radians().cos();
        let tol = MATCH_M * s.k;
        for &j in cand {
            let (seg, i) = self.segs[j as usize];
            if Some(i) == own || out.iter().any(|x| x.0 == i) {
                continue;
            }
            let (ex, ey) = (seg[1][0] - seg[0][0], seg[1][1] - seg[0][1]);
            let e = ex.hypot(ey);
            let dot = ex * s.dx + ey * s.dy;
            if e <= 0.0 || dot.abs() < cos * e {
                continue;
            }
            let t = (((s.x - seg[0][0]) * ex + (s.y - seg[0][1]) * ey) / (e * e)).clamp(0.0, 1.0);
            if (s.x - seg[0][0] - t * ex).hypot(s.y - seg[0][1] - t * ey) <= tol {
                out.push((i, dot > 0.0));
            }
        }
        out.sort_unstable_by_key(|x| x.0);
        out
    }

    /// Line `l`'s lanes beside the others' (zero for a line not among them).
    pub fn split(&self, l: u32) -> Split {
        self.split_upto(l, u32::MAX)
    }

    /// `split` beside the lines before position `upto` in the order only.
    pub fn split_upto(&self, l: u32, upto: u32) -> Split {
        let Some(i) = self.lines.iter().position(|&x| x == l) else { return Split::default() };
        let mut out = Split::default();
        for s in &self.steps[i] {
            let mut by = self.sharers(s, Some(i as u32));
            by.retain(|&j| j < upto);
            out.town += s.m;
            if by.first().map_or(true, |&j| j > i as u32) {
                out.first += s.m;
            }
            if by.is_empty() {
                out.alone += s.m;
            }
            out.share += s.m / (1 + by.len()) as f64;
        }
        out
    }

    /// The road metres of `steps` (another route's, `town_steps`) that no line here has lanes on.
    pub fn new_m(&self, steps: &[Step]) -> f64 {
        self.new_m_upto(steps, None, u32::MAX)
    }

    /// `new_m` beside the lines before position `upto` in the order, line `own`'s position left out.
    pub fn new_m_upto(&self, steps: &[Step], own: Option<u32>, upto: u32) -> f64 {
        steps.iter().filter(|s| self.sharers(s, own).iter().all(|&j| j >= upto)).map(|s| s.m).sum()
    }

    /// The position of line `l` in the order.
    pub fn position(&self, l: u32) -> Option<u32> {
        self.lines.iter().position(|&x| x == l).map(|i| i as u32)
    }

    /// The lines, in the order built.
    pub fn lines(&self) -> &[u32] {
        &self.lines
    }
}

/// T-3378 stage 2: buses an hour one way a BRT lane takes (Anita, 11-21: "60 buses an hour per lane each way, lines
/// on it cut to fit past that, as trains do on full track"). Past it each line on the lane runs fewer.
pub const LANE_BUSES_H: f64 = 60.0;

/// One way of one stretch of lane: the lines whose buses run along it that way (ids, ascending), and the middle
/// of one of its steps (frame) for where a panel names it.
#[derive(Clone, Debug, PartialEq)]
pub struct LaneWay {
    pub lines: Vec<u32>,
    pub x: f64,
    pub y: f64,
}

/// T-3378 stage 2: the room on the constructed BRT lines' lanes (`Network::brt_room`): every distinct set of lines
/// running one way along a stretch, as track's capacity resources are for trains (`Capacity::fill` cuts the lines'
/// rates to fit them, `line_limits` counts them).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Room {
    pub ways: Vec<LaneWay>,
}

impl Room {
    /// The ways of the lanes of constructed BRT lines `lines`. A line runs both ways along its route unless it is
    /// a loop run one way (`Network::line_one_way`).
    pub fn build(net: &Network, lines: &[u32]) -> Room {
        let lanes = Lanes::build(net, lines);
        let both: Vec<bool> = lines
            .iter()
            .map(|&l| !(net.line_bus.get(l as usize).and_then(|b| b.as_ref()).is_some_and(|r| super::is_closed(&r.points)) && net.line_one_way.get(l as usize).copied().unwrap_or(false)))
            .collect();
        let mut ways: Vec<LaneWay> = vec![];
        for (i, steps) in lanes.steps.iter().enumerate() {
            for s in steps {
                let by = lanes.sharers_way(s, Some(i as u32));
                // the way along the step (true) and the other: who runs each
                for fwd in [true, false] {
                    if !both[i] && !fwd {
                        continue;
                    }
                    let mut on = vec![lines[i]];
                    on.extend(by.iter().filter(|&&(j, same)| both[j as usize] || same == fwd).map(|&(j, _)| lines[j as usize]));
                    on.sort_unstable();
                    if !ways.iter().any(|w| w.lines == on) {
                        ways.push(LaneWay { lines: on, x: s.x, y: s.y });
                    }
                }
            }
        }
        Room { ways }
    }

    /// The fullest way line `l` runs on at rates `tph` (buses an hour each way by line): (buses an hour through
    /// it, its middle).
    pub fn fullest(&self, l: u32, tph: impl Fn(u32) -> f64) -> Option<(f64, f64, f64)> {
        self.ways
            .iter()
            .filter(|w| w.lines.contains(&l))
            .map(|w| (w.lines.iter().map(|&x| tph(x)).sum::<f64>(), w.x, w.y))
            .max_by(|a, b| a.0.total_cmp(&b.0))
    }
}
