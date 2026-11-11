//! Capacity and delay (SPEC 6.2, notes/T-008.md 5): resources, utilisation per demand level from
//! the scheduled trains per hour, Kingman's delay, and holds for the run-time profiles. Each
//! capacity pass uses fixed rates; whole-line count schedules repeat it within the bounded
//! rate/round-trip solve in world.rs (T-104). Legacy frequency schedules need one pass.
//!
//! Incremental (T-050): after an edit, `Capacity::update` throws away only the resources around
//! what it touched (sections reaching a touched edge or node, platforms, termini and junction
//! moves at touched nodes, crossings of touched edges), builds those again with the same code as
//! the global pass, and re-solves them and the resources a rescheduled line uses. The result is
//! the global pass's, in another order (tests below check it on random edit sequences).

use super::geom;
use super::lanes::ends_in;
use super::net::{path_dir, path_edge, CrossKind, Crossing, LoopKind, Network};
use super::params::*;
use super::profile::{self, Hold};
use super::overtake::{self, Plan};
use super::service::LineSvc;
use super::upgrades;
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResKind {
    /// A run of directed double track over which the set of lines does not change.
    Section,
    /// Single track between passing points, both directions.
    SingleTrack,
    /// One platform track at a station: stopping and passing trains.
    Platform,
    /// Reversing trains at a line's end.
    Terminus,
    /// A junction move, loaded by the moves that cross it at the same level.
    Junction,
    /// One direction of one route through a flat crossing, loaded by the other route.
    Crossing,
    /// T-1924: a faster line's direction behind slower lines on a stretch of track it shares
    /// with them, where it catches them up and follows (the following loss, notes/T-1900.md N1).
    Follow,
    /// T-2231: a slower line's direction waiting at a passing station for a faster line to pass
    /// (`overtake`): the one owner the slower line at its stop, the faster line the other user.
    Wait,
}

/// A line direction using a resource. Owners get the delay; the others only load it.
/// On a `ResKind::Follow` resource (T-1924) the one owner is the faster line's stretch, from `at`
/// to `occ` (path offsets in its run), where its loss slows it (T-2018, `holds_for`), and each
/// other user is a slower line ahead: `occ` is how much longer it takes over the shared stretch
/// (s), `at` the spacing a train following it keeps (s).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct User {
    pub line: u32,
    pub run: u8,
    pub owner: bool,
    /// Occupancy per train, s.
    pub occ: f64,
    /// Where an owner waits: path offset in its run (ignored when `stop` >= 0).
    pub at: f64,
    /// Stop index whose dwell absorbs the delay (platform holds), or -1.
    pub stop: i32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Resource {
    pub kind: ResKind,
    pub x: f64,
    pub y: f64,
    /// The node (junction, station) or u32::MAX.
    pub node: u32,
    /// An edge on it (sections, crossings) or u32::MAX.
    pub edge: u32,
    pub first: u32,
    pub count: u32,
}

/// What a resource was built from, so the incremental pass knows what an edit invalidates.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Tag {
    /// A section: its directed track keys (`edge * 2 + direction`, `edge * 2` on single track),
    /// a span of `Capacity::keys`.
    Section { first: u32, count: u32 },
    /// A terminus or junction move at `Resource::node`.
    Node,
    /// A platform at `Resource::node`; `reverses` is never set now (T-3107: a turn holds no platform).
    Platform { reverses: bool },
    /// A flat crossing of `Resource::edge` with `other`. `yields`: a tram route crossing a train route
    /// (T-3032): its non-owners, the train passes, load it but are never limited by it.
    Crossing { other: u32, yields: bool },
    /// T-1924: line `line`'s following loss on one stretch (`build_follow`); rebuilt whenever any
    /// edge the line runs over is.
    Follow { line: u32 },
    /// T-2231: a slower line's wait for faster line `line` at its plan's point `point`, built and
    /// rebuilt with that line's following losses.
    Wait { line: u32, point: u16 },
}

/// Every resource with its users, utilisation and delay. Resource indices are valid until the
/// next update; the incremental pass does not keep their order.
#[derive(Clone, Debug, Default)]
pub struct Capacity {
    pub res: Vec<Resource>,
    pub users: Vec<User>,
    /// Utilisation per demand level (high, medium, low).
    pub rho: Vec<[f64; 3]>,
    /// Delay per train per level: continuous for resources used by count schedules; legacy-only
    /// resources round to whole seconds and omit holds under 2 s.
    pub delay: Vec<[f64; 3]>,
    tag: Vec<Tag>,
    keys: Vec<u32>,
    /// T-2231: per resource, the overtake plans it reads: a following loss's, one per slower user
    /// marked `FOLLOW_PASSING` in order; a wait's, its one plan. Empty for every other resource.
    plans: Vec<Vec<Arc<Plan>>>,
    /// Entries of `users` and `keys` no resource points at any more; compacted past half.
    dead_users: usize,
    dead_keys: usize,
}

impl Capacity {
    pub fn users_of(&self, r: usize) -> &[User] {
        let x = &self.res[r];
        &self.users[x.first as usize..(x.first + x.count) as usize]
    }

    /// The edges a resource caps the track of (T-819): a section's, and a flat crossing's two;
    /// none for stations, termini and junction moves.
    pub fn track_edges(&self, r: usize) -> Vec<u32> {
        match self.tag[r] {
            Tag::Section { first, count } => self.keys[first as usize..(first + count) as usize].iter().map(|k| k / 2).collect(),
            Tag::Crossing { yields: true, .. } => vec![self.res[r].edge],
            Tag::Crossing { other, .. } => vec![self.res[r].edge, other],
            _ => vec![],
        }
    }

    /// Whether the resource can hold back user `u`'s line: not the trains at a tram's flat crossing
    /// of their track (T-3032), which only take its room.
    pub fn caps(&self, r: usize, u: &User) -> bool {
        u.owner || !matches!(self.tag[r], Tag::Crossing { yields: true, .. })
    }

    /// Whether the resource caps throughput (T-517). A station crossover's load counts through
    /// trains that never conflict with each other, so it only delays.
    /// A following loss (T-1924) or a wait for faster trains (T-2231) neither fills nor caps
    /// anything either.
    pub fn limits(&self, r: usize) -> bool {
        !matches!(self.tag[r], Tag::Follow { .. } | Tag::Wait { .. })
    }

    /// The kind the app words alerts by (`ResKind as u8` for 0 to 5), finer where the cause differs:
    /// 7 track several lines run over, 11 a faster line held behind slower ones (T-1924; never a
    /// map dot, its utilisation is 0), 12 track several lines join from different edges at its
    /// start, a merge, where they wait for it (T-2185; `Resource::node` is that node), 13 a slower
    /// line waiting at a passing station for faster trains to pass (T-2231; never a map dot;
    /// `Resource::node` is the station). Kinds 6, 8, 9, 10 and 14 (turning variants) are no longer
    /// made (T-3107: every turn is a terminus, 3); their numbers stay for old saves.
    pub fn wire_kind(&self, r: usize) -> u8 {
        let kind = self.res[r].kind;
        match self.tag[r] {
            Tag::Follow { .. } => 11,
            Tag::Wait { .. } => 13,
            Tag::Section { .. } if kind == ResKind::Section && self.res[r].node != u32::MAX => 12,
            Tag::Section { .. } if kind == ResKind::Section => {
                let users = self.users_of(r);
                if users.iter().any(|u| u.line != users[0].line) { 7 } else { 0 }
            }
            _ => kind as u8,
        }
    }

    /// Trains an hour through a resource at a level, each line direction once.
    pub fn trains(&self, net: &Network, r: usize, lev: usize) -> f64 {
        let mut seen: Vec<(u32, u8)> = vec![];
        let mut n = 0.0;
        for u in self.users_of(r) {
            if !seen.contains(&(u.line, u.run)) {
                seen.push((u.line, u.run));
                n += net.line_tph[u.line as usize][lev] as f64;
            }
        }
        n
    }

    /// Per line and level, its fullest limiting resource: (utilisation, resource), or (0, MAX).
    pub fn bottlenecks(&self, lines: usize) -> Vec<[(f64, u32); 3]> {
        let mut out = vec![[(0.0, u32::MAX); 3]; lines];
        for r in (0..self.res.len()).filter(|&r| self.limits(r)) {
            for u in self.users_of(r).iter().filter(|u| self.caps(r, u)) {
                let b = &mut out[u.line as usize];
                for lev in 0..LEVELS {
                    if self.rho[r][lev] > b[lev].0 {
                        b[lev] = (self.rho[r][lev], r as u32);
                    }
                }
            }
        }
        out
    }

    /// T-679: per line and level, the most trains an hour it can run with every other line's rate
    /// as it stands, and the limiting resource that sets it: (trains an hour, resource), or
    /// (infinity, MAX) where nothing limits it. `fill` cuts a line at the same rate.
    pub fn line_limits(&self, net: &Network) -> Vec<[(f64, u32); 3]> {
        let mut out = vec![[(f64::INFINITY, u32::MAX); 3]; net.line_count()];
        let mut own: Vec<(u32, f64, bool)> = vec![];
        for r in (0..self.res.len()).filter(|&r| self.limits(r)) {
            own.clear();
            for u in self.users_of(r) {
                match own.iter_mut().find(|o| o.0 == u.line) {
                    Some(o) => o.1 += u.occ / 3600.0,
                    None => own.push((u.line, u.occ / 3600.0, self.caps(r, u))),
                }
            }
            for lev in 0..LEVELS {
                let load: f64 = own.iter().map(|&(l, a, _)| net.line_tph[l as usize][lev] as f64 * a).sum();
                for &(l, a, _) in own.iter().filter(|o| o.1 > 0.0 && o.2) {
                    let most = (net.line_tph[l as usize][lev] as f64 + (1.0 - load) / a).max(0.0);
                    let b = &mut out[l as usize][lev];
                    if most < b.0 {
                        *b = (most, r as u32);
                    }
                }
            }
        }
        out
    }

    /// T-517: the rates `want` (line, trains an hour) at one level, cut so that no limiting
    /// resource exceeds its capacity. Progressive filling: every line's rate grows in proportion
    /// to its wanted rate until a resource it uses fills; its lines then stop there. Lines not in
    /// `want` keep their rates and only take room. A rate that fits comes back unchanged.
    pub fn fill(&self, net: &Network, want: &[(u32, f64)], lev: usize) -> Vec<f64> {
        let mut slot = vec![u32::MAX; net.line_count()];
        for (i, w) in want.iter().enumerate() {
            slot[w.0 as usize] = i as u32;
        }
        // per limiting resource: utilisation from fixed lines, and per wanted line its share per train an hour
        let mut res: Vec<(f64, Vec<(u32, f64)>)> = vec![];
        for r in (0..self.res.len()).filter(|&r| self.limits(r)) {
            let (mut fixed, mut coef) = (0.0, vec![]);
            for u in self.users_of(r) {
                match if self.caps(r, u) { slot[u.line as usize] } else { u32::MAX } {
                    u32::MAX => fixed += net.line_tph[u.line as usize][lev] as f64 * u.occ / 3600.0,
                    s => match coef.iter_mut().find(|c: &&mut (u32, f64)| c.0 == s) {
                        Some(c) => c.1 += u.occ / 3600.0,
                        None => coef.push((s, u.occ / 3600.0)),
                    },
                }
            }
            if !coef.is_empty() {
                res.push((fixed, coef));
            }
        }
        let mut scale: Vec<Option<f64>> = vec![None; want.len()];
        loop {
            // the common scale at which each resource fills
            let at: Vec<f64> = res.iter().map(|(fixed, coef)| {
                let (mut used, mut grow) = (*fixed, 0.0);
                for &(s, a) in coef {
                    match scale[s as usize] {
                        Some(x) => used += want[s as usize].1 * a * x,
                        None => grow += want[s as usize].1 * a,
                    }
                }
                if grow > 0.0 { ((1.0 - used) / grow).max(0.0) } else { f64::INFINITY }
            }).collect();
            let first = at.iter().copied().fold(f64::INFINITY, f64::min);
            if first >= 1.0 {
                break;
            }
            for (k, (_, coef)) in res.iter().enumerate() {
                if at[k] <= first {
                    for &(s, _) in coef {
                        scale[s as usize].get_or_insert(first);
                    }
                }
            }
        }
        want.iter().zip(&scale).map(|(w, x)| w.1 * x.unwrap_or(1.0)).collect()
    }
}

/// Delay per train in seconds (SPEC 6.2): Kingman with a small variability term, a straight line
/// above 90%, plus an hour's fluid queue above 100%.
pub fn kingman(rho: f64, mix: f64, s: f64) -> f64 {
    let c = KINGMAN_C0 * mix;
    let w = if rho <= RHO_KNEE {
        c * s * rho / (1.0 - rho)
    } else {
        let k = 1.0 - RHO_KNEE;
        c * s * RHO_KNEE / k + c * s / (k * k) * (rho - RHO_KNEE)
    };
    w + if rho > 1.0 { OVERLOAD_WINDOW_S / 2.0 * (1.0 - 1.0 / rho) } else { 0.0 }
}

struct Uf(Vec<u32>);
impl Uf {
    fn find(&mut self, mut a: u32) -> u32 {
        while self.0[a as usize] != a {
            let p = self.0[self.0[a as usize] as usize];
            self.0[a as usize] = p;
            a = p;
        }
        a
    }
    fn union(&mut self, a: u32, b: u32) {
        let (a, b) = (self.find(a), self.find(b));
        if a != b {
            let (lo, hi) = if a < b { (a, b) } else { (b, a) };
            self.0[hi as usize] = lo;
        }
    }
}

/// T-3103: whether two moves through junction `n` could cross at its level, by the rule
/// `build_nodes` holds them to, over every move in its table (run by a line or not, T-3137). The
/// map's flat-junction badge; a flying junction has none.
pub fn junction_crosses(net: &Network, n: u32) -> bool {
    let p = &net.node_ports[n as usize];
    if p.n < 3 || net.node_flying[n as usize] {
        return false;
    }
    let m = &net.node_moves[n as usize];
    let moves: Vec<(f64, f64)> = (0..p.n as usize).flat_map(|i| ends_in(m.to[i]).map(move |j| m.at(p, i, j))).collect();
    moves.iter().enumerate().any(|(k, x)| moves[k + 1..].iter().any(|y| (x.0 - y.0) * (x.1 - y.1) < 0.0))
}

/// Trains on the edge hold only their own direction's track: double track, or single track that runs one way (T-3094, T-3137).
fn directed(net: &Network, e: u32) -> bool {
    net.edge_tracks[e as usize] != 1 || net.edge_way(e) != 0
}

/// The directed track a path entry runs on (shared by both directions on two-way single track).
fn key_of(net: &Network, p: u32) -> u32 {
    let e = path_edge(p);
    if !directed(net, e) {
        e * 2
    } else {
        e * 2 + path_dir(p)
    }
}

#[allow(clippy::too_many_arguments)]
fn push_res(cap: &mut Capacity, kind: ResKind, x: f64, y: f64, node: u32, edge: u32, users: &[User], tag: Tag) {
    if users.is_empty() {
        return;
    }
    cap.res.push(Resource { kind, x, y, node, edge, first: cap.users.len() as u32, count: users.len() as u32 });
    cap.users.extend_from_slice(users);
    cap.tag.push(tag);
    cap.plans.push(vec![]);
    cap.rho.push([0.0; 3]);
    cap.delay.push([0.0; 3]);
}

fn hold_back(lines: &[LineSvc], l: usize) -> f64 {
    lines[l].train_len / 2.0 + HOLD_MARGIN_M
}

/// Track sections over the keys `in_k` accepts, from the runs of `cand` (which must hold every
/// running line on those keys). Sections join across two-ended nodes where the set of line
/// directions does not change; a key outside `in_k` is never joined to (the incremental pass
/// passes whole sections).
/// T-1805: a node's edge ends that a running train line uses. A branch no train runs on yet
/// (blueprint, or constructed with no line) leaves the track through the node one section.
fn live_ports(net: &Network, lines: &[LineSvc], node: u32) -> usize {
    net.node_ports[node as usize].ends().iter().filter(|&&c| {
        net.lines_on_edge(c >> 1).iter().any(|&l| lines.get(l as usize).is_some_and(|ls| ls.running && !ls.is_bus()))
    }).count()
}

fn build_sections(net: &Network, lines: &[LineSvc], cand: &[u32], in_k: &dyn Fn(u32) -> bool, cap: &mut Capacity) {
    // (key, line, run, seg index)
    let mut occ: Vec<(u32, u32, u8, u32)> = vec![];
    for &l in cand {
        let ls = &lines[l as usize];
        if !ls.running || ls.is_bus() {
            continue;
        }
        for r in 0..ls.live_runs() {
            for (i, &p) in ls.runs[r].segs.iter().enumerate() {
                let k = key_of(net, p);
                if in_k(k) {
                    occ.push((k, l, r as u8, i as u32));
                }
            }
        }
    }
    if occ.is_empty() {
        return;
    }
    occ.sort_unstable();
    let mut keys: Vec<u32> = occ.iter().map(|o| o.0).collect();
    keys.dedup();
    // Per key, a hash of its (line, run) set: sections join where the set does not change.
    let mut sig = vec![0u64; keys.len()];
    {
        let mut k = 0;
        let mut prev = (u32::MAX, u8::MAX);
        for o in &occ {
            while keys[k] != o.0 {
                k += 1;
                prev = (u32::MAX, u8::MAX);
            }
            if (o.1, o.2) != prev {
                prev = (o.1, o.2);
                let x = ((o.1 as u64) << 8 | o.2 as u64).wrapping_add(0x9E37_79B9_7F4A_7C15);
                sig[k] = (sig[k] ^ x).wrapping_mul(0x1000_0000_01B3).rotate_left(17);
            }
        }
    }
    let kidx = |k: u32| keys.binary_search(&k).ok().map(|i| i as u32);
    let mut uf = Uf((0..keys.len() as u32).collect());
    for &l in cand {
        let ls = &lines[l as usize];
        if !ls.running || ls.is_bus() {
            continue;
        }
        // a circle's runs carry on through their first stop (T-798)
        let wrap = ls.kind == LoopKind::Circle;
        for run in ls.runs.iter().take(ls.live_runs()) {
            let ns = run.segs.len();
            for i in 0..if wrap { ns } else { ns.saturating_sub(1) } {
                let node = run.nodes[i + 1] as usize;
                if net.node_ports[node].n != 2 && live_ports(net, lines, node as u32) != 2 {
                    continue;
                }
                let (p, q) = (run.segs[i], run.segs[(i + 1) % ns]);
                if net.turns_back(p, q) {
                    continue;
                }
                let (Some(a), Some(b)) = (kidx(key_of(net, p)), kidx(key_of(net, q))) else { continue };
                let (te, tf) = (directed(net, path_edge(p)), directed(net, path_edge(q)));
                let join = if te && tf { sig[a as usize] == sig[b as usize] } else { !te && !tf && net.node_platform[node] == 0 };
                if join {
                    uf.union(a, b);
                }
            }
        }
    }
    // A component's root is its smallest key index, so its keys and users sort by root.
    let mut members: Vec<(u32, u32)> = (0..keys.len() as u32).map(|i| (uf.find(i), keys[i as usize])).collect();
    members.sort_unstable();
    let mut grouped: Vec<(u32, u32, u8, u32)> = occ.iter().map(|o| (uf.find(kidx(o.0).unwrap()), o.1, o.2, o.3)).collect();
    grouped.sort_unstable();
    let mut g = 0;
    let mut m = 0;
    let mut users = vec![];
    let mut joins: Vec<(u32, u32)> = vec![]; // per user: the node it enters at, the edge it comes from (MAX: starts there)
    while g < grouped.len() {
        let root = grouped[g].0;
        let end = grouped[g..].partition_point(|x| x.0 == root) + g;
        let first_edge = keys[root as usize] / 2;
        let single = !directed(net, first_edge);
        users.clear();
        joins.clear();
        let mut u = g;
        while u < end {
            let (l, r) = (grouped[u].1, grouped[u].2);
            let ue = grouped[u..end].partition_point(|x| (x.1, x.2) == (l, r)) + u;
            let ls = &lines[l as usize];
            let run = &ls.runs[r as usize];
            let (mut entry, mut exit) = (f64::INFINITY, f64::NEG_INFINITY);
            for x in &grouped[u..ue] {
                let i = x.3 as usize;
                entry = entry.min(run.seg_off[i]);
                exit = exit.max(run.node_s[i + 1]);
            }
            let i0 = grouped[u..ue].iter().map(|x| x.3).min().unwrap_or(0) as usize;
            let from = match (i0, run.segs.last()) {
                (0, Some(&p)) if ls.kind == LoopKind::Circle => path_edge(p),
                (0, _) => u32::MAX,
                _ => path_edge(run.segs[i0 - 1]),
            };
            // (arriving on the track it leaves on: it turns back there, no join)
            joins.push((run.nodes[i0], if from == path_edge(run.segs[i0]) { u32::MAX } else { from }));
            let ph = &run.free.phases;
            let occ_s = if single {
                profile::time_at(ph, exit) - profile::time_at(ph, entry) + SINGLE_TRACK_MARGIN_S
            } else {
                net.edge_section_s(path_edge(run.segs[i0])) + profile::v_max(ph, entry, exit) / SIGNAL_BRAKE
            };
            users.push(User { line: l, run: r, owner: true, occ: occ_s, at: entry - hold_back(lines, l as usize), stop: -1 });
            u = ue;
        }
        let me = members[m..].partition_point(|x| x.0 == root) + m;
        let kf = cap.keys.len() as u32;
        cap.keys.extend(members[m..me].iter().map(|x| x.1));
        m = me;
        // T-2185: lines joining this track from two or more edges at one node (a merge) wait there
        // for it, so it sits at that node and is worded as the merge (`node` set, wire kind 12)
        let merge = !single && joins.iter().all(|j| j.0 == joins[0].0) && {
            let mut from: Vec<u32> = joins.iter().map(|j| j.1).filter(|&e| e != u32::MAX).collect();
            from.sort_unstable();
            from.dedup();
            from.len() >= 2
        };
        let (x, y, node) = if merge {
            let n = joins[0].0;
            (net.node_x[n as usize], net.node_y[n as usize], n)
        } else {
            let (x, y, _) = geom::pos_at(net.edge_pieces(first_edge), net.edge_len[first_edge as usize] / 2.0);
            (x, y, u32::MAX)
        };
        let kind = if single { ResKind::SingleTrack } else { ResKind::Section };
        push_res(cap, kind, x, y, node, first_edge, &users, Tag::Section { first: kf, count: (cap.keys.len() as u32) - kf });
        g = end;
    }
}

/// Active station capacity follows constructed track; planned extensions cannot add platforms.
pub(super) fn active_node_tracks(net: &Network, node: u32) -> u8 {
    net.node_ports[node as usize].ends().iter().filter_map(|&p| {
        let e = (p >> 1) as usize;
        net.edge_built[e].then_some(net.edge_tracks[e])
    }).max().unwrap_or(1)
}

/// The places a turning train can stand at node `n` (T-3107, Anita 2026-11-10: a dead end and a
/// through station turn alike): with a terminal every platform track, without one a single track
/// of each pair. A track end with no station has one.
pub(super) fn turn_places(net: &Network, n: u32) -> u8 {
    if net.node_platform[n as usize] == 0 { return 1; }
    let tracks = active_node_tracks(net, n);
    if net.node_upgrades[n as usize] & upgrades::TERMINAL != 0 { tracks } else { tracks.div_ceil(2).max(1) }
}

type PlatUse = (u32, u8, u32, u8, i32, f64, f64, bool);
type TermUse = (u32, u32, u8, f64, f64);
/// What a train of line `l`'s run `r` turning at node `n` holds (SPEC 6.2), approaching from `at`:
/// the pooled turnaround over `turn_places`, the same at a dead end and at a through station.
fn turn_at(net: &Network, n: u32, l: usize, r: u8, at: f64, term: &mut Vec<TermUse>) {
    let turn = net.line_turn[l] as f64;
    term.push((n, l as u32, r, at, upgrades::pooled_turn_s(turn, TERMINUS_MARGIN_S, turn_places(net, n) as f64)));
}

/// Platforms, termini and junction moves at the nodes `in_n` accepts, from the runs of `cand`
/// (which must hold every running line through those nodes).
fn build_nodes(net: &Network, lines: &[LineSvc], cand: &[u32], in_n: &dyn Fn(u32) -> bool, cap: &mut Capacity) {
    let mut plat: Vec<PlatUse> = vec![]; // node, track, line, run, stop, at, occ, reverses here
    let mut term: Vec<TermUse> = vec![]; // node, line, run, at, occ
    let mut moves: Vec<(u32, f64, f64, u32, u8, f64)> = vec![]; // node, pA, pB, line, run, at
    for &l in cand {
        let ls = &lines[l as usize];
        if !ls.running || ls.is_bus() {
            continue;
        }
        let l = l as usize;
        let dwell = net.line_dwell[l] as f64;
        for (r, run) in ls.runs.iter().enumerate().take(ls.live_runs()) {
            let k_last = run.nodes.len() - 1;
            for k in 1..k_last {
                let n = run.nodes[k];
                if !in_n(n) {
                    continue;
                }
                if net.turns_back(run.segs[k - 1], run.segs[k]) {
                    // T-797: reversing partway turns as a line's end does; it is no junction move and no
                    // through train. At a bare track end (B3) the stub is a one-track terminus.
                    let stand = match run.stop_idx.iter().position(|&s| s as usize == k) {
                        Some(j) => run.stop_s[j],
                        None => match run.ends.iter().find(|x| x.0 as usize == k) {
                            Some(x) => x.1,
                            None => continue,
                        },
                    };
                    let reach = net.node_platform[n as usize] as f64;
                    let at = stand - reach.max(ls.train_len) - HOLD_MARGIN_M;
                    turn_at(net, n, l, r as u8, at, &mut term);
                    continue;
                }
                if net.node_platform[n as usize] > 0 {
                    let prev = run.segs[k - 1];
                    let side_in = net.port_side(path_edge(prev), 1 - path_dir(prev)).unwrap_or(0);
                    let track = if active_node_tracks(net, n) == 1 { 0 } else { side_in };
                    // T-2231: passing tracks give each direction two platform tracks, a stop pooled
                    // over them, and a train that does not stop runs past on its own track
                    let passing = overtake::passing_at(net, n);
                    let (stop, at, occ) = match run.stop_idx.iter().position(|&s| s as usize == k) {
                        Some(j) => (j as i32, run.stop_s[j], if passing { (dwell + net.node_margin_s(n as u32)) / 2.0 } else { dwell + net.node_margin_s(n as u32) }),
                        None => (-1, run.node_s[k] - hold_back(lines, l), PASSING_S),
                    };
                    if !(passing && stop < 0) {
                        plat.push((n, track, l as u32, r as u8, stop, at, occ, false));
                    }
                }
                if net.node_ports[n as usize].n >= 3 && !net.node_flying[n as usize] {
                    let (pa, pb) = net.move_at(n, run.segs[k - 1], run.segs[k]);
                    moves.push((n, pa, pb, l as u32, r as u8, run.node_s[k] - hold_back(lines, l)));
                }
            }
            let n = run.nodes[k_last];
            if in_n(n) && ls.kind == LoopKind::Circle {
                // T-798: a circle's first stop is a stop on the way round, and the move from the
                // last entry into the first a junction move; the timing hold there is not counted
                let (p, q) = (run.segs[k_last - 1], run.segs[0]);
                let side_in = net.port_side(path_edge(p), 1 - path_dir(p)).unwrap_or(0);
                let tracks = active_node_tracks(net, n);
                let track = if tracks == 1 { 0 } else { side_in };
                let at = run.stop_s.last().unwrap() - hold_back(lines, l);
                plat.push((n, track, l as u32, r as u8, -1, at, dwell + net.node_margin_s(n as u32), false));
                if net.node_ports[n as usize].n >= 3 && !net.node_flying[n as usize] {
                    let (pa, pb) = net.move_at(n, p, q);
                    moves.push((n, pa, pb, l as u32, r as u8, run.node_s[k_last] - hold_back(lines, l)));
                }
            } else if in_n(n) {
                let reach = net.node_platform[n as usize] as f64;
                let at = run.stop_s.last().unwrap() - reach.max(ls.train_len) - HOLD_MARGIN_M;
                turn_at(net, n, l, r as u8, at, &mut term);
            }
        }
    }
    plat.sort_by(|a, b| (a.0, a.1, a.2, a.3, a.4).cmp(&(b.0, b.1, b.2, b.3, b.4)));
    let mut users: Vec<User>;
    let mut i = 0;
    while i < plat.len() {
        let end = plat[i..].partition_point(|x| (x.0, x.1) == (plat[i].0, plat[i].1)) + i;
        users = plat[i..end].iter().map(|x| User { line: x.2, run: x.3, owner: true, occ: x.6, at: x.5, stop: x.4 }).collect();
        let n = plat[i].0 as usize;
        let reverses = plat[i..end].iter().any(|x| x.7);
        push_res(cap, ResKind::Platform, net.node_x[n], net.node_y[n], n as u32, u32::MAX, &users, Tag::Platform { reverses });
        i = end;
    }
    term.sort_by(|a, b| (a.0, a.1, a.2).cmp(&(b.0, b.1, b.2)));
    let mut i = 0;
    while i < term.len() {
        let end = term[i..].partition_point(|x| x.0 == term[i].0) + i;
        users = term[i..end].iter().map(|x| User { line: x.1, run: x.2, owner: true, occ: x.4, at: x.3, stop: -1 }).collect();
        let n = term[i].0 as usize;
        push_res(cap, ResKind::Terminus, net.node_x[n], net.node_y[n], n as u32, u32::MAX, &users, Tag::Node);
        i = end;
    }
    moves.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.total_cmp(&b.1)).then(a.2.total_cmp(&b.2)).then((a.3, a.4).cmp(&(b.3, b.4))));
    let mut users = vec![];
    let mut i = 0;
    while i < moves.len() {
        let n = moves[i].0;
        let conflict = net.node_conflict_s(n);
        let end = moves[i..].partition_point(|x| x.0 == n) + i;
        let at_node = &moves[i..end];
        let mut distinct: Vec<(f64, f64)> = at_node.iter().map(|m| (m.1, m.2)).collect();
        distinct.dedup();
        for &(pa, pb) in &distinct {
            let crosses = |m: &(u32, f64, f64, u32, u8, f64)| (pa - m.1) * (pb - m.2) < 0.0;
            if !at_node.iter().any(crosses) {
                continue;
            }
            users.clear();
            for m in at_node {
                if (m.1, m.2) == (pa, pb) {
                    users.push(User { line: m.3, run: m.4, owner: true, occ: conflict, at: m.5, stop: -1 });
                } else if crosses(m) {
                    users.push(User { line: m.3, run: m.4, owner: false, occ: conflict, at: m.5, stop: -1 });
                }
            }
            push_res(cap, ResKind::Junction, net.node_x[n as usize], net.node_y[n as usize], n, u32::MAX, &users, Tag::Node);
        }
        i = end;
    }
}

/// Flat crossings `sel` accepts: each direction of each route, loaded by the other route. A train
/// route crossed by a tram route has none: the tram route's holds the tram passes at the tram's
/// figure and the train passes at the train's (T-3032). `cand` must hold every running line on
/// both edges of those crossings.
fn build_crossings(net: &Network, lines: &[LineSvc], cand: &[u32], sel: &dyn Fn(&Crossing) -> bool, cap: &mut Capacity) {
    let mut on_edge: Vec<(u32, u32, u32, u8, u32)> = vec![]; // edge, dir, line, run, seg index
    for &l in cand {
        let ls = &lines[l as usize];
        if !ls.running || ls.is_bus() {
            continue;
        }
        for (r, run) in ls.runs.iter().enumerate().take(ls.live_runs()) {
            for (i, &p) in run.segs.iter().enumerate() {
                on_edge.push((path_edge(p), path_dir(p), l, r as u8, i as u32));
            }
        }
    }
    on_edge.sort_unstable();
    let on = |e: u32| {
        let a = on_edge.partition_point(|o| o.0 < e);
        let b = on_edge.partition_point(|o| o.0 <= e);
        &on_edge[a..b]
    };
    let mut users = vec![];
    for c in &net.crossings {
        if !matches!(c.kind, CrossKind::Flat(_)) || !sel(c) {
            continue;
        }
        for (own, s_own, other) in [(c.e1, c.s1, c.e2), (c.e2, c.s2, c.e1)] {
            // T-3032: a tram yields to a train at a flat crossing. The train's resource is not loaded by
            // trams (none is built if only trams cross it); the tram's is loaded by every train pass at
            // the train's own figure, on top of its own passes at the tram's.
            let (fam_own, fam_other) = (net.edge_kind[own as usize].spec().family, net.edge_kind[other as usize].spec().family);
            if fam_own == 0 && fam_other != 0 {
                continue;
            }
            let yields = fam_own != 0 && fam_other == 0;
            let (conflict, other_conflict) = (net.edge_conflict_s(own), net.edge_conflict_s(other));
            // T-1805: a crossing no train runs over the other way holds nothing (blueprint, or
            // constructed track no line uses); the lines on its own track already share their section
            if on(other).is_empty() {
                continue;
            }
            let len = net.edge_len[own as usize];
            for dir in 0..2 {
                users.clear();
                for o in on(own).iter().filter(|o| o.1 == dir) {
                    let run = &lines[o.2 as usize].runs[o.3 as usize];
                    let at = run.seg_off[o.4 as usize] + if dir == 0 { s_own } else { len - s_own } - hold_back(lines, o.2 as usize);
                    users.push(User { line: o.2, run: o.3, owner: true, occ: conflict, at, stop: -1 });
                }
                if users.is_empty() {
                    continue;
                }
                for o in on(other) {
                    users.push(User { line: o.2, run: o.3, owner: false, occ: other_conflict, at: 0.0, stop: -1 });
                }
                push_res(cap, ResKind::Crossing, c.x, c.y, u32::MAX, own, &users, Tag::Crossing { other, yields });
            }
        }
    }
}

/// T-1966: the junction moves that cross at the same level (the pairs `build_nodes` loads each
/// other's junction resource with), over every line that is ok, running or not, for the map's
/// train guard: per pair, the node and each move's arriving and leaving path entries as its run
/// lists them.
pub fn move_conflicts(net: &Network, lines: &[LineSvc]) -> Vec<[u32; 5]> {
    let mut moves: Vec<(u32, f64, f64, u32, u32)> = vec![]; // node, pA, pB, in entry, out entry
    for ls in lines {
        if !ls.ok || ls.is_bus() {
            continue;
        }
        for run in ls.runs.iter().take(ls.live_runs()) {
            for k in 1..run.nodes.len().saturating_sub(1) {
                let n = run.nodes[k];
                let (p, q) = (run.segs[k - 1], run.segs[k]);
                if net.node_ports[n as usize].n < 3 || net.node_flying[n as usize] || net.turns_back(p, q) {
                    continue;
                }
                let (pa, pb) = net.move_at(n, p, q);
                moves.push((n, pa, pb, p, q));
            }
        }
    }
    moves.sort_by(|a, b| (a.0, a.3, a.4).cmp(&(b.0, b.3, b.4)));
    moves.dedup_by(|a, b| (a.0, a.3, a.4) == (b.0, b.3, b.4));
    let mut out = vec![];
    let mut i = 0;
    while i < moves.len() {
        let end = moves[i..].partition_point(|x| x.0 == moves[i].0) + i;
        for a in i..end {
            for b in a + 1..end {
                let (x, y) = (&moves[a], &moves[b]);
                if (x.1 - y.1) * (x.2 - y.2) < 0.0 {
                    out.push([x.0, x.3, x.4, y.3, y.4]);
                }
            }
        }
        i = end;
    }
    out
}

/// When a run leaves offset `s` (after standing there), from its free profile.
pub(super) fn leave_at(run: &super::service::Run, s: f64) -> f64 {
    let first = run.stop_s.first().copied().unwrap_or(0.0);
    let last = run.stop_s.last().copied().unwrap_or(run.len);
    profile::time_at(&run.free.phases, s.clamp(first, last))
}

/// When a run first reaches offset `s` (before standing there), from its free profile.
pub(super) fn reach_at(run: &super::service::Run, s: f64) -> f64 {
    let first = run.stop_s.first().copied().unwrap_or(0.0);
    let last = run.stop_s.last().copied().unwrap_or(run.len);
    let s = s.clamp(first, last);
    if s <= first { return 0.0; }
    profile::time_at(&run.free.phases, (s - 0.5).max(first))
}

/// One shared stretch of a faster line's run behind one slower run: path entries `i0..=i1` of the
/// faster run (from entry `j0` of the slower one), how much longer the slower one takes over it,
/// the spacing behind it, and where it starts and ends along the faster run.
pub(super) struct Stretch {
    pub i0: usize,
    pub i1: usize,
    pub j0: usize,
    pub line: u32,
    pub run: u8,
    longer: f64,
    spacing: f64,
    span: (f64, f64),
    /// T-2231: the slower run stops at a passing station inside: the overtake plan
    plan: Option<Plan>,
}

/// T-2419: on a following resource, a slower user's `stop` when it is the only slower line there and
/// stops at no passing station inside: the two are timetabled together, the faster trains placed
/// just ahead of slower ones (`follow_best`).
pub const FOLLOW_TIMED: i32 = 1;
/// T-2231: on a following resource, a slower user's `stop` when it stops at a passing station
/// inside the stretch: the loss is its overtake plan's (`Capacity::plans`).
pub const FOLLOW_PASSING: i32 = 2;
/// T-2231: the least a faster line's trains are placed after a slower line's they pass (s), so a
/// plan's offset of 0 still reads as a pair (`follow_targets`).
const PASS_OFFSET_MIN_S: f64 = 1.0;
/// T-2419: two lines that differ by less than this over a stretch (s) are taken as one pattern, not
/// timetabled one behind the other.
const FOLLOW_TIMED_MIN_S: f64 = 30.0;

/// T-1924 (notes/T-1900.md N1): the stretches each run of line `f` shares with slower lines `live`
/// accepts, in the same direction, by run. A shared stretch is a run of path entries both lines take
/// one after the other (same edge, same direction), cut where either turns back or they part; quad
/// track is two edges, so lines on its two pairs share nothing. `plans`: work out the overtake plan
/// of each stretch with a passing station inside (T-2231).
pub(super) fn stretches(net: &Network, lines: &[LineSvc], f: u32, live: &dyn Fn(&LineSvc) -> bool, plans: bool) -> [Vec<Stretch>; 2] {
    let mut out: [Vec<Stretch>; 2] = Default::default();
    let Some(fs) = lines.get(f as usize).filter(|l| live(l) && !l.is_bus()) else { return out };
    // the lines on its track, indexed by path entry
    let mut seen: Vec<u32> = vec![];
    for &p in &fs.runs[0].segs {
        for &l in net.lines_on_edge(path_edge(p)) {
            if l != f && !seen.contains(&l) && lines.get(l as usize).is_some_and(|x| live(x) && !x.is_bus()) {
                seen.push(l);
            }
        }
    }
    if seen.is_empty() {
        return out;
    }
    seen.sort_unstable();
    let mut occ: Vec<(u32, u32, u8, u32)> = vec![]; // path entry, line, run, entry index
    for &l in &seen {
        for (r, run) in lines[l as usize].runs.iter().enumerate().take(lines[l as usize].live_runs()) {
            occ.extend(run.segs.iter().enumerate().map(|(j, &p)| (p, l, r as u8, j as u32)));
        }
    }
    occ.sort_unstable();
    let mut chains: Vec<(u32, u8, usize, usize, usize)> = vec![]; // line, run, i0, j0, last i
    for (fr, run) in fs.runs.iter().enumerate().take(fs.live_runs()) {
        let stretches = &mut out[fr];
        chains.clear();
        let ns = run.segs.len();
        for i in 0..=ns {
            // close the chains that do not carry on to entry i (or past the end)
            let turn = i > 0 && i < ns && net.turns_back(run.segs[i - 1], run.segs[i]);
            let mut k = 0;
            while k < chains.len() {
                let (l, r, i0, j0, last) = chains[k];
                let next = if i < ns && !turn && last + 1 == i {
                    let j = j0 + (i - i0);
                    let s_run = &lines[l as usize].runs[r as usize];
                    j < s_run.segs.len() && s_run.segs[j] == run.segs[i] && !net.turns_back(s_run.segs[j - 1], s_run.segs[j])
                } else {
                    false
                };
                if next {
                    chains[k].4 = i;
                    k += 1;
                    continue;
                }
                chains.swap_remove(k);
                let s_run = &lines[l as usize].runs[r as usize];
                let j1 = j0 + (last - i0);
                let (fa, fb) = (run.seg_off[i0], run.node_s[last + 1]);
                let (sa, sb) = (s_run.seg_off[j0], s_run.node_s[j1 + 1]);
                // (where either run ends it arrives and goes no further: no wait for the slower one
                // to leave; the platform's own queue covers the rest)
                let s_end = if j1 + 1 == s_run.segs.len() || last + 1 == ns { reach_at(s_run, sb) } else { leave_at(s_run, sb) };
                let longer = (s_end - leave_at(s_run, sa)) - (reach_at(run, fb) - leave_at(run, fa));
                if longer < FOLLOW_MIN_S {
                    continue;
                }
                let spacing = net.edge_section_s(path_edge(s_run.segs[j0])) + profile::v_max(&s_run.free.phases, sa, sb) / SIGNAL_BRAKE;
                // T-2231: the slower one steps aside at passing stations inside, so the pair can part
                let plan = if plans {
                    let s = net.edge_section_s(path_edge(run.segs[i0])) + profile::v_max(&run.free.phases, fa, fb) / SIGNAL_BRAKE;
                    overtake::plan(net, (f, fr as u8), run, (l, r), s_run, i0, last, j0, s)
                } else {
                    None
                };
                stretches.push(Stretch { i0, i1: last, j0, line: l, run: r, longer, spacing, span: (fa, fb), plan });
            }
            if i == ns {
                break;
            }
            // start chains for the runs on entry i that no chain carries on
            let p = run.segs[i];
            let a = occ.partition_point(|o| o.0 < p);
            for o in occ[a..].iter().take_while(|o| o.0 == p) {
                let j = o.3 as usize;
                if chains.iter().any(|c| (c.0, c.1) == (o.1, o.2) && c.3 + (i - c.2) == j) {
                    continue;
                }
                chains.push((o.1, o.2, i, j, i));
            }
        }
    }
    out
}

/// T-1924: the following loss of each line in `owners` behind slower lines on track it shares with
/// them (`stretches`). Stretches of one run with different slower lines that overlap are one
/// resource, its one owner spanning them all. T-2231: a slower run that stops at a passing station
/// inside its stretch waits there for the faster one (`overtake`), a wait resource at each.
/// T-2419: a stretch behind one slower line and no other is timetabled with it (`follow_best`).
fn build_follow(net: &Network, lines: &[LineSvc], owners: &[u32], cap: &mut Capacity) {
    let mut users: Vec<User> = vec![];
    for &f in owners {
        for (fr, mut stretches) in stretches(net, lines, f, &|l| l.running, true).into_iter().enumerate() {
            let run = &lines[f as usize].runs[fr];
            // overlapping stretches are one resource
            stretches.sort_by(|a, b| (a.i0, a.i1, a.line, a.run).cmp(&(b.i0, b.i1, b.line, b.run)));
            let mut g = 0;
            while g < stretches.len() {
                let mut end = g + 1;
                let mut hi = stretches[g].i1;
                while end < stretches.len() && stretches[end].i0 <= hi {
                    hi = hi.max(stretches[end].i1);
                    end += 1;
                }
                let a = stretches[g..end].iter().map(|s| s.span.0).fold(f64::INFINITY, f64::min);
                let b = stretches[g..end].iter().map(|s| s.span.1).fold(f64::NEG_INFINITY, f64::max);
                // T-2419: behind one slower line the two are timetabled together; behind several, one
                // phase cannot suit them all, so they keep the random-meeting average, as do lines
                // that differ by too little to tell apart
                let one = stretches[g..end].iter().all(|s| s.line == stretches[g].line);
                users.clear();
                users.push(User { line: f, run: fr as u8, owner: true, occ: b, at: a, stop: -1 });
                users.extend(stretches[g..end].iter().map(|s| {
                    let stop = if s.plan.is_some() {
                        FOLLOW_PASSING
                    } else if one && s.longer >= FOLLOW_TIMED_MIN_S {
                        FOLLOW_TIMED
                    } else {
                        -1
                    };
                    User { line: s.line, run: s.run, owner: false, occ: s.longer, at: s.spacing, stop }
                }));
                let plans: Vec<Arc<Plan>> = stretches[g..end].iter_mut().filter_map(|s| s.plan.take()).map(Arc::new).collect();
                let mid = run.nodes[(stretches[g].i0 + hi + 1) / 2] as usize;
                push_res(cap, ResKind::Follow, net.node_x[mid], net.node_y[mid], u32::MAX, path_edge(run.segs[stretches[g].i0]), &users, Tag::Follow { line: f });
                let fi = cap.res.len() - 1;
                for p in &plans {
                    for (k, pt) in p.points.iter().enumerate().filter(|x| x.1.passing) {
                        let us = [
                            User { line: p.slow.0, run: p.slow.1, owner: true, occ: 0.0, at: pt.at, stop: pt.stop },
                            User { line: f, run: fr as u8, owner: false, occ: 0.0, at: 0.0, stop: -1 },
                        ];
                        let n = pt.node as usize;
                        push_res(cap, ResKind::Wait, net.node_x[n], net.node_y[n], pt.node, u32::MAX, &us, Tag::Wait { line: f, point: k as u16 });
                        *cap.plans.last_mut().unwrap() = vec![p.clone()];
                    }
                }
                cap.plans[fi] = plans;
                g = end;
            }
        }
    }
}

/// T-1924: a following resource's loss per train at a level, and the slower line that sets it
/// (u32::MAX for none). A train of the faster line meets the slower trains at a random place in
/// their headway `H`; over a stretch the slower one takes `D` longer on, it closes the gap less the
/// spacing `s` it must keep and then follows: on average `D - (H - s)/2` when `D` is at least
/// `H - s` (it always catches up), else `D² / 2(H - s)`. Slower lines are taken together from the
/// slowest: the loss behind those at least as slow as each is worked out with their trains
/// together, and the largest counts. Behind the one slower line it is timetabled with
/// (`FOLLOW_TIMED`, T-2419) it is `follow_best`'s; behind one that steps aside at passing stations
/// (`FOLLOW_PASSING`, T-2231) its overtake plan's (`plans`, one per such user in order); the largest
/// of those and the rest counts.
pub fn follow_loss(us: &[User], net: &Network, lev: usize, plans: &[Arc<Plan>]) -> (f64, u32) {
    let Some(f) = us.iter().find(|u| u.owner) else { return (0.0, u32::MAX) };
    if net.line_tph[f.line as usize][lev] <= 0.0 {
        return (0.0, u32::MAX);
    }
    let mut slow: Vec<(f64, f64, f64, u32, u8)> = vec![]; // longer, spacing, tph, line, run
    let (mut passed, mut by) = (0.0, u32::MAX);
    let mut plan = plans.iter();
    for u in us.iter().filter(|u| !u.owner) {
        if u.stop == FOLLOW_TIMED || u.stop == FOLLOW_PASSING {
            let loss = if u.stop == FOLLOW_TIMED { timed(net, f.line, u, lev).0 } else { plan.next().map_or(0.0, |p| p.outcome(net, lev).loss) };
            if loss > passed {
                (passed, by) = (loss, u.line);
            }
            continue;
        }
        let tph = net.line_tph[u.line as usize][lev] as f64;
        if tph <= 0.0 {
            continue;
        }
        match slow.iter_mut().find(|x| (x.3, x.4) == (u.line, u.run)) {
            Some(x) => {
                if u.occ > x.0 {
                    (x.0, x.1) = (u.occ, u.at);
                }
            }
            None => slow.push((u.occ, u.at, tph, u.line, u.run)),
        }
    }
    slow.sort_by(|a, b| b.0.total_cmp(&a.0).then((a.3, a.4).cmp(&(b.3, b.4))));
    let (mut best, mut who, mut tph) = (0.0, u32::MAX, 0.0);
    for &(d, s, t, l, _) in &slow {
        tph += t;
        let gap = (3600.0 / tph - s).max(0.0);
        let loss = if d >= gap { d - 0.5 * gap } else { d * d / (2.0 * gap) };
        if loss > best {
            (best, who) = (loss, l);
        }
    }
    if passed > best {
        (best, who) = (passed, by);
    }
    (best, who)
}

/// T-2419 (notes/T-2413.md): a faster line timetabled with the one slower line on a stretch with no
/// passing station, both at the headways they keep (`overtake::pattern_tph`): its loss per train and
/// how long after a slower train its first train of a slower headway reaches the stretch (s), or
/// (0, 0) when either does not run. See `follow_best`.
fn timed(net: &Network, fast: u32, slow: &User, lev: usize) -> (f64, f64) {
    let (tf, ts) = (overtake::pattern_tph(net, fast, lev) as f64, overtake::pattern_tph(net, slow.line, lev) as f64);
    if !(tf > 0.0 && ts > 0.0) {
        return (0.0, 0.0);
    }
    follow_best(slow.occ, slow.at, 3600.0 / ts, 3600.0 / tf)
}

/// T-2419: the faster trains' mean loss over a stretch the slower one takes `d` longer on, with
/// spacing `s`, slower headway `hs` and faster `hf`, when each slower headway's faster trains leave
/// as late as they can, the last `s` before the next slower train: that one closes the gap `hs - 2s`
/// and loses `d + 2s - hs` if more, each one before it a faster headway more. A fractional last
/// train counts by its share, so the loss runs smoothly with the rates (count schedules settle on
/// it). No passing: a faster line that gains more than the gap still ends up behind, as before.
/// Returns the loss and where the first of them leaves after a slower train.
pub fn follow_best(d: f64, s: f64, hs: f64, hf: f64) -> (f64, f64) {
    let a = d + 2.0 * s - hs;
    let k = hs / hf;
    let n = k.floor();
    let sum: f64 = (0..n as usize).map(|j| (a + j as f64 * hf).max(0.0)).sum::<f64>() + (k - n) * (a + n * hf).max(0.0);
    let at = if k <= 1.0 { hs - s } else { (hs - s - (k.ceil() - 1.0) * hf).max(s) };
    (sum / k, at)
}

impl Capacity {
    /// T-1924: the slower line a following resource's loss is behind at a level, or None.
    pub fn follow_leader(&self, net: &Network, r: usize, lev: usize) -> Option<u32> {
        if !matches!(self.tag[r], Tag::Follow { .. }) {
            return None;
        }
        Some(follow_loss(self.users_of(r), net, lev, &self.plans[r]).1).filter(|&l| l != u32::MAX)
    }

    /// T-2231: the overtake plans a following loss or a wait reads.
    pub fn plans_of(&self, r: usize) -> &[Arc<Plan>] {
        &self.plans[r]
    }

    /// T-2231: the faster line a wait at a passing station is for, or None.
    pub fn wait_for(&self, r: usize) -> Option<u32> {
        matches!(self.tag[r], Tag::Wait { .. }).then(|| self.users_of(r).iter().find(|u| !u.owner).map(|u| u.line)).flatten()
    }

    /// T-2231, T-2419: following pairs whose trains belong at a set place behind one another at a
    /// level (`spacing::place`): (faster line, run), (slower line, run), and how long after the
    /// slower one's trains the faster one's reach their shared track (s): just ahead of a slower
    /// one (`follow_best`), or with passing stations their overtake plan's offset.
    pub fn follow_targets(&self, net: &Network, lev: usize) -> Vec<((u32, u8), (u32, u8), f64)> {
        let mut out = vec![];
        for r in (0..self.res.len()).filter(|&r| matches!(self.tag[r], Tag::Follow { .. })) {
            let us = self.users_of(r);
            let Some(f) = us.iter().find(|u| u.owner) else { continue };
            let mut plan = self.plans[r].iter();
            for u in us.iter().filter(|u| !u.owner) {
                let at = match u.stop {
                    FOLLOW_TIMED => timed(net, f.line, u, lev).1,
                    FOLLOW_PASSING => plan.next().map_or(0.0, |p| p.outcome(net, lev).offset),
                    _ => continue,
                };
                out.push(((f.line, f.run), (u.line, u.run), at.max(PASS_OFFSET_MIN_S)));
            }
        }
        out
    }
}

/// Every resource and who uses it, from the lines' runs (paths, stops, free profiles).
pub fn build(net: &Network, lines: &[LineSvc]) -> Capacity {
    let mut cap = Capacity::default();
    let all: Vec<u32> = (0..lines.len() as u32).collect();
    build_sections(net, lines, &all, &|_| true, &mut cap);
    build_nodes(net, lines, &all, &|_| true, &mut cap);
    build_crossings(net, lines, &all, &|_| true, &mut cap);
    build_follow(net, lines, &all, &mut cap);
    cap
}

/// Utilisation and delay of one resource's users for each demand level.
fn solve_one(us: &[User], net: &Network, streams: &mut Vec<((u32, u8), f64)>, tag: Tag, plans: &[Arc<Plan>]) -> ([f64; 3], [f64; 3]) {
    let (mut rho_r, mut delay_r) = ([0.0; 3], [0.0; 3]);
    let continuous = us.iter().any(|u| net.line_trains[u.line as usize].is_some());
    if matches!(tag, Tag::Follow { .. } | Tag::Wait { .. }) {
        // T-1924: no utilisation, so no map dot; only the faster line's loss, or (T-2231) the
        // slower line's wait at a passing station
        for (lev, d) in delay_r.iter_mut().enumerate() {
            let w = match tag {
                Tag::Wait { point, .. } => plans.first().map_or(0.0, |p| p.outcome(net, lev).waits[point as usize]),
                _ => follow_loss(us, net, lev, plans).0,
            };
            *d = if continuous { w } else if w.round() < HOLD_MIN_S { 0.0 } else { w.round() };
        }
        return (rho_r, delay_r);
    }
    for lev in 0..LEVELS {
        let (mut load, mut trains) = (0.0, 0.0);
        streams.clear();
        for u in us {
            let tph = net.line_tph[u.line as usize][lev] as f64;
            if tph <= 0.0 {
                continue;
            }
            load += tph * u.occ;
            trains += tph;
            let key = (u.line, u.run);
            match streams.iter_mut().find(|s| s.0 == key) {
                Some(s) => s.1 = s.1.max(tph),
                None => streams.push((key, tph)),
            }
        }
        if trains <= 0.0 {
            continue;
        }
        let rho = load / 3600.0;
        let tot: f64 = streams.iter().map(|s| s.1).sum();
        let mix = 1.0 - streams.iter().map(|s| (s.1 / tot).powi(2)).sum::<f64>();
        let raw = kingman(rho, mix, load / trains);
        let w = if continuous { raw } else { raw.round() };
        rho_r[lev] = rho;
        delay_r[lev] = if !continuous && w < HOLD_MIN_S { 0.0 } else { w };
    }
    (rho_r, delay_r)
}

/// Utilisation and delay of every resource for each demand level.
pub fn solve(cap: &mut Capacity, net: &Network) {
    let mut streams = vec![];
    let n = cap.res.len();
    cap.rho.resize(n, [0.0; 3]);
    cap.delay.resize(n, [0.0; 3]);
    for r in 0..n {
        let (rho, delay) = solve_one(cap.users_of(r), net, &mut streams, cap.tag[r], &cap.plans[r]);
        cap.rho[r] = rho;
        cap.delay[r] = delay;
    }
}

impl Capacity {
    /// Bring the resources up to date after an edit, rebuilding only around what it touched.
    /// `edges` and `nodes`: what the edit touched (geometry, ports, tracks, platforms, flying
    /// flags, crossings) plus every edge and node on the old and new paths of the lines that were
    /// rebuilt; `lines` already holds the rebuilt lines. `rescheduled`: lines whose trains per
    /// hour changed. Returns, per line, whether it owns a resource that was rebuilt or re-solved
    /// to a different delay, i.e. whether its holds may have changed.
    pub fn update(&mut self, net: &Network, lines: &[LineSvc], edges: &[u32], nodes: &[u32], rescheduled: &[u32]) -> Vec<bool> {
        let (ne, nn) = (net.edge_count(), net.node_count());
        let mut de = vec![false; ne];
        for &e in edges {
            if let Some(x) = de.get_mut(e as usize) {
                *x = true;
            }
        }
        let mut dn = vec![false; nn];
        for &n in nodes {
            if let Some(x) = dn.get_mut(n as usize) {
                *x = true;
            }
        }
        for e in 0..ne {
            if de[e] {
                for n in [net.edge_a[e], net.edge_b[e]] {
                    if let Some(x) = dn.get_mut(n as usize) {
                        *x = true;
                    }
                }
            }
        }
        // Hot edges: touched, or ending at a touched node (a section through it may change).
        let mut he = de.clone();
        for n in 0..nn {
            if dn[n] {
                for &c in net.node_ports[n].ends() {
                    he[(c >> 1) as usize] = true;
                }
            }
        }
        let mut kk = vec![false; ne * 2];
        for e in 0..ne {
            if he[e] {
                kk[e * 2] = true;
                kk[e * 2 + 1] = true;
            }
        }
        let flag = |v: &[bool], i: u32| v.get(i as usize).copied().unwrap_or(true);
        // T-1924: a line's following losses come from every line on its track, so all of them go
        // and are built again when any edge it runs over is hot
        let mut follow = vec![false; lines.len()];
        for e in (0..ne).filter(|&e| he[e]) {
            for &l in net.lines_on_edge(e as u32) {
                if let Some(x) = follow.get_mut(l as usize) {
                    *x = true;
                }
            }
        }

        // ---- drop what the edit invalidates
        let mut affected = vec![false; lines.len()];
        let mut kill = vec![];
        for r in 0..self.res.len() {
            let dead = match self.tag[r] {
                Tag::Section { first, count } => {
                    let ks = &self.keys[first as usize..(first + count) as usize];
                    if ks.iter().any(|&k| flag(&he, k / 2)) {
                        for &k in ks {
                            if let Some(x) = kk.get_mut(k as usize) {
                                *x = true;
                            }
                        }
                        true
                    } else {
                        false
                    }
                }
                Tag::Node | Tag::Platform { .. } => flag(&dn, self.res[r].node),
                Tag::Crossing { other, .. } => flag(&de, self.res[r].edge) || flag(&de, other),
                Tag::Follow { line } | Tag::Wait { line, .. } => flag(&follow, line) || !lines.get(line as usize).is_some_and(|l| l.running),
            };
            if dead {
                kill.push(r);
                for u in self.users_of(r) {
                    if u.owner {
                        affected[u.line as usize] = true;
                    }
                }
            }
        }
        for &r in kill.iter().rev() {
            self.dead_users += self.res[r].count as usize;
            if let Tag::Section { count, .. } = self.tag[r] {
                self.dead_keys += count as usize;
            }
            self.res.swap_remove(r);
            self.tag.swap_remove(r);
            self.plans.swap_remove(r);
            self.rho.swap_remove(r);
            self.delay.swap_remove(r);
        }

        // ---- build them again
        let first_new = self.res.len();
        let mut mark = vec![false; lines.len()];
        let mut cand = |edges: &mut dyn Iterator<Item = u32>| -> Vec<u32> {
            mark.iter_mut().for_each(|m| *m = false);
            let mut v = vec![];
            for e in edges {
                for &l in net.lines_on_edge(e) {
                    if !mark[l as usize] {
                        mark[l as usize] = true;
                        v.push(l);
                    }
                }
            }
            v.sort_unstable();
            v
        };
        let c = cand(&mut (0..ne as u32).filter(|&e| kk[e as usize * 2] || kk[e as usize * 2 + 1]));
        build_sections(net, lines, &c, &|k| kk.get(k as usize).copied().unwrap_or(false), self);
        let c = cand(&mut (0..nn as u32).filter(|&n| dn[n as usize]).flat_map(|n| net.node_ports[n as usize].ends().iter().map(|&c| c >> 1)));
        build_nodes(net, lines, &c, &|n| flag(&dn, n), self);
        let sel = |x: &Crossing| flag(&de, x.e1) || flag(&de, x.e2);
        let c = cand(&mut net.crossings.iter().filter(|x| matches!(x.kind, CrossKind::Flat(_)) && sel(x)).flat_map(|x| [x.e1, x.e2]));
        build_crossings(net, lines, &c, &sel, self);
        let owners: Vec<u32> = (0..lines.len() as u32).filter(|&l| follow[l as usize]).collect();
        build_follow(net, lines, &owners, self);

        // ---- solve the new ones, and the old ones a rescheduled line uses
        let mut streams = vec![];
        for r in first_new..self.res.len() {
            let (rho, delay) = solve_one(self.users_of(r), net, &mut streams, self.tag[r], &self.plans[r]);
            self.rho[r] = rho;
            self.delay[r] = delay;
            for u in self.users_of(r) {
                if u.owner {
                    affected[u.line as usize] = true;
                }
            }
        }
        if !rescheduled.is_empty() {
            let mut rs = vec![false; lines.len()];
            for &l in rescheduled {
                if let Some(x) = rs.get_mut(l as usize) {
                    *x = true;
                }
            }
            for r in 0..first_new {
                if !self.users_of(r).iter().any(|u| rs[u.line as usize]) {
                    continue;
                }
                let (rho, delay) = solve_one(self.users_of(r), net, &mut streams, self.tag[r], &self.plans[r]);
                self.rho[r] = rho;
                if delay != self.delay[r] {
                    self.delay[r] = delay;
                    for u in self.users_of(r) {
                        if u.owner {
                            affected[u.line as usize] = true;
                        }
                    }
                }
            }
        }
        self.compact();
        affected
    }

    /// Drop dead users and keys once they are more than half of their vectors.
    fn compact(&mut self) {
        if self.dead_users * 2 > self.users.len() {
            let mut users = Vec::with_capacity(self.users.len() - self.dead_users);
            for r in &mut self.res {
                let f = users.len() as u32;
                users.extend_from_slice(&self.users[r.first as usize..(r.first + r.count) as usize]);
                r.first = f;
            }
            self.users = users;
            self.dead_users = 0;
        }
        if self.dead_keys * 2 > self.keys.len() {
            let mut keys = Vec::with_capacity(self.keys.len() - self.dead_keys);
            for t in &mut self.tag {
                if let Tag::Section { first, count } = t {
                    let f = keys.len() as u32;
                    keys.extend_from_slice(&self.keys[*first as usize..(*first + *count) as usize]);
                    *first = f;
                }
            }
            self.keys = keys;
            self.dead_keys = 0;
        }
    }

    /// The resources as a sorted list of (resource, users, rho, delay), for comparing two passes
    /// that built them in different orders (tests and checks).
    pub fn canonical(&self) -> Vec<(Resource, Vec<User>, [f64; 3], [f64; 3])> {
        let mut v: Vec<_> = (0..self.res.len())
            .map(|r| {
                let mut res = self.res[r].clone();
                res.first = 0;
                (res, self.users_of(r).to_vec(), self.rho[r], self.delay[r])
            })
            .collect();
        let key = |x: &(Resource, Vec<User>, [f64; 3], [f64; 3])| {
            let us: Vec<_> = x.1.iter().map(|u| (u.line, u.run, u.owner, u.stop, u.at.to_bits(), u.occ.to_bits())).collect();
            (x.0.kind as u8, x.0.node, x.0.edge, x.0.x.to_bits(), x.0.y.to_bits(), us)
        };
        v.sort_by(|a, b| key(a).cmp(&key(b)));
        v
    }
}

/// Holds per (line, run, level): track holds and extra dwell per stop, from the solved delays.
/// With `want`, only for the lines it marks (the others come back empty).
pub type LineHolds = [[(Vec<Hold>, Vec<f64>); 3]; 2];

pub fn holds(cap: &Capacity, lines: &[LineSvc]) -> Vec<LineHolds> {
    holds_for(cap, lines, None)
}

pub fn holds_for(cap: &Capacity, lines: &[LineSvc], want: Option<&[bool]>) -> Vec<LineHolds> {
    let wanted = |l: usize| want.map_or(true, |w| w.get(l).copied().unwrap_or(false));
    let mut out: Vec<LineHolds> = lines
        .iter()
        .enumerate()
        .map(|(l, ls)| {
            std::array::from_fn(|r| std::array::from_fn(|_| (vec![], if wanted(l) { vec![0.0; ls.runs[r].stop_s.len()] } else { vec![] })))
        })
        .collect();
    for r in 0..cap.res.len() {
        if let Tag::Follow { line } = cap.tag[r] {
            // T-2018: the loss as slower running over the whole stretch, never a stand between
            // stops (the app's trainGuard keeps it behind the train it follows)
            if !wanted(line as usize) {
                continue;
            }
            for u in cap.users_of(r).iter().filter(|u| u.owner) {
                for lev in 0..LEVELS {
                    let d = cap.delay[r][lev];
                    if d > 0.0 {
                        out[u.line as usize][u.run as usize][lev].0.push(Hold { s: u.at, delay: d, span: (u.occ - u.at).max(0.0) });
                    }
                }
            }
            continue;
        }
        for u in cap.users_of(r) {
            if !u.owner || !wanted(u.line as usize) {
                continue;
            }
            for lev in 0..LEVELS {
                let d = cap.delay[r][lev];
                if d <= 0.0 {
                    continue;
                }
                let slot = &mut out[u.line as usize][u.run as usize][lev];
                if u.stop >= 0 {
                    slot.1[u.stop as usize] += d;
                } else {
                    slot.0.push(Hold { s: u.at, delay: d, span: 0.0 });
                }
            }
        }
    }
    for l in &mut out {
        for r in l.iter_mut() {
            for lev in r.iter_mut() {
                lev.0.sort_by(|a, b| a.s.total_cmp(&b.s).then(a.delay.total_cmp(&b.delay)));
            }
        }
    }
    out
}

#[cfg(test)]
mod tests;
