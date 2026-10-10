//! Capacity and delay (SPEC 6.2, notes/T-008.md 5): resources, utilisation per demand level from
//! the scheduled trains per hour, Kingman's delay, and holds for the run-time profiles. One pass,
//! no iteration: delays change run times and fleet, never trains per hour.
//!
//! Incremental (T-050): after an edit, `Capacity::update` throws away only the resources around
//! what it touched (sections reaching a touched edge or node, platforms, termini and junction
//! moves at touched nodes, crossings of touched edges), builds those again with the same code as
//! the global pass, and re-solves them and the resources a rescheduled line uses. The result is
//! the global pass's, in another order (tests below check it on random edit sequences).

use super::geom;
use super::net::{path_dir, path_edge, CrossKind, Crossing, Network};
use super::params::*;
use super::profile::{self, Hold};
use super::service::LineSvc;

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
}

/// A line direction using a resource. Owners get the delay; the others only load it.
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
    /// A platform, terminus or junction move at `Resource::node`.
    Node,
    /// A flat crossing of `Resource::edge` with `other`.
    Crossing { other: u32 },
}

/// Every resource with its users, utilisation and delay. Resource indices are valid until the
/// next update; the incremental pass does not keep their order.
#[derive(Clone, Debug, Default)]
pub struct Capacity {
    pub res: Vec<Resource>,
    pub users: Vec<User>,
    /// Utilisation per demand level (high, medium, low).
    pub rho: Vec<[f64; 3]>,
    /// Delay per train per level, whole seconds, 0 under 2 s.
    pub delay: Vec<[f64; 3]>,
    tag: Vec<Tag>,
    keys: Vec<u32>,
    /// Entries of `users` and `keys` no resource points at any more; compacted past half.
    dead_users: usize,
    dead_keys: usize,
}

impl Capacity {
    pub fn users_of(&self, r: usize) -> &[User] {
        let x = &self.res[r];
        &self.users[x.first as usize..(x.first + x.count) as usize]
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

/// Lateral track position of an edge end at a node, for the crossing rule
/// `(pA1 - pA2)(pB1 - pB2) < 0`: ports right to left, each double-track port's two tracks by
/// running side, a single-track port in between.
fn track_pos(net: &Network, node: u32, e: u32, which: u32, plus_h: bool) -> f64 {
    let p = &net.node_ports[node as usize];
    let rank = p.find(e, which).map_or(0, |k| p.rank[k]) as f64 * 2.0;
    if net.edge_tracks[e as usize] == 1 {
        rank + 0.5
    } else if plus_h == net.right_hand {
        rank
    } else {
        rank + 1.0
    }
}

/// The directed track a path entry runs on (shared by both directions on single track).
fn key_of(net: &Network, p: u32) -> u32 {
    let e = path_edge(p);
    if net.edge_tracks[e as usize] == 1 {
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
fn build_sections(net: &Network, lines: &[LineSvc], cand: &[u32], in_k: &dyn Fn(u32) -> bool, cap: &mut Capacity) {
    // (key, line, run, seg index)
    let mut occ: Vec<(u32, u32, u8, u32)> = vec![];
    for &l in cand {
        let ls = &lines[l as usize];
        if !ls.running {
            continue;
        }
        for r in 0..2 {
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
        if !ls.running {
            continue;
        }
        for run in &ls.runs {
            for i in 0..run.segs.len().saturating_sub(1) {
                let node = run.nodes[i + 1] as usize;
                if net.node_ports[node].n != 2 {
                    continue;
                }
                let (p, q) = (run.segs[i], run.segs[i + 1]);
                let (Some(a), Some(b)) = (kidx(key_of(net, p)), kidx(key_of(net, q))) else { continue };
                let (te, tf) = (net.edge_tracks[path_edge(p) as usize], net.edge_tracks[path_edge(q) as usize]);
                let join = if te == 2 && tf == 2 { sig[a as usize] == sig[b as usize] } else { te == 1 && tf == 1 && net.node_platform[node] == 0 };
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
    while g < grouped.len() {
        let root = grouped[g].0;
        let end = grouped[g..].partition_point(|x| x.0 == root) + g;
        let first_edge = keys[root as usize] / 2;
        let single = net.edge_tracks[first_edge as usize] == 1;
        users.clear();
        let mut u = g;
        while u < end {
            let (l, r) = (grouped[u].1, grouped[u].2);
            let ue = grouped[u..end].partition_point(|x| (x.1, x.2) == (l, r)) + u;
            let run = &lines[l as usize].runs[r as usize];
            let (mut entry, mut exit) = (f64::INFINITY, f64::NEG_INFINITY);
            for x in &grouped[u..ue] {
                let i = x.3 as usize;
                entry = entry.min(run.seg_off[i]);
                exit = exit.max(run.node_s[i + 1]);
            }
            let ph = &run.free.phases;
            let occ_s = if single {
                profile::time_at(ph, exit) - profile::time_at(ph, entry) + SINGLE_TRACK_MARGIN_S
            } else {
                SECTION_BASE_S + profile::v_max(ph, entry, exit) / SIGNAL_BRAKE
            };
            users.push(User { line: l, run: r, owner: true, occ: occ_s, at: entry - hold_back(lines, l as usize), stop: -1 });
            u = ue;
        }
        let me = members[m..].partition_point(|x| x.0 == root) + m;
        let kf = cap.keys.len() as u32;
        cap.keys.extend(members[m..me].iter().map(|x| x.1));
        m = me;
        let pcs = net.edge_pieces(first_edge);
        let (x, y, _) = geom::pos_at(pcs, net.edge_len[first_edge as usize] / 2.0);
        let kind = if single { ResKind::SingleTrack } else { ResKind::Section };
        push_res(cap, kind, x, y, u32::MAX, first_edge, &users, Tag::Section { first: kf, count: (cap.keys.len() as u32) - kf });
        g = end;
    }
}

/// Platforms, termini and junction moves at the nodes `in_n` accepts, from the runs of `cand`
/// (which must hold every running line through those nodes).
fn build_nodes(net: &Network, lines: &[LineSvc], cand: &[u32], in_n: &dyn Fn(u32) -> bool, cap: &mut Capacity) {
    let mut plat: Vec<(u32, u8, u32, u8, i32, f64, f64)> = vec![]; // node, track, line, run, stop, at, occ
    let mut term: Vec<(u32, u32, u8, f64, f64)> = vec![]; // node, line, run, at, occ
    let mut moves: Vec<(u32, f64, f64, u32, u8, f64)> = vec![]; // node, pA, pB, line, run, at
    for &l in cand {
        let ls = &lines[l as usize];
        if !ls.running {
            continue;
        }
        let l = l as usize;
        let dwell = net.line_dwell[l] as f64;
        for (r, run) in ls.runs.iter().enumerate() {
            let k_last = run.nodes.len() - 1;
            for k in 1..k_last {
                let n = run.nodes[k];
                if !in_n(n) {
                    continue;
                }
                if net.node_platform[n as usize] > 0 {
                    let prev = run.segs[k - 1];
                    let side_in = net.port_side(path_edge(prev), 1 - path_dir(prev)).unwrap_or(0);
                    let track = if net.node_tracks(n) == 1 { 0 } else { side_in };
                    match run.stop_idx.iter().position(|&s| s as usize == k) {
                        Some(j) => plat.push((n, track, l as u32, r as u8, j as i32, run.stop_s[j], dwell + PLATFORM_MARGIN_S)),
                        None => plat.push((n, track, l as u32, r as u8, -1, run.node_s[k] - hold_back(lines, l), PASSING_S)),
                    }
                }
                if net.node_ports[n as usize].n >= 3 && !net.node_flying[n as usize] {
                    let (p, q) = (run.segs[k - 1], run.segs[k]);
                    let (ein, win) = (path_edge(p), 1 - path_dir(p));
                    let (eout, wout) = (path_edge(q), path_dir(q));
                    let side_in = net.port_side(ein, win).unwrap_or(0);
                    let (pa, pb) = if side_in == 0 {
                        (track_pos(net, n, ein, win, true), track_pos(net, n, eout, wout, true))
                    } else {
                        (track_pos(net, n, eout, wout, false), track_pos(net, n, ein, win, false))
                    };
                    moves.push((n, pa, pb, l as u32, r as u8, run.node_s[k] - hold_back(lines, l)));
                }
            }
            let n = run.nodes[k_last];
            if in_n(n) {
                let reach = net.node_platform[n as usize] as f64;
                let at = run.stop_s.last().unwrap() - reach.max(ls.train_len) - HOLD_MARGIN_M;
                let occ_s = (net.line_turn[l] as f64 + TERMINUS_MARGIN_S) / net.node_tracks(n) as f64;
                term.push((n, l as u32, r as u8, at, occ_s));
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
        push_res(cap, ResKind::Platform, net.node_x[n], net.node_y[n], n as u32, u32::MAX, &users, Tag::Node);
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
                    users.push(User { line: m.3, run: m.4, owner: true, occ: CONFLICT_S, at: m.5, stop: -1 });
                } else if crosses(m) {
                    users.push(User { line: m.3, run: m.4, owner: false, occ: CONFLICT_S, at: m.5, stop: -1 });
                }
            }
            push_res(cap, ResKind::Junction, net.node_x[n as usize], net.node_y[n as usize], n, u32::MAX, &users, Tag::Node);
        }
        i = end;
    }
}

/// Flat crossings `sel` accepts: each direction of each route, loaded by the other route. `cand`
/// must hold every running line on both edges of those crossings.
fn build_crossings(net: &Network, lines: &[LineSvc], cand: &[u32], sel: &dyn Fn(&Crossing) -> bool, cap: &mut Capacity) {
    let mut on_edge: Vec<(u32, u32, u32, u8, u32)> = vec![]; // edge, dir, line, run, seg index
    for &l in cand {
        let ls = &lines[l as usize];
        if !ls.running {
            continue;
        }
        for (r, run) in ls.runs.iter().enumerate() {
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
            let len = net.edge_len[own as usize];
            for dir in 0..2 {
                users.clear();
                for o in on(own).iter().filter(|o| o.1 == dir) {
                    let run = &lines[o.2 as usize].runs[o.3 as usize];
                    let at = run.seg_off[o.4 as usize] + if dir == 0 { s_own } else { len - s_own } - hold_back(lines, o.2 as usize);
                    users.push(User { line: o.2, run: o.3, owner: true, occ: CONFLICT_S, at, stop: -1 });
                }
                if users.is_empty() {
                    continue;
                }
                for o in on(other) {
                    users.push(User { line: o.2, run: o.3, owner: false, occ: CONFLICT_S, at: 0.0, stop: -1 });
                }
                push_res(cap, ResKind::Crossing, c.x, c.y, u32::MAX, own, &users, Tag::Crossing { other });
            }
        }
    }
}

/// Every resource and who uses it, from the lines' runs (paths, stops, free profiles).
pub fn build(net: &Network, lines: &[LineSvc]) -> Capacity {
    let mut cap = Capacity::default();
    let all: Vec<u32> = (0..lines.len() as u32).collect();
    build_sections(net, lines, &all, &|_| true, &mut cap);
    build_nodes(net, lines, &all, &|_| true, &mut cap);
    build_crossings(net, lines, &all, &|_| true, &mut cap);
    cap
}

/// Utilisation and delay of one resource's users for each demand level.
fn solve_one(us: &[User], net: &Network, streams: &mut Vec<((u32, u8), f64)>) -> ([f64; 3], [f64; 3]) {
    let (mut rho_r, mut delay_r) = ([0.0; 3], [0.0; 3]);
    let continuous = us.iter().any(|u| net.line_trains[u.line as usize].is_some());
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
            match streams.iter_mut().find(|s| s.0 == (u.line, u.run)) {
                Some(s) => s.1 = s.1.max(tph),
                None => streams.push(((u.line, u.run), tph)),
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
        let (rho, delay) = solve_one(cap.users_of(r), net, &mut streams);
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
                Tag::Node => flag(&dn, self.res[r].node),
                Tag::Crossing { other } => flag(&de, self.res[r].edge) || flag(&de, other),
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

        // ---- solve the new ones, and the old ones a rescheduled line uses
        let mut streams = vec![];
        for r in first_new..self.res.len() {
            let (rho, delay) = solve_one(self.users_of(r), net, &mut streams);
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
                let (rho, delay) = solve_one(self.users_of(r), net, &mut streams);
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
                    slot.0.push(Hold { s: u.at, delay: d });
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
mod tests {
    use super::*;

    #[test]
    fn spec_delay_table() {
        // SPEC 6.2: a 90 s junction shared by two equal lines (mix 0.5).
        let w = |rho| kingman(rho, 0.5, 90.0);
        assert!((w(0.75) - 27.0).abs() < 1e-9);
        assert!((w(0.9) - 81.0).abs() < 1e-9);
        assert!((w(1.0) - 171.0).abs() < 1e-9);
        assert!((w(1.1) - 425.0).abs() < 1.0);
        // One line alone: nothing until over capacity.
        assert_eq!(kingman(0.95, 0.0, 90.0), 0.0);
        assert!(kingman(1.2, 0.0, 90.0) > 0.0);
        // T-008's 80 km/h section, two lines.
        let s = 90.0 + (80.0 / 3.6) / 0.6;
        assert!((kingman(0.75, 0.5, s) - 38.0).abs() < 1.0);
    }

    use super::super::bench::synth;
    use super::super::net::StationData;
    use super::super::service::Service;
    use super::super::world::{Op, TrackWorld};

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn below(&mut self, n: usize) -> usize {
            (self.next() % n.max(1) as u64) as usize
        }
        fn unit(&mut self) -> f64 {
            (self.next() >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    /// A random edit of the kinds that move resources: PIs, schedules, splits, deletes, flying
    /// junctions, single or double track, platforms, stops, removing a line.
    fn random_op(w: &mut TrackWorld, rng: &mut Rng) -> Option<Op> {
        let n = &w.net;
        let edges: Vec<u32> = (0..n.edge_count() as u32).filter(|&e| n.edge_ok(e)).collect();
        let lines: Vec<u32> = (0..n.line_count() as u32).filter(|&l| n.line_ok(l)).collect();
        let nodes: Vec<u32> = (0..n.node_count() as u32).filter(|&x| n.node_ok(x)).collect();
        // Busy edges are where the interesting changes are: prefer edges with lines.
        let used: Vec<u32> = edges.iter().copied().filter(|&e| !n.lines_on_edge(e).is_empty()).collect();
        let pick_edge = |rng: &mut Rng| if !used.is_empty() && rng.below(4) > 0 { used[rng.below(used.len())] } else { edges[rng.below(edges.len())] };
        Some(match rng.below(9) {
            0 => {
                let e = pick_edge(rng);
                let mut d = n.edge_data(e)?;
                if d.pis.is_empty() {
                    return None;
                }
                let k = rng.below(d.pis.len());
                d.pis[k].x = quant(d.pis[k].x + rng.unit() * 40.0 - 20.0);
                Op::Edge { id: e, data: Some(d) }
            }
            1 | 2 => {
                let l = *lines.get(rng.below(lines.len()))?;
                let t = n.line_tph[l as usize];
                let tph = [(t[0] + rng.below(9) as f32 - 4.0).clamp(0.0, 30.0), (t[1] + rng.below(5) as f32 - 2.0).clamp(0.0, 30.0), t[2]];
                Op::Schedule { id: l, tph }
            }
            3 => {
                let e = pick_edge(rng);
                let s = n.edge_len[e as usize] * (0.2 + 0.6 * rng.unit());
                let node = w.net.alloc_node();
                let new_edge = w.net.alloc_edge();
                Op::Split { edge: e, s, node, new_edge }
            }
            4 => Op::DeleteEdge { id: pick_edge(rng) },
            5 => {
                let j: Vec<u32> = nodes.iter().copied().filter(|&x| n.node_ports[x as usize].n >= 3).collect();
                let x = *j.get(rng.below(j.len()))?;
                let mut d = n.node_data(x)?;
                d.flying = !d.flying;
                Op::Node { id: x, data: Some(d) }
            }
            6 => {
                let e = pick_edge(rng);
                let mut d = n.edge_data(e)?;
                d.tracks = 3 - d.tracks;
                Op::Edge { id: e, data: Some(d) }
            }
            7 => {
                let st: Vec<u32> = nodes.iter().copied().filter(|&x| n.node_platform[x as usize] > 0).collect();
                let x = *st.get(rng.below(st.len()))?;
                let d = n.station_data(x)?;
                Op::Station { node: x, data: Some(StationData { platform: if d.platform > 120 { d.platform - 40 } else { d.platform + 40 }, ..d }) }
            }
            _ => {
                let l = *lines.get(rng.below(lines.len()))?;
                let mut d = n.line_data(l)?;
                if rng.below(3) == 0 || d.stops.len() <= 3 {
                    Op::Line { id: l, data: None }
                } else {
                    // Drop a middle stop; the path stays (it still passes the station).
                    d.stops.remove(1 + rng.below(d.stops.len() - 2));
                    Op::Line { id: l, data: Some(d) }
                }
            }
        })
    }

    /// The incremental state against everything recomputed from the network.
    fn same_as_global(w: &TrackWorld, what: &str) {
        let mut g = Service::default();
        g.update_all(&w.net);
        let (a, b) = (w.svc.cap.canonical(), g.cap.canonical());
        assert_eq!(a.len(), b.len(), "{what}: resource count");
        for (x, y) in a.iter().zip(&b) {
            assert_eq!(x, y, "{what}: resource");
        }
        for (l, (x, y)) in w.svc.lines.iter().zip(&g.lines).enumerate() {
            assert_eq!((x.ok, x.running), (y.ok, y.running), "{what}: line {l} state");
            if !x.ok {
                continue;
            }
            for r in 0..2 {
                assert_eq!(x.runs[r].holds, y.runs[r].holds, "{what}: line {l} run {r} holds");
                assert_eq!(x.runs[r].stop_holds, y.runs[r].stop_holds, "{what}: line {l} run {r} stop holds");
                assert_eq!(x.runs[r].prof, y.runs[r].prof, "{what}: line {l} run {r} profiles");
            }
            assert_eq!((x.round_trip, x.trains, x.delay), (y.round_trip, y.trains, y.delay), "{what}: line {l} round trip");
        }
    }

    /// T-050: random edit sequences (with undo and redo) on synthetic networks; after every
    /// step the incremental capacity pass must equal the global one.
    #[test]
    fn incremental_equals_global() {
        let mut applied = 0;
        for seed in 1..=6u64 {
            let mut w = TrackWorld::new(synth(250.0, 70, 12, seed));
            let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
            same_as_global(&w, "start");
            for step in 0..80 {
                let what = format!("seed {seed} step {step}");
                match rng.below(10) {
                    0 => {
                        let _ = w.undo();
                    }
                    1 => {
                        let _ = w.redo();
                    }
                    _ => {
                        let Some(op) = random_op(&mut w, &mut rng) else { continue };
                        if w.apply(op).is_ok() {
                            applied += 1;
                        }
                    }
                }
                same_as_global(&w, &what);
            }
        }
        assert!(applied > 200, "too few edits applied: {applied}");
    }
}
