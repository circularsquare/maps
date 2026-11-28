//! T-1447: a pick of track and stations copied out of one game and into another as blueprint (notes/T-1447.md).
//! A clip is a network the save writer writes; both ways go through the ordinary checked edit, which leaves out
//! what it refuses by name and tries the rest again.

use super::*;
use crate::track::net::{path_dir, path_edge};
use crate::track::twin::Twin;
use std::collections::{HashMap, HashSet};

/// Most rounds of leaving out what the checks refuse.
const ROUNDS: usize = 32;

/// Source records to add: track pieces, stations (their nodes), lines, links and kept-apart pairs.
#[derive(Clone, Default)]
struct Want {
    edges: Vec<u32>,
    stations: Vec<u32>,
    lines: Vec<u32>,
    links: Vec<[u32; 2]>,
    apart: Vec<[u32; 2]>,
}

/// Source id to new id, handed out once and kept across rounds.
#[derive(Default)]
struct Ids {
    node: HashMap<u32, u32>,
    edge: HashMap<u32, u32>,
    line: HashMap<u32, u32>,
}

/// The lines wholly on `edges` and stopping only at `stations`: rail and tram lines, never a quad line's hidden twin.
fn lines_within(src: &Network, edges: &HashSet<u32>, stations: &HashSet<u32>) -> Vec<u32> {
    (0..src.line_count() as u32)
        .filter(|&l| {
            let Some(d) = src.line_data(l) else { return false };
            d.bus.is_none() && d.ferry.is_none() && !matches!(d.twin, Twin::Of(_)) && d.stops.len() >= 2 && !d.path.is_empty()
                && d.stops.iter().all(|s| stations.contains(s)) && d.path.iter().all(|&p| edges.contains(&path_edge(p)))
        })
        .collect()
}

fn want_of(src: &Network, edges: &[u32], stations: &[u32]) -> Want {
    let edges: Vec<u32> = edges.iter().copied().filter(|&e| src.edge_ok(e)).collect();
    let ends: HashSet<u32> = edges.iter().flat_map(|&e| [src.edge_a[e as usize], src.edge_b[e as usize]]).collect();
    // a station on track goes with some of its track; a port or a BRT station stands on none
    let stations: Vec<u32> = stations.iter().copied().filter(|&n| src.station_data(n).is_some_and(|s| s.berth.is_some() || s.road.is_some() || ends.contains(&n))).collect();
    let (es, ss): (HashSet<u32>, HashSet<u32>) = (edges.iter().copied().collect(), stations.iter().copied().collect());
    let both = |p: &[u32; 2]| ss.contains(&p[0]) && ss.contains(&p[1]);
    Want { lines: lines_within(src, &es, &ss), links: src.links_live().filter(both).collect(), apart: src.apart_live().filter(both).collect(), edges, stations }
}

impl Want {
    fn drop_edge(&mut self, src: &Network, e: u32) -> bool {
        let Some(i) = self.edges.iter().position(|&x| x == e) else { return false };
        self.edges.swap_remove(i);
        self.lines.retain(|&l| !src.line_path(l).iter().any(|&p| path_edge(p) == e));
        true
    }
    fn drop_station(&mut self, src: &Network, n: u32) -> bool {
        let Some(i) = self.stations.iter().position(|&x| x == n) else { return false };
        self.stations.swap_remove(i);
        self.lines.retain(|&l| !src.line_stops(l).contains(&n));
        self.links.retain(|p| !p.contains(&n));
        self.apart.retain(|p| !p.contains(&n));
        true
    }
    fn is_empty(&self) -> bool {
        self.edges.is_empty() && self.stations.is_empty()
    }
}

/// The edges, nodes and lines an issue names.
fn named(i: &Issue) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    use Issue::*;
    match *i {
        Geom { edge, .. } | GroundOverWater { edge, .. } | WaterTooLong { edge } | NoSuchEdge { edge } | StationTrack { edge } | EdgeLoop { edge }
        | Tracks { edge } | BuswayTrack { edge } | SplitPoint { edge } | SplitRamp { edge } | Crossover { edge } | NoParallel { edge }
        | Alongside { edge } | HighSpeed { edge } | NoSingle { edge, .. } | NodeHeading { edge, .. } => (vec![edge], vec![], vec![]),
        Crossing { e1, e2, .. } => (vec![e1, e2], vec![], vec![]),
        NoSuchNode { node } | TooManyPorts { node } | NodeOneSided { node } | NodeLevel { node } | NodeInUse { node } | StationPorts { node }
        | StationFamily { node } | BuswayJoin { node } | BuswayStation { node } | StationJunction { node } | PlatformLength { node }
        | PlatformTooLong { node } | PlatformStation { node } | PlatformJunction { node } | PlatformJoint { node } | PlatformNotLevel { node }
        | PlatformCrossing { node } | StationGroundOverWater { node } | OneTrackHere { node } | BranchNoRoom { node } | NotAtPort { node }
        | PortBerth { node } | NotAtBrtStation { node } => (vec![], vec![node], vec![]),
        NoSuchLine { line } | LineStops { line } | StopTwice { line, .. } | StopNotStation { line, .. } | StopsOffPath { line } | LinePath { line }
        | TramTrack { line } | PortStop { line } | NoWayBack { line } | OneWayNoWay { line } | OneTrackStop { line, .. } | LineSchedule { line }
        | StationInUse { line, .. } => (vec![], vec![], vec![line]),
        _ => (vec![], vec![], vec![]),
    }
}

#[wasm_bindgen]
impl TrackApi {
    /// T-1447: the picked track (`edges`) and stations (`stations`), with the lines wholly on them and the links between
    /// them, as network bytes the save writer wrote at `save_level`, every piece blueprint. Empty when none of it could
    /// be copied; `clip_left_out` says how many picked pieces and stations the checks left out.
    pub fn clip(&mut self, edges: &[u32], stations: &[u32]) -> Vec<u8> {
        let src = &self.w.net;
        let asked = (edges.iter().filter(|&&e| src.edge_ok(e)).count() + stations.iter().filter(|&&n| src.is_station(n)).count()) as u32;
        let want = want_of(src, edges, stations);
        let mut tmp = TrackApi::new(src.origin_lon, src.origin_lat, src.right_hand);
        tmp.w = TrackWorld::new(Network::new_in(src.frame_kind, src.origin_lon, src.origin_lat, src.right_hand));
        tmp.unlimited = true;
        let placed = tmp.place(src, want, false);
        self.clip_left = asked - placed.as_ref().map_or(0, |w| (w.edges.len() + w.stations.len()) as u32);
        if placed.is_none() { vec![] } else { tmp.save() }
    }

    /// T-1447: picked pieces and stations the last `clip` left out.
    pub fn clip_left_out(&self) -> u32 {
        self.clip_left
    }

    /// T-1447: whether a clip written at save level `level` can be pasted: 0 it can, 1 a newer build wrote it (a level
    /// above this build's, a newer network format or bytes this build does not read), 2 it does not read.
    pub fn clip_check(bytes: &[u8], level: u32) -> u32 {
        if level > save::LEVEL || save::is_newer_magic(bytes) {
            return 1;
        }
        if save::decode_at(bytes, level).is_err() {
            return 2;
        }
        if TrackApi::save_unread(bytes) > 0 { 1 } else { 0 }
    }

    /// T-1447: add a clip (`clip`'s bytes, written at save level `level`) to the network as blueprint, where it was, in
    /// one undoable edit; what the checks refuse is left out and the rest tried again, and `last_paste` lists what went
    /// in. 0 when anything went in; else the last refusal's issue count, at least 1.
    pub fn paste(&mut self, bytes: &[u8], level: u32) -> u32 {
        self.pasted = vec![0, 0, 0, 0];
        let Ok(src) = save::decode_at(bytes, level) else {
            self.issues.clear();
            return 1;
        };
        if src.frame_kind != self.w.net.frame_kind {
            self.issues.clear();
            return 1;
        }
        let edges: Vec<u32> = (0..src.edge_count() as u32).filter(|&e| src.edge_ok(e)).collect();
        let stations: Vec<u32> = (0..src.node_count() as u32).filter(|&n| src.station_data(n).is_some()).collect();
        let want = want_of(&src, &edges, &stations);
        let asked = (want.edges.len() + want.stations.len()) as u32;
        let ids = self.place_ids(&src, want, true);
        let Some((w, ids)) = ids else {
            self.pasted = vec![0, 0, 0, asked];
            return (self.issues.len() as u32).max(1);
        };
        let left = asked - (w.edges.len() + w.stations.len()) as u32;
        let mut out = vec![w.edges.len() as u32];
        out.extend(w.edges.iter().map(|e| ids.edge[e]));
        out.push(w.stations.len() as u32);
        out.extend(w.stations.iter().map(|n| ids.node[n]));
        out.push(w.lines.len() as u32);
        out.extend(w.lines.iter().map(|l| ids.line[l]));
        out.push(left);
        self.pasted = out;
        0
    }

    /// T-1447: what the last `paste` added: the count of new track pieces then their ids, the same for stations (node
    /// ids) and lines, then how many of the clip's pieces and stations were left out.
    pub fn last_paste(&self) -> Vec<u32> {
        self.pasted.clone()
    }
}

impl TrackApi {
    fn place(&mut self, src: &Network, want: Want, clear_sep: bool) -> Option<Want> {
        self.place_ids(src, want, clear_sep).map(|(w, _)| w)
    }

    /// Add `want` from `src` as blueprint: the whole in one edit, leaving out what the checks name; where they name
    /// nothing of it, each connected part on its own, joined to the first part that goes in as one undo step. What went
    /// in, and the new ids; None when nothing did.
    fn place_ids(&mut self, src: &Network, want: Want, clear_sep: bool) -> Option<(Want, Ids)> {
        let mut ids = Ids::default();
        let whole = self.copy_round(src, want.clone(), &mut ids, clear_sep, false);
        let placed = match whole {
            Ok(w) => Some(w),
            Err(left) if left.is_empty() => None,
            Err(left) => {
                let mut done: Option<Want> = None;
                for part in parts(src, &left) {
                    if let Ok(w) = self.copy_round(src, part, &mut ids, clear_sep, done.is_some()) {
                        let d = done.get_or_insert_with(Want::default);
                        d.edges.extend(w.edges);
                        d.stations.extend(w.stations);
                        d.lines.extend(w.lines);
                    }
                }
                done
            }
        };
        // ids no op claimed go back
        let net = &mut self.w.net;
        for &n in ids.node.values() { net.release_node(n); }
        for &e in ids.edge.values() { net.release_edge(e); }
        placed.map(|w| (w, ids))
    }

    /// Rounds of one edit adding `want`, each leaving out what the last refusal named. `join`: taken back with the
    /// edit before (`run`'s `join`). Ok with what went in; Err with what is left when a refusal named none of it.
    fn copy_round(&mut self, src: &Network, mut want: Want, ids: &mut Ids, clear_sep: bool, join: bool) -> Result<Want, Want> {
        for _ in 0..ROUNDS {
            if want.is_empty() {
                return Err(want);
            }
            let op = Op::Batch(self.ops_for(src, &want, ids, clear_sep));
            let was = std::mem::replace(&mut self.join, join);
            let r = self.run(op);
            self.join = was;
            if r == 0 {
                return Ok(want);
            }
            let rev = |m: &HashMap<u32, u32>| -> HashMap<u32, u32> { m.iter().map(|(&k, &v)| (v, k)).collect() };
            let (edge_src, node_src, line_src) = (rev(&ids.edge), rev(&ids.node), rev(&ids.line));
            let mut dropped = false;
            for i in std::mem::take(&mut self.issues) {
                let (es, ns, ls) = named(&i);
                for e in es {
                    if let Some(&s) = edge_src.get(&e) { dropped |= want.drop_edge(src, s); }
                }
                for n in ns {
                    let Some(&s) = node_src.get(&n) else { continue };
                    if want.stations.contains(&s) {
                        dropped |= want.drop_station(src, s);
                    } else {
                        let at: Vec<u32> = want.edges.iter().copied().filter(|&e| src.edge_a[e as usize] == s || src.edge_b[e as usize] == s).collect();
                        for e in at { dropped |= want.drop_edge(src, e); }
                    }
                }
                for l in ls {
                    if let Some(&s) = line_src.get(&l) {
                        if let Some(k) = want.lines.iter().position(|&x| x == s) { want.lines.swap_remove(k); dropped = true; }
                    }
                }
            }
            if !dropped {
                return Err(want);
            }
        }
        Err(want)
    }

    /// The ops adding `want`: nodes, track, stations, lines, then links and kept-apart pairs, every new id from `ids`.
    fn ops_for(&mut self, src: &Network, want: &Want, ids: &mut Ids, clear_sep: bool) -> Vec<Op> {
        let net = &mut self.w.net;
        let mut ops = vec![];
        let mut nodes: Vec<u32> = want.edges.iter().flat_map(|&e| [src.edge_a[e as usize], src.edge_b[e as usize]]).chain(want.stations.iter().copied()).collect();
        nodes.sort_unstable();
        nodes.dedup();
        for &n in &nodes {
            let id = *ids.node.entry(n).or_insert_with(|| net.alloc_node());
            ops.push(Op::Node { id, data: src.node_data(n) });
        }
        for &e in &want.edges {
            let id = *ids.edge.entry(e).or_insert_with(|| net.alloc_edge());
            let Some(mut d) = src.edge_data(e) else { continue };
            (d.a, d.b) = (ids.node[&d.a], ids.node[&d.b]);
            d.off &= !d.built;
            d.built = false;
            d.day = 0;
            d.sep &= !clear_sep;
            ops.push(Op::Edge { id, data: Some(d) });
        }
        for &n in &want.stations {
            let Some(mut s) = src.station_data(n) else { continue };
            s.off &= !s.built;
            s.built = false;
            s.day = 0;
            // the fare areas are the source game's
            s.fare = 0;
            ops.push(Op::Station { node: ids.node[&n], data: Some(s) });
        }
        for &l in &want.lines {
            let id = *ids.line.entry(l).or_insert_with(|| net.alloc_line());
            let Some(mut d) = src.line_data(l) else { continue };
            let node = |n: &u32| ids.node[n];
            d.stops = d.stops.iter().map(node).collect();
            d.path = d.path.iter().map(|&p| ids.edge[&path_edge(p)] << 1 | path_dir(p)).collect();
            for (n, _) in d.ones.iter_mut().chain(d.only.iter_mut()) { *n = ids.node[n]; }
            // the Lines tab's groups are the source game's; a quad line's twin is not copied
            (d.group, d.group_look, d.twin, d.day) = (0, Default::default(), Twin::None, 0);
            d.group_name.clear();
            d.outer.clear();
            ops.push(Op::Line { id, data: Some(d) });
        }
        for &[a, b] in &want.links {
            let (a, b) = (ids.node[&a], ids.node[&b]);
            ops.push(Op::Link { pair: [a.min(b), a.max(b)], on: true });
        }
        for &[a, b] in &want.apart {
            let (a, b) = (ids.node[&a], ids.node[&b]);
            ops.push(Op::Apart { pair: [a.min(b), a.max(b)], on: true });
        }
        ops
    }
}

/// `want` split into its connected parts (track and stations joined through nodes), each with its lines, links and
/// kept-apart pairs.
fn parts(src: &Network, want: &Want) -> Vec<Want> {
    let mut root: HashMap<u32, u32> = HashMap::new();
    fn find(root: &mut HashMap<u32, u32>, n: u32) -> u32 {
        let p = *root.entry(n).or_insert(n);
        if p == n { return n; }
        let r = find(root, p);
        root.insert(n, r);
        r
    }
    for &e in &want.edges {
        let (a, b) = (find(&mut root, src.edge_a[e as usize]), find(&mut root, src.edge_b[e as usize]));
        root.insert(a, b);
    }
    let mut by: HashMap<u32, Want> = HashMap::new();
    let mut order = vec![];
    for &e in &want.edges {
        let r = find(&mut root, src.edge_a[e as usize]);
        if !by.contains_key(&r) { order.push(r); }
        by.entry(r).or_default().edges.push(e);
    }
    for &n in &want.stations {
        let r = find(&mut root, n);
        if !by.contains_key(&r) { order.push(r); }
        by.entry(r).or_default().stations.push(n);
    }
    for &l in &want.lines {
        if let Some(&s) = src.line_stops(l).first() {
            let r = find(&mut root, s);
            if let Some(p) = by.get_mut(&r) { p.lines.push(l); }
        }
    }
    order.into_iter().filter_map(|r| by.remove(&r)).collect()
}
