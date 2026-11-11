//! Tracks inside a piece, and what trains can do where pieces meet (T-3137, notes/T-3137.md (c)).
//!
//! A piece (edge) is one alignment the player places, prices and deletes whole. Its tracks are
//! lanes at fixed offsets from that centre line, derived from its track count and the running side,
//! never stored. At a node, each edge end's lanes get a position across the node, and the node's
//! move table says which ends a train arriving through each end may leave by. Routing, line checks,
//! the flat-junction badge, junction capacity and the map's train guard all read the table and the
//! positions, so they agree by construction. A path entry stays `(edge, direction)`: on double
//! track the direction names the lane, single track has one, running the way its run takes from
//! the tracks of double track its ends join (`derive_ways`) unless the player set one.

use super::net::{path_dir, path_edge, Network, Ports, MAX_PORTS};
use super::params::{LANE_HALF_M, PORT_PITCH_M};

/// Directions a lane carries: a to b, b to a.
pub const WAY_AB: u8 = 1;
pub const WAY_BA: u8 = 2;

/// `Network::node_unusable`: one-way track pointing into the node with no way on, and pointing
/// out of it with no way in.
pub const UNUSABLE_IN: u8 = 1;
pub const UNUSABLE_OUT: u8 = 2;

/// One track of a piece.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Lane {
    /// From the centre line, m, positive to the left looking from a to b.
    pub off: f64,
    /// `WAY_AB`, `WAY_BA` or both.
    pub ways: u8,
}

/// A piece's lanes, right to left looking from a to b.
#[derive(Clone, Copy, Debug)]
pub struct Lanes {
    n: u8,
    lane: [Lane; 2],
}

impl Lanes {
    pub fn all(&self) -> &[Lane] {
        &self.lane[..self.n as usize]
    }
}

/// A node's move table, derived with its ports and indexed as its `Ports` orders the edge ends.
#[derive(Clone, Copy, Debug, Default)]
pub struct Moves {
    /// Where across the node the lane lies that a train arriving through each end runs on, and the
    /// one a train leaving through it takes: m, positive to the left looking along the node's
    /// heading. Every edge end meets the node at its centre, so an end's lanes sit around a slot of
    /// its own, `rank` x `PORT_PITCH_M` on its side: the ends' lanes stay together, in the order the
    /// ends part from the node.
    pub arrive_at: [f32; MAX_PORTS],
    pub leave_at: [f32; MAX_PORTS],
    /// Per arriving end, a bit per end a train may leave by without turning back: an end on the
    /// node's other side, with a lane running each way the move needs.
    pub to: [u8; MAX_PORTS],
    /// `to` whichever way the lanes run: the ends each end's lanes connect to.
    pub link: [u8; MAX_PORTS],
}

impl Moves {
    /// The crossing rule's positions of the move from arriving end `i` to leaving end `j`: its
    /// lane's position on the node's side 0, then on side 1. Two moves cross at the node when
    /// `(a1 - a2)(b1 - b2) < 0`.
    pub fn at(&self, ports: &Ports, i: usize, j: usize) -> (f64, f64) {
        let (a, b) = (self.arrive_at[i] as f64, self.leave_at[j] as f64);
        if ports.side[i] == 0 { (a, b) } else { (b, a) }
    }
}

/// The ends set in a bit mask of `Ports` indices, ascending.
pub fn ends_in(mask: u8) -> impl Iterator<Item = usize> {
    (0..MAX_PORTS).filter(move |&k| mask >> k & 1 != 0)
}

impl Network {
    /// The piece's lanes. Double track keeps to the running side: with right-hand running the
    /// right lane (looking from a to b) carries a to b. Single track's one lane runs its way
    /// (`edge_way`).
    pub fn lanes(&self, e: u32) -> Lanes {
        let i = e as usize;
        if self.edge_tracks[i] == 1 {
            let ways = match self.edge_way(e) {
                1 => WAY_AB,
                2 => WAY_BA,
                _ => WAY_AB | WAY_BA,
            };
            let lane = Lane { off: 0.0, ways };
            return Lanes { n: 1, lane: [lane; 2] };
        }
        let (right, left) = if self.right_hand { (WAY_AB, WAY_BA) } else { (WAY_BA, WAY_AB) };
        Lanes { n: 2, lane: [Lane { off: -LANE_HALF_M, ways: right }, Lane { off: LANE_HALF_M, ways: left }] }
    }

    /// The lane a path entry runs on; None where no lane of its piece runs its way.
    pub fn lane_of(&self, p: u32) -> Option<Lane> {
        let way = if path_dir(p) == 0 { WAY_AB } else { WAY_BA };
        self.lanes(path_edge(p)).all().iter().copied().find(|l| l.ways & way != 0)
    }

    /// Node `n`'s move table from its ports (`derive_node`, after the sides and ranks).
    pub(super) fn derive_moves(&mut self, n: u32) {
        let p = self.node_ports[n as usize];
        let k = p.n as usize;
        let mut m = Moves::default();
        for i in 0..k {
            let c = p.end[i];
            // a piece runs a to b outward from its end a, and an end on side 1 leaves along the
            // heading: the lane's offset as seen along the heading
            let sign = if (c & 1 == 0) == (p.side[i] == 1) { 1.0 } else { -1.0 };
            let slot = p.rank[i] as f64 * PORT_PITCH_M;
            // a way no lane runs is never used, so its position is never read
            let at = |entry: u32| (slot + sign * self.lane_of(entry).map_or(0.0, |l| l.off)) as f32;
            m.arrive_at[i] = at(c ^ 1);
            m.leave_at[i] = at(c);
        }
        for i in 0..k {
            let arrives = self.lane_of(p.end[i] ^ 1).is_some();
            for j in (0..k).filter(|&j| p.side[j] != p.side[i]) {
                m.link[i] |= 1 << j;
                if arrives && self.lane_of(p.end[j]).is_some() {
                    m.to[i] |= 1 << j;
                }
            }
        }
        self.node_moves[n as usize] = m;
    }

    /// T-3164: one-way single track at node `n` that no train can use, from its move table. Bit
    /// `UNUSABLE_IN`: an end whose track brings trains in with no move on (one-way pieces pointing
    /// into each other); bit `UNUSABLE_OUT`: an end whose track takes trains out that no move
    /// reaches (pointing away from each other). A bare end (a stub still being drawn) and a station
    /// (a line may end there) say nothing.
    pub fn node_unusable(&self, n: u32) -> u8 {
        let i = n as usize;
        let p = &self.node_ports[i];
        let k = p.n as usize;
        if k < 2 || self.node_platform[i] > 0 {
            return 0;
        }
        let m = &self.node_moves[i];
        let reached = m.to[..k].iter().fold(0u8, |acc, &t| acc | t);
        let mut out = 0;
        for j in 0..k {
            let (c, e) = (p.end[j], (p.end[j] >> 1) as usize);
            if self.edge_tracks[e] != 1 || self.edge_crossover[e] || self.edge_way(e as u32) == 0 {
                continue;
            }
            if self.lane_of(c ^ 1).is_some() && m.to[j] == 0 {
                out |= UNUSABLE_IN;
            }
            if self.lane_of(c).is_some() && reached >> j & 1 == 0 {
                out |= UNUSABLE_OUT;
            }
        }
        out
    }

    /// The node a path entry arrives at, its index there, and the ends a train may leave by
    /// without turning back (`Moves::to`).
    pub fn moves_after(&self, p: u32) -> (u32, Option<usize>, u8) {
        let (_, node) = self.path_nodes(p);
        let i = self.node_ports[node as usize].find(path_edge(p), 1 - path_dir(p));
        (node, i, i.map_or(0, |i| self.node_moves[node as usize].to[i]))
    }

    /// Whether a train may run on from path entry `p` into `q` through their node without turning
    /// back.
    pub fn move_ok(&self, p: u32, q: u32) -> bool {
        let (node, _, to) = self.moves_after(p);
        self.node_ports[node as usize].find(path_edge(q), path_dir(q)).is_some_and(|j| to >> j & 1 != 0)
    }

    /// The crossing rule's positions (`Moves::at`) of the move from path entry `p` into `q` at
    /// node `n`.
    pub fn move_at(&self, n: u32, p: u32, q: u32) -> (f64, f64) {
        let ports = &self.node_ports[n as usize];
        let i = ports.find(path_edge(p), 1 - path_dir(p)).unwrap_or(0);
        let j = ports.find(path_edge(q), path_dir(q)).unwrap_or(0);
        self.node_moves[n as usize].at(ports, i, j)
    }

    // ---- the way single track runs (T-3137, notes/T-3135.md "Direction")

    /// Single track's way: the player's `oneway` where set (3 pins both ways), else the way its run
    /// takes by itself (`derive_ways`). 0 both, 1 a to b, 2 b to a.
    pub fn edge_way(&self, e: u32) -> u8 {
        let i = e as usize;
        match self.edge_oneway[i] {
            w @ 1..=2 => w,
            3 => 0,
            _ => self.edge_infer[i],
        }
    }

    /// Whether edge end `c` (`edge << 1 | end`) is joined to one track of double track, and then
    /// whether trains run from the node into the piece there (else out of it into the node). A join
    /// counts at a junction with double track on the end's other side. The double's track to the
    /// right, looking out along the piece, carries trains away from the node with right-hand running.
    pub fn join_away(&self, c: u32) -> Option<bool> {
        let join = self.edge_join[(c >> 1) as usize][(c & 1) as usize];
        (join != 0 && self.join_applies(c)).then_some((join == 2) == self.right_hand)
    }

    /// Whether a join at edge end `c` would count: a node with double track on the end's other side, a
    /// junction or (T-3190) the loose end of the double track itself.
    pub fn join_applies(&self, c: u32) -> bool {
        let p = &self.node_ports[self.end_node(c >> 1, c & 1) as usize];
        let Some(i) = p.find(c >> 1, c & 1) else { return false };
        let double = |k: usize| self.edge_tracks[(p.end[k] >> 1) as usize] == 2 && !self.edge_crossover[(p.end[k] >> 1) as usize];
        p.n >= 2 && (0..p.n as usize).any(|k| p.side[k] != p.side[i] && double(k))
    }

    fn in_run(&self, e: u32) -> bool {
        self.edge_ok(e) && self.edge_tracks[e as usize] == 1 && !self.edge_crossover[e as usize]
    }

    /// What the node at run end `c` says about the run's way.
    fn end_vote(&self, c: u32) -> Vote {
        let n = self.end_node(c >> 1, c & 1) as usize;
        if self.node_platform[n] > 0 {
            return Vote::Both;
        }
        match self.node_ports[n].n {
            1 => Vote::None,
            // double track or a crossover on: the run meets track that runs both ways, unless it
            // continues one track of the double (T-3190)
            _ => self.join_away(c).map_or(Vote::Both, Vote::Joined),
        }
    }

    /// The run of single track through `e`: pieces (`in_run`) through nodes where just two of them
    /// meet, as (piece, walked a to b) from one end to the other; whether a station is inside it;
    /// whether it closes on itself.
    fn run_through(&self, e: u32) -> (Vec<(u32, bool)>, bool, bool) {
        let (mut ahead, mut behind) = (vec![(e, true)], vec![]);
        let (mut station, mut closed) = (false, false);
        for back in [false, true] {
            let (mut cur, mut at_b) = (e, !back);
            loop {
                let n = self.end_node(cur, at_b as u32) as usize;
                let p = &self.node_ports[n];
                let here = cur << 1 | at_b as u32;
                let Some(&o) = p.ends().iter().find(|&&c| c != here).filter(|_| p.n == 2) else { break };
                if !self.in_run(o >> 1) {
                    break;
                }
                station |= self.node_platform[n] > 0;
                if o >> 1 == e {
                    closed = true;
                    break;
                }
                // entered through its a: walked a to b, away from `e`
                let f = o & 1 == 0;
                if back { behind.push((o >> 1, !f)) } else { ahead.push((o >> 1, f)) }
                (cur, at_b) = (o >> 1, f);
            }
            if closed {
                break;
            }
        }
        behind.reverse();
        behind.extend(ahead);
        (behind, station, closed)
    }

    /// Each run of single track with a piece in `edges` or an end at `nodes` takes its way again:
    /// one way where every join at its ends sends trains that way (at least one) and nothing says
    /// both; both ways where a station is on it, it meets other track that runs both ways (a taper,
    /// a junction with no join, a crossover) or its joins disagree. A bare end says nothing. Returns
    /// the pieces whose way changed.
    pub(super) fn derive_ways(&mut self, edges: &[u32], nodes: &[u32]) -> Vec<u32> {
        let mut seeds = vec![];
        for &e in edges {
            if self.in_run(e) {
                seeds.push(e);
            } else if (e as usize) < self.edge_count() {
                self.edge_infer[e as usize] = 0;
            }
        }
        for &n in nodes {
            if self.node_ok(n) {
                seeds.extend(self.node_ports[n as usize].ends().iter().map(|&c| c >> 1).filter(|&e| self.in_run(e)));
            }
        }
        seeds.sort_unstable();
        seeds.dedup();
        let mut seen = std::collections::HashSet::new();
        let mut changed = vec![];
        for e in seeds {
            if seen.contains(&e) {
                continue;
            }
            let (run, station, closed) = self.run_through(e);
            let mut both = station || closed;
            let mut way = 0i8;
            if !closed {
                let (e0, f0) = run[0];
                let (e1, f1) = run[run.len() - 1];
                for (c, start) in [(e0 << 1 | !f0 as u32, true), (e1 << 1 | f1 as u32, false)] {
                    match self.end_vote(c) {
                        Vote::None => {}
                        Vote::Both => both = true,
                        Vote::Joined(away) => {
                            let w = if away == start { 1 } else { -1 };
                            both |= way != 0 && way != w;
                            way = w;
                        }
                    }
                }
            }
            for &(p, f) in &run {
                seen.insert(p);
                let infer = match (both, way, f) {
                    (false, 1, true) | (false, -1, false) => 1,
                    (false, 1, false) | (false, -1, true) => 2,
                    _ => 0,
                };
                let was = self.edge_way(p);
                self.edge_infer[p as usize] = infer;
                if self.edge_way(p) != was {
                    changed.push(p);
                }
            }
        }
        changed.sort_unstable();
        changed
    }
}

#[derive(Clone, Copy)]
enum Vote {
    None,
    Both,
    /// A join: whether trains run from the node into the run.
    Joined(bool),
}

/// The join (`EdgeData::join`) that sends trains from the node into the piece (`away`) or out of it.
pub fn join_for(away: bool, right_hand: bool) -> u8 {
    if away == right_hand { 2 } else { 1 }
}
