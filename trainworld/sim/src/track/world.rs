//! Edit operations with inverses, undo and redo, and what each edit dirties (SPEC 6.5,
//! notes/T-008.md 7). `TrackWorld` is what the clock worker owns: the network, the service
//! derived from it, and the undo history.

use super::capacity::ResKind;
use super::net::{path_edge, EdgeData, Issue, LineData, Network, NodeData, StationData};
use super::params::*;
use super::service::{Service, ServiceDirty};

/// One edit. The primitive setters replace an object whole (`None` removes it); `Split` and
/// `DeleteEdge` expand into primitives when applied. Every applied op yields its inverse, which
/// is again an `Op`.
#[derive(Clone, Debug, PartialEq)]
pub enum Op {
    Node { id: u32, data: Option<NodeData> },
    Edge { id: u32, data: Option<EdgeData> },
    Station { node: u32, data: Option<StationData> },
    Line { id: u32, data: Option<LineData> },
    Schedule { id: u32, tph: [f32; 3] },
    /// Split `edge` at chainage `s` with a new node `node`; the far part becomes `new_edge`.
    /// Line paths through it are rewritten. Geometry does not change.
    Split { edge: u32, s: f64, node: u32, new_edge: u32 },
    /// Remove an edge; each line using it is rerouted by the fastest path or left broken.
    DeleteEdge { id: u32 },
    Batch(Vec<Op>),
}

/// What an edit touched (SPEC 6.5).
#[derive(Clone, Debug, Default)]
pub struct Dirty {
    /// Zoom-12 render tiles of the old and new extent.
    pub tiles: Vec<(u32, u32)>,
    pub edges: Vec<u32>,
    pub nodes: Vec<u32>,
    /// Positions of crossings removed or added.
    pub crossings: Vec<[f64; 2]>,
    /// Resources (indices into `svc.cap.res`, valid until the next edit) on touched track or used
    /// by a directly changed line.
    pub resources: Vec<u32>,
    /// Lines whose path, stops, geometry or schedule changed.
    pub lines: Vec<u32>,
    pub service: ServiceDirty,
}

#[derive(Clone, Debug)]
pub struct EditResult {
    pub dirty: Dirty,
    /// Build cost of the whole network (blueprint included) before and after, US$M.
    pub cost_before: f64,
    pub cost_after: f64,
    /// Cost of what is constructed before and after (`Network::built_cost`).
    pub built_before: f64,
    pub built_after: f64,
    /// What the edit costs: the increase in constructed cost. Blueprint edits cost nothing and
    /// removing constructed track refunds nothing (SPEC 6.4, Anita 2026-10-09).
    pub charge: f64,
}

pub struct TrackWorld {
    pub net: Network,
    pub svc: Service,
    undo: Vec<Op>,
    redo: Vec<Op>,
}

/// Why an edit that changes what is constructed was refused.
#[derive(Clone, Debug, PartialEq)]
pub enum ChargeError {
    Issues(Vec<Issue>),
    /// The edit costs `need` US$M and there is only `have`.
    Money { need: f64, have: f64 },
}

impl TrackWorld {
    pub fn new(net: Network) -> TrackWorld {
        let mut w = TrackWorld { net, svc: Service::default(), undo: vec![], redo: vec![] };
        w.reload();
        w
    }

    /// Derive everything from the inputs (after a load or a bulk build).
    pub fn reload(&mut self) {
        self.net.touch_all();
        let t = self.net.take_touch();
        self.net.derive(&t);
        self.svc.update_all(&self.net);
    }

    pub fn can_undo(&self) -> bool {
        !self.undo.is_empty()
    }
    pub fn can_redo(&self) -> bool {
        !self.redo.is_empty()
    }

    /// Apply, validate, recompute. On problems nothing changes and the problems come back.
    pub fn apply(&mut self, op: Op) -> Result<EditResult, Vec<Issue>> {
        let (r, inv) = self.commit(op)?;
        self.undo.push(inv);
        self.redo.clear();
        Ok(r)
    }

    /// Apply an edit that changes what is constructed (construct, remove constructed track, make
    /// a constructed junction flying), paying `charge` out of `cash` US$M. Such an edit is final:
    /// it is not undone, and it clears the undo history, so no undo can rebuild or unbuild
    /// anything for free (SPEC 6.4). Refused, with nothing changed, if the cash is short.
    pub fn apply_charged(&mut self, op: Op, cash: f64) -> Result<EditResult, ChargeError> {
        self.apply_paid(op, cash, true, &|_| 0.0).map(|r| r.0)
    }

    /// Apply an edit and pay for it: its construction charge plus `extra`, evaluated on the
    /// edited network (the trains it makes the lines need, T-028). Refused, with nothing changed,
    /// if that comes to more than `cash`; an edit costing nothing is never refused for money, so
    /// a network in debt can still be edited. `final_` edits clear the undo history (SPEC 6.4);
    /// the others go on the undo stack. Returns the result and what `extra` came to.
    pub fn apply_paid(&mut self, op: Op, cash: f64, final_: bool, extra: &dyn Fn(&TrackWorld) -> f64) -> Result<(EditResult, f64), ChargeError> {
        let (r, inv) = self.commit(op).map_err(ChargeError::Issues)?;
        let x = extra(self);
        let need = r.charge + x;
        if need > 1e-9 && need > cash + 1e-6 {
            let _ = self.commit(inv);
            return Err(ChargeError::Money { need, have: cash });
        }
        if final_ {
            self.undo.clear();
        } else {
            self.undo.push(inv);
        }
        self.redo.clear();
        Ok((r, x))
    }

    /// Undo (or redo) and pay `extra` as `apply_paid` does: undoing a line's removal can need
    /// trains again. Nothing changes if it is refused.
    pub fn step_paid(&mut self, redo: bool, cash: f64, extra: &dyn Fn(&TrackWorld) -> f64) -> Option<Result<(EditResult, f64), ChargeError>> {
        let op = if redo { self.redo.pop()? } else { self.undo.pop()? };
        let back = |w: &mut TrackWorld, op: Op| if redo { w.redo.push(op) } else { w.undo.push(op) };
        Some(match self.commit(op.clone()) {
            Ok((r, inv)) => {
                let x = extra(self);
                let need = r.charge + x;
                if need > 1e-9 && need > cash + 1e-6 {
                    let _ = self.commit(inv);
                    back(self, op);
                    Err(ChargeError::Money { need, have: cash })
                } else {
                    if redo {
                        self.undo.push(inv);
                    } else {
                        self.redo.push(inv);
                    }
                    Ok((r, x))
                }
            }
            Err(e) => {
                back(self, op);
                Err(ChargeError::Issues(e))
            }
        })
    }

    /// What an edit would do, without doing it: applied, measured and rolled back. The drawing
    /// tool's preview, with every check a real edit gets (crossings, ports, platforms).
    pub fn trial(&mut self, op: Op) -> Result<EditResult, Vec<Issue>> {
        let (r, inv) = self.commit(op)?;
        let _ = self.commit(inv);
        Ok(r)
    }

    pub fn undo(&mut self) -> Option<Result<EditResult, Vec<Issue>>> {
        let op = self.undo.pop()?;
        Some(match self.commit(op.clone()) {
            Ok((r, inv)) => {
                self.redo.push(inv);
                Ok(r)
            }
            Err(e) => {
                self.undo.push(op);
                Err(e)
            }
        })
    }

    pub fn redo(&mut self) -> Option<Result<EditResult, Vec<Issue>>> {
        let op = self.redo.pop()?;
        Some(match self.commit(op.clone()) {
            Ok((r, inv)) => {
                self.undo.push(inv);
                Ok(r)
            }
            Err(e) => {
                self.redo.push(op);
                Err(e)
            }
        })
    }

    fn commit(&mut self, op: Op) -> Result<(EditResult, Op), Vec<Issue>> {
        let cost_before = self.net.total_cost();
        let built_before = self.net.built_cost();
        let again = op.clone();
        let (mut inv, mut t, mut der) = self.apply_and_derive(op)?;
        let issues = self.check(&t, &der);
        if !issues.is_empty() {
            // Refuse only what the edit adds: problems already there (an old save under a newer
            // water mask, say) must not block fixing things nearby.
            let _ = self.apply_raw(inv);
            let t2 = self.net.take_touch();
            self.net.derive(&t2);
            let before = self.check(&t, &der);
            let added: Vec<Issue> = issues.into_iter().filter(|i| !before.iter().any(|b| same_issue(i, b))).collect();
            if !added.is_empty() {
                return Err(added);
            }
            (inv, t, der) = self.apply_and_derive(again)?;
        }
        // Lines to rebuild: set directly, on touched track, or stopping at a touched station.
        let mut lines: Vec<u32> = t.lines.clone();
        for &e in &der.edges {
            lines.extend_from_slice(self.net.lines_on_edge(e));
        }
        if !t.stations.is_empty() {
            for l in 0..self.net.line_count() as u32 {
                if self.net.line_ok(l) && self.net.line_stops(l).iter().any(|s| t.stations.binary_search(s).is_ok()) {
                    lines.push(l);
                }
            }
        }
        lines.sort_unstable();
        lines.dedup();
        let junction_touched = der.nodes.iter().any(|&n| self.net.node_ok(n) && self.net.node_ports[n as usize].n >= 3);
        // A station touched with no line stopping there still changes passing lines' platforms.
        let cap_dirty = !der.crossings.is_empty() || junction_touched || !t.stations.is_empty();
        // The capacity pass rebuilds only around what was touched (T-050).
        let mut cap_nodes = der.nodes.clone();
        cap_nodes.extend_from_slice(&t.stations);
        let svc = self.svc.update_scoped(&self.net, &lines, &t.schedules, cap_dirty, Some((&der.edges, &cap_nodes)));
        let mut direct = lines.clone();
        direct.extend_from_slice(&t.schedules);
        direct.sort_unstable();
        direct.dedup();
        let mut resources = vec![];
        for (r, res) in self.svc.cap.res.iter().enumerate() {
            let on_track = (res.edge != u32::MAX && der.edges.binary_search(&res.edge).is_ok())
                || (res.node != u32::MAX && der.nodes.binary_search(&res.node).is_ok());
            if on_track || self.svc.cap.users_of(r).iter().any(|u| direct.binary_search(&u.line).is_ok()) {
                resources.push(r as u32);
            }
        }
        let cost_after = self.net.total_cost();
        let built_after = self.net.built_cost();
        let r = EditResult {
            dirty: Dirty { tiles: der.tiles, edges: der.edges, nodes: der.nodes, crossings: der.crossings, resources, lines: direct, service: svc },
            cost_before,
            cost_after,
            built_before,
            built_after,
            charge: (built_after - built_before).max(0.0),
        };
        Ok((r, inv))
    }

    fn apply_and_derive(&mut self, op: Op) -> Result<(Op, super::net::Touch, super::net::Derived), Vec<Issue>> {
        let inv = match self.apply_raw(op) {
            Ok(i) => i,
            Err(e) => {
                self.net.take_touch();
                return Err(vec![e]);
            }
        };
        let t = self.net.take_touch();
        let der = self.net.derive(&t);
        Ok((inv, t, der))
    }

    /// Problems on what an edit touched: its edges, their nodes and stations, their crossings,
    /// and lines it set.
    fn check(&self, t: &super::net::Touch, der: &super::net::Derived) -> Vec<Issue> {
        let alive: Vec<u32> = der.edges.iter().copied().filter(|&e| self.net.edge_ok(e)).collect();
        let mut issues = self.net.validate(&alive, &der.nodes, &[]);
        for &l in &t.lines {
            if self.net.line_ok(l) {
                if let Err(i) = self.net.check_line(l) {
                    issues.push(i);
                }
            }
        }
        issues
    }

    /// Apply without deriving or validating; returns the inverse. A failing batch is rolled back.
    fn apply_raw(&mut self, op: Op) -> Result<Op, Issue> {
        let n = &mut self.net;
        Ok(match op {
            Op::Node { id, data } => Op::Node { id, data: n.set_node(id, data)? },
            Op::Edge { id, data } => Op::Edge { id, data: n.set_edge(id, data)? },
            Op::Station { node, data } => Op::Station { node, data: n.set_station(node, data)? },
            Op::Line { id, data } => Op::Line { id, data: n.set_line(id, data)? },
            Op::Schedule { id, tph } => Op::Schedule { id, tph: n.set_schedule(id, tph)? },
            Op::Split { edge, s, node, new_edge } => {
                if n.node_ok(node) {
                    return Err(Issue::NoSuchNode { node });
                }
                if n.edge_ok(new_edge) {
                    return Err(Issue::NoSuchEdge { edge: new_edge });
                }
                let (nd, mut first, mut second) = n.split_plan(edge, s)?;
                first.b = node;
                second.a = node;
                let mut ops = vec![Op::Node { id: node, data: Some(nd) }, Op::Edge { id: edge, data: Some(first) }, Op::Edge { id: new_edge, data: Some(second) }];
                for l in lines_using(n, edge) {
                    let mut d = n.line_data(l).unwrap();
                    let mut path = Vec::with_capacity(d.path.len() + 1);
                    for &p in &d.path {
                        if path_edge(p) == edge {
                            let dir = p & 1;
                            if dir == 0 {
                                path.push(edge << 1);
                                path.push(new_edge << 1);
                            } else {
                                path.push(new_edge << 1 | 1);
                                path.push(edge << 1 | 1);
                            }
                        } else {
                            path.push(p);
                        }
                    }
                    d.path = path;
                    ops.push(Op::Line { id: l, data: Some(d) });
                }
                self.apply_raw(Op::Batch(ops))?
            }
            Op::DeleteEdge { id } => {
                if !n.edge_ok(id) {
                    return Err(Issue::NoSuchEdge { edge: id });
                }
                let users = lines_using(n, id);
                let mut inv = vec![Op::Edge { id, data: n.set_edge(id, None)? }];
                for l in users {
                    let mut d = self.net.line_data(l).unwrap();
                    if let Some(path) = self.net.route(&d.stops) {
                        d.path = path;
                        let old = self.net.set_line(l, Some(d))?;
                        inv.push(Op::Line { id: l, data: old });
                    }
                }
                inv.reverse();
                Op::Batch(inv)
            }
            Op::Batch(ops) => {
                let mut inv = Vec::with_capacity(ops.len());
                for o in ops {
                    match self.apply_raw(o) {
                        Ok(i) => inv.push(i),
                        Err(e) => {
                            for i in inv.into_iter().rev() {
                                let _ = self.apply_raw(i);
                            }
                            return Err(e);
                        }
                    }
                }
                inv.reverse();
                Op::Batch(inv)
            }
        })
    }

    /// Cost and validity of an alignment not yet built (the drawing tool's preview runs the same
    /// code): the fitted geometry, its track cost (water included) and its geometry problems.
    pub fn preview(&self, vertices: &[super::geom::Pi], tracks: u8) -> (super::geom::Fit, f64, bool) {
        let f = super::geom::fit(vertices);
        let mut wet = vec![];
        super::cost::water_spans(f.len, |s| { let (x, y, _) = super::geom::pos_at(&f.pieces, s); (x, y) }, self.net.water_mask.as_ref(), &mut wet);
        let c = super::cost::track_cost(f.len, &f.vert, &wet, tracks);
        let ground_wet = wet.iter().any(|w| {
            let mut s = w[0];
            while s <= w[1] {
                if super::geom::height_at(&f.vert, s).abs() < LEVEL_H / 2.0 {
                    return true;
                }
                s += WATER_STEP / 2.0;
            }
            false
        });
        let ok = f.issues.is_empty() && !ground_wet;
        (f, c, ok)
    }

    /// Resources at or above 75% for a level (what T-031 marks on the map).
    pub fn busy(&self, level: usize) -> Vec<(usize, ResKind, f64, f64)> {
        let c = &self.svc.cap;
        (0..c.res.len()).filter(|&r| c.rho[r][level] >= 0.75).map(|r| (r, c.res[r].kind, c.rho[r][level], c.delay[r][level])).collect()
    }
}

/// Two problems are the same one, ignoring where exactly a moved crossing now is.
fn same_issue(a: &Issue, b: &Issue) -> bool {
    match (a, b) {
        (Issue::Crossing { e1, e2, overlap, .. }, Issue::Crossing { e1: f1, e2: f2, overlap: o2, .. }) => (e1, e2, overlap) == (f1, f2, o2),
        (Issue::GroundOverWater { edge, .. }, Issue::GroundOverWater { edge: f, .. }) => edge == f,
        _ => a == b,
    }
}

fn lines_using(n: &Network, e: u32) -> Vec<u32> {
    (0..n.line_count() as u32).filter(|&l| n.line_ok(l) && n.line_path(l).iter().any(|&p| path_edge(p) == e)).collect()
}
