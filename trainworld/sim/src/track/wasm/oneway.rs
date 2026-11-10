//! One-way single track (T-3094): the flag on an edge, and the lines it sends another way.

use super::*;
use crate::track::net::path_edge;

#[wasm_bindgen]
impl TrackApi {
    /// An edge's one-way flag: 0 both ways, 1 a to b only, 2 b to a only.
    pub fn edge_oneway(&self, edge: u32) -> u8 {
        let n = &self.w.net;
        if n.edge_ok(edge) { n.edge_oneway[edge as usize] } else { 0 }
    }

    /// The edges that run one way, as pairs: edge, flag.
    pub fn oneway_edges(&self) -> Vec<u32> {
        let n = &self.w.net;
        (0..n.edge_count() as u32).filter(|&e| n.edge_ok(e) && n.edge_oneway[e as usize] != 0).flat_map(|e| [e, n.edge_oneway[e as usize] as u32]).collect()
    }

    /// Set single track to run one way (`dir` 1 a to b, 2 b to a) or both (0). Free and one undoable
    /// edit: the lines that ran the other way through it take another way through the same stops in
    /// the same edit, and it is refused (`OneWayNoWay`) when one has none.
    pub fn set_edge_oneway(&mut self, edge: u32, dir: u8) -> u32 {
        match self.oneway_ops(edge, dir) {
            Ok((ops, _)) => self.run(Op::Batch(ops)),
            Err(i) => self.refuse(i),
        }
    }

    /// What `set_edge_oneway` would do to lines, as pairs: line, 1 if it can take another way and 0 if it
    /// has none. Empty when no line runs against it, or the edit is refused for another reason.
    pub fn oneway_effects(&mut self, edge: u32, dir: u8) -> Vec<f64> {
        match self.oneway_plan(edge, dir) {
            Ok((_, rows)) => rows.iter().flat_map(|r| [r.0 as f64, r.1.is_some() as u8 as f64]).collect(),
            Err(_) => vec![],
        }
    }
}

impl TrackApi {
    /// The edge's new data and the lines it changes, each with the path through its stops that does not
    /// run against it (None: it has no way). A line that goes both ways runs every track it uses
    /// both ways, so setting track one way changes any line over it: its forward path moves
    /// off an entry against the flag, and its return run takes its own way back
    /// (`Network::back_path_of`). A loop that runs one way only needs its path to avoid the flag.
    #[allow(clippy::type_complexity)]
    fn oneway_plan(&mut self, edge: u32, dir: u8) -> Result<(Op, Vec<(u32, Option<Vec<u32>>)>), Issue> {
        let net = &self.w.net;
        if !net.edge_ok(edge) {
            return Err(Issue::NoSuchEdge { edge });
        }
        let mut d = net.edge_data(edge).ok_or(Issue::NoSuchEdge { edge })?;
        if dir > 2 || (dir != 0 && (d.tracks != 1 || d.crossover)) {
            return Err(Issue::Tracks { edge });
        }
        d.oneway = dir;
        let op = Op::Edge { id: edge, data: Some(d) };
        let rail = |l: u32| net.line_ok(l) && net.line_bus.get(l as usize).map_or(true, |b| b.is_none()) && net.line_ferry.get(l as usize).map_or(true, |f| f.is_none());
        // lines whose return run goes round other one-way track may now go round this
        let mut hit: Vec<u32> = (0..net.line_count() as u32).filter(|&l| rail(l) && net.line_path(l).iter().any(|&p| path_edge(p) == edge)).collect();
        if dir != 0 {
            hit.extend(net.back_lines());
        }
        hit.sort_unstable();
        hit.dedup();
        let lines: Vec<(u32, Vec<u32>, Vec<u32>, bool, bool)> = hit.iter().map(|&l| (l, net.line_stops(l).to_vec(), net.line_path(l).to_vec(), net.line_manual[l as usize], net.line_runs_once(l))).collect();
        let was: Vec<Option<Vec<u32>>> = lines.iter().map(|x| net.back_path_of(&x.1, &x.2, x.3)).collect();
        let inv = self.w.sketch(op.clone(), false).map_err(|mut v| v.swap_remove(0))?;
        let net = &self.w.net;
        let mut rows = vec![];
        for (k, (l, stops, path, manual, once)) in lines.into_iter().enumerate() {
            let fwd_ok = path.iter().all(|&p| net.dir_ok(p));
            let new = if fwd_ok { Some(path.clone()) } else { net.route_hops_turn(&stops, if manual { 0.0 } else { REVERSAL_SEARCH_S }).ok() };
            let new = new.filter(|np| once || net.back_path_of(&stops, np, manual).is_some());
            let moved = match &new {
                Some(np) => *np != path || (!once && net.back_path_of(&stops, np, manual) != was[k]),
                None => true,
            };
            if moved {
                rows.push((l, new));
            }
        }
        self.w.unsketch(inv);
        Ok((op, rows))
    }

    /// `oneway_plan` as the ops of one edit; `OneWayNoWay` names the first line with no other way.
    fn oneway_ops(&mut self, edge: u32, dir: u8) -> Result<(Vec<Op>, usize), Issue> {
        let (op, rows) = self.oneway_plan(edge, dir)?;
        let mut ops = vec![op];
        for (l, path) in &rows {
            let Some(path) = path else { return Err(Issue::OneWayNoWay { line: *l }) };
            let mut ld = self.w.net.line_data(*l).ok_or(Issue::NoSuchLine { line: *l })?;
            if ld.path != *path {
                ld.path = path.clone();
                ops.push(Op::Line { id: *l, data: Some(ld) });
            }
        }
        Ok((ops, rows.len()))
    }
}
