//! The lines related to a line (T-1168): the routes for `related::related`.

use super::*;
use crate::track::net::path_edge;
use crate::track::related::{self, Route, NONE};
use crate::track::service::Run;

/// A run as a related-lines route: its edges sampled every 200 m at most (tighter on curves).
pub(super) fn route_of(net: &Network, line: u32, run: &Run) -> Route {
    let mut r = Route { line, nodes: run.nodes.clone(), node_s: run.node_s.clone(), segs: run.segs.clone(), stop_at: vec![-1; run.nodes.len()], ..Default::default() };
    for (i, &k) in run.stop_idx.iter().enumerate() {
        if let Some(x) = r.stop_at.get_mut(k as usize) {
            if *x < 0 {
                *x = i as i32;
            }
        }
    }
    let mut pts: Vec<[f64; 2]> = vec![];
    for (j, &p) in run.segs.iter().enumerate() {
        pts.clear();
        geom::sample(net.edge_pieces(path_edge(p)), 200.0, 2.0, 0.0, &mut pts);
        if p & 1 == 1 {
            pts.reverse();
        }
        let (s0, s1) = (run.node_s[j], run.node_s[j + 1]);
        let chord: Vec<f64> = std::iter::once(0.0)
            .chain(pts.windows(2).scan(0.0, |a, w| {
                *a += (w[1][0] - w[0][0]).hypot(w[1][1] - w[0][1]);
                Some(*a)
            }))
            .collect();
        let total = chord.last().copied().unwrap_or(0.0).max(1e-9);
        let skip = if r.pts.is_empty() { 0 } else { 1 };
        for (q, c) in pts.iter().zip(&chord).skip(skip) {
            let s = s0 + (s1 - s0) * c / total;
            // offsets strictly increasing, for the searches along the route
            if r.ps.last().map_or(true, |&last| s > last) {
                r.pts.push(*q);
                r.ps.push(s);
            }
        }
    }
    if r.pts.is_empty() {
        if let Some(&n) = run.nodes.first() {
            r.pts.push([net.node_x[n as usize], net.node_y[n as usize]]);
            r.ps.push(0.0);
        }
    }
    r
}

#[wasm_bindgen]
impl TrackApi {
    /// The lines related to `line` (T-1168), flattened as `related::related` says. `complex_of`
    /// is each node's station complex (every station, blueprints too), `u32::MAX` for none.
    /// Only lines through a complex on its route or on one of its edges are measured.
    pub fn line_related(&self, line: u32, complex_of: &[u32]) -> Vec<f64> {
        let (net, svc) = (&self.w.net, &self.w.svc);
        let Some(ls) = svc.lines.get(line as usize).filter(|l| l.ok) else { return vec![] };
        let cx = |n: u32| complex_of.get(n as usize).copied().unwrap_or(NONE);
        let run = &ls.runs[0];
        let lcx: std::collections::HashSet<u32> = run.nodes.iter().map(|&n| cx(n)).filter(|&c| c != NONE).collect();
        let ledges: std::collections::HashSet<u32> = run.segs.iter().map(|&p| path_edge(p)).collect();
        let others: Vec<Route> = svc
            .lines
            .iter()
            .enumerate()
            .filter(|&(o, os)| {
                o != line as usize && os.ok && (os.runs[0].nodes.iter().any(|&n| lcx.contains(&cx(n))) || os.runs[0].segs.iter().any(|&p| ledges.contains(&path_edge(p))))
            })
            .map(|(o, os)| route_of(net, o as u32, &os.runs[0]))
            .collect();
        let merc = net.frame_kind == FrameKind::Merc;
        let k = |y: f64| if merc { crate::track::frame::merc_k(y) } else { 1.0 };
        related::related(&route_of(net, line, run), &others, &cx, &k)
    }
}
