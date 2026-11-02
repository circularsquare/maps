//! The gap between a line and a station no track joins it to (T-1896, docs/ui-building.md "Stops").

use super::*;

/// Points tried on a piece with no loose ends at most; more are thinned evenly.
const MAX_POINTS: usize = 2000;

/// One side of a gap: the track a search reached, as the line router moves over it.
struct Side {
    nodes: Vec<u32>,
    edges: Vec<u32>,
}

impl Side {
    fn reached(net: &Network, from: &[u32]) -> Side {
        let (seen_node, seen) = reach_from(net, from);
        let nodes = (0..net.node_count() as u32).filter(|&n| seen_node[n as usize]).collect();
        let edges = (0..net.edge_count() as u32).filter(|&e| net.edge_ok(e) && (seen[2 * e as usize] || seen[2 * e as usize + 1])).collect();
        Side { nodes, edges }
    }

    /// Where the track stops: nodes with one track end or none; every node (thinned) if there is none.
    fn ends(&self, net: &Network) -> Vec<u32> {
        let loose: Vec<u32> = self.nodes.iter().copied().filter(|&n| net.node_ports[n as usize].n <= 1).collect();
        if !loose.is_empty() {
            return loose;
        }
        let step = self.nodes.len().div_ceil(MAX_POINTS).max(1);
        self.nodes.iter().copied().step_by(step).collect()
    }

    /// The point of this track nearest (x, y), and its distance (plane metres).
    fn nearest(&self, net: &Network, x: f64, y: f64) -> Option<([f64; 2], f64)> {
        let mut best: Option<([f64; 2], f64)> = None;
        let xy = |n: u32| [net.node_x[n as usize], net.node_y[n as usize]];
        if self.edges.is_empty() {
            for &n in &self.nodes {
                let p = xy(n);
                let d = (p[0] - x).hypot(p[1] - y);
                if best.is_none_or(|b| d < b.1) {
                    best = Some((p, d));
                }
            }
            return best;
        }
        for &e in &self.edges {
            let pieces = net.edge_pieces(e);
            // no point of the edge is nearer than its ends less half its length (plane metres)
            let plen: f64 = pieces.iter().map(|p| p.plen).sum();
            let (a, b) = (xy(net.edge_a[e as usize]), xy(net.edge_b[e as usize]));
            let floor = ((a[0] - x).hypot(a[1] - y) + (b[0] - x).hypot(b[1] - y) - plen) / 2.0;
            if best.is_some_and(|b| floor >= b.1) {
                continue;
            }
            let (s, d) = geom::project(pieces, x, y);
            if best.is_none_or(|b| d < b.1) {
                let (px, py, _) = geom::pos_at(pieces, s);
                best = Some(([px, py], d));
            }
        }
        best
    }
}

/// The gap between the track reachable from `stops` and the track reachable from station `node`,
/// as the plane points `[ax, ay, bx, by]` (a on the line's side): of the loose ends of either side
/// (or all its nodes where it has none), the one nearest the other side's track, and that nearest
/// point. None when the two sides share track, or either has none at all.
pub(super) fn gap_between(net: &Network, stops: &[u32], node: u32) -> Option<[f64; 4]> {
    if stops.is_empty() || !net.node_ok(node) {
        return None;
    }
    let a = Side::reached(net, stops);
    let b = Side::reached(net, &[node]);
    if a.nodes.is_empty() || b.nodes.is_empty() {
        return None;
    }
    let in_a: std::collections::HashSet<u32> = a.nodes.iter().copied().collect();
    if b.nodes.iter().any(|n| in_a.contains(n)) {
        return None;
    }
    let mut best: Option<([f64; 4], f64)> = None;
    for (from, to, flip) in [(&b, &a, false), (&a, &b, true)] {
        for n in from.ends(net) {
            let (x, y) = (net.node_x[n as usize], net.node_y[n as usize]);
            let Some((p, d)) = to.nearest(net, x, y) else { continue };
            if best.is_none_or(|b| d < b.1) {
                best = Some((if flip { [x, y, p[0], p[1]] } else { [p[0], p[1], x, y] }, d));
            }
        }
    }
    best.map(|b| b.0)
}

#[wasm_bindgen]
impl TrackApi {
    /// T-1896: where the gap is between line `line` and a station `node` no track joins it to:
    /// `[lon a, lat a, lon b, lat b]`, a on the line's side, or empty when there is no gap to show.
    pub fn stop_gap(&self, line: u32, node: u32) -> Vec<f64> {
        let net = &self.w.net;
        if !net.line_ok(line) {
            return vec![];
        }
        let Some([ax, ay, bx, by]) = gap_between(net, net.line_stops(line), node) else { return vec![] };
        let f = net.frame();
        let (lon_a, lat_a) = f.to_lonlat(ax, ay);
        let (lon_b, lat_b) = f.to_lonlat(bx, by);
        vec![lon_a, lat_a, lon_b, lat_b]
    }
}
