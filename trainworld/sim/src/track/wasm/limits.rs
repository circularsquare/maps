//! A speed limit on a piece of track (T-3405): the player's, free, on any piece but a crossover.

use super::*;

#[wasm_bindgen]
impl TrackApi {
    /// The pieces with a speed limit of their own, as pairs: edge, km/h.
    pub fn edge_limits(&self) -> Vec<u32> {
        let n = &self.w.net;
        (0..n.edge_count() as u32).filter(|&e| n.edge_ok(e) && n.edge_limit[e as usize] > 0).flat_map(|e| [e, n.edge_limit[e as usize] as u32]).collect()
    }

    /// Hold trains on these pieces to `kmh` (40 up to the kind of track's top speed; 0 takes the limit
    /// off). Free and one undoable edit, constructed track or blueprint; refused on a crossover. Returns
    /// 1 when nothing changed.
    pub fn set_edges_limit(&mut self, ids: &[u32], kmh: u32) -> u32 {
        let mut ops = vec![];
        for &e in ids {
            let Some(mut d) = self.w.net.edge_data(e) else { return self.refuse(Issue::NoSuchEdge { edge: e }) };
            if d.crossover {
                return self.refuse(Issue::Crossover { edge: e });
            }
            let top = (d.kind.v_cap() * 3.6).round() as u32;
            let lim = if kmh == 0 { 0 } else { kmh.clamp(EDGE_LIMIT_MIN as u32, top) as u16 };
            if lim != d.limit {
                d.limit = lim;
                ops.push(Op::Edge { id: e, data: Some(d) });
            }
        }
        if ops.is_empty() {
            self.money_short = None;
            self.issues = vec![];
            return 1;
        }
        self.run(Op::Batch(ops))
    }
}
