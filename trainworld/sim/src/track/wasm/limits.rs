//! A speed limit on a piece of track (T-3405): the player's, free, on any piece but a crossover.

use super::*;

#[wasm_bindgen]
impl TrackApi {
    /// The pieces with a speed limit of their own, as pairs: edge, km/h.
    pub fn edge_limits(&self) -> Vec<u32> {
        let n = &self.w.net;
        (0..n.edge_count() as u32).filter(|&e| n.edge_ok(e) && n.edge_limit[e as usize] > 0).flat_map(|e| [e, n.edge_limit[e as usize] as u32]).collect()
    }

    /// T-3893: the pieces with a ramp length of their own (`EdgeData::ramp` not 0), as pairs: edge, m. Pieces join
    /// across a joint only with equal ramp, so the page's guess at a platform's room needs it.
    pub fn edge_ramps(&self) -> Vec<u32> {
        let n = &self.w.net;
        (0..n.edge_count() as u32).filter(|&e| n.edge_ok(e) && n.edge_ramp[e as usize] > 0).flat_map(|e| [e, n.edge_ramp[e as usize] as u32]).collect()
    }

    /// Hold trains on these pieces to `kmh` (at least 40; 0, or the kind of track's top speed or more,
    /// takes the limit off, T-3525). Free and one undoable edit, constructed track or blueprint; refused
    /// on a crossover. Returns 1 when nothing changed.
    pub fn set_edges_limit(&mut self, ids: &[u32], kmh: u32) -> u32 {
        let mut ops = vec![];
        for &e in ids {
            let Some(mut d) = self.w.net.edge_data(e) else { return self.refuse(Issue::NoSuchEdge { edge: e }) };
            if d.crossover {
                return self.refuse(Issue::Crossover { edge: e });
            }
            let lim = crate::track::net::piece_limit(kmh.min(u16::MAX as u32) as u16, d.kind, false);
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
