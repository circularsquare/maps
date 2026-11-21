//! T-3378 stage 4: the constructed busways (`TrackKind::Busway`) as short straight stretches in a grid, for finding
//! the pace steps of a bus route that run along one (`Network::busways`). The sim decides it from the busways built,
//! never from the bits the bus tool set, so deleting a busway takes its speed away.

use crate::track::geom;
use crate::track::net::Network;
use crate::track::params::{BUSWAY_MATCH_DEG, BUSWAY_MATCH_M};
use std::collections::HashMap;

/// Ground metres between the points a busway is cut into.
const SAMPLE_M: f64 = 10.0;
/// The grid's square, ground metres: more than `BUSWAY_MATCH_M`, so a point looks in its own square only.
const CELL_M: f64 = 50.0;

#[derive(Debug, Default)]
pub struct Busways {
    segs: Vec<[[f64; 2]; 2]>,
    grid: HashMap<(i64, i64), Vec<u32>>,
    /// the grid's square, frame units
    cell: f64,
}

impl Busways {
    /// The constructed busways of `net`.
    pub fn build(net: &Network) -> Busways {
        let mut out = Busways::default();
        for e in net.built_busways() {
            let (pcs, len) = (net.edge_pieces(e), net.edge_len[e as usize]);
            if pcs.is_empty() || len <= 0.0 {
                continue;
            }
            let n = (len / SAMPLE_M).ceil().max(1.0) as usize;
            let at = |j: usize| {
                let (x, y, _) = geom::pos_at(pcs, len * j as f64 / n as f64);
                [x, y]
            };
            let mut prev = at(0);
            for j in 1..=n {
                let p = at(j);
                out.segs.push([prev, p]);
                prev = p;
            }
        }
        let Some(first) = out.segs.first() else { return out };
        out.cell = CELL_M * net.k_at(first[0][1]);
        for (j, s) in out.segs.iter().enumerate() {
            let pad = BUSWAY_MATCH_M * net.k_at(0.5 * (s[0][1] + s[1][1])) * 1.05;
            let (x0, x1) = (s[0][0].min(s[1][0]) - pad, s[0][0].max(s[1][0]) + pad);
            let (y0, y1) = (s[0][1].min(s[1][1]) - pad, s[0][1].max(s[1][1]) + pad);
            let c = out.cell;
            for cx in (x0 / c).floor() as i64..=(x1 / c).floor() as i64 {
                for cy in (y0 / c).floor() as i64..=(y1 / c).floor() as i64 {
                    out.grid.entry((cx, cy)).or_default().push(j as u32);
                }
            }
        }
        out
    }

    pub fn is_empty(&self) -> bool {
        self.segs.is_empty()
    }

    /// Whether a route at (x, y) heading (dx, dy) (frame, any length) runs along a busway: within `BUSWAY_MATCH_M`
    /// of one (`k` frame units a ground metre there) and within `BUSWAY_MATCH_DEG` of its way, either way, so a route
    /// passing under a busway on a bridge does not count.
    pub fn on(&self, x: f64, y: f64, dx: f64, dy: f64, k: f64) -> bool {
        if self.cell <= 0.0 {
            return false;
        }
        let Some(cand) = self.grid.get(&((x / self.cell).floor() as i64, (y / self.cell).floor() as i64)) else { return false };
        let d = dx.hypot(dy);
        let cos = BUSWAY_MATCH_DEG.to_radians().cos();
        let tol = BUSWAY_MATCH_M * k;
        cand.iter().any(|&j| {
            let s = self.segs[j as usize];
            let (ex, ey) = (s[1][0] - s[0][0], s[1][1] - s[0][1]);
            let e = ex.hypot(ey);
            if e <= 0.0 || (d > 0.0 && (ex * dx + ey * dy).abs() < cos * e * d) {
                return false;
            }
            let t = (((x - s[0][0]) * ex + (y - s[0][1]) * ey) / (e * e)).clamp(0.0, 1.0);
            (x - s[0][0] - t * ex).hypot(y - s[0][1] - t * ey) <= tol
        })
    }
}
