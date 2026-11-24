//! T-4080 (design notes/T-4080-design.md): kept bus stops. A plain bus stop's id lasts only until the next edit
//! places its line's stops again, so what the player names or connects is a point: a kept stop, with its name (or
//! none) and its connections. Every plain bus stop within `KEEP_M` of it is that stop, and a line that places its
//! own stops holds one at the route's point nearest it, as at a station (`plan_by`). Changed only by `Op::Keep`
//! (undoable), saved in `LBK1`.

use super::super::net::Network;
use super::{Poly, CROSS_NEAR_M};

/// A plain bus stop this near a kept stop (ground m) is that stop: `CROSS_NEAR_M`, half demand's 100 m merge, so
/// the stops of different lines one kept stop takes are always one place for demand.
pub const KEEP_M: f64 = CROSS_NEAR_M;
/// The farthest a connection reaches, ground m (the app's `LINK_MAX_M`, a passage's reach).
pub const LINK_MAX_M: f64 = 1000.0;

/// What a kept stop is connected to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KeepLink {
    /// a station, by node: no longer one once the station is removed (`Network::link_live`)
    Station(u32),
    /// another kept stop, by id
    Keep(u32),
}

/// A kept stop: its place (plane, on the 1 mm grid: the bus stop's place when it was made), the name the player
/// gave it ("" none) and the connections made from it; one made from another kept stop is listed there.
#[derive(Clone, Debug, PartialEq)]
pub struct KeptStop {
    pub x: f64,
    pub y: f64,
    pub name: String,
    pub links: Vec<KeepLink>,
}

/// The kept stops within `KEEP_M` of a route, once per pass (`Poly::project_passes`, as stations): (road m along
/// it, id), in road order.
pub(super) fn near_route(net: &Network, poly: &Poly) -> Vec<(f64, u32)> {
    if net.keeps.is_empty() || poly.pts.len() < 2 {
        return vec![];
    }
    let bb = poly.bbox();
    let pad = KEEP_M * net.k_at(0.5 * (bb[1] + bb[3])) * 1.1 + 1.0;
    let mut out = vec![];
    for id in net.keep_ids() {
        let Some(k) = net.keep(id) else { continue };
        if k.x < bb[0] - pad || k.x > bb[2] + pad || k.y < bb[1] - pad || k.y > bb[3] + pad {
            continue;
        }
        for (s, d) in poly.project_passes(net, k.x, k.y) {
            if d <= KEEP_M {
                out.push((s, id));
            }
        }
    }
    out.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    out
}

/// Of the kept stops `ids`, the one nearest (x, y) within `KEEP_M` ground metres.
pub fn nearest_of(net: &Network, ids: &[u32], x: f64, y: f64) -> Option<u32> {
    let k = net.k_at(y);
    ids.iter()
        .filter_map(|&id| net.keep(id).map(|p| (id, (p.x - x).hypot(p.y - y) / k)))
        .filter(|c| c.1 <= KEEP_M)
        .min_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)))
        .map(|c| c.0)
}

