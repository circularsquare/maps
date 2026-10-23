//! T-1056 measurement, scratch copy only. `T1056=1` counts the rail trips whose walking in all
//! (to the station, between stations and platforms, from the station) is longer than the straight
//! line between the two ends; `T1056=2` also takes those station pairs out of each trip's choice.
use super::kernel::{departures, Graph, Params, Skims};
use std::sync::Mutex;

const NONE: u32 = u32::MAX;

pub fn mode() -> u8 {
    std::env::var("T1056").ok().and_then(|v| v.parse().ok()).unwrap_or(0)
}

/// Expected straight metres walked between stations and platforms, station to station, S x S.
pub static XW: Mutex<Vec<f32>> = Mutex::new(Vec::new());
/// Per layer (0 commutes, 1 everyday) and period: rail, rail walking further; then the same two by
/// the direct distance's band (`BANDS`).
static OUT: Mutex<Vec<[f64; 12]>> = Mutex::new(Vec::new());

pub const BANDS: [f32; 4] = [500.0, 1000.0, 2000.0, 5000.0];
pub fn band(d: f32) -> usize {
    BANDS.iter().position(|&e| d < e).unwrap_or(4)
}

pub fn put(layer: usize, period: usize, acc: [f64; 12]) {
    let mut o = OUT.lock().unwrap();
    let i = layer * 16 + period;
    if o.len() <= i {
        o.resize(i + 1, [0.0; 12]);
    }
    o[i] = acc;
}

pub fn get(layer: usize, period: usize) -> [f64; 12] {
    OUT.lock().unwrap().get(layer * 16 + period).copied().unwrap_or([0.0; 12])
}

pub fn set_xw(g: &Graph, p: &Params, sk: &Skims) {
    let (s, n) = (g.n_st, g.n);
    let mut xw = vec![f32::INFINITY; s * s];
    let mut dep = vec![];
    let to_m = |cost: f32| cost / p.w_xfer * p.xfer_mps * 60.0 / p.walk_detour;
    for dst in 0..s {
        let slot = sk.slot[dst];
        if slot == NONE {
            continue;
        }
        let st = sk.strategy(slot as usize);
        let next = st.next.to_vec();
        let order = st.order.to_vec();
        let cn = st.change;
        let mut wm = vec![f32::INFINITY; n + cn.len()];
        let on_train = |v: usize| (next[v] as usize) >= s && (next[v] as usize) < n;
        for &id in &order {
            let id = id as usize;
            if id == dst {
                wm[id] = 0.0;
                continue;
            }
            if id < s || id >= n {
                departures(g, st.board, cn, id, &mut dep);
                let (mut f, mut w) = (0f32, 0f32);
                for &(e, a) in dep.iter() {
                    let r = g.adj_to[e as usize] as usize - s;
                    let k = if on_train(s + r) { wm[s + r] } else { wm[g.rn_next[r] as usize] };
                    if k.is_finite() {
                        f += a;
                        w += a * k;
                    }
                }
                if f > 0.0 {
                    wm[id] = w / f;
                }
            } else {
                let nx = next[id] as usize;
                if nx == NONE as usize {
                    continue;
                }
                if on_train(id) {
                    wm[id] = wm[nx];
                    continue;
                }
                let r = id - s;
                let here = g.rn_station[r] as usize;
                let to = if nx < s { nx } else { cn.st[nx - n] as usize };
                let walk = if to != here {
                    (g.adj_off[id] as usize..g.adj_off[id + 1] as usize).find(|&e| g.adj_to[e] as usize == to).map_or(0.0, |e| to_m(g.adj_cost[e]))
                } else {
                    0.0
                };
                let at = if to == here { (g.rn_dx[r], g.rn_dy[r]) } else { (0.0, 0.0) };
                let mut plat = 0.0;
                if nx != dst {
                    departures(g, st.board, cn, nx, &mut dep);
                    let (mut f, mut w) = (0f32, 0f32);
                    for &(e, a) in dep.iter() {
                        let r2 = g.adj_to[e as usize] as usize - s;
                        f += a;
                        w += a * (g.rn_dx[r2] - at.0).hypot(g.rn_dy[r2] - at.1);
                    }
                    if f > 0.0 {
                        plat = w / f;
                    }
                }
                wm[id] = walk + plat + wm[nx];
            }
        }
        for a in 0..s {
            xw[a * s + dst] = wm[a];
        }
    }
    *XW.lock().unwrap() = xw;
}
