//! T-4080: kept bus stops (`bus::keep`) through the clock worker: naming a bus stop, connecting it to a station or
//! another bus stop, and which kept stop a bus stop is. Each edit is one undo step and free; a place with a kept stop
//! within `KEEP_M` edits that one, so two are never made at one place.

use super::*;
use crate::track::bus::{self, keep::{KeepLink, KeptStop, KEEP_M, LINK_MAX_M}};

/// Whether kept stop `id` in `keeps` (the network's as an edit would leave them) has no name and no connection
/// that holds: such a kept stop goes in that edit.
fn keep_empty(net: &Network, keeps: &[Option<KeptStop>], id: u32) -> bool {
    let live = |l: &KeepLink| match *l {
        KeepLink::Station(n) => net.is_station(n),
        KeepLink::Keep(k) => k != id && keeps.get(k as usize).is_some_and(|p| p.is_some()),
    };
    let Some(Some(k)) = keeps.get(id as usize) else { return false };
    k.name.trim().is_empty() && !k.links.iter().any(live) && !keeps.iter().enumerate().any(|(o, p)| o as u32 != id && p.as_ref().is_some_and(|p| p.links.contains(&KeepLink::Keep(id))))
}

#[wasm_bindgen]
impl TrackApi {
    /// The kept stops, 3 numbers each: id, x, y.
    pub fn keeps(&self) -> Vec<f64> {
        let net = &self.w.net;
        net.keep_ids().flat_map(|k| { let p = net.keep(k).unwrap(); [k as f64, p.x, p.y] }).collect()
    }
    /// A kept stop's name; "" for none, or no such kept stop.
    pub fn keep_name(&self, id: u32) -> String {
        self.w.net.keep(id).map_or(String::new(), |k| k.name.clone())
    }
    /// The kept stop nearest (x, y) within `KEEP_M`; `u32::MAX` for none.
    pub fn keep_at(&self, x: f64, y: f64) -> u32 {
        self.w.net.keep_near(x, y).unwrap_or(u32::MAX)
    }
    /// The kept stop bus stop `stop` (an id `bus_stops` gives) is, as its line's service places it; `u32::MAX`
    /// for none, a station's stop, or no such stop.
    pub fn bus_stop_keep(&self, stop: u32) -> u32 {
        if !bus::is_stop_id(stop) {
            return u32::MAX;
        }
        let (line, k) = ((stop & !bus::STOP_ID_BASE) >> 10, (stop & 1023) as usize);
        let Some(plan) = self.w.svc.lines.get(line as usize).and_then(|l| l.bus.as_ref()) else { return u32::MAX };
        plan.stops.get(k).filter(|s| s.id == stop).and_then(|s| s.keep).unwrap_or(u32::MAX)
    }
    /// The kept stop of each of a bus line's stops, in `bus_stops`' order: its id, or -1.
    pub fn bus_stop_keeps(&self, line: u32) -> Vec<f64> {
        let Some(plan) = self.w.svc.lines.get(line as usize).and_then(|l| l.bus.as_ref()) else { return vec![] };
        plan.stops.iter().map(|s| s.keep.map_or(-1.0, |k| k as f64)).collect()
    }
    /// Changes with every change to the kept stops, a name or a connection included (an edit, its undo or redo, a
    /// load): read them again when it moves.
    pub fn keep_epoch(&self) -> u32 {
        self.w.net.keep_epoch
    }
    /// The kept stop the last `name_keep` or `connect_keep` named or connected from (`u32::MAX`: none, or it went).
    pub fn last_keep(&self) -> u32 {
        self.last_keep
    }
    /// A kept stop's connections that hold, made from either side, 2 numbers each: kind (0 a station, 1 a kept stop)
    /// and the station's node or the kept stop's id.
    pub fn keep_links(&self, id: u32) -> Vec<u32> {
        self.w.net.keep_links(id).into_iter().flat_map(|l| match l { KeepLink::Station(n) => [0, n], KeepLink::Keep(k) => [1, k] }).collect()
    }
    /// The kept stops connected to station `node`.
    pub fn station_keeps(&self, node: u32) -> Vec<u32> {
        self.w.net.station_keeps(node)
    }

    /// Name kept stop `id`, or the bus stop at (x, y) with `id` `u32::MAX` (its kept stop within `KEEP_M`, else a
    /// new one there). An empty name clears it, and a kept stop left with no name and no connection goes. 0 done,
    /// or nothing to change; else refused.
    pub fn name_keep(&mut self, id: u32, x: f64, y: f64, name: String) -> u32 {
        self.last_keep = u32::MAX;
        let net = &self.w.net;
        let name = if name.trim().is_empty() { String::new() } else { name };
        let id = if id == u32::MAX { net.keep_near(x, y).unwrap_or(u32::MAX) } else { id };
        if id == u32::MAX {
            if name.is_empty() {
                return 0;
            }
            let k = self.w.net.alloc_keep();
            let r = self.run(Op::Keep { id: k, data: Some(KeptStop { x: quant(x), y: quant(y), name, links: vec![] }) });
            if r == 0 { self.last_keep = k } else { self.w.net.release_keep(k) }
            return r;
        }
        let Some(old) = net.keep(id).cloned() else { return self.refuse(Issue::NoSuchKeep { keep: id }) };
        if old.name == name {
            self.last_keep = id;
            return 0;
        }
        let mut after = net.keeps.clone();
        after[id as usize] = Some(KeptStop { name, ..old });
        let gone = keep_empty(net, &after, id);
        let r = self.run(Op::Keep { id, data: if gone { None } else { after[id as usize].take() } });
        if r == 0 && !gone { self.last_keep = id }
        r
    }

    /// Connect kept stop `id` (`u32::MAX`: the bus stop at (x, y), as `name_keep`) to a station (`kind` 0, `target`
    /// its node) or another kept stop (`kind` 1, `target` its id, or `u32::MAX` for the bus stop at (tx, ty)), up
    /// to `LINK_MAX_M` apart. 0 done; refused where the target is no station or kept stop, the same kept stop (or one
    /// within `KEEP_M`), already connected or too far.
    pub fn connect_keep(&mut self, id: u32, x: f64, y: f64, kind: u8, target: u32, tx: f64, ty: f64) -> u32 {
        self.last_keep = u32::MAX;
        let net = &self.w.net;
        let id = if id == u32::MAX { net.keep_near(x, y).unwrap_or(u32::MAX) } else { id };
        let from = match net.keep(id) {
            Some(k) => k.clone(),
            None if id == u32::MAX => KeptStop { x: quant(x), y: quant(y), name: String::new(), links: vec![] },
            None => return self.refuse(Issue::NoSuchKeep { keep: id }),
        };
        // the target: a station, a kept stop, or a new kept stop at (tx, ty)
        let (to, at) = match kind {
            0 if net.is_station(target) => (Some(KeepLink::Station(target)), (net.node_x[target as usize], net.node_y[target as usize])),
            0 => return self.refuse(Issue::NoSuchNode { node: target }),
            1 => match if target == u32::MAX { net.keep_near(tx, ty) } else { Some(target) } {
                Some(t) => match net.keep(t) {
                    Some(p) => (Some(KeepLink::Keep(t)), (p.x, p.y)),
                    None => return self.refuse(Issue::NoSuchKeep { keep: t }),
                },
                None => (None, (quant(tx), quant(ty))),
            },
            _ => return self.refuse(Issue::NoSuchKeep { keep: target }),
        };
        let far = (at.0 - from.x).hypot(at.1 - from.y) / net.k_at(0.5 * (at.1 + from.y));
        // a kept stop at the same place is the same stop
        let same = (id != u32::MAX && to == Some(KeepLink::Keep(id))) || (kind == 1 && far <= KEEP_M);
        let already = id != u32::MAX && to.is_some_and(|l| net.keep_links(id).contains(&l));
        if same || already || !(far <= LINK_MAX_M) {
            return self.refuse(Issue::NoSuchKeep { keep: id });
        }
        let mut made = vec![];
        let mut ops = vec![];
        let to = match to {
            Some(l) => l,
            None => {
                let k = self.w.net.alloc_keep();
                made.push(k);
                ops.push(Op::Keep { id: k, data: Some(KeptStop { x: at.0, y: at.1, name: String::new(), links: vec![] }) });
                KeepLink::Keep(k)
            }
        };
        let src = if id == u32::MAX {
            let k = self.w.net.alloc_keep();
            made.push(k);
            k
        } else {
            id
        };
        let mut data = from;
        data.links.push(to);
        ops.push(Op::Keep { id: src, data: Some(data) });
        let r = self.run(Op::Batch(ops));
        if r == 0 {
            self.last_keep = src;
        } else {
            for k in made { self.w.net.release_keep(k) }
        }
        r
    }

    /// Remove the connection between kept stop `id` and a station (`kind` 0) or kept stop (`kind` 1), made from
    /// either side; a kept stop it leaves with no name and no connection goes. 0 done; 1 if they were not connected.
    pub fn disconnect_keep(&mut self, id: u32, kind: u8, target: u32) -> u32 {
        let net = &self.w.net;
        let link = match kind {
            0 => KeepLink::Station(target),
            1 => KeepLink::Keep(target),
            _ => return 1,
        };
        let mut after = net.keeps.clone();
        let mut changed = vec![];
        if let Some(Some(p)) = after.get_mut(id as usize).filter(|p| p.as_ref().is_some_and(|p| p.links.contains(&link))) {
            p.links.retain(|l| *l != link);
            changed.push(id);
        }
        if kind == 1 {
            if let Some(Some(p)) = after.get_mut(target as usize).filter(|p| p.as_ref().is_some_and(|p| p.links.contains(&KeepLink::Keep(id)))) {
                p.links.retain(|l| *l != KeepLink::Keep(id));
                changed.push(target);
            }
        }
        if changed.is_empty() {
            return 1;
        }
        let ends: Vec<u32> = if kind == 1 { vec![id, target] } else { vec![id] };
        let gone: Vec<u32> = ends.into_iter().filter(|&k| keep_empty(net, &after, k)).collect();
        for &k in &gone {
            if !changed.contains(&k) { changed.push(k) }
        }
        let ops = changed.into_iter().map(|k| Op::Keep { id: k, data: if gone.contains(&k) { None } else { after[k as usize].clone() } }).collect();
        self.run(Op::Batch(ops))
    }
}
