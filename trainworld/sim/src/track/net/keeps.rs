//! T-4080: the kept bus stops (`bus::keep`): ids, the setter `Op::Keep` applies, and the reads.

use super::*;
use super::bus::keep::{KeepLink, KeptStop};

impl Network {
    pub fn keep(&self, id: u32) -> Option<&KeptStop> {
        self.keeps.get(id as usize).and_then(|k| k.as_ref())
    }
    /// The kept stops' ids, ascending.
    pub fn keep_ids(&self) -> impl Iterator<Item = u32> + '_ {
        self.keeps.iter().enumerate().filter(|(_, k)| k.is_some()).map(|(i, _)| i as u32)
    }
    /// A free id for a new kept stop (claimed when the op adding it is applied; `release_keep` if it is not).
    pub fn alloc_keep(&mut self) -> u32 {
        self.keep_free.pop().unwrap_or_else(|| {
            self.keeps.push(None);
            self.keeps.len() as u32 - 1
        })
    }
    pub fn release_keep(&mut self, id: u32) {
        if self.keep(id).is_none() && (id as usize) < self.keeps.len() && !self.keep_free.contains(&id) {
            self.keep_free.push(id);
        }
    }

    /// Replace kept stop `id` whole (None removes it); the old one comes back. A kept stop made, removed or
    /// moved places again the stops of the bus lines passing near (`Touch::keeps`); a name or connections alone
    /// change no stop.
    pub fn set_keep(&mut self, id: u32, data: Option<KeptStop>) -> Result<Option<KeptStop>, Issue> {
        let old = self.keep(id).cloned();
        self.keep_epoch = self.keep_epoch.wrapping_add(1);
        match data {
            None => {
                let Some(o) = &old else { return Ok(None) };
                self.touch.keeps.push([o.x, o.y]);
                self.keeps[id as usize] = None;
                self.keep_free.push(id);
            }
            Some(d) => {
                if !d.x.is_finite() || !d.y.is_finite() || !self.inside_world(d.x, d.y) || id == u32::MAX {
                    return Err(Issue::NoSuchKeep { keep: id });
                }
                while self.keeps.len() <= id as usize {
                    self.keeps.push(None);
                    let n = self.keeps.len() as u32 - 1;
                    if n != id {
                        self.keep_free.push(n);
                    }
                }
                self.keep_free.retain(|&k| k != id);
                match &old {
                    Some(o) if (o.x, o.y) == (d.x, d.y) => {}
                    Some(o) => self.touch.keeps.extend([[o.x, o.y], [d.x, d.y]]),
                    None => self.touch.keeps.push([d.x, d.y]),
                }
                self.keeps[id as usize] = Some(d);
            }
        }
        Ok(old)
    }

    /// Whether a connection holds now: a station's while it is one, a kept stop's while it is there.
    pub fn link_live(&self, l: KeepLink) -> bool {
        match l {
            KeepLink::Station(n) => self.is_station(n),
            KeepLink::Keep(k) => self.keep(k).is_some(),
        }
    }
    /// Kept stop `id`'s connections that hold, made from either side: its own, then the kept stops listing it.
    pub fn keep_links(&self, id: u32) -> Vec<KeepLink> {
        let Some(k) = self.keep(id) else { return vec![] };
        let mut out: Vec<KeepLink> = k.links.iter().copied().filter(|&l| self.link_live(l) && l != KeepLink::Keep(id)).collect();
        for o in self.keep_ids().filter(|&o| o != id) {
            if self.keep(o).is_some_and(|p| p.links.contains(&KeepLink::Keep(id))) && !out.contains(&KeepLink::Keep(o)) {
                out.push(KeepLink::Keep(o));
            }
        }
        out
    }
    /// The kept stops connected to station `node`.
    pub fn station_keeps(&self, node: u32) -> Vec<u32> {
        if !self.is_station(node) {
            return vec![];
        }
        self.keep_ids().filter(|&k| self.keep(k).is_some_and(|p| p.links.contains(&KeepLink::Station(node)))).collect()
    }
    /// The kept stop nearest (x, y) within `KEEP_M` ground metres.
    pub fn keep_near(&self, x: f64, y: f64) -> Option<u32> {
        let ids: Vec<u32> = self.keep_ids().collect();
        bus::keep::nearest_of(self, &ids, x, y)
    }

    /// T-4080: at a load, after every line's stops are placed: the kept stops no bus stop took (`taken`) go, with
    /// the connections to them. None of them holds a stop, so no line's stops move.
    pub fn drop_keeps_except(&mut self, taken: &[u32]) -> usize {
        let gone: Vec<u32> = self.keep_ids().filter(|k| !taken.contains(k)).collect();
        if !gone.is_empty() {
            self.keep_epoch = self.keep_epoch.wrapping_add(1);
        }
        for &k in &gone {
            self.keeps[k as usize] = None;
            self.keep_free.push(k);
        }
        for p in self.keeps.iter_mut().flatten() {
            p.links.retain(|l| !matches!(l, KeepLink::Keep(k) if gone.contains(k)));
        }
        gone.len()
    }
}
