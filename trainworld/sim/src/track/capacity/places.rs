//! Where each resource sits, kept by the edge, node and line it belongs to (T-4444; the "C2" index of
//! notes/T-4267.md, beside the limiting resources per line in `limits.rs`): `Capacity::update` finds what an edit
//! drops and what a line's rates re-solve by looking these up, not by going over every resource.

use super::*;
use std::cell::RefCell;

#[derive(Clone, Debug, Default)]
pub(super) struct Where {
    built: bool,
    /// per edge: the sections over it and the flat crossings it is one of
    by_edge: Vec<Vec<u32>>,
    /// per node: its junction moves, terminus and platforms
    by_node: Vec<Vec<u32>>,
    /// per line: every resource it is a user of
    by_line: Vec<Vec<u32>>,
    /// per line: its following losses and waits
    by_tag: Vec<Vec<u32>>,
}

pub(super) type Places = RefCell<Where>;

const EDGE: u8 = 0;
const NODE: u8 = 1;
const LINE: u8 = 2;
const TAG: u8 = 3;

impl Where {
    fn lists(&mut self, k: u8) -> &mut Vec<Vec<u32>> {
        match k {
            EDGE => &mut self.by_edge,
            NODE => &mut self.by_node,
            LINE => &mut self.by_line,
            _ => &mut self.by_tag,
        }
    }
    fn list(&mut self, k: u8, i: u32) -> &mut Vec<u32> {
        let v = self.lists(k);
        if v.len() <= i as usize {
            v.resize(i as usize + 1, vec![]);
        }
        &mut v[i as usize]
    }
    fn get(&self, k: u8, i: u32) -> &[u32] {
        let v = match k {
            EDGE => &self.by_edge,
            NODE => &self.by_node,
            LINE => &self.by_line,
            _ => &self.by_tag,
        };
        v.get(i as usize).map_or(&[], |x| &x[..])
    }
}

impl Capacity {
    /// What resource `r` is listed under, each once.
    fn spots(&self, r: usize) -> Vec<(u8, u32)> {
        let mut v: Vec<(u8, u32)> = vec![];
        match self.tag[r] {
            Tag::Section { first, count } => v.extend(self.keys[first as usize..(first + count) as usize].iter().map(|k| (EDGE, k / 2))),
            Tag::Node | Tag::Platform { .. } => v.push((NODE, self.res[r].node)),
            Tag::Crossing { other, .. } => v.extend([(EDGE, self.res[r].edge), (EDGE, other)]),
            Tag::Follow { line } | Tag::Wait { line, .. } => v.push((TAG, line)),
        }
        v.extend(self.users_of(r).iter().map(|u| (LINE, u.line)));
        v.sort_unstable();
        v.dedup();
        v
    }

    pub(super) fn ensure_places(&self) {
        if self.places.borrow().built {
            return;
        }
        let mut w = Where::default();
        for r in 0..self.res.len() {
            for (k, i) in self.spots(r) {
                w.list(k, i).push(r as u32);
            }
        }
        w.built = true;
        *self.places.borrow_mut() = w;
    }

    /// Resources `first_new..` were just built.
    pub(super) fn places_attach(&mut self, first_new: usize) {
        let mut w = std::mem::take(self.places.get_mut());
        if w.built {
            for r in first_new..self.res.len() {
                for (k, i) in self.spots(r) {
                    w.list(k, i).push(r as u32);
                }
            }
        }
        *self.places.get_mut() = w;
    }

    /// Resource `r` is about to be dropped (`swap_remove`: the last resource takes its place).
    pub(super) fn places_drop(&mut self, r: usize) {
        let mut w = std::mem::take(self.places.get_mut());
        if w.built {
            for (k, i) in self.spots(r) {
                w.list(k, i).retain(|&x| x != r as u32);
            }
            let last = self.res.len() - 1;
            if r != last {
                for (k, i) in self.spots(last) {
                    for x in w.list(k, i).iter_mut().filter(|x| **x == last as u32) {
                        *x = r as u32;
                    }
                }
            }
        }
        *self.places.get_mut() = w;
    }

    /// The resources over or at the given edges, nodes: sections over an edge, crossings of it, moves and platforms at
    /// a node; sorted, each once.
    pub(super) fn resources_at(&self, edges: &[u32], nodes: &[u32]) -> Vec<u32> {
        self.ensure_places();
        let w = self.places.borrow();
        let mut v: Vec<u32> = edges.iter().flat_map(|&e| w.get(EDGE, e).iter().copied()).collect();
        v.extend(nodes.iter().flat_map(|&n| w.get(NODE, n).iter().copied()));
        v.sort_unstable();
        v.dedup();
        v
    }

    /// Every resource any of `lines` is a user of, sorted, each once. With `with_tagged`, also the following losses
    /// and waits tagged with one of them.
    pub fn resources_of_lines(&self, lines: impl Iterator<Item = u32>, with_tagged: bool) -> Vec<u32> {
        self.ensure_places();
        let w = self.places.borrow();
        let mut v: Vec<u32> = vec![];
        for l in lines {
            v.extend_from_slice(w.get(LINE, l));
            if with_tagged {
                v.extend_from_slice(w.get(TAG, l));
            }
        }
        v.sort_unstable();
        v.dedup();
        v
    }

    /// Tests: the kept lists are what a fresh pass over the resources gives.
    #[cfg(test)]
    pub(super) fn places_match_fresh(&self) -> bool {
        self.ensure_places();
        let kept = self.places.borrow().clone();
        let mut fresh = Where::default();
        for r in 0..self.res.len() {
            for (k, i) in self.spots(r) {
                fresh.list(k, i).push(r as u32);
            }
        }
        let norm = |w: &Where| {
            [&w.by_edge, &w.by_node, &w.by_line, &w.by_tag].map(|v| {
                let mut v: Vec<Vec<u32>> = v.iter().map(|x| { let mut x = x.clone(); x.sort_unstable(); x }).collect();
                while v.last().is_some_and(|x| x.is_empty()) {
                    v.pop();
                }
                v
            })
        };
        norm(&kept) == norm(&fresh)
    }

    /// The following losses and waits of the lines `drop` says yes to, sorted.
    pub(super) fn tagged_where(&self, mut drop: impl FnMut(u32) -> bool) -> Vec<u32> {
        self.ensure_places();
        let w = self.places.borrow();
        let mut v: Vec<u32> = vec![];
        for (l, rs) in w.by_tag.iter().enumerate() {
            if !rs.is_empty() && drop(l as u32) {
                v.extend_from_slice(rs);
            }
        }
        v.sort_unstable();
        v.dedup();
        v
    }
}

/// T-4444: on the real New York save (trainworld/T-007-real-nyc.save), updating the capacity for three lines whose paths
/// are touched and whose rates moved gives the capacity built whole from scratch, to the bit (resources, users,
/// utilisation, delays, and every line's holds). On request, on the save's track bytes (gunzipped, after its JSON header):
/// `T4444_TRACK=<file> cargo test --profile agent update_on_the_real_save -- --ignored --nocapture`.
#[cfg(test)]
#[test]
#[ignore]
fn update_on_the_real_save() {
    use super::super::{net::path_edge, save, world::TrackWorld};
    use std::time::Instant;
    let mut w = TrackWorld::new(save::decode(&std::fs::read(std::env::var("T4444_TRACK").unwrap()).unwrap()).unwrap());
    let n = w.net.line_count();
    let moved: Vec<u32> = (0..n as u32).filter(|&l| w.svc.lines[l as usize].running).step_by((n / 3).max(1)).take(3).collect();
    let mut edges: Vec<u32> = vec![];
    let mut nodes: Vec<u32> = vec![];
    for &l in &moved {
        for run in &w.svc.lines[l as usize].runs {
            edges.extend(run.segs.iter().map(|&p| path_edge(p)));
            nodes.extend_from_slice(&run.nodes);
        }
        for s in w.net.slots().collect::<Vec<_>>() {
            w.net.line_tph[l as usize][s] += 1.5;
        }
    }
    edges.sort_unstable();
    edges.dedup();
    nodes.sort_unstable();
    nodes.dedup();
    let t = Instant::now();
    let affected = w.svc.cap.update(&w.net, &w.svc.lines, &edges, &nodes, &moved);
    let took = t.elapsed();
    let mut whole = super::build(&w.net, &w.svc.lines);
    super::solve(&mut whole, &w.net);
    let (a, b) = (w.svc.cap.canonical(), whole.canonical());
    assert_eq!(a.len(), b.len(), "resource count");
    assert!(a == b, "the updated capacity differs from the whole build");
    assert!(w.svc.cap.places_match_fresh());
    let (ha, hb) = (super::holds(&w.svc.cap, &w.svc.lines), super::holds(&whole, &w.svc.lines));
    assert!(ha == hb, "holds differ");
    // the holds of only the lines the update says moved are the whole pass's
    let some = super::holds_for(&w.svc.cap, &w.svc.lines, Some(&affected));
    for l in (0..n).filter(|&l| affected[l]) {
        assert!(some[l] == hb[l], "holds_for differs for line {l}");
    }
    eprintln!("lines {n}, resources {}, {} edges and {} nodes touched, {} lines' rates moved, {} lines affected; update {took:?}", w.svc.cap.res.len(), edges.len(), nodes.len(), moved.len(), affected.iter().filter(|&&a| a).count());
}
