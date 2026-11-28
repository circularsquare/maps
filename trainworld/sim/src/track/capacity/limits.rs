//! The track limits of each line (`Capacity::line_limits`) kept per line, with the index from a line to the limiting
//! resources it uses (T-4309; the "B1" index of notes/T-4267.md: later stages that work by the lines an edit reaches
//! read `Capacity::resources_of_line`).
//!
//! A line's limit is the tightest over the limiting resources it uses, and a resource's figure for a line depends on
//! the resource's users and on those lines' rates, so it moves when a resource the line uses is built or dropped
//! (`Capacity::update`) or when a line sharing one has its rates changed. Nothing else is read: the rates and the
//! one-train test (`occ_at`) are checked against what the last pass saw, so a caller that changes a line's rates
//! without telling the capacity still gets a right answer.

use super::*;
use std::cell::RefCell;

/// What `occ_at` and the rate read of a line, per slot: its trains an hour, and whether it has one train or none.
type Key = ([f32; SLOTS], [bool; SLOTS]);

#[derive(Clone, Debug, Default)]
pub(super) struct LineIndex {
    built: bool,
    /// per line, the limiting resources it uses (in no order)
    by_line: Vec<Vec<u32>>,
    /// per line, its limits as of the last pass; stale where `dirty`
    most: Vec<[(f64, u32); SLOTS]>,
    key: Vec<Key>,
    dirty: Vec<bool>,
    slots: [bool; SLOTS],
}

pub(super) type Index = RefCell<LineIndex>;

impl LineIndex {
    fn resize(&mut self, n: usize) {
        self.by_line.resize(n, vec![]);
        self.most.resize(n, [(f64::INFINITY, u32::MAX); SLOTS]);
        self.key.resize(n, ([f32::NAN; SLOTS], [false; SLOTS]));
        self.dirty.resize(n, true);
    }
    fn mark(&mut self, l: u32) {
        let l = l as usize;
        if l >= self.dirty.len() {
            self.resize(l + 1);
        }
        self.dirty[l] = true;
    }
}

fn key_of(net: &Network, l: usize) -> Key {
    let tph = net.line_tph.get(l).copied().unwrap_or([0.0; SLOTS]);
    let one = std::array::from_fn(|s| net.line_trains.get(l).is_some_and(|t| t.is_some()) && net.slot_worked(s) && net.trains_most(l, s) <= 1);
    (tph, one)
}

fn lines_of(us: &[User]) -> impl Iterator<Item = u32> + '_ {
    us.iter().enumerate().filter(|&(i, u)| !us[..i].iter().any(|x| x.line == u.line)).map(|(_, u)| u.line)
}

impl Capacity {
    /// The limiting resources line `l` uses (indices into `res`, valid until the next update), in no order. The
    /// index is built on first use and kept by `update`.
    pub fn resources_of_line(&self, net: &Network, l: u32) -> Vec<u32> {
        self.ensure_index(net.line_count());
        self.lim.borrow().by_line.get(l as usize).cloned().unwrap_or_default()
    }

    pub(crate) fn ensure_index(&self, n: usize) {
        let mut ix = self.lim.borrow_mut();
        if ix.built && ix.by_line.len() == n {
            return;
        }
        if !ix.built {
            *ix = LineIndex::default();
            ix.resize(n);
            for r in (0..self.res.len()).filter(|&r| self.limits(r)) {
                for l in lines_of(self.users_of(r)) {
                    ix.mark(l);
                    ix.by_line[l as usize].push(r as u32);
                }
            }
            ix.built = true;
        } else {
            // the number of lines moved: every line's limits are worked out again
            ix.resize(n);
            ix.by_line.truncate(n);
            ix.most.truncate(n);
            ix.key.truncate(n);
            ix.dirty.truncate(n);
            ix.dirty.iter_mut().for_each(|d| *d = true);
        }
    }

    /// Resources `first_new..` were just built: their users' limits are to be worked out again.
    pub(super) fn index_attach(&mut self, first_new: usize) {
        let mut ix = std::mem::take(self.lim.get_mut());
        if ix.built {
            for r in (first_new..self.res.len()).filter(|&r| self.limits(r)) {
                for l in lines_of(self.users_of(r)) {
                    ix.mark(l);
                    ix.by_line[l as usize].push(r as u32);
                }
            }
        }
        *self.lim.get_mut() = ix;
    }

    /// Resource `r` is about to be dropped (`swap_remove`: the last resource takes its place).
    pub(super) fn index_drop(&mut self, r: usize) {
        let mut ix = std::mem::take(self.lim.get_mut());
        if ix.built {
            let last = self.res.len() - 1;
            if self.limits(r) {
                for l in lines_of(self.users_of(r)) {
                    ix.mark(l);
                    ix.by_line[l as usize].retain(|&x| x != r as u32);
                }
            }
            if r != last && self.limits(last) {
                for l in lines_of(self.users_of(last)) {
                    let l = l as usize;
                    if let Some(x) = ix.by_line[l].iter_mut().find(|x| **x == last as u32) {
                        *x = r as u32;
                    }
                    for m in ix.most[l].iter_mut().filter(|m| m.1 == last as u32) {
                        m.1 = r as u32;
                    }
                }
            }
        }
        *self.lim.get_mut() = ix;
    }

    /// Per line and level, the most trains an hour it can run (`line_limits_lone` with `lone`), worked out again only
    /// for the lines whose limits can have moved since the last call.
    pub fn line_limits(&self, net: &Network) -> Vec<[(f64, u32); SLOTS]> {
        let n = net.line_count();
        self.ensure_index(n);
        let out = self.limits_kept(net, n);
        #[cfg(test)]
        assert!(out == self.line_limits_lone(net, true), "the kept track limits differ from the whole pass");
        out
    }

    pub(crate) fn limits_kept(&self, net: &Network, n: usize) -> Vec<[(f64, u32); SLOTS]> {
        let mut ix = self.lim.borrow_mut();
        let slots: [bool; SLOTS] = std::array::from_fn(|s| net.slot_worked(s));
        if ix.slots != slots {
            ix.slots = slots;
            ix.dirty.iter_mut().for_each(|d| *d = true);
        }
        // a line whose rates moved moves the limits of every line that shares a resource with it
        for l in 0..n {
            let k = key_of(net, l);
            if k != ix.key[l] {
                ix.key[l] = k;
                ix.dirty[l] = true;
                let rs = std::mem::take(&mut ix.by_line[l]);
                for &r in &rs {
                    for u in self.users_of(r as usize) {
                        ix.dirty[u.line as usize] = true;
                    }
                }
                ix.by_line[l] = rs;
            }
        }
        let todo: Vec<usize> = (0..n).filter(|&l| ix.dirty[l]).collect();
        if !todo.is_empty() {
            let mut rs: Vec<u32> = if todo.len() == n {
                (0..self.res.len() as u32).filter(|&r| self.limits(r as usize)).collect()
            } else {
                let mut v: Vec<u32> = todo.iter().flat_map(|&l| ix.by_line[l].iter().copied()).collect();
                v.sort_unstable();
                v.dedup();
                v
            };
            rs.sort_unstable();
            for &l in &todo {
                ix.most[l] = [(f64::INFINITY, u32::MAX); SLOTS];
            }
            let mut own = vec![];
            for &r in &rs {
                self.each_limit_at(net, true, r as usize, &mut own, |l, lev, most, r| {
                    if ix.dirty[l as usize] {
                        let b = &mut ix.most[l as usize][lev];
                        if most < b.0 {
                            *b = (most, r as u32);
                        }
                    }
                });
            }
            for &l in &todo {
                ix.dirty[l] = false;
            }
        }
        ix.most.clone()
    }

    /// `each_limit` over the resources of the lines `want` only (`at` gets every line of those resources).
    pub(super) fn each_limit_of(&self, net: &Network, lone: bool, want: &[u32], mut at: impl FnMut(u32, usize, f64, usize)) {
        self.ensure_index(net.line_count());
        let mut rs: Vec<u32> = {
            let ix = self.lim.borrow();
            want.iter().flat_map(|&l| ix.by_line.get(l as usize).into_iter().flatten().copied()).collect()
        };
        rs.sort_unstable();
        rs.dedup();
        let mut own = vec![];
        for &r in &rs {
            self.each_limit_at(net, lone, r as usize, &mut own, &mut at);
        }
    }
}

/// T-4309: on the real New York save (trainworld/T-007-real-nyc.save), the kept limits are the whole pass's to the bit,
/// at load and after the rates of a few lines move, and what each costs. On request, on the save's track bytes
/// (gunzipped, after its JSON header): `T4309_TRACK=<file> cargo test --profile agent limits_on_the_real_save -- --ignored --nocapture`.
#[cfg(test)]
#[test]
#[ignore]
fn limits_on_the_real_save() {
    use super::super::{save, world::TrackWorld};
    use std::time::Instant;
    let mut w = TrackWorld::new(save::decode(&std::fs::read(std::env::var("T4309_TRACK").unwrap()).unwrap()).unwrap());
    let cap = &w.svc.cap;
    let t = Instant::now();
    let whole = cap.line_limits_lone(&w.net, true);
    let t_whole = t.elapsed();
    // (`line_limits` itself also runs the whole pass in tests, to compare: the times are of the kept pass alone)
    let n = w.net.line_count();
    cap.ensure_index(n);
    let t = Instant::now();
    let first = cap.limits_kept(&w.net, n);
    let t_first = t.elapsed();
    assert!(first == whole);
    let t = Instant::now();
    let again = cap.limits_kept(&w.net, n);
    let t_again = t.elapsed();
    assert!(again == whole);
    let moved = [3 % n, n / 2, n - 1];
    for &l in &moved {
        for s in w.net.slots().collect::<Vec<_>>() {
            w.net.line_tph[l][s] += 1.5;
        }
    }
    let cap = &w.svc.cap;
    let t = Instant::now();
    let after = cap.limits_kept(&w.net, n);
    let t_moved = t.elapsed();
    assert!(after == cap.line_limits_lone(&w.net, true));
    let changed = (0..n).filter(|&l| after[l] != whole[l]).count();
    eprintln!("lines {n}, resources {}, limiting lines moved {} of {n} by {} rate changes", cap.res.len(), changed, moved.len());
    eprintln!("whole {t_whole:?}, kept first {t_first:?}, kept unchanged {t_again:?}, kept after 3 lines' rates moved {t_moved:?}");
}
