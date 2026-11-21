//! Departure spacing (T-1925, notes/T-1900.md D1): each line's departures at a demand level leave
//! a phase after its own pattern's times (`Lattice::phase`), picked so its trains and those of the
//! lines it shares track, a platform or a flat junction with do not reach the place they meet
//! together (and are drawn on top of each other). Real timetabling, done greedily: only the lines
//! an edit changed are placed, against every other line as it runs, so the rest keep their times.
//! Waits are by headway, so demand does not move. A merge, where trains queue in sight, comes
//! first (T-2960). A headway line's reverse departures also move with where it stands its spare
//! time, which is placed with the phase, so both directions can take even turns (T-2972).

use super::capacity::ResKind;
use super::net::Network;
use super::params::SLOTS;
use super::profile;
use super::service::Service;

/// Phases tried per headway.
const STEPS: usize = 24;
/// A line keeps its phase unless another one cuts its crowding by more than this share.
const KEEP: f64 = 0.9;
/// Trains of the line looked at per meeting: an hour's, at most this many.
const MAX_TRAINS: usize = 60;
/// A merge's weight against other meetings (T-2960): so large that they only choose among phases
/// nearly as good at the merge (its trains off by at most 1/26 of what the others pull).
const MERGE_FIRST: f64 = 25.0;
/// T-2972: shares of a headway line's spare time stood at its last stop tried, in this many steps
/// from none to all (half is one of them).
const SPLITS: usize = 8;

/// Where one direction of the line meets one direction of another line: when its trains get
/// there after its pattern's own times (`a`, without its phase), when the other line's do (`b`,
/// with its phase) and the other line's headway.
struct Meet {
    a: f64,
    b: f64,
    hb: f64,
    /// T-2231, T-2419: the two lines are a following pair timetabled together
    /// (`Capacity::follow_targets`): the faster one's trains belong this far (s) after the slower
    /// one's; positive when this line follows, negative when it leads, 0 for any other pair
    behind: f64,
    /// `MERGE_FIRST` at a merge, else 1
    weight: f64,
    /// how far apart its trains are clear of each other, s: a flat junction's crossing time, or
    /// infinite (half the smaller headway)
    clear: f64,
    /// this line's run there: run 1's departures move with where its spare time stands (T-2972)
    run: u8,
}

/// How crowded a phase leaves the line's meetings: per train, how far it is short of half the
/// smaller headway (or of `clear`) from the nearest train of the other line there, squared, as a
/// share, times the meeting's weight. For a following pair timetabled together, how far each
/// following train is from its place after a leading one, as a share of the same half headway, squared.
fn crowding(meets: &[Meet], h: f64, phase: f64, shift: f64) -> f64 {
    let n = ((3600.0 / h).ceil() as usize).clamp(1, MAX_TRAINS);
    let mut sum = 0.0;
    for m in meets {
        // run 1's departures `shift` later than the pattern's (T-2972)
        let phase = if m.run == 1 { phase + shift } else { phase };
        let g = 0.5 * h.min(m.hb);
        if m.behind > 0.0 {
            for k in 0..n {
                let x = (m.a + phase + k as f64 * h - m.b - m.behind).rem_euclid(m.hb);
                sum += (x.min(m.hb - x) / g).powi(2);
            }
            continue;
        }
        if m.behind < 0.0 {
            // the other line's trains each want one of this line's ahead of them
            let nb = ((3600.0 / m.hb).ceil() as usize).clamp(1, MAX_TRAINS);
            for k in 0..nb {
                let x = (m.b + k as f64 * m.hb + m.behind - m.a - phase).rem_euclid(h);
                sum += (x.min(h - x) / g).powi(2);
            }
            continue;
        }
        let g = g.min(m.clear);
        for k in 0..n {
            let x = (m.a + phase + k as f64 * h - m.b).rem_euclid(m.hb);
            let d = x.min(m.hb - x);
            if d < g {
                sum += m.weight * ((g - d) / g).powi(2);
            }
        }
    }
    sum
}

/// Place the departures of `lines` (any order; placed in id order, each against the phases as
/// they then stand) into `phase` and where their spare time stands into `split` (by line id, grown
/// as needed). The lines they meet with spare time then move only where theirs stands, against them,
/// and `lines` are placed once more (T-2972: two branches into one terminus both stand theirs at
/// the far ends, which neither reaches alone). Returns the lines whose phase or split moved.
pub fn place(net: &Network, svc: &Service, lines: &[u32], phase: &mut Vec<[f32; SLOTS]>, split: &mut Vec<[f32; SLOTS]>) -> Vec<u32> {
    phase.resize(phase.len().max(net.line_count()), [0.0; SLOTS]);
    split.resize(split.len().max(net.line_count()), [0.5; SLOTS]);
    // where its spare time would stand with nothing else known: at the end it does not share
    for &l in lines {
        seed_split(net, svc, l, split);
    }
    let mut moved = place_pass(net, svc, lines, phase, split, false);
    let cap = &svc.cap;
    let mut mine = vec![false; net.line_count()];
    for &l in lines {
        if let Some(m) = mine.get_mut(l as usize) { *m = true; }
    }
    let mut others: Vec<u32> = vec![];
    for r in 0..cap.res.len() {
        let us = cap.users_of(r);
        if !us.iter().any(|u| mine.get(u.line as usize).copied().unwrap_or(false)) {
            continue;
        }
        for u in us {
            if !mine[u.line as usize] && net.slots().any(|lev| svc.spare(net, u.line as usize, lev) >= 1.0) {
                others.push(u.line);
            }
        }
    }
    others.sort_unstable();
    others.dedup();
    for &o in &others {
        if seed_split(net, svc, o, split) {
            moved.push(o);
        }
    }
    if !others.is_empty() {
        let o = place_pass(net, svc, &others, phase, split, true);
        if !o.is_empty() {
            moved.extend(o);
            moved.extend(place_pass(net, svc, lines, phase, split, false));
        }
    }
    moved.sort_unstable();
    moved.dedup();
    moved
}

/// T-2972: a line with spare time whose track is shared at one end only stands it at the other (a
/// branch into a shared terminus waits out at its own far end, so the order its trains leave the
/// terminus in is the order they come back in). True when its split moved.
fn seed_split(net: &Network, svc: &Service, l: u32, split: &mut [[f32; SLOTS]]) -> bool {
    // T-3474: the ends of the part of its plan it runs
    let path = match svc.lines.get(l as usize).filter(|x| x.part.is_some()) {
        Some(x) => &x.runs[0].segs[..],
        None => net.line_path(l),
    };
    let (Some(&a), Some(&b)) = (path.first(), path.last()) else { return false };
    let shared = |e: u32| net.lines_on_edge(e).iter().any(|&o| o != l && svc.lines.get(o as usize).is_some_and(|x| x.running && !x.is_bus() && x.runs_on(e)));
    let want = match (shared(a >> 1), shared(b >> 1)) {
        (true, false) => 1.0,
        (false, true) => 0.0,
        _ => return false,
    };
    let mut moved = false;
    for lev in net.slots() {
        if svc.spare(net, l as usize, lev) >= 1.0 && split[l as usize][lev] != want {
            split[l as usize][lev] = want;
            moved = true;
        }
    }
    moved
}

/// One pass of `place`; `split_only`: each line keeps its phase.
fn place_pass(net: &Network, svc: &Service, lines: &[u32], phase: &mut Vec<[f32; SLOTS]>, split: &mut Vec<[f32; SLOTS]>, split_only: bool) -> Vec<u32> {
    let cap = &svc.cap;
    let mut todo: Vec<u32> = lines.iter().copied().filter(|&l| svc.lines.get(l as usize).is_some_and(|x| x.running && !x.is_bus())).collect();
    todo.sort_unstable();
    todo.dedup();
    if todo.is_empty() {
        return vec![];
    }
    let mut want = vec![false; net.line_count()];
    for &l in &todo {
        want[l as usize] = true;
    }
    // the resources several lines meet at, by line of the ones to place: shared track, platforms,
    // junction moves
    let mut by_line: Vec<Vec<usize>> = vec![vec![]; net.line_count()];
    for r in 0..cap.res.len() {
        if !matches!(cap.res[r].kind, ResKind::Section | ResKind::SingleTrack | ResKind::Platform | ResKind::Junction) {
            continue;
        }
        let us = cap.users_of(r);
        if us.iter().all(|u| u.line == us[0].line) {
            continue;
        }
        for u in us {
            if want[u.line as usize] && by_line[u.line as usize].last() != Some(&r) {
                by_line[u.line as usize].push(r);
            }
        }
    }
    // T-2231, T-2419: following pairs timetabled together, by level: (faster line, its run), (slower
    // line, its run), how far behind it belongs (s)
    let paired: [Vec<((u32, u8), (u32, u8), f64)>; SLOTS] = std::array::from_fn(|lev| {
        if !net.slot_worked(lev) {
            return vec![];
        }
        let mut v = cap.follow_targets(net, lev);
        v.retain(|x| want[x.0 .0 as usize] || want[x.1 .0 as usize]);
        v
    });
    // this line's run `ur` against line `v`'s run `vr`: how far behind (+) or ahead (-) it belongs
    let behind = |lev: usize, l: u32, ur: u8, v: u32, vr: u8| {
        paired[lev].iter().find_map(|x| if (x.0, x.1) == ((l, ur), (v, vr)) { Some(x.2) } else if (x.0, x.1) == ((v, vr), (l, ur)) { Some(-x.2) } else { None }).unwrap_or(0.0)
    };
    let time = |l: u32, run: u8, lev: usize, at: f64| profile::time_at(&svc.lines[l as usize].runs[run as usize].prof[lev].phases, at.max(0.0));
    // each line's pattern, worked out once
    let mut lats: Vec<Option<Option<super::service::Lattice>>> = vec![None; net.line_count()];
    for &l in &todo {
        for &r in &by_line[l as usize] {
            for u in cap.users_of(r) {
                let x = &mut lats[u.line as usize];
                if x.is_none() {
                    *x = Some(svc.lattice(net, u.line as usize));
                }
            }
        }
    }
    let mut moved = vec![];
    // (other line, its run, own run), own time, a, b, weight, clear
    let mut first: Vec<((u32, u8, u8), f64, f64, f64, f64, f64)> = vec![];
    for &l in &todo {
        let Some(Some(lat)) = lats[l as usize] else { continue };
        let mut p = phase[l as usize];
        let mut x = split[l as usize];
        for lev in net.slots() {
            let tph = lat.tph[lev] as f64;
            if tph <= 0.0 {
                continue;
            }
            let h = 3600.0 / tph;
            // (the lattices were made with every line's spare time half at each end: `split` is out
            // of the network while it is placed, as `phase` is)
            let base = |lat: &super::service::Lattice, line: u32, run: u8| {
                let share = if line == l { x[lev] } else { split[line as usize][lev] } as f64;
                -lat.lead - lat.shift + if run == 1 { lat.off[lev] + (share - 0.5) * svc.spare(net, line as usize, lev) } else { 0.0 }
            };
            // per pair of directions, the first place they meet along this line's run
            first.clear();
            for &r in &by_line[l as usize] {
                let us = cap.users_of(r);
                let junction = cap.res[r].kind == ResKind::Junction;
                let weight = if cap.wire_kind(r) == 12 { MERGE_FIRST } else { 1.0 };
                for u in us.iter().filter(|u| u.line == l) {
                    let tu = time(l, u.run, lev, u.at);
                    for v in us.iter().filter(|v| v.line != l && (!junction || v.owner != u.owner)) {
                        if net.line_tph[v.line as usize][lev] <= 0.0 || !svc.lines[v.line as usize].running {
                            continue;
                        }
                        let key = (v.line, v.run, u.run);
                        let Some(Some(other)) = &lats[v.line as usize] else { continue };
                        let b = base(other, v.line, v.run) + phase[v.line as usize][lev] as f64 + time(v.line, v.run, lev, v.at);
                        let a = base(&lat, l, u.run) + tu;
                        let clear = if junction { u.occ.max(v.occ) } else { f64::INFINITY };
                        match first.iter_mut().find(|x| x.0 == key) {
                            Some(x) if x.1 <= tu => {}
                            Some(x) => *x = (key, tu, a, b, weight, clear),
                            None => first.push((key, tu, a, b, weight, clear)),
                        }
                    }
                }
            }
            if first.is_empty() {
                continue;
            }
            let meets: Vec<Meet> = first.iter().map(|x| Meet { a: x.2, b: x.3, hb: 3600.0 / net.line_tph[x.0 .0 as usize][lev] as f64, behind: behind(lev, l, x.0 .2, x.0 .0, x.0 .1), weight: x.4, clear: x.5, run: x.0 .2 }).collect();
            let cur = (p[lev] as f64).rem_euclid(h);
            // T-2972: with spare time, the share stood at the last stop is tried too: run 1 leaves
            // (share - its share now) x spare later
            let (spare, xs) = (svc.spare(net, l as usize, lev), x[lev] as f64);
            let shares: Vec<f64> = if spare >= 1.0 { (0..=SPLITS).map(|j| j as f64 / SPLITS as f64).collect() } else { vec![xs] };
            let now = crowding(&meets, h, cur, 0.0);
            let (mut best, mut at, mut share) = (now, cur, xs);
            for k in 0..if split_only { 1 } else { STEPS } {
                let c = if split_only { cur } else { h * k as f64 / STEPS as f64 };
                for &sh in &shares {
                    let v = crowding(&meets, h, c, (sh - xs) * spare);
                    if v < best - 1e-9 {
                        (best, at, share) = (v, c, sh);
                    }
                }
            }
            if best < KEEP * now - 1e-9 {
                p[lev] = at as f32;
                x[lev] = share as f32;
            }
        }
        if p != phase[l as usize] || x != split[l as usize] {
            phase[l as usize] = p;
            split[l as usize] = x;
            moved.push(l);
        }
    }
    moved
}

#[cfg(test)]
mod tests {
    use super::super::service::Trip;
    use super::super::tests::{edge, line, node, station, world};
    use super::super::world::{Op, TrackWorld};

    /// The departures of a line's first run from its first stop in the hour from 08:00 of day 1.
    fn deps(w: &TrackWorld, l: u32) -> Vec<f64> {
        let mut out: Vec<Trip> = vec![];
        let t0 = 86400.0 + 8.0 * 3600.0;
        w.svc.run_deps(&w.net, l as usize, 0, t0, t0 + 3600.0, &mut out);
        out.iter().map(|t| t.dep).collect()
    }

    /// The closest any departure of `a` comes to one of `b`, s.
    fn closest(a: &[f64], b: &[f64]) -> f64 {
        a.iter().flat_map(|x| b.iter().map(move |y| (x - y).abs())).fold(f64::INFINITY, f64::min)
    }

    /// T-1925: two lines on the same track at the same headway leave half a headway apart, and
    /// keep it through a save; one on its own track is not moved. Set by headway, so the headways
    /// are the same.
    #[test]
    fn lines_on_shared_track_leave_apart() {
        use super::super::tests::headway_line;
        let mut w = world();
        let n: Vec<u32> = (0..=4).map(|k| node(&mut w, k as f64 * 1500.0, 0.0, 0)).collect();
        for k in 0..4 { edge(&mut w, n[k], n[k + 1], &[], 0).unwrap(); }
        for &x in &n { station(&mut w, x, 200); }
        let q = [node(&mut w, 0.0, 600.0, 0), node(&mut w, 6000.0, 600.0, 0)];
        edge(&mut w, q[0], q[1], &[], 0).unwrap();
        for &x in &q { station(&mut w, x, 200); }
        crate::track::tests::terminals_at_dead_ends(&mut w);
        let a = headway_line(&mut w, &n, [6.0, 4.0, 2.0]);
        let alone = headway_line(&mut w, &q, [6.0, 4.0, 2.0]);
        let b = headway_line(&mut w, &n, [6.0, 4.0, 2.0]);
        let (da, db) = (deps(&w, a), deps(&w, b));
        assert_eq!(da.len(), 6);
        let gap = closest(&da, &db);
        assert!(gap > 0.4 * 600.0, "the two lines leave {gap:.0} s apart: {da:?} {db:?}");
        assert_eq!(w.net.line_phase_of(a), [0.0; crate::track::params::SLOTS], "the first line keeps its times");
        assert_eq!(w.net.line_phase_of(alone), [0.0; crate::track::params::SLOTS], "nothing shares its track");
        // a third line on the same track takes the gaps that are left
        let c = headway_line(&mut w, &n, [6.0, 4.0, 2.0]);
        let dc = deps(&w, c);
        // (a third of the headway at best; phases are tried in 24ths of it)
        assert!(closest(&dc, &da).min(closest(&dc, &db)) > 0.2 * 600.0, "{da:?} {db:?} {dc:?} {:?}", [a, b, c].map(|l| w.net.line_phase_of(l)));
        // kept through a save and a load
        let back = TrackWorld::new(super::super::save::decode(&super::super::save::encode(&w.net)).unwrap());
        for l in [a, b, c] {
            assert_eq!(back.net.line_phase_of(l), w.net.line_phase_of(l));
            assert_eq!(deps(&back, l), deps(&w, l));
        }
        // an edit elsewhere moves nobody
        let before: Vec<_> = [a, b, c].iter().map(|&l| w.net.line_phase_of(l)).collect();
        crate::track::tests::set_headway(&mut w, alone, [4.0, 4.0, 2.0]);
        assert_eq!(before, [a, b, c].iter().map(|&l| w.net.line_phase_of(l)).collect::<Vec<_>>());
    }

    /// T-2960: two headway lines on branches of different lengths joining one track at J take turns
    /// through the merge at even gaps, though the other direction or a flat junction's crossing
    /// would want other phases (1.65:1 and 2.41:1 at J in these two before the merge came first).
    #[test]
    fn trains_take_even_turns_at_a_merge() {
        use super::super::net::NodeData;
        use super::super::params::HIGH;
        use super::super::profile;
        for (headway, flying, m_x, br_x) in [(360, false, 6000.0, 4400.0), (420, true, 7000.0, 3700.0)] {
            let mut w = world();
            let t = node(&mut w, 0.0, 0.0, 0);
            let j = node(&mut w, 3000.0, 0.0, 0);
            let m = node(&mut w, m_x, 0.0, 0);
            let br = node(&mut w, br_x, 1500.0, 0);
            edge(&mut w, t, j, &[], 0).unwrap();
            edge(&mut w, j, m, &[], 0).unwrap();
            edge(&mut w, j, br, &[(3400.0, 0.0)], 0).unwrap();
            for x in [t, m, br] { station(&mut w, x, 200); }
            crate::track::tests::terminals_at_dead_ends(&mut w);
            if flying {
                w.apply(Op::Node { id: j, data: Some(NodeData { x: 3000.0, y: 0.0, level: 0, flying: true }) }).unwrap();
            }
            let mut ls = vec![];
            for end in [br, m] {
                let l = line(&mut w, &[t, end], [0.0; 3]);
                w.apply(Op::AutoTrains { id: l, auto: Some([headway, 2 * headway, 0]), trains: Some([4, 2, 0]) }).unwrap();
                ls.push(l);
            }
            let hw: Vec<f64> = ls.iter().map(|&l| 3600.0 / w.net.line_tph[l as usize][HIGH] as f64).collect();
            assert!(hw.iter().all(|&h| (h - headway as f64).abs() < 0.01), "both keep {headway} s: {hw:?}");
            // each train's time leaving its hold short of J towards T, 09:00 to 10:00 of a weekday
            let t0 = 86400.0 + 9.0 * 3600.0;
            let mut at_j: Vec<(f64, u32)> = vec![];
            for tr in w.svc.trips(&w.net, t0 - 7200.0, t0 + 3600.0).into_iter().filter(|tr| tr.run == 1) {
                let run = &w.svc.lines[tr.line as usize].runs[1];
                let i = run.nodes.iter().position(|&x| x == j).unwrap();
                let s = tr.draw + profile::time_at(&w.svc.lines[tr.line as usize].prof_of(1, tr.level as usize, tr.var).phases, run.node_s[i]);
                if s >= t0 && s < t0 + 3600.0 { at_j.push((s, tr.line)); }
            }
            at_j.sort_by(|x, y| x.0.total_cmp(&y.0));
            let gaps: Vec<f64> = at_j.windows(2).map(|p| p[1].0 - p[0].0).collect();
            assert!(at_j.windows(2).all(|p| p[0].1 != p[1].1), "the lines take turns: {at_j:?}");
            let (lo, hi) = gaps.iter().fold((f64::INFINITY, 0.0f64), |(a, b), &g| (a.min(g), b.max(g)));
            assert!(hi / lo < 1.25, "headway {headway}, flying {flying}: gaps at J {gaps:.0?}, phases {:?} shares {:?} spare {:?}", ls.iter().map(|&l| w.net.line_phase_of(l)).collect::<Vec<_>>(), ls.iter().map(|&l| w.net.line_split_of(l)).collect::<Vec<_>>(), ls.iter().map(|&l| w.svc.spare(&w.net, l as usize, HIGH)).collect::<Vec<_>>());
            // T-2972: and leaving T the other way, the spare time stood at the branch ends (was 1.3:1
            // and 8:1 in these two)
            let mut from_t: Vec<(f64, u32)> = w.svc.trips(&w.net, t0, t0 + 3600.0).into_iter().filter(|tr| tr.run == 0 && tr.dep >= t0).map(|tr| (tr.dep, tr.line)).collect();
            from_t.sort_by(|x, y| x.0.total_cmp(&y.0));
            let gaps: Vec<f64> = from_t.windows(2).map(|p| p[1].0 - p[0].0).collect();
            let (lo, hi) = gaps.iter().fold((f64::INFINITY, 0.0f64), |(a, b), &g| (a.min(g), b.max(g)));
            // (on a flat junction the crossing's 90 s weighs against it too)
            assert!(hi / lo < if flying { 1.25 } else { 1.5 }, "headway {headway}, flying {flying}: gaps leaving T {gaps:.0?}, shares {:?}", ls.iter().map(|&l| w.net.line_split_of(l)).collect::<Vec<_>>());
            // kept through a save and a load
            let back = TrackWorld::new(super::super::save::decode(&super::super::save::encode(&w.net)).unwrap());
            for &l in &ls {
                assert_eq!(back.net.line_split_of(l), w.net.line_split_of(l));
            }
        }
    }

    /// T-2973: two lines set by train count on branches of different lengths, 4 trains each, run
    /// one headway when theirs are within 3% (5.1 and 6.0 km branches: 301.2 and 302.3 s, whose
    /// gaps at J drifted from 1.6:1 at 09h to 30:1 at 17h), and take even turns at J all day; 3.8%
    /// apart (4.8 and 7.0 km) they keep their own.
    #[test]
    fn count_lines_at_a_merge_run_in_step() {
        use super::super::params::HIGH;
        use super::super::profile;
        for (m_x, br_x, in_step) in [(6000.0, 4400.0, true), (7000.0, 3700.0, false)] {
            let mut w = world();
            let t = node(&mut w, 0.0, 0.0, 0);
            let j = node(&mut w, 3000.0, 0.0, 0);
            let m = node(&mut w, m_x, 0.0, 0);
            let br = node(&mut w, br_x, 1500.0, 0);
            edge(&mut w, t, j, &[], 0).unwrap();
            edge(&mut w, j, m, &[], 0).unwrap();
            edge(&mut w, j, br, &[(3400.0, 0.0)], 0).unwrap();
            for x in [t, m, br] { station(&mut w, x, 200); }
            crate::track::tests::terminals_at_dead_ends(&mut w);
            let mut ls = vec![];
            for end in [br, m] {
                let l = line(&mut w, &[t, end], [0.0; 3]);
                w.apply(Op::TrainCounts { id: l, trains: Some([4, 2, 0]), tph: [6.0, 3.0, 0.0] }).unwrap();
                ls.push(l);
            }
            let hw: Vec<f64> = ls.iter().map(|&l| 3600.0 / w.net.line_tph[l as usize][HIGH] as f64).collect();
            let natural: Vec<f64> = ls.iter().map(|&l| w.svc.lines[l as usize].round_trip[HIGH] / 4.0).collect();
            if !in_step {
                assert!((hw[0] - hw[1]).abs() > 5.0 && ls.iter().all(|&l| w.net.line_pace_of(l)[HIGH] == 0.0), "3.8% apart, each its own: {hw:?}");
                continue;
            }
            assert!((hw[0] - hw[1]).abs() < 0.01, "one headway: {hw:?} (round trip over trains {natural:?})");
            for h0 in [9.0, 13.0, 17.0] {
                let t0 = 86400.0 + h0 * 3600.0;
                let mut at_j: Vec<(f64, u32)> = vec![];
                for tr in w.svc.trips(&w.net, t0 - 7200.0, t0 + 3600.0).into_iter().filter(|tr| tr.run == 1) {
                    let run = &w.svc.lines[tr.line as usize].runs[1];
                    let i = run.nodes.iter().position(|&x| x == j).unwrap();
                    let s = tr.draw + profile::time_at(&w.svc.lines[tr.line as usize].prof_of(1, tr.level as usize, tr.var).phases, run.node_s[i]);
                    if s >= t0 && s < t0 + 3600.0 { at_j.push((s, tr.line)); }
                }
                at_j.sort_by(|x, y| x.0.total_cmp(&y.0));
                let gaps: Vec<f64> = at_j.windows(2).map(|p| p[1].0 - p[0].0).collect();
                let (lo, hi) = gaps.iter().fold((f64::INFINITY, 0.0f64), |(a, b), &g| (a.min(g), b.max(g)));
                assert!(hi / lo < 1.25, "{h0}h: gaps at J {gaps:.0?}");
            }
            // T-3233: the pacing rides in the save with the rates, and a load keeps both
            let bytes = super::super::save::encode(&w.net);
            assert!(bytes.windows(4).any(|x| x == b"LPC1"));
            let back = TrackWorld::new(super::super::save::decode(&bytes).unwrap());
            for &l in &ls {
                assert!(back.net.line_pace_of(l)[HIGH] > 0.0);
                assert_eq!((back.net.line_pace_of(l), back.net.line_tph[l as usize]), (w.net.line_pace_of(l), w.net.line_tph[l as usize]));
            }
            assert_eq!(super::super::save::encode(&back.net), bytes, "and writes back the same bytes");
            // T-3388: an edit refused for money leaves the pacing and the rates as they were; once
            // the pacing has held through a settle, an edit that moves nothing paces nothing again
            let state = |w: &TrackWorld| ls.iter().map(|&l| (w.net.line_pace_of(l), w.net.line_tph[l as usize], w.net.line_trains[l as usize])).collect::<Vec<_>>();
            let before = state(&w);
            let tph = crate::track::params::slots_to_rows(w.net.line_tph[ls[0] as usize]);
            let refused = w.apply_paid(Op::TrainCounts { id: ls[0], trains: Some([5, 2, 0]), tph }, 0.0, false, &|_| 1.0);
            assert!(matches!(refused, Err(super::super::world::ChargeError::Money { .. })));
            assert_eq!(before, state(&w));
            let same = |w: &mut TrackWorld| w.apply(Op::TrainCounts { id: ls[1], trains: w.net.line_trains[ls[1] as usize], tph: crate::track::params::slots_to_rows(w.net.line_tph[ls[1] as usize]) }).unwrap();
            same(&mut w);
            let held = state(&w);
            assert!(held.iter().all(|x| x.0[HIGH] > 0.0), "still in step: {held:?}");
            same(&mut w);
            assert_eq!(held, state(&w));
        }
    }

    /// T-3584: two count lines on branches 5.1 and 6.0 km, 4 trains each on weekdays, run one headway
    /// there (T-2973); at 3 trains each on weekends they are paced again from the weekend headways
    /// (the weekday pace 302 s would leave them 400 s apart by 1.6 s), the weekdays' left as they were.
    #[test]
    fn weekend_lines_at_a_merge_are_paced_from_their_weekend_headways() {
        use super::super::net::Weekend;
        use super::super::params::{HIGH, WEEKEND};
        let mut w = world();
        let t = node(&mut w, 0.0, 0.0, 0);
        let j = node(&mut w, 3000.0, 0.0, 0);
        let m = node(&mut w, 6000.0, 0.0, 0);
        let br = node(&mut w, 4400.0, 1500.0, 0);
        edge(&mut w, t, j, &[], 0).unwrap();
        edge(&mut w, j, m, &[], 0).unwrap();
        edge(&mut w, j, br, &[(3400.0, 0.0)], 0).unwrap();
        for x in [t, m, br] { station(&mut w, x, 200); }
        crate::track::tests::terminals_at_dead_ends(&mut w);
        let mut ls = vec![];
        for end in [br, m] {
            let l = line(&mut w, &[t, end], [0.0; 3]);
            w.apply(Op::TrainCounts { id: l, trains: Some([4, 2, 0]), tph: [6.0, 3.0, 0.0] }).unwrap();
            ls.push(l);
        }
        let week = |w: &TrackWorld| ls.iter().map(|&l| (w.net.line_pace_of(l), w.net.line_tph[l as usize], w.net.line_trains[l as usize])).collect::<Vec<_>>();
        let before = week(&w);
        assert!(before.iter().all(|x| x.0[HIGH] > 0.0), "weekdays in step");
        // the weekend headways: the long branch with 5 trains runs as often as the short with 3
        let trip: Vec<f64> = ls.iter().map(|&l| w.svc.lines[l as usize].round_trip[HIGH]).collect();
        let counts = [3u32, 3];
        let own: Vec<f64> = (0..2).map(|i| trip[i] / counts[i] as f64).collect();
        let (lo, hi) = (own[0].min(own[1]), own[0].max(own[1]));
        // (the two headways are what the branches' lengths make them: skip the premise if not close)
        for (i, &l) in ls.iter().enumerate() {
            w.apply(Op::Weekend { id: l, weekend: Weekend { share: 100, counts: Some([counts[i], 2, 0]), headways: [0; 3] } }).unwrap();
        }
        assert_eq!(before, week(&w), "the weekdays are left as they were");
        assert!(hi / lo - 1.0 < 0.03, "weekend headways {own:?} within the pacing share");
        let rest: Vec<f64> = ls.iter().map(|&l| 3600.0 / w.net.line_tph_on(l as usize, WEEKEND, HIGH) as f64).collect();
        assert!((rest[0] - rest[1]).abs() < 0.01, "one weekend headway: {rest:?} (own {own:?})");
        assert!(w.net.line_pace_rest.iter().any(|p| p[HIGH] > 0.0), "weekend pacing kept");
        // the same through an edit elsewhere on the weekends' kept service
        let kept = rest.clone();
        w.apply(Op::Weekend { id: ls[0], weekend: Weekend { share: 100, counts: Some([counts[0], 2, 0]), headways: [0; 3] } }).unwrap();
        let again: Vec<f64> = ls.iter().map(|&l| 3600.0 / w.net.line_tph_on(l as usize, WEEKEND, HIGH) as f64).collect();
        assert!(again.iter().zip(&kept).all(|(a, b)| (a - b).abs() < 1.0), "{again:?} against {kept:?}");
        assert_eq!(before, week(&w));
    }

    /// T-3735: two headway lines into one track at J (T-2960's branches, 6 min) run 12 min on
    /// weekends: there they take even turns through J and leaving T, their departures placed on the
    /// weekend headway and its spare time, the weekdays' left to the bit; a save keeps them.
    #[test]
    fn weekend_departures_are_placed_on_the_weekend_service() {
        use super::super::net::Weekend;
        use super::super::params::{HIGH, WEEKEND};
        use super::super::profile;
        let mut w = world();
        let t = node(&mut w, 0.0, 0.0, 0);
        let j = node(&mut w, 3000.0, 0.0, 0);
        let m = node(&mut w, 6000.0, 0.0, 0);
        let br = node(&mut w, 4400.0, 1500.0, 0);
        edge(&mut w, t, j, &[], 0).unwrap();
        edge(&mut w, j, m, &[], 0).unwrap();
        edge(&mut w, j, br, &[(3400.0, 0.0)], 0).unwrap();
        for x in [t, m, br] { station(&mut w, x, 200); }
        crate::track::tests::terminals_at_dead_ends(&mut w);
        let mut ls = vec![];
        for end in [br, m] {
            let l = line(&mut w, &[t, end], [0.0; 3]);
            w.apply(Op::AutoTrains { id: l, auto: Some([360, 720, 0]), trains: Some([4, 2, 0]) }).unwrap();
            ls.push(l);
        }
        // the trips of an hour from `h0` of `day`, at J towards T and leaving T
        let hour = |w: &TrackWorld, day: f64, h0: f64| {
            let t0 = day * 86400.0 + h0 * 3600.0;
            let (mut at_j, mut from_t): (Vec<(f64, u32)>, Vec<(f64, u32)>) = (vec![], vec![]);
            for tr in w.svc.trips(&w.net, t0 - 7200.0, t0 + 3600.0) {
                if tr.run == 0 {
                    if tr.dep >= t0 && tr.dep < t0 + 3600.0 { from_t.push((tr.dep, tr.line)); }
                    continue;
                }
                let run = &w.svc.lines[tr.line as usize].runs[1];
                let i = run.nodes.iter().position(|&x| x == j).unwrap();
                let s = tr.draw + profile::time_at(&w.svc.lines[tr.line as usize].prof_of(1, tr.level as usize, tr.var).phases, run.node_s[i]);
                if s >= t0 && s < t0 + 3600.0 { at_j.push((s, tr.line)); }
            }
            for v in [&mut at_j, &mut from_t] { v.sort_by(|x, y| x.0.total_cmp(&y.0)); }
            (at_j, from_t)
        };
        let ratio = |v: &[(f64, u32)]| {
            let gaps: Vec<f64> = v.windows(2).map(|p| p[1].0 - p[0].0).collect();
            let (lo, hi) = gaps.iter().fold((f64::INFINITY, 0.0f64), |(a, b), &g| (a.min(g), b.max(g)));
            (hi / lo, gaps)
        };
        let tuesday: Vec<_> = [9.0, 13.0].map(|h| hour(&w, 2.0, h)).into();
        let trips: Vec<_> = ls.iter().map(|&l| w.svc.lines[l as usize].round_trip).collect();
        for &l in &ls {
            w.apply(Op::Weekend { id: l, weekend: Weekend { share: 50, counts: None, headways: [0; 3] } }).unwrap();
        }
        let rest: Vec<f64> = ls.iter().map(|&l| 3600.0 / w.net.line_tph_on(l as usize, WEEKEND, HIGH) as f64).collect();
        assert!(rest.iter().all(|&h| (h - 720.0).abs() < 0.01), "12 min on weekends: {rest:?}");
        // Saturday
        for h0 in [9.0, 13.0] {
            let (at_j, from_t) = hour(&w, 6.0, h0);
            assert_eq!(at_j.len(), 10, "{h0}h: {at_j:?}");
            assert!(at_j.windows(2).all(|p| p[0].1 != p[1].1), "{h0}h: the lines take turns at J: {at_j:?}");
            let (r, gaps) = ratio(&at_j);
            eprintln!("{h0}h: Saturday gaps at J {gaps:.0?}");
            assert!(r < 1.25, "{h0}h: gaps at J {gaps:.0?}, weekend phases {:?} splits {:?}", ls.iter().map(|&l| w.net.line_phase_rest_of(l)).collect::<Vec<_>>(), ls.iter().map(|&l| w.net.line_split_rest_of(l)).collect::<Vec<_>>());
            let (r, gaps) = ratio(&from_t);
            assert!(r < 1.5, "{h0}h: gaps leaving T {gaps:.0?}");
        }
        // every Saturday midday train turns around into its next departure, standing at the far end
        // the turnaround and its weekend share of the spare time a 12-minute headway leaves it
        let t0 = 6.0 * 86400.0 + 10.0 * 3600.0;
        for &l in &ls {
            let ls_ = &w.svc.lines[l as usize];
            let trip = ls_.trip_len(&w.net, l, 0, HIGH) + ls_.trip_len(&w.net, l, 1, HIGH);
            let spare = ((trip / 720.0 - 1e-3).ceil() * 720.0 - trip).clamp(0.0, 720.0);
            let share = w.net.line_split_rest_of(l)[HIGH] as f64;
            let turn = ls_.close_stand(&w.net, l);
            for tr in w.svc.trips(&w.net, t0, t0 + 3600.0).into_iter().filter(|tr| tr.line == l && tr.dep >= t0 && tr.dep < t0 + 3000.0) {
                let next = tr.next.unwrap_or_else(|| panic!("line {l}: a Saturday train leaves service: {tr:?}"));
                let stand = next - (tr.draw + ls_.prof_of(tr.run as usize, HIGH, tr.var).duration);
                let want = turn + if tr.run == 0 { share } else { 1.0 - share } * spare;
                assert!((stand - want).abs() < 1.0, "line {l} run {}: stands {stand:.1} s, want {want:.1} (turnaround {turn}, weekend spare {spare:.1}, share {share})", tr.run);
            }
        }
        // the weekdays as they were, round trips and every trip
        assert_eq!(trips, ls.iter().map(|&l| w.svc.lines[l as usize].round_trip).collect::<Vec<_>>());
        assert_eq!(tuesday, Vec::from([9.0, 13.0].map(|h| hour(&w, 2.0, h))));
        // a save keeps the weekend departures to the bit
        let bytes = super::super::save::encode(&w.net);
        assert!(bytes.windows(4).any(|x| x == b"LWP1"));
        let back = TrackWorld::new(super::super::save::decode(&bytes).unwrap());
        assert_eq!(hour(&back, 6.0, 9.0), hour(&w, 6.0, 9.0));
        assert_eq!(hour(&back, 2.0, 9.0), hour(&w, 2.0, 9.0));
        // and weekends back at 100% run the weekdays' departures, with nothing saved for them
        for &l in &ls {
            w.apply(Op::Weekend { id: l, weekend: Weekend::default() }).unwrap();
        }
        assert!(!w.net.rest_solved && w.net.line_phase_rest.is_empty());
        assert!(!super::super::save::encode(&w.net).windows(4).any(|x| x == b"LWP1"));
    }
}
