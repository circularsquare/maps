//! Departure spacing (T-1925, notes/T-1900.md D1): each line's departures at a demand level leave
//! a phase after its own pattern's times (`Lattice::phase`), picked so its trains and those of the
//! lines it shares track, a platform or a flat junction with do not reach the place they meet
//! together (and are drawn on top of each other). Real timetabling, done greedily: only the lines
//! an edit changed are placed, against every other line as it runs, so the rest keep their times.
//! Waits are by headway, so demand does not move.

use super::capacity::ResKind;
use super::net::Network;
use super::params::LEVELS;
use super::profile;
use super::service::Service;

/// Phases tried per headway.
const STEPS: usize = 24;
/// A line keeps its phase unless another one cuts its crowding by more than this share.
const KEEP: f64 = 0.9;
/// Trains of the line looked at per meeting: an hour's, at most this many.
const MAX_TRAINS: usize = 60;

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
}

/// How crowded a phase leaves the line's meetings: per train, how far it is short of half the
/// smaller headway from the nearest train of the other line there, squared, as a share. For a
/// following pair timetabled together, how far each following train is from its place after a
/// leading one, as a share of the same half headway, squared.
fn crowding(meets: &[Meet], h: f64, phase: f64) -> f64 {
    let n = ((3600.0 / h).ceil() as usize).clamp(1, MAX_TRAINS);
    let mut sum = 0.0;
    for m in meets {
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
        for k in 0..n {
            let x = (m.a + phase + k as f64 * h - m.b).rem_euclid(m.hb);
            let d = x.min(m.hb - x);
            if d < g {
                sum += ((g - d) / g).powi(2);
            }
        }
    }
    sum
}

/// Place the departures of `lines` (any order; placed in id order, each against the phases as
/// they then stand) into `phase` (by line id, grown as needed). Returns the lines whose phase moved.
pub fn place(net: &Network, svc: &Service, lines: &[u32], phase: &mut Vec<[f32; LEVELS]>) -> Vec<u32> {
    let cap = &svc.cap;
    phase.resize(phase.len().max(net.line_count()), [0.0; LEVELS]);
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
    let paired: [Vec<((u32, u8), (u32, u8), f64)>; LEVELS] = std::array::from_fn(|lev| {
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
    let mut first: Vec<((u32, u8, u8), f64, f64, f64)> = vec![]; // (other line, its run, own run), own time, a, b
    for &l in &todo {
        let Some(Some(lat)) = lats[l as usize] else { continue };
        let mut p = phase[l as usize];
        for lev in 0..LEVELS {
            let tph = lat.tph[lev] as f64;
            if tph <= 0.0 {
                continue;
            }
            let h = 3600.0 / tph;
            let base = |lat: &super::service::Lattice, run: u8| -lat.lead - lat.shift + if run == 1 { lat.off[lev] } else { 0.0 };
            // per pair of directions, the first place they meet along this line's run
            first.clear();
            for &r in &by_line[l as usize] {
                let us = cap.users_of(r);
                let junction = cap.res[r].kind == ResKind::Junction;
                for u in us.iter().filter(|u| u.line == l) {
                    let tu = time(l, u.run, lev, u.at);
                    for v in us.iter().filter(|v| v.line != l && (!junction || v.owner != u.owner)) {
                        if net.line_tph[v.line as usize][lev] <= 0.0 || !svc.lines[v.line as usize].running {
                            continue;
                        }
                        let key = (v.line, v.run, u.run);
                        let Some(Some(other)) = &lats[v.line as usize] else { continue };
                        let b = base(other, v.run) + phase[v.line as usize][lev] as f64 + time(v.line, v.run, lev, v.at);
                        let a = base(&lat, u.run) + tu;
                        match first.iter_mut().find(|x| x.0 == key) {
                            Some(x) if x.1 <= tu => {}
                            Some(x) => *x = (key, tu, a, b),
                            None => first.push((key, tu, a, b)),
                        }
                    }
                }
            }
            if first.is_empty() {
                continue;
            }
            let meets: Vec<Meet> = first.iter().map(|x| Meet { a: x.2, b: x.3, hb: 3600.0 / net.line_tph[x.0 .0 as usize][lev] as f64, behind: behind(lev, l, x.0 .2, x.0 .0, x.0 .1) }).collect();
            let cur = (p[lev] as f64).rem_euclid(h);
            let now = crowding(&meets, h, cur);
            let (mut best, mut at) = (now, cur);
            for k in 0..STEPS {
                let c = h * k as f64 / STEPS as f64;
                let x = crowding(&meets, h, c);
                if x < best - 1e-9 {
                    (best, at) = (x, c);
                }
            }
            if best < KEEP * now - 1e-9 {
                p[lev] = at as f32;
            }
        }
        if p != phase[l as usize] {
            phase[l as usize] = p;
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
    /// keep it through a save; one on its own track is not moved.
    #[test]
    fn lines_on_shared_track_leave_apart() {
        let mut w = world();
        let n: Vec<u32> = (0..=4).map(|k| node(&mut w, k as f64 * 1500.0, 0.0, 0)).collect();
        for k in 0..4 { edge(&mut w, n[k], n[k + 1], &[], 0).unwrap(); }
        for &x in &n { station(&mut w, x, 200); }
        let q = [node(&mut w, 0.0, 600.0, 0), node(&mut w, 6000.0, 600.0, 0)];
        edge(&mut w, q[0], q[1], &[], 0).unwrap();
        for &x in &q { station(&mut w, x, 200); }
        let a = line(&mut w, &n, [6.0, 4.0, 2.0]);
        let alone = line(&mut w, &q, [6.0, 4.0, 2.0]);
        let b = line(&mut w, &n, [6.0, 4.0, 2.0]);
        let (da, db) = (deps(&w, a), deps(&w, b));
        assert_eq!(da.len(), 6);
        let gap = closest(&da, &db);
        assert!(gap > 0.4 * 600.0, "the two lines leave {gap:.0} s apart: {da:?} {db:?}");
        assert_eq!(w.net.line_phase_of(a), [0.0; 3], "the first line keeps its times");
        assert_eq!(w.net.line_phase_of(alone), [0.0; 3], "nothing shares its track");
        // a third line on the same track takes the gaps that are left
        let c = line(&mut w, &n, [6.0, 4.0, 2.0]);
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
        w.apply(Op::Schedule { id: alone, tph: [4.0, 4.0, 2.0] }).unwrap();
        assert_eq!(before, [a, b, c].iter().map(|&l| w.net.line_phase_of(l)).collect::<Vec<_>>());
    }
}
