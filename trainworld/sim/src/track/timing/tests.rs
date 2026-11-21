use super::super::overtake::tests::{set_passing, shinkansen, shinkansen_rates};
use super::super::overtake::PASS_CLEAR_S;
use super::super::profile;
use super::super::service::Trip;
use super::super::world::{Op, TrackWorld};
use super::*;

/// A fixed-rate line's trains an hour (`shinkansen_rates`).
fn set_tph(w: &mut TrackWorld, id: u32, tph: [f32; 3]) {
    let mut d = w.net.line_data(id).unwrap();
    d.tph = tph;
    w.apply(Op::Line { id, data: Some(d) }).unwrap();
}

/// The times a trip is at each of `nodes` (game time): reaching it, and leaving or passing it.
fn at_nodes(w: &TrackWorld, t: &Trip, nodes: &[u32], mean: bool) -> Vec<(f64, f64)> {
    let ls = &w.svc.lines[t.line as usize];
    let run = &ls.runs[t.run as usize];
    let p = ls.prof_of(t.run as usize, t.level as usize, if mean { 0 } else { t.var });
    nodes
        .iter()
        .map(|n| {
            let s = run.node_s[run.nodes.iter().position(|x| x == n).unwrap()];
            (t.draw + profile::time_at(&p.phases, (s - 0.5).max(0.0)), t.draw + profile::time_at(&p.phases, s))
        })
        .collect()
}

/// Where the timetable has an express pass a local other than at a passing station with the local
/// standing there and leaving `PASS_CLEAR_S` after it: per direction, over `[t0, t1)`; and how
/// many good passes there were. `mean`: every train on its level's timetable, as before T-3346.
fn bad_passes(w: &TrackWorld, local: u32, express: u32, stations: &[u32], passing: &[u32], t0: f64, t1: f64, mean: bool) -> (Vec<String>, usize) {
    let (mut out, mut good) = (vec![], 0);
    let lt = w.svc.line_trips(&w.net, local as usize, t0 - 7200.0, t1);
    let xt = w.svc.line_trips(&w.net, express as usize, t0 - 7200.0, t1);
    for r in 0..2u8 {
        let order: Vec<u32> = if r == 0 { stations.to_vec() } else { stations.iter().rev().copied().collect() };
        for l in lt.iter().filter(|t| t.run == r) {
            let lp = at_nodes(w, l, &order, mean);
            if lp[0].1 < t0 || lp[0].1 >= t1 {
                continue;
            }
            for x in xt.iter().filter(|t| t.run == r) {
                let xp = at_nodes(w, x, &order, mean);
                for i in 0..order.len() {
                    // the express catches up from behind on the line before node i
                    if i > 0 && xp[i - 1].1 > lp[i - 1].1 && xp[i].0 < lp[i].0 - 1e-6 {
                        out.push(format!("run {r}: express {:.0} through local {:.0} before node {i}", x.dep, l.dep));
                    }
                    // or passes it standing at node i
                    if i > 0 && i + 1 < order.len() && xp[i].0 > lp[i].0 && xp[i].1 < lp[i].1 {
                        let ok = passing.contains(&order[i]) && lp[i].1 >= xp[i].1 + PASS_CLEAR_S - 1e-6;
                        good += ok as usize;
                        if !ok {
                            out.push(format!("run {r}: express {:.0} passes local {:.0} at node {i}, local leaves {:.0} s after", x.dep, l.dep, lp[i].1 - xp[i].1));
                        }
                    }
                }
            }
        }
    }
    (out, good)
}

/// T-3346 stage 2's done-when (notes/T-3346.md): on the 180 km test line with passing tracks at B
/// to F, at 10/10, 30/10 and 10/30 (local/express minutes), every local leaves each passing station
/// at least `PASS_CLEAR_S` after the last express passes, both directions, in the timetable: it
/// once stood every local the plan's mean wait (T-3343). Demand's stop times stay the means.
#[test]
fn every_local_waits_for_its_express() {
    for (hl, hx) in [(10.0f32, 10.0f32), (30.0, 10.0), (10.0, 30.0)] {
        let (mut w, n, local, express) = shinkansen_rates();
        for (id, h) in [(local, hl), (express, hx)] {
            set_tph(&mut w, id, [60.0 / h, 60.0 / h, 0.0]);
        }
        set_passing(&mut w, &n[1..6], true);
        let t0 = 86400.0 + 8.0 * 3600.0;
        let (bad, good) = bad_passes(&w, local, express, &n, &n[1..6], t0, t0 + 4.0 * 3600.0, false);
        let (was, _) = bad_passes(&w, local, express, &n, &n[1..6], t0, t0 + 4.0 * 3600.0, true);
        eprintln!("{hl}/{hx}: {good} passes at passing stations in 4 h; with the averaged waits {} bad", was.len());
        assert!(good >= 8, "{hl}/{hx}: the expresses pass the locals");
        let lr = &w.svc.lines[local as usize].runs;
        eprintln!("{hl}/{hx}: variants {:?}, patterns {:?}", [lr[0].vars[0].len(), lr[1].vars[0].len()], [&lr[0].pattern[0][..12.min(lr[0].pattern[0].len())], &lr[1].pattern[0][..12.min(lr[1].pattern[0].len())]]);
        for r in 0..2 {
            for v in &lr[r].vars[0] {
                let extra: Vec<f64> = v.stop_holds.iter().zip(&lr[r].stop_holds[0]).map(|(a, b)| (a - b).round()).collect();
                eprintln!("  run {r} variant: waits over the mean {extra:?}");
            }
        }
        assert!(bad.is_empty(), "{hl}/{hx}: {} bad passes, first {:?}", bad.len(), &bad[..bad.len().min(6)]);
        assert!(lr[0].vars[0].len() <= MAX_VARS && !lr[0].vars[0].is_empty(), "{hl}/{hx}: the local is timed train by train");
        // demand's stop times are the level's, as before
        let trips = w.svc.line_trips(&w.net, local as usize, t0, t0 + 3600.0);
        assert!(trips.iter().any(|t| t.var > 0));
    }
}

/// Without passing tracks the express is timed behind the local ahead of it train by train
/// (T-3271): the timetable never has it through a local, where the averaged following loss did.
#[test]
fn an_express_with_nowhere_to_pass_follows_its_local() {
    let (mut w, n, local, express) = shinkansen();
    for (id, h) in [(local, 10.0f32), (express, 10.0f32)] {
        crate::track::tests::set_headway(&mut w, id, [60.0 / h, 60.0 / h, 0.0]);
    }
    let t0 = 86400.0 + 8.0 * 3600.0;
    let er = &w.svc.lines[express as usize].runs;
    let mean: Vec<f64> = (0..2).map(|r| er[r].prof[0].duration - er[r].free.duration).collect();
    eprintln!("express variants {:?}, mean loss {mean:.0?} s", [er[0].vars[0].len(), er[1].vars[0].len()]);
    for r in 0..2 {
        for v in &er[r].vars[0] {
            eprintln!("  run {r}: {:.0} s over free", v.prof.duration - er[r].free.duration);
        }
    }
    let (bad, _) = bad_passes(&w, local, express, &n, &[], t0, t0 + 4.0 * 3600.0, false);
    let (was, _) = bad_passes(&w, local, express, &n, &[], t0, t0 + 4.0 * 3600.0, true);
    let xs = &w.svc.lines[express as usize];
    let own: Vec<f64> = w.svc.line_trips(&w.net, express as usize, t0, t0 + 4.0 * 3600.0).iter().map(|t| xs.prof_of(t.run as usize, 0, t.var).duration - xs.runs[t.run as usize].free.duration).collect();
    eprintln!("with the averaged loss: {} bad; its trains' own loss a run, mean {:.0} s over {} trips", was.len(), own.iter().sum::<f64>() / own.len().max(1) as f64, own.len());
    assert!(bad.is_empty(), "{} bad passes, first {:?}", bad.len(), &bad[..bad.len().min(6)]);
}

/// Variants are derived: a saved and loaded game times its trains the same.
#[test]
fn variants_come_back_after_a_load() {
    let (mut w, n, local, express) = shinkansen();
    crate::track::tests::set_headway(&mut w, express, [2.0, 2.0, 0.0]);
    set_passing(&mut w, &n[1..6], true);
    let back = TrackWorld::new(super::super::save::decode(&super::super::save::encode(&w.net)).unwrap());
    let t0 = 86400.0 + 8.0 * 3600.0;
    for l in [local, express] {
        assert_eq!(w.svc.line_trips(&w.net, l as usize, t0, t0 + 7200.0), back.svc.line_trips(&back.net, l as usize, t0, t0 + 7200.0));
        assert_eq!(w.svc.lines[l as usize].runs[0].vars, back.svc.lines[l as usize].runs[0].vars);
    }
}

#[test]
fn cluster_keeps_every_departure_under_its_variant() {
    let each: Vec<Vec<i64>> = (0..40).map(|i| vec![(i * 7) % 13, (i * 3) % 5, 0]).collect();
    let (reps, which) = cluster(&each);
    assert!(reps.len() <= MAX_VARS);
    for (e, &k) in each.iter().zip(&which) {
        assert!(reps[k].iter().zip(e).all(|(r, x)| r >= x), "{e:?} under {:?}", reps[k]);
    }
}

/// T-3271: t1965-follow's track (8 km of ordinary double track, a station every km, a local at all
/// of them and an express at the ends, one train each): the express, timed at its own 907 s round
/// trip against the local's 1,885 s, was drawn through the local where it has nowhere to pass. Now
/// each express trip is timed behind the local ahead of it, losing no more than its stand at the
/// far end allows (its round trip is sized on the averaged loss): every train still makes its next
/// departure, and the timetable has the express through the local no more often than with the
/// averaged loss (past the cap it is drawn held behind it). Prints the figures for the report.
#[test]
fn t1965_track_express_never_through_its_local() {
    use super::super::tests::{edge, line, node, station, world};
    for (nl, nx) in [(1u32, 1u32), (4, 2)] {
        let mut w = world();
        let n: Vec<u32> = (0..9).map(|k| node(&mut w, k as f64 * 1000.0, 0.0, 0)).collect();
        for k in 0..8 {
            edge(&mut w, n[k], n[k + 1], &[], 0).unwrap();
        }
        for &x in &n {
            station(&mut w, x, 200);
        }
        super::super::tests::terminals_at_dead_ends(&mut w);
        let local = line(&mut w, &n, [4.0, 4.0, 4.0]);
        let express = line(&mut w, &[n[0], n[8]], [4.0, 4.0, 4.0]);
        for (id, k) in [(local, nl), (express, nx)] {
            w.apply(Op::TrainCounts { id, trains: Some([k; 3]), tph: [4.0; 3] }).unwrap();
        }
        let t0 = 86400.0 + 8.0 * 3600.0;
        let (bad, _) = bad_passes(&w, local, express, &n, &[], t0, t0 + 4.0 * 3600.0, false);
        let (was, _) = bad_passes(&w, local, express, &n, &[], t0, t0 + 4.0 * 3600.0, true);
        let (ls, xs) = (&w.svc.lines[local as usize], &w.svc.lines[express as usize]);
        let trips = w.svc.line_trips(&w.net, express as usize, t0, t0 + 4.0 * 3600.0);
        let own: Vec<f64> = trips.iter().map(|t| xs.prof_of(t.run as usize, 0, t.var).duration - xs.runs[t.run as usize].free.duration).collect();
        let mean_own = own.iter().sum::<f64>() / own.len().max(1) as f64;
        eprintln!(
            "{nl}+{nx}: round trips local {:.0} s express {:.0} s; express timetabled loss {:.0?} s a run, its trains' own mean {mean_own:.0} s (min {:.0}, max {:.0}); bad passes {} (averaged: {})",
            ls.round_trip[0], xs.round_trip[0],
            (0..2).map(|r| xs.runs[r].prof[0].duration - xs.runs[r].free.duration).collect::<Vec<_>>(),
            own.iter().cloned().fold(f64::INFINITY, f64::min), own.iter().cloned().fold(0.0, f64::max), bad.len(), was.len()
        );
        // every train makes its next departure: its own run no longer than the slot's by more than
        // its stand at the far end less `TURN_MIN_S`
        let late: Vec<f64> = trips
            .iter()
            .map(|t| xs.prof_of(t.run as usize, 0, t.var).duration - xs.runs[t.run as usize].prof[0].duration - (w.net.line_turn[express as usize] as f64 - TURN_MIN_S))
            .filter(|&x| x > 0.0)
            .collect();
        eprintln!("{nl}+{nx}: trips missing their next departure {}", late.len());
        assert!(late.is_empty(), "{nl}+{nx}: {} express trips run past their next departure, by {late:?} s", late.len());
        assert!(bad.len() <= was.len(), "{nl}+{nx}: {} bad passes against {} averaged, first {:?}", bad.len(), was.len(), &bad[..bad.len().min(6)]);
    }
}

/// T-3346 stage 2, open: both lines keeping headways (as players set them, t2233's 30/10) run at
/// the phase spacing places them, not the plan's best; the trains' own waits there against the
/// averaged waits the round trip is sized on.
#[test]
fn headway_lines_waits_at_the_real_phase() {
    for (hl, hx) in [(600u32, 600u32), (1800, 600), (600, 1800)] {
        let (mut w, n, local, express) = shinkansen();
        for (id, h) in [(local, hl), (express, hx)] {
            let k = w.net.line_trains[id as usize].unwrap_or([6, 6, 0]);
            w.apply(Op::AutoTrains { id, auto: Some([h, h, 0]), trains: Some(k) }).unwrap();
        }
        set_passing(&mut w, &n[1..6], true);
        let ls = &w.svc.lines[local as usize];
        for r in 0..2 {
            let mean: f64 = ls.runs[r].prof[0].duration;
            let t0 = 86400.0 + 8.0 * 3600.0;
            let own: Vec<f64> = w.svc.line_trips(&w.net, local as usize, t0, t0 + 4.0 * 3600.0).iter().filter(|t| t.run as usize == r).map(|t| ls.prof_of(r, 0, t.var).duration - mean).collect();
            eprintln!("{}/{} run {r}: trips' own run over the averaged, s: {:?}; turnaround {} s", hl / 60, hx / 60, own.iter().map(|x| x.round()).collect::<Vec<_>>(), w.net.line_turn[local as usize]);
        }
    }
}

/// T-3542 meets stage 2: at four rows the rush is one slot with two stretches a day (6-10 and 16-20),
/// so a run's pattern names each departure by its stretch as well as its place in it: the
/// afternoon's locals wait for their own expresses, not the morning's.
#[test]
fn both_rush_stretches_are_timed_on_their_own() {
    let (mut w, n, local, express) = shinkansen_rates();
    for (id, h) in [(local, 30.0f32), (express, 10.0f32)] {
        set_tph(&mut w, id, [60.0 / h, 60.0 / h, 0.0]);
    }
    set_passing(&mut w, &n[1..6], true);
    let op = w.rows_ops(None, 4).unwrap();
    w.apply(op).unwrap();
    assert_eq!(w.net.period_slot(), [0, 1, 0, 3, 4]);
    let lr = &w.svc.lines[local as usize].runs;
    let stretches: Vec<u32> = lr[0].pattern[0].iter().map(|x| x.0 >> 16).collect::<std::collections::BTreeSet<_>>().into_iter().collect();
    assert!(stretches.len() >= 2, "the rush's two stretches each have places: {stretches:?}");
    for h in [7.0, 16.5] {
        let t0 = 86400.0 + h * 3600.0;
        let (bad, good) = bad_passes(&w, local, express, &n, &n[1..6], t0, t0 + 3.0 * 3600.0, false);
        eprintln!("rush from {h}: {good} passes, {} bad", bad.len());
        assert!(good >= 4, "from {h}: the expresses pass the locals");
        assert!(bad.is_empty(), "from {h}: {} bad passes, first {:?}", bad.len(), &bad[..bad.len().min(6)]);
    }
}

/// Lines running their own weekends: weekday trains timed train by train, weekend trains their
/// slot's timetable (the variants were worked out against the weekday trains).
#[test]
fn own_weekends_run_the_slot_timetable() {
    use super::super::net::Weekend;
    let weekday = 86400.0 + 8.0 * 3600.0;
    // local day 6 is a Saturday (`params::row_of_day`)
    let saturday = 6.0 * 86400.0 + 8.0 * 3600.0;
    let timed = |w: &TrackWorld, l: u32, t: f64| w.svc.line_trips(&w.net, l as usize, t, t + 3.0 * 3600.0).iter().filter(|t| t.var > 0).count();
    // both lines keeping 10-minute headways, so a weekend share runs fewer trains
    let headways = |w: &mut TrackWorld, ids: [u32; 2]| {
        for id in ids {
            let k = w.net.line_trains[id as usize].unwrap_or([6, 6, 0]);
            w.apply(Op::AutoTrains { id, auto: Some([600, 600, 0]), trains: Some(k) }).unwrap();
        }
    };
    let (mut w, n, local, express) = shinkansen();
    headways(&mut w, [local, express]);
    set_passing(&mut w, &n[1..6], true);
    for id in [local, express] {
        w.apply(Op::Weekend { id, weekend: Weekend { share: 50, ..Weekend::default() } }).unwrap();
    }
    for l in [local, express] {
        assert!(w.svc.lattice(&w.net, l as usize).unwrap().rest_on, "line {l} runs its own weekends");
    }
    assert!(timed(&w, local, weekday) > 0, "weekday locals timed train by train");
    assert_eq!(timed(&w, local, saturday), 0, "weekend locals run the slot's timetable");
    // one line's own weekends give the other its own weekend pattern too (the weekends are paced
    // on their own, T-3584), so both run the slot's timetable on Saturdays
    let (mut w, n, local, express) = shinkansen();
    headways(&mut w, [local, express]);
    set_passing(&mut w, &n[1..6], true);
    w.apply(Op::Weekend { id: express, weekend: Weekend { share: 50, ..Weekend::default() } }).unwrap();
    assert!(timed(&w, local, weekday) > 0);
    assert_eq!(timed(&w, local, saturday), 0);
}

/// A game with nothing to time (no passing waits, no timed following) has no variants and every
/// trip runs its slot's timetable: master's figures.
#[test]
fn nothing_to_time_leaves_no_variant() {
    let (mut w, _, local, express) = shinkansen();
    // the express alone: no trains of the local to wait or follow
    w.apply(Op::AutoTrains { id: local, auto: None, trains: Some([0; 3]) }).unwrap();
    for l in [local, express] {
        assert!(w.svc.lines[l as usize].runs.iter().all(|r| r.vars.iter().all(|v| v.is_empty())));
        assert!(w.svc.line_trips(&w.net, l as usize, 86400.0, 2.0 * 86400.0).iter().all(|t| t.var == 0));
    }
}

/// A line rebuilt in place (a reload of a live world) loses its variants with its service; the next
/// retime works them out again even though nothing it reads moved.
#[test]
fn variants_come_back_after_a_reload_in_place() {
    let (mut w, n, local, _) = shinkansen();
    set_passing(&mut w, &n[1..6], true);
    let t0 = 86400.0 + 8.0 * 3600.0;
    let timed = |w: &TrackWorld| w.svc.line_trips(&w.net, local as usize, t0, t0 + 7200.0).iter().filter(|t| t.var > 0).count();
    let before = timed(&w);
    assert!(before > 0);
    w.reload();
    let d = w.net.line_data(local).unwrap();
    w.apply(Op::Line { id: local, data: Some(d) }).unwrap();
    assert_eq!(timed(&w), before);
}

/// T-3346 stage 2's 30/10 (t2233-passing-map): a local whose own timetable runs longer one way than
/// its round trip leaves room for would miss its next departure at that end and stand a whole
/// headway there, holding the expresses behind it. That run keeps the slot's timetable (`floor`),
/// and every local in steady service takes its next departure.
#[test]
fn headway_locals_make_their_next_departure() {
    for (hl, hx) in [(600u32, 600u32), (1800, 600), (600, 1800)] {
        let (mut w, n, local, express) = shinkansen_rates();
        for (id, h) in [(local, hl), (express, hx)] {
            let k = w.net.line_trains[id as usize].unwrap_or([6, 6, 0]);
            w.apply(Op::AutoTrains { id, auto: Some([h, h, 0]), trains: Some(k) }).unwrap();
        }
        set_passing(&mut w, &n[1..6], true);
        let ends: Vec<Option<f64>> = (0..2).map(|r| stand_at_end(&w.net, &w.svc, local as usize, r, 0)).collect();
        let t0 = 86400.0 + 8.0 * 3600.0;
        let trips = w.svc.line_trips(&w.net, local as usize, t0, t0 + 4.0 * 3600.0);
        let lost = trips.iter().filter(|t| t.next.is_none() && t.dep < t0 + 3.0 * 3600.0).count();
        let timed: Vec<usize> = (0..2u8).map(|r| trips.iter().filter(|t| t.run == r && t.var > 0).count()).collect();
        // what 30/10 did before the floor: a train missing its next departure and standing at the
        // end for the one after (half a headway more than the slot's timetable stands there)
        let ls = &w.svc.lines[local as usize];
        let most = ls.close_stand(&w.net, local) + w.svc.spare(&w.net, local as usize, 0) + 0.5 * hl as f64;
        let long = trips.iter().filter(|t| t.next.is_some() && t.end - (t.draw + ls.prof_of(t.run as usize, t.level as usize, t.var).duration) > most).count();
        eprintln!("{}/{}: stands at the ends {ends:.0?}, trips timed per run {timed:?} of {}, {lost} leave service, {long} stand longer than {most:.0} s", hl / 60, hx / 60, trips.len());
        assert_eq!(long, 0, "{}/{}: no local stands a headway at an end", hl / 60, hx / 60);
        assert!(ends.iter().flatten().all(|&s| s >= TURN_MIN_S - 1.0));
        assert!(timed.iter().sum::<usize>() > 0, "{}/{}: some run timed train by train", hl / 60, hx / 60);
    }
}

/// T-3542's moved hours meet stage 2: with the morning from 5 the hour 5-6 is a piece of the night's
/// slot run at the morning's own rate; its locals are timed train by train like the morning's.
#[test]
fn a_moved_morning_hour_is_timed_too() {
    let (mut w, n, local, express) = shinkansen_rates();
    set_passing(&mut w, &n[1..6], true);
    let op = w.rows_ops(None, 5).unwrap();
    w.apply(op).unwrap();
    w.apply(Op::Hours { line: None, cuts: [5, 10, 16, 20, 24] }).unwrap();
    let t0 = 86400.0 + 5.0 * 3600.0;
    let trips = w.svc.line_trips(&w.net, local as usize, t0, t0 + 3600.0);
    let early: Vec<(u8, u8)> = trips.iter().filter(|t| t.dep >= t0 + 600.0).map(|t| (t.level, t.var)).collect();
    eprintln!("5-6 locals (slot, variant): {early:?}");
    assert!(early.iter().any(|x| x.1 > 0), "locals in the moved hour timed train by train");
    let (bad, good) = bad_passes(&w, local, express, &n, &n[1..6], t0, t0 + 2.0 * 3600.0, false);
    eprintln!("5-7: {good} passes, {} bad", bad.len());
    assert!(bad.is_empty(), "{} bad passes, first {:?}", bad.len(), &bad[..bad.len().min(4)]);
}

/// T-3896: the variants are in a line's phase tables, which the page reads again only for the lines
/// noted as changed (`Service::take_changed`, T-1540), so `retime` notes every line it times
/// otherwise, whatever the edit touched (a station renamed on Tokyo's network timed a line far from
/// it again). Unnoted, the page laid the tables out at their old lengths and read the lines after
/// them from the wrong place: the Speed view's fastest train at 856 km/h.
#[test]
fn a_line_timed_otherwise_is_noted_changed() {
    let (mut w, n, local, _) = shinkansen();
    set_passing(&mut w, &n[1..6], true);
    let timed = w.svc.lines[local as usize].runs[0].vars[0].len();
    assert!(timed > 1);
    // the local's variants as if worked out otherwise before: the next retime puts them back
    w.svc.lines[local as usize].runs[0].vars[0].pop();
    w.svc.timing_key.clear();
    w.svc.take_changed();
    w.retime();
    assert_eq!(w.svc.lines[local as usize].runs[0].vars[0].len(), timed);
    assert_eq!(w.svc.take_changed(), vec![local]);
    // timed the same: nothing noted
    w.svc.timing_key.clear();
    w.retime();
    assert!(w.svc.take_changed().is_empty());
}

