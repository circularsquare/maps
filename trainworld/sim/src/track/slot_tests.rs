//! T-3542: the service's slots (`params::SLOTS`). Every line sets three rows, so only the slots
//! some period uses are worked out (`Network::period_slot`); the others stay as made.

use super::bench::synth;
use super::net::Network;
use super::params::*;
use super::world::{Op, TrackWorld};

/// A synthetic network with shared track, two of its lines switched to headways and one line's
/// weekends at 70%, so the settle, the pacing, the spacing and the weekend settle all run.
fn busy(slots: Option<[usize; 5]>) -> TrackWorld {
    let mut net: Network = synth(400.0, 90, 14, 11);
    if let Some(p) = slots {
        // the lines were set under the three-row slots: each row's rate into its slots again
        net.set_period_slot(p);
        for l in 0..net.line_count() {
            net.line_tph[l] = net.rows_in(slots_to_rows(net.line_tph[l]), 0.0);
        }
    }
    let mut w = TrackWorld::new(net);
    w.allocate_unallocated();
    let ok: Vec<u32> = (0..w.net.line_count() as u32).filter(|&l| w.svc.lines[l as usize].ok).collect();
    assert!(ok.len() >= 8, "{} lines run", ok.len());
    for &l in &ok[..2] {
        let trains = w.net.line_trains[l as usize];
        w.apply(Op::AutoTrains { id: l, auto: Some([300, 600, 1200]), trains }).unwrap();
    }
    w.apply(Op::Weekend { id: ok[2], weekend: super::net::Weekend { share: 70, counts: None, headways: [0; 3] } }).unwrap();
    let t = w.net.line_trains[ok[3] as usize].unwrap();
    let tph = w.net.rows_of(w.net.line_tph[ok[3] as usize]);
    w.apply(Op::TrainCounts { id: ok[3], trains: Some([t[0] + 2, t[1] + 1, t[2]]), tph }).unwrap();
    w
}

/// A slot no period uses is never written: its rates, round trips, fleets, delays, holds,
/// profiles, utilisation, pacing and phases keep their defaults through loads and edits, and a
/// day's high periods are one stretch of departures.
#[test]
fn t3542_slots_no_period_uses_stay_as_made() {
    let w = busy(None);
    let idle: Vec<usize> = (0..SLOTS).filter(|&s| !w.net.slot_worked(s)).collect();
    assert_eq!(idle, if SLOTS == 5 { vec![1, 2] } else { vec![] });
    for l in 0..w.net.line_count() {
        let ls = &w.svc.lines[l];
        for &s in &idle {
            assert_eq!(w.net.line_tph[l][s].to_bits(), 0, "line {l} slot {s} rate");
            assert_eq!(w.net.line_tph_rest[l][s].to_bits(), 0, "line {l} slot {s} weekend rate");
            assert_eq!(w.net.line_phase_of(l as u32)[s].to_bits(), 0);
            assert_eq!(w.net.line_pace_of(l as u32)[s].to_bits(), 0);
            assert_eq!((ls.round_trip[s].to_bits(), ls.trains[s], ls.delay[s].to_bits()), (0, 0, 0), "line {l} slot {s}");
            for r in &ls.runs {
                assert!(r.holds[s].is_empty() && r.prof[s].phases.is_empty() && r.prof[s].dep.is_empty(), "line {l} slot {s} profile");
            }
        }
        if let Some(lat) = w.svc.lattice(&w.net, l) {
            let lv = lat.levels_on(3);
            assert!(lv[0] == lv[1] && lv[1] == lv[2] && lv[0] % SLOTS == HIGH, "6-20 is one slot: {lv:?}");
        }
    }
    for r in 0..w.svc.cap.res.len() {
        for &s in &idle {
            assert_eq!((w.svc.cap.rho[r][s].to_bits(), w.svc.cap.delay[r][s].to_bits()), (0, 0), "resource {r} slot {s}");
        }
    }
    assert!(w.net.settle_kept.iter().all(|k| w.net.slot_worked(k.1)));
}

/// Every period worked out in a slot of its own (the cost stage 1 pays once some line sets five
/// rows), with each 3-row line's high count in all three of the day's high slots: the morning,
/// midday and afternoon slots settle exactly as the one high slot does, and evening and night as
/// before, to the bit.
#[test]
fn t3542_five_slots_settle_as_three() {
    let three = busy(None);
    let five = busy(Some([0, 1, 2, 3, 4]));
    assert!((0..5).all(|s| five.net.slot_worked(s)));
    let of = |w: &TrackWorld, l: usize, s: usize| {
        let ls = &w.svc.lines[l];
        (w.net.line_tph[l][s].to_bits(), w.net.line_tph_rest[l][s].to_bits(), ls.round_trip[s].to_bits(), ls.trains[s], ls.delay[s].to_bits(), ls.runs.iter().map(|r| r.prof[s].duration.to_bits()).collect::<Vec<_>>(), w.net.line_pace_of(l as u32)[s].to_bits())
    };
    for l in 0..three.net.line_count() {
        for (s5, s3) in [(0, HIGH), (1, HIGH), (2, HIGH), (3, MEDIUM), (4, LOW)] {
            assert_eq!(of(&five, l, s5), of(&three, l, s3), "line {l}: slot {s5} of five against {s3} of three");
        }
        assert_eq!(five.net.line_trains[l], three.net.line_trains[l], "line {l} counts");
    }
}

/// The lines that run (a line in `busy`'s order).
fn running(w: &TrackWorld) -> Vec<u32> {
    (0..w.net.line_count() as u32).filter(|&l| w.svc.lines[l as usize].ok && w.net.line_trains[l as usize].is_some()).collect()
}

/// Everything a save and its load keep of the service, to the bit.
fn kept(w: &TrackWorld) -> (Vec<u8>, Vec<Vec<u32>>, Vec<(u32, usize, u32, u64, u32)>) {
    let rates = (0..w.net.line_count()).map(|l| w.net.line_tph[l].iter().chain(&w.net.line_tph_rest[l]).map(|x| x.to_bits()).collect()).collect();
    let mut k = w.net.settle_kept.clone();
    k.sort_unstable();
    (super::save::encode(&w.net), rates, k)
}

/// T-3542: a game of five rows, one line's midday its own, saves and loads to the bit (`LPD1`), and
/// every line's rows and slots come back.
#[test]
fn t3542_five_rows_round_trip() {
    let mut w = busy(None);
    let op = w.rows_ops(None, 5).unwrap();
    w.apply(op).unwrap();
    assert_eq!(w.net.period_slot(), [0, 1, 2, 3, 4]);
    let l = running(&w)[3] as usize;
    let mut x = w.net.line_x[l];
    x.trains[0] += 2;
    w.apply(Op::Extra { id: l as u32, x }).unwrap();
    assert_eq!(w.net.trains_at(l, 1), w.net.trains_at(l, 0) + 2, "midday its own");
    // the fleet (what the line is charged for, `cars_needed`) is the most of every slot's
    assert_eq!(w.svc.lines[l].trains.iter().max(), Some(&w.net.trains_at(l, 1)), "the fleet counts midday's");
    let bytes = super::save::encode(&w.net);
    let back = TrackWorld::new(super::save::decode(&bytes).unwrap());
    assert_eq!(back.net.period_slot(), [0, 1, 2, 3, 4]);
    assert_eq!(back.net.rows_default, 5);
    for k in 0..w.net.line_count() as u32 {
        assert_eq!(back.net.line_data(k), w.net.line_data(k), "line {k}");
    }
    assert_eq!(kept(&back), kept(&w));
}

/// T-3542: switching every line to five rows and back leaves the game as it was, to the bit; and an
/// old save (no `LPD1`) loads as three rows.
#[test]
fn t3542_five_and_back_as_never() {
    let a = busy(None);
    let mut b = busy(None);
    let op = b.rows_ops(None, 5).unwrap();
    b.apply(op).unwrap();
    let op = b.rows_ops(None, 3).unwrap();
    b.apply(op).unwrap();
    assert_eq!(b.net.period_slot(), PERIOD_SLOT_3);
    assert_eq!(kept(&b), kept(&a));
    let bytes = super::save::encode(&a.net);
    assert!(!bytes.windows(4).any(|w| w == b"LPD1"), "a game of three rows writes no LPD1");
    let old = super::save::decode(&bytes).unwrap();
    assert_eq!((old.rows_default, old.period_slot()), (3, PERIOD_SLOT_3));
}

/// T-3542: switching up copies a row into the rows it splits into; switching down gives a joined row
/// the most trains of its rows, and the shortest headway; undo puts the five rows back.
#[test]
fn t3542_switch_up_copies_down_takes_most() {
    let mut w = busy(None);
    let ls = running(&w);
    let (count, auto) = (ls[3] as usize, ls[0] as usize);
    let before = (w.net.trains_of(count).unwrap(), w.net.auto_of(auto).unwrap());
    let op = w.rows_ops(Some(count as u32), 5).unwrap();
    w.apply(op).unwrap();
    let op = w.rows_ops(Some(auto as u32), 5).unwrap();
    w.apply(op).unwrap();
    let t = w.net.trains_of(count).unwrap();
    assert_eq!((t[3], t[4]), (before.0[0], before.0[0]), "midday and afternoon copy high");
    let a = w.net.auto_of(auto).unwrap();
    assert_eq!((a[3], a[4]), (before.1[0], before.1[0]));
    // the afternoon busier, the midday's headway shorter
    let mut x = w.net.line_x[count];
    x.trains[1] = t[0] + 3;
    w.apply(Op::Extra { id: count as u32, x }).unwrap();
    let mut x = w.net.line_x[auto];
    x.auto[0] = a[0] / 2;
    w.apply(Op::Extra { id: auto as u32, x }).unwrap();
    let five = (w.net.trains_of(count).unwrap(), w.net.auto_of(auto).unwrap());
    let op = w.rows_ops(Some(count as u32), 3).unwrap();
    w.apply(op).unwrap();
    let op = w.rows_ops(Some(auto as u32), 3).unwrap();
    w.apply(op).unwrap();
    assert_eq!(w.net.trains_of(count).unwrap()[0], five.0[0].max(five.0[3]).max(five.0[4]), "the most of the three");
    assert_eq!(w.net.auto_of(auto).unwrap()[0], five.1[0].min(five.1[3]).min(five.1[4]), "the shortest headway");
    assert_eq!(w.net.period_slot(), PERIOD_SLOT_3);
    w.undo().unwrap().unwrap();
    w.undo().unwrap().unwrap();
    assert_eq!((w.net.trains_of(count).unwrap(), w.net.auto_of(auto).unwrap()), five, "undo puts them back");
    assert_eq!(w.net.period_slot(), [0, 1, 2, 3, 4]);
}

/// T-3542: a line of three rows in a game of five joins its morning, midday and afternoon into one
/// stretch of departures while they run alike; a line of five rows keeps them apart.
#[test]
fn t3542_three_rows_join_in_five() {
    let mut w = busy(None);
    let ls = running(&w);
    let five = ls[3];
    let op = w.rows_ops(Some(five), 5).unwrap();
    w.apply(op).unwrap();
    assert_eq!(w.net.period_slot(), [0, 1, 2, 3, 4]);
    let mut joined = 0;
    for &l in &ls {
        let Some(lat) = w.svc.lattice(&w.net, l as usize) else { continue };
        let lv = lat.levels_on(3);
        if l == five {
            assert!(lv[0] != lv[1] && lv[1] != lv[2], "five rows apart: {lv:?}");
        } else if lv[0] == lv[1] && lv[1] == lv[2] {
            joined += 1;
        }
    }
    assert!(joined >= ls.len() / 2, "{joined} of {} lines of three rows run 6-20 as one", ls.len() - 1);
}
