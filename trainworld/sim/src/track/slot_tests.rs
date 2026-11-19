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
