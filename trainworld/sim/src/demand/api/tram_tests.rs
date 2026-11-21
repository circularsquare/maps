//! T-2926: tram lines in demand.

use super::tests::small;
use super::*;

/// `small()`'s two metro lines plus one more line of `n` stops from (x0, y) in `step` m steps along x,
/// 90 s a hop, 8 trains an hour; returns the arguments `set_network` takes and the new line's index.
#[allow(clippy::type_complexity)]
fn with_line(n: usize, x0: f32, y: f32, step: f32, cars: f32) -> (DemandApi, Vec<f32>, Vec<u32>, Vec<u32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let (api, mut xy, mut counts, mut stops, mut times, mut tph, mut c) = small();
    let first = (xy.len() / 2) as u32;
    for k in 0..n {
        xy.extend([x0 + k as f32 * step, y]);
    }
    stops.extend(first..first + n as u32);
    counts.push(n as u32);
    for run in 0..2 {
        for _lev in 0..3 {
            for k in 0..n {
                let end = if run == 0 { k == n - 1 } else { k == 0 };
                times.push(if end { 0.0 } else { 90.0 });
            }
        }
    }
    tph.extend([8.0, 8.0, 4.0]);
    c.push(cars);
    (api, xy, counts, stops, times, tph, c)
}

/// A cell keeps at most `TRAM_SLOTS` tram-only stops among its K nearest, and the metro stop a short
/// walk farther off stays on its list.
#[test]
fn a_street_of_tram_stops_does_not_crowd_out_the_metro() {
    // five tram stops 150 m apart across the centre, 100 m north of it
    let (mut api, xy, counts, stops, times, tph, cars) = with_line(5, -300.0, 300.0, 150.0, 2.0);
    api.set_tram_lines(&[0, 0, 1]);
    api.set_network(&xy, &counts, &stops, &times, &tph, &cars).unwrap();
    let net = DemandApi::period_net(api.svc.as_ref().unwrap(), 0);
    let trams: Vec<usize> = (0..net.st_x.len()).filter(|&s| net.tram_only(s)).collect();
    assert_eq!(trams.len(), 5, "only the five new stops are tram-only");
    let acc = build_access(&api.city, &net, &api.base);
    let mut near_centre = 0;
    for c in 0..api.city.len() {
        let list = &acc.st[c * acc.w..c * acc.w + acc.n[c] as usize];
        assert!(list.iter().filter(|&&s| net.tram_only(s as usize)).count() <= TRAM_SLOTS, "cell {c}");
        if (api.city.x[c].powi(2) + api.city.y[c].powi(2)).sqrt() < 400.0 {
            near_centre += 1;
            assert!(list.iter().any(|&s| !net.tram_only(s as usize) && (net.st_x[s as usize].abs() + net.st_y[s as usize].abs()) < 1.0), "the metro stop at the centre stays, cell {c}");
            assert!(list.iter().filter(|&&s| !net.tram_only(s as usize)).count() >= acc.w - TRAM_SLOTS, "cell {c} keeps its other stations");
        }
    }
    assert!(near_centre > 0);
    // without the flags the same stops are ordinary and fill the list
    let (mut plain, xy, counts, stops, times, tph, cars) = with_line(5, -300.0, 300.0, 150.0, 2.0);
    plain.set_network(&xy, &counts, &stops, &times, &tph, &cars).unwrap();
    let net = DemandApi::period_net(plain.svc.as_ref().unwrap(), 0);
    assert!((0..net.st_x.len()).all(|s| !net.tram_only(s)));
}

/// A tram has three quarters of a car per section, and a tram stop that a metro line also serves is
/// not a tram-only stop.
#[test]
fn tram_sections_are_three_quarters_of_a_car() {
    let (mut api, xy, counts, stops, times, tph, cars) = with_line(5, -300.0, 300.0, 150.0, 2.0);
    api.set_tram_lines(&[0, 0, 1]);
    api.set_network(&xy, &counts, &stops, &times, &tph, &cars).unwrap();
    assert_eq!(api.svc.as_ref().unwrap().cars, vec![10.0, 8.0, 1.5]);
    // another tram line through a metro station makes that station an ordinary one
    let (mut api, xy, counts, stops, times, tph, cars) = with_line(5, -300.0, 0.0, 150.0, 2.0);
    api.set_tram_lines(&[0, 0, 1]);
    api.set_network(&xy, &counts, &stops, &times, &tph, &cars).unwrap();
    let net = DemandApi::period_net(api.svc.as_ref().unwrap(), 0);
    assert_eq!((0..net.st_x.len()).filter(|&s| net.tram_only(s)).count(), 4, "the stop merged into the metro station is not tram-only");
}

/// A tram along a corridor carries at least what the bus it replaces does when the two board alike: same
/// stops, same trains an hour, same hop times; the bus is a third of a car, the tram two sections of three
/// quarters. Since T-3910 a tram trip pays the 2 + 2 boarding minutes at its ends and a bus trip none (T-3856), so
/// with the cost on the tram carries less than the bus on this corridor, but still a good part of it.
#[test]
fn a_tram_carries_at_least_the_bus_it_replaces() {
    let load = |tram: bool, boarding: bool| {
        let cars = if tram { 2.0 } else { crate::demand::network::BUS_CARS };
        let (mut api, xy, counts, stops, times, tph, c) = with_line(8, -3500.0, 600.0, 1000.0, cars);
        if tram {
            api.set_tram_lines(&[0, 0, 1]);
        }
        if !boarding {
            api.base.train_in_min = 0.0;
            api.base.train_out_min = 0.0;
        }
        // (T-3888: its stops on the street, at grade, as a bus's; the metro's below)
        let grade: Vec<u8> = (0..stops.len()).map(|k| (k >= stops.len() - 8) as u8).collect();
        api.set_stairs(&[], &grade);
        api.set_network(&xy, &counts, &stops, &times, &tph, &c).unwrap();
        assert!(api.solve(0));
        // the third line's route nodes are the last 16
        let b = api.board(0);
        b[b.len() - 16..].iter().sum::<f32>()
    };
    // boarding alike (none): the tram carries at least the bus
    let (bus, tram) = (load(false, false), load(true, false));
    assert!(tram >= bus * 0.9999, "no boarding cost: tram {tram} against bus {bus}");
    // with the boarding cost on the tram only: less than the bus, not by more than half
    let (bus, tram) = (load(false, true), load(true, true));
    assert!(tram >= bus * 0.5, "boarding cost: tram {tram} against bus {bus}");
}

/// T-3903: a per-line or per-stop input of the wrong length is an error, not read as none (the page's copy of a
/// field for a view once dropped the tram flags unseen); empty is none.
#[test]
fn a_per_line_input_of_the_wrong_length_is_an_error() {
    let (mut api, xy, counts, stops, times, tph, cars) = with_line(5, -300.0, 300.0, 150.0, 2.0);
    api.set_tram_lines(&[1, 0]);
    let err = api.set_network(&xy, &counts, &stops, &times, &tph, &cars).unwrap_err();
    assert!(err.contains("tram lines: 2 entries, one per line wants 3"), "{err}");
    api.set_tram_lines(&[]);
    api.set_stairs(&[], &[1, 0]);
    let err = api.set_network(&xy, &counts, &stops, &times, &tph, &cars).unwrap_err();
    assert!(err.contains("stops at grade"), "{err}");
    api.set_stairs(&[], &[]);
    api.set_network(&xy, &counts, &stops, &times, &tph, &cars).unwrap();
}

/// T-3909: the everyday layer, given the tram flags as the worker gives both populations, solves on the network the
/// commutes do (a tram's three quarters of a car, the same tram-only stops); with no flags it would hold the tram as
/// a train, and its tram riders differ.
#[test]
fn everyday_trips_see_the_tram_lines_as_commutes_do() {
    use crate::demand::everyday::EverydayApi;
    // the small pack with errand and leisure places: a cell's jobs each
    let (h, mut b) = super::tests::small_pack();
    let jobs = DemandApi::new(&h, &b).unwrap().city.jobs.clone();
    let mut h: serde_json::Value = serde_json::from_str(&h).unwrap();
    for name in ["errand_jobs", "leisure_jobs"] {
        let offset = b.len();
        b.extend(jobs.iter().flat_map(|v| v.to_le_bytes()));
        h["arrays"].as_array_mut().unwrap().push(serde_json::json!({"name": name, "dtype": "f32", "per": "cell", "count": jobs.len(), "offset": offset}));
    }
    let h = h.to_string();
    let open = |tram: bool| {
        let (mut api, xy, counts, stops, times, tph, cars) = with_line(5, -300.0, 300.0, 150.0, 2.0);
        let mut ev = EverydayApi::beside_with(&api, &h, &b, &[]).unwrap();
        if tram {
            api.set_tram_lines(&[0, 0, 1]);
            ev.set_tram_lines(&[0, 0, 1]);
        }
        api.set_network(&xy, &counts, &stops, &times, &tph, &cars).unwrap();
        ev.set_network(&xy, &counts, &stops, &times, &tph, &cars).unwrap();
        (api, ev)
    };
    let (api, mut ev) = open(true);
    let (a, e) = (api.svc.as_ref().unwrap(), ev.engine().svc.as_ref().unwrap());
    assert_eq!(e.cars, vec![10.0, 8.0, 1.5], "the everyday layer's tram: three quarters of a car a section");
    assert_eq!(e.st_tram, a.st_tram);
    assert_eq!(e.st_tram.iter().filter(|&&t| t).count(), 5);
    assert!(e.same_layout(a), "the everyday layer's network is the commutes'");
    let (_, mut plain) = open(false);
    assert!(ev.solve(0) && plain.solve(0));
    // the tram line's route nodes are the last 10
    let on = |b: Vec<f32>| b[b.len() - 10..].iter().sum::<f32>();
    let (with, without) = (on(ev.board(0)), on(plain.board(0)));
    assert!(with > 0.0 && (with - without).abs() > 1e-3 * without, "tram riders {with} with the flags, {without} without");
}

/// Solve time with and without a tram line of 20 stops on top of the 34 metro ones (printed, not
/// asserted: timings are noisy). `cargo test --release --features demand --lib tram_solve_time -- --ignored --nocapture`.
#[test]
#[ignore]
fn tram_solve_time() {
    let time = |tram: bool| {
        let (mut api, xy, counts, stops, times, tph, cars) = with_line(20, -2000.0, 300.0, 200.0, 2.0);
        if tram {
            api.set_tram_lines(&[0, 0, 1]);
        }
        let (mut xy, mut counts, mut stops, mut times, mut tph, mut cars) = (xy, counts, stops, times, tph, cars);
        if !tram {
            // the metro alone
            xy.truncate(xy.len() - 40);
            counts.pop();
            stops.truncate(stops.len() - 20);
            times.truncate(times.len() - 120);
            tph.truncate(tph.len() - 3);
            cars.pop();
        }
        api.set_network(&xy, &counts, &stops, &times, &tph, &cars).unwrap();
        let t = std::time::Instant::now();
        for q in 0..5 {
            assert!(api.solve(q));
        }
        t.elapsed().as_secs_f64()
    };
    let (a, b) = (time(false), time(true));
    println!("metro only {a:.3} s, with 20 tram stops {b:.3} s, ratio {:.2}, stops 34 -> 54 ({:.2})", b / a, 54.0 / 34.0);
}
