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

/// A tram along a corridor beats the bus it replaces: same stops, same trains an hour, same hop times;
/// the bus is a third of a car with the bus boarding cost, the tram two sections of three quarters.
#[test]
fn a_tram_carries_more_than_the_bus_it_replaces() {
    let load = |tram: bool| {
        let cars = if tram { 2.0 } else { crate::demand::network::BUS_CARS };
        let (mut api, xy, counts, stops, times, tph, c) = with_line(8, -3500.0, 600.0, 1000.0, cars);
        if tram {
            api.set_tram_lines(&[0, 0, 1]);
        }
        api.set_network(&xy, &counts, &stops, &times, &tph, &c).unwrap();
        assert!(api.solve(0));
        // the third line's route nodes are the last 16
        let b = api.board(0);
        b[b.len() - 16..].iter().sum::<f32>()
    };
    let (bus, tram) = (load(false), load(true));
    assert!(tram > bus, "tram {tram} against bus {bus}");
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
