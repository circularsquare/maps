use super::*;
use crate::demand::{network, synth};

/// T-434: the first round always runs; a later one is skipped once the loads move under 3%.
#[test]
fn crowding_rounds_stop_once_settled() {
    let mut seen = vec![];
    assert!(!settled(&mut seen, &[100.0, 100.0]));
    assert!(!settled(&mut seen, &[100.0, 90.0]));
    assert!(settled(&mut seen, &[100.0, 87.0]));
    assert!(seen == [100.0, 90.0]);
    assert!(!settled(&mut seen, &[100.0, 80.0]));
}

fn common_test_net(lines: &[&[u32]], tph: &[f32], ride: &[f32]) -> Network {
    let mut net = Network {
        st_x: vec![0.0, 2000.0, 4000.0, 4000.0], st_y: vec![0.0, 0.0, 0.0, 2000.0],
        line_name: vec![String::new(); lines.len()], line_off: vec![0], line_stops: vec![],
        hop_min: vec![], hop_back: vec![], line_speed_kmh: vec![0.0; lines.len()],
        line_headway_min: tph.iter().map(|t| [if *t > 0.0 { 60.0 / t } else { f32::INFINITY }; network::PERIODS]).collect(),
        line_seats: vec![100.0; lines.len()], line_rated: vec![], line_crush: vec![200.0; lines.len()],
        st_group: vec![], stop_x: vec![], stop_y: vec![], line_wrap: vec![], span: vec![], st_ups: vec![], st_passage: vec![], catches: vec![], st_bus: vec![], st_tram: vec![], back_stops: vec![], line_pier: vec![], stop_street: vec![], bus_walks: None,
    };
    for (l, stops) in lines.iter().enumerate() {
        net.line_stops.extend_from_slice(stops);
        net.line_off.push(net.line_stops.len() as u32);
        net.hop_min.extend((0..stops.len()).map(|k| if k + 1 < stops.len() { ride[l] } else { 0.0 }));
    }
    net
}

/// T-4050: `net` with line `l`'s `i`-th stop at a platform of its own, 8 m from its station, so
/// lines on the same stops through it are no evenly spaced service there (both of its hops).
fn platform_apart(mut net: Network, l: usize, i: usize) -> Network {
    if net.stop_x.is_empty() {
        net.stop_x = net.line_stops.iter().map(|&st| net.st_x[st as usize]).collect();
        net.stop_y = net.line_stops.iter().map(|&st| net.st_y[st as usize]).collect();
    }
    net.stop_x[net.line_off[l] as usize + i] += 8.0;
    net
}

fn common_test_params() -> Params {
    // (without a train's way in and out, which `a_train_trip_pays_its_way_in_and_out` tests)
    Params { xfer_walk_m: 0.0, train_in_min: 0.0, train_out_min: 0.0, ..Params::default() }
}

/// T-4041: two departures every `h` minutes, `d` and `d + x` from boarding (x < h): the expected
/// cost from the platform and the slower one's share. Of the riders who take the first to arrive,
/// (h - x)^2 / 2h^2 take it, at d + h/2 - (h - x)^3 / 6h^2; of those who board the first train
/// that can still arrive by d + h, y (1 - y / 2h) / h with y = h - x, waiting
/// y - y^2 / h + y^3 / 3h^2 + x^3 / 2h^2.
fn first_two(d: f32, x: f32, h: f32) -> (f32, f32) {
    let (a, y) = (FIRST_ARRIVAL_SHARE, h - x);
    let (best, best_share) = (d + h / 2.0 - y.powi(3) / (6.0 * h * h), y * y / (2.0 * h * h));
    let lazy_share = y * (1.0 - y / (2.0 * h)) / h;
    let lazy = d + lazy_share * x + y - y * y / h + y.powi(3) / (3.0 * h * h) + x.powi(3) / (2.0 * h * h);
    (a * best + (1.0 - a) * lazy, a * best_share + (1.0 - a) * lazy_share)
}

/// T-4041: `pool_cost` worked out on a grid of every line's wait (`n` steps each), unplanned at
/// wait factor 1: the cost from the platform, each line's share and the expected wait.
fn pool_by_grid(lines: &[(f32, f32)], n: usize) -> (f64, Vec<f64>, f64) {
    let a = FIRST_ARRIVAL_SHARE as f64;
    let l: Vec<(f64, f64)> = lines.iter().map(|&(d, f)| (d as f64, 60.0 / f as f64)).collect();
    let cut = l.iter().map(|x| x.0 + x.1).fold(f64::INFINITY, f64::min);
    let (mut cost, mut share, mut wait) = (0.0, vec![0.0; l.len()], 0.0);
    let mut w = vec![0.0; l.len()];
    let cells = n.pow(l.len() as u32);
    for c in 0..cells {
        let mut k = c;
        for (j, x) in l.iter().enumerate() {
            w[j] = ((k % n) as f64 + 0.5) / n as f64 * x.1;
            k /= n;
        }
        let best = (0..l.len()).filter(|&j| l[j].0 < cut).min_by(|&i, &j| (l[i].0 + w[i]).total_cmp(&(l[j].0 + w[j]))).unwrap();
        let lazy = (0..l.len()).filter(|&j| l[j].0 + w[j] <= cut).min_by(|&i, &j| w[i].total_cmp(&w[j])).unwrap();
        cost += a * (l[best].0 + w[best]) + (1.0 - a) * (l[lazy].0 + w[lazy]);
        wait += a * w[best] + (1.0 - a) * w[lazy];
        share[best] += a;
        share[lazy] += 1.0 - a;
    }
    let m = cells as f64;
    (cost / m, share.iter().map(|x| x / m).collect(), wait / m)
}

/// T-4041: riders turning up at random take whichever line gets them to their stop first (three
/// quarters) or the first train that comes (the rest), each line's wait uniform over its headway
/// and independent of the others' (worked out in notes/T-4019.md and T-4041.md). Newark to
/// Trenton, local 41.2 minutes every 22.6 and express 31.2 every 35.9: the express 59% of the first
/// group, (11.3 + 10) / 35.9, 31% of the second, 52% in all. A tram 10 minutes every 10 beside a
/// bus 13 every 5: the tram 55% and 25%, 47.5%. Equal rides, every 7.5 and every 3.75: 25% on the
/// less frequent in both groups.
#[test]
fn riders_take_the_line_that_gets_them_there_first() {
    let p = common_test_params();
    let mut w = PoolWork::default();
    let share = |lines: &[(f32, f32)], w: &mut PoolWork| {
        let mut sh = vec![0f32; lines.len()];
        let (c, wait) = pool_cost(lines, 1.0, 1.0, false, &p, w, Some(&mut sh));
        (sh, c - p.change_at(1.0), wait)
    };
    let a = FIRST_ARRIVAL_SHARE;
    let newark = [(41.2, 60.0 / 22.6), (31.2, 60.0 / 35.9)];
    let (sh, _, _) = share(&newark, &mut w);
    let express = a * (11.3 + 10.0) / 35.9 + (1.0 - a) * 22.6 / 71.8;
    assert!((sh[1] - express).abs() < 1e-4 && (sh[1] - 0.524).abs() < 1e-3, "{sh:?}");
    let (sh, _, _) = share(&[(10.0, 6.0), (13.0, 12.0)], &mut w);
    assert!((sh[0] - (a * 0.55 + (1.0 - a) * 0.25)).abs() < 1e-4 && (sh[0] - 0.475).abs() < 1e-4, "{sh:?}");
    // by a grid of the waits, every case below: cost, shares and wait
    let grid = |lines: &[(f32, f32)], n: usize, tol: f64, w: &mut PoolWork| {
        let (sh, c, wait) = share(lines, w);
        let (gc, gsh, gw) = pool_by_grid(lines, n);
        assert!((c as f64 - gc).abs() < tol * 30.0 && (wait as f64 - gw).abs() < tol * 30.0, "{lines:?}: {c} {wait} against {gc} {gw}");
        for (x, y) in sh.iter().zip(&gsh) {
            assert!((*x as f64 - y).abs() < tol, "{lines:?}: {sh:?} against {gsh:?}");
        }
    };
    grid(&newark, 1000, 2e-3, &mut w);
    grid(&[(10.0, 6.0), (13.0, 12.0)], 1000, 2e-3, &mut w);
    // her case (notes/T-4041.md): an express every 40, a local every 10 35 and 38 minutes slower
    let (sh, _, _) = share(&[(0.0, 1.5), (35.0, 6.0)], &mut w);
    assert!((sh[1] - (a * 0.03125 + (1.0 - a) * (5.0 - 25.0 / 80.0) / 10.0)).abs() < 1e-4, "{sh:?}");
    grid(&[(0.0, 1.5), (38.0, 6.0)], 1000, 2e-3, &mut w);
    grid(&[(41.2, 60.0 / 22.6), (31.2, 60.0 / 35.9), (36.0, 3.0)], 120, 1e-2, &mut w);
    grid(&[(5.0, 6.0), (9.0, 2.0), (12.0, 12.0)], 120, 1e-2, &mut w);
    let (sh, _, _) = share(&[(5.0, 8.0), (5.0, 16.0)], &mut w);
    assert!((sh[0] - 0.25).abs() < 1e-4, "{sh:?}");
    // both every 30, one 10 minutes faster
    let (sh, _, _) = share(&[(20.0, 2.0), (30.0, 2.0)], &mut w);
    assert!((sh[0] - (1.0 - first_two(20.0, 10.0, 30.0).1)).abs() < 1e-4, "{sh:?}");
    // two lines every 10 with the same ride wait a third of the headway, not a quarter: they are
    // independent; and the cost is the shares' rides plus the expected wait
    let (sh, c, wait) = share(&[(4.0, 6.0), (4.0, 6.0)], &mut w);
    assert!((c - (4.0 + 10.0 / 3.0)).abs() < 1e-4 && (wait - 10.0 / 3.0).abs() < 1e-4 && (sh[0] - 0.5).abs() < 1e-6, "{c} {wait} {sh:?}");
    let lines = [(41.2, 60.0 / 22.6), (31.2, 60.0 / 35.9), (36.0, 3.0)];
    let (sh, c, wait) = share(&lines, &mut w);
    let ride: f32 = lines.iter().zip(&sh).map(|(l, s)| l.0 * s).sum();
    assert!((ride + wait - c).abs() < 1e-3, "{ride} + {wait} against {c}");
    assert!((sh.iter().sum::<f32>() - 1.0).abs() < 1e-5);
    // the order given changes nothing; a line slower than another's ride and whole headway is never first
    let (sh2, c2, _) = share(&[lines[2], lines[0], lines[1]], &mut w);
    assert_eq!((c2.to_bits(), [sh2[1], sh2[2], sh2[0]]), (c.to_bits(), [sh[0], sh[1], sh[2]]));
    let (sh, _, _) = share(&[(10.0, 6.0), (20.5, 12.0)], &mut w);
    assert_eq!(sh, [1.0, 0.0]);
    // one line: the plain and the planned wait as before
    assert_eq!(pool_cost(&[(10.0, 1.0)], 1.0, 1.0, false, &p, &mut w, None).0, 10.0 + board_cost(60.0, 1.0, 1.0, &p));
    assert_eq!(pool_cost(&[(10.0, 1.0)], 1.0, 1.0, true, &p, &mut w, None).0, 10.0 + planned_board_cost(60.0, 1.0, 1.0, &p));
    // her case in the search (notes/T-4041.md): an express every 40 and a local every 10 that is
    // 25 minutes slower; the local is offered after station 0 has left the queue and is loaded
    // all the same, 40.6% (25% of those who take the first to arrive, 87.5% of the others)
    let net = common_test_net(&[&[0, 1], &[0, 1]], &[1.5, 6.0], &[10.0, 35.0]);
    let g = build_graph(&net, &p);
    let sk = skims(&g, &p);
    let mut od = vec![0.0; g.n_st * g.n_st];
    od[1] = 100.0;
    let loads = assign(&g, &sk, &od);
    let local = loads.rn_board[4];
    assert!((local - 100.0 * (a * 0.25 + (1.0 - a) * 0.875)).abs() < 0.01 && (loads.rn_board[0] + local - 100.0).abs() < 1e-3, "{:?}", loads.rn_board);
    let (sh, c, _) = share(&[(10.0, 1.5), (35.0, 6.0)], &mut w);
    assert!((sk.r[1] - c).abs() < 1e-4 && (sh[1] * 100.0 - local).abs() < 0.01, "{} against {c}", sk.r[1]);
    // the search and the loading: Newark to Trenton on two lines from 0 to 1, 100 riders
    let net = common_test_net(&[&[0, 1], &[0, 1]], &[60.0 / 22.6, 60.0 / 35.9], &[41.2, 31.2]);
    let g = build_graph(&net, &p);
    let sk = skims(&g, &p);
    let s = g.n_st;
    let mut od = vec![0.0; s * s];
    od[1] = 100.0;
    let loads = assign(&g, &sk, &od);
    let (local, express) = (loads.rn_board[0], loads.rn_board[4]);
    let (sh, c, _) = share(&newark, &mut w);
    assert!((express - 100.0 * sh[1]).abs() < 0.01 && (local + express - 100.0).abs() < 1e-3, "{local} {express}");
    assert!((sk.r[1] - c).abs() < 1e-4, "{} against {c}", sk.r[1]);
    // the trip explorer splits them the same
    let r = super::super::trip::routes(&g, &sk, &p, 0, 1, 0.005);
    assert_eq!(r.len(), 2);
    assert!((r[0].0 - sh[1]).abs() < 1e-5, "{}", r[0].0);
}

/// T-3888, T-3910: a trip pays `train_in_min` where its first boarding is a train or tram (at any
/// level, but not at a tram stop on the street, T-4041) and `train_out_min` where its last ride leaves one, never at a
/// change; a bus and a ferry pay neither; the real minutes of the walking rule are untouched;
/// the trip explorer's legs carry them and still add up.
#[test]
fn a_train_trip_pays_its_way_in_and_out() {
    let plain = common_test_params();
    let p = Params { train_in_min: 2.0, train_out_min: 3.0, ..plain.clone() };
    // line A 0-1, B 1-2, 12 tph and 2 minutes each
    let mut net = common_test_net(&[&[0, 1], &[1, 2]], &[12.0, 12.0], &[2.0, 2.0]);
    let s = net.n_stations();
    let base = skims(&build_graph(&net, &plain), &plain);
    let g = build_graph(&net, &p);
    let sk = skims(&g, &p);
    let at = |sk: &Skims, a: usize, b: usize| sk.r[a * s + b];
    assert!((at(&sk, 0, 1) - at(&base, 0, 1) - 5.0).abs() < 1e-4, "one ride: in and out");
    assert!((at(&sk, 0, 2) - at(&base, 0, 2) - 5.0).abs() < 1e-4, "a change pays neither: {} {}", at(&sk, 0, 2), at(&base, 0, 2));
    assert_eq!(sk.walk, base.walk, "the walking rule's real minutes");
    let legs = &super::super::trip::routes(&g, &sk, &p, 0, 2, 0.005)[0].1;
    assert_eq!((legs[0].ends, legs[legs.len() - 1].ends), (2.0, 3.0));
    let total: f32 = legs.iter().map(|l| l.perc + l.wait_perc + l.change + l.ends).sum();
    assert!((total - at(&sk, 0, 2)).abs() < 1e-4, "the legs add up to the strategy: {total}");
    // B a bus: the trip onto it from A pays the way in only, B alone neither
    net.line_crush[1] = crate::demand::network::BUS_CARS * super::super::api::CRUSH_PER_CAR;
    let (b0, b1) = (skims(&build_graph(&net, &plain), &plain), skims(&build_graph(&net, &p), &p));
    assert!((at(&b1, 0, 2) - at(&b0, 0, 2) - 2.0).abs() < 1e-4 && (at(&b1, 1, 2) - at(&b0, 1, 2)).abs() < 1e-4);
    assert!((at(&b1, 2, 0) - at(&b0, 2, 0) - 3.0).abs() < 1e-4, "from the bus onto the train: the way out only");
    // A a ferry, B a train again
    net.line_crush[1] = net.line_crush[0];
    net.line_pier = vec![true, false];
    let (t0, t1) = (skims(&build_graph(&net, &plain), &plain), skims(&build_graph(&net, &p), &p));
    assert!((at(&t1, 0, 1) - at(&t0, 0, 1)).abs() < 1e-4 && (at(&t1, 0, 2) - at(&t0, 0, 2) - 3.0).abs() < 1e-4);
    // A a tram again, its stop at station 0 on the street (stops: A 0, 1; B 1, 2): the way in from 0
    // is free, the way out at 1 is paid, and the trip from 2 pays its way in on B (T-4041)
    net.line_pier = vec![];
    net.stop_street = vec![true, false, false, false];
    let (g0, g1) = (skims(&build_graph(&net, &plain), &plain), skims(&build_graph(&net, &p), &p));
    let d = |a: usize, b: usize| at(&g1, a, b) - at(&g0, a, b);
    assert!((d(0, 1) - 3.0).abs() < 1e-4 && (d(1, 0) - 2.0).abs() < 1e-4 && (d(2, 1) - 5.0).abs() < 1e-4);
    // nothing to pay: no per-node table at all
    assert!(build_graph(&net, &plain).rn_in.is_empty());
    net.stop_street = vec![true; 4];
    assert!(build_graph(&net, &p).rn_in.is_empty());
}

/// T-3856: boarding a bus costs no more than boarding a train, in the cost of boarding and in each
/// crowding round's.
#[test]
fn a_bus_line_boards_as_a_train_does() {
    let p = common_test_params();
    let mut net = common_test_net(&[&[0, 2], &[0, 1]], &[6.0, 6.0], &[2.0, 2.0]);
    net.st_bus = vec![false, true, false, false];
    assert!(net.bus_line(1));
    let g = build_graph(&net, &p);
    let nr = g.rn_line.len();
    assert!(g.rn_left.iter().all(|&l| l == 0.0));
    let (_, left) = round_costs(&g, &net, &p, &vec![0.0; nr]).unwrap();
    assert!(left.iter().all(|&l| l == 0.0));
}

/// T-1577: a bus between two train stations has no bus-only stop, and is a bus all the same: the
/// snapshot's cars for a bus, whose crush load is a bus's, mark it.
#[test]
fn a_short_bus_between_two_train_stations_is_a_bus() {
    let mut net = common_test_net(&[&[0, 2], &[0, 1]], &[6.0, 6.0], &[2.0, 2.0]);
    net.line_crush[1] = network::BUS_CARS * crate::demand::api::CRUSH_PER_CAR;
    assert!(net.st_bus.is_empty() && net.bus_line(1) && !net.bus_line(0));
    // both its stations have the train line too; a bus alone would give them none
    assert_eq!(on_train(&net), vec![true, false, true, false]);
    net.line_crush[0] = network::BUS_CARS * crate::demand::api::CRUSH_PER_CAR;
    assert_eq!(on_train(&net), vec![false; 4]);
}

/// T-883, T-1143: a fare that steepens past 30 km raises only the longer pairs' costs, by the extra
/// rate on the km past 30, with two lines pooled at the shared stations; a curve on the straight
/// line changes nothing. The assignment puts each pair's extra where its riders first board, split
/// as they split, and it adds up to the trips times the extra at their km.
#[test]
fn a_bent_fare_curve_charges_long_trips_and_counts_its_extra() {
    let p = common_test_params();
    let net = common_test_net(&[&[0, 1, 2, 3], &[0, 1, 2]], &[6.0, 12.0], &[2.0, 2.0]);
    let (base, rate, steep) = (4.5f32, 0.3f32, 0.9f32);
    let graph = |knots: &[(f32, f32)]| {
        let mut g = build_graph(&net, &p);
        g.rn_fare = vec![20.0 * rate; g.rn_ride.len()];
        let ones = vec![1.0; g.rn_ride.len()];
        set_crowding(&mut g, &ones);
        if knots.len() >= 2 {
            g.fare_curve = FareCurve { km: knots.iter().map(|k| k.0).collect(), min: knots.iter().map(|k| k.1).collect(), base, rate, ..Default::default() };
            g.rn_km = vec![20.0; g.rn_ride.len()];
        }
        g
    };
    let line = skims(&graph(&[]), &p).r;
    let straight = skims(&graph(&[(0.0, base), (20.0, base + 20.0 * rate), (100.0, base + 100.0 * rate)]), &p).r;
    let knee = base + 30.0 * rate;
    let bent_g = graph(&[(0.0, base), (30.0, knee), (100.0, knee + 70.0 * steep)]);
    let bent_sk = skims(&bent_g, &p);
    for (a, b) in line.iter().zip(&straight) {
        assert!(a == b || (a - b).abs() < 1e-3, "{a} {b}");
    }
    let at = |r: &[f32], i: usize, j: usize| r[i * 4 + j];
    let more = steep - rate;
    assert!((at(&bent_sk.r, 0, 1) - at(&line, 0, 1)).abs() < 1e-3);
    assert!((at(&bent_sk.r, 0, 2) - at(&line, 0, 2) - more * 10.0).abs() < 1e-3);
    assert!((at(&bent_sk.r, 0, 3) - at(&line, 0, 3) - more * 30.0).abs() < 1e-3);
    assert!((at(&bent_sk.r, 3, 0) - at(&line, 3, 0) - more * 30.0).abs() < 1e-3);

    let mut od = vec![0.0f32; 16];
    od[2] = 10.0;
    od[3] = 5.0;
    od[4] = 7.0; // 1 -> 0, 20 km: no extra
    let l = assign(&bent_g, &bent_sk, &od);
    let total: f32 = l.rn_fare_x.iter().sum();
    let want = 10.0 * more * 10.0 + 5.0 * more * 30.0;
    assert!((total - want).abs() < 1e-3, "{total} against {want}");
    // 0 -> 2 rides either line from 0, the same track and stops every 10 and every 5 minutes: one
    // evenly spaced service, a third and two thirds by trains an hour (T-4050); 0 -> 3 waits for
    // line 0, since line 1's riders may not change at 2 onto the line right behind them (T-4064).
    // Each trip's extra is spread over the stretches it rides by their km, all 20 km (T-1516)
    let (x02, x03) = (10.0 * more * 10.0, 5.0 * more * 30.0);
    for (r, want) in [(8, x02 * 2.0 / 3.0 / 2.0), (9, x02 * 2.0 / 3.0 / 2.0), (0, x02 / 3.0 / 2.0 + x03 / 3.0), (1, x02 / 3.0 / 2.0 + x03 / 3.0), (2, x03 / 3.0)] {
        assert!((l.rn_fare_x[r] - want).abs() < 1e-3, "route node {r}: {:?}", l.rn_fare_x);
    }
    assert!(assign(&graph(&[]), &skims(&graph(&[]), &p), &od).rn_fare_x.iter().all(|&x| x == 0.0));
    // T-3758: under fare zones with every line on the distance fare, every ride is a distance ride and
    // the trips pay the curve on their km as above, split over the two lines alike
    let mut zg = graph(&[(0.0, base), (30.0, knee), (100.0, knee + 70.0 * steep)]);
    zg.fare_curve.zones = FareZones { st: vec![1 << 8 | 1; 4], off: vec![0, 1], min: vec![base], dist: vec![true; zg.rn_ride.len()], ..Default::default() };
    let zsk = skims(&zg, &p);
    for (a, b) in zsk.r.iter().zip(&bent_sk.r) {
        assert!((a - b).abs() < 1e-3, "{a} {b}");
    }
    let zl = assign(&zg, &zsk, &od);
    for (r, (a, b)) in zl.rn_fare_x.iter().zip(&l.rn_fare_x).enumerate() {
        assert!((a - b).abs() < 1e-3, "route node {r}: {a} {b}");
    }
}

/// T-3740, T-3758: fare zones on a straight curve. One line 0-1-2-3 of 20 km stretches; stations 0, 1,
/// 2 in area 1 at zones 2, 1, 2 and station 3 in area 2. A trip counts the zones from the lowest to the
/// highest it touches: 0 -> 2 rides through zone 1 and pays 2 zones, as 0 -> 1 does; 2 -> 3 crosses
/// areas and pays the line. The route cost carries the zone price in the line's place, the assignment
/// puts the difference on the km ridden, and the trip explorer's legs add up to the price over the
/// base. No zones is no change.
#[test]
fn fare_zones_count_every_zone_a_trip_touches_in_the_curves_place() {
    let p = common_test_params();
    let net = common_test_net(&[&[0, 1, 2, 3]], &[12.0], &[6.0]);
    let (base, rate) = (4.5f32, 0.05f32);
    let km = [20.0, 20.0, 20.0, 0.0, 20.0, 20.0, 20.0, 0.0];
    let graph = |zones: Option<FareZones>| {
        let mut g = build_graph(&net, &p);
        g.rn_fare = km.iter().map(|k| k * rate).collect();
        let ones = vec![1.0; g.rn_ride.len()];
        set_crowding(&mut g, &ones);
        if let Some(z) = zones {
            g.fare_curve = FareCurve { base, rate, zones: z, ..Default::default() };
            g.rn_km = km.to_vec();
        }
        g
    };
    let code = |area: u32, zone: u32| area << 8 | zone;
    let zones = FareZones { st: vec![code(1, 2), code(1, 1), code(1, 2), code(2, 1)], off: vec![0, 3, 4], min: vec![5.0, 7.0, 9.0, 6.0], ..Default::default() };
    let plain_g = graph(None);
    let plain = skims(&plain_g, &p);
    let g = graph(Some(zones.clone()));
    let sk = skims(&g, &p);
    let at = |r: &[f32], i: usize, j: usize| r[i * 4 + j];
    for (i, j, price, k) in [(0, 2, 7.0, 40.0), (2, 0, 7.0, 40.0), (0, 1, 7.0, 20.0), (1, 2, 7.0, 20.0), (1, 0, 7.0, 20.0)] {
        let d = at(&sk.r, i, j) - at(&plain.r, i, j);
        let want = price - base - rate * k;
        assert!((d - want).abs() < 1e-4, "{i} -> {j} adds {d}, want {want}");
    }
    for (i, j) in [(2, 3), (3, 2), (0, 3), (3, 0)] {
        assert_eq!(at(&sk.r, i, j).to_bits(), at(&plain.r, i, j).to_bits(), "{i} -> {j} pays the line");
    }
    // the assignment: 10 trips 0 -> 2 and 4 trips 2 -> 3, the latter adding nothing
    let mut od = vec![0.0f32; 16];
    od[2] = 10.0;
    od[2 * 4 + 3] = 4.0;
    let l = assign(&g, &sk, &od);
    let x = 10.0 * (7.0 - base - rate * 40.0);
    let total: f32 = l.rn_fare_x.iter().sum();
    assert!((total - x).abs() < 1e-3, "{total} against {x}: {:?}", l.rn_fare_x);
    assert!((l.rn_fare_x[0] - x / 2.0).abs() < 1e-3 && (l.rn_fare_x[1] - x / 2.0).abs() < 1e-3, "{:?}", l.rn_fare_x);
    // the trip explorer: the legs' fares add up to the zone price over the base
    let routes = crate::demand::trip::routes(&g, &sk, &p, 0, 2, 0.01);
    assert!(!routes.is_empty());
    for (_, legs) in &routes {
        let fare: f32 = legs.iter().map(|l| l.fare).sum();
        assert!((fare - (7.0 - base)).abs() < 1e-4, "explorer fare {fare}");
    }
    // prices for fewer zones than counted: the last price holds; between two counts the line between
    let short = FareZones { off: vec![0, 1, 2], min: vec![5.0, 6.0], ..zones.clone() };
    assert_eq!(short.price_of(1, 1.0, 2.0), Some(5.0));
    assert_eq!(zones.price_of(1, 1.0, 2.5), Some(8.0));
    assert_eq!(zones.price_of(3, 1.0, 1.0), None);
    // all zones empty is the plain line to the bit
    let none = skims(&graph(Some(FareZones::default())), &p);
    assert!(none.r.iter().zip(&plain.r).all(|(a, b)| a.to_bits() == b.to_bits()));
}

/// T-3758: lines that pay the distance fare. Three lines in a row, A 0-1, B 1-2, C 2-3, 20 km each, all
/// stations in area 1 at zones 1, 2, 3, 3 (prices 5, 7, 9). With every line on zones, 0 -> 3 is one
/// run of 3 zones. With B on distance it is two runs, zones 1-2 on A and zone 3 on C, the base once and
/// B's 20 km at the rate; a station C passes without stopping counts for C's run. The assignment's
/// extra adds up to the trips' and goes on the zone rides by km, and the trip explorer agrees.
#[test]
fn distance_lines_split_a_trip_into_its_zone_runs() {
    let p = common_test_params();
    let net = common_test_net(&[&[0, 1], &[1, 2], &[2, 3]], &[12.0, 12.0, 12.0], &[6.0, 6.0, 6.0]);
    let (base, rate) = (4.5f32, 0.05f32);
    // route nodes: A 0, 1 and back 2, 3; B 4, 5, back 6, 7; C 8, 9, back 10, 11
    let km: Vec<f32> = (0..12).map(|r| if r % 2 == 0 { 20.0 } else { 0.0 }).collect();
    let code = |zone: u32| 1 << 8 | zone;
    let graph = |dist: &[usize], pass: Option<(usize, u32)>| {
        let mut g = build_graph(&net, &p);
        g.rn_fare = km.iter().map(|k| k * rate).collect();
        let ones = vec![1.0; g.rn_ride.len()];
        set_crowding(&mut g, &ones);
        let mut z = FareZones { st: vec![code(1), code(2), code(3), code(3)], off: vec![0, 3], min: vec![5.0, 7.0, 9.0], ..Default::default() };
        if !dist.is_empty() {
            z.dist = (0..12).map(|r| dist.contains(&r)).collect();
        }
        if let Some((r, c)) = pass {
            z.pass_off = (0..=12).map(|k| u32::from(k > r)).collect();
            z.pass = vec![c];
        }
        g.fare_curve = FareCurve { base, rate, zones: z, ..Default::default() };
        g.rn_km = km.clone();
        g
    };
    let plain = {
        let mut g = build_graph(&net, &p);
        g.rn_fare = km.iter().map(|k| k * rate).collect();
        let ones = vec![1.0; g.rn_ride.len()];
        set_crowding(&mut g, &ones);
        skims(&g, &p).r
    };
    let at = |r: &[f32], i: usize, j: usize| r[i * 4 + j];
    let b_dist = [4, 5, 6, 7];
    for (dist, pass, want03, want12) in [
        (&[][..], None, 9.0 - base - rate * 60.0, 7.0 - base - rate * 20.0),
        (&b_dist[..], None, (7.0 - base - rate * 20.0) + (5.0 - base - rate * 20.0), 0.0),
        (&b_dist[..], Some((8, code(1))), (7.0 - base - rate * 20.0) + (9.0 - base - rate * 20.0), 0.0),
    ] {
        let g = graph(dist, pass);
        let sk = skims(&g, &p);
        for (i, j, want) in [(0, 3, want03), (3, 0, if pass.is_some() { (7.0 - base - rate * 20.0) + (5.0 - base - rate * 20.0) } else { want03 }), (1, 2, want12)] {
            let d = at(&sk.r, i, j) - at(&plain, i, j);
            assert!((d - want).abs() < 1e-4, "{dist:?} {pass:?}: {i} -> {j} adds {d}, want {want}");
        }
        let mut od = vec![0.0f32; 16];
        od[3] = 10.0;
        let l = assign(&g, &sk, &od);
        let total: f32 = l.rn_fare_x.iter().sum();
        assert!((total - 10.0 * want03).abs() < 1e-3, "{total} against {}: {:?}", 10.0 * want03, l.rn_fare_x);
        let zone_rides: Vec<usize> = [0, 4, 8].into_iter().filter(|r| !dist.contains(r)).collect();
        for r in [0, 4, 8] {
            let want = if zone_rides.contains(&r) { 10.0 * want03 / zone_rides.len() as f32 } else { 0.0 };
            assert!((l.rn_fare_x[r] - want).abs() < 1e-3, "{dist:?} route node {r}: {:?}", l.rn_fare_x);
        }
        let routes = crate::demand::trip::routes(&g, &sk, &p, 0, 3, 0.01);
        assert_eq!(routes.len(), 1);
        let fare: f32 = routes[0].1.iter().map(|l| l.fare).sum();
        assert!((fare - (want03 + rate * 60.0)).abs() < 1e-4, "explorer fare {fare}");
    }
}

/// T-3784: a change inside a fare zone run pays no transfer fare; every other change pays it. Lines A
/// 0-1, B 1-3 (12 trains an hour) and D 1-3 (6), 20 km a stretch, all stations in area 1: from 0 to 3
/// riders change at 1 onto B or D by their trains, 2/3 and 1/3. With every line on zones both
/// changes are free; with D on the distance fare the change onto it pays. The route cost carries the
/// fare on the paid share alone, the assignment takes it off B's boarding, where its counted
/// transfers charge it (so B earns none, D all), and the explorer marks the change onto B free. Then
/// a walk between two runs: A 0-1 in area 1, 100 m on foot to C 2-3 in area 2 pays the fare; with 2
/// and 3 in area 1 it is one run and free.
#[test]
fn a_change_inside_a_zone_run_pays_no_transfer_fare() {
    let xf = 3.0f32;
    let (base, rate) = (4.5f32, 0.05f32);
    let code = |area: u32, zone: u32| area << 8 | zone;
    let km: Vec<f32> = (0..12).map(|r| if r % 2 == 0 { 20.0 } else { 0.0 }).collect();
    let graph = |net: &Network, p: &Params, z: FareZones| {
        let mut g = build_graph(net, p);
        let km = &km[..g.rn_ride.len()];
        g.rn_fare = km.iter().map(|k| k * rate).collect();
        let ones = vec![1.0; g.rn_ride.len()];
        set_crowding(&mut g, &ones);
        g.fare_curve = FareCurve { base, rate, zones: z, ..Default::default() };
        g.rn_km = km.to_vec();
        g
    };
    let p0 = common_test_params();
    let px = Params { xfer_fare: xf, ..p0.clone() };
    // route nodes: A 0, 1 and back 2, 3; B 4-7; D 8-11
    // (D at a platform of its own at 3, so B and D are independent, T-4050)
    let net = platform_apart(common_test_net(&[&[0, 1], &[1, 3], &[1, 3]], &[12.0, 12.0, 6.0], &[6.0, 6.0, 6.0]), 2, 1);
    let zones = |d_dist: bool| FareZones {
        st: vec![code(1, 1), code(1, 1), code(1, 1), code(1, 2)],
        off: vec![0, 3],
        min: vec![5.0, 7.0, 9.0],
        dist: if d_dist { (0..12).map(|r| r >= 8).collect() } else { vec![] },
        ..Default::default()
    };
    let at = |r: &[f32], i: usize, j: usize| r[i * 4 + j];
    // B every 5 minutes is first three times in four beside D every 10, the same ride (T-4041)
    let pb = 0.75;
    for (d_dist, paid) in [(false, 0.0), (true, 1.0 - pb)] {
        let (g0, gx) = (graph(&net, &p0, zones(d_dist)), graph(&net, &px, zones(d_dist)));
        let (sk0, skx) = (skims(&g0, &p0), skims(&gx, &px));
        let d = at(&skx.r, 0, 3) - at(&sk0.r, 0, 3);
        assert!((d - xf * paid).abs() < 1e-4, "D on distance {d_dist}: 0 -> 3 adds {d}, want {}", xf * paid);
        let mut od = vec![0.0f32; 16];
        od[3] = 10.0;
        let ix = xfer_index(&gx);
        let (mut x0, mut xx) = (vec![0.0; ix.len], vec![0.0; ix.len]);
        let l0 = assign_with_transfers(&g0, &sk0, &od, &ix, &mut x0);
        let lx = assign_with_transfers(&gx, &skx, &od, &ix, &mut xx);
        let free_b = 10.0 * pb * xf;
        let free_d = if d_dist { 0.0 } else { 10.0 * (1.0 - pb) * xf };
        for r in 0..12 {
            let want = match r {
                4 => -free_b,
                8 => -free_d,
                _ => 0.0,
            };
            let got = lx.rn_fare_x[r] - l0.rn_fare_x[r];
            assert!((got - want).abs() < 1e-3, "D on distance {d_dist}: route node {r} gets {got}, want {want}");
        }
        // income: the refund and the counted transfers' fare cancel on B; D keeps its fare when paid
        let (u0, ux) = (crate::demand::api::fare_extra_usd(&l0.rn_fare_x, &p0, &g0, &ix, &x0), crate::demand::api::fare_extra_usd(&lx.rn_fare_x, &px, &gx, &ix, &xx));
        let paid_d = if d_dist { 10.0 * (1.0 - pb) * xf / px.fare_w } else { 0.0 };
        assert!((ux[4] - u0[4]).abs() < 1e-3 && (ux[8] - u0[8] - paid_d).abs() < 1e-3, "D on distance {d_dist}: B {} -> {}, D {} -> {}", u0[4], ux[4], u0[8], ux[8]);
        // the explorer: the change onto B is free, onto D free only on zones
        let routes = crate::demand::trip::routes(&gx, &skx, &px, 0, 3, 0.01);
        assert_eq!(routes.len(), 2);
        for (_, legs) in &routes {
            let rides: Vec<_> = legs.iter().filter(|l| l.kind == 1).collect();
            let want = rides[1].rn_from == 4 || !d_dist;
            assert_eq!(rides.iter().map(|l| l.free).collect::<Vec<_>>(), vec![false, want], "D on distance {d_dist}: rides from {:?}", rides.iter().map(|l| l.rn_from).collect::<Vec<_>>());
        }
    }

    // a walk between two runs pays; the same walk inside one run is free
    let p0 = Params { xfer_walk_m: 300.0, ..p0 };
    let px = Params { xfer_fare: xf, ..p0.clone() };
    let mut net = common_test_net(&[&[0, 1], &[2, 3]], &[12.0, 12.0], &[6.0, 6.0]);
    (net.st_x, net.st_y) = (vec![0.0, 2000.0, 2100.0, 4100.0], vec![0.0; 4]);
    let two = FareZones { st: vec![code(1, 1), code(1, 1), code(2, 1), code(2, 1)], off: vec![0, 1, 2], min: vec![5.0, 6.0], ..Default::default() };
    let one = FareZones { st: vec![code(1, 1); 4], off: vec![0, 1], min: vec![5.0], ..Default::default() };
    for (z, paid) in [(two, 1.0), (one, 0.0)] {
        let d = at(&skims(&graph(&net, &px, z.clone()), &px).r, 0, 3) - at(&skims(&graph(&net, &p0, z), &p0).r, 0, 3);
        assert!((d - xf * paid).abs() < 1e-4, "walk: 0 -> 3 adds {d}, want {}", xf * paid);
    }
}

/// T-3740: a population's fare zones reach the graph through `put_fare_curve` on a straight curve
/// too: its base and rate set, the fare km filled, every price in minutes and held to the base; the
/// merge gives a demand station the first zone among the stations merged into it.
#[test]
fn fare_zones_reach_the_graph_on_a_straight_curve() {
    let p = common_test_params();
    let net = common_test_net(&[&[0, 1, 2, 3]], &[12.0], &[6.0]);
    let mut g = build_graph(&net, &p);
    let hop: Vec<f32> = vec![20.0; g.rn_ride.len()];
    let z = crate::demand::api::demand_fare_zones(&[0, 1, 1, 2, 3], 4, &[0, 0, 1 << 8 | 3, 1 << 8 | 1, 1 << 8 | 2], &[0, 2], &[1.0, 3.0], &[0; 8], &[], &[]);
    assert!(z.dist.is_empty() && z.pass.is_empty());
    assert_eq!(z.st, vec![0, 1 << 8 | 3, 1 << 8 | 1, 1 << 8 | 2]);
    crate::demand::api::put_fare_curve(&mut g, &[], &z, &hop, 6.0, 0.6, 3.0);
    assert!(g.fare_curve.is_line() && !g.fare_curve.flat());
    assert_eq!((g.fare_curve.base, g.fare_curve.rate), (6.0, 0.6));
    assert_eq!(g.fare_curve.zones.min, vec![6.0, 9.0]);
    assert_eq!(g.rn_km.len(), g.rn_ride.len());
    let check = |st: &[u32], dist: &[u8], off: &[u32], pass: &[u32]| crate::demand::api::check_fare_zones(st, &[0, 2], &[1.0, 3.0], dist, off, pass, 4);
    assert!(check(&[3 << 8 | 1], &[], &[], &[]).is_err());
    assert!(check(&[1 << 8], &[], &[], &[]).is_err());
    assert!(crate::demand::api::check_fare_zones(&[], &[], &[], &[], &[], &[], 4).is_ok());
    // T-3758: a mode per route node, passed codes as stations'
    assert!(check(&[1 << 8 | 1], &[0, 1, 0, 0], &[0, 1, 1, 1, 1], &[1 << 8 | 2]).is_ok());
    assert!(check(&[1 << 8 | 1], &[0, 1, 0], &[], &[]).is_err());
    assert!(check(&[1 << 8 | 1], &[], &[0, 1, 1, 1], &[1 << 8 | 2]).is_err());
    assert!(check(&[1 << 8 | 1], &[], &[0, 1, 1, 1, 1], &[3 << 8 | 2]).is_err());
}

/// T-1516: a 5 km feeder 0-1 into a line 1-2-3 of 20 km stretches, under a fare curve that is flat
/// past 5 km. A trip 0 -> 3 (45 km) pays the curve's extra on each stretch by its km, so the feeder
/// takes 5/45 of it and no stretch's rate plus extra is below zero (it took all of it, and went
/// below, while the extra was put where trips first board).
#[test]
fn a_feeders_share_of_the_fare_curve_extra_goes_by_its_km() {
    let p = common_test_params();
    let net = common_test_net(&[&[0, 1], &[1, 2, 3]], &[12.0, 12.0], &[2.0, 6.0]);
    let (base, rate) = (4.5f32, 0.3f32);
    let mut g = build_graph(&net, &p);
    // route nodes: the feeder's 0, 1 and back 2, 3; the line's 4, 5, 6 and back 7, 8, 9
    let km = [5.0, 0.0, 5.0, 0.0, 20.0, 20.0, 0.0, 20.0, 20.0, 0.0];
    g.rn_fare = km.iter().map(|k| k * rate).collect();
    let ones = vec![1.0; g.rn_ride.len()];
    set_crowding(&mut g, &ones);
    g.fare_curve = FareCurve { km: vec![0.0, 5.0, 200.0], min: vec![base, base + 5.0 * rate, base + 5.0 * rate], base, rate, ..Default::default() };
    g.rn_km = km.to_vec();
    let sk = skims(&g, &p);
    let mut od = vec![0.0f32; 16];
    od[3] = 10.0; // 0 -> 3
    let l = assign(&g, &sk, &od);
    let extra = 10.0 * (base + 5.0 * rate - base - rate * 45.0);
    let total: f32 = l.rn_fare_x.iter().sum();
    assert!((total - extra).abs() < 1e-3, "{total} against {extra}: {:?}", l.rn_fare_x);
    for (r, want) in [(0, extra * 5.0 / 45.0), (4, extra * 20.0 / 45.0), (5, extra * 20.0 / 45.0)] {
        assert!((l.rn_fare_x[r] - want).abs() < 1e-3, "route node {r}: {:?}", l.rn_fare_x);
    }
    for r in 0..km.len() {
        assert!(l.seg[r] * km[r] * rate + l.rn_fare_x[r] >= -1e-3, "route node {r}: {:?}", l.rn_fare_x);
    }
}

/// T-3578: a line's own multiplier scales the base, a trip paying it times the largest of the lines
/// it rides: the feeder 0-1 at 1.5 and the line 1-2-3 at 2, so 0 -> 3 pays 2 bases, 0 -> 1 one and a
/// half and 1 -> 3 two. The route costs carry the base's part past one base (below it at 0), on a
/// straight curve or a bent one; the assignment puts it on the km ridden, a line at 0 included, and
/// it adds up to the trips times it. Parallel lines from one station average by their trains.
#[test]
fn a_lines_own_multiplier_scales_the_base_by_the_largest_on_the_trip() {
    let p = common_test_params();
    let net = common_test_net(&[&[0, 1], &[1, 2, 3]], &[12.0, 12.0], &[2.0, 6.0]);
    let (base, rate) = (4.5f32, 0.3f32);
    // route nodes: the feeder's 0, 1 and back 2, 3; the line's 4, 5, 6 and back 7, 8, 9
    let km = [5.0, 0.0, 5.0, 0.0, 20.0, 20.0, 0.0, 20.0, 20.0, 0.0];
    let graph = |mul: Option<[f32; 2]>, bent: bool| {
        let mut g = build_graph(&net, &p);
        g.rn_fare = km.iter().map(|k| k * rate).collect();
        let ones = vec![1.0; g.rn_ride.len()];
        set_crowding(&mut g, &ones);
        if bent {
            g.fare_curve = FareCurve { km: vec![0.0, 5.0, 200.0], min: vec![base, base + 5.0 * rate, base + 5.0 * rate], base, rate, ..Default::default() };
            g.rn_km = km.to_vec();
        }
        if let Some(m) = mul {
            g.rn_base_mul = g.rn_line.iter().map(|&l| m[l as usize]).collect();
            g.base_min = base;
            g.rn_km = km.to_vec();
        }
        g
    };
    let at = |r: &[f32], i: usize, j: usize| r[i * 4 + j];
    for bent in [false, true] {
        let plain_g = graph(None, bent);
        let plain = skims(&plain_g, &p);
        let ones = skims(&graph(Some([1.0, 1.0]), bent), &p);
        assert!(plain.r.iter().zip(&ones.r).all(|(a, b)| a.to_bits() == b.to_bits()), "all 1 is no change");
        let g = graph(Some([1.5, 2.0]), bent);
        let sk = skims(&g, &p);
        for (i, j, m) in [(0, 3, 2.0), (0, 1, 1.5), (1, 3, 2.0), (3, 0, 2.0), (1, 0, 1.5)] {
            let d = at(&sk.r, i, j) - at(&plain.r, i, j);
            assert!((d - base * (m - 1.0)).abs() < 1e-4, "bent {bent}: {i} -> {j} adds {d}, want {}", base * (m - 1.0));
        }
        // the largest, not the sum: the feeder at 3 and the line at 0.5 pays 3 bases
        let sk3 = skims(&graph(Some([3.0, 0.5]), bent), &p);
        assert!((at(&sk3.r, 0, 3) - at(&plain.r, 0, 3) - base * 2.0).abs() < 1e-4);
        // a free feeder alone costs less than nothing past its wait and ride
        let free = skims(&graph(Some([0.0, 1.0]), bent), &p);
        assert!((at(&free.r, 0, 1) - at(&plain.r, 0, 1) + base).abs() < 1e-4);
        assert!((at(&free.r, 0, 3) - at(&plain.r, 0, 3)).abs() < 1e-4, "the line at 1 sets the trip's");

        let mut od = vec![0.0f32; 16];
        od[3] = 10.0; // 0 -> 3
        od[1] = 4.0; // 0 -> 1
        let l0 = assign(&plain_g, &plain, &od);
        let l = assign(&g, &sk, &od);
        assert!(l0.rn_base_x.iter().all(|&v| v == 0.0) && l.rn_fare_x == l0.rn_fare_x, "bent {bent}: the curve's extra is its own");
        let x = l.rn_base_x.clone();
        let total: f32 = x.iter().sum();
        let want = 10.0 * base * 1.0 + 4.0 * base * 0.5;
        assert!((total - want).abs() < 1e-3, "bent {bent}: {total} against {want}: {x:?}");
        // 0 -> 3's by km: 5 of 45 on the feeder; 0 -> 1's all on it
        for (r, want) in [(0, 10.0 * base * 5.0 / 45.0 + 4.0 * base * 0.5), (4, 10.0 * base * 20.0 / 45.0), (5, 10.0 * base * 20.0 / 45.0)] {
            assert!((x[r] - want).abs() < 1e-3, "bent {bent}: route node {r}: {x:?}");
        }
        let lf = assign(&graph(Some([0.0, 1.0]), bent), &free, &od);
        let xf: f32 = lf.rn_base_x.iter().sum();
        assert!((xf + 4.0 * base).abs() < 1e-3, "bent {bent}: only 0 -> 1 rides the free line alone: {xf}");
    }
    // two lines on one track and stops from 0 to 2, 6 and 12 trains an hour: one evenly spaced
    // service, riders split a third and two thirds by trains an hour (T-4050), and so does the base's
    // multiplier
    let net = common_test_net(&[&[0, 1, 2], &[0, 1, 2]], &[6.0, 12.0], &[2.0, 2.0]);
    let mut g = build_graph(&net, &p);
    let plain = skims(&g, &p);
    g.rn_base_mul = g.rn_line.iter().map(|&l| if l == 0 { 3.0 } else { 1.5 }).collect();
    g.base_min = base;
    g.rn_km = vec![1.0; g.rn_line.len()];
    let sk = skims(&g, &p);
    let want = base * ((3.0 + 2.0 * 1.5) / 3.0 - 1.0);
    assert!((at(&sk.r, 0, 2) - at(&plain.r, 0, 2) - want).abs() < 1e-4, "{} against {want}", at(&sk.r, 0, 2) - at(&plain.r, 0, 2));
}

/// T-798: a circle 0-1-2-3-4-5-0, 12 tph, 2 minutes a stop and half a minute standing at 0.
/// Riders take the short way round through 0 without changing; a line that is not a circle
/// sends them the long way.
#[test]
fn a_circle_rides_the_short_way_through_its_first_stop() {
    let p = common_test_params();
    let ring = |wrap: bool| {
        let st: Vec<(f32, f32)> = (0..6).map(|i| { let a = i as f32 * std::f32::consts::PI / 3.0; (3000.0 * a.cos(), 3000.0 * a.sin()) }).collect();
        let stops = [0u32, 1, 2, 3, 4, 5, 0];
        let ride: Vec<f32> = (0..7).map(|k| if k < 6 { 2.0 } else if wrap { 0.5 } else { 0.0 }).collect();
        let back: Vec<f32> = (0..7).map(|k| if k > 0 { 2.0 } else if wrap { 0.5 } else { 0.0 }).collect();
        Network {
            st_x: st.iter().map(|s| s.0).collect(), st_y: st.iter().map(|s| s.1).collect(),
            line_name: vec![String::new()], line_off: vec![0, 7], line_stops: stops.to_vec(),
            hop_min: ride, hop_back: back, line_speed_kmh: vec![0.0],
            line_headway_min: vec![[5.0; network::PERIODS]], line_seats: vec![100.0], line_rated: vec![], line_crush: vec![200.0],
            st_group: vec![], stop_x: vec![], stop_y: vec![], line_wrap: vec![wrap], span: vec![], st_ups: vec![], st_passage: vec![], catches: vec![], st_bus: vec![], st_tram: vec![], back_stops: vec![], line_pier: vec![], stop_street: vec![], bus_walks: None,
        }
    };
    let g = build_graph(&ring(true), &p);
    let s = g.n_st;
    assert_eq!(g.n - s, 14, "two route nodes a stop a direction, as for any line");
    let sk = skims(&g, &p);
    let first = first_two;
    // 5 to 1: two stops and the stand at 0 (4.5 from boarding), or now and then the long way
    // round when its train comes first (8)
    let (short, long_share) = first(2.0 + 0.5 + 2.0, 3.5, 5.0);
    let alone = 2.5 + 2.0 + 0.5 + 2.0;
    assert!((sk.r[5 * s + 1] - short).abs() < 1e-4, "{} against {short}", sk.r[5 * s + 1]);
    assert!((sk.r[1 * s + 5] - short).abs() < 1e-4, "the other way: {}", sk.r[1 * s + 5]);
    let mut od = vec![0.0; s * s];
    od[5 * s + 1] = 100.0;
    od[1 * s + 5] = 40.0;
    let loads = assign(&g, &sk, &od);
    // route nodes: run 0 = 0..7 (stations 0,1,2,3,4,5,0), run 1 = 7..14 (0,5,4,3,2,1,0)
    let near = |a: f32, b: f32| (a - b).abs() < 1e-3;
    let k = 1.0 - long_share;
    assert!(near(loads.seg[6], 100.0 * k), "riders stay aboard through 0 on run 0: {}", loads.seg[6]);
    assert!(near(loads.seg[13], 40.0 * k), "and on run 1");
    assert!(near(loads.rn_board.iter().sum::<f32>(), 140.0), "one boarding each");
    assert!(near(loads.rn_board[5], 100.0 * k) && near(loads.rn_alight[1], 100.0 * k) && near(loads.rn_board[8], 100.0 * long_share));
    assert!(near(loads.rn_board[12], 40.0 * k) && near(loads.rn_alight[8], 40.0 * k));
    // nobody boards at a wrapped last stop: the first stop's route node has the same trains
    assert_eq!(loads.rn_board[6] + loads.rn_board[13], 0.0);
    // without the wrap trains turn back at 0: riders split between the long way (four stops, 8
    // from boarding) and a change at 0 onto the line out to 1 (2, 2 + 2.5, 2: T-1268)
    let plain = skims(&build_graph(&ring(false), &p), &p);
    let want = first(8.0, 0.5, 5.0).0;
    assert!((plain.r[5 * s + 1] - want).abs() < 1e-4, "{} against {want}", plain.r[5 * s + 1]);
    // T-2981: run one way, run 1 has no train due anywhere (`track::span`), so 1 to 5 rides round the
    // long way on run 0, four stops; 5 to 1 is the short way as before
    let mut one = ring(true);
    one.span = (0..network::PERIODS).flat_map(|_| (0..14).map(|r| if r < 7 { 1.0 } else { 0.0 })).collect();
    let g1 = build_graph(&one, &p);
    let sk1 = skims(&g1, &p);
    assert!((sk1.r[5 * s + 1] - alone).abs() < 1e-4, "{}", sk1.r[5 * s + 1]);
    let long = 2.5 + 4.0 * 2.0;
    assert!((sk1.r[1 * s + 5] - long).abs() < 1e-4, "{} against {long}", sk1.r[1 * s + 5]);
    let loads = assign(&g1, &sk1, &od);
    assert_eq!(loads.rn_board[7..].iter().sum::<f32>(), 0.0, "nobody boards run 1");
    assert_eq!((loads.rn_board[1], loads.rn_alight[5]), (40.0, 40.0));
}

/// T-3332: a line 0-1-2 whose westbound trains pass 1 on the other track (`span` -1 there): nobody
/// boards or alights there westbound and 2 to 0 rides through; 2 to 1 and 1 to 0 have no ride on this
/// line alone (riding it back the other way is no route), where eastbound 0 to 1 still does.
#[test]
fn a_passed_stop_boards_and_alights_nobody() {
    let p = common_test_params();
    let mut net = Network {
        st_x: vec![0.0, 3000.0, 6000.0], st_y: vec![0.0; 3],
        line_name: vec![String::new()], line_off: vec![0, 3], line_stops: vec![0, 1, 2],
        hop_min: vec![2.0, 2.0, 0.0], hop_back: vec![0.0, 1.5, 1.5], line_speed_kmh: vec![0.0],
        line_headway_min: vec![[5.0; network::PERIODS]], line_seats: vec![100.0], line_rated: vec![], line_crush: vec![200.0],
        st_group: vec![], stop_x: vec![], stop_y: vec![], line_wrap: vec![false], span: vec![], st_ups: vec![], st_passage: vec![], catches: vec![], st_bus: vec![], st_tram: vec![], back_stops: vec![], line_pier: vec![], stop_street: vec![], bus_walks: None,
    };
    // route nodes: run 0 = 0..3 (stations 0, 1, 2), run 1 = 3..6 (2, 1, 0)
    net.span = (0..network::PERIODS).flat_map(|_| (0..6).map(|r| if r == 4 { -1.0 } else { 1.0 })).collect();
    let g = build_graph(&net, &p);
    let s = g.n_st;
    let sk = skims(&g, &p);
    let mut od = vec![0.0; s * s];
    od[2 * s + 1] = 10.0;
    od[1 * s] = 20.0;
    od[2 * s] = 30.0;
    let loads = assign(&g, &sk, &od);
    assert_eq!((loads.rn_board[4], loads.rn_alight[4]), (0.0, 0.0), "nobody at the passed stop");
    assert!((sk.r[2 * s] - (2.5 + 3.0)).abs() < 1e-4, "2 to 0 rides through: {}", sk.r[2 * s]);
    assert!(!(sk.r[2 * s + 1] < 1e3) && !(sk.r[s] < 1e3), "no ride 2 to 1 or 1 to 0: {} {}", sk.r[2 * s + 1], sk.r[s]);
    assert!((sk.r[1] - (2.5 + 2.0)).abs() < 1e-4, "0 to 1 eastbound: {}", sk.r[1]);
    assert!((loads.seg[4] - 30.0).abs() < 1e-3, "the westbound riders stay aboard through 1: {:?}", loads.seg);
    assert_eq!(loads.rn_alight[5], 30.0);
}

/// T-4013: a line 0-1-2 with stop 1 kept to boarding only (`span` `SPAN_NO_ALIGHT` plus the share) or
/// getting off only (`SPAN_NO_BOARD`) on both runs, and station 3 200 m from 1 on no line, a walk away.
fn only_net(at_1: f32) -> Network {
    let mut net = Network {
        st_x: vec![0.0, 3000.0, 6000.0, 3200.0], st_y: vec![0.0; 4],
        line_name: vec![String::new()], line_off: vec![0, 3], line_stops: vec![0, 1, 2],
        hop_min: vec![2.0, 2.0, 0.0], hop_back: vec![0.0, 2.0, 2.0], line_speed_kmh: vec![0.0],
        line_headway_min: vec![[5.0; network::PERIODS]], line_seats: vec![100.0], line_rated: vec![], line_crush: vec![200.0],
        st_group: vec![], stop_x: vec![], stop_y: vec![], line_wrap: vec![false], span: vec![], st_ups: vec![], st_passage: vec![], catches: vec![], st_bus: vec![], st_tram: vec![], back_stops: vec![], line_pier: vec![], stop_street: vec![], bus_walks: None,
    };
    // route nodes: run 0 = 0..3 (stations 0, 1, 2), run 1 = 3..6 (2, 1, 0); stop 1 is 1 and 4
    net.span = (0..network::PERIODS).flat_map(|_| (0..6).map(move |r| if r == 1 || r == 4 { at_1 } else { 1.0 })).collect();
    net
}

/// T-4013: at a boarding-only stop nobody gets off, by train or by walking on: trips to it, or to a
/// station a walk from it, have no ride on the line, while trips from it ride both ways.
#[test]
fn a_boarding_only_stop_lets_nobody_off() {
    let p = common_test_params();
    let net = only_net(crate::track::span::SPAN_NO_ALIGHT + 1.0);
    let g = build_graph(&net, &p);
    let s = g.n_st;
    let sk = skims(&g, &p);
    let none = |a: usize, b: usize| !(sk.r[a * s + b] < 1e3);
    assert!(none(0, 1) && none(2, 1) && none(0, 3) && none(2, 3), "nothing gets off at 1: {:?}", sk.r);
    assert!((sk.r[s + 2] - (2.5 + 2.0)).abs() < 1e-4 && (sk.r[s] - (2.5 + 2.0)).abs() < 1e-4, "1 to 2 and 1 to 0 ride");
    let mut od = vec![0.0; s * s];
    (od[1], od[2 * s + 1], od[3], od[s + 2], od[s]) = (10.0, 15.0, 5.0, 20.0, 30.0);
    let loads = assign(&g, &sk, &od);
    assert_eq!((loads.rn_alight[1], loads.rn_alight[4]), (0.0, 0.0), "nobody alights at 1");
    assert_eq!((loads.rn_board[1], loads.rn_board[4]), (20.0, 30.0), "riders board at 1 both ways");
}

/// T-4013: at a getting-off-only stop nobody boards: trips from it have no ride on the line, while
/// trips to it ride both ways.
#[test]
fn a_getting_off_only_stop_boards_nobody() {
    let p = common_test_params();
    let net = only_net(crate::track::span::SPAN_NO_BOARD);
    let g = build_graph(&net, &p);
    let s = g.n_st;
    let sk = skims(&g, &p);
    assert!(!(sk.r[s + 2] < 1e3) && !(sk.r[s] < 1e3) && !(sk.r[3 * s + 2] < 1e3), "nothing boards at 1: {:?}", sk.r);
    assert!((sk.r[1] - (2.5 + 2.0)).abs() < 1e-4 && (sk.r[2 * s + 1] - (2.5 + 2.0)).abs() < 1e-4, "0 to 1 and 2 to 1 ride");
    let mut od = vec![0.0; s * s];
    (od[1], od[2 * s + 1], od[s + 2], od[s], od[2]) = (10.0, 15.0, 20.0, 30.0, 7.0);
    let loads = assign(&g, &sk, &od);
    assert_eq!((loads.rn_board[1], loads.rn_board[4]), (0.0, 0.0), "nobody boards at 1");
    assert_eq!((loads.rn_alight[1], loads.rn_alight[4]), (10.0, 15.0), "riders get off at 1 both ways");
    assert!((loads.seg[1] - 7.0).abs() < 1e-3, "0 to 2 rides through 1: {:?}", loads.seg);
}

#[test]
fn everyday_rates_by_name() {
    let mut p = Params::default();
    p.set_extra("rate_social", 0.42).unwrap();
    p.set_extra("decay_medical", 3500.0).unwrap();
    assert_eq!((p.everyday_rate[1], p.everyday_decay_m[5]), (0.42, 3500.0));
    assert_eq!(p.everyday_rate[0], super::super::everyday::RATE[0]);
    assert!(p.set_extra("rate_shopping", 0.1).is_err());
}

/// T-710: the station choice's scale is 20 over the trip's rough length (25 + 3 x km
/// perceived minutes), held from 60 minutes on, by four distance bands; off, one band at
/// `station_theta` with no offset.
#[test]
fn station_choice_scale_follows_the_trip_length() {
    let p = Params::default();
    let km = |d: f32| p.band_of_d2(d * d * 1e6);
    assert_eq!((km(0.0), km(2.9), km(3.0), km(6.0), km(11.9), km(12.0), km(80.0)), (0, 0, 1, 1, 2, 3, 3));
    let (th0, off0) = p.band(0);
    assert!((th0 - 20.0 / 29.5).abs() < 1e-6 && (off0 - 60.0 / th0).abs() < 1e-4);
    assert!((p.band(1).0 - 0.5).abs() < 1e-6);
    assert!((p.band(2).0 - 20.0 / 53.5).abs() < 1e-6);
    // past 60 minutes the scale softens as sqrt(60 / T): the band from 12 km is 79 minutes
    assert!((p.band(3).0 - 20.0 / 60.0 * (60.0f32 / 79.0).sqrt()).abs() < 1e-6);
    assert_eq!(p.soften(0.5, 50.0), 0.5);
    assert!((p.soften(1.0 / 3.0, 240.0) - 1.0 / 6.0).abs() < 1e-6);
    // `rel_soft` 0: past 60 minutes a minute counts the same however long the trip
    let hard = Params { rel_soft: 0.0, ..Params::default() };
    assert!((hard.band(3).0 - 20.0 / 60.0).abs() < 1e-6 && hard.soften(0.5, 240.0) == 0.5);
    let off = Params { rel_gamma: 0.0, ..Params::default() };
    assert_eq!((off.n_bands(), off.band_of_d2(1e12), off.band(0)), (1, 0, (0.2, 0.0)));
}

/// T-710: a train costs nothing extra until 70% of crush (or its seats, if more), then
/// 1 + 2 x(1+x)/2 with x from there to crush: 3 at crush.
#[test]
fn crowding_starts_near_full() {
    let p = common_test_params();
    let net = common_test_net(&[&[0, 1]], &[6.0], &[2.0]);
    let g = build_graph(&net, &p);
    // an hour's load at the busiest hour from the period's riders
    let at = |hour: f32, p: &Params| {
        let mut l = Loads::zero(g.n_st, g.rn_line.len());
        l.seg[0] = hour * p.period_hours / p.peak_hour_factor;
        crowd_factors(&g, &net, &l, p).0[0]
    };
    // 6 trains an hour: 600 seats, 1,200 at crush
    assert_eq!(at(720.0, &p), 1.0);
    assert_eq!(at(840.0, &p), 1.0);
    assert!((at(1200.0, &p) - 3.0).abs() < 1e-4);
    let seats = Params { crowd_start: 0.0, ..common_test_params() };
    assert!((at(720.0, &seats) - 1.24).abs() < 1e-4);
}

/// T-710: a band's offset keeps a long trip's rail inside f32 with a sharp scale: a pair whose
/// station-to-station cost is 300 perceived minutes is still priced, at its own cost.
#[test]
fn a_long_trip_keeps_its_rail_cost_in_a_sharp_band() {
    let p = Params::default();
    let (th, off) = p.band(3);
    // 300 minutes at this scale is e^-100 without the offset: clamped, as if 240
    assert_eq!(fast_exp(-th * 300.0), fast_exp(-80.0));
    let e = fast_exp(-th * (300.0 - off));
    assert!(e > 1e-30);
    assert!((-fast_ln(e) / th + off - 300.0).abs() < 0.05);
    // and a short one in the same band does not overflow
    assert!(fast_exp(-th * (0.0 - off)).is_finite());
}

/// Setting the worldwide rail constant by name keeps the country's part on top.
#[test]
fn country_rail_constant() {
    let mut p = Params::default();
    assert_eq!(super::super::api::country_asc_rail("JP"), 0.0);
    p.set_extra("country_asc_rail", 0.5).unwrap();
    assert_eq!(p.asc_rail, super::super::api::ASC_RAIL + 0.5);
    p.set_extra("asc_rail", 0.0).unwrap();
    assert_eq!(p.asc_rail, 0.5);
    p.set_extra("country_asc_rail", 0.0).unwrap();
    assert_eq!(p.asc_rail, 0.0);
}

/// T-380's parking: by jobs per km² of land, the paid price, the free share and the cordon;
/// with nothing to pay the two-logit mix is the one logit to the bit.
#[test]
fn parking_by_land_density() {
    let n = 5;
    let land = 4.0 / CELL_KM2;
    let ncell = vec![land.round() as u32, land.round() as u32, land.round() as u32, (land / 4.0).round() as u32, land.round() as u32];
    let km2: Vec<f64> = ncell.iter().map(|&c| c as f64 * CELL_KM2 as f64).collect();
    let dens = [2000.0, 10000.0, 200000.0, 10000.0, 8000.0];
    let z = Zones {
        zone_m: 2000.0, n, cell_zone: vec![], gx: vec![0; n], gy: vec![0; n], region: vec![0; n], w: 1, h: 1, x0: 0.0, y0: 0.0,
        pop: vec![0.0; n], jobs: (0..n).map(|j| dens[j] * km2[j]).collect(), ncell, size: vec![],
    };
    let mut p = Params::default();
    let near = |a: f32, b: f32| (a - b).abs() < 0.05;
    let (f0, paid0, free0) = p.park_parts(&z, 0);
    assert!(f0 == p.park_min + p.car_own_min() && paid0 == 0.0 && near(free0, 0.8));
    let (_, paid1, _) = p.park_parts(&z, 1);
    assert!(near(paid1, 15.0 * (10000.0f32 / 3000.0).ln()));
    assert_eq!(p.park_parts(&z, 2).1, 50.0);
    // a quarter of the square on land: the same jobs per km² of land, the same price
    assert!(near(p.park_parts(&z, 3).1, paid1));
    assert!(near(p.park_parts(&z, 4).2, 0.5));
    p.cordon = Some(std::sync::Arc::new(vec![0.0, 0.0, 13.5, 0.0, 0.0]));
    assert_eq!(p.park_parts(&z, 2).0, 13.5 + p.park_min + p.car_own_min());
    let ring = |name: &str| CORDONS.iter().find(|c| c.0 == name).unwrap().2;
    // Times Square and Wall St in, Downtown Brooklyn and the Upper East Side out
    assert!(in_ring(ring("Manhattan"), -73.985, 40.758) && in_ring(ring("Manhattan"), -74.009, 40.706));
    assert!(!in_ring(ring("Manhattan"), -73.990, 40.693) && !in_ring(ring("Manhattan"), -73.958, 40.775));
    // Bank and Oxford Circus in, Canary Wharf out
    assert!(in_ring(ring("London"), -0.089, 51.513) && in_ring(ring("London"), -0.142, 51.515));
    assert!(!in_ring(ring("London"), -0.024, 51.505));
    // no walk: drivers parking free and paying choose apart, mixed by the free share
    for rail in [20.0f32, 30.0, 50.0] {
        let util = p.beta * (30.0 - rail);
        assert_eq!(p.car_choice(util, -1e5, p.beta, 0.0, 0.3, rail, 30.0), p.ratio_share(rail, 30.0));
        let r = p.car_choice(util, -1e5, p.beta, 20.0, 0.3, rail, 30.0);
        let (r0, r1) = (p.ratio_share(rail, 30.0), p.ratio_share(rail, 50.0));
        assert!(r > r0 && r < r1 && (r - (0.3 * r0 + 0.7 * r1)).abs() < 1e-6);
    }
}

/// T-2677: parking by income takes the fares' power of income from game/income.ts, so the two
/// cannot drift apart; India's factor (GDP $2,702 a head) is (2,702 / 35,000)^0.5.
#[test]
fn money_power_is_the_fares() {
    let src = include_str!("../../../../app/src/game/income.ts");
    let fare_e: f32 = src.lines().find_map(|l| l.strip_prefix("export const FARE_E = ")).and_then(|v| v.trim_end_matches(';').trim().parse().ok()).expect("FARE_E in game/income.ts");
    assert_eq!(MONEY_E, fare_e);
    assert!((park_income_of(Some(2702.0)) - 0.278).abs() < 0.001);
    assert_eq!(park_income_of(Some(90_027.0)), 1.0);
    assert_eq!(park_income_of(None), 1.0);
}

/// T-621: ownership is dollars a trip in the destination's fixed part plus dollars a km of
/// road in the drive, FARE_MIN_PER_USD minutes a dollar; at 0 both change nothing to the bit.
#[test]
fn car_ownership_adds_dollars_per_trip_and_km() {
    let z = Zones {
        zone_m: 2000.0, n: 1, cell_zone: vec![], gx: vec![0], gy: vec![0], region: vec![0], w: 1, h: 1, x0: 0.0, y0: 0.0,
        pop: vec![0.0], jobs: vec![0.0], ncell: vec![38], size: vec![],
    };
    let mut p = Params::default();
    p.set_extra("car_own_usd_trip", 0.0).unwrap();
    p.set_extra("car_own_usd_km", 0.0).unwrap();
    let base = p.park_parts(&z, 0);
    assert_eq!(p.car_drive(17.3, 12.1).to_bits(), (17.3f32 * p.w_drive).to_bits());
    p.set_extra("car_own_usd_trip", 1.5).unwrap();
    p.set_extra("car_own_usd_km", 0.1).unwrap();
    let usd = crate::demand::api::FARE_MIN_PER_USD;
    assert!((p.car_own_min() - 1.5 * usd).abs() < 1e-5);
    let with = p.park_parts(&z, 0);
    assert!((with.0 - base.0 - 1.5 * usd).abs() < 1e-5 && with.1 == base.1 && with.2 == base.2);
    // a 15 km road: $1.50 + $1.50 = 9 perceived minutes on top of the drive's time
    assert!((p.car_drive(20.0, 15.0) - 20.0 * p.w_drive - 1.5 * usd).abs() < 1e-4);
    assert!((p.car_own_min() + 15.0 * p.car_own_km_min() - 9.0).abs() < 1e-4);
}

/// T-854: home parking is 0.4 of the paid-space rule on residents. T-918: rail against the car
/// is 1 / (1 + q^3) on the cost ratio, the same at every trip length, mirrored for q and 1 / q,
/// and gone from 3 to 4 times either way; walking keeps its three-way share.
#[test]
fn home_parking_and_the_ratio_curve() {
    let p = Params::default();
    assert_eq!(p.home_park_at(3000.0), 0.0);
    assert!((p.home_park_at(40_000.0) - 0.4 * 15.0 * (40.0f32 / 3.0).ln()).abs() < 1e-4);
    let near = |a: f32, b: f32| (a - b).abs() < 1e-5;
    // twice as dear: 1 in 9; three times: 1 in 28; equal: half; any trip length
    for car in [10.0f32, 30.0, 200.0] {
        assert!(near(p.ratio_share(car, car), 0.5));
        assert!(near(p.ratio_share(2.0 * car, car), 1.0 / 9.0));
        assert!(near(p.ratio_share(3.0 * car, car), 1.0 / 28.0));
        assert!(near(p.ratio_share(3.5 * car, car), 0.5 / (1.0 + 3.5f32.powi(3))));
        assert_eq!(p.ratio_share(4.0 * car, car), 0.0);
        assert_eq!(p.ratio_share(9.0 * car, car), 0.0);
        // the mirror: the car's share at q is rail's at 1 / q
        for q in [0.2f32, 0.3, 0.5, 0.9, 1.7, 3.2] {
            assert!(near(1.0 - p.ratio_share(q * car, car), p.ratio_share(car, q * car)), "{car} {q}");
        }
    }
    // walking keeps the logit's share: rail and the car split what is left
    let (rail, road) = (40.0f32, 30.0f32);
    let bt = p.mode_beta(road);
    let (util, uw) = (bt * (road - rail), bt * (road - 35.0) + p.asc_walk());
    let w = fast_exp(uw - util) / (1.0 + fast_exp(-util) + fast_exp(uw - util));
    assert!(near(p.car_choice(util, uw, bt, 0.0, 1.0, rail, road), (1.0 - w) * p.ratio_share(rail, road)));
}

/// T-636: the walking constant is `walk_bonus_min` perceived minutes whatever the scale, and a
/// walk costs `walk_trip_w` a minute at every length.
#[test]
fn walk_constant_is_in_minutes_and_the_walk_cost_is_flat() {
    let mut p = Params { walk_bonus_min: 10.0, ..Params::default() };
    assert!((p.asc_walk() - 10.0 * p.beta).abs() < 1e-6);
    p.set_extra("beta", 0.05).unwrap();
    assert!((p.asc_walk() - 0.5).abs() < 1e-6);
    p.set_extra("walk_bonus_min", 20.0).unwrap();
    assert!((p.asc_walk() - 1.0).abs() < 1e-6);
    // 1 km straight is 1.3 km at 6 km/h: 13 minutes, and 2 km twice that
    let (a, b) = (p.walk_trip_cost(1.0), p.walk_trip_cost(2.0));
    assert!((a - 13.0 * p.walk_trip_w).abs() < 1e-3 && (b - 2.0 * a).abs() < 1e-3);
    assert_eq!(p.walk_trip_cost(WALK_REACH_KM + 0.1), NO_WALK);
}

/// T-4041: the same ride every 10 and every 5 minutes, from platforms of their own (T-4050): riders
/// take whichever comes first, a quarter and three quarters, and wait on average 5 - 3.75 + 125/150
/// minutes for it.
#[test]
fn common_lines_pool_wait_and_split_by_first_arrival() {
    let p = common_test_params();
    let net = platform_apart(common_test_net(&[&[0, 1], &[0, 1]], &[6.0, 12.0], &[2.0, 2.0]), 1, 1);
    let g = build_graph(&net, &p);
    let sk = skims(&g, &p);
    assert!((sk.r[1] - (2.0 + 5.0 - 3.75 + 125.0 / 150.0)).abs() < 1e-4, "{}", sk.r[1]);
    let mut od = vec![0.0; 16]; od[1] = 90.0; od[4] = 30.0;
    let loads = assign(&g, &sk, &od);
    assert!((loads.seg[0] - 22.5).abs() < 1e-3);
    assert!((loads.seg[4] - 67.5).abs() < 1e-3);
    assert!((loads.seg[2] - 7.5).abs() < 1e-3);
    assert!((loads.seg[6] - 22.5).abs() < 1e-3);
    assert_eq!(loads.board.iter().sum::<f32>(), 120.0);
    assert_eq!(loads.alight.iter().sum::<f32>(), 120.0);
    assert!((loads.rn_board.iter().sum::<f32>() - 120.0).abs() < 1e-3);
    assert!((loads.rn_alight.iter().sum::<f32>() - 120.0).abs() < 1e-3);
    // Two identical 6-tph lines at platforms of their own wait a third of their headway, independent
    // of each other (on one platform, a quarter: T-4050); one 12-tph line half its own.
    let one = skims(&build_graph(&common_test_net(&[&[0, 1]], &[12.0], &[2.0]), &p), &p);
    let two = skims(&build_graph(&platform_apart(common_test_net(&[&[0, 1], &[0, 1]], &[6.0, 6.0], &[2.0, 2.0]), 1, 1), &p), &p);
    assert!((one.r[1] - 4.5).abs() < 1e-5 && (two.r[1] - (2.0 + 10.0 / 3.0)).abs() < 1e-4, "{} {}", one.r[1], two.r[1]);
}

/// T-4050: lines leaving one platform for one platform at the next stop are one evenly spaced
/// service where their riders get off alike: a trunk of two lines every 10 minutes waits 2.5, not
/// 3.3, and its riders split by trains an hour; riders for a branch wait for their own line, since
/// the other's riders may not change onto it at the split, right behind them (T-4064). A line skipping the stop (the 4 beside the 6), another platform, or a bus stays
/// independent.
#[test]
fn lines_on_one_track_and_stops_are_evenly_spaced() {
    let p = common_test_params();
    let net = common_test_net(&[&[0, 1, 2], &[0, 1, 3]], &[6.0, 6.0], &[2.0, 2.0]);
    let g = build_graph(&net, &p);
    assert_eq!(g.rn_even.iter().filter(|&&c| c != NONE).count(), 4, "the trunk hops both ways, nothing else: {:?}", g.rn_even);
    let sk = skims(&g, &p);
    assert!((sk.r[1] - 4.5).abs() < 1e-5, "trunk: half of a 5 minute combined headway, {}", sk.r[1]);
    assert!((sk.r[2] - (4.0 + 5.0)).abs() < 1e-4, "branch: its own line's ride and half headway, {}", sk.r[2]);
    let mut od = vec![0.0; 16];
    od[1] = 90.0;
    od[2] = 100.0;
    let loads = assign(&g, &sk, &od);
    // route node 0 is line 0 out of 0, 6 line 1 out of 0
    assert!((loads.rn_board[0] - 145.0).abs() < 1e-3 && (loads.rn_board[6] - 45.0).abs() < 1e-3, "{:?}", loads.rn_board);
    // a branch every 60 minutes beside one every 4: its riders wait for it at their own station, from
    // the last trunk station too (not riding the other back a stop to catch it there)
    let sk = skims(&build_graph(&common_test_net(&[&[0, 1, 2], &[0, 1, 3]], &[15.0, 1.0], &[2.0, 2.0]), &p), &p);
    assert!((sk.r[3] - (4.0 + 30.0)).abs() < 1e-4 && (sk.r[7] - (2.0 + 30.0)).abs() < 1e-4, "{} {}", sk.r[3], sk.r[7]);
    // unequal: every 10 and every 5, 18 an hour, riders a third and two thirds
    let g = build_graph(&common_test_net(&[&[0, 1], &[0, 1]], &[6.0, 12.0], &[2.0, 2.0]), &p);
    let sk = skims(&g, &p);
    assert!((sk.r[1] - (2.0 + 30.0 / 18.0)).abs() < 1e-5, "{}", sk.r[1]);
    let mut od = vec![0.0; 16];
    od[1] = 90.0;
    let loads = assign(&g, &sk, &od);
    assert!((loads.rn_board[0] - 30.0).abs() < 1e-3 && (loads.rn_board[4] - 60.0).abs() < 1e-3, "{:?}", loads.rn_board);
    // one line dearer by a fare: taken while dearer by less than the other's half headway (5),
    // else never (its trains come between the cheaper one's)
    for (extra, want) in [(3.0, 2.0 + 1.5 + 2.5), (6.0, 2.0 + 5.0)] {
        let mut g = build_graph(&common_test_net(&[&[0, 1], &[0, 1]], &[6.0, 6.0], &[2.0, 2.0]), &p);
        g.rn_fare[4] = extra;
        let ones = vec![1.0; g.rn_ride.len()];
        set_crowding(&mut g, &ones);
        let r = skims(&g, &p).r[1];
        assert!((r - want).abs() < 1e-4, "dearer by {extra}: {r} against {want}");
    }
    // the 4 and the 6: an express to 2 as fast as the local through 1, both every 10 minutes
    let six = skims(&build_graph(&common_test_net(&[&[0, 1, 2], &[0, 2]], &[6.0, 6.0], &[2.0, 4.0]), &p), &p);
    assert!((six.r[2] - (4.0 + 10.0 / 3.0)).abs() < 1e-4, "{}", six.r[2]);
    // the same stops at platforms of their own, and a bus on the train's stops
    let mut own = common_test_net(&[&[0, 1], &[0, 1]], &[6.0, 6.0], &[2.0, 2.0]);
    (own.stop_x, own.stop_y) = (vec![0.0, 2000.0, 8.0, 2008.0], vec![0.0; 4]);
    assert!(build_graph(&own, &p).rn_even.is_empty());
    let mut bus = common_test_net(&[&[0, 1], &[0, 1]], &[6.0, 6.0], &[2.0, 2.0]);
    bus.line_crush[1] = network::BUS_CARS * crate::demand::api::CRUSH_PER_CAR;
    assert!(build_graph(&bus, &p).rn_even.is_empty());
}

/// T-1281: a wait in a waiting room counts at 0.8 of its minutes, and changing there too; a park
/// and ride takes 10 minutes off a ride's fixed cost.
#[test]
fn a_waiting_room_shortens_perceived_waits_and_a_park_and_ride_rides() {
    let p = common_test_params();
    let mut net = common_test_net(&[&[0, 1, 2], &[1, 3]], &[6.0, 6.0], &[2.0, 2.0]);
    let plain = skims(&build_graph(&net, &p), &p);
    assert!((plain.r[1] - 7.0).abs() < 1e-5);
    net.st_ups = vec![upgrades::WAITING_ROOM, upgrades::WAITING_ROOM, 0, 0];
    let room = skims(&build_graph(&net, &p), &p);
    assert!((room.r[1] - 6.0).abs() < 1e-5, "{}", room.r[1]);
    assert!((plain.r[3] - room.r[3] - 2.0).abs() < 1e-4, "both 5-minute waits a minute shorter: {} {}", plain.r[3], room.r[3]);
    assert_eq!(room.r[2 * 4], plain.r[2 * 4], "a wait elsewhere is as it was");
    assert_eq!(ride_fixed(&p, &net, 0), p.p_ride);
    net.st_ups[2] = upgrades::PARK_RIDE;
    assert_eq!(ride_fixed(&p, &net, 2), p.p_ride - 10.0);
}

/// T-1430: a trip's first boarding at the planned wait (the least of half the headway, 5 + a sixth
/// and 30, no tolerance penalty; with several lines, each line's wait spread over twice that,
/// T-4041) gives slow lines less; a change keeps half the headway, and a train every 10 minutes
/// waits as before.
#[test]
fn an_inter_city_start_waits_the_planned_wait() {
    let p = common_test_params();
    let entry = |sk: &Skims| sk.entry_r().into_owned();
    // hourly: 30 plus the tolerance penalty at the platform, 15 planned
    let mut net = common_test_net(&[&[0, 1]], &[1.0], &[2.0]);
    net.span = vec![0.5; network::PERIODS * 4];
    let sk = skims(&build_graph(&net, &p), &p);
    assert!((sk.r[1] - (32.0 + 2f32.ln() / p.beta)).abs() < 1e-4, "{}", sk.r[1]);
    assert!((entry(&sk)[1] - 17.0).abs() < 1e-4, "{}", entry(&sk)[1]);
    // an hourly fast line and an hourly slow one, 18 minutes slower: whichever comes first at the
    // platform (each wait over 60 minutes), and planned (over 30) the slow one less often
    let sk = skims(&build_graph(&common_test_net(&[&[0, 1], &[0, 1]], &[1.0, 1.0], &[2.0, 20.0]), &p), &p);
    let (platform, planned) = (first_two(2.0, 18.0, 60.0).0, first_two(2.0, 18.0, 30.0).0);
    assert!((sk.r[1] - platform).abs() < 1e-4, "{} against {platform}", sk.r[1]);
    assert!((entry(&sk)[1] - planned).abs() < 1e-4, "{} against {planned}", entry(&sk)[1]);
    // T-4041: and they are loaded so: 100 commuters and 100 inter-city trips from 0 to 1, the slow
    // line (route node 4) taking (60 - 18)^2 / 2 60^2 of the first and (30 - 18)^2 / 2 30^2 of the second
    let g = build_graph(&common_test_net(&[&[0, 1], &[0, 1]], &[1.0, 1.0], &[2.0, 20.0]), &p);
    let sk = skims(&g, &p);
    let (plain_slow, planned_slow) = (first_two(2.0, 18.0, 60.0).1, first_two(2.0, 18.0, 30.0).1);
    let mut a = DestAssign::new(&g, 0, false);
    assert!(a.take(1, |x| if x == 0 { 100.0 } else { 0.0 }) && a.take_entry(1, |x| if x == 0 { 100.0 } else { 0.0 }));
    a.walk(&g, 1, sk.strategy(sk.slot[1] as usize), None);
    let want = 100.0 * (plain_slow + planned_slow);
    assert!((a.l.rn_board[4] - want).abs() < 1e-3 && (a.l.rn_board[0] + a.l.rn_board[4] - 200.0).abs() < 1e-3, "{} against {want}", a.l.rn_board[4]);
    assert!((a.l.seg[4] - want).abs() < 1e-3 && (a.l.board[0] - 200.0).abs() < 1e-3);
    assert!((planned_board_cost(240.0, 1.0, 1.0, &p) - p.transfer_min - 30.0).abs() < 1e-5);
    // every 10 minutes, then a change onto an hourly line: the change waits 30 as before
    let sk = skims(&build_graph(&common_test_net(&[&[0, 1], &[1, 2]], &[6.0, 1.0], &[2.0, 2.0]), &p), &p);
    let s = sk.s;
    assert!((sk.r[2] - (5.0 + 2.0 + p.transfer_min + 30.0 + 2.0)).abs() < 1e-4, "{}", sk.r[2]);
    assert_eq!(entry(&sk)[2], sk.r[2]);
    assert!(sk.entry.iter().flatten().all(|&(src, _)| src != 0), "only starts on the hourly line are cheaper");
    assert!((entry(&sk)[s + 2] - 17.0).abs() < 1e-4);
}

/// T-1281: a change through a passage walks the straight line at 2 m/s; between other stations
/// of a complex, the line x 1.3 at the transfer pace.
#[test]
fn a_passage_walks_the_straight_line_faster() {
    let p = common_test_params();
    let mut net = common_test_net(&[&[0, 2]], &[6.0], &[2.0]);
    net.st_group = vec![0, 0, 2, 3];
    let walk = |net: &Network| {
        let g = build_graph(net, &p);
        let rn = g.rn_station.iter().position(|&s| s == 0).unwrap() + g.n_st;
        (g.adj_off[rn] as usize..g.adj_off[rn + 1] as usize).find(|&e| g.adj_to[e] == 1).map(|e| g.adj_cost[e]).unwrap()
    };
    assert!((walk(&net) - p.xfer_min(2000.0) * p.w_xfer).abs() < 1e-4);
    net.st_passage = vec![[0, 1]];
    assert!((walk(&net) - 2000.0 / 2.0 / 60.0 * p.w_xfer).abs() < 1e-4, "{}", walk(&net));
}

/// T-4079: a bus stop's walks reach `BUS_WALK_M`, not `xfer_walk_m`; a walk with no passage at a bus stop or a
/// BRT station takes `CONNECTION_MIN` more; a passage there walks as any passage; between stations, as before.
#[test]
fn a_bus_stops_walks_are_shorter_and_slower_without_a_passage() {
    use crate::demand::network::{BUS_WALK_M, CONNECTION_MIN};
    let p = Params { xfer_walk_m: 300.0, ..common_test_params() };
    let walk = |d: f32, bus: bool, brt: bool, passage: bool| {
        let mut net = common_test_net(&[&[0, 2]], &[6.0], &[2.0]);
        net.st_x[1] = d;
        if bus { net.st_bus = vec![false, true, false, false] }
        if brt { net.st_ups = vec![upgrades::BRT_STATION, 0, 0, 0] }
        if passage { net.st_passage = vec![[0, 1]] }
        let g = build_graph(&net, &p);
        let rn = g.rn_station.iter().position(|&s| s == 0).unwrap() + g.n_st;
        (g.adj_off[rn] as usize..g.adj_off[rn + 1] as usize).find(|&e| g.adj_to[e] == 1).map(|e| g.adj_cost[e] / p.w_xfer)
    };
    let near = BUS_WALK_M - 20.0;
    assert_eq!(walk(BUS_WALK_M + 50.0, true, false, false), None, "a bus stop past its reach");
    assert!((walk(BUS_WALK_M + 50.0, false, false, false).unwrap() - p.xfer_min(BUS_WALK_M + 50.0)).abs() < 1e-4, "stations walk as before");
    for (bus, brt) in [(true, false), (false, true)] {
        assert!((walk(near, bus, brt, false).unwrap() - p.xfer_min(near) - CONNECTION_MIN).abs() < 1e-4, "{bus} {brt}");
        assert!((walk(near, bus, brt, true).unwrap() - near / 2.0 / 60.0).abs() < 1e-4, "{bus} {brt}: a passage");
    }
}

#[test]
fn common_lines_are_destination_specific_and_ignore_unhelpful_service() {
    let p = common_test_params();
    let net = common_test_net(&[&[0, 1, 2], &[0, 1, 3]], &[6.0, 6.0], &[2.0, 2.0]);
    let g = build_graph(&net, &p); let sk = skims(&g, &p);
    assert!((sk.r[1] - 4.5).abs() < 1e-4, "shared trunk, the same track and stops: one service every 5 minutes (T-4050)");
    // T-4064: to the branch, its own line only: the other's riders may not change onto it at 1,
    // where it runs right behind them
    assert!((sk.r[2] - (4.0 + 5.0)).abs() < 1e-4, "branch: {}", sk.r[2]);
    let mut od = vec![0.0; 16]; od[2] = 100.0;
    let loads = assign(&g, &sk, &od);
    assert!((loads.rn_board[0] - 100.0).abs() < 1e-3 && loads.rn_board[6] == 0.0, "{:?}", loads.rn_board);
    // A train that is off, or too slow, adds no attractive departures.
    for (frequency, slow) in [(0.0, 2.0), (6.0, 100.0)] {
        let g = build_graph(&common_test_net(&[&[0, 1], &[0, 1]], &[6.0, frequency], &[2.0, slow]), &p);
        let sk = skims(&g, &p);
        assert!((sk.r[1] - 7.0).abs() < 1e-5);
        assert_eq!(sk.board_tph(&g, 1)[4], 0.0);
    }
}

/// T-552, T-575: riders never change onto the line they left, so where full trains leave them
/// behind they wait for the direct train rather than ride the other way to board upstream.
/// Line 0-1-2-3, 12 tph, 2 minutes a stop, riders 1 -> 3; the southbound at 1 leaves riders
/// behind for 60 minutes.
#[test]
fn riders_wait_for_a_full_line_rather_than_ride_back() {
    let p = common_test_params();
    {
        let net = common_test_net(&[&[0, 1, 2, 3]], &[12.0], &[2.0]);
        let mut g = build_graph(&net, &p);
        let (south, north) = (1, 6);
        g.rn_left[south] = 60.0;
        let sk = skims(&g, &p);
        let s = g.n_st;
        let tph = sk.board_tph(&g, sk.slot[3] as usize);
        // the direct train with its wait: 2.5 a headway, 60 for the full trains, 4 riding
        let direct = 2.5 + 60.0 + 4.0;
        assert!((sk.r[1 * s + 3] - direct).abs() < 1e-3, "{} against {direct}", sk.r[1 * s + 3]);
        assert_eq!((tph[south], tph[north]), (12.0, 0.0));
        let mut od = vec![0.0; s * s];
        od[1 * s + 3] = 100.0;
        let loads = assign(&g, &sk, &od);
        assert_eq!(loads.rn_board[north], 0.0, "nobody rides back");
        assert_eq!(loads.rn_board[south], 100.0);
        assert_eq!(loads.board.iter().sum::<f32>(), 100.0, "one boarding each");
        // the trip explorer's route: one ride, its wait including the full trains'
        let r = super::super::trip::routes(&g, &sk, &p, 1, 3, 0.005);
        assert_eq!(r.len(), 1);
        let legs = &r[0].1;
        assert_eq!(legs.len(), 1);
        assert!((legs[0].wait_min - 62.5).abs() < 1e-4, "{}", legs[0].wait_min);
        // a shorter wait: still the direct train, its cost with the shorter wait
        g.rn_left[south] = 2.0;
        let sk = skims(&g, &p);
        let tph = sk.board_tph(&g, sk.slot[3] as usize);
        assert!((sk.r[1 * s + 3] - (2.5 + 2.0 + 4.0)).abs() < 1e-3, "{}", sk.r[1 * s + 3]);
        assert_eq!((tph[south], tph[north]), (12.0, 0.0));
    }
}

/// T-575: a rider leaving a line at a station whose strategy boards that line's other direction
/// changes there onto the station's other lines only. Line L 0-1-2 (12 tph, half a minute a
/// stop), C 0-3 (4 tph, 2 minutes), M 2-3 (4 tph, 3.5 minutes); riders 1 -> 3. At 2 the
/// strategy pools M with L back towards C; a rider off L at 2 takes M alone. Changes cost no
/// fixed minutes here, or the change onto C would keep L back out of the pool.
#[test]
fn a_change_off_a_line_takes_the_stations_other_lines() {
    let p = Params { transfer_min: 0.0, ..common_test_params() };
    let net = common_test_net(&[&[0, 1, 2], &[0, 3], &[2, 3]], &[12.0, 4.0, 4.0], &[0.5, 2.0, 3.5]);
    let g = build_graph(&net, &p);
    let s = g.n_st;
    let sk = skims(&g, &p);
    let slot = sk.slot[3] as usize;
    let tph = sk.board_tph(&g, slot);
    let (l_up, l_back_at_2, m_at_2) = (1, 3, 10);
    assert_eq!((tph[l_back_at_2], tph[m_at_2]), (12.0, 4.0), "2 pools M with L back");
    // from 1: L back to C (10 from boarding) and L on to M at 2 (0.5 + 11), both every 5 minutes,
    // whichever comes first (T-4041)
    let (want, on) = first_two(10.0, 1.5, 5.0);
    assert!((sk.r[1 * s + 3] - want).abs() < 1e-4, "{} against {want}", sk.r[1 * s + 3]);
    assert_eq!(tph[l_up], 12.0);
    let mut od = vec![0.0; s * s];
    od[1 * s + 3] = 100.0;
    let loads = assign(&g, &sk, &od);
    assert!((loads.rn_board[m_at_2] - 100.0 * on).abs() < 1e-3, "{}", loads.rn_board[m_at_2]);
    assert_eq!(loads.rn_board[l_back_at_2], 0.0, "nobody off L at 2 rides L back");
    assert!((loads.board[2] - 100.0 * on).abs() < 1e-3);
    assert!((loads.alight[3] - 100.0).abs() < 1e-3);
    let r = super::super::trip::routes(&g, &sk, &p, 1, 3, 0.005);
    assert_eq!(r.len(), 2);
    for (share, legs) in &r {
        assert!(((share - on).abs() < 1e-5 || (share - (1.0 - on)).abs() < 1e-5) && legs.len() == 2, "{share}");
    }
    let via_m = r.iter().find(|x| x.1[1].st_from == 2).unwrap();
    assert!((via_m.1[1].wait_min - 7.5).abs() < 1e-4, "M's own headway: {}", via_m.1[1].wait_min);
}

/// T-1268: a lollipop line, out along a trunk T0-T1-J, round a loop of eight stops L1..L8 and back
/// to J (and, as a balloon, back down the trunk to T0), 12 tph, 2 minutes a stop. A rider may
/// change onto the same line at J where it leaves for a stop that is neither the one they came
/// from nor the one their train goes on to: from L1 to L8 (or L8 to T0) they ride to J and take the
/// line on, rather than ride round the loop.
#[test]
fn a_lollipop_line_changes_onto_itself_at_its_junction() {
    let p = common_test_params();
    // stations: T0 0, T1 1, J 2, L1..L8 3..=10
    let lolly = |balloon: bool| {
        let mut stops: Vec<u32> = (0..=10).collect();
        stops.push(2);
        if balloon {
            stops.extend([1, 0]);
        }
        let n = stops.len();
        let mut st: Vec<(f32, f32)> = vec![(-4000.0, 0.0), (-2000.0, 0.0), (0.0, 0.0)];
        st.extend((0..8).map(|i| { let a = (i as f32 + 0.5) * std::f32::consts::PI / 4.0 - std::f32::consts::PI; (3000.0 + 3000.0 * a.cos(), 3000.0 * a.sin()) }));
        Network {
            st_x: st.iter().map(|s| s.0).collect(), st_y: st.iter().map(|s| s.1).collect(),
            line_name: vec![String::new()], line_off: vec![0, n as u32], line_stops: stops,
            hop_min: (0..n).map(|k| if k + 1 < n { 2.0 } else { 0.0 }).collect(), hop_back: vec![], line_speed_kmh: vec![0.0],
            line_headway_min: vec![[5.0; network::PERIODS]], line_seats: vec![100.0], line_rated: vec![], line_crush: vec![200.0],
            st_group: vec![], stop_x: vec![], stop_y: vec![], line_wrap: vec![false], span: vec![], st_ups: vec![], st_passage: vec![], catches: vec![], st_bus: vec![], st_tram: vec![], back_stops: vec![], line_pier: vec![], stop_street: vec![], bus_walks: None,
        }
    };
    let g = build_graph(&lolly(true), &p);
    let s = g.n_st;
    let sk = skims(&g, &p);
    // L1 to L8: 2.5 waiting, 2 to J, the change's 2 and 2.5, 2 to L8; round the loop it is 16.5
    let across = 2.5 + 2.0 + 2.0 + 2.5 + 2.0;
    assert!((sk.r[3 * s + 10] - across).abs() < 1e-4, "{} against {across}", sk.r[3 * s + 10]);
    // trips with a direct train do not change: T0 to L1, L8 to T0 through J, T1 to T0 on both
    // (T-4041: a few take the train round the other way to J and change there when it comes
    // first, and the cost is a little over the direct train's)
    assert!(sk.r[3] >= 2.5 + 6.0 && sk.r[3] < 2.5 + 6.0 + 0.1, "{}", sk.r[3]);
    assert!(sk.r[10 * s] >= 2.5 + 6.0 && sk.r[10 * s] < 2.5 + 6.0 + 0.1, "{}", sk.r[10 * s]);
    assert!((sk.r[s] - first_two(2.0, 0.0, 5.0).0).abs() < 1e-4, "{}", sk.r[s]);
    let mut od = vec![0.0; s * s];
    od[3 * s + 10] = 100.0;
    od[10 * s] = 30.0;
    od[3] = 20.0;
    let loads = assign(&g, &sk, &od);
    // route nodes: run 0 = 0..14 (T0 T1 J L1..L8 J T1 T0), run 1 = 14..28 (T0 T1 J L8..L1 J T1 T0)
    let (l1_back, j_on_to_l8) = (24, 16);
    assert_eq!((loads.rn_board[l1_back], loads.rn_board[j_on_to_l8]), (100.0, 100.0));
    let boards = loads.rn_board.iter().sum::<f32>();
    assert!((250.0..253.0).contains(&boards), "a change for L1 to L8, and for the few round the other way to J: {boards}");
    assert_eq!(loads.seg[3] + loads.seg[4], 0.0, "nobody rides round to L8");
    let r = super::super::trip::routes(&g, &sk, &p, 3, 10, 0.005);
    assert_eq!(r.len(), 1);
    let legs = &r[0].1;
    assert_eq!((legs.len(), legs[0].change, legs[1].change), (2, 0.0, 2.0));
    assert_eq!((legs[0].st_to, legs[1].st_from), (2, 2));

    // as a 6, ending at J: L8 to T0 rides to J and takes the line back down the trunk; round
    // the loop it is 22.5
    let g6 = build_graph(&lolly(false), &p);
    let sk6 = skims(&g6, &p);
    let down = 2.5 + 2.0 + 2.0 + 2.5 + 4.0;
    assert!((sk6.r[10 * s] - down).abs() < 1e-4, "{} against {down}", sk6.r[10 * s]);

    // no riding out to board upstream on the line (T-575): the trains in from T1 leave riders
    // behind for 20 minutes, and riding out to J for a train back in would cost 13
    let mut g = g;
    let (in_run0, in_run1) = (12, 14 + 12);
    g.rn_left[in_run0] = 20.0;
    g.rn_left[in_run1] = 20.0;
    let sk = skims(&g, &p);
    // (T-4041: a train out round the whole loop takes a few riders when it comes first, those who
    // take the first train a few more, so about the two trains in alone)
    let two_in = first_two(22.0, 0.0, 5.0).0;
    assert!(sk.r[s] > 22.0 && (sk.r[s] - two_in).abs() < 0.05, "{} against {two_in}", sk.r[s]);
}

/// T-3295: a line that bends back near itself, P(0)-A(1)-...-B(5)-T(6), A and B `gap` metres apart
/// and every other pair of stations far apart, 12 tph, 3 minutes a hop outside the bend and 6
/// round it.
fn bent_line(gap: f32) -> Network {
    let st: [(f32, f32); 7] = [(-3000.0, -1500.0), (0.0, 0.0), (3000.0, -1500.0), (5000.0, gap / 2.0), (3000.0, gap + 1500.0), (0.0, gap), (-3000.0, gap + 1500.0)];
    Network {
        st_x: st.iter().map(|s| s.0).collect(), st_y: st.iter().map(|s| s.1).collect(),
        line_name: vec![String::new()], line_off: vec![0, 7], line_stops: (0..7).collect(),
        hop_min: vec![3.0, 6.0, 6.0, 6.0, 6.0, 3.0, 0.0], hop_back: vec![], line_speed_kmh: vec![0.0],
        line_headway_min: vec![[5.0; network::PERIODS]], line_seats: vec![100.0], line_rated: vec![], line_crush: vec![200.0],
        st_group: vec![], stop_x: vec![], stop_y: vec![], line_wrap: vec![false], span: vec![], st_ups: vec![], st_passage: vec![], catches: vec![], st_bus: vec![], st_tram: vec![], back_stops: vec![], line_pier: vec![], stop_street: vec![], bus_walks: None,
    }
}

/// T-3295: off a line that bends back near itself, riders walk across the bend and take the same
/// line on from its other stop, either way along it, where that beats riding round; through a
/// passage too, past the walking limit. Route nodes: run 0 is 0..7 by stop, run 1 7..14 (stop
/// 6 - i at 7 + i).
#[test]
fn a_line_bent_back_near_itself_is_changed_onto_across_the_bend() {
    let p = Params { xfer_walk_m: 300.0, ..common_test_params() };
    let g = build_graph(&bent_line(200.0), &p);
    let s = g.n_st;
    let sk = skims(&g, &p);
    let walk = p.xfer_min(200.0) * p.w_xfer;
    // P to T: 2.5 waiting, 3 to A, the walk, 2.5 and the change's 2, 3 to T; round the bend 32.5
    let across = 2.5 + 3.0 + walk + 2.5 + p.transfer_min + 3.0;
    assert!(across < 32.5);
    assert!((sk.r[6] - across).abs() < 1e-3, "P to T {} against {across}", sk.r[6]);
    assert!((sk.r[6 * s] - across).abs() < 1e-3, "T to P {} against {across}", sk.r[6 * s]);
    // P to stop 4: across, then the line back the other way one stop of 6 (round, 23.5)
    let back = across - 3.0 + 6.0;
    assert!(back < 23.5);
    assert!((sk.r[4] - back).abs() < 1e-3, "P to 4 {} against {back}", sk.r[4]);
    // P to stop 3 rides round (2.5 + 3 + 12); stop 2 too
    assert!((sk.r[3] - 17.5).abs() < 1e-3, "{}", sk.r[3]);
    let mut od = vec![0.0; s * s];
    od[6] = 100.0;
    od[4] = 10.0;
    od[6 * s] = 50.0;
    let loads = assign(&g, &sk, &od);
    let (b_on, b_back, a_back) = (5, 8, 12);
    assert_eq!((loads.rn_board[b_on], loads.rn_board[b_back], loads.rn_board[a_back]), (100.0, 10.0, 50.0));
    assert_eq!(loads.seg[2] + loads.seg[3], 0.0, "nobody rides round the bend");
    let r = super::super::trip::routes(&g, &sk, &p, 0, 6, 0.005);
    assert_eq!(r.len(), 1);
    let legs = &r[0].1;
    let kinds: Vec<_> = legs.iter().map(|l| (l.kind, l.st_from, l.st_to)).collect();
    assert_eq!(kinds, vec![(1, 0, 1), (2, 1, 5), (1, 5, 6)], "ride to A, walk to B, ride on");

    // 600 m apart: too far to walk, so round the bend; linked by a passage, across it
    let mut net = bent_line(600.0);
    let sk = skims(&build_graph(&net, &p), &p);
    assert!((sk.r[6] - 32.5).abs() < 1e-3, "{}", sk.r[6]);
    net.st_group = vec![0, 1, 2, 3, 4, 1, 6];
    net.st_passage = vec![[1, 5]];
    let sk = skims(&build_graph(&net, &p), &p);
    let across = 2.5 + 3.0 + 600.0 / upgrades::PASSAGE_MPS / 60.0 * p.w_xfer + 2.5 + p.transfer_min + 3.0;
    assert!((sk.r[6] - across).abs() < 1e-3, "{} against {across}", sk.r[6]);
}

/// T-3295: a walk off a line to another of its stops never rides back over the line: X(0)-Y(1)
/// 200 m apart, then Z(2) and W(3) far on, 12 tph, 2 minutes a stop. Riders Z -> W whose trains
/// leave riders behind for 20 minutes wait for them, rather than ride back to Y, walk to X and
/// board there (20.0), nor does a rider off the line at Y walk to Z's side to ride back.
#[test]
fn a_walk_off_a_line_never_rides_it_back() {
    let p = Params { xfer_walk_m: 300.0, ..common_test_params() };
    let mut net = common_test_net(&[&[0, 1, 2, 3]], &[12.0], &[2.0]);
    (net.st_x, net.st_y) = (vec![0.0, 200.0, 2200.0, 4200.0], vec![0.0; 4]);
    let mut g = build_graph(&net, &p);
    let s = g.n_st;
    let z_on = 2;
    g.rn_left[z_on] = 20.0;
    let sk = skims(&g, &p);
    let direct = 2.5 + 20.0 + 2.0;
    assert!((sk.r[2 * s + 3] - direct).abs() < 1e-3, "{} against {direct}", sk.r[2 * s + 3]);
    let mut od = vec![0.0; s * s];
    od[2 * s + 3] = 100.0;
    let loads = assign(&g, &sk, &od);
    assert_eq!(loads.board.iter().sum::<f32>(), 100.0, "one boarding each");
    assert_eq!(loads.rn_board[z_on], 100.0);
}

/// T-3718: a trip may start with the walk it may end with, so the way home mirrors the way out. A
/// 0-1-2-3 west to east 1 km a stop, B 4-5-6 from 160 m south of A's 1 and 2, then south. Out
/// from 6 to 1 rides B to 4 and walks; home from 1 to 6 walks to 4 and rides B, never riding A a
/// stop to change at 2 and 5 (a paid change mode choice never saw, T-3689). Its riders enter at 1
/// and board at 4. With B ending at 4, home had no way at all.
#[test]
fn a_trip_starts_with_the_walk_it_may_end_with() {
    let p = Params { xfer_walk_m: 300.0, xfer_fare: 1000.0, ..common_test_params() };
    let mut net = common_test_net(&[&[0, 1, 2, 3], &[4, 5, 6]], &[12.0, 12.0], &[2.0, 2.0]);
    (net.st_x, net.st_y) = (vec![0.0, 1000.0, 2000.0, 3000.0, 1000.0, 2000.0, 2000.0], vec![0.0, 0.0, 0.0, 0.0, -160.0, -160.0, -2160.0]);
    let g = build_graph(&net, &p);
    let s = g.n_st;
    let sk = skims(&g, &p);
    let walk = p.xfer_min(160.0) * p.w_xfer;
    let (out, home) = (sk.r[6 * s + 1], sk.r[s + 6]);
    let want = 2.5 + 4.0 + walk;
    assert!((out - want).abs() < 1e-3 && (home - want).abs() < 1e-3, "out {out} home {home} against {want}");
    let mut od = vec![0.0; s * s];
    od[s + 6] = 100.0;
    od[6 * s + 1] = 100.0;
    let ix = xfer_index(&g);
    let mut xfer = vec![0.0; ix.len];
    let loads = assign_with_transfers(&g, &sk, &od, &ix, &mut xfer);
    assert_eq!(xfer.iter().sum::<f32>(), 0.0, "no changes");
    assert_eq!((loads.entry[1], loads.board[4], loads.board.iter().sum::<f32>()), (100.0, 100.0, 200.0));
    assert_eq!((loads.exit[1], loads.alight[4]), (100.0, 100.0));
    let legs = &super::super::trip::routes(&g, &sk, &p, 1, 6, 0.005)[0].1;
    let kinds: Vec<_> = legs.iter().map(|l| (l.kind, l.st_from, l.st_to)).collect();
    assert_eq!(kinds, vec![(2, 1, 4), (1, 4, 6)], "walk to B, ride");
    assert!((legs[0].perc + legs[1].perc + legs[1].wait_perc - home).abs() < 1e-3);
    // nobody takes the walk on a change, nor to the destination
    assert!(sk.r[3 * s + 6].is_infinite() || sk.r[3 * s + 6] > 1000.0, "A's 3 to B's 6 changes, at its fare: {}", sk.r[3 * s + 6]);
    assert!(sk.r[s + 4] > 1000.0, "1 to 4 is no walk-only rail trip: {}", sk.r[s + 4]);

    // B ending at 4 beside A's 1: home from 1 to 4's line had no way; now the walk
    let mut net = common_test_net(&[&[0, 1, 2], &[4, 3]], &[12.0, 12.0], &[2.0, 2.0]);
    (net.st_x, net.st_y) = (vec![0.0, 1000.0, 2000.0, 1000.0, 1000.0], vec![0.0, 0.0, 0.0, -2160.0, -160.0]);
    let g = build_graph(&net, &p);
    let s = g.n_st;
    let sk = skims(&g, &p);
    assert!((sk.r[s + 3] - (2.5 + 2.0 + walk)).abs() < 1e-3, "{}", sk.r[s + 3]);
    assert!((sk.r[3 * s + 1] - sk.r[s + 3]).abs() < 1e-3);
}

/// T-3718: no walk follows another. P, Q and R in a row 250 m apart (P and R out of reach), each
/// with a line of its own to D, faster from P to R: Q's riders walk to R, and P's walk to Q and
/// board there, whatever order the stations come in.
#[test]
fn a_start_walk_never_walks_on() {
    let p = Params { xfer_walk_m: 300.0, ..common_test_params() };
    for order in [[0u32, 1, 2], [2, 1, 0]] {
        let [pi, qi, ri] = order;
        let mut net = common_test_net(&[&[pi, 3], &[qi, 3], &[ri, 3]], &[12.0; 3], &[30.0, 20.0, 10.0]);
        let mut x = vec![0.0; 4];
        (x[pi as usize], x[qi as usize], x[ri as usize], x[3]) = (0.0, 250.0, 500.0, 250.0);
        (net.st_x, net.st_y) = (x, vec![0.0, 0.0, 0.0, 9000.0]);
        let g = build_graph(&net, &p);
        let s = g.n_st;
        let sk = skims(&g, &p);
        let walk = p.xfer_min(250.0) * p.w_xfer;
        let (pi, qi, ri) = (pi as usize, qi as usize, ri as usize);
        assert!((sk.r[qi * s + 3] - (walk + 12.5)).abs() < 1e-3, "Q walks to R: {}", sk.r[qi * s + 3]);
        assert!((sk.r[pi * s + 3] - (walk + 22.5)).abs() < 1e-3, "P walks to Q, no further: {}", sk.r[pi * s + 3]);
        let mut od = vec![0.0; s * s];
        od[pi * s + 3] = 100.0;
        let loads = assign(&g, &sk, &od);
        assert_eq!((loads.board[qi], loads.board[ri], loads.entry[pi]), (100.0, 0.0, 100.0), "{order:?}");
    }
}

#[test]
fn common_lines_use_expected_ride_cost_and_respect_fare_and_crowding() {
    let p = common_test_params();
    let mut g = build_graph(&common_test_net(&[&[0, 1], &[0, 1]], &[6.0, 12.0], &[2.0, 3.0]), &p);
    let sk = skims(&g, &p);
    // T-4041: 2 every 10 and 3 every 5: the first arrival 2 + 0.95 (one line can be first, to
    // 3) + 1.8333 (either, to 8); the first train that can arrive by 8, the second line's three
    // quarters of the time, 0.25 x 2 + 0.75 x 3 plus a wait of 2.0833
    let want = FIRST_ARRIVAL_SHARE * (2.0 + 0.95 + 5.5 / 3.0) + (1.0 - FIRST_ARRIVAL_SHARE) * (2.75 + 25.0 / 12.0);
    assert!((sk.r[1] - want).abs() < 1e-4, "{} against {want}", sk.r[1]);
    g.rn_fare[4] = 100.0;
    let uncrowded = vec![1.0; g.n - g.n_st];
    set_crowding(&mut g, &uncrowded);
    let expensive = skims(&g, &p);
    assert!((expensive.r[1] - 7.0).abs() < 1e-5);
    g.rn_fare[4] = 0.0;
    let mut factor = vec![1.0; g.n - g.n_st]; factor[4] = 100.0;
    set_crowding(&mut g, &factor);
    let crowded = skims(&g, &p);
    assert!((crowded.r[1] - 7.0).abs() < 1e-5);
}

/// T-119: strategies searched towards some stations only are the full ones for those stations,
/// the cache extends what it has instead of starting again, and flows into searched stations
/// load the same as with every station searched.
#[test]
fn partial_skims_equal_full_skims_where_searched() {
    let p = common_test_params();
    let net = common_test_net(&[&[0, 1, 2], &[0, 1, 3]], &[6.0, 6.0], &[2.0, 2.0]);
    let g = build_graph(&net, &p);
    let full = skims(&g, &p);
    let s = g.n_st;
    let part = skims_for(&g, &p, &[false, true, false, true]);
    for dst in 0..s {
        for src in 0..s {
            let (a, b) = (part.r[src * s + dst], full.r[src * s + dst]);
            if dst == 1 || dst == 3 {
                assert_eq!(a, b, "r {src} -> {dst}");
            } else {
                assert!(a.is_infinite(), "r {src} -> {dst} was not searched");
            }
        }
    }
    assert_eq!(part.order.len(), 2);
    let mut od = vec![0.0; s * s];
    od[1] = 40.0; // station 0 to 1
    od[3] = 25.0; // station 0 to 3
    od[s * 2 + 3] = 5.0;
    let (lf, lp) = (assign(&g, &full, &od), assign(&g, &part, &od));
    assert_eq!(lf.seg, lp.seg);
    assert_eq!(lf.board, lp.board);
    assert_eq!(lf.rn_alight, lp.rn_alight);
    // the cache: ask for 1 and 3, then for 2 as well; every answer is the full table's column
    clear_skim_cache();
    with_skims(&g, &p, &[false, true, false, true], |sk| assert_eq!(sk.slot.iter().filter(|&&v| v != NONE).count(), 2));
    with_skims(&g, &p, &[false, false, true, false], |sk| {
        assert_eq!(sk.slot.iter().filter(|&&v| v != NONE).count(), 3);
        for dst in [1, 2, 3] {
            for src in 0..s {
                assert_eq!(sk.r[src * s + dst], full.r[src * s + dst]);
            }
        }
        assert_eq!(assign(&g, sk, &od).seg, lf.seg);
    });
    // a different graph (another crowding) is a different entry
    let mut g2 = build_graph(&net, &p);
    let mut factor = vec![1.0; g2.n - g2.n_st];
    factor[0] = 3.0;
    set_crowding(&mut g2, &factor);
    let full2 = skims(&g2, &p);
    with_skims(&g2, &p, &[true; 4], |sk| assert_eq!(sk.r, full2.r));
    with_skims(&g, &p, &[true; 4], |sk| assert_eq!(sk.r, full.r));
    // T-405: a third graph wanting fewer searches than either kept table evicts neither
    let mut g3 = build_graph(&net, &p);
    factor[0] = 2.0;
    set_crowding(&mut g3, &factor);
    let full3 = skims(&g3, &p);
    with_skims(&g3, &p, &[false; 4], |sk| assert_eq!(sk.slot.iter().filter(|&&v| v != NONE).count(), 0));
    with_skims(&g3, &p, &[false, true, false, false], |sk| {
        assert_eq!(sk.slot.iter().filter(|&&v| v != NONE).count(), 1);
        assert_eq!((0..s).map(|src| sk.r[src * s + 1]).collect::<Vec<_>>(), (0..s).map(|src| full3.r[src * s + 1]).collect::<Vec<_>>());
    });
    with_skims(&g2, &p, &[false; 4], |sk| assert_eq!(sk.r, full2.r));
    with_skims(&g, &p, &[false; 4], |sk| assert_eq!(sk.r, full.r));
    clear_skim_cache();
}

/// T-176: transfers are counted platform to platform. A to C changes at B from line 0 to line 1
/// (and back on the way home); riders who end at B do not transfer.
#[test]
fn transfers_between_route_nodes() {
    let p = common_test_params();
    // line 0: 0-1-2, line 1: 1-3 (stations 1 and 2 at 2 and 4 km on x, 3 is 2 km north of 2)
    let net = common_test_net(&[&[0, 1, 2], &[1, 3]], &[6.0, 6.0], &[2.0, 2.0]);
    let g = build_graph(&net, &p);
    let sk = skims(&g, &p);
    let s = g.n_st;
    let mut od = vec![0.0; s * s];
    od[3] = 100.0; // 0 to 3: line 0 to B, line 1 on
    od[s * 3] = 40.0; // 3 to 0: line 1 to B, line 0 back
    od[1] = 7.0; // 0 to 1: no transfer
    let ix = xfer_index(&g);
    let mut x = vec![0.0; ix.len];
    let l = assign_with_transfers(&g, &sk, &od, &ix, &mut x);
    let plain = assign(&g, &sk, &od);
    assert_eq!((l.seg, l.board, l.rn_board, l.rn_alight), (plain.seg.clone(), plain.board.clone(), plain.rn_board.clone(), plain.rn_alight.clone()));
    let mut out = vec![];
    transfer_pairs(&g, &ix, &x, &vec![1u8; g.n - s], &mut out);
    // route nodes: line 0 run 0 = 0,1,2 (stations 0,1,2), run 1 = 3,4,5 (2,1,0); line 1 run 0 =
    // 6,7 (1,3), run 1 = 8,9 (3,1)
    let mut pairs: Vec<(u32, u32, f32)> = out.chunks(3).map(|c| (c[0] as u32, c[1] as u32, c[2])).collect();
    pairs.sort_by_key(|p| (p.0, p.1));
    assert_eq!(pairs, vec![(1, 6, 100.0), (9, 4, 40.0)]);
    // only pairs touching the selection
    let mut sel = vec![0u8; g.n - s];
    sel[4] = 1;
    let mut out = vec![];
    transfer_pairs(&g, &ix, &x, &sel, &mut out);
    assert_eq!(out, vec![9.0, 4.0, 40.0]);
}

/// T-717: a change costs the fixed minutes, its wait and the walk between its two platforms.
#[test]
fn a_change_costs_fixed_minutes_and_the_walk_between_platforms() {
    let p = common_test_params();
    assert_eq!(p.transfer_min, 2.0);
    // line A 0-1, B 1-2, 12 tph and 2 minutes each; riders 0 -> 2 change at 1
    let mut net = common_test_net(&[&[0, 1], &[1, 2]], &[12.0, 12.0], &[2.0, 2.0]);
    let s = net.n_stations();
    let one_platform = skims(&build_graph(&net, &p), &p).r[2];
    assert!((one_platform - (2.5 + 2.0 + 2.0 + 2.5 + 2.0)).abs() < 1e-4, "{one_platform}");
    // station 1 is two platforms 80 m apart, A's at y 0 and B's at y 80, the station between
    net.st_y[1] = 40.0;
    net.stop_x = net.line_stops.iter().map(|&st| net.st_x[st as usize]).collect();
    net.stop_y = vec![0.0, 0.0, 80.0, 0.0];
    let g = build_graph(&net, &p);
    assert!(g.st_spread[1] && !g.st_spread[0]);
    let sk = skims(&g, &p);
    let walk = p.xfer_min(80.0) * p.w_xfer;
    assert!((sk.r[2] - (one_platform + walk)).abs() < 1e-4, "{} against {}", sk.r[2], one_platform + walk);
    assert!((sk.r[s + 2] - 4.5).abs() < 1e-4, "a trip starting at the station walks no platform: {}", sk.r[s + 2]);
    // the trip explorer: ride, the walk between the platforms, ride with the change's 2 minutes
    let r = super::super::trip::routes(&g, &sk, &p, 0, 2, 0.005);
    let legs = &r[0].1;
    assert_eq!(legs.iter().map(|l| l.kind).collect::<Vec<_>>(), vec![1, 2, 1]);
    assert!((legs[1].perc - walk).abs() < 1e-5 && legs[1].st_from == 1 && legs[1].st_to == 1);
    assert_eq!((legs[0].change, legs[2].change), (0.0, 2.0));
    let total: f32 = legs.iter().map(|l| l.perc + l.wait_perc + l.change + l.ends).sum();
    assert!((total - sk.r[2]).abs() < 1e-4, "the legs add up to the strategy: {total}");
}

/// T-2220: a change at a station in a station building counts 0.8 of its fixed minutes (1.6); its
/// platform walk, the wait, the first boarding, the transfer fare and stations outside are as
/// before.
#[test]
fn a_change_in_a_station_building_counts_less() {
    let p = common_test_params();
    let mut net = common_test_net(&[&[0, 1], &[1, 2]], &[12.0, 12.0], &[2.0, 2.0]);
    let s = net.n_stations();
    net.st_y[1] = 40.0;
    net.stop_x = net.line_stops.iter().map(|&st| net.st_x[st as usize]).collect();
    net.stop_y = vec![0.0, 0.0, 80.0, 0.0];
    let plain = skims(&build_graph(&net, &p), &p);
    let walk = p.xfer_min(80.0) * p.w_xfer;
    // waits 2.5, rides 2, the change's fixed 2 and its walk
    assert!((plain.r[2] - (2.5 + 2.0 + 2.0 + walk + 2.5 + 2.0)).abs() < 1e-4, "{}", plain.r[2]);
    // a building round the first station only: the first boarding pays no change, and the
    // building is that station's waiting room (T-2276)
    net.st_ups = vec![upgrades::IN_BUILDING, 0, 0, 0];
    let first = skims(&build_graph(&net, &p), &p);
    net.st_ups = vec![upgrades::WAITING_ROOM, 0, 0, 0];
    let room = skims(&build_graph(&net, &p), &p);
    assert!(first.r.iter().zip(&room.r).all(|(a, b)| a == b || (a - b).abs() < 1e-4), "a building where nobody changes is a waiting room");
    net.st_ups = vec![0, upgrades::IN_BUILDING, 0, 0];
    let g = build_graph(&net, &p);
    assert_eq!((g.change_f(1), g.change_f(0)), (0.5, 1.0));
    let sk = skims(&g, &p);
    // the change's fixed 2 count 1, the wait there 0.8 of 2.5
    assert!((sk.r[2] - (2.5 + 2.0 + 1.0 + walk + 2.0 + 2.0)).abs() < 1e-4, "{}", sk.r[2]);
    assert!((sk.r[s + 2] - (plain.r[s + 2] - 0.5)).abs() < 1e-4, "a trip starting in the building pays no change, and waits in it");
    // the trip explorer: 1 on the change, the walk whole
    let legs = &super::super::trip::routes(&g, &sk, &p, 0, 2, 0.005)[0].1;
    assert_eq!(legs.iter().map(|l| l.kind).collect::<Vec<_>>(), vec![1, 2, 1]);
    assert!((legs[2].change - 1.0).abs() < 1e-6 && legs[0].change == 0.0);
    assert!((legs[1].perc - walk).abs() < 1e-5);
    let total: f32 = legs.iter().map(|l| l.perc + l.wait_perc + l.change + l.ends).sum();
    assert!((total - sk.r[2]).abs() < 1e-4, "the legs add up to the strategy: {total}");
    // a transfer fare is paid whole
    let pf = Params { xfer_fare: 3.0, ..p };
    let fare = skims(&build_graph(&net, &pf), &pf);
    assert!((fare.r[2] - sk.r[2] - 3.0).abs() < 1e-4, "{} {}", fare.r[2], sk.r[2]);
}

/// T-2220: a walk between two stations in one building is whole; only the boarding's fixed minutes
/// at the far one count less (half, T-2276), and its wait as in a waiting room.
#[test]
fn a_walk_between_building_stations_is_whole() {
    let p = Params { xfer_walk_m: 300.0, ..common_test_params() };
    let mut net = common_test_net(&[&[0, 1], &[2, 3]], &[6.0, 6.0], &[2.0, 2.0]);
    net.st_group = vec![0, 1, 1, 3];
    let walk_edge = |g: &Graph, r: usize, to: u32| (g.adj_off[g.n_st + r] as usize..g.adj_off[g.n_st + r + 1] as usize).find(|&e| g.adj_to[e] == to).map(|e| g.adj_cost[e]).unwrap();
    let plain = skims(&build_graph(&net, &p), &p);
    net.st_ups = vec![0, upgrades::IN_BUILDING, upgrades::IN_BUILDING, 0];
    let g = build_graph(&net, &p);
    // route nodes: line 0 run 0 = 0, 1 (stations 0, 1); line 1 run 0 = 4, 5 (2, 3)
    assert_eq!(walk_edge(&g, 1, 2), p.xfer_min(2000.0) * p.w_xfer);
    let sk = skims(&g, &p);
    // 1 of the fixed 2, and 0.2 of the 5 minute wait at station 2
    assert!((plain.r[3] - sk.r[3] - 2.0).abs() < 1e-4, "{} {}", plain.r[3], sk.r[3]);
}

/// T-161: stations of one complex are walking transfers whatever the distance, costed like any
/// walk; without the complex 2 km is out of reach and the trip has no rail.
#[test]
fn complex_members_are_walking_transfers() {
    let p = Params { xfer_walk_m: 300.0, ..common_test_params() };
    // line 0: 0-1, line 1: 2-3; stations 1 and 2 are 2 km apart
    let mut net = common_test_net(&[&[0, 1], &[2, 3]], &[6.0, 6.0], &[2.0, 2.0]);
    let walk_edge = |g: &Graph, r: usize, to: u32| (g.adj_off[g.n_st + r] as usize..g.adj_off[g.n_st + r + 1] as usize).find(|&e| g.adj_to[e] == to).map(|e| g.adj_cost[e]);
    let g = build_graph(&net, &p);
    assert_eq!(walk_edge(&g, 1, 2), None);
    assert!(skims(&g, &p).r[3].is_infinite());
    net.st_group = vec![0, 1, 1, 3];
    let g = build_graph(&net, &p);
    // route nodes: line 0 run 0 = 0, 1 (stations 0, 1); line 1 run 0 = 4, 5 (2, 3)
    assert_eq!(walk_edge(&g, 1, 2), Some(p.xfer_min(2000.0) * p.w_xfer));
    assert_eq!(walk_edge(&g, 4, 1), Some(p.xfer_min(2000.0) * p.w_xfer));
    assert_eq!(walk_edge(&g, 1, 0), None, "a station outside the complex stays out of reach");
    let sk = skims(&g, &p);
    assert!(sk.r[3].is_finite());
    let s = g.n_st;
    let mut od = vec![0.0; s * s];
    od[3] = 100.0;
    let ix = xfer_index(&g);
    let mut x = vec![0.0; ix.len];
    assign_with_transfers(&g, &sk, &od, &ix, &mut x);
    let mut out = vec![];
    transfer_pairs(&g, &ix, &x, &vec![1u8; g.n - s], &mut out);
    assert_eq!(out, vec![1.0, 4.0, 100.0]);
    // a group within the walk limit adds nothing: no edge twice
    net.st_x = vec![0.0, 2000.0, 2100.0, 4000.0];
    let a = build_graph(&net, &p).adj_to.len();
    net.st_group = vec![];
    assert_eq!(build_graph(&net, &p).adj_to.len(), a);
    assert_eq!(complex_members(&[5, 2, 5, 7, 2, 2]), vec![vec![0, 2], vec![1, 4, 5]]);
}

/// T-1055 and T-200: a boarding rides at least one stop. Station 1 is a walk from station 0,
/// within 300 m or 1 km off in its complex; line 0 runs 2-0-3 and line 1 3-1. From 0 to 1 the
/// rail trip rides line 0 to 3 and line 1 back: boarding line 0 only to walk off is no trip.
/// Riders already on line 0 from 2 still leave it at 0 and walk over.
#[test]
fn a_boarding_rides_at_least_one_stop() {
    let p = Params { xfer_walk_m: 300.0, ..common_test_params() };
    for (y, group) in [(200.0, vec![]), (1000.0, vec![0, 0, 2, 3])] {
        let mut net = common_test_net(&[&[2, 0, 3], &[3, 1]], &[12.0, 12.0], &[2.0, 2.0]);
        net.st_x = vec![0.0, 0.0, -3000.0, 3000.0];
        net.st_y = vec![0.0, y, 0.0, 0.0];
        net.st_group = group;
        let g = build_graph(&net, &p);
        let s = g.n_st;
        let sk = skims(&g, &p);
        let walk = p.xfer_min(y) * p.w_xfer;
        // wait 2.5, ride 2, change (2 and a wait of 2.5), ride 2
        assert!((sk.r[1] - 11.0).abs() < 1e-4, "y {y}: 0 to 1 costs {}", sk.r[1]);
        // from 2, the riders on board at 0 walk or stay on (2, then the change, then 2)
        assert!((sk.r[2 * s + 1] - (2.5 + 2.0 + walk.min(8.5))).abs() < 1e-4, "y {y}: 2 to 1 costs {}", sk.r[2 * s + 1]);
        // route nodes: line 0 run 0 = 0, 1, 2 (stations 2, 0, 3); line 1 run 0 = 6, 7 (3, 1)
        let mut od = vec![0.0; s * s];
        od[1] = 100.0;
        od[2 * s + 1] = 10.0;
        let l = assign(&g, &sk, &od);
        let on_foot = y < 300.0;
        let want_seg = |r: usize| match r {
            0 => 10.0,
            1 | 6 => if on_foot { 100.0 } else { 110.0 },
            _ => 0.0,
        };
        for r in 0..g.n - s {
            assert!((l.seg[r] - want_seg(r)).abs() < 1e-3, "y {y}: segment {r} carries {}", l.seg[r]);
        }
        for a in [0, 2] {
            for (_, route) in super::super::trip::routes(&g, &sk, &p, a, 1, 0.005) {
                assert!(route.iter().filter(|l| l.kind == 1).all(|l| l.stops >= 1), "y {y}: a ride of no stops from {a}");
                assert_eq!(route.iter().filter(|l| l.kind == 1).count(), if a == 2 && on_foot { 1 } else { 2 }, "y {y} from {a}");
            }
        }
    }
}

/// T-1059, T-2265: the strategies' real minutes between stations, and the rule's test on them.
/// Station 1 is a walk from station 0 (200 m, then 1 km in its complex); line 0 runs 2-0-3, line 1
/// 3-1, as in `a_boarding_rides_at_least_one_stop`: 12 trains an hour (waits 2.5), 2 minute hops.
/// Station 1's platform for line 1 is 50 m off.
#[test]
fn the_walking_rule_reads_the_real_minutes_between_stations() {
    let p = Params { xfer_walk_m: 300.0, ..common_test_params() };
    let eq = |min: f32| p.walk_eq_m(min);
    let near = |a: u16, b: f32| (a as f32 - b).abs() <= 1.0;
    for (y, group) in [(200.0f32, vec![]), (1000.0, vec![0, 0, 2, 3])] {
        let mut net = common_test_net(&[&[2, 0, 3], &[3, 1]], &[12.0, 12.0], &[2.0, 2.0]);
        net.st_x = vec![0.0, 0.0, -3000.0, 3000.0];
        net.st_y = vec![0.0, y, 0.0, 0.0];
        net.st_group = group;
        let g = build_graph(&net, &p);
        let s = g.n_st;
        let sk = skims(&g, &p);
        let m = |a: usize, t: usize| sk.walk[a * s + t];
        // from 0 the trip waits, rides, waits and rides; from 2 it walks over at 0 when that is
        // cheaper (200 m, at the change's pace) and rides on otherwise. No fixed change minutes.
        assert!(near(m(0, 1), eq(2.5 + 2.0 + 2.5 + 2.0)), "y {y}: {}", m(0, 1));
        let from2 = if y < 300.0 { 2.5 + 2.0 + p.xfer_min(200.0) } else { 2.5 + 2.0 + 2.0 + 2.5 + 2.0 };
        assert!(near(m(2, 1), eq(from2)), "y {y}: {} against {}", m(2, 1), eq(from2));
        assert_eq!(m(1, 1), 0);
        // the rule: 150 m to station 2, its minutes over and 100 m from 1
        let w = m(2, 1) as f32;
        assert!(walk_ok(150.0, m(2, 1), 100.0, 250.0 + w));
        assert!(!walk_ok(150.0, m(2, 1), 100.0, 249.0 + w));
        assert!(!walk_ok(0.0, u16::MAX, 0.0, 1e9), "no route is no option");
        // a trip's share kept, over two access and one egress entry: the pair from 0 (2 of 3
        // parts) just kept, the one from 2 kept where it is the quicker
        let direct = 200.0 + m(0, 1) as f32;
        let acc = [(2u32, 1.0f32, 150.0f32, 0u32), (0, 2.0, 100.0, 1)];
        let eg = [(1u32, 1.0f32, 100.0f32, 2u32)];
        let kept = walk_kept(&acc, &eg, |_, _| 1.0, &sk.walk, s, direct);
        let want = if 250.0 + w <= direct { 1.0 } else { 2.0 / 3.0 };
        assert!((kept - want).abs() < 1e-6, "y {y}: {kept}");
        // T-2307: the same pair from 2 refused at a thousandth of the weight is a sliver, let be
        let acc = [(2u32, 0.002f32, 150.0f32, 0u32), (0, 2.0, 100.0, 1)];
        assert_eq!(walk_kept(&acc, &eg, |_, _| 1.0, &sk.walk, s, direct), 1.0, "y {y}");
    }
    // platforms: a change at station 3 walks from line 0's platform to line 1's
    let mut net = common_test_net(&[&[2, 0, 3], &[3, 1]], &[12.0, 12.0], &[2.0, 2.0]);
    net.st_x = vec![0.0, 0.0, -3000.0, 3000.0];
    net.st_y = vec![0.0, 2000.0, 0.0, 0.0];
    net.stop_x = net.line_stops.iter().map(|&st| net.st_x[st as usize]).collect();
    net.stop_y = vec![0.0, 0.0, -25.0, 25.0, 2000.0];
    let g = build_graph(&net, &p);
    let sk = skims(&g, &p);
    assert!(near(sk.walk[1], eq(9.0 + p.xfer_min(50.0))), "0 to 1 changes across 50 m at station 3: {}", sk.walk[1]);
}

/// T-1059 on a small city: the rule takes rail only from the trips it touches, and the slow
/// lanes load what they keep: the station matrix and the access entries add up to the rail
/// trips with the rule on as off, and the pair choice agrees (`trip.rs`).
#[test]
fn the_walking_rule_keeps_the_loads_whole() {
    let city = synth::synth_city(1).inner(15_000.0);
    let net = network::synth_network();
    let on = Params::default();
    assert!(on.walk_rule > 0.0);
    let off = Params { walk_rule: 0.0, ..on.clone() };
    let z = build_zones(&city, on.zone_m);
    let mut gr = gravity(&z, &on, 100, 1e-4);
    gr.set_walk(&z, &on);
    let acc = build_access(&city, &net, &on);
    let sz = build_subzones(&city, &z, &acc, &net, &on);
    let g = build_graph(&net, &on);
    let sk = skims(&g, &on);
    let er = station_exp(&sk, &on);
    let nr = NoRail::new(&z, &on);
    let (a, b) = (mode_choice(&z, &gr, &sz, &er, &sk.walk, &sk.walk_max, g.n_st, &off, &nr), mode_choice(&z, &gr, &sz, &er, &sk.walk, &sk.walk_max, g.n_st, &on, &nr));
    assert!(b.rail < a.rail && b.rail > 0.8 * a.rail, "rail {} with the rule, {} without", b.rail, a.rail);
    for mo in [&a, &b] {
        let od: f64 = mo.od.iter().map(|&x| x as f64).sum();
        let home: f64 = mo.acc_home.iter().map(|&x| x as f64).sum();
        let work: f64 = mo.acc_work.iter().map(|&x| x as f64).sum();
        let bands: f64 = mo.rail_band.iter().sum();
        let to: f64 = mo.rail_to_sub.iter().map(|&x| x as f64).sum();
        for (what, x) in [("station pairs", od), ("access", home), ("egress", work), ("bands", bands), ("by work subzone", to)] {
            assert!((x - mo.rail).abs() < 1e-4 * mo.rail, "{what} {x} against rail {} (rule {})", mo.rail, std::ptr::eq(mo, &b));
        }
    }
}

/// T-176 on a small city with walking transfers: the loads are `assign`'s to the bit, and at
/// every station the transfers leaving are the boardings that are not first boardings.
#[test]
fn transfers_account_for_every_boarding_after_the_first() {
    let city = synth::synth_city(1).inner(15_000.0);
    let net = network::synth_network();
    let p = Params::default();
    assert!(p.xfer_walk_m > 0.0);
    let z = build_zones(&city, p.zone_m);
    let mut gr = gravity(&z, &p, 100, 1e-4);
    gr.set_walk(&z, &p);
    let acc = build_access(&city, &net, &p);
    let sz = build_subzones(&city, &z, &acc, &net, &p);
    let g = build_graph(&net, &p);
    let sk = skims(&g, &p);
    let mo = mode_choice(&z, &gr, &sz, &station_exp(&sk, &p), &sk.walk, &sk.walk_max, g.n_st, &p, &NoRail::new(&z, &p));
    let ix = xfer_index(&g);
    let mut x = vec![0.0; ix.len];
    let l = assign_with_transfers(&g, &sk, &mo.od, &ix, &mut x);
    let plain = assign(&g, &sk, &mo.od);
    assert_eq!(l.seg, plain.seg);
    assert_eq!(l.board, plain.board);
    assert_eq!(l.alight, plain.alight);
    assert_eq!(l.rn_board, plain.rn_board);
    assert_eq!(l.rn_alight, plain.rn_alight);
    assert_eq!(l.entry, plain.entry);
    assert_eq!(l.exit, plain.exit);
    let s = g.n_st;
    let mut out = vec![];
    transfer_pairs(&g, &ix, &x, &vec![1u8; g.n - s], &mut out);
    let mut leaving = vec![0f64; s];
    let mut from = vec![0f64; g.n - s];
    let mut walked = 0.0;
    for c in out.chunks(3) {
        let (a, b) = (c[0] as usize, c[1] as usize);
        leaving[g.rn_station[b] as usize] += c[2] as f64;
        from[a] += c[2] as f64;
        if g.rn_station[a] != g.rn_station[b] {
            walked += c[2] as f64;
        }
    }
    let total: f64 = leaving.iter().sum();
    // (the synthetic network has no stations close enough to walk between: the real network
    // test `transfers_real_network` counts walks)
    assert!(total > 0.0, "transfers {total}, walked {walked}");
    for st in 0..s {
        let want = (l.board[st] - l.entry[st]) as f64;
        assert!((leaving[st] - want).abs() <= 1e-3 * (l.board[st] as f64).max(1.0), "station {st}: {} leaving, {want} boardings after the first", leaving[st]);
    }
    for r in 0..g.n - s {
        assert!(from[r] <= l.rn_alight[r] as f64 * 1.0001 + 1e-3, "route node {r}: {} transfer, {} alight", from[r], l.rn_alight[r]);
    }
}

#[test]
fn fast_exp_is_close() {
    let mut x = -30.0f32;
    while x < 30.0 {
        let (a, b) = (fast_exp(x), x.exp());
        assert!(((a - b) / b).abs() < 3e-4, "x {x}: {a} vs {b}");
        x += 0.0137;
    }
}

/// Small end-to-end run: gravity conserves trips, rail trips are a share of them, every rail
/// trip shows up as a boarding.
#[test]
fn kernel_runs_on_a_small_city() {
    let city = synth::synth_city(1).inner(15_000.0);
    let net = network::synth_network();
    let p = Params::default();
    let z = build_zones(&city, p.zone_m);
    let mut gr = gravity(&z, &p, 100, 1e-4);
    gr.set_walk(&z, &p);
    let (_, tj) = city.totals();
    assert!((gr.total / tj - 1.0).abs() < 1e-3);
    let acc = build_access(&city, &net, &p);
    let sz = build_subzones(&city, &z, &acc, &net, &p);
    let g = build_graph(&net, &p);
    let sk = skims(&g, &p);
    let er = station_exp(&sk, &p);
    let mo = mode_choice(&z, &gr, &sz, &er, &sk.walk, &sk.walk_max, g.n_st, &p, &NoRail::new(&z, &p));
    assert!(mo.rail > 0.0 && mo.rail < mo.trips_access && mo.trips_access <= mo.trips_total * 1.0001);
    let l = assign(&g, &sk, &mo.od);
    let od_sum: f64 = mo.od.iter().map(|&v| v as f64).sum();
    // the split back onto access stations conserves rail trips
    assert!((od_sum / mo.rail - 1.0).abs() < 1e-3, "od {od_sum} vs rail {}", mo.rail);
    let bands: f64 = mo.rail_band.iter().sum();
    assert!((bands / mo.rail - 1.0).abs() < 1e-3);
    // rail trips add up the same by home subzone, work subzone and access entry at each end
    let sum = |v: &[f32]| v.iter().map(|&x| x as f64).sum::<f64>();
    for (what, v) in [("by", &mo.rail_by_sub), ("to", &mo.rail_to_sub), ("home", &mo.acc_home), ("work", &mo.acc_work)] {
        assert!((sum(v) / mo.rail - 1.0).abs() < 1e-3, "{what}: {} vs {}", sum(v), mo.rail);
    }
    // T-558: rail takes walkers, the same by home and by work subzone, and never more than
    // walk with no rail
    let (wb, wt) = (sum(&mo.walk_lost_by_sub), sum(&mo.walk_lost_to_sub));
    assert!(mo.rail_walk > 0.0 && (wb / mo.rail_walk - 1.0).abs() < 1e-3 && (wt / mo.rail_walk - 1.0).abs() < 1e-3, "{wb} {wt} {}", mo.rail_walk);
    let nr = NoRail::new(&z, &p);
    let walk0: f64 = gr.no_rail_by_zone(&z, &p, &nr)[0].iter().sum::<f64>() * p.period_share as f64;
    assert!(mo.rail_walk < walk0, "{} of {walk0}", mo.rail_walk);
    let boards: f64 = l.board.iter().map(|&v| v as f64).sum();
    assert!(boards >= od_sum * 0.999);
    // T-874: small pairs dealt in chunks visit fewer pairs, keep the trips offered rail and
    // rail itself close, and deal the same chunks every pass
    let every = Params { commute_chunk: 0.0, ..p.clone() };
    let big = Params { commute_chunk: 50.0, ..p.clone() };
    let mo0 = mode_choice(&z, &gr, &sz, &er, &sk.walk, &sk.walk_max, g.n_st, &every, &nr);
    let (mo1, mo2) = (mode_choice(&z, &gr, &sz, &er, &sk.walk, &sk.walk_max, g.n_st, &big, &nr), mode_choice(&z, &gr, &sz, &er, &sk.walk, &sk.walk_max, g.n_st, &big, &nr));
    assert!(mo1.pairs < mo0.pairs * 9 / 10, "pairs {} of {}", mo1.pairs, mo0.pairs);
    assert!((mo1.trips_access / mo0.trips_access - 1.0).abs() < 0.01, "offered {} vs {}", mo1.trips_access, mo0.trips_access);
    assert!((mo1.rail / mo0.rail - 1.0).abs() < 0.02, "rail {} vs {}", mo1.rail, mo0.rail);
    assert!(mo1.od == mo2.od && mo1.rail_by_sub == mo2.rail_by_sub);
}

/// T-162: a region border (here a north-south line at x = 300 m, through the middle of a
/// column of zones) splits the squares it cuts, the factor cuts trips across it, both margins
/// still balance, and the pack round trip rebuilds exactly the trips solved.
#[test]
fn pull_cap_leaves_a_far_job_island_short() {
    // T-1000: a town 60 km out with ten times the jobs of its homes
    let mut city = synth::synth_city(1).inner(10_000.0);
    for k in 0..20 {
        city.h3.push(city.h3[k]);
        city.x.push(60_000.0 + 400.0 * (k % 5) as f32);
        city.y.push(400.0 * (k / 5) as f32);
        city.pop.push(100.0);
        city.jobs.push(1000.0);
    }
    let p = Params::default();
    let z = build_zones(&city, p.zone_m);
    let g = gravity(&z, &p, 1000, 1e-10);
    assert!(g.max_err < 1e-10, "{} after {} iterations", g.max_err, g.iters);
    let [lo, hi, fill, held] = g.pull.unwrap();
    assert!(held > 0.0 && fill > 1.0, "held {held}, fill {fill}");
    let island = |j: usize| z.gx[j] as f32 * z.zone_m + z.x0 > 50_000.0;
    let o: Vec<f64> = { let (tp, tj): (f64, f64) = (z.pop.iter().sum(), z.jobs.iter().sum()); z.pop.iter().map(|&v| v * tj / tp).collect() };
    for j in (0..z.n).filter(|&j| z.jobs[j] > 0.0) {
        let (rs, cs): (f64, f64) = ((0..z.n).map(|k| g.t(&z, j, k)).sum(), (0..z.n).map(|i| g.t(&z, i, j)).sum());
        assert!(o[j] == 0.0 || (rs / o[j] - 1.0).abs() < 1e-8);
        let pull = g.col[j] / z.jobs[j];
        assert!(pull >= lo * (1.0 - 1e-9) && pull <= hi * (1.0 + 1e-9));
        if island(j) {
            assert!(pull >= hi * (1.0 - 1e-9) && cs < z.jobs[j], "island zone {j}: {cs} of {}", z.jobs[j]);
        } else if pull < hi * (1.0 - 1e-9) && pull > lo * (1.0 + 1e-9) {
            assert!((cs / z.jobs[j] / fill - 1.0).abs() < 1e-6, "zone {j}: {} against {fill}", cs / z.jobs[j]);
        }
    }
}

#[test]
fn region_border_factor() {
    let mut city = synth::synth_city(1).inner(15_000.0);
    let p1 = Params::default();
    let z0 = build_zones(&city, p1.zone_m);
    let g0 = gravity(&z0, &p1, 200, 1e-6);
    city.region = city.x.iter().map(|&x| (x > 300.0) as u8).collect();
    let z = build_zones(&city, p1.zone_m);
    assert!(z.n > z0.n, "the border splits squares");
    let (off, _) = z.by_square();
    assert_eq!(off.windows(2).filter(|w| w[1] > w[0]).count(), z0.n, "same squares");
    let cross = |z: &Zones, g: &Gravity| -> f64 {
        let mut s = 0.0;
        for i in 0..z.n {
            for j in 0..z.n {
                if z.region[i] != z.region[j] {
                    s += g.t(z, i, j);
                }
            }
        }
        s
    };
    // factor 1: the split changes nothing but where the intrazonal trips of cut squares go
    let g1 = gravity(&z, &p1, 200, 1e-6);
    assert!((g1.total / g0.total - 1.0).abs() < 1e-6);
    let pk = Params { border_factor: 0.25, ..Params::default() };
    let gk = gravity(&z, &pk, 200, 1e-6);
    assert!((gk.total / g0.total - 1.0).abs() < 1e-6);
    let (c1, ck) = (cross(&z, &g1), cross(&z, &gk));
    assert!(ck < 0.6 * c1, "cross-border trips {ck} with the factor, {c1} without");
    // both margins hold with the factor
    let (tp, tj): (f64, f64) = (z.pop.iter().sum(), z.jobs.iter().sum());
    for i in 0..z.n {
        let (r, c): (f64, f64) = ((0..z.n).map(|j| gk.t(&z, i, j)).sum(), (0..z.n).map(|j| gk.t(&z, j, i)).sum());
        assert!((r - z.pop[i] * tj / tp).abs() <= 1e-3 * z.pop[i] * tj / tp + 1e-6, "row {i}");
        assert!((c - z.jobs[i]).abs() <= 1e-3 * z.jobs[i] + 1e-6, "column {i}");
    }
    // shipped and rebuilt: the same zones, factor and trips
    city.gravity = Some(to_pack_gravity(&z, &gk, &pk));
    let (h, b) = crate::demand::pack::write(&city);
    let back = crate::demand::pack::parse(&h, &b).unwrap();
    let (zb, gb, from_pack) = city_gravity(&back, &p1);
    assert!(from_pack && zb.n == z.n && zb.region == z.region && gb.border == 0.25);
    for (i, j) in [(0, 1), (3, z.n - 1), (z.n / 2, z.n / 3)] {
        assert!((gb.t(&zb, i, j) / gk.t(&z, i, j).max(1e-30) - 1.0).abs() < 1e-5 || gk.t(&z, i, j) < 1e-12);
    }
}

/// T-852: the synthetic city's inner 15 km, two regions (border at x = 300 m), with every
/// aligned block of zones whose centres are over 6 km out made one zone.
fn mixed_city() -> (crate::demand::pack::City, Zones, Zones) {
    let mut city = synth::synth_city(1).inner(15_000.0);
    city.region = city.x.iter().map(|&x| (x > 300.0) as u8).collect();
    let z0 = build_zones(&city, 2000.0);
    let out = |i: usize| (z0.x0 + (z0.gx[i] as f32 + 0.5) * 2000.0).hypot(z0.y0 + (z0.gy[i] as f32 + 0.5) * 2000.0) > 6000.0;
    let ends: Vec<f64> = (0..z0.n).map(|i| if out(i) { 10.0 } else { 1e6 }).collect();
    let z = coarsen(&z0, &ends, 1000.0);
    (city, z0, z)
}

/// T-852: zones of two sizes. Every pair's table entry is the mean of the decay between their
/// squares and its distance that of their centres; the gravity balances to the same trips as
/// the squares'; near pairs, a square's zone and the pack round trip hold.
#[test]
fn mixed_zones() {
    let (mut city, z0, z) = mixed_city();
    let blocks = z.size.iter().filter(|&&s| s == 2).count();
    assert!(blocks > 20 && z.n < z0.n - blocks && z.w % 2 == 0, "{blocks} blocks, {} of {} zones", z.n, z0.n);
    assert!((z.pop.iter().sum::<f64>() / z0.pop.iter().sum::<f64>() - 1.0).abs() < 1e-9);
    for c in 0..city.len() {
        let i = z.cell_zone[c] as usize;
        let (gx, gy) = (((city.x[c] - z.x0) / 2000.0) as i32, ((city.y[c] - z.y0) / 2000.0) as i32);
        assert!(gx - z.gx[i] < z.side(i) && gx >= z.gx[i] && gy - z.gy[i] < z.side(i) && gy >= z.gy[i] && z.region[i] == city.region[c]);
    }
    let p = Params::default();
    let g = gravity(&z, &p, 300, 1e-7);
    let g0 = gravity(&z0, &p, 300, 1e-7);
    assert!((g.total / g0.total - 1.0).abs() < 1e-5 && (g.mean_km / g0.mean_km - 1.0).abs() < 0.04, "{} km against {}", g.mean_km, g0.mean_km);
    let (_, dtab) = decay_tables(&z, Curve { km: 1.0, pow: 0.0, flat_km: 0.0 });
    let f = |d: f32| (-d / p.decay_km).exp() * d.powf(-p.decay_pow);
    let sq = |i: usize| -> Vec<(i32, i32)> {
        let (s, gx, gy) = (z.side(i), z.gx[i], z.gy[i]);
        (0..s).flat_map(|a| (0..s).map(move |b| (gx + a, gy + b))).collect()
    };
    for i in 0..z.n {
        for j in 0..z.n {
            let k = g.base(&z, i) + g.key(&z, j);
            assert!(k >= 0 && (k as usize) < g.ftab.len());
            let (si, sj) = (sq(i), sq(j));
            let mean = si.iter().flat_map(|a| sj.iter().map(move |b| (a, b))).map(|(a, b)| {
                let d = if a == b { 1.04 } else { 2.0 * (((a.0 - b.0).pow(2) + (a.1 - b.1).pow(2)) as f32).sqrt() };
                f(d)
            }).sum::<f32>() / (si.len() * sj.len()) as f32;
            let got = g.ftab[k as usize];
            assert!((got / mean - 1.0).abs() < 1e-5, "{i} {j}: {got} vs {mean}");
            let d = (z.d2_m(i, j).sqrt() / 1000.0).max(if z.d2_m(i, j) == 0.0 { 1.04 * z.side(i) as f32 } else { 0.0 });
            assert!((dtab[k as usize] - d).abs() < 1e-4 && (g.dist_km(k as usize) - d).abs() < 1e-4, "{i} {j}: {} km vs {d}", dtab[k as usize]);
        }
        // both margins
        let r: f64 = (0..z.n).map(|j| g.t(&z, i, j)).sum();
        let c: f64 = (0..z.n).map(|j| g.t(&z, j, i)).sum();
        assert!((c - z.jobs[i]).abs() <= 1e-4 * z.jobs[i] + 1e-6 && r >= 0.0, "column {i}");
    }
    // near pairs by squares, and the walk between cells for each of them
    let mut gw = g.clone();
    gw.set_walk(&z, &p);
    gw.set_walk_fine(&city, &z, &p, &NoRail::new(&z, &p));
    let byq = z.by_square();
    for i in 0..z.n {
        let mut got = vec![];
        z.near(&byq, i, 2, |j| got.push(j));
        let (si, mut want) = (sq(i), vec![]);
        for j in 0..z.n {
            if sq(j).iter().any(|b| si.iter().any(|a| (a.0 - b.0).abs() <= 2 && (a.1 - b.1).abs() <= 2)) {
                want.push(j);
            }
        }
        got.sort();
        assert_eq!(got, want, "near {i}");
        let fine = &gw.wf_j[gw.wf_off[i] as usize..gw.wf_off[i + 1] as usize];
        for &j in fine {
            let k = (gw.base(&z, i) + gw.key(&z, j as usize)) as usize;
            assert!(gw.fine_at(i, j as usize, k).is_some(), "fine {i} {j}");
        }
    }
    // a square's zone (one region: the squares west of the border)
    for c in (0..city.len()).filter(|&c| city.region[c] == 0) {
        let (gx, gy) = (((city.x[c] - z.x0) / 2000.0) as i32, ((city.y[c] - z.y0) / 2000.0) as i32);
        let at = z.zone_at(&byq, gx, gy).unwrap();
        assert!(at == z.cell_zone[c] as usize || z.region[at] != 0);
    }
    // shipped and rebuilt
    city.gravity = Some(to_pack_gravity(&z, &g, &p));
    let (h, b) = crate::demand::pack::write(&city);
    assert!(h.contains("\"zone_size\""));
    let back = crate::demand::pack::parse(&h, &b).unwrap();
    let (zb, gb, from_pack) = city_gravity(&back, &p);
    assert!(from_pack && zb.size == z.size && zb.w == z.w);
    for (i, j) in [(0, 1), (3, z.n - 1), (z.n / 2, z.n / 3), (z.n - 1, z.n - 1)] {
        assert!((gb.t(&zb, i, j) / g.t(&z, i, j).max(1e-30) - 1.0).abs() < 1e-5 || g.t(&z, i, j) < 1e-12);
    }
}

/// T-1091: the flat near end weighs a job in the home square as one in the next square and
/// leaves pairs 2 km apart and more as they were; a block's pairs take the mean over its
/// squares of the flat curve; the US curve takes none; a pack solved with it keeps fewer
/// commutes at home and opens from its factors only for the same flat near end.
#[test]
fn flat_near_end() {
    let (mut city, z0, z) = mixed_city();
    assert_eq!(flat_near_km(US_DECAY_KM, US_DECAY_POW), 0.0);
    assert_eq!(flat_near_km(80.0, 1.8), FLAT_NEAR_KM);
    let c = Curve { km: 16.0, pow: 1.4, flat_km: FLAT_NEAR_KM };
    let c0 = Curve { flat_km: 0.0, ..c };
    let (f, d) = decay_table(z0.w, z0.h, z0.zone_m, c);
    let (f0, _) = decay_table(z0.w, z0.h, z0.zone_m, c0);
    for k in 0..f.len() {
        if d[k] >= 2.0 {
            assert_eq!(f[k], f0[k]);
        } else {
            assert!(f[k] == c.at(2.0) && f0[k] > 2.0 * f[k], "{} km: {} against {}", d[k], f[k], f0[k]);
        }
    }
    // a block's own pair: of its 16 square pairs, 4 own and 8 side by side at 2 km, 4 diagonal
    let fb = decay_ftab(&z, c);
    let pk = PairKey::of(&z);
    let b = (0..z.n).find(|&i| z.size[i] == 2).unwrap();
    let own = fb[(pk.base(&z, b) + pk.key(&z, b)) as usize];
    let want = (12.0 * c.at(2.0) + 4.0 * c.at(2.0 * 2f32.sqrt())) / 16.0;
    assert!((own / want - 1.0).abs() < 1e-6, "{own} against {want}");
    // solved: fewer commutes in their own zone, the mean a little longer
    let p0 = Params { decay_km: c.km, decay_pow: c.pow, ..Params::default() };
    let p = Params { decay_flat_km: c.flat_km, ..p0.clone() };
    let (g0, g) = (gravity(&z, &p0, 300, 1e-7), gravity(&z, &p, 300, 1e-7));
    let own_share = |g: &Gravity| (0..z.n).map(|i| g.t(&z, i, i)).sum::<f64>() / g.total;
    assert!(own_share(&g) < 0.7 * own_share(&g0) && g.mean_km > g0.mean_km, "own {} against {}", own_share(&g), own_share(&g0));
    // shipped with its flat near end, opened from its factors with it and re-solved without
    city.gravity = Some(to_pack_gravity(&z, &g, &p));
    let (h, bin) = crate::demand::pack::write(&city);
    assert!(h.contains("\"decay_flat_km\": 2.0"));
    let back = crate::demand::pack::parse(&h, &bin).unwrap();
    assert_eq!(back.gravity.as_ref().unwrap().curve(), c);
    let (zb, gb, from_pack) = city_gravity(&back, &p);
    assert!(from_pack && gb.ftab == g.ftab);
    assert!((own_share(&gb) / own_share(&g) - 1.0).abs() < 1e-5 && zb.n == z.n);
    assert!(!city_gravity(&back, &p0).2);
}

/// T-852: commutes on the synthetic network with the outer zones in blocks keep their rail
/// and their trips offered rail close to the squares'.
#[test]
fn mixed_zones_ride_like_squares() {
    let (city, z0, z) = mixed_city();
    let net = network::synth_network();
    let p = Params::default();
    let run = |z: &Zones| {
        let mut gr = gravity(z, &p, 200, 1e-5);
        gr.set_walk(z, &p);
        let nr = NoRail::new(z, &p);
        gr.set_walk_fine(&city, z, &p, &nr);
        let acc = build_access(&city, &net, &p);
        let sz = build_subzones(&city, z, &acc, &net, &p);
        let g = build_graph(&net, &p);
        let sk = skims(&g, &p);
        let er = station_exp(&sk, &p);
        let mo = mode_choice(z, &gr, &sz, &er, &sk.walk, &sk.walk_max, g.n_st, &p, &nr);
        let walk: f64 = gr.no_rail_by_zone(z, &p, &nr)[0].iter().sum();
        (mo.rail, mo.trips_access, walk)
    };
    let (r0, a0, w0) = run(&z0);
    let (r, a, w) = run(&z);
    assert!(r0 > 0.0 && (r / r0 - 1.0).abs() < 0.05 && (a / a0 - 1.0).abs() < 0.05 && (w / w0 - 1.0).abs() < 0.05, "rail {r} vs {r0}, offered {a} vs {a0}, walk {w} vs {w0}");
}

/// The ring search finds the same K nearest stations within the bound as a full scan.
#[test]
fn access_matches_a_full_scan() {
    let city = synth::synth_city(3);
    let net = network::synth_network();
    for cut in [800.0f32, 2500.0, 5000.0] {
        let p = Params { walk_cutoff_m: cut, ..Params::default() };
        let a = build_access(&city, &net, &p);
        for c in (0..city.len()).step_by(37) {
            let mut d: Vec<(f32, u32)> = (0..net.n_stations())
                .map(|s| ((net.st_x[s] - city.x[c]).hypot(net.st_y[s] - city.y[c]), s as u32))
                .filter(|e| e.0 <= cut)
                .collect();
            d.sort_by(|x, y| x.partial_cmp(y).unwrap());
            d.truncate(K);
            assert_eq!(a.n[c] as usize, d.len(), "cell {c} cut {cut}");
            for (k, e) in d.iter().enumerate() {
                assert!((a.walk[c * K + k] - p.walk_min(e.0)).abs() < 1e-3, "cell {c} k {k}");
            }
        }
    }
}

/// The walk's perceived cost: `w_walk` a minute for the first `walk_easy_min`, `w_walk_far`
/// after (T-371: the same, a linear walk); the drive is slower in dense places and on short
/// trips.
#[test]
fn access_and_drive_costs() {
    let p = Params::default();
    let easy_m = p.walk_easy_min * 60.0 * p.walk_mps / p.walk_detour;
    assert!((p.access_cost(easy_m) - p.walk_easy_min * p.w_walk).abs() < 1e-3);
    let km = |d: f32| p.access_cost(d + 1000.0) - p.access_cost(d);
    assert!((km(easy_m + 100.0) / km(0.0) - p.w_walk_far / p.w_walk).abs() < 1e-3);
    // 13 real minutes a straight km at 6 km/h on 1.3x the line
    assert!((p.walk_min(1000.0) - 13.0).abs() < 1e-3);
    let (dense, open) = (p.place_min_per_km(50_000.0), p.place_min_per_km(100.0));
    assert!(dense.0 > open.0 && dense.1 > open.1);
    // 30 km between open places is much quicker a km than 3 km
    assert!(p.drive_min(30_000.0, open, open) / 30.0 < 0.6 * p.drive_min(3_000.0, open, open) / 3.0);
    assert!(p.drive_min(10_000.0, dense, dense) > 2.0 * p.drive_min(10_000.0, open, open));
}

/// T-370: a long drive into a dense place is slow only near its ends (real peak drives of
/// 40-60 km into Manhattan, Philadelphia or Boston take 50-90 minutes); a short one is all
/// at its ends' speeds.
#[test]
fn long_drive_slow_only_near_its_ends() {
    let p = Params::default();
    let (city, suburb) = (p.place_min_per_km(30_000.0), p.place_min_per_km(1_000.0));
    let t = p.drive_min(60_000.0, suburb, city);
    assert!((60.0..90.0).contains(&t), "60 km into a 30,000/km² place: {t} min");
    let road = 60.0 * p.road_detour;
    let t_end = p.road_local_km * (suburb.0 + city.0) + p.road_end_km * (suburb.1 + city.1);
    let open = (road - 2.0 * (p.road_local_km + p.road_end_km)) * 60.0 / p.road_open_kmh;
    assert!((t - t_end - open).abs() < 1e-3);
    let road = 15.0 * p.road_detour;
    let short = p.road_local_km * (suburb.0 + city.0) + (road - 2.0 * p.road_local_km) * 0.5 * (suburb.1 + city.1);
    assert!((p.drive_min(15_000.0, suburb, city) - short).abs() < 1e-3);
}

#[test]
fn fast_ln_is_close() {
    let mut x = 1e-30f32;
    while x < 1e30 {
        let (a, b) = (fast_ln(x), x.ln());
        assert!((a - b).abs() < 4e-6 * b.abs().max(1.0), "x {x}: {a} vs {b}");
        x *= 1.0137;
    }
}

/// T-2652: with leaf stations out of the queue every strategy is the plain search's to the bit
/// (`DestSearch::check_plain`, every destination) and the loads too: on the synthetic city's
/// network with its walks, on a circle (all but one station leaves) with and without its wrap, and
/// on two lines with a branch; each again with waits of their own at some boardings (`rn_left`), so
/// a leaf's offers come at their boardings' turns.
#[test]
fn leaf_stations_leave_every_strategy_as_it_was() {
    let p = Params::default();
    let ring = |wrap: bool| {
        let st: Vec<(f32, f32)> = (0..6).map(|i| { let a = i as f32 * std::f32::consts::PI / 3.0; (3000.0 * a.cos(), 3000.0 * a.sin()) }).collect();
        Network {
            st_x: st.iter().map(|s| s.0).collect(), st_y: st.iter().map(|s| s.1).collect(),
            line_name: vec![String::new()], line_off: vec![0, 7], line_stops: vec![0, 1, 2, 3, 4, 5, 0],
            hop_min: (0..7).map(|k| if k < 6 { 2.0 } else if wrap { 0.5 } else { 0.0 }).collect(), hop_back: vec![], line_speed_kmh: vec![0.0],
            line_headway_min: vec![[5.0; network::PERIODS]], line_seats: vec![100.0], line_rated: vec![], line_crush: vec![200.0],
            st_group: vec![], stop_x: vec![], stop_y: vec![], line_wrap: vec![wrap], span: vec![], st_ups: vec![], st_passage: vec![], catches: vec![], st_bus: vec![], st_tram: vec![], back_stops: vec![], line_pier: vec![], stop_street: vec![], bus_walks: None,
        }
    };
    let nets = [network::synth_network(), ring(true), ring(false), common_test_net(&[&[0, 1, 2, 3], &[0, 1, 2]], &[6.0, 12.0], &[2.0, 3.0])];
    super::search::CHECK_PLAIN.with(|c| c.set(true));
    for (i, net) in nets.iter().enumerate() {
        let p = if i == 0 { p.clone() } else { common_test_params() };
        for waits in [false, true] {
            let mut g = build_graph(net, &p);
            if waits {
                for r in (0..g.rn_left.len()).step_by(3) {
                    g.rn_left[r] += 1.5;
                }
            }
            let s = g.n_st;
            let (_, leaves, _) = DestSearch::new(&g, &p).counts();
            assert!(leaves > 0, "network {i}: no leaf stations");
            let od: Vec<f32> = (0..s * s).map(|k| if k / s == k % s { 0.0 } else { 1.0 + (k % 7) as f32 }).collect();
            let a = assign(&g, &skims(&g, &p), &od);
            set_plain_search(true);
            let b = assign(&g, &skims(&g, &p), &od);
            set_plain_search(false);
            let bits = |l: &Loads| [&l.seg, &l.board, &l.alight, &l.rn_board, &l.rn_alight, &l.entry, &l.exit].map(|v| v.iter().map(|x| x.to_bits()).collect::<Vec<u32>>());
            assert_eq!(bits(&a), bits(&b), "network {i} (waits {waits}): loads");
        }
    }
    super::search::CHECK_PLAIN.with(|c| c.set(false));
}


/// T-3348: a line whose reverse run stops at a platform beside the listed stop (one-way track): the
/// reverse run's route node is at that station, riders board and alight there going back, and
/// the listed stop has no reverse-run service.
#[test]
fn a_reverse_run_boards_at_its_own_platform_beside_the_stop() {
    let p = common_test_params();
    let mut net = common_test_net(&[&[0, 1, 2]], &[6.0], &[2.0]);
    // station 4: the westbound platform 150 m from station 1
    net.st_x.push(2000.0);
    net.st_y.push(150.0);
    net.back_stops = vec![0, 4, 2];
    assert_eq!(net.back_only(0).collect::<Vec<_>>(), vec![4]);
    let g = build_graph(&net, &p);
    assert_eq!(g.rn_station, vec![0, 1, 2, 2, 4, 0]);
    let sk = skims(&g, &p);
    let at = |i: usize, j: usize| sk.r[i * 5 + j];
    assert!(at(4, 0).is_finite() && at(2, 4).is_finite() && at(0, 1).is_finite());
    // from the listed stop west only by riding to the end and back, past the platform beside
    assert!(at(1, 0) > at(4, 0) + 2.0 * 2.0, "{} {}", at(1, 0), at(4, 0));
    let mut od = vec![0.0f32; 25];
    od[4 * 5] = 10.0; // 4 -> 0
    od[2 * 5 + 4] = 5.0; // 2 -> 4
    od[1] = 3.0; // 0 -> 1
    let l = assign(&g, &sk, &od);
    let near = |a: f32, b: f32| (a - b).abs() < 1e-3;
    assert!(near(l.rn_board[4], 10.0) && near(l.rn_alight[4], 5.0), "{:?} {:?}", l.rn_board, l.rn_alight);
    assert!(near(l.board[4], 10.0) && near(l.alight[4], 5.0) && near(l.alight[1], 3.0) && near(l.board[1], 0.0));
    // the station is one of the line's for access, as its listed stops are
    assert_eq!(on_train(&net), vec![true, true, true, false, true]);
}
