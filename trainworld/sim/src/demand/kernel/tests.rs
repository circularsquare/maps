use super::*;
use crate::demand::{network, synth};

/// T-434: the first round always runs; a later one is skipped once the loads move under 1%.
#[test]
fn crowding_rounds_stop_once_settled() {
    let mut seen = vec![];
    assert!(!settled(&mut seen, &[100.0, 100.0]));
    assert!(!settled(&mut seen, &[100.0, 98.0]));
    assert!(settled(&mut seen, &[100.0, 96.5]));
    assert!(seen == [100.0, 98.0]);
    assert!(!settled(&mut seen, &[100.0, 95.0]));
}

fn common_test_net(lines: &[&[u32]], tph: &[f32], ride: &[f32]) -> Network {
    let mut net = Network {
        st_x: vec![0.0, 2000.0, 4000.0, 4000.0], st_y: vec![0.0, 0.0, 0.0, 2000.0],
        line_name: vec![String::new(); lines.len()], line_off: vec![0], line_stops: vec![],
        hop_min: vec![], hop_back: vec![], line_speed_kmh: vec![0.0; lines.len()],
        line_headway_min: tph.iter().map(|t| [if *t > 0.0 { 60.0 / t } else { f32::INFINITY }; network::PERIODS]).collect(),
        line_seats: vec![100.0; lines.len()], line_crush: vec![200.0; lines.len()],
        st_group: vec![], stop_x: vec![], stop_y: vec![], line_wrap: vec![], span: vec![],
    };
    for (l, stops) in lines.iter().enumerate() {
        net.line_stops.extend_from_slice(stops);
        net.line_off.push(net.line_stops.len() as u32);
        net.hop_min.extend((0..stops.len()).map(|k| if k + 1 < stops.len() { ride[l] } else { 0.0 }));
    }
    net
}

fn common_test_params() -> Params {
    Params { xfer_walk_m: 0.0, ..Params::default() }
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
            g.fare_curve = FareCurve { km: knots.iter().map(|k| k.0).collect(), min: knots.iter().map(|k| k.1).collect(), base, rate };
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
    // 0 -> 2 rides either line from 0 by its trains an hour (6 and 12); 0 -> 3 only line 0
    assert!((l.rn_fare_x[8] - 10.0 * more * 10.0 * 2.0 / 3.0).abs() < 1e-3, "{:?}", l.rn_fare_x);
    assert!((l.rn_fare_x[0] - (10.0 * more * 10.0 / 3.0 + 5.0 * more * 30.0)).abs() < 1e-3, "{:?}", l.rn_fare_x);
    assert!(assign(&graph(&[]), &skims(&graph(&[]), &p), &od).rn_fare_x.iter().all(|&x| x == 0.0));
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
            line_headway_min: vec![[5.0; network::PERIODS]], line_seats: vec![100.0], line_crush: vec![200.0],
            st_group: vec![], stop_x: vec![], stop_y: vec![], line_wrap: vec![wrap], span: vec![],
        }
    };
    let g = build_graph(&ring(true), &p);
    let s = g.n_st;
    assert_eq!(g.n - s, 14, "two route nodes a stop a direction, as for any line");
    let sk = skims(&g, &p);
    // 5 to 1: half a headway's wait at 5, two stops and the stand at 0
    let short = 2.5 + 2.0 + 0.5 + 2.0;
    assert!((sk.r[5 * s + 1] - short).abs() < 1e-4, "{} against {short}", sk.r[5 * s + 1]);
    assert!((sk.r[1 * s + 5] - short).abs() < 1e-4, "the other way: {}", sk.r[1 * s + 5]);
    let mut od = vec![0.0; s * s];
    od[5 * s + 1] = 100.0;
    od[1 * s + 5] = 40.0;
    let loads = assign(&g, &sk, &od);
    // route nodes: run 0 = 0..7 (stations 0,1,2,3,4,5,0), run 1 = 7..14 (0,5,4,3,2,1,0)
    assert_eq!(loads.seg[6], 100.0, "riders stay aboard through 0 on run 0");
    assert_eq!(loads.seg[13], 40.0, "and on run 1");
    assert_eq!(loads.rn_board.iter().sum::<f32>(), 140.0, "one boarding each");
    assert_eq!((loads.rn_board[5], loads.rn_alight[1]), (100.0, 100.0));
    assert_eq!((loads.rn_board[12], loads.rn_alight[8]), (40.0, 40.0));
    // nobody boards at a wrapped last stop: the first stop's route node has the same trains
    assert_eq!(loads.rn_board[6] + loads.rn_board[13], 0.0);
    // without the wrap trains turn back at 0: riders split between the long way (four stops, 8
    // from boarding) and a change at 0 onto the line out to 1 (2, 2 + 2.5, 2: T-1268)
    let plain = skims(&build_graph(&ring(false), &p), &p);
    let want = (8.0 + 8.5) / 2.0 + 1.25;
    assert!((plain.r[5 * s + 1] - want).abs() < 1e-4, "{} against {want}", plain.r[5 * s + 1]);
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
    let mut p = Params::default();
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

#[test]
fn common_lines_pool_wait_and_split_by_frequency() {
    let p = common_test_params();
    let net = common_test_net(&[&[0, 1], &[0, 1]], &[6.0, 12.0], &[2.0, 2.0]);
    let g = build_graph(&net, &p);
    let sk = skims(&g, &p);
    assert!((sk.r[1] - (2.0 + 30.0 / 18.0)).abs() < 1e-5);
    let mut od = vec![0.0; 16]; od[1] = 90.0; od[4] = 30.0;
    let loads = assign(&g, &sk, &od);
    assert!((loads.seg[0] - 30.0).abs() < 1e-5);
    assert!((loads.seg[4] - 60.0).abs() < 1e-5);
    assert!((loads.seg[2] - 10.0).abs() < 1e-5);
    assert!((loads.seg[6] - 20.0).abs() < 1e-5);
    assert_eq!(loads.board.iter().sum::<f32>(), 120.0);
    assert_eq!(loads.alight.iter().sum::<f32>(), 120.0);
    assert_eq!(loads.rn_board.iter().sum::<f32>(), 120.0);
    assert_eq!(loads.rn_alight.iter().sum::<f32>(), 120.0);
    // Two identical 6-tph lines give exactly the same cost as one 12-tph line.
    let one = skims(&build_graph(&common_test_net(&[&[0, 1]], &[12.0], &[2.0]), &p), &p);
    let two = skims(&build_graph(&common_test_net(&[&[0, 1], &[0, 1]], &[6.0, 6.0], &[2.0, 2.0]), &p), &p);
    assert_eq!(one.r, two.r);
}

#[test]
fn common_lines_are_destination_specific_and_ignore_unhelpful_service() {
    let p = common_test_params();
    let net = common_test_net(&[&[0, 1, 2], &[0, 1, 3]], &[6.0, 6.0], &[2.0, 2.0]);
    let g = build_graph(&net, &p); let sk = skims(&g, &p);
    assert!((sk.r[1] - 4.5).abs() < 1e-5, "shared trunk has combined 12 tph");
    assert!((sk.r[2] - 9.0).abs() < 1e-5, "branch has only its useful 6 tph");
    let mut od = vec![0.0; 16]; od[2] = 100.0;
    let loads = assign(&g, &sk, &od);
    assert_eq!(loads.rn_board[0], 100.0);
    assert_eq!(loads.rn_board[6], 0.0);
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
    // from 1: L back to C (10 from boarding) and L on to M at 2 (0.5 + 11), both 12 tph
    let want = (10.0 + 11.5) / 2.0 + 1.25;
    assert!((sk.r[1 * s + 3] - want).abs() < 1e-4, "{} against {want}", sk.r[1 * s + 3]);
    assert_eq!(tph[l_up], 12.0);
    let mut od = vec![0.0; s * s];
    od[1 * s + 3] = 100.0;
    let loads = assign(&g, &sk, &od);
    assert_eq!(loads.rn_board[m_at_2], 50.0);
    assert_eq!(loads.rn_board[l_back_at_2], 0.0, "nobody off L at 2 rides L back");
    assert_eq!(loads.board[2], 50.0);
    assert_eq!(loads.alight[3], 100.0);
    let r = super::super::trip::routes(&g, &sk, &p, 1, 3, 0.005);
    assert_eq!(r.len(), 2);
    for (share, legs) in &r {
        assert!((share - 0.5).abs() < 1e-6 && legs.len() == 2);
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
            line_headway_min: vec![[5.0; network::PERIODS]], line_seats: vec![100.0], line_crush: vec![200.0],
            st_group: vec![], stop_x: vec![], stop_y: vec![], line_wrap: vec![false], span: vec![],
        }
    };
    let g = build_graph(&lolly(true), &p);
    let s = g.n_st;
    let sk = skims(&g, &p);
    // L1 to L8: 2.5 waiting, 2 to J, the change's 2 and 2.5, 2 to L8; round the loop it is 16.5
    let across = 2.5 + 2.0 + 2.0 + 2.5 + 2.0;
    assert!((sk.r[3 * s + 10] - across).abs() < 1e-4, "{} against {across}", sk.r[3 * s + 10]);
    // trips with a direct train do not change: T0 to L1, L8 to T0 through J, T1 to T0 on both
    assert!((sk.r[3] - (2.5 + 6.0)).abs() < 1e-4, "{}", sk.r[3]);
    assert!((sk.r[10 * s] - (2.5 + 6.0)).abs() < 1e-4, "{}", sk.r[10 * s]);
    assert!((sk.r[s] - (1.25 + 2.0)).abs() < 1e-4, "{}", sk.r[s]);
    let mut od = vec![0.0; s * s];
    od[3 * s + 10] = 100.0;
    od[10 * s] = 30.0;
    od[3] = 20.0;
    let loads = assign(&g, &sk, &od);
    // route nodes: run 0 = 0..14 (T0 T1 J L1..L8 J T1 T0), run 1 = 14..28 (T0 T1 J L8..L1 J T1 T0)
    let (l1_back, j_on_to_l8) = (24, 16);
    assert_eq!((loads.rn_board[l1_back], loads.rn_board[j_on_to_l8]), (100.0, 100.0));
    assert_eq!(loads.rn_board.iter().sum::<f32>(), 250.0, "a change for L1 to L8 only");
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
    assert!((sk.r[s] - (1.25 + 20.0 + 2.0)).abs() < 1e-3, "{}", sk.r[s]);
}

#[test]
fn common_lines_use_expected_ride_cost_and_respect_fare_and_crowding() {
    let p = common_test_params();
    let mut g = build_graph(&common_test_net(&[&[0, 1], &[0, 1]], &[6.0, 12.0], &[2.0, 3.0]), &p);
    let sk = skims(&g, &p);
    assert!((sk.r[1] - (30.0 / 18.0 + (6.0 * 2.0 + 12.0 * 3.0) / 18.0)).abs() < 1e-5);
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
    let total: f32 = legs.iter().map(|l| l.perc + l.wait_perc + l.change).sum();
    assert!((total - sk.r[2]).abs() < 1e-4, "the legs add up to the strategy: {total}");
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

/// T-1059: the strategies' walks between stations, and the rule's test on them. Station 1 is a
/// walk from station 0 (200 m, then 1 km in its complex); line 0 runs 2-0-3, line 1 3-1, as in
/// `a_boarding_rides_at_least_one_stop`. Station 1's platform for line 1 is 50 m off.
#[test]
fn the_walking_rule_reads_the_walks_between_stations() {
    let p = Params { xfer_walk_m: 300.0, ..common_test_params() };
    for (y, group) in [(200.0f32, vec![]), (1000.0, vec![0, 0, 2, 3])] {
        let mut net = common_test_net(&[&[2, 0, 3], &[3, 1]], &[12.0, 12.0], &[2.0, 2.0]);
        net.st_x = vec![0.0, 0.0, -3000.0, 3000.0];
        net.st_y = vec![0.0, y, 0.0, 0.0];
        net.st_group = group;
        let g = build_graph(&net, &p);
        let s = g.n_st;
        let sk = skims(&g, &p);
        let m = |a: usize, t: usize| sk.walk[a * s + t];
        // from 0 the trip rides both lines and walks nothing; from 2 it walks over at 0 when
        // that is cheaper (200 m) and rides on otherwise
        assert_eq!(m(0, 1), 0, "y {y}");
        assert_eq!(m(2, 1), if y < 300.0 { 200 } else { 0 }, "y {y}");
        assert_eq!(m(1, 1), 0);
        // the rule: 150 m to station 2, its 200 m over and 100 m from 1 is 450 m
        assert!(walk_ok(150.0, m(2, 1), 100.0, 460.0) == (y >= 300.0 || 450.0 <= 460.0));
        assert!(!walk_ok(150.0, m(2, 1), 100.0, 440.0) || y >= 300.0);
        assert!(!walk_ok(0.0, u16::MAX, 0.0, 1e9), "no route is no option");
        // a trip's share kept, over two access and one egress entry: 2 of 3 parts
        let acc = [(2u32, 1.0f32, 150.0f32, 0u32), (0, 2.0, 100.0, 1)];
        let eg = [(1u32, 1.0f32, 100.0f32, 2u32)];
        let kept = walk_kept(&acc, &eg, |_, _| 1.0, &sk.walk, s, 300.0);
        assert!((kept - if y < 300.0 { 2.0 / 3.0 } else { 1.0 }).abs() < 1e-6, "y {y}: {kept}");
    }
    // platforms: a change at station 3 walks from line 0's platform to line 1's
    let mut net = common_test_net(&[&[2, 0, 3], &[3, 1]], &[12.0, 12.0], &[2.0, 2.0]);
    net.st_x = vec![0.0, 0.0, -3000.0, 3000.0];
    net.st_y = vec![0.0, 2000.0, 0.0, 0.0];
    net.stop_x = net.line_stops.iter().map(|&st| net.st_x[st as usize]).collect();
    net.stop_y = vec![0.0, 0.0, -25.0, 25.0, 2000.0];
    let g = build_graph(&net, &p);
    let sk = skims(&g, &p);
    assert_eq!(sk.walk[1], 50, "0 to 1 changes across 50 m at station 3");
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
