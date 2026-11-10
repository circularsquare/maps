//! T-2926: tram track drawn by hand, its stops, its fleet and how it meets rail track.
use super::*;

fn api() -> TrackApi {
    TrackApi::new(-73.985, 40.758, true)
}
fn free(x: f64, y: f64) -> [f64; 5] {
    [0.0, 0.0, x, y, 0.0]
}
fn route(a: &mut TrackApi, from: (f64, f64), to: (f64, f64), tracks: u8) -> u32 {
    let ends: Vec<f64> = free(from.0, from.1).iter().chain(free(to.0, to.1).iter()).copied().collect();
    a.add_route(&ends, &[], tracks, false)
}
fn live(a: &TrackApi) -> Vec<u32> {
    (0..a.w.net.edge_count() as u32).filter(|&e| a.w.net.edge_ok(e)).collect()
}
fn edge_at(a: &TrackApi, x: f64, y: f64) -> u32 {
    let net = &a.w.net;
    live(a).into_iter().min_by(|&p, &q| net.project(p, x, y).1.total_cmp(&net.project(q, x, y).1)).unwrap()
}

/// A 3 km tram route with a 40 m stop at each end of its middle kilometre, and a line over them.
fn tram_line() -> (TrackApi, [u32; 2]) {
    let mut a = api();
    a.set_route_kind(3);
    assert_eq!(route(&mut a, (0.0, 0.0), (3000.0, 0.0), 2), 0, "{}", a.issues());
    let mut stops = [0u32; 2];
    for (k, x) in [1000.0, 2000.0].into_iter().enumerate() {
        let e = edge_at(&a, x, 0.0);
        assert_eq!(a.add_station(2, e, x, 0.0, 40, format!("T{k}")), 0, "{}", a.issues());
        let net = &a.w.net;
        stops[k] = (0..net.node_count() as u32).find(|&n| net.node_ok(n) && net.node_platform[n as usize] > 0 && (k == 0 || n != stops[0])).unwrap();
    }
    (a, stops)
}

#[test]
fn tram_track_is_drawn_double_priced_by_its_table_and_saved() {
    let mut plain = api();
    assert_eq!(route(&mut plain, (0.0, 0.0), (3000.0, 0.0), 2), 0);
    let mut a = api();
    a.set_route_kind(3);
    // single and quad are asked for and come out as double tram track
    for (y, tracks) in [(0.0, 1u8), (2000.0, 4)] {
        assert_eq!(route(&mut a, (0.0, y), (3000.0, y), tracks), 0, "{}", a.issues());
    }
    let edges = live(&a);
    assert!(!edges.is_empty());
    assert!(edges.iter().all(|&e| a.w.net.edge_kind[e as usize] == TrackKind::Tram && a.w.net.edge_tracks[e as usize] == 2));
    assert!(edges.iter().all(|&e| a.edge_kind(e) == 3));
    // ground tram track is 0.5 x US$75M a km, as ground rail, and no road crossing is charged
    assert!((a.blueprint_cost() - 2.0 * plain.blueprint_cost()).abs() < 1e-6, "{} against {}", a.blueprint_cost(), plain.blueprint_cost());
    let bytes = a.save();
    let mut b = api();
    assert!(b.load(&bytes));
    assert!(live(&b).iter().all(|&e| b.w.net.edge_kind[e as usize] == TrackKind::Tram));
    assert_eq!(b.save(), bytes);
    // no doubling up or reshaping into other kinds
    assert_ne!(a.double_up(edges[0]), 0);
    // rail drawn after it is rail again
    a.set_route_kind(0);
    assert_eq!(route(&mut a, (0.0, 4000.0), (3000.0, 4000.0), 2), 0);
    assert!(live(&a).iter().any(|&e| a.w.net.edge_kind[e as usize] == TrackKind::Regular));
}

#[test]
fn tram_stops_are_short_and_flat_priced() {
    let mut a = api();
    a.set_route_kind(3);
    assert_eq!(route(&mut a, (0.0, 0.0), (3000.0, 0.0), 2), 0);
    let e = edge_at(&a, 1000.0, 0.0);
    assert_ne!(a.add_station(2, e, 1000.0, 0.0, 200, "Long".into()), 0, "a 200 m platform is rail's");
    assert!(a.issue_kinds().contains("PlatformLength"), "{}", a.issue_kinds());
    let before = a.blueprint_cost();
    assert_eq!(a.add_station(2, edge_at(&a, 1000.0, 0.0), 1000.0, 0.0, 80, "Stop".into()), 0, "{}", a.issues());
    // US$2M at ground, whatever the platform
    assert!((a.blueprint_cost() - before - 2.0).abs() < 1e-6, "{}", a.blueprint_cost() - before);
    assert_ne!(a.add_station(2, edge_at(&a, 2000.0, 0.0), 2000.0, 0.0, 100, "Too long".into()), 0);
}

#[test]
fn rail_and_tram_track_do_not_join_but_cross_flat() {
    let mut a = api();
    a.set_route_kind(3);
    let ends: Vec<f64> = free(0.0, 0.0).iter().chain(free(2000.0, 0.0).iter()).copied().collect();
    assert_eq!(a.add_route(&ends, &[], 2, false), 0);
    let tram_end = a.route_end();
    a.set_route_kind(0);
    // rail from the tram's end node is refused
    let from_end = vec![1.0, tram_end as f64, 0.0, 0.0, 0.0, 0.0, 0.0, 3000.0, 0.0, 0.0];
    assert_ne!(a.add_route(&from_end, &[], 2, false), 0);
    assert!(a.issue_kinds().contains("TrackFamily"), "{}", a.issue_kinds());
    // rail across it crosses flat
    assert_eq!(route(&mut a, (1000.0, -1000.0), (1000.0, 1000.0), 2), 0, "{}", a.issues());
    assert!(a.w.net.crossings.iter().any(|c| matches!(c.kind, CrossKind::Flat(_))));
    assert!(!a.issue_kinds().contains("Crossing"));
}

#[test]
fn a_tram_line_has_its_own_fleet_and_defaults() {
    let d = TrackApi::kind_defaults(3);
    assert_eq!((d[0], d[1], d[2], d[3], d[4]), (15.0, 120.0, 20.0, 80.0, 2.0));
    assert_eq!(TrackApi::kind_defaults(0)[..2], [30.0, 180.0]);
    let (mut a, stops) = tram_line();
    let l = a.new_line_id();
    assert_eq!(a.set_line_with_trains(l, &stops, 3, 2, 1, 15.0, 120.0, 2, 0xff0000, "Tram".into()), 0, "{}", a.issues());
    assert!(a.line_tram(l));
    assert_eq!(a.construct_all(), 0, "{}", a.issues());
    assert_eq!((a.trams_needed(), a.cars_needed(), a.buses_needed()), (6, 0, 0));
    // a high-speed tram line is refused
    let mut d = a.w.net.line_data(l).unwrap();
    d.fast = true;
    assert!(a.w.net.set_line(l, Some(d)).is_err());
}

/// The flat crossing of a train line and a tram line is a resource both lines use, held for the
/// train's 90 s.
#[test]
fn a_tram_crossing_a_train_holds_both_for_the_train_figure() {
    let mut a = api();
    a.set_route_kind(3);
    assert_eq!(route(&mut a, (0.0, 0.0), (2000.0, 0.0), 2), 0);
    a.set_route_kind(0);
    assert_eq!(route(&mut a, (1000.0, -1000.0), (1000.0, 1000.0), 2), 0, "{}", a.issues());
    let mut stops = vec![];
    for (x, y, platform) in [(300.0, 0.0, 40u16), (1700.0, 0.0, 40), (1000.0, -600.0, 100), (1000.0, 600.0, 100)] {
        let e = edge_at(&a, x, y);
        assert_eq!(a.add_station(2, e, x, y, platform, format!("S{x}{y}")), 0, "{}", a.issues());
        let net = &a.w.net;
        stops.push((0..net.node_count() as u32).find(|&n| net.node_ok(n) && net.node_platform[n as usize] > 0 && (net.node_x[n as usize] - x).abs() < 30.0 && (net.node_y[n as usize] - y).abs() < 30.0).unwrap());
    }
    for (stops, cars, name) in [(&stops[..2], 2u8, "Tram"), (&stops[2..], 4, "Train")] {
        let l = a.new_line_id();
        assert_eq!(a.set_line_with_trains(l, stops, 6, 3, 1, 20.0, 120.0, cars, 0x00ff00, name.into()), 0, "{}", a.issues());
    }
    assert_eq!(a.construct_all(), 0, "{}", a.issues());
    let cap = &a.w.svc.cap;
    let crossing: Vec<usize> = (0..cap.res.len()).filter(|&r| cap.res[r].kind == crate::track::capacity::ResKind::Crossing).collect();
    assert!(!crossing.is_empty(), "no crossing resource");
    for r in crossing {
        assert!(cap.users_of(r).iter().all(|u| u.occ == 90.0), "{:?}", cap.users_of(r).iter().map(|u| u.occ).collect::<Vec<_>>());
        assert!(cap.users_of(r).len() >= 2);
    }
}

fn route_pis(a: &mut TrackApi, from: (f64, f64), pis: &[f64], to: (f64, f64)) -> u32 {
    let ends: Vec<f64> = free(from.0, from.1).iter().chain(free(to.0, to.1).iter()).copied().collect();
    a.add_route(&ends, pis, 2, false)
}

/// T-2927, T-2997: tram kinds take curves down to 18 m; rail keeps its 50.
#[test]
fn a_20_m_curve_is_tram_track_only() {
    let corner = [500.0, 0.0, 20.0, 0.0];
    let mut a = api();
    assert_ne!(route_pis(&mut a, (0.0, 0.0), &corner, (500.0, 500.0)), 0);
    assert!(a.issue_kinds().contains("RadiusTooSmall"), "{}", a.issue_kinds());
    for kind in [2u8, 3] {
        let mut a = api();
        a.set_route_kind(kind);
        assert_eq!(route_pis(&mut a, (0.0, 0.0), &corner, (500.0, 500.0)), 0, "kind {kind}: {}", a.issues());
        let mut a = api();
        a.set_route_kind(kind);
        assert_ne!(route_pis(&mut a, (0.0, 0.0), &[500.0, 0.0, 15.0, 0.0], (500.0, 500.0)), 0, "15 m is under the tram's 18");
    }
}

/// T-2927: drawn on streets, a corner whose curve leaves the street's line by more than
/// `OFF_STREET_M` is a piece of tram track of its own, priced as one; a gentle bend stays street.
#[test]
fn street_track_turning_a_corner_is_tram_track_there() {
    let mut a = api();
    a.set_route_kind(2);
    // a right angle at 30 m (its middle 8.8 m off the legs), then a 20 degree bend at 100 m (1.5 m)
    let (bx, by) = (500.0 + 500.0 * (70f64).to_radians().cos(), 500.0 + 500.0 * (70f64).to_radians().sin());
    let pis = [500.0, 0.0, 30.0, 0.0, 500.0, 500.0, 100.0, 0.0];
    assert_eq!(route_pis(&mut a, (0.0, 0.0), &pis, (bx, by)), 0, "{}", a.issues());
    let net = &a.w.net;
    let edges = live(&a);
    let own: Vec<u32> = edges.iter().copied().filter(|&e| net.edge_kind[e as usize] == TrackKind::Tram).collect();
    assert_eq!(own.len(), 1, "{:?}", edges.iter().map(|&e| (net.edge_kind[e as usize], net.edge_len[e as usize])).collect::<Vec<_>>());
    // the corner's arc (47 m) and a metre of straight each side
    let arc = 30.0 * std::f64::consts::FRAC_PI_2;
    assert!((net.edge_len[own[0] as usize] - arc - 2.0).abs() < 0.05, "{}", net.edge_len[own[0] as usize]);
    assert!(edges.iter().filter(|&&e| net.edge_kind[e as usize] == TrackKind::StreetTram).count() >= 2);
    // street track at 0.1, tram track at 0.5 of US$75M a km
    let (st, tr): (f64, f64) = edges.iter().fold((0.0, 0.0), |(s, t), &e| if net.edge_kind[e as usize] == TrackKind::Tram { (s, t + net.edge_len[e as usize]) } else { (s + net.edge_len[e as usize], t) });
    let want = 75.0 * (0.1 * st + 0.5 * tr) / 1000.0;
    assert!((a.blueprint_cost() - want).abs() < 1e-6, "{} against {want}", a.blueprint_cost());
    // shaped by hand, a street piece is tram track
    let e = edges.iter().copied().find(|&e| net.edge_kind[e as usize] == TrackKind::StreetTram).unwrap();
    let (ea, eb) = (net.edge_a[e as usize], net.edge_b[e as usize]);
    let (mx, my) = ((net.node_x[ea as usize] + net.node_x[eb as usize]) / 2.0, (net.node_y[ea as usize] + net.node_y[eb as usize]) / 2.0);
    assert_eq!(a.set_edge_pis(e, &[mx, my, 0.0, 0.0]), 0, "{}", a.issues());
    assert_eq!(a.w.net.edge_kind[e as usize], TrackKind::Tram);
}

/// T-2927: two corners too close for a street piece between are one piece of tram track.
#[test]
fn close_corners_leave_the_street_together() {
    let mut a = api();
    a.set_route_kind(2);
    // a jog: right, then left 40 m on
    let pis = [500.0, 0.0, 25.5, 0.0, 500.0, 60.0, 25.5, 0.0];
    assert_eq!(route_pis(&mut a, (0.0, 0.0), &pis, (1000.0, 60.0)), 0, "{}", a.issues());
    let net = &a.w.net;
    let kinds: Vec<(TrackKind, f64)> = live(&a).iter().map(|&e| (net.edge_kind[e as usize], net.edge_len[e as usize])).collect();
    assert_eq!(kinds.iter().filter(|k| k.0 == TrackKind::Tram).count(), 1, "{kinds:?}");
    assert_eq!(kinds.iter().filter(|k| k.0 == TrackKind::StreetTram).count(), 2, "{kinds:?}");
}

/// T-2927: with the street it was fitted to, a straight that cuts a corner of the street by more than
/// `STREET_HALF_M` is tram track; one along the street stays street track.
#[test]
fn a_straight_that_cuts_the_street_is_tram_track() {
    let kinds = |street: &[(f64, f64)]| {
        let mut a = api();
        let ll: Vec<f64> = street.iter().flat_map(|&(x, y)| a.to_lonlat(x, y)).collect();
        a.set_route_kind(2);
        a.set_route_street(&ll);
        assert_eq!(route_pis(&mut a, (0.0, 0.0), &[], (1000.0, 0.0)), 0, "{}", a.issues());
        live(&a).iter().map(|&e| a.w.net.edge_kind[e as usize]).collect::<Vec<_>>()
    };
    assert_eq!(kinds(&[(0.0, 0.0), (1000.0, 0.0)]), vec![TrackKind::StreetTram]);
    assert_eq!(kinds(&[(0.0, 0.0), (500.0, 6.0), (1000.0, 0.0)]), vec![TrackKind::StreetTram]);
    assert_eq!(kinds(&[(0.0, 0.0), (500.0, 40.0), (1000.0, 0.0)]), vec![TrackKind::Tram]);
}

/// T-3004: tram track pays a quarter of the terrain surcharge, street and off-street; water is half on
/// streets (T-2928) and whole off them.
#[test]
fn tram_track_pays_a_quarter_of_the_hills() {
    use crate::track::cost::{terrain_surcharge, track_cost_t};
    let vert = [[0.0, 0.0], [1000.0, 0.0]];
    let (hills, water) = ([[0.0, 1000.0, 3.0]], [[0.0, 1000.0]]);
    for (k, t, w) in [(TrackKind::StreetTram, 0.25, 0.5), (TrackKind::Tram, 0.25, 1.0), (TrackKind::Regular, 1.0, 1.0)] {
        let flat = track_cost_t(1000.0, &vert, &[], &[], 2, k);
        let hill = track_cost_t(1000.0, &vert, &[], &hills, 2, k);
        let wet = track_cost_t(1000.0, &vert, &water, &[], 2, k);
        assert!((hill - flat - BASE_COST_PER_KM * terrain_surcharge(0, 3) * t).abs() < 1e-6, "{k:?} hills {}", hill - flat);
        assert!((wet - flat - BASE_COST_PER_KM * WATER_SURCHARGE * w).abs() < 1e-6, "{k:?} water {}", wet - flat);
    }
}

/// T-2997: chained on from the end of street track, a street route turning right at that end (the
/// next corner 40 m on) turns at the smallest radius, dropping the corner, instead of being refused;
/// a route back the way it came is still refused.
#[test]
fn a_street_route_turning_at_a_chained_end_rounds_the_turn() {
    let mut a = api();
    a.set_route_kind(2);
    assert_eq!(route_pis(&mut a, (0.0, -500.0), &[], (0.0, 0.0)), 0, "{}", a.issues());
    let end = a.route_end();
    let from_node = |x: f64, y: f64| -> Vec<f64> { [1.0, end as f64, 0.0, 0.0, 0.0].iter().chain(free(x, y).iter()).copied().collect() };
    // east along the street from the end, a jog north 40 m on, then east again
    let pis = [40.0, 0.0, 0.0, 0.0, 60.0, 30.0, 0.0, 0.0];
    assert_eq!(a.add_route(&from_node(600.0, 30.0), &pis, 2, false), 0, "{}", a.issues());
    let net = &a.w.net;
    assert!(live(&a).iter().any(|&e| net.edge_kind[e as usize] == TrackKind::Tram), "the turn leaves the street");
    let mut b = api();
    b.set_route_kind(2);
    assert_eq!(route_pis(&mut b, (0.0, -500.0), &[], (0.0, 0.0)), 0);
    let end = b.route_end();
    let back: Vec<f64> = [1.0, end as f64, 0.0, 0.0, 0.0].iter().chain(free(0.0, -300.0).iter()).copied().collect();
    assert_ne!(b.add_route(&back, &[0.0, -150.0, 0.0, 0.0], 2, false), 0);
}

/// T-2927: a street's staircase traced for street tram track: curves of 18 m or more, close to the
/// street, and placed as track.
#[test]
fn a_street_staircase_traces_for_street_tram_track() {
    // the legacy frame's (lon, lat) for plane points
    let mut a = api();
    let mut ll = vec![];
    let mut push = |x: f64, y: f64| ll.extend(a.to_lonlat(x, y));
    for k in 0..6 {
        let (x, y) = (k as f64 * 200.0, k as f64 * 150.0);
        push(x, y);
        push(x + 200.0, y);
    }
    push(1200.0, 900.0);
    let mut rt = a.trace_kind(&ll, 10.0, 2, 0);
    while !rt.step() {}
    let t = rt.t.take().expect("traced");
    assert!(t.worst <= 20.0, "worst {}", t.worst);
    let mut v = vec![Pi::new(0.0, 0.0, 0.0, 0)];
    v.extend(t.pis.iter().cloned());
    let (ex, ey) = (1200.0, 900.0);
    v.push(Pi::new(ex, ey, 0.0, 0));
    let f = geom::fit_min(&v, false, R_CAP, TRAM_MIN_RADIUS);
    assert!(f.issues.is_empty(), "{:?}", f.issues);
    assert!(f.radius.iter().all(|&r| r == 0.0 || r >= TRAM_MIN_RADIUS - 1e-6));
    a.set_route_kind(2);
    let flat: Vec<f64> = t.pis.iter().flat_map(|p| [p.x, p.y, p.radius, p.level as f64]).collect();
    assert_eq!(route_pis(&mut a, (0.0, 0.0), &flat, (ex, ey)), 0, "{}", a.issues());
    assert!(live(&a).iter().any(|&e| a.w.net.edge_kind[e as usize] == TrackKind::Tram));
}
