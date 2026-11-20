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
    // quad is two tram doubles (T-3166)
    assert_eq!(route(&mut a, (0.0, 0.0), (3000.0, 0.0), 2), 0, "{}", a.issues());
    let single = live(&a);
    assert_eq!(route(&mut a, (0.0, 2000.0), (3000.0, 2000.0), 4), 0, "{}", a.issues());
    let edges = live(&a);
    assert!(edges.len() >= single.len() + 2);
    assert!(edges.iter().all(|&e| a.w.net.edge_kind[e as usize] == TrackKind::Tram && a.w.net.edge_tracks[e as usize] == 2));
    assert!(edges.iter().all(|&e| a.edge_kind(e) == 3));
    // ground tram track is 0.5 x US$75M a km, as ground rail, and no road crossing is charged; quad is twice that
    assert!((a.blueprint_cost() - 3.0 * plain.blueprint_cost()).abs() < 0.1, "{} against {}", a.blueprint_cost(), plain.blueprint_cost());
    let bytes = a.save();
    let mut b = api();
    assert!(b.load(&bytes));
    assert!(live(&b).iter().all(|&e| b.w.net.edge_kind[e as usize] == TrackKind::Tram));
    assert_eq!(b.save(), bytes);
    // doubling up adds a tram pair (T-3166)
    let before = live(&a).len();
    let price = a.preview_double_up(single[0]);
    assert!(price > 0.0, "{}", a.issues());
    assert_eq!(a.double_up(single[0]), 0, "{}", a.issues());
    let now = live(&a);
    assert_eq!(now.len(), before + single.len());
    assert!(now.iter().all(|&e| a.w.net.edge_kind[e as usize] == TrackKind::Tram));
    assert_ne!(a.double_up(single[0]), 0);
    assert!(a.issue_kinds().contains("Alongside"), "{}", a.issue_kinds());
    let mut b = api();
    assert!(b.load(&a.save()));
    assert_eq!(live(&b).len(), now.len());
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
fn rail_and_tram_track_join_and_cross_flat() {
    let mut a = api();
    a.set_route_kind(3);
    let ends: Vec<f64> = free(0.0, 0.0).iter().chain(free(2000.0, 0.0).iter()).copied().collect();
    assert_eq!(a.add_route(&ends, &[], 2, false), 0);
    let tram_end = a.route_end();
    a.set_route_kind(0);
    // T-3889: rail from the tram's end node joins it there
    let from_end = vec![1.0, tram_end as f64, 0.0, 0.0, 0.0, 0.0, 0.0, 3000.0, 0.0, 0.0];
    assert_eq!(a.add_route(&from_end, &[], 2, false), 0, "{}", a.issues());
    let kinds: Vec<TrackKind> = a.w.net.node_ports[tram_end as usize].ends().iter().map(|&c| a.w.net.edge_kind[(c >> 1) as usize]).collect();
    assert!(kinds.contains(&TrackKind::Tram) && kinds.contains(&TrackKind::Regular), "{kinds:?}");
    assert_eq!(a.w.net.node_kind(tram_end), TrackKind::Regular);
    // no station where they meet: its kind would hang on which track came last
    assert_ne!(a.add_station(1, tram_end, 2000.0, 0.0, 80, "Joint".into()), 0);
    assert!(a.issue_kinds().contains("StationFamily"), "{}", a.issue_kinds());
    // rail across it crosses flat
    assert_eq!(route(&mut a, (1000.0, -1000.0), (1000.0, 1000.0), 2), 0, "{}", a.issues());
    assert!(a.w.net.crossings.iter().any(|c| matches!(c.kind, CrossKind::Flat(_))));
    assert!(!a.issue_kinds().contains("Crossing"));
}

fn station_at(a: &TrackApi, x: f64, y: f64) -> u32 {
    let net = &a.w.net;
    (0..net.node_count() as u32).find(|&n| net.node_ok(n) && net.node_platform[n as usize] > 0 && (net.node_x[n as usize] - x).abs() < 30.0 && (net.node_y[n as usize] - y).abs() < 30.0).unwrap()
}

/// T-3889: regular track from 0 to 5 km east with a rail station at 4 km, tram track drawn on from
/// its west end to -2 km with a tram stop at -1 km, and regular track on again to -4 km with a rail
/// station at -3 km: (west rail station, tram stop, east rail station).
fn tram_between_rail() -> (TrackApi, [u32; 3]) {
    let mut a = api();
    a.set_cash(1e6);
    assert_eq!(route(&mut a, (0.0, 0.0), (5000.0, 0.0), 2), 0, "{}", a.issues());
    let start = (0..a.w.net.node_count() as u32).find(|&n| a.w.net.node_ok(n) && a.w.net.node_x[n as usize] == 0.0).unwrap();
    a.set_route_kind(3);
    let ends = [1.0, start as f64, 0.0, 0.0, 0.0, 0.0, 0.0, -2000.0, 0.0, 0.0];
    assert_eq!(a.add_route(&ends, &[], 2, false), 0, "{}", a.issues());
    let west = a.route_end();
    a.set_route_kind(0);
    let ends = [1.0, west as f64, 0.0, 0.0, 0.0, 0.0, 0.0, -4000.0, 0.0, 0.0];
    assert_eq!(a.add_route(&ends, &[], 2, false), 0, "{}", a.issues());
    for (x, platform, name) in [(-3000.0, 200u16, "West"), (-1000.0, 40, "Stop"), (4000.0, 200, "East")] {
        assert_eq!(a.add_station(2, edge_at(&a, x, 0.0), x, 0.0, platform, name.into()), 0, "{name}: {}", a.issues());
    }
    let st = [station_at(&a, -3000.0, 0.0), station_at(&a, -1000.0, 0.0), station_at(&a, 4000.0, 0.0)];
    assert!(!a.w.net.tram_stop(st[0]) && a.w.net.tram_stop(st[1]) && !a.w.net.tram_stop(st[2]));
    (a, st)
}

/// T-3889: a tram line runs from a tram stop onto rail track to a rail station, at its 80 km/h
/// there, with tram cars from the tram fleet; a train line between the two rail stations, which only
/// tram track joins, is refused, saying why, until it stops at the tram stop (a tram line then).
#[test]
fn a_tram_line_runs_onto_rail_track_and_a_train_line_keeps_off_tram_track() {
    let (mut a, st) = tram_between_rail();
    let tram = a.new_line_id();
    assert_eq!(a.set_line_with_trains(tram, &st[1..], 6, 3, 1, 15.0, 120.0, 2, 0xff0000, "Tram-train".into()), 0, "{}", a.issues());
    assert!(a.line_tram(tram));
    let net = &a.w.net;
    assert!(net.line_path(tram).iter().any(|&p| net.edge_kind[(p >> 1) as usize] == TrackKind::Regular), "it runs on the rail");
    assert_eq!(a.construct_all(), 0, "{}", a.issues());
    let l = &a.w.svc.lines[tram as usize];
    assert!(l.running);
    let top = l.runs[0].limits.iter().map(|x| x[1]).fold(0.0, f64::max);
    assert!((top - 80.0 / 3.6).abs() < 1e-9, "top {} km/h", top * 3.6);
    assert!(a.trams_needed() > 0 && a.cars_needed() == 0);

    let train = a.new_line_id();
    assert_ne!(a.set_line_with_trains(train, &[st[0], st[2]], 2, 1, 0, 30.0, 180.0, 0, 0, "Train".into()), 0);
    assert!(a.issue_kinds().contains("TramTrack"), "{}", a.issue_kinds());
    // a one-stop line, then the far station clicked: refused the same way
    assert_eq!(a.set_line_with_trains(train, &st[..1], 2, 1, 0, 30.0, 180.0, 0, 0, "Train".into()), 0, "{}", a.issues());
    assert_ne!(a.add_stop(train, st[2]), 0);
    assert!(a.issue_kinds().contains("TramTrack"), "{}", a.issue_kinds());
    // the tram stop clicked: a tram line, with the tram's dwell and turnaround, that then takes the far station
    assert_eq!(a.add_stop(train, st[1]), 0, "{}", a.issues());
    assert!(a.line_tram(train));
    assert_eq!((a.w.net.line_dwell[train as usize], a.w.net.line_turn[train as usize]), (15.0, 120.0));
    assert_eq!(a.add_stop(train, st[2]), 0, "{}", a.issues());
    assert_eq!(a.line_stops(train), st);
    // and back: without its tram stop it would run trains over tram track
    assert_ne!(a.drop_stop(train, 1), 0);
    assert!(a.issue_kinds().contains("TramTrack"), "{}", a.issue_kinds());
    // a high-speed tram line is refused
    let mut d = a.w.net.line_data(train).unwrap();
    d.fast = true;
    assert!(a.w.net.set_line(train, Some(d)).is_err());

    // the joined network and its lines load the same
    let bytes = a.save();
    let mut b = api();
    assert!(b.load(&bytes));
    assert_eq!(b.save(), bytes);
    for l in [tram, train] {
        assert_eq!((b.line_tram(l), b.line_stops(l)), (a.line_tram(l), a.line_stops(l)));
        assert_eq!(b.w.net.line_path(l), a.w.net.line_path(l));
    }
}

/// T-3889: tram track branching off regular and high-speed track makes a junction, priced as rail's,
/// and one undo takes it out.
#[test]
fn tram_track_branches_off_regular_and_high_speed_track() {
    for hsr in [false, true] {
        let mut a = api();
        a.set_route_hsr(hsr);
        assert_eq!(route(&mut a, (0.0, 0.0), (4000.0, 0.0), 2), 0, "{}", a.issues());
        let main = edge_at(&a, 2000.0, 0.0);
        assert_eq!(a.w.net.edge_kind[main as usize], if hsr { TrackKind::HighSpeed } else { TrackKind::Regular });
        let before = (live(&a).len(), a.blueprint_cost());
        a.set_route_kind(3);
        let ends = [2.0, main as f64, 2000.0, 0.0, 0.0, 0.0, 0.0, 2600.0, 600.0, 0.0];
        assert_eq!(a.add_route(&ends, &[], 2, false), 0, "hsr {hsr}: {}", a.issues());
        let net = &a.w.net;
        let junction = (0..net.node_count() as u32).find(|&n| net.node_ok(n) && net.node_ports[n as usize].n == 3).expect("a junction");
        let kinds: Vec<TrackKind> = net.node_ports[junction as usize].ends().iter().map(|&c| net.edge_kind[(c >> 1) as usize]).collect();
        assert!(kinds.contains(&TrackKind::Tram), "{kinds:?}");
        assert_ne!(net.node_kind(junction), TrackKind::Tram);
        let rows = a.blueprint_piece_costs();
        assert!(rows.chunks(5).any(|r| r[0] == 2.0 && r[1] == junction as f64 && r[3] > 0.0), "the junction is priced");
        a.undo();
        assert_eq!((live(&a).len(), a.blueprint_cost()), before);
    }
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

/// A tram line across a train line at a flat crossing, built (the tram line's id, the train line's).
fn tram_over_train(tram_line: bool) -> (TrackApi, u32, u32) {
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
    let mut ids = [u32::MAX; 2];
    for (k, (stops, cars, name)) in [(&stops[..2], 2u8, "Tram"), (&stops[2..], 4, "Train")].into_iter().enumerate() {
        if k == 0 && !tram_line {
            continue;
        }
        let l = a.new_line_id();
        assert_eq!(a.set_line_with_trains(l, stops, 6, 3, 1, 20.0, 120.0, cars, 0x00ff00, name.into()), 0, "{}", a.issues());
        ids[k] = l;
    }
    assert_eq!(a.construct_all(), 0, "{}", a.issues());
    (a, ids[0], ids[1])
}

/// T-3032: at a flat crossing a tram yields. The train's capacity through it is what it would be
/// with no crossing, whatever the tram frequency; the tram's resource holds each tram pass at the
/// tram's 45 s and each train pass at the train's 120 s.
#[test]
fn a_tram_yields_to_a_train_at_a_flat_crossing() {
    use crate::track::capacity::ResKind;
    let (alone, _, train_alone) = tram_over_train(false);
    let limit = |a: &TrackApi, l: u32| a.w.svc.cap.line_limits(&a.w.net)[l as usize][0];
    let base = limit(&alone, train_alone);
    for trams in [(1, 1, 1), (6, 3, 1), (40, 20, 10)] {
        let (mut a, tram, train) = tram_over_train(true);
        assert_eq!(a.set_train_counts(tram, trams.0, trams.1, trams.2), 0, "{}", a.issues());
        let cap = &a.w.svc.cap;
        assert_eq!(limit(&a, train).0, base.0, "train capacity with {trams:?} trams");
        assert!(cap.wire_kind(limit(&a, train).1 as usize) != 4, "the crossing limits the train");
        let crossing: Vec<usize> = (0..cap.res.len()).filter(|&r| cap.res[r].kind == ResKind::Crossing).collect();
        assert!(!crossing.is_empty(), "no crossing resource");
        for r in crossing {
            let us = cap.users_of(r);
            assert!(us.iter().all(|u| u.line == tram || !u.owner), "a train owns a crossing a tram crosses");
            assert!(us.iter().filter(|u| u.owner).all(|u| u.occ == 45.0));
            assert!(us.iter().filter(|u| !u.owner).all(|u| u.occ == 120.0 && u.line == train));
            let tph = |l: u32| a.w.net.line_tph[l as usize][0] as f64;
            // one direction of the tram line: its trams at 45 s, the train line's passes at 120 s
            assert!((cap.rho[r][0] - (tph(tram) * 45.0 + 2.0 * tph(train) * 120.0) / 3600.0).abs() < 1e-9, "{}", cap.rho[r][0]);
        }
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

/// T-3352: single tram and single high-speed track are drawn single, priced at 0.6x their kind, saved and loaded.
#[test]
fn single_tram_and_high_speed_track_are_drawn_priced_and_saved() {
    for (kind, hsr) in [(3u8, false), (0, true)] {
        let mut dbl = api();
        dbl.set_route_kind(kind);
        dbl.set_route_hsr(hsr);
        assert_eq!(route(&mut dbl, (0.0, 0.0), (3000.0, 0.0), 2), 0, "{}", dbl.issues());
        let mut a = api();
        a.set_route_kind(kind);
        a.set_route_hsr(hsr);
        assert_eq!(route(&mut a, (0.0, 0.0), (3000.0, 0.0), 1), 0, "{}", a.issues());
        let edges = live(&a);
        assert!(edges.iter().all(|&e| a.w.net.edge_tracks[e as usize] == 1 && (a.w.net.edge_kind[e as usize].is_hsr() == hsr) && (a.w.net.edge_kind[e as usize].spec().family == (kind == 3) as u8)));
        assert!((a.blueprint_cost() - 0.6 * dbl.blueprint_cost()).abs() < 0.05, "{} against {}", a.blueprint_cost(), dbl.blueprint_cost());
        let bytes = a.save();
        let mut b = api();
        assert!(b.load(&bytes), "{}", b.issues());
        assert_eq!(b.save(), bytes);
        // a stop or station on it
        let e = edge_at(&a, 1500.0, 0.0);
        let platform = if kind == 3 { 40 } else { 200 };
        assert_eq!(a.add_station(2, e, 1500.0, 0.0, platform, "S".into()), 0, "{}", a.issues());
    }
}

/// T-3574: a tram stop's terminal is 1 km of its own track kind and its passing tracks the rail formula at that
/// kind's table; with the terminal a line turning there turns 30 an hour where it turned 15.
#[test]
fn a_tram_stop_takes_a_terminal_and_passing_tracks_priced_as_tram_track() {
    use crate::track::capacity::ResKind;
    use crate::track::upgrades;
    let (mut a, stops) = tram_line();
    let t = stops[1];
    assert!((a.station_upgrade_cost(t, upgrades::TERMINAL) - 37.5).abs() < 1e-9, "{}", a.station_upgrade_cost(t, upgrades::TERMINAL));
    assert!((a.station_upgrade_cost(t, upgrades::PASSING) - 75.0 * 0.5 * 0.44).abs() < 1e-9);
    assert_eq!(a.construct_all(), 0, "{}", a.issues());
    let id = a.new_line_id();
    assert_eq!(a.set_line_with_trains(id, &stops, 4, 2, 0, 15.0, 120.0, 2, 0, "L".into()), 0, "{}", a.issue_kinds());
    let turns = |a: &TrackApi| {
        let cap = &a.w.svc.cap;
        let r = (0..cap.res.len()).find(|&r| cap.res[r].node == t && cap.res[r].kind == ResKind::Terminus).unwrap();
        3600.0 / cap.users_of(r).iter().map(|u| u.occ).fold(f64::NAN, |_, o| o)
    };
    assert!((turns(&a) - 15.0).abs() < 1e-9, "{}", turns(&a));
    let cash = a.cash();
    assert_eq!(a.set_station_upgrade(&[t], upgrades::TERMINAL, true), 0, "{}", a.issues());
    assert!((cash - a.cash() - 37.5).abs() < 1e-6, "paid 1 km of tram track");
    assert!((turns(&a) - 30.0).abs() < 1e-9, "{}", turns(&a));
    let cash = a.cash();
    assert_eq!(a.set_station_upgrade(&[t], upgrades::PASSING, true), 0, "{}", a.issues());
    assert!((cash - a.cash() - 16.5).abs() < 1e-6);
    // rail's price is unchanged
    let mut rail = api();
    assert_eq!(route(&mut rail, (0.0, 0.0), (3000.0, 0.0), 2), 0);
    assert_eq!(rail.add_station(2, edge_at(&rail, 1000.0, 0.0), 1000.0, 0.0, 200, "R".into()), 0);
    let r = (0..rail.w.net.node_count() as u32).find(|&n| rail.w.net.node_platform[n as usize] > 0).unwrap();
    assert!((rail.station_upgrade_cost(r, upgrades::TERMINAL) - 37.5).abs() < 1e-9);
    assert!((rail.station_upgrade_cost(r, upgrades::PASSING) - 22.5).abs() < 1e-9);
}

/// T-3575: tram track takes crossovers as rail does: a tram crossover is tram track at the tram table's price (rail's
/// at ground level off street), and a tram line whose stops lie on the two pairs of a quad runs over it.
#[test]
fn tram_track_takes_crossovers_priced_as_tram_track() {
    let build = |kind: u8| {
        let mut a = api();
        a.set_cash(1e6);
        a.set_route_kind(kind);
        assert_eq!(route(&mut a, (0.0, 0.0), (3000.0, 0.0), 4), 0, "{}", a.issues());
        let mut st = [0u32; 2];
        for (k, (x, y)) in [(800.0, 0.0), (2200.0, 4.27)].into_iter().enumerate() {
            let e = edge_at(&a, x, y);
            assert_eq!(a.add_station(2, e, x, y, if kind == 3 { 40 } else { 200 }, format!("Q{k}")), 0, "{}", a.issues());
            st[k] = (0..a.w.net.node_count() as u32).find(|&n| a.w.net.node_ok(n) && a.w.net.node_platform[n as usize] > 0 && (k == 0 || n != st[0])).unwrap();
        }
        assert_eq!(a.construct_all(), 0, "{}", a.issues());
        let id = a.new_line_id();
        let e = edge_at(&a, 1500.0, 0.0);
        let preview = a.preview_crossover(e, 1500.0, 0.0);
        assert_eq!(preview[0], 1.0, "{}", a.issue_kinds());
        assert_eq!(a.add_crossover(e, 1500.0, 0.0), 0, "{}", a.issues());
        let xs = a.crossover_edges();
        assert_eq!(xs.len(), 2);
        assert!(xs.iter().all(|&x| a.w.net.edge_kind[x as usize].spec().family == (kind == 3) as u8));
        let price: f64 = xs.iter().map(|&x| a.w.net.edge_cost[x as usize]).sum();
        assert_eq!(a.construct_all(), 0, "{}", a.issues());
        assert_eq!(a.set_line_with_trains(id, &st, 2, 1, 0, 15.0, 120.0, 2, 0, "L".into()), 0, "{}", a.issue_kinds());
        assert!(a.w.net.line_path(id).iter().any(|&p| xs.contains(&(p >> 1))), "kind {kind}: path {:?} crossovers {xs:?} stops {st:?}", a.w.net.line_path(id));
        let bytes = a.save();
        let mut b = api();
        assert!(b.load(&bytes), "{}", b.issues());
        assert_eq!(b.crossover_edges().len(), 2);
        assert_eq!(b.save(), bytes);
        price
    };
    let (rail, tram) = (build(0), build(3));
    // off-street tram track is rail's price at ground level; street tram track is a fifth of it
    assert!(rail > 0.0 && (tram - rail).abs() < 1e-6, "{tram} against {rail}");
    use crate::track::cost::crossover_cost;
    assert!((crossover_cost(0, false, TrackKind::StreetTram) - 0.25 * 75.0 * 0.1).abs() < 1e-9);
    assert!((crossover_cost(0, false, TrackKind::Regular) - 0.25 * 37.5).abs() < 1e-9);
}
