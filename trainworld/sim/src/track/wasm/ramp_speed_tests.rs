//! T-3027: drawn ramps are sized to the target speed where there is room, shorter and slower where
//! there is not, and a piece's ramp length is kept through edits and saves.

use super::*;

fn api() -> TrackApi {
    TrackApi::new(-73.985, 40.758, true)
}
fn base(a: &TrackApi) -> (f64, f64) {
    let p = a.from_lonlat(-73.985, 40.758);
    (p[0], p[1])
}
/// Open ground at level `l0` to open ground at `l1`, along x.
fn ends(a: &TrackApi, from: f64, to: f64, l0: i8, l1: i8) -> [f64; 10] {
    let (bx, by) = base(a);
    [0.0, 0.0, bx + from, by, l0 as f64, 0.0, 0.0, bx + to, by, l1 as f64]
}
fn alive(a: &TrackApi) -> Vec<u32> {
    (0..a.w.net.edge_count() as u32).filter(|&e| a.w.net.edge_ok(e)).collect()
}
/// The slowest limit over the built track, km/h, and each piece's stored ramp per level.
fn built(a: &TrackApi) -> (f64, Vec<u16>) {
    let es = alive(a);
    let slow = es.iter().flat_map(|&e| a.w.net.edge_limits(e, a.w.net.v_cap(e))).map(|l| l.v * 3.6).fold(f64::INFINITY, f64::min);
    (slow, es.iter().map(|&e| a.w.net.edge_ramp[e as usize]).collect())
}
fn slowest_preview(a: &TrackApi) -> f64 {
    a.preview_render_speeds().iter().copied().filter(|&v| v >= 0.0).fold(f32::INFINITY, f32::min) as f64
}

#[test]
fn high_speed_track_at_320_gets_400_m_ramps_and_keeps_its_speed() {
    let mut a = api();
    a.set_design_speed(320.0);
    a.set_route_hsr(true);
    let e = ends(&a, 0.0, 2000.0, 0, 1);
    assert_eq!(a.preview_route(&e, &[], 2, false)[0], 1.0, "{}", a.issue_kinds());
    assert!((slowest_preview(&a) - 320.0).abs() < 0.5, "{}", slowest_preview(&a));
    assert_eq!(a.add_route(&e, &[], 2, false), 0, "{}", a.issue_kinds());
    let (slow, ramps) = built(&a);
    assert_eq!(ramps, vec![400]);
    assert!((slow - 320.0).abs() < 1e-6, "{slow}");
    let e0 = alive(&a)[0];
    let spans: Vec<(f64, f64, f64)> = geom::ramp_spans(a.w.net.edge_vert(e0)).collect();
    assert_eq!(spans.len(), 1);
    assert!((spans[0].1 - spans[0].0 - 400.0).abs() < 1e-6, "{spans:?}");
}

#[test]
fn a_short_ramp_is_steeper_and_slower_down_to_the_floor() {
    // 250 m of room on high-speed track at 320: the ramp takes all of it, 200 km/h
    let mut a = api();
    a.set_design_speed(320.0);
    a.set_route_hsr(true);
    let e = ends(&a, 0.0, 250.0, 0, 1);
    assert_eq!(a.preview_route(&e, &[], 2, false)[0], 1.0, "{}", a.issue_kinds());
    assert!((slowest_preview(&a) - 200.0).abs() < 0.5, "{}", slowest_preview(&a));
    assert_eq!(a.add_route(&e, &[], 2, false), 0, "{}", a.issue_kinds());
    assert!((built(&a).0 - 200.0).abs() < 1e-6);
    // 130 m is under rail's 160 m floor: refused as before
    let mut a = api();
    a.set_design_speed(320.0);
    a.set_route_hsr(true);
    let e = ends(&a, 0.0, 130.0, 0, 1);
    assert_eq!(a.preview_route(&e, &[], 2, false)[0], 0.0);
    assert!(a.issue_kinds().contains("RampTooShort"), "{}", a.issue_kinds());
    // 170 m on ordinary track: 136 km/h, where 200 m ramps were refused
    let mut a = api();
    let e = ends(&a, 0.0, 170.0, 0, 1);
    assert_eq!(a.add_route(&e, &[], 2, false), 0, "{}", a.issue_kinds());
    assert!((built(&a).0 - 136.0).abs() < 1e-6, "{}", built(&a).0);
    // tram track: 100 m a level, 80 km/h by the ramp rule, held to the 75 cap
    let mut a = api();
    a.set_route_kind(3);
    let e = ends(&a, 0.0, 600.0, 0, 1);
    assert_eq!(a.add_route(&e, &[], 2, false), 0, "{}", a.issue_kinds());
    let (slow, ramps) = built(&a);
    assert_eq!(ramps, vec![100]);
    assert!((slow - 75.0).abs() < 1e-6);
}

#[test]
fn old_track_keeps_its_ramps_and_their_speed_through_a_save() {
    let mut a = api();
    a.set_route_hsr(true);
    a.set_design_speed(320.0);
    let e = ends(&a, 0.0, 2000.0, 0, 1);
    assert_eq!(a.add_route(&e, &[], 2, false), 0, "{}", a.issue_kinds());
    let e0 = alive(&a)[0];
    // a piece from before T-3027 (ramp 0): 200 m, no limit on high-speed track
    let d = a.w.net.edge_data(e0).unwrap();
    a.w.net.set_edge(e0, Some(EdgeData { ramp: 0, ..d.clone() })).unwrap();
    a.w.reload();
    let spans: Vec<(f64, f64, f64)> = geom::ramp_spans(a.w.net.edge_vert(e0)).collect();
    assert!((spans[0].1 - spans[0].0 - 200.0).abs() < 1e-6, "{spans:?}");
    assert!((built(&a).0 - 320.0).abs() < 1e-6);
    // and the drawn one comes back with its 400 m through a save
    a.w.net.set_edge(e0, Some(d)).unwrap();
    a.w.reload();
    let bytes = a.save();
    assert!(bytes.windows(4).any(|w| w == b"LRA1"));
    let mut b = api();
    assert!(b.load(&bytes));
    assert_eq!(TrackApi::save_unread(&bytes), 0);
    assert_eq!(b.w.net.edge_ramp[e0 as usize], 400);
    let spans: Vec<(f64, f64, f64)> = geom::ramp_spans(b.w.net.edge_vert(e0)).collect();
    assert!((spans[0].1 - spans[0].0 - 400.0).abs() < 1e-6, "{spans:?}");
    assert_eq!(b.save(), bytes);
    // ordinary track's 200 m ramps store nothing, so its saves are as they were
    let mut c = api();
    assert_eq!(c.add_route(&ends(&c, 0.0, 2000.0, 0, 1), &[], 2, false), 0, "{}", c.issue_kinds());
    assert!(!c.save().windows(4).any(|w| w == b"LRA1"));
}

/// T-3883: the ends of a bridge's ramps and of a ramp over track lie on the millimetre grid, a hair off
/// the straight; that is no curve, so a 400 m ramp on a slanted route keeps 320 for the track and the
/// trains (each end had held them at 160 for a train's length).
#[test]
fn ramp_ends_laid_by_bridges_and_crossings_hold_no_train() {
    let ang: f64 = 0.27;
    let at = |a: &TrackApi, d: f64| {
        let (bx, by) = base(a);
        (bx + d * ang.cos(), by + d * ang.sin())
    };
    let route = |a: &TrackApi, d0: f64, d1: f64| {
        let (p, q) = (at(a, d0), at(a, d1));
        [0.0, 0.0, p.0, p.1, 0.0, 0.0, 0.0, q.0, q.1, 0.0]
    };
    // over water: a bridge
    let mut a = api();
    let bx = base(&a).0;
    a.w.net.set_water_mask(Box::new(move |x: f64, _y: f64| x > bx + 3000.0 && x < bx + 3200.0));
    a.w.reload();
    a.set_design_speed(320.0);
    a.set_route_hsr(true);
    let e = route(&a, 0.0, 8000.0);
    assert_eq!(a.add_route(&e, &[], 2, true), 0, "{}", a.issue_kinds());
    assert_eq!(geom::ramp_spans(a.w.net.edge_vert(alive(&a)[0])).count(), 2);
    assert!((built(&a).0 - 320.0).abs() < 1e-6, "bridge: {}", built(&a).0);
    // its line's trains are held by nothing under 320 between the stations
    a.set_cash(1e6);
    let mut st = [0; 2];
    for (i, d) in [1000.0, 7000.0].into_iter().enumerate() {
        let (x, y) = at(&a, d);
        let on = alive(&a).into_iter().min_by(|&p, &q| a.w.net.project(p, x, y).1.total_cmp(&a.w.net.project(q, x, y).1)).unwrap();
        assert_eq!(a.add_station(2, on, x, y, 200, format!("S{i}")), 0, "{}", a.issue_kinds());
        let n = &a.w.net;
        st[i] = (0..n.node_count() as u32).filter(|&k| n.node_ok(k) && n.node_platform[k as usize] > 0).min_by(|&p, &q| {
            let d = |k: u32| (n.node_x[k as usize] - x).hypot(n.node_y[k as usize] - y);
            d(p).total_cmp(&d(q))
        }).unwrap();
    }
    assert_eq!(a.construct_all(), 0, "{}", a.issue_kinds());
    let line = a.new_line_id();
    assert_eq!(a.set_line_with_trains(line, &st, 2, 1, 0, 30.0, 180.0, 0, 0, "Test".into()), 0, "{}", a.issue_kinds());
    assert_eq!(a.set_line_fast(line, true), 0, "{}", a.issue_kinds());
    let lims = &a.w.svc.lines[line as usize].runs[0].limits;
    assert!(lims.iter().all(|l| l[1] * 3.6 > 319.0), "{lims:?}");
    // over track: a ramp up and down across it
    let mut a = api();
    let (c, s) = (at(&a, 3000.0), (ang.sin(), ang.cos()));
    let cross = [0.0, 0.0, c.0 - 1500.0 * s.0, c.1 + 1500.0 * s.1, 0.0, 0.0, 0.0, c.0 + 1500.0 * s.0, c.1 - 1500.0 * s.1, 0.0];
    assert_eq!(a.add_route(&cross, &[], 2, false), 0, "{}", a.issue_kinds());
    a.set_design_speed(320.0);
    a.set_route_hsr(true);
    a.set_route_ramp(true);
    assert_eq!(a.add_route(&route(&a, 0.0, 6000.0), &[], 2, true), 0, "{}", a.issue_kinds());
    let hsr: Vec<u32> = alive(&a).into_iter().filter(|&e| a.w.net.edge_kind[e as usize].is_hsr()).collect();
    assert_eq!(hsr.iter().map(|&e| geom::ramp_spans(a.w.net.edge_vert(e)).count()).sum::<usize>(), 2);
    let slow = hsr.iter().flat_map(|&e| a.w.net.edge_limits(e, a.w.net.v_cap(e))).map(|l| l.v * 3.6).fold(f64::INFINITY, f64::min);
    assert!((slow - 320.0).abs() < 1e-6, "over track: {slow}");
}

#[test]
fn a_split_keeps_the_ramp_where_it_was() {
    let mut a = api();
    a.set_route_hsr(true);
    a.set_design_speed(320.0);
    let e = ends(&a, 0.0, 2000.0, 0, 1);
    assert_eq!(a.add_route(&e, &[], 2, false), 0, "{}", a.issue_kinds());
    let e0 = alive(&a)[0];
    let (bx, by) = base(&a);
    assert_eq!(a.add_point(e0, bx + 1900.0, by), 0, "{}", a.issue_kinds());
    let es = alive(&a);
    assert_eq!(es.len(), 2);
    let spans: Vec<(f64, f64, f64)> = es.iter().flat_map(|&e| geom::ramp_spans(a.w.net.edge_vert(e)).collect::<Vec<_>>()).collect();
    assert_eq!(spans.len(), 1);
    assert!((spans[0].1 - spans[0].0 - 400.0).abs() < 0.05, "{spans:?}");
    assert!(es.iter().all(|&e| a.w.net.edge_ramp[e as usize] == 400));
    assert!((built(&a).0 - 320.0).abs() < 0.1, "{}", built(&a).0);
}
