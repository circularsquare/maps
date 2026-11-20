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
    // tram track: 100 m a level, 80 km/h
    let mut a = api();
    a.set_route_kind(3);
    let e = ends(&a, 0.0, 600.0, 0, 1);
    assert_eq!(a.add_route(&e, &[], 2, false), 0, "{}", a.issue_kinds());
    let (slow, ramps) = built(&a);
    assert_eq!(ramps, vec![100]);
    assert!((slow - 80.0).abs() < 1e-6);
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
