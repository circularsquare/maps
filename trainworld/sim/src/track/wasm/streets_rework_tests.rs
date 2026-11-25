//! T-4155: street speeds landing time again only the bus lines whose route meets them, to the same figures a
//! rework of every bus line gives.

use super::*;
use crate::track::bus;

fn api() -> TrackApi {
    TrackApi::new(-73.985, 40.758, true)
}

/// A bus line along x from `x` for 6 km, buses every 10 minutes.
fn bus(a: &mut TrackApi, x: f64) -> u32 {
    let id = a.new_line_id();
    let pts = [x, 0.0, bus::K_CORNER as f64, x + 6000.0, 0.0, bus::K_CORNER as f64];
    assert_eq!(a.set_bus_line(id, &pts, 600, 600, 600, 20.0, 60.0, 0, "B".into()), 0, "{}", a.issues());
    id
}

/// One pack's street speeds about (`lon`, `lat`): a dense square, then open ones, 8 km along x.
fn grid(a: &mut TrackApi, lon: f64, lat: f64) -> i32 {
    let (pop, jobs) = ([40_000.0f32, 10.0, 10.0, 10.0], [40_000.0f32, 10.0, 10.0, 10.0]);
    a.add_pack_streets(2000.0, -1000.0, -1000.0, 4, 1, &pop, &jobs, &[0, 1, 2, 3], &[0, 1, 2, 3], &[0, 0, 0, 0], &[], &[1, 1, 1, 1], lon, lat)
}

fn times(a: &TrackApi, lines: &[u32]) -> Vec<(Vec<f64>, Vec<u32>)> {
    lines.iter().map(|&l| {
        let s = &a.w.svc.lines[l as usize];
        (s.round_trip.to_vec(), s.trains.to_vec())
    }).collect()
}

#[test]
fn streets_landing_time_again_only_the_bus_lines_they_meet() {
    let lon_far = |a: &TrackApi, x: f64| a.w.net.frame().to_lonlat(x, 0.0).0;
    let mut a = api();
    let near = bus(&mut a, 0.0);
    let far = bus(&mut a, 200_000.0);
    let lines = [near, far];
    // the first speeds: every bus samples streets from then on
    let (lon_nowhere, _) = a.w.net.frame().to_lonlat(-600_000.0, 0.0);
    assert_eq!(grid(&mut a, lon_nowhere, 40.758), 2);
    // speeds over the near line only
    let mut whole = api();
    let (n2, f2) = (bus(&mut whole, 0.0), bus(&mut whole, 200_000.0));
    assert_eq!(grid(&mut whole, lon_nowhere, 40.758), 2);
    let before = times(&a, &lines);
    assert_eq!(grid(&mut a, -73.985, 40.758), 1, "only the near line");
    assert_ne!(times(&a, &lines)[0], before[0], "the near line runs at the new speeds");
    assert_eq!(times(&a, &lines)[1], before[1]);
    grid(&mut whole, -73.985, 40.758);
    whole.w.rework_buses(None);
    assert_eq!(times(&a, &lines), times(&whole, &[n2, f2]), "the same figures as timing every bus line again");
    // speeds over neither: nothing timed again, nothing changed
    let after = times(&a, &lines);
    let lon_mid = lon_far(&a, 100_000.0);
    assert_eq!(grid(&mut a, lon_mid, 40.758), 0);
    assert_eq!(times(&a, &lines), after);
    // and over the far one
    let lon_f = lon_far(&a, 201_000.0);
    assert_eq!(grid(&mut a, lon_f, 40.758), 1);
    grid(&mut whole, lon_mid, 40.758);
    grid(&mut whole, lon_f, 40.758);
    whole.w.rework_buses(None);
    assert_eq!(times(&a, &lines), times(&whole, &[n2, f2]));
}

/// T-4193: a save's bus routes' points, read without deriving it, are the loaded network's bus routes' points
#[test]
fn save_bus_points_are_the_loaded_bus_routes() {
    let mut a = api();
    bus(&mut a, 0.0);
    let gone = bus(&mut a, 100_000.0);
    bus(&mut a, 200_000.0);
    a.remove_line(gone);
    let bytes = a.save();
    let mut b = api();
    assert!(b.load(&bytes));
    let routes: Vec<f64> = b.bus_lines().iter().flat_map(|&l| b.bus_route(l)[1..].chunks(3).flat_map(|p| [p[0], p[1]]).collect::<Vec<_>>()).collect();
    assert_eq!(b.bus_lines().len(), 2);
    assert_eq!(TrackApi::save_bus_points(&bytes), routes);
    assert!(TrackApi::save_bus_points(&api().save()).is_empty(), "no bus line");
    assert!(TrackApi::save_bus_points(b"not a save").is_empty());
}

/// T-4193: street speeds put in before a load time its buses in the load's one settle, to the figures the load
/// and then the speeds give, where the settle holds still
#[test]
fn streets_before_a_load_give_the_figures_of_streets_after_it() {
    let mut a = api();
    let lines = [bus(&mut a, 0.0), bus(&mut a, 200_000.0)];
    let bytes = a.save();
    let tph = |a: &TrackApi| lines.iter().map(|&l| a.w.net.line_tph[l as usize]).collect::<Vec<_>>();
    let mut plain = api();
    assert!(plain.load(&bytes));
    let mut after = api();
    assert!(after.load(&bytes));
    assert_eq!(grid(&mut after, -73.985, 40.758), 2, "the first speeds time every bus line again");
    let mut before = api();
    assert_eq!(grid(&mut before, -73.985, 40.758), 0, "an empty network has no bus to time");
    assert!(before.load(&bytes));
    assert_ne!(times(&before, &lines)[0], times(&plain, &lines)[0], "the load ran its buses at the speeds");
    assert_eq!(times(&before, &lines), times(&after, &lines));
    assert_eq!(tph(&before), tph(&after));
}
