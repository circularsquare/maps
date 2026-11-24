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
