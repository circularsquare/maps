use super::*;

#[test]
fn a_bus_runs_at_nine_tenths_of_the_car_or_its_lane() {
    assert!((bus_kmh(car_street_kmh(0.0)) - 39.6).abs() < 1e-9);
    assert_eq!(bus_kmh(car_street_kmh(1e9)), BUS_LANE_KMH);
    assert_eq!(bus_kmh(car_street_kmh(5000.0)), BUS_LANE_KMH);
    assert!((bus_kmh(100.0) - 90.0).abs() < 1e-9);
}

#[test]
fn main_roads_follow_the_local_streets_density() {
    assert_eq!(car_main_kmh(CAR_CITY_KMH), CAR_CITY_KMH);
    assert_eq!(car_main_kmh(CAR_TOWN_KMH), CAR_FAR_KMH);
    assert!((car_main_kmh(car_street_kmh(5000.0)) - 52.25).abs() < 1e-9);
}

#[cfg(feature = "demand")]
#[test]
fn the_car_speed_is_demands() {
    let p = crate::demand::kernel::Params::default();
    for d in [0.0f32, 500.0, 2000.0, 5000.0, 12_000.0, 30_000.0, 90_000.0] {
        let want = 60.0 / p.place_min_per_km(d).0 as f64;
        assert!((car_street_kmh(d as f64) - want).abs() < 1e-3, "{d}: {} {want}", car_street_kmh(d as f64));
    }
    assert_eq!(CELL_KM2 as f32, crate::demand::kernel::CELL_KM2);
    assert_eq!((CAR_FAR_KMH as f32, CAR_OPEN_KMH as f32), (p.road_far_kmh, p.road_open_kmh));
    assert_eq!((CAR_LOCAL_M as f32, CAR_END_M as f32), (p.road_local_km * 1000.0, p.road_end_km * 1000.0));
}

/// Two zones side by side on a 2 km grid of 4 x 2 squares, the west one dense.
fn grid() -> StreetGrid {
    // cells 0 and 1 in zone 0, 2 and 3 in zone 1
    let (pop, jobs) = ([30_000.0f32, 12_000.0, 100.0, 50.0], [20_000.0f32, 8_000.0, 10.0, 5.0]);
    StreetGrid::from_zones(2000.0, 0.0, 0.0, 4, 2, &pop, &jobs, &[0, 0, 1, 1], &[0, 1], &[0, 0], &[], &[2, 1], 10.0, 50.0).unwrap()
}

#[test]
fn a_grid_gives_each_square_its_zones_street_speed() {
    let g = grid();
    let pack = Frame::equirect(10.0, 50.0);
    let at = |x: f64, y: f64| {
        let (lon, lat) = pack.to_lonlat(x, y);
        g.at(lon, lat)
    };
    // zone 0: 100,000 people and jobs over 2 cells of 0.1053 km²: far past dense
    assert!((at(1000.0, 1000.0).unwrap() - 16.9).abs() < 0.3);
    // zone 1: 165 over one cell, 1,570 a km2
    assert!((at(3000.0, 1000.0).unwrap() - 37.4).abs() < 0.3);
    // beyond the grid, and a square with no zone
    assert_eq!(at(9000.0, 1000.0), None);
    assert_eq!(at(-100.0, 1000.0), None);
    assert_eq!(at(5000.0, 3000.0), None);
}

#[test]
fn arrays_that_do_not_fit_make_no_grid() {
    assert!(StreetGrid::from_zones(2000.0, 0.0, 0.0, 4, 2, &[1.0], &[1.0], &[0], &[0], &[0], &[], &[], 0.0, 0.0).is_none());
    assert!(StreetGrid::from_zones(2000.0, 0.0, 0.0, 4, 2, &[1.0], &[1.0], &[3], &[0], &[0], &[], &[1], 0.0, 0.0).is_none());
}
