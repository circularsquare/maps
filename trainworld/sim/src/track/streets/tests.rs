use super::*;

#[test]
fn a_bus_runs_at_its_share_of_the_car_or_its_lane() {
    assert!((bus_kmh(car_street_kmh(0.0)) - 44.0 * BUS_SHARE).abs() < 1e-9);
    assert_eq!(bus_kmh(car_street_kmh(1e9)), BUS_LANE_KMH);
    assert_eq!(bus_kmh(car_street_kmh(5000.0)), (BUS_SHARE * car_street_kmh(5000.0)).max(BUS_LANE_KMH));
    assert!((bus_kmh(100.0) - 100.0 * BUS_SHARE).abs() < 1e-9);
}

/// T-3907: the trunk road cap is the line's own street speed times the share, so the floor and the share are the
/// two numbers to set: at a 25 floor and 1.4x a dense place gives 25 / 35 on the street and 35 / 49 on a trunk road
/// for a plain bus / a BRT one; open country's main roads keep their speed.
#[test]
fn a_trunk_road_lifts_each_line_to_its_share_over_its_own_street_speed() {
    let floor = 25.0;
    let brt = floor + crate::track::params::BRT_LANE_BONUS_KMH;
    let car = car_street_kmh(1e9);
    let road = car_main_kmh(car);
    assert!((bus_kmh_at(car, floor) - 25.0).abs() < 1e-9 && (bus_kmh_at(car, brt) - 35.0).abs() < 1e-9);
    assert!((bus_fast_kmh(road, car, floor, 1.4) - 35.0).abs() < 1e-9, "{}", bus_fast_kmh(road, car, floor, 1.4));
    assert!((bus_fast_kmh(road, car, brt, 1.4) - 49.0).abs() < 1e-9, "{}", bus_fast_kmh(road, car, brt, 1.4));
    assert!((bus_fast_kmh(car, car, brt, 1.4) - 49.0).abs() < 1e-9);
    // today's figures: 30 and 1.5
    assert!((bus_trunk_kmh(road, car, lane_kmh(false)) - BUS_TRUNK_MAX_X * BUS_LANE_KMH).abs() < 1e-9);
    // open country: the trunk's street at 1.5 x 37.4, its main road at the share of 88 as without the trunk
    let open = car_street_kmh(0.0);
    assert!((bus_fast_kmh(open, open, floor, 1.5) - 1.5 * BUS_SHARE * open).abs() < 1e-9);
    assert!((bus_fast_kmh(car_main_kmh(open), open, floor, 1.5) - BUS_SHARE * CAR_FAR_KMH).abs() < 1e-9);
    // never over the share of the trunk's car
    assert!((bus_fast_kmh(open, open, floor, 3.0) - BUS_SHARE * CAR_TRUNK_KMH).abs() < 1e-9);
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

/// T-3913: the in-town squares come out as one polygon per patch joined side to side, its outline first and then its
/// holes, a patch meeting another only at a corner kept apart; squares an earlier grid covers are left to it.
#[test]
fn the_town_outline_merges_squares_into_polygons_with_holes() {
    // 4 x 4 squares, one zone each: a 3 x 3 block in town around an open square, and one more in town touching its
    // corner
    let town = |x: u16, y: u16| (x < 3 && y < 3 && (x, y) != (1, 1)) || (x, y) == (3, 3);
    let (mut gx, mut gy, mut pop) = (vec![], vec![], vec![]);
    for x in 0..4u16 {
        for y in 0..4u16 {
            gx.push(x);
            gy.push(y);
            pop.push(if town(x, y) { 40_000.0f32 } else { 1.0 });
        }
    }
    let zone: Vec<u32> = (0..16).collect();
    let grid = || StreetGrid::from_zones(1000.0, 0.0, 0.0, 4, 4, &pop, &pop, &zone, &gx, &gy, &[], &[1; 16], 10.0, 0.0).unwrap();
    let out = grid().town_outline(&[]);
    let (mut at, mut polys) = (1, vec![]);
    for _ in 0..out[0] as usize {
        let mut rings = vec![];
        let n = out[at] as usize;
        at += 1;
        for _ in 0..n {
            let k = out[at] as usize;
            rings.push(out[at + 1..at + 1 + 2 * k].to_vec());
            at += 1 + 2 * k;
        }
        polys.push(rings);
    }
    assert_eq!(at, out.len());
    let sizes: Vec<Vec<usize>> = polys.iter().map(|p| p.iter().map(|r| r.len() / 2).collect()).collect();
    assert_eq!(sizes, vec![vec![5, 5], vec![5]]);
    // the block's outline runs from the frame's origin to 3 km east and north
    let lons: Vec<f64> = polys[0][0].iter().step_by(2).copied().collect();
    let east = Frame::equirect(10.0, 0.0).to_lonlat(3000.0, 0.0).0;
    assert!(lons.iter().any(|&l| (l - 10.0).abs() < 1e-9) && lons.iter().any(|&l| (l - east).abs() < 1e-9));
    // the same squares again in a second grid: all of them are the first grid's
    let streets = Streets { grids: vec![Arc::new(grid()), Arc::new(grid())] };
    assert_eq!(streets.last_town_outline(), vec![0.0]);
}
