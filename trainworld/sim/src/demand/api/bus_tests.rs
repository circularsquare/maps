//! T-1464: bus corridors in demand.

use super::tests::small;
use super::*;

/// A feeder bus north from the station at (x0, 0): its stops every `step` m up to `len`, each with
/// its stretch of route (the station first, with none). Returns the line's stops' positions, the
/// catch counts per stop and the catch points.
fn feeder(x0: f32, y0: f32, dir: (f32, f32), len: f32, step: f32) -> (Vec<[f32; 2]>, Vec<u32>, Vec<f32>) {
    let n = (len / step).round() as usize;
    let at = |g: f32| [x0 + dir.0 * g, y0 + dir.1 * g];
    let stops: Vec<[f32; 2]> = (0..=n).map(|k| at(k as f32 * step)).collect();
    let (mut cn, mut cxy) = (vec![0u32], vec![]);
    for k in 1..=n {
        let a = if k == 1 { 0.0 } else { (k as f32 - 0.5) * step };
        let b = if k == n { len } else { (k as f32 + 0.5) * step };
        cn.push(2);
        cxy.extend(at(a));
        cxy.extend(at(b));
    }
    (stops, cn, cxy)
}

/// Bus seconds a hop: road 1.2 times the drawn metres at 18 km/h.
fn bus_s(m: f32) -> f32 {
    m * 1.2 / 5.0
}

#[test]
fn a_feeder_bus_brings_riders_to_rail_and_no_bus_changes_nothing() {
    let (mut api, xy, n, stops, times, tph, cars) = small();
    api.set_network(&xy, &n, &stops, &times, &tph, &cars).unwrap();
    assert!(api.solve(0));
    let base = (api.board(0), api.summary(0));
    // an empty bus list is no bus at all
    let (mut api2, ..) = small();
    api2.set_bus(&[], &[]);
    api2.set_network(&xy, &n, &stops, &times, &tph, &cars).unwrap();
    assert!(api2.solve(0));
    assert!(same_bits(&api2.board(0), &base.0) && api2.summary(0)[..4] == base.1[..4]);
    // a bus north 6 km from the station at (4 km, 0)
    let (bstops, bcn, bcxy) = feeder(4000.0, 0.0, (0.0, 1.0), 6000.0, 2000.0);
    let mut xy3 = xy.clone();
    let first = (xy3.len() / 2) as u32;
    for p in &bstops[1..] {
        xy3.extend(p);
    }
    let mut stops3 = stops.clone();
    let bus: Vec<u32> = std::iter::once(12).chain(first..first + 3).collect();
    stops3.extend(&bus);
    let mut times3 = times.clone();
    let m = bus.len();
    for run in 0..2 {
        for _ in 0..3 {
            for k in 0..m {
                let end = if run == 0 { k == m - 1 } else { k == 0 };
                times3.push(if end { 0.0 } else { bus_s(2000.0) });
            }
        }
    }
    let mut tph3 = tph.clone();
    tph3.extend([8.0, 8.0, 4.0]);
    let mut cars3 = cars.clone();
    cars3.push(0.5);
    let mut n3 = n.clone();
    n3.push(m as u32);
    let mut cn = vec![0u32; stops.len()];
    cn.extend(&bcn);
    let (mut api3, ..) = small();
    api3.set_bus(&cn, &bcxy);
    api3.set_network(&xy3, &n3, &stops3, &times3, &tph3, &cars3).unwrap();
    assert!(api3.solve(0));
    let s3 = api3.summary(0);
    let b3 = api3.board(0);
    // route nodes: line a 34, line b 34, the bus 8 (its 4 stops in order, then reversed)
    let bus_board: f32 = b3[68..76].iter().sum();
    assert!(bus_board > 0.0, "riders board the bus");
    assert!(s3[2] > base.1[2], "rail trips rise with the feeder: {} against {}", s3[2], base.1[2]);
    let sz = api3.sz.as_ref().unwrap();
    let net = DemandApi::period_net(api3.svc.as_ref().unwrap(), 0);
    assert!(net.bus_only(net.st_x.len() - 1) && !net.bus_only(12));
    // a bus stop is reached only through its corridor: the subzones with it lie near the route
    // (x = 4 km), at most two bus stops each
    for u in 0..sz.n {
        let bus: Vec<usize> = sz.walk(u).filter(|&k| net.bus_only(sz.acc_st[k] as usize)).collect();
        assert!(bus.len() <= BUS_SLOTS);
        if !bus.is_empty() {
            assert!((sz.cx[u] - 4000.0).abs() < 1500.0, "subzone at {} reaches the bus", sz.cx[u]);
        }
    }
}
