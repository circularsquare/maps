//! T-1464: bus corridors in demand.

use super::tests::small;
use super::*;
use crate::demand::network::{route_nodes, RealNet};
use crate::demand::pack;

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

/// T-1464 B1's gate: the real New York network as the commute layer solves it (five periods, free
/// flow and three crowding rounds), then with 30 feeder buses, timed. Demand time may rise a third
/// at most. `cargo test --release --features demand t1464_bus_cost -- --ignored --nocapture`
#[test]
#[ignore]
fn t1464_bus_cost() {
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/..");
    let nyc = pack::load_raw(&format!("{root}/data/packs/nyc.json")).unwrap();
    let (RealNet { st_xy, line_n, stops, times, tph, cars }, _) = RealNet::load(&format!("{root}/data/work/realnet/nyc_real_track.json"));
    let ns = st_xy.len() / 2;
    // 30 stations spread over the network, each with a 6 km feeder pointing away from the centre
    let (mut cx, mut cy) = (0.0f32, 0.0f32);
    for i in 0..ns {
        cx += st_xy[2 * i];
        cy += st_xy[2 * i + 1];
    }
    (cx, cy) = (cx / ns as f32, cy / ns as f32);
    let mut order: Vec<usize> = (0..ns).collect();
    order.sort_by(|&a, &b| (st_xy[2 * a] + 0.37 * st_xy[2 * a + 1]).total_cmp(&(st_xy[2 * b] + 0.37 * st_xy[2 * b + 1])));
    let picks: Vec<usize> = (0..30).map(|k| order[(k * ns) / 30 + ns / 60]).collect();
    let solve = |with_bus: bool| -> (f64, f64, f64, f64) {
        let mut api = DemandApi::new(&nyc.0, &nyc.1).unwrap();
        let (mut xy, mut n, mut st, mut ti, mut tp, mut ca) = (st_xy.clone(), line_n.clone(), stops.clone(), times.clone(), tph.clone(), cars.clone());
        let mut cn = vec![0u32; stops.len()];
        let mut cxy = vec![];
        if with_bus {
            for &s in &picks {
                let (x0, y0) = (st_xy[2 * s], st_xy[2 * s + 1]);
                let (dx, dy) = (x0 - cx, y0 - cy);
                let d = dx.hypot(dy).max(1.0);
                // across the line of the centre, so the feeders fan out rather than run along rails
                let dir = (-dy / d, dx / d);
                let (bstops, bcn, bcxy) = feeder(x0, y0, dir, 6000.0, 2000.0);
                let first = (xy.len() / 2) as u32;
                for p in &bstops[1..] {
                    xy.extend(p);
                }
                let ids: Vec<u32> = std::iter::once(s as u32).chain(first..first + 3).collect();
                st.extend(&ids);
                let m = ids.len();
                for run in 0..2 {
                    for _ in 0..3 {
                        for k in 0..m {
                            let end = if run == 0 { k == m - 1 } else { k == 0 };
                            ti.push(if end { 0.0 } else { bus_s(2000.0) });
                        }
                    }
                }
                tp.extend([8.0, 8.0, 4.0]);
                ca.push(0.5);
                n.push(m as u32);
                cn.extend(&bcn);
                cxy.extend(bcxy);
            }
            api.set_bus(&cn, &cxy);
        }
        let t0 = now_ms();
        let set = api.set_network(&xy, &n, &st, &ti, &tp, &ca).unwrap();
        let hop_km = route_nodes(&xy, &n, &st).2;
        api.set_fares(2.9, 0.1, &hop_km).unwrap();
        let mut rail = 0.0;
        for q in 0..PERIODS {
            api.solve(q);
            for _ in 0..3 {
                api.crowd(q);
            }
            rail += api.summary(q)[2] as f64;
        }
        (now_ms() - t0, set, rail, api.n_subzones() as f64)
    };
    let (t_base, set_base, rail_base, sz_base) = solve(false);
    let (t_bus, set_bus, rail_bus, sz_bus) = solve(true);
    println!("without buses: {t_base:.0} ms (set_network {set_base:.0}), rail {rail_base:.0}, {sz_base} subzones");
    println!("with 30 buses: {t_bus:.0} ms (set_network {set_bus:.0}), rail {rail_bus:.0}, {sz_bus} subzones");
    println!("demand time x{:.3}", t_bus / t_base);
}
