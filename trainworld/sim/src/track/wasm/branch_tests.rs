use super::draw::BRANCH_CLEAR;
use super::*;

fn api() -> TrackApi {
    let mut a = TrackApi::new(-73.985, 40.758, true);
    a.set_cash(1e4);
    a.set_route_tangent(true);
    a
}

fn free(x: f64, y: f64) -> [f64; 5] {
    [0.0, 0.0, x, y, 0.0]
}

fn ends(a: [f64; 5], b: [f64; 5]) -> Vec<f64> {
    a.iter().chain(b.iter()).copied().collect()
}

fn node(n: u32) -> [f64; 5] {
    [1.0, n as f64, 0.0, 0.0, 0.0]
}

/// The live edge whose ends lie either side of x.
fn edge_over(a: &TrackApi, x: f64) -> u32 {
    let n = &a.w.net;
    (0..n.edge_count() as u32)
        .find(|&e| {
            let (xa, xb) = (n.node_x[n.edge_a[e as usize] as usize], n.node_x[n.edge_b[e as usize] as usize]);
            n.edge_ok(e) && xa.min(xb) < x && xa.max(xb) > x && n.node_y[n.edge_a[e as usize] as usize].abs() < 1.0
        })
        .unwrap()
}

fn station_at(a: &TrackApi, x: f64) -> u32 {
    let n = &a.w.net;
    (0..n.node_count() as u32).find(|&m| n.node_ok(m) && n.node_platform[m as usize] > 0 && (n.node_x[m as usize] - x).abs() < 1.0).unwrap()
}

/// The junctions (three edge ends or more) and the ground length of the track from the station to each.
fn junctions(a: &TrackApi, st: u32) -> Vec<(u32, f64)> {
    let n = &a.w.net;
    let mut out = vec![];
    for &c in n.node_ports[st as usize].ends() {
        let (e, which) = (c >> 1, c & 1);
        let other = n.end_node(e, 1 - which);
        if n.node_ports[other as usize].n >= 3 {
            out.push((other, n.edge_len[e as usize]));
        }
    }
    out
}

/// A straight double track 0..3000 (plane m) with a 200 m station at 1500.
fn through_station() -> (TrackApi, u32) {
    let mut a = api();
    assert_eq!(a.add_route(&ends(free(0.0, 0.0), free(3000.0, 0.0)), &[], 2, false), 0, "{}", a.issues());
    let e = edge_over(&a, 1500.0);
    assert_eq!(a.add_station(2, e, 1500.0, 0.0, 200, "Mid".into()), 0, "{}", a.issues());
    let st = station_at(&a, 1500.0);
    (a, st)
}

#[test]
fn from_a_through_station_node_the_branch_leaves_past_the_platform() {
    for (to, east) in [((2200.0, 700.0), true), ((800.0, 700.0), false)] {
        let (mut a, st) = through_station();
        let edges = a.w.net.edge_count();
        let pv = a.preview_route(&ends(node(st), free(to.0, to.1)), &[], 2, false);
        assert_eq!(pv[0], 1.0, "{}", a.issues());
        assert_eq!(a.add_route(&ends(node(st), free(to.0, to.1)), &[], 2, false), 0, "{}", a.issues());
        assert_eq!(a.w.net.node_ports[st as usize].n, 2, "the station keeps two track ends");
        let j = junctions(&a, st);
        assert_eq!(j.len(), 1, "{j:?}");
        assert!((j[0].1 - (100.0 + BRANCH_CLEAR)).abs() < 0.5, "{j:?}");
        let jx = a.w.net.node_x[j[0].0 as usize];
        assert_eq!(jx > 1500.0, east, "the side the click lies on: {jx}");
        assert_eq!(a.undo(), 0);
        assert!(junctions(&a, st).is_empty());
        assert!(a.w.net.edge_count() <= edges + 2);
    }
}

#[test]
fn a_click_on_the_platform_and_an_end_on_the_station_branch_the_same_way() {
    let (mut a, st) = through_station();
    let e = edge_over(&a, 1450.0);
    assert_eq!(a.add_route(&ends([2.0, e as f64, 1450.0, 0.0, 0.0], free(2200.0, 700.0)), &[], 2, false), 0, "{}", a.issues());
    let j = junctions(&a, st);
    assert!(j.len() == 1 && a.w.net.node_x[j[0].0 as usize] > 1500.0, "{j:?}");
    assert_eq!(a.undo(), 0);
    // ending on the station: the junction on the side the route comes from
    assert_eq!(a.add_route(&ends(free(800.0, 700.0), node(st)), &[], 2, false), 0, "{}", a.issues());
    let j = junctions(&a, st);
    assert!(j.len() == 1 && a.w.net.node_x[j[0].0 as usize] < 1500.0, "{j:?}");
    assert_eq!(a.w.net.node_ports[st as usize].n, 2);
}

#[test]
fn no_room_past_the_platform_is_refused() {
    let (mut a, st) = through_station();
    // a second station whose platform starts 20 m past the first's (ground metres)
    let k = a.w.net.k_at(0.0);
    let x = 1500.0 + 220.0 * k;
    let e = edge_over(&a, x);
    assert_eq!(a.add_station(2, e, x, 0.0, 200, "Next".into()), 0, "{}", a.issues());
    assert_eq!(a.add_route(&ends(node(st), free(2200.0, 700.0)), &[], 2, false), 1);
    assert!(a.issue_kinds().contains("BranchNoRoom"), "{}", a.issue_kinds());
    // the other side is clear
    assert_eq!(a.add_route(&ends(node(st), free(800.0, 700.0)), &[], 2, false), 0, "{}", a.issues());
}

#[test]
fn an_end_station_extends_and_branches_behind_its_platform() {
    let mut a = api();
    assert_eq!(a.add_route(&ends(free(0.0, 0.0), free(3000.0, 0.0)), &[], 2, false), 0, "{}", a.issues());
    let end = a.route_end();
    assert_eq!(a.add_station(1, end, 0.0, 0.0, 200, "End".into()), 0, "{}", a.issues());
    // its node: the line extends
    assert_eq!(a.add_route(&ends(node(end), free(3800.0, 300.0)), &[], 2, false), 0, "{}", a.issues());
    assert_eq!(a.w.net.node_ports[end as usize].n, 2);
    assert!(junctions(&a, end).is_empty());
    assert_eq!(a.undo(), 0);
    // its platform, drawn back the way the track comes: a branch past the platform's far end
    let e = edge_over(&a, 2950.0);
    assert_eq!(a.add_route(&ends([2.0, e as f64, 2950.0, 0.0, 0.0], free(2000.0, 700.0)), &[], 2, false), 0, "{}", a.issues());
    let j = junctions(&a, end);
    assert!(j.len() == 1 && (j[0].1 - (200.0 + BRANCH_CLEAR)).abs() < 0.5, "{j:?}");
    assert_eq!(a.w.net.node_ports[end as usize].n, 1);
}

#[test]
fn a_station_on_its_own_piece_carries_on_from_the_loose_end() {
    let mut a = api();
    // the platform and 10 m past each end, as the station tool lays it
    assert_eq!(a.add_free_station(&ends(free(0.0, 0.0), free(220.0, 0.0)), 2, 200, "Own".into()), 0, "{}", a.issues());
    let st = station_at(&a, 110.0);
    assert_eq!(a.add_route(&ends(node(st), free(900.0, 300.0)), &[], 2, false), 0, "{}", a.issues());
    assert!(junctions(&a, st).is_empty(), "{:?}", junctions(&a, st));
    let n = &a.w.net;
    assert!((0..n.node_count()).all(|m| !n.node_ok(m as u32) || n.node_ports[m].n <= 2), "no junction anywhere");
}

#[test]
fn quad_and_high_speed_branch_past_the_platform() {
    // quad: the station and its twin, the branch from the clicked pair's station
    let mut a = api();
    assert_eq!(a.add_route(&ends(free(0.0, 0.0), free(3000.0, 0.0)), &[], 4, false), 0, "{}", a.issues());
    let e = edge_over(&a, 1500.0);
    assert_eq!(a.add_station(2, e, 1500.0, 0.0, 200, "Mid".into()), 0, "{}", a.issues());
    let st = station_at(&a, 1500.0);
    assert_eq!(a.add_route(&ends(node(st), free(2200.0, -700.0)), &[], 4, false), 0, "{}", a.issues());
    assert_eq!(a.w.net.node_ports[st as usize].n, 2);
    // the crossover past the station is a junction too, 50 m beyond the platform
    let j = junctions(&a, st);
    assert_eq!(j.iter().filter(|j| (j.1 - (100.0 + BRANCH_CLEAR)).abs() < 0.5).count(), 1, "{j:?}");
    // high speed
    let mut a = api();
    a.set_route_hsr(true);
    assert_eq!(a.add_route(&ends(free(0.0, 0.0), free(3000.0, 0.0)), &[], 2, false), 0, "{}", a.issues());
    let e = edge_over(&a, 1500.0);
    assert_eq!(a.add_station(2, e, 1500.0, 0.0, 200, "Mid".into()), 0, "{}", a.issues());
    let st = station_at(&a, 1500.0);
    assert_eq!(a.add_route(&ends(node(st), free(2200.0, 700.0)), &[], 2, false), 0, "{}", a.issues());
    let j = junctions(&a, st);
    assert_eq!(j.len(), 1);
    let n = &a.w.net;
    assert!((0..n.edge_count()).filter(|&e| n.edge_ok(e as u32)).all(|e| n.edge_hsr[e]), "the split keeps high speed");
}
