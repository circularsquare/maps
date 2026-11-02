use super::*;
use std::f64::consts::FRAC_1_SQRT_2;

fn live(a: &TrackApi) -> Vec<u32> {
    (0..a.w.net.edge_count() as u32).filter(|&e| a.w.net.edge_ok(e)).collect()
}
fn nearest(a: &TrackApi, edges: &[u32], x: f64, y: f64) -> u32 {
    *edges.iter().min_by(|&&p, &&q| a.w.net.project(p, x, y).1.total_cmp(&a.w.net.project(q, x, y).1)).unwrap()
}

/// T-1685: a route drawn beside a curved run of single track (two pieces) runs `beside::BESIDE_SPACING`
/// from it all along, curves included, on the side its end is; without `set_route_beside` the
/// same ends make a plain route that does not.
#[test]
fn route_beside_track_runs_parallel() {
    let mut a = TrackApi::new(-73.985, 40.758, true);
    a.set_route_tangent(true);
    assert_eq!(a.add_route(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1000.0, 0.0, 0.0], &[], 1, false), 0, "{}", a.issues());
    let j = a.route_end();
    assert_eq!(a.add_route(&[1.0, j as f64, 0.0, 0.0, 0.0, 0.0, 0.0, 2600.0, 1400.0, 0.0], &[1800.0, 0.0, 600.0, 0.0], 1, false), 0, "{}", a.issues());
    let local = live(&a);
    let (s, e) = ((300.0, 20.0), (2350.0, 1150.0));
    let ends = [0.0, 0.0, s.0, s.1, 0.0, 0.0, 0.0, e.0, e.1, 0.0];
    a.set_route_beside(nearest(&a, &local, s.0, s.1), nearest(&a, &local, e.0, e.1));
    assert_eq!(a.add_route(&ends, &[], 2, false), 0, "{}", a.issues());
    let new: Vec<u32> = live(&a).into_iter().filter(|e| !local.contains(e)).collect();
    let mut worst: f64 = 0.0;
    let mut n = 0;
    for &p in &new {
        for q in a.edge_render(p).chunks(3) {
            let off = local.iter().map(|&l| a.w.net.project(l, q[0], q[1]).1).fold(f64::INFINITY, f64::min);
            worst = worst.max((off - beside::BESIDE_SPACING).abs());
            n += 1;
            assert!(q[1] > 0.0, "on the end's side");
        }
    }
    assert!(n > 20 && worst < 0.05, "{n} points, worst {worst} m off the spacing");
    // the same ends drawn plain: a straight across the curve
    a.set_route_beside(u32::MAX, u32::MAX);
    let pv = a.preview_route(&ends, &[], 2, false);
    assert!(pv[3] <= 3.0, "no corners: {}", pv[3]);
    // T-1721: from a track end 300 m off the run, coming in and then running along it
    let far = (900.0, -300.0);
    assert_eq!(a.add_route(&[0.0, 0.0, 300.0, -300.0, 0.0, 0.0, 0.0, far.0, far.1, 0.0], &[], 2, false), 0, "{}", a.issues());
    let tip = a.route_end();
    let near = (2500.0, 1000.0); // right of the curve's second leg (the left has the first route)
    let before = live(&a);
    a.set_route_beside(u32::MAX, nearest(&a, &local, near.0, near.1));
    assert_eq!(a.add_route(&[1.0, tip as f64, 0.0, 0.0, 0.0, 0.0, 0.0, near.0, near.1, 0.0], &[], 2, false), 0, "{}", a.issues());
    let came: Vec<u32> = live(&a).into_iter().filter(|e| !before.contains(e)).collect();
    let pts: Vec<f64> = came.iter().flat_map(|&p| a.edge_render(p)).collect();
    let offs: Vec<f64> = pts.chunks(3).map(|q| local.iter().map(|&l| a.w.net.project(l, q[0], q[1]).1).fold(f64::INFINITY, f64::min)).collect();
    let on = offs.iter().filter(|o| (*o - beside::BESIDE_SPACING).abs() < 0.05).count();
    assert!(on * 3 > offs.len(), "{on} of {} points along the line", offs.len());
    let end = a.route_end() as usize;
    let off = local.iter().map(|&l| a.w.net.project(l, a.w.net.node_x[end], a.w.net.node_y[end]).1).fold(f64::INFINITY, f64::min);
    assert!((off - beside::BESIDE_SPACING).abs() < 0.05, "end {off} m off");
    // the plain route again, for the rest of this test
    a.set_route_beside(u32::MAX, u32::MAX);
    let pv = a.preview_route(&ends, &[], 2, false);
    assert!(pv[3] <= 3.0, "no corners: {}", pv[3]);
    // the end to the right of the track: the start goes to the right too
    let r = (2500.0, 1000.0);
    a.set_route_beside(nearest(&a, &local, s.0, s.1), nearest(&a, &local, r.0, r.1));
    let pv = a.preview_route(&[0.0, 0.0, s.0, s.1, 0.0, 0.0, 0.0, r.0, r.1, 0.0], &[], 2, false);
    assert!((pv[6] + beside::BESIDE_SPACING).abs() < 0.01, "start at {}", pv[6]);
}

/// T-1781: coming in from open ground 150 m off a straight, the route turns onto the line beside it
/// in a wide curve (the turn at the start's own distance along gave about 500 m), still on its side.
#[test]
fn coming_in_beside_takes_the_roundest_turn() {
    let mut a = TrackApi::new(-73.985, 40.758, true);
    a.set_route_tangent(true);
    assert_eq!(a.add_route(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3000.0, 0.0, 0.0], &[], 1, false), 0, "{}", a.issues());
    let local = live(&a);
    let end = (2800.0, 10.0);
    a.set_route_beside(u32::MAX, nearest(&a, &local, end.0, end.1));
    let pv = a.preview_route(&[0.0, 0.0, 300.0, 150.0, 0.0, 0.0, 0.0, end.0, end.1, 0.0], &[], 2, false);
    assert_eq!(pv[0], 1.0, "{}", a.issues());
    let nv = pv[3] as usize;
    let radius = (1..nv - 1).map(|i| pv[5 + i * 4 + 2]).fold(0.0, f64::max);
    assert!(radius > 1000.0, "turn in at {radius} m");
    let p0 = 5 + nv * 4;
    assert!(pv[p0..].chunks(3).all(|q| q[1] > 0.0), "stays on its side");
}

/// T-1708: a route ramping over a line with a station near the crossing keeps a level stretch on
/// the bridge at least that station's platform long; with no station there it is shorter.
#[test]
fn bridge_beside_a_station_has_room_for_one() {
    let level_run = |station: bool| -> f64 {
        let mut a = TrackApi::new(-73.985, 40.758, true);
        a.set_route_tangent(true);
        a.set_route_ramp(true);
        assert_eq!(a.add_route(&[0.0, 0.0, 1500.0, -800.0, 0.0, 0.0, 0.0, 1500.0, 800.0, 0.0], &[], 2, false), 0, "{}", a.issues());
        if station {
            assert_eq!(a.add_station(2, live(&a)[0], 1500.0, 160.0, 200, "S".into()), 0, "{}", a.issues());
        }
        let before = live(&a);
        assert_eq!(a.add_route(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3000.0, 0.0, 0.0], &[], 2, false), 0, "{}", a.issues());
        let xs: Vec<f64> = live(&a).into_iter().filter(|e| !before.contains(e)).flat_map(|e| a.edge_render(e)).collect::<Vec<f64>>().chunks(3).filter(|q| (q[2] - LEVEL_H).abs() < 0.01).map(|q| q[0]).collect();
        xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max) - xs.iter().cloned().fold(f64::INFINITY, f64::min)
    };
    let (plain, room) = (level_run(false), level_run(true));
    assert!(room >= 200.0 && room > plain, "level stretch {room} m with a station, {plain} m without");
}

/// T-1722: beside track that ramps over a line on a slight curve, the route beside it ramps over
/// that line too.
#[test]
fn route_beside_bridges_on_a_slight_curve() {
    for turn in [0.0f64, 1.0, 3.0, 5.0, 8.0, 12.0, -3.0, -8.0] {
        let mut a = TrackApi::new(-73.985, 40.758, true);
        a.set_route_tangent(true);
        a.set_route_ramp(true);
        assert_eq!(a.add_route(&[0.0, 0.0, 1500.0, -800.0, 0.0, 0.0, 0.0, 1500.0, 800.0, 0.0], &[], 2, false), 0, "{}", a.issues());
        let t = turn.to_radians();
        let end = (1500.0 + 1500.0 * t.cos(), 1500.0 * t.sin());
        assert_eq!(a.add_route(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, end.0, end.1, 0.0], &[1500.0, 0.0, 0.0, 0.0], 2, false), 0, "turn {turn}: {}", a.issues());
        let local = live(&a);
        let (s, e) = ((200.0, 14.0), (end.0 - 200.0 * t.cos() - 14.0 * t.sin(), end.1 - 200.0 * t.sin() + 14.0 * t.cos()));
        a.set_route_beside(u32::MAX, nearest(&a, &local, e.0, e.1));
        assert_eq!(a.add_route(&[0.0, 0.0, s.0, s.1, 0.0, 0.0, 0.0, e.0, e.1, 0.0], &[], 2, false), 0, "turn {turn}: {}", a.issues());
    }
}

/// T-1895: a chain beside track keeps to its side when the pointer strays across the track, and a
/// start coming in from the far side stays on its own side: neither crosses the track.
#[test]
fn beside_keeps_to_the_start_side() {
    let mut a = TrackApi::new(-73.985, 40.758, true);
    a.set_route_tangent(true);
    assert_eq!(a.add_route(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4000.0, 0.0, 0.0], &[], 2, false), 0, "{}", a.issues());
    let local = live(&a);
    let gap = beside::BESIDE_SPACING;
    a.set_route_beside(u32::MAX, local[0]);
    assert_eq!(a.add_route(&[0.0, 0.0, 300.0, gap, 0.0, 0.0, 0.0, 1000.0, 20.0, 0.0], &[], 2, false), 0, "{}", a.issues());
    let n = a.route_end();
    for (start, ptr) in [([1.0, n as f64, 0.0, 0.0, 0.0], (1700.0, -20.0)), ([0.0, 0.0, 1000.0, 120.0, 0.0], (2500.0, -25.0))] {
        a.set_route_beside(u32::MAX, local[0]);
        let mut ends = start.to_vec();
        ends.extend([0.0, 0.0, ptr.0, ptr.1, 0.0]);
        let pv = a.preview_route(&ends, &[], 2, false);
        assert_eq!(pv[0], 1.0, "{}", a.issues());
        let p0 = 5 + pv[3] as usize * 4;
        assert!(pv[p0..].chunks(3).all(|q| q[1] > gap - 0.05), "from {start:?}: stays on its side");
        assert!((pv[p0..].chunks(3).last().unwrap()[1] - gap).abs() < 0.05, "ends on the line");
    }
}

/// T-1895: the pointer level with the middle of a curve: the route ends there on the line, concentric
/// all the way (cut from the polyline of corners, the line used to stop where the curve's corner leg
/// did, and its last curve's set radius had no room).
#[test]
fn beside_ends_inside_a_curve() {
    let mut a = TrackApi::new(-73.985, 40.758, true);
    assert_eq!(a.add_route(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3000.0, 3000.0, 0.0], &[3000.0, 0.0, 0.0, 0.0], 2, false), 0, "{}", a.issues());
    let local = live(&a);
    let gap = beside::BESIDE_SPACING;
    // the curve (the auto radius, capped) turns left about its centre; 30 m inside and outside its middle
    let c = (3000.0 - R_CAP, R_CAP);
    let at = |r: f64| (c.0 + r * FRAC_1_SQRT_2, c.1 - r * FRAC_1_SQRT_2);
    for (sx, sy, ptr) in [(300.0, gap, at(R_CAP - 30.0)), (300.0, -gap, at(R_CAP + 30.0))] {
        a.set_route_beside(u32::MAX, local[0]);
        let pv = a.preview_route(&[0.0, 0.0, sx, sy, 0.0, 0.0, 0.0, ptr.0, ptr.1, 0.0], &[], 2, false);
        assert_eq!(pv[0], 1.0, "{}", a.issues());
        let p0 = 5 + pv[3] as usize * 4;
        let worst = pv[p0..].chunks(3).map(|q| (a.w.net.project(local[0], q[0], q[1]).1 - gap).abs()).fold(0.0, f64::max);
        assert!(worst < 0.05, "{worst} m off the spacing");
        let end = pv[p0..].chunks(3).last().unwrap();
        assert!((end[0] - ptr.0).hypot(end[1] - ptr.1) < 30.0, "ends level with the pointer");
    }
}

/// Whether polyline `p` (x, y, h triples) crosses polyline `q`, away from (x, y) within `skip` m.
fn polylines_cross(p: &[f64], q: &[f64], skip: Option<(f64, f64)>) -> bool {
    let side = |a: (f64, f64), b: (f64, f64), c: (f64, f64)| (b.0 - a.0) * (c.1 - a.1) - (b.1 - a.1) * (c.0 - a.0);
    let (p, q): (Vec<(f64, f64)>, Vec<(f64, f64)>) = (p.chunks(3).map(|c| (c[0], c[1])).collect(), q.chunks(3).map(|c| (c[0], c[1])).collect());
    p.windows(2).any(|u| {
        if skip.is_some_and(|s| (u[0].0 - s.0).hypot(u[0].1 - s.1) < 3.0) {
            return false;
        }
        q.windows(2).any(|w| side(u[0], u[1], w[0]) * side(u[0], u[1], w[1]) < 0.0 && side(w[0], w[1], u[0]) * side(w[0], w[1], u[1]) < 0.0)
    })
}

/// T-1895: drawing beside track never crosses the track it follows. A sweep over the real save
/// (trainworld/T-007-real-nyc.save, Anita's): every `STEP` m along every piece, a pointer 300 m on
/// at 30 m to either side, from starts on the line, coming in from 70 m off either side, and from
/// a loose end on the line (a segment drawn beside first, then taken back). Counts previews that
/// cross the run followed, by start. Runs on request, given the save's network bytes
/// (`python trainworld/checks/save_net.py <save> <out>`) in `T1895_NET`:
/// `cargo test --manifest-path sim/Cargo.toml --release beside_sweep -- --ignored --nocapture`.
#[test]
#[ignore]
fn beside_sweep_real_save() {
    const STEP: f64 = 500.0;
    let raw = std::fs::read(std::env::var("T1895_NET").expect("T1895_NET: the save's network bytes")).unwrap();
    let mut a = TrackApi::new(-73.985, 40.758, true);
    assert!(a.load(&super::save::convert_frame(&raw, FrameKind::Merc).unwrap()));
    a.cash = 1e9;
    let debug = std::env::var("T1895_DEBUG").ok();
    a.set_route_tangent(true);
    a.set_route_ramp(true);
    let edges = live(&a);
    let scaled = a.w.net.frame().scaled();
    let names = ["on line", "in, same side", "in, other side", "loose end", "end, other side"];
    let (mut tried, mut ok, mut crossed) = ([0usize; 5], [0usize; 5], [0usize; 5]);
    let mut examples: Vec<String> = vec![];
    let mut shown = [0usize; 5];
    let mut refused: std::collections::BTreeMap<String, usize> = Default::default();
    let mut why: std::collections::BTreeMap<String, usize> = Default::default();
    for &e in &edges {
        if a.w.net.edge_crossover[e as usize] {
            continue;
        }
        let Some((_, run)) = a.beside_run(e, e, scaled) else { continue };
        let run_pts: Vec<Vec<f64>> = run.iter().map(|&r| a.edge_render(r)).collect();
        let len = a.w.net.edge_len[e as usize];
        let mut s = STEP / 2.0;
        while s < len {
            let k = a.w.net.k_at(a.w.net.node_y[a.w.net.edge_a[e as usize] as usize]);
            let pieces = a.w.net.edge_pieces(e).to_vec();
            let at = |s: f64, off: f64| {
                let (x, y, th) = geom::pos_at(&pieces, s.clamp(0.0, len));
                (x - off * k * th.sin(), y + off * k * th.cos())
            };
            let s2 = if s + 300.0 < len { s + 300.0 } else { s - 300.0 };
            for side in [1.0, -1.0] {
                let q = at(s2, 30.0 * side);
                let free = |p: (f64, f64)| [0.0, 0.0, p.0, p.1, 0.0];
                let mut starts: Vec<(usize, [f64; 5], Option<(f64, f64)>)> = vec![
                    (0, free(at(s, 20.0 * side)), None),
                    (1, free(at(s, 70.0 * side)), None),
                    (2, free(at(s, -70.0 * side)), None),
                ];
                // a loose end on the line: a segment drawn beside from 200 m back, kept for the previews
                let back = if s2 > s { s - 200.0 } else { s + 200.0 };
                let mut undo = false;
                if (0.0..len).contains(&back) {
                    let (p0, p1) = (at(back, 8.0 * side), at(s, 8.0 * side));
                    a.set_route_beside(u32::MAX, e);
                    if a.add_route(&[0.0, 0.0, p0.0, p0.1, 0.0, 0.0, 0.0, p1.0, p1.1, 0.0], &[], 2, false) == 0 {
                        undo = true;
                        let n = a.route_end();
                        let at_n = Some((a.w.net.node_x[n as usize], a.w.net.node_y[n as usize]));
                        starts.push((3, [1.0, n as f64, 0.0, 0.0, 0.0], at_n));
                        starts.push((4, [1.0, n as f64, 0.0, 0.0, 0.0], at_n));
                    } else {
                        *refused.entry(a.issue_kinds()).or_insert(0) += 1;
                    }
                }
                for (kind, st, skip) in starts {
                    tried[kind] += 1;
                    a.set_route_beside(u32::MAX, e);
                    let mut ends = st.to_vec();
                    // the last start: the pointer drifted to the track's other side
                    let q = if kind == 4 { at(s2, -30.0 * side) } else { q };
                    ends.extend([0.0, 0.0, q.0, q.1, 0.0]);
                    let pv = a.preview_route(&ends, &[], 2, false);
                    if pv[0] != 1.0 {
                        *why.entry(format!("{}: {}", names[kind], a.issue_kinds())).or_insert(0) += 1;
                        continue;
                    }
                    ok[kind] += 1;
                    let nv = pv[3] as usize;
                    let line = &pv[5 + nv * 4..];
                    if run_pts.iter().any(|r| polylines_cross(line, r, skip)) {
                        crossed[kind] += 1;
                        if debug.as_deref() == Some(&format!("{kind}-{e}-{s}-{side}")) {
                            let nv = pv[3] as usize;
                            println!("route verts: {:?}", pv[5..5 + nv * 4].chunks(4).map(|c| { let l = a.to_lonlat(c[0], c[1]); (l[0], l[1], c[2]) }).collect::<Vec<_>>());
                            let ll = |v: &[f64]| v.chunks(3).map(|c| { let l = a.to_lonlat(c[0], c[1]); format!("[{:.6},{:.6}]", l[0], l[1]) }).collect::<Vec<_>>().join(",");
                            println!("route line: [{}]", ll(line));
                            for r in &run_pts {
                                println!("run piece: [{}]", ll(r));
                            }
                            println!("start {:?} pointer {:?}", a.to_lonlat(st[2], st[3]), a.to_lonlat(q.0, q.1));
                            println!("beside_ends: {:?}", a.beside_ends(&ends, e).map(|r| (r.1.len() / 4, r.2, r.3)));
                        }
                        if shown[kind] < 4 {
                            shown[kind] += 1;
                            let ll = a.to_lonlat(q.0, q.1);
                            examples.push(format!("{}: edge {e} s {s:.0} side {side} pointer {:.5},{:.5}", names[kind], ll[0], ll[1]));
                        }
                    }
                }
                if undo {
                    assert_eq!(a.undo(), 0);
                }
            }
            s += STEP;
        }
    }
    for k in 0..5 {
        println!("{:16} tried {:5} placed {:5} crossing the track followed {:4}", names[k], tried[k], ok[k], crossed[k]);
    }
    for x in &examples {
        println!("  {x}");
    }
    println!("loose ends not drawn: {refused:?}");
    for (k, n) in &why {
        println!("refused {n:5} {k}");
    }
}
