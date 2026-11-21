use super::*;

/// T-255: quad track is the drawn double track and an express double track beside it, end to
/// end; a station on one pair gets its twin on the other.
#[test]
fn quad_track_and_twin_stations() {
    let ends: Vec<f64> = free(0.0, 0.0).iter().chain(free(12000.0, 0.0).iter()).copied().collect();
    let pis = vec![3000.0, 400.0, 0.0, 0.0, 6000.0, 0.0, 0.0, 0.0, 9000.0, 400.0, 0.0, 0.0];
    let mut a = api();
    // continuing an existing track end, so the start becomes a junction
    let lead: Vec<f64> = free(-3000.0, 0.0).iter().chain(free(0.0, 0.0).iter()).copied().collect();
    assert_eq!(a.add_route(&lead, &[], 2, false), 0);
    let start = a.nodes_info().chunks(10).find(|c| c[1].abs() < 1e-6 && c[2].abs() < 1e-6).unwrap()[0];
    let mut quad_ends = ends.clone();
    quad_ends[..5].copy_from_slice(&[1.0, start, 0.0, 0.0, 0.0]);
    let pv = a.preview_route(&quad_ends, &pis, 4, false);
    assert_eq!(pv[0], 1.0, "{}", a.issues());
    assert_eq!(a.preview_render_speeds().len(), (pv.len() - 5 - 4 * pv[3] as usize) / 3 - 1);
    let before = a.total_cost();
    assert_eq!(a.add_route(&quad_ends, &pis, 4, false), 0, "{}", a.issues());
    let mut b = api();
    let (n0, n1, e) = (b.new_node_id(), b.new_node_id(), b.new_edge_id());
    assert_eq!(b.set_node(n0, 0.0, 0.0, 0, false) + b.set_node(n1, 12000.0, 0.0, 0, false) + b.set_edge(e, n0, n1, 2, &pis), 0);
    let on_local = |a: &TrackApi, p: u32| { let r = a.edge_render(p); let k = r.len() / 6 * 3; b.w.net.project(e, r[k], r[k + 1]).1 < 2.0 };
    let new: Vec<u32> = live_edges(&a).into_iter().filter(|&p| p != 0).collect();
    let (local, express) = (new[..3].to_vec(), new[3..].to_vec());
    assert!(local[1..].iter().all(|&p| on_local(&a, p)) && express.iter().all(|&p| !on_local(&a, p)));
    assert!(express.len() >= 3, "{express:?}");
    // beside it, 8 m to the left, away from the tapers at the ends
    let len = b.w.net.edge_len[e as usize];
    for &p in &express {
        for q in a.edge_render(p).chunks(3) {
            let (s, _) = b.w.net.project(e, q[0], q[1]);
            let off = local.iter().map(|&l| a.w.net.project(l, q[0], q[1]).1).fold(f64::INFINITY, f64::min);
            if s > QUAD_TAPER + 300.0 && s < len - QUAD_TAPER - 50.0 {
                assert!((off - QUAD_SPACING).abs() < 0.05, "express {p} {off} m off at {s}");
            }
        }
    }
    // two double tracks' price, and a junction where it leaves the old track
    let added = a.total_cost() - before;
    assert!((added / b.total_cost() - 2.0).abs() < 0.1, "{added} against {}", b.total_cost());
    let j = a.nodes_info().chunks(10).find(|c| c[0] == start).unwrap()[4];
    assert_eq!(j, 3.0, "the old track end is a junction now");
    // stations: on the pair clicked only (T-1687)
    let mut stops = vec![];
    for (piece, at, name) in [(local[1], 60.0, "First"), (local[2], 400.0, "Last")] {
        let p = piece; let (x, y, _) = geom::pos_at(a.w.net.edge_pieces(p), at);
        assert_eq!(a.add_station(2, p, x, y, 200, name.into()), 0, "{} on {p} at {:?}: {:?} {:?}", a.issues(), a.w.net.project(p, x, y), a.w.net.edge_data(p), a.edge_pis(p));
        let named: Vec<u32> = (0..a.w.net.node_count() as u32).filter(|&k| a.w.net.node_ok(k) && a.w.net.node_platform[k as usize] > 0 && a.w.net.node_name[k as usize] == name).collect();
        assert_eq!(named.len(), 1, "{name} alone");
        stops.push(named[0]);
    }
    // and on a joint node of the drawn pair
    let joint = (0..a.w.net.node_count() as u32).find(|&k| a.w.net.node_ok(k) && a.w.net.node_ports[k as usize].n == 2 && a.w.net.node_platform[k as usize] == 0 && b.w.net.project(e, a.w.net.node_x[k as usize], a.w.net.node_y[k as usize]).1 < 0.5 && (1000.0..11000.0).contains(&a.w.net.node_x[k as usize])).unwrap();
    assert_eq!(a.add_station(1, joint, 0.0, 0.0, 200, "Joint".into()), 0, "{}", a.issues());
    assert_eq!((0..a.w.net.node_count() as u32).filter(|&k| a.w.net.node_ok(k) && a.w.net.node_platform[k as usize] > 0 && a.w.net.node_name[k as usize] == "Joint").count(), 1);
    assert_eq!(a.undo(), 0);
    let id = a.new_line_id();
    assert_eq!(a.set_line_with_trains(id, &stops, 6, 4, 2, 30.0, 180.0, 0, 0, "Local".into()), 0, "{}", a.issues());
    assert!(a.line_path(id).iter().all(|&p| on_local(&a, p >> 1)));
    // deleting an express piece leaves the drawn pair
    assert_eq!(a.delete_edge(express[express.len() / 2]), 0, "{}", a.issues());
    assert!(!a.w.net.line_broken[id as usize]);
}

/// T-423: quad drawn through points. A chain of four segments carries both pairs on at every
/// joint (no swing, no junction: two loose ends at each end of the chain, plain joints
/// between, the express pair 8 m beside all the way, each local segment one arc with the
/// express arc concentric); a segment from the express pair's loose end carries both on the
/// other way; one ending on other track joins it as a junction; twin stations go across a
/// joint and on the loose ends; a crossover goes on it.
#[test]
fn quad_snapped_through_corners_carries_both_pairs_on() {
    // T-3261: a route through corners (the real-network or road snap's) on quad track ends both pairs apart on
    // open ground, and the next carries both on from there: no junction where they meet, four loose ends
    let mut a = api();
    a.set_route_tangent(true);
    let ends: Vec<f64> = free(0.0, 0.0).iter().chain(free(1500.0, 0.0).iter()).copied().collect();
    let pis = vec![500.0, 0.0, 0.0, 0.0, 1000.0, 200.0, 0.0, 0.0];
    assert_eq!(a.add_route(&ends, &pis, 4, false), 0, "{}", a.issues());
    let end = a.route_end();
    let (x0, y0) = (a.w.net.node_x[end as usize], a.w.net.node_y[end as usize]);
    let pis2 = vec![x0 + 600.0, y0, 0.0, 0.0, x0 + 1100.0, y0 - 300.0, 0.0, 0.0];
    assert_eq!(a.add_route(&from_node(end, x0 + 1800.0, y0 - 300.0), &pis2, 4, false), 0, "{}", a.issues());
    let nodes: Vec<u32> = (0..a.w.net.node_count() as u32).filter(|&k| a.w.net.node_ok(k)).collect();
    let ports: Vec<u8> = nodes.iter().map(|&k| a.w.net.node_ports[k as usize].n as u8).collect();
    assert!(ports.iter().all(|&p| p <= 2), "a junction where the pairs meet: {ports:?}");
    assert_eq!(ports.iter().filter(|&&p| p == 1).count(), 4, "{ports:?}");
}

#[test]
fn quad_drawn_through_points_carries_both_pairs_on() {
    let mut a = api();
    a.set_route_tangent(true);
    let net_nodes = |a: &TrackApi| (0..a.w.net.node_count() as u32).filter(|&k| a.w.net.node_ok(k)).collect::<Vec<u32>>();
    let nearest = |a: &TrackApi, x: f64, y: f64| -> u32 { net_nodes(a).into_iter().min_by(|&p, &q| { let d = |k: u32| (a.w.net.node_x[k as usize] - x).hypot(a.w.net.node_y[k as usize] - y); d(p).total_cmp(&d(q)) }).unwrap() };
    let ends: Vec<f64> = free(0.0, 0.0).iter().chain(free(600.0, 0.0).iter()).copied().collect();
    assert_eq!(a.add_route(&ends, &[], 4, false), 0, "{}", a.issues());
    assert_eq!(live_edges(&a).len(), 2);
    let (mut end, mut heading) = (a.route_end(), 0.0f64);
    let mut local = vec![live_edges(&a)[0]];
    let mut express = vec![live_edges(&a)[1]];
    for turn in [20.0f64, -15.0, 10.0] {
        let off = turn.to_radians();
        let (x0, y0) = (a.w.net.node_x[end as usize], a.w.net.node_y[end as usize]);
        let (x, y) = (x0 + 500.0 * (heading + off).cos(), y0 + 500.0 * (heading + off).sin());
        let before = live_edges(&a);
        assert_eq!(a.add_route(&from_node(end, x, y), &[], 4, false), 0, "{turn}: {}", a.issues());
        let new: Vec<u32> = live_edges(&a).into_iter().filter(|e| !before.contains(e)).collect();
        assert_eq!(new.len(), 2, "{turn}: one edge each pair");
        let (l, e) = if a.w.net.edge_b[new[0] as usize] == a.route_end() { (new[0], new[1]) } else { (new[1], new[0]) };
        // the local arc is the tangent arc, the express one concentric with it
        let want = 500.0 / (2.0 * off.abs().sin());
        let rl = a.edge_pis(l)[5];
        assert!((rl - want).abs() < 0.01 * want, "{turn}: local {rl} against {want}");
        let pe = a.edge_pis(e);
        assert_eq!(pe.len(), 9, "{turn}: one corner on the express pair too");
        let inside = turn > 0.0;
        assert!((pe[5] - if inside { rl - QUAD_SPACING } else { rl + QUAD_SPACING }).abs() < 0.5, "{turn}: express radius {} beside {rl}", pe[5]);
        local.push(l);
        express.push(e);
        heading += 2.0 * off;
        end = a.route_end();
    }
    // plain joints, no junctions; four loose ends
    let ports: Vec<u8> = net_nodes(&a).iter().map(|&k| a.w.net.node_ports[k as usize].n as u8).collect();
    assert!(ports.iter().all(|&p| p <= 2), "{ports:?}");
    assert_eq!(ports.iter().filter(|&&p| p == 1).count(), 4, "{ports:?}");
    for &p in &express {
        for q in a.edge_render(p).chunks(3) {
            let off = local.iter().map(|&l| a.w.net.project(l, q[0], q[1]).1).fold(f64::INFINITY, f64::min);
            assert!((off - QUAD_SPACING).abs() < 0.05, "express {p} {off} m off");
        }
    }
    // the express pair is to the left: its last loose end 8 m left of the local's
    let (ex, ey) = (a.w.net.node_x[end as usize] - QUAD_SPACING * heading.sin(), a.w.net.node_y[end as usize] + QUAD_SPACING * heading.cos());
    let ex_end = nearest(&a, ex, ey);
    assert!((a.w.net.node_x[ex_end as usize] - ex).hypot(a.w.net.node_y[ex_end as usize] - ey) < 0.05);
    // twin stations: across the first joint, and on the far loose ends
    let (sx, sy, _) = geom::pos_at(a.w.net.edge_pieces(local[0]), a.w.net.edge_len[local[0] as usize] - 40.0);
    assert_eq!(a.add_station(2, local[0], sx, sy, 200, "Joint".into()), 0, "{}", a.issues());
    let named = |a: &TrackApi, name: &str| net_nodes(a).into_iter().filter(|&k| a.w.net.node_platform[k as usize] > 0 && a.w.net.node_name[k as usize] == name).count();
    assert_eq!(named(&a, "Joint"), 1, "the pair clicked only (T-1687)");
    assert_eq!(a.add_station(1, end, 0.0, 0.0, 120, "End".into()), 0, "{}", a.issues());
    assert_eq!(named(&a, "End"), 1);
    // the express pair's loose end, clicked next: its own station, named as the one beside
    assert_eq!(a.add_station(1, ex_end, 0.0, 0.0, 120, "Other".into()), 0, "{}", a.issues());
    assert_eq!(named(&a, "End"), 2, "the loose ends both");
    assert!(a.w.net.node_platform[ex_end as usize] > 0);
    for _ in 0..3 {
        assert_eq!(a.undo(), 0);
    }
    // a crossover on the new quad track, mid-segment
    let mid = local[2];
    let (cx, cy, _) = geom::pos_at(a.w.net.edge_pieces(mid), a.w.net.edge_len[mid as usize] / 2.0);
    assert_eq!(a.add_crossover(mid, cx, cy), 0, "{}", a.issues());
    assert_eq!(a.undo(), 0);
    // from the express pair's loose end at the start, the other way: both pairs carry on
    let ex0 = nearest(&a, 0.0, QUAD_SPACING);
    let before = live_edges(&a);
    assert_eq!(a.add_route(&from_node(ex0, -500.0, -150.0), &[], 4, false), 0, "{}", a.issues());
    assert_eq!(live_edges(&a).len(), before.len() + 2);
    assert!(net_nodes(&a).iter().all(|&k| a.w.net.node_ports[k as usize].n != 1 || a.w.net.node_x[k as usize] < -100.0 || a.w.net.node_x[k as usize] > 100.0), "no loose end left at the start");
    // ending on a plain double track ahead: both pairs meet it at one new junction
    let (x0, y0) = (a.w.net.node_x[end as usize], a.w.net.node_y[end as usize]);
    let (ux, uy) = (heading.cos(), heading.sin());
    // the ghost of a segment from the loose end: both pairs, no step drawn between their ends
    let pv = a.preview_route(&from_node(end, x0 + 400.0 * (heading + 0.1).cos(), y0 + 400.0 * (heading + 0.1).sin()), &[], 4, false);
    assert_eq!(pv[0], 1.0, "{}", a.issues());
    let sp = a.preview_render_speeds();
    assert_eq!(sp.len(), (pv.len() - 5 - 4 * pv[3] as usize) / 3 - 1);
    assert_eq!(sp.iter().filter(|&&v| v < 0.0).count(), 1);
    assert_eq!(a.route_arc().len(), 3, "the local arc's circle");
    let (px, py) = (x0 + 700.0 * ux, y0 + 700.0 * uy);
    let cross: Vec<f64> = free(px + 1500.0 * uy, py - 1500.0 * ux).iter().chain(free(px - 1500.0 * uy, py + 1500.0 * ux).iter()).copied().collect();
    assert_eq!(a.add_route(&cross, &[], 2, false), 0, "{}", a.issues());
    let other = *live_edges(&a).last().unwrap();
    let (tx, ty) = (px - 200.0 * uy, py + 200.0 * ux);
    let to = vec![1.0, end as f64, 0.0, 0.0, 0.0, 2.0, other as f64, tx, ty, 0.0];
    assert_eq!(a.add_route(&to, &[], 4, false), 0, "{}", a.issues());
    let meet = a.route_end();
    assert_eq!(a.w.net.node_ports[meet as usize].n, 4, "the crossed track, the local and the express pair");
    assert_eq!(net_nodes(&a).iter().filter(|&&k| a.w.net.node_ports[k as usize].n == 1).count(), 4, "the new start's two and the crossed track's two");
}

/// T-338: a crossover on quad track joins the two pairs mid-stretch, so a line can change from
/// the local pair to the express pair there; it runs at 50 km/h at most over it, is priced as
/// two junctions, has a fixed shape, saves, deletes as a whole and undoes in one step. Track
/// with nothing beside it is refused.
#[test]
fn crossover_joins_the_pairs() {
    let mut a = api();
    let ends: Vec<f64> = free(0.0, 0.0).iter().chain(free(4000.0, 0.0).iter()).copied().collect();
    assert_eq!(a.add_route(&ends, &[], 4, false), 0, "{}", a.issues());
    let edge_at = |t: &TrackApi, x: f64, y: f64| -> u32 {
        live_edges(t).into_iter().filter(|&e| !t.w.net.edge_crossover[e as usize]).min_by(|&p, &q| t.w.net.project(p, x, y).1.total_cmp(&t.w.net.project(q, x, y).1)).unwrap()
    };
    let station_at = |t: &TrackApi, x: f64, y: f64| -> u32 {
        t.nodes_info().chunks(10).filter(|c| c[7] > 0.0).min_by(|p, q| (p[1] - x).hypot(p[2] - y).total_cmp(&(q[1] - x).hypot(q[2] - y))).unwrap()[0] as u32
    };
    for (x, y) in [(500.0, 0.0), (3500.0, QUAD_SPACING)] {
        assert_eq!(a.add_station(2, edge_at(&a, x, y), x, y, 120, format!("S{x}")), 0, "{}", a.issues());
    }
    let (w_local, e_express) = (station_at(&a, 500.0, 0.0), station_at(&a, 3500.0, QUAD_SPACING));
    assert!((a.w.net.node_y[e_express as usize] - QUAD_SPACING).abs() < 0.1);
    let id = a.new_line_id();
    assert_ne!(a.set_line_with_trains(id, &[w_local, e_express], 6, 4, 2, 30.0, 180.0, 0, 0, "X".into()), 0, "no way across yet");
    // nothing beside plain double track
    let mut plain = api();
    assert_eq!(plain.add_route(&ends, &[], 2, false), 0);
    let only = live_edges(&plain)[0];
    assert_eq!(plain.preview_crossover(only, 2000.0, 0.0)[0], 0.0);
    assert_eq!(plain.issue_kinds(), "NoParallel");
    assert_ne!(plain.add_crossover(only, 2000.0, 0.0), 0);
    // on the local pair, in the middle
    let local = edge_at(&a, 2000.0, 0.0);
    let (edges0, cost0, cash0) = (live_edges(&a).len(), a.total_cost(), a.cash);
    let pv = a.preview_crossover(local, 2000.0, 0.0);
    assert_eq!(pv[0], 1.0, "{}", a.issues());
    let two_junctions = 2.0 * super::super::super::cost::junction_cost(0, false, false);
    assert!((pv[1] - two_junctions).abs() < 1e-6, "priced {} not {two_junctions}", pv[1]);
    assert_eq!(live_edges(&a).len(), edges0, "a preview changes nothing");
    assert_eq!(a.add_crossover(local, 2000.0, 0.0), 0, "{}", a.issues());
    assert!((a.total_cost() - cost0 - two_junctions).abs() < 1e-6);
    assert_eq!(a.cash, cash0, "a blueprint is free until constructed");
    let xs = a.crossover_edges();
    assert_eq!(xs.len(), 2);
    assert_eq!(live_edges(&a).len(), edges0 + 6, "four splits and two diagonals");
    for &x in &xs {
        let len = a.w.net.edge_len[x as usize];
        assert!(len > 110.0 && len < 130.0, "diagonal {len} m");
        assert!(a.w.net.edge_time[x as usize] >= len / V_CROSSOVER - 1e-6);
        assert_eq!(crossover_partner(&a.w.net, x), xs.iter().copied().find(|&o| o != x));
    }
    // a line from the local pair to the express pair now runs across it, at 50 km/h at most
    assert_eq!(a.set_line_with_trains(id, &[w_local, e_express], 6, 4, 2, 30.0, 180.0, 0, 0, "X".into()), 0, "{}", a.issues());
    let path = a.line_path(id);
    let over: Vec<u32> = path.iter().map(|p| p >> 1).filter(|e| xs.contains(e)).collect();
    assert_eq!(over.len(), 1, "one diagonal: {path:?}");
    let run = &a.w.svc.lines[id as usize].runs[0];
    let k = run.segs.iter().position(|&p| p >> 1 == over[0]).unwrap();
    let (s0, s1) = (run.seg_off[k], run.seg_off[k] + a.w.net.edge_len[over[0] as usize]);
    let v = profile::v_max(&run.free.phases, s0, s1);
    assert!(v <= V_CROSSOVER + 1e-6, "{} km/h over the crossover", v * 3.6);
    // fixed shape
    assert_ne!(a.set_edge_pis(xs[0], &[]), 0);
    assert_eq!(a.issue_kinds(), "Crossover");
    let n = a.w.net.edge_a[xs[0] as usize];
    assert_ne!(a.move_node(n, a.w.net.node_x[n as usize] + 5.0, a.w.net.node_y[n as usize]), 0);
    // saved and loaded with the flag
    let bytes = a.save();
    let mut b = api();
    assert!(b.load(&bytes));
    assert_eq!(b.crossover_edges().len(), 2);
    assert!((b.total_cost() - a.total_cost()).abs() < 1e-6);
    // deleting one diagonal takes the whole crossover; the line goes back to no route
    assert_eq!(a.remove_line(id), 0);
    assert_eq!(a.delete_edge(xs[0]), 0, "{}", a.issues());
    assert!(a.crossover_edges().is_empty());
    assert_eq!(a.undo(), 0);
    assert_eq!(a.crossover_edges().len(), 2);
    // constructing one diagonal builds both (T-391)
    assert_eq!(b.crossover_partners(), vec![xs[1], xs[0]]);
    b.cash = 1e6;
    assert_eq!(b.construct(&[xs[0]], &[]), 0, "{}", b.issues());
    assert!(xs.iter().all(|&x| b.w.net.edge_built[x as usize]));
    // undoing the placement takes it all away
    assert_eq!(a.undo(), 0); // the line's removal
    assert_eq!(a.undo(), 0); // the line
    assert_eq!(a.undo(), 0); // the crossover
    assert!(a.crossover_edges().is_empty());
    assert_eq!(live_edges(&a).len(), edges0);
    assert!((a.total_cost() - cost0).abs() < 1e-6);
}

/// T-244: dragging a corner of a piece moves the joints next to it along, so the stroke
/// reshapes as if it were still one edge.
#[test]
fn reshaping_a_piece_moves_its_joints() {
    let (ends, pis) = stroke(&|_| 0);
    let mut a = api();
    assert_eq!(a.add_route(&ends, &pis, 2, false), 0, "{}", a.issues());
    let edges = live_edges(&a);
    let mid = edges[1];
    let moved = [2700.0, -900.0, 0.0, 0.0];
    let pv = a.preview_edge(mid, &moved);
    assert_eq!(pv[0], 1.0, "{}", a.issues());
    assert_eq!(a.preview_render_speeds().len(), (pv.len() - 5 - 4 * pv[3] as usize) / 3 - 1);
    assert_eq!(a.set_edge_pis(mid, &moved), 0, "{}", a.issues());
    let mut whole = pis.clone();
    whole[4..8].copy_from_slice(&moved);
    let (b, e) = one_edge(&whole);
    same_track(&a, &b, e, 0.01);
    assert_eq!(live_edges(&a), edges, "the same pieces");
    assert_eq!(a.undo(), 0);
    let (b, e) = one_edge(&pis);
    same_track(&a, &b, e, 0.005);
    // a joint drags like any blueprint node, keeping its heading
    let j = a.w.net.edge_b[edges[0] as usize];
    let (x, y) = (a.w.net.node_x[j as usize], a.w.net.node_y[j as usize]);
    assert_eq!(a.move_node(j, x + 40.0, y + 60.0), 0, "{}", a.issues());
    // a line still routes over the pieces through the joints
    let end = |t: &TrackApi, x: f64| t.nodes_info().chunks(10).find(|c| (c[1] - x).abs() < 1e-6).unwrap()[0] as u32;
    let (w, east) = (end(&a, 0.0), end(&a, 7600.0));
    assert_eq!(a.add_station(1, w, 0.0, 0.0, 200, "West".into()), 0);
    assert_eq!(a.add_station(1, east, 0.0, 0.0, 200, "East".into()), 0);
    let id = a.new_line_id();
    assert_eq!(a.set_line_with_trains(id, &[w, east], 6, 4, 2, 30.0, 180.0, 0, 0, "L".into()), 0, "{}", a.issues());
    assert_eq!(a.line_path(id).len(), 4);
}

/// T-1161: removing the piece past a station on the track's last stretch stops at the platform's
/// far end, so the station stays where it was; between two stations both keep theirs.
#[test]
fn removing_past_a_station_stops_at_its_platform_end() {
    for built in [false, true] {
        let mut a = api();
        a.set_cash(1e6);
        let ends: Vec<f64> = free(0.0, 0.0).iter().chain(free(4000.0, 0.0).iter()).copied().collect();
        assert_eq!(a.add_route(&ends, &[], 2, false), 0, "{}", a.issues());
        for x in [1000.0, 3000.0] {
            let e = (0..a.w.net.edge_count() as u32).find(|&e| {
                let n = &a.w.net;
                n.edge_ok(e) && n.node_x[n.edge_a[e as usize] as usize] < x && n.node_x[n.edge_b[e as usize] as usize] > x
            }).unwrap();
            assert_eq!(a.add_station(2, e, x, 0.0, 200, format!("S{x}")), 0, "{}", a.issues());
        }
        if built {
            assert_eq!(a.construct_all(), 0, "{}", a.issues());
        }
        let station = |t: &TrackApi, x: f64| (0..t.w.net.node_count() as u32).find(|&n| t.w.net.node_ok(n) && t.w.net.node_platform[n as usize] > 0 && (t.w.net.node_x[n as usize] - x).abs() < 1e-6).unwrap();
        let edge_between = |t: &TrackApi, p: u32, q: u32| live_edges(t).into_iter().find(|&e| {
            let (ea, eb) = (t.w.net.edge_a[e as usize], t.w.net.edge_b[e as usize]);
            (ea == p && eb == q) || (ea == q && eb == p)
        }).unwrap();
        let loose = |t: &TrackApi| -> Vec<f64> {
            let mut v: Vec<f64> = (0..t.w.net.node_count() as u32).filter(|&n| t.w.net.node_ok(n) && t.w.net.node_ports[n as usize].n == 1).map(|n| t.w.net.node_x[n as usize]).collect();
            v.sort_by(f64::total_cmp);
            v
        };
        let (s1, s3) = (station(&a, 1000.0), station(&a, 3000.0));
        let far = (0..a.w.net.node_count() as u32).find(|&n| a.w.net.node_ok(n) && a.w.net.node_x[n as usize] == 4000.0).unwrap();
        // the last stretch, past the station at 3 km
        assert_eq!(a.delete_edge(edge_between(&a, s3, far)), 0, "{}", a.issues());
        assert_eq!(a.w.net.node_x[s3 as usize], 3000.0, "the station keeps its place");
        assert_eq!(a.w.net.node_ports[s3 as usize].n, 2);
        let ends = loose(&a);
        assert_eq!(ends.len(), 2, "{ends:?}");
        let past = ends[1] - 3000.0 - 100.0;
        assert!(past > 1.0 && past <= super::links::STATION_TRIM_MARGIN_M + 0.01, "track ends {past} m past the platform");
        // between the two stations: both stay, with the platform's length of track past each
        let mid = edge_between(&a, s1, s3);
        assert_eq!(a.delete_edge(mid), 0, "{}", a.issues());
        assert_eq!((a.w.net.node_x[s1 as usize], a.w.net.node_x[s3 as usize]), (1000.0, 3000.0));
        assert_eq!((a.w.net.node_ports[s1 as usize].n, a.w.net.node_ports[s3 as usize].n), (2, 2));
        let ends = loose(&a);
        assert_eq!(ends.len(), 4, "{ends:?}");
        assert!(ends[1] > 1101.0 && ends[1] <= 1104.01 && ends[2] < 2899.0 && ends[2] >= 2895.99, "{ends:?}");
        if !built {
            // undone in two steps, back to the whole track
            assert_eq!(a.undo(), 0);
            assert_eq!(a.undo(), 0);
            assert_eq!(loose(&a), [0.0, 4000.0]);
        }
    }
}

/// T-3648: a set removed in one edit leaves the track that removing its pieces one by one leaves.
#[test]
fn removing_a_set_leaves_what_one_by_one_leaves() {
    for (built, xs) in [(false, vec![2000.0]), (true, vec![2000.0]), (false, vec![1000.0, 3000.0]), (true, vec![1000.0, 3000.0])] {
        let setup = || {
            let mut a = api();
            a.set_cash(1e6);
            let ends: Vec<f64> = free(0.0, 0.0).iter().chain(free(4000.0, 0.0).iter()).copied().collect();
            assert_eq!(a.add_route(&ends, &[], 2, false), 0, "{}", a.issues());
            for &x in &xs {
                let e = (0..a.w.net.edge_count() as u32).find(|&e| {
                    let n = &a.w.net;
                    n.edge_ok(e) && n.node_x[n.edge_a[e as usize] as usize] < x && n.node_x[n.edge_b[e as usize] as usize] > x
                }).unwrap();
                assert_eq!(a.add_station(2, e, x, 0.0, 200, format!("S{x}")), 0, "{}", a.issues());
            }
            if built {
                assert_eq!(a.construct_all(), 0, "{}", a.issues());
            }
            a
        };
        // what is left: each piece's x span and length, sorted
        let left = |t: &TrackApi| -> Vec<(i64, i64)> {
            let n = &t.w.net;
            let mut v: Vec<(i64, i64)> = live_edges(t).into_iter().map(|e| {
                let (xa, xb) = (n.node_x[n.edge_a[e as usize] as usize], n.node_x[n.edge_b[e as usize] as usize]);
                (xa.min(xb).round() as i64, xa.max(xb).round() as i64)
            }).collect();
            v.sort();
            v
        };
        let mut a = setup();
        let ids = live_edges(&a);
        assert_eq!(a.delete_pieces(&ids, &[]), 0, "{}", a.issues());
        let set = left(&a);
        let mut b = setup();
        for e in live_edges(&b) {
            if b.w.net.edge_ok(e) {
                b.delete_edge(e);
            }
        }
        // the pieces not yet touched are gone by id; the one-by-one run may keep what a trim left
        let one = left(&b);
        assert_eq!(set, one, "built {built}, stations {xs:?}");
    }
}

/// T-1667: track drawn on from a station at a dead end, at either end of its edge, leaves the
/// platform where it was: the station's node moves back to the platform's middle, keeping its id
/// and its line, and the new track starts from the old end.
#[test]
fn drawing_on_from_a_dead_end_station_keeps_its_platform() {
    for (built, from_a) in [(false, false), (true, false), (false, true), (true, true)] {
        let mut a = api();
        a.set_cash(1e6);
        let ends: Vec<f64> = free(0.0, 0.0).iter().chain(free(1000.0, 0.0).iter()).copied().collect();
        assert_eq!(a.add_route(&ends, &[], 2, false), 0, "{}", a.issues());
        let node_at = |t: &TrackApi, x: f64| (0..t.w.net.node_count() as u32).find(|&n| t.w.net.node_ok(n) && (t.w.net.node_x[n as usize] - x).abs() < 1e-6).unwrap();
        let (s0, s1) = (node_at(&a, 0.0), node_at(&a, 1000.0));
        assert_eq!(a.add_station(1, s0, 0.0, 0.0, 120, "West".into()), 0, "{}", a.issues());
        assert_eq!(a.add_station(1, s1, 0.0, 0.0, 120, "East".into()), 0, "{}", a.issues());
        let l = a.new_line_id();
        assert_eq!(a.set_line_with_trains(l, &[s0, s1], 2, 0, 0, 30.0, 180.0, 0, 0xd7263d, "Line".into()), 0, "{}", a.issues());
        if built {
            assert_eq!(a.construct_all(), 0, "{}", a.issues());
        }
        let (st, x0, far) = if from_a { (s0, 0.0, -1500.0) } else { (s1, 1000.0, 2500.0) };
        let e: Vec<f64> = [1.0, st as f64, 0.0, 0.0, 0.0].iter().chain(free(far, 400.0).iter()).copied().collect();
        assert_eq!(a.add_route(&e, &[], 2, false), 0, "{}", a.issues());
        let back = if from_a { 60.0 } else { -60.0 };
        assert!((a.w.net.node_x[st as usize] - (x0 + back)).abs() < 2.01, "station at {}", a.w.net.node_x[st as usize]);
        assert_eq!(a.w.net.node_ports[st as usize].n, 2);
        assert_eq!(a.w.net.station_data(st).unwrap().built, built);
        let old_end = node_at(&a, x0 - back / 30.0);
        assert_eq!((a.w.net.node_ports[old_end as usize].n, a.w.net.node_platform[old_end as usize]), (2, 0), "the route starts 2 m past the old end");
        assert_eq!(a.line_stops(l), [s0, s1]);
        assert!(!a.w.net.line_broken[l as usize]);
        let nodes: Vec<u32> = (0..a.w.net.node_count() as u32).filter(|&n| a.w.net.node_ok(n)).collect();
        let bad = a.w.net.validate(&live_edges(&a), &nodes, &[]);
        assert!(bad.is_empty(), "{bad:?}");
        assert_eq!(a.undo(), 0);
        assert_eq!((a.w.net.node_x[st as usize], a.w.net.node_ports[st as usize].n), (x0, 1), "one undo step");
    }
}

/// T-1805: blueprint off two lines' shared constructed track, branching mid-edge or crossing it
/// at grade, changes no train's run, so nothing goes to demand.
#[test]
fn blueprint_beside_shared_track_wakes_no_demand() {
    let mut a = api();
    a.set_cash(1e6);
    let ends: Vec<f64> = free(0.0, 0.0).iter().chain(free(6000.0, 0.0).iter()).copied().collect();
    assert_eq!(a.add_route(&ends, &[], 2, false), 0, "{}", a.issues());
    let nodes = a.nodes_info();
    let (n0, n1) = (nodes[0] as u32, nodes[10] as u32);
    assert_eq!(a.add_station(1, n0, 0.0, 0.0, 200, "West".into()), 0, "{}", a.issues());
    assert_eq!(a.add_station(1, n1, 0.0, 0.0, 200, "East".into()), 0, "{}", a.issues());
    let l4 = a.new_line_id();
    assert_eq!(a.set_line_with_trains(l4, &[n0, n1], 8, 6, 4, 30.0, 180.0, 0, 0x00933c, "4".into()), 0);
    let l5 = a.new_line_id();
    assert_eq!(a.set_line_with_trains(l5, &[n0, n1], 8, 6, 4, 30.0, 180.0, 0, 0x00933c, "5".into()), 0);
    assert_eq!(a.construct_line(l4), 0, "{}", a.issues());
    let times = |a: &TrackApi| [a.stop_times(l4, 0, 0), a.stop_times(l5, 0, 0), a.stop_times(l4, 1, 1)];
    let before = times(&a);
    let edge = a.edges_info().chunks(10).find(|c| c[5] > 5000.0).unwrap()[0];
    // a branch from the middle of the shared edge, splitting it
    let branch = [2.0, edge, 3000.0, 0.0, 0.0, 0.0, 0.0, 4500.0, 900.0, 0.0];
    assert_eq!(a.add_route(&branch, &[], 2, false), 0, "{}", a.issues());
    assert!(a.dirty_demand_lines().is_empty(), "{:?}", a.dirty_demand_lines());
    assert_eq!(times(&a), before);
    // a blueprint across the track at grade
    let across: Vec<f64> = free(1500.0, -800.0).iter().chain(free(1500.0, 800.0).iter()).copied().collect();
    assert_eq!(a.add_route(&across, &[], 2, false), 0, "{}", a.issues());
    assert!(!a.w.net.crossings.is_empty());
    assert!(a.dirty_demand_lines().is_empty(), "{:?}", a.dirty_demand_lines());
    assert_eq!(times(&a), before);
    // and the same track constructed, with no line on it
    assert_eq!(a.construct_all(), 0, "{}", a.issues());
    assert!(a.dirty_demand_lines().is_empty(), "{:?}", a.dirty_demand_lines());
    assert_eq!(times(&a), before);
}
