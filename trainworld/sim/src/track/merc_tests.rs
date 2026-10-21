//! The Mercator world frame through the track model (T-306, docs/world.md): rhumb-line lengths,
//! the same track near New York in the conic and in Mercator, London and New York in one save,
//! TWT6, converting a conic save, and the refusals at the plane's edges.

use super::frame::{great_circle, merc_k, FrameKind, MERC_MAX_LAT};
use super::geom::{self, rhumb_ground, rhumb_plane, Pi};
use super::net::{EdgeData, Issue, Network, NodeData};
use super::params::*;
use super::tests::{station, line};
use super::world::{Op, TrackWorld};

fn world_in(kind: FrameKind) -> TrackWorld {
    TrackWorld::new(Network::new_in(kind, -73.985, 40.758, true))
}

/// A node at a longitude and latitude.
fn place(w: &mut TrackWorld, lon: f64, lat: f64) -> u32 {
    let (x, y) = w.net.frame().from_lonlat(lon, lat);
    let id = w.net.alloc_node();
    w.apply(Op::Node { id, data: Some(NodeData { x: quant(x), y: quant(y), level: 0, flying: false }) }).unwrap();
    id
}

/// An edge through PIs given as (lon, lat, ground radius or 0 for auto).
fn edge_ll(w: &mut TrackWorld, a: u32, b: u32, pis: &[(f64, f64, f64)], level: i8) -> Result<u32, Vec<Issue>> {
    let f = w.net.frame();
    let pis = pis
        .iter()
        .map(|&(lon, lat, r)| {
            let (x, y) = f.from_lonlat(lon, lat);
            Pi::new(x, y, r, level).quantised()
        })
        .collect();
    let id = w.net.alloc_edge();
    w.apply(Op::Edge { id, data: Some(EdgeData { a, b, tracks: 2, pis, built: true, off: false, thru: vec![], crossover: false, hsr: false }) })?;
    Ok(id)
}

#[test]
fn rhumb_lengths_are_exact_and_invert() {
    let r = EARTH_R;
    // North-south: R times the latitude change, at any latitude.
    for lat0 in [-60.0f64, 0.0, 19.07, 40.758, 51.5] {
        let (y0, y1) = (r * lat0.to_radians().sin().atanh(), r * (lat0 + 2.0).to_radians().sin().atanh());
        let g = rhumb_ground(y0, 1.0, y1 - y0);
        assert!((g - r * 2f64.to_radians()).abs() < 1e-6, "{lat0}: {g}");
        assert!((rhumb_plane(lat0.to_radians(), 1.0, g) - (y1 - y0)).abs() < 1e-6);
    }
    // East-west: plane over k, exactly, through the u = 0 branch and a hair off it.
    let y = r * 51.5f64.to_radians().sin().atanh();
    for u in [0.0, 1e-13, 1e-9, 1e-5] {
        let g = rhumb_ground(y, u, 10000.0);
        assert!((g - 10000.0 / merc_k(y)).abs() < 1e-3, "u {u}: {g}");
    }
    // Any bearing: against a fine numerical integral of plane metres over k.
    for (lat0, th) in [(51.5f64, 0.7f64), (40.758, -2.1), (19.07, 1.2), (-33.9, 2.9)] {
        let (y0, u) = (r * lat0.to_radians().sin().atanh(), th.sin());
        let d = 300_000.0;
        let n = 200_000;
        let num: f64 = (0..n).map(|i| (d / n as f64) / merc_k(y0 + (i as f64 + 0.5) * d / n as f64 * u)).sum();
        let g = rhumb_ground(y0, u, d);
        assert!((g / num - 1.0).abs() < 1e-9, "{lat0} {th}: {g} vs {num}");
        assert!((rhumb_plane(lat0.to_radians(), u, g) - d).abs() < 1e-6);
    }
}

/// Penn Station to Jamaica and on, with curves and a crossing line: the same lon/lat inputs in the
/// conic (T-125) and in Mercator.
fn nyc_network(kind: FrameKind) -> (TrackWorld, Vec<u32>, Vec<u32>) {
    let mut w = world_in(kind);
    let penn = place(&mut w, -73.9935, 40.7506);
    let jam = place(&mut w, -73.8080, 40.6995);
    let north = place(&mut w, -73.9150, 40.7800);
    let south = place(&mut w, -73.9000, 40.7100);
    let main = &[(-73.9700, 40.7400, 0.0), (-73.9400, 40.7470, 0.0), (-73.9030, 40.7450, 0.0), (-73.8600, 40.7300, 1200.0), (-73.8300, 40.7000, 0.0)];
    let mut edges = vec![edge_ll(&mut w, penn, jam, main, 0).unwrap()];
    // a north-south line crossing the first one on the level (a flat crossing)
    edges.push(edge_ll(&mut w, north, south, &[(-73.9080, 40.7480, 0.0)], 0).unwrap());
    for n in [penn, jam, north, south] {
        station(&mut w, n, 200);
    }
    let lines = vec![line(&mut w, &[penn, jam], [12.0, 6.0, 3.0]), line(&mut w, &[north, south], [8.0, 4.0, 2.0])];
    (w, edges, lines)
}

#[test]
fn track_near_new_york_is_the_same_in_mercator_as_in_the_conic() {
    let (lcc, edges, lines) = nyc_network(FrameKind::Lcc);
    let (merc, _, _) = nyc_network(FrameKind::Merc);
    let rel = |a: f64, b: f64| (a / b - 1.0).abs();
    for &e in &edges {
        let (i, a, b) = (e as usize, &lcc.net, &merc.net);
        println!(
            "edge {e}: {:.2} m conic, {:.2} m Mercator ({:+.4}%); cost {:+.4}%; time {:+.4}%",
            a.edge_len[i],
            b.edge_len[i],
            (b.edge_len[i] / a.edge_len[i] - 1.0) * 100.0,
            (b.edge_cost[i] / a.edge_cost[i] - 1.0) * 100.0,
            (b.edge_time[i] / a.edge_time[i] - 1.0) * 100.0
        );
        // The conic's own scale here is under 1 + 1e-5; the rest is the rhumb line against the
        // conic's straight and the millimetre grid.
        assert!(rel(b.edge_len[i], a.edge_len[i]) < 1e-4, "edge {e} length");
        assert!(rel(b.edge_cost[i], a.edge_cost[i]) < 1e-4, "edge {e} cost");
        assert!(rel(b.edge_time[i], a.edge_time[i]) < 1e-4, "edge {e} time");
    }
    for &l in &lines {
        let (a, b) = (&lcc.svc.lines[l as usize], &merc.svc.lines[l as usize]);
        assert!(a.ok && b.ok);
        for lev in 0..LEVELS {
            assert!(rel(b.round_trip[lev], a.round_trip[lev]) < 1e-4, "line {l} round trip");
            assert_eq!(a.trains[lev], b.trains[lev], "line {l} fleet");
        }
    }
    // The same crossings, of the same kinds, at the same chainages to a few centimetres.
    assert_eq!(lcc.net.crossings.len(), merc.net.crossings.len());
    assert!(!lcc.net.crossings.is_empty());
    for (a, b) in lcc.net.crossings.iter().zip(&merc.net.crossings) {
        assert_eq!((a.e1, a.e2), (b.e1, b.e2));
        assert_eq!(a.kind, b.kind);
        assert!((a.s1 - b.s1).abs() < 0.2 && (a.s2 - b.s2).abs() < 0.2, "{} {} vs {} {}", a.s1, a.s2, b.s1, b.s2);
    }
    // The curve set to 1,200 m is 1,200 m on the ground in both.
    for w in [&lcc, &merc] {
        let r = w.net.edge_pieces(edges[0]).iter().filter(|p| p.k != 0.0).map(|p| p.ground_radius()).min_by(|a, b| (a - 1200.0).abs().total_cmp(&(b - 1200.0).abs())).unwrap();
        assert!((r - 1200.0).abs() < 0.5, "{:?}: {r}", w.net.frame_kind);
    }
}

/// The radius of the circle through three ground points (lon, lat), metres.
fn ground_radius(a: (f64, f64), b: (f64, f64), c: (f64, f64)) -> f64 {
    let (ab, bc, ca) = (great_circle(a.0, a.1, b.0, b.1), great_circle(b.0, b.1, c.0, c.1), great_circle(c.0, c.1, a.0, a.1));
    let s = (ab + bc + ca) / 2.0;
    ab * bc * ca / (4.0 * (s * (s - ab) * (s - bc) * (s - ca)).sqrt())
}

#[test]
fn london_and_new_york_share_one_save_with_true_lengths_and_curves() {
    let (mut w, nyc_edges, nyc_lines) = nyc_network(FrameKind::Merc);
    let nyc_len: Vec<f64> = nyc_edges.iter().map(|&e| w.net.edge_len[e as usize]).collect();
    // London: King's Cross to Stratford by a 1,500 m curve near Highbury, then on to Canary Wharf.
    let kx = place(&mut w, -0.1240, 51.5308);
    let st = place(&mut w, -0.0035, 51.5413);
    let (sf, cw) = (place(&mut w, -0.0080, 51.5450), place(&mut w, -0.0195, 51.5035));
    let e1 = edge_ll(&mut w, kx, st, &[(-0.0900, 51.5480, 1500.0)], 0).unwrap();
    let e2 = edge_ll(&mut w, sf, cw, &[], 0).unwrap();
    for n in [kx, st, sf, cw] {
        station(&mut w, n, 200);
    }
    let ll = line(&mut w, &[kx, st], [12.0, 6.0, 3.0]);
    // New York is untouched by London being built.
    for (k, &e) in nyc_edges.iter().enumerate() {
        assert_eq!(w.net.edge_len[e as usize], nyc_len[k]);
    }
    let f = w.net.frame();
    // A straight is a rhumb line: its length is the great circle's to a few millimetres per km
    // over a few km.
    let gc = great_circle(-0.0080, 51.5450, -0.0195, 51.5035);
    let len = w.net.edge_len[e2 as usize];
    println!("Stratford - Canary Wharf: {len:.3} m, great circle {gc:.3} m");
    assert!((len / gc - 1.0).abs() < 1e-5);
    // The curve is 1,500 m on the ground (not 1,500 plane metres, which would be 934 m here),
    // and its speed limit is the 1,500 m one.
    let arc = *w.net.edge_pieces(e1).iter().find(|p| p.k != 0.0).unwrap();
    let pt = |s: f64| {
        let (x, y, _) = arc.at(arc.len * s);
        f.to_lonlat(x, y)
    };
    let measured = ground_radius(pt(0.0), pt(0.5), pt(1.0));
    println!("London curve: ground radius {measured:.2} m, plane radius {:.1}, k {:.4}", arc.radius(), arc.g);
    // A circle in a conformal plane is a circle on the ground up to the plane straight's own
    // curvature on the ground (a rhumb line bends by up to tan(lat) / R, 2e-7 per metre here):
    // 0.03% on a 1,500 m curve, sub-metre.
    assert!((measured / 1500.0 - 1.0).abs() < 5e-4, "{measured}");
    assert!((arc.ground_radius() / 1500.0 - 1.0).abs() < 2e-4);
    assert!((arc.v_limit() / geom::curve_speed(1500.0) - 1.0).abs() < 1e-4);
    // The edge as a whole against the ground, sampled every metre.
    let mut ground = 0.0;
    let mut prev;
    let pcs = w.net.edge_pieces(e1).to_vec();
    let total = w.net.edge_len[e1 as usize];
    prev = { let (x, y, _) = geom::pos_at(&pcs, 0.0); f.to_lonlat(x, y) };
    let n = total.ceil() as usize;
    for i in 1..=n {
        let (x, y, _) = geom::pos_at(&pcs, total * i as f64 / n as f64);
        let p = f.to_lonlat(x, y);
        ground += great_circle(prev.0, prev.1, p.0, p.1);
        prev = p;
    }
    println!("King's Cross - Stratford: {total:.3} m, walked on the ground {ground:.3} m");
    assert!((total - ground).abs() < 0.05 * total / 1000.0, "{total} vs {ground}");
    // The line runs, and a train covers ground metres: the line's length is the sum of its edges'.
    let ls = &w.svc.lines[ll as usize];
    assert!(ls.ok && ls.round_trip[HIGH] > 0.0);
    let sum = w.net.edge_len[e1 as usize];
    assert!((ls.runs[0].len - sum).abs() < 1e-6, "{} vs {sum}", ls.runs[0].len);
    // The same shape built at New York takes the same time: a straight 5 km east then 3 km north
    // with a 1,000 m curve, laid out in ground metres at both places.
    let time_at = |w: &mut TrackWorld, lon0: f64, lat0: f64| {
        let off = |dx: f64, dy: f64| {
            let lat = lat0 + (dy / EARTH_R).to_degrees();
            (lon0 + (dx / (EARTH_R * ((lat0 + lat) / 2.0).to_radians().cos())).to_degrees(), lat)
        };
        let (a, b, c) = (off(0.0, 0.0), off(5000.0, 0.0), off(5000.0, 3000.0));
        let na = place(w, a.0, a.1);
        let nc = place(w, c.0, c.1);
        let e = edge_ll(w, na, nc, &[(b.0, b.1, 1000.0)], 0).unwrap();
        station(w, na, 200);
        station(w, nc, 200);
        let l = line(w, &[na, nc], [12.0, 6.0, 3.0]);
        (w.net.edge_len[e as usize], w.svc.lines[l as usize].round_trip[HIGH])
    };
    let (len_ny, rt_ny) = time_at(&mut w, -73.70, 40.90);
    let (len_ld, rt_ld) = time_at(&mut w, 0.10, 51.60);
    println!("same shape: New York {len_ny:.2} m {rt_ny:.2} s, London {len_ld:.2} m {rt_ld:.2} s");
    assert!((len_ld / len_ny - 1.0).abs() < 1e-3 && (rt_ld / rt_ny - 1.0).abs() < 1e-3);
    // Both cities in one TWT6 save, and back with everything derived the same.
    let bytes = super::save::encode(&w.net);
    assert_eq!(&bytes[..4], b"TWT6");
    let back = TrackWorld::new(super::save::decode(&bytes).unwrap());
    assert_eq!(back.net.frame_kind, FrameKind::Merc);
    assert_eq!(super::save::encode(&back.net), bytes);
    for e in 0..w.net.edge_count() {
        assert_eq!(back.net.edge_len[e], w.net.edge_len[e]);
        assert_eq!(back.net.edge_cost[e], w.net.edge_cost[e]);
        assert_eq!(back.net.edge_time[e], w.net.edge_time[e]);
    }
    for l in nyc_lines.iter().chain([&ll]) {
        assert_eq!(back.svc.lines[*l as usize].round_trip, w.svc.lines[*l as usize].round_trip);
    }
}

#[test]
fn track_stops_short_of_the_poles_and_the_antimeridian() {
    let mut w = world_in(FrameKind::Merc);
    let try_node = |w: &mut TrackWorld, x: f64, y: f64| {
        let id = w.net.alloc_node();
        w.apply(Op::Node { id, data: Some(NodeData { x, y, level: 0, flying: false }) })
    };
    let f = w.net.frame();
    let (x, y) = f.from_lonlat(15.6, MERC_MAX_LAT - 0.5);
    assert!(try_node(&mut w, x, y).is_ok(), "Svalbard is fine");
    let (x, y) = f.from_lonlat(15.6, MERC_MAX_LAT + 0.5);
    let err = try_node(&mut w, x, y).unwrap_err();
    assert!(err.iter().any(|i| matches!(i, Issue::OutsideWorld { .. })), "{err:?}");
    let err = try_node(&mut w, std::f64::consts::PI * EARTH_R + 10.0, 0.0).unwrap_err();
    assert!(err.iter().any(|i| matches!(i, Issue::OutsideWorld { .. })));
    // The other frames have no such edge.
    let mut c = world_in(FrameKind::Lcc);
    assert!(try_node(&mut c, 0.0, 6.0e6).is_ok());
}

#[test]
fn a_conic_save_converts_to_mercator_with_lengths_moving_by_the_conic_scale() {
    let (lcc, edges, _) = nyc_network(FrameKind::Lcc);
    let mut w = lcc;
    // Boston, 1.6 degrees north of the conic's origin: the conic draws it 0.04% long.
    let a = place(&mut w, -71.0552, 42.3519);
    let b = place(&mut w, -71.0040, 42.3880);
    let eb = edge_ll(&mut w, a, b, &[], 0).unwrap();
    let bytes = super::save::encode(&w.net);
    assert_eq!(&bytes[..4], b"TWT5");
    let merc = super::save::convert_frame(&bytes, FrameKind::Merc).unwrap();
    assert_eq!(&merc[..4], b"TWT6");
    let m = TrackWorld::new(super::save::decode(&merc).unwrap());
    for &e in edges.iter().chain([&eb]) {
        let (o, n) = (w.net.edge_len[e as usize], m.net.edge_len[e as usize]);
        println!("edge {e}: conic {o:.3} m, Mercator {n:.3} m ({:+.4}%)", (n / o - 1.0) * 100.0);
        assert!((n / o - 1.0).abs() < 6e-4);
    }
    // Boston's edge is shorter in Mercator by the conic's scale there, about 0.04%.
    let d = m.net.edge_len[eb as usize] / w.net.edge_len[eb as usize] - 1.0;
    assert!(d < -2e-4 && d > -6e-4, "{d}");
    // Converting back to the conic puts every node within a few millimetres.
    let again = super::save::decode(&super::save::convert_frame(&merc, FrameKind::Lcc).unwrap()).unwrap();
    for i in 0..w.net.node_count() {
        assert!((again.node_x[i] - w.net.node_x[i]).hypot(again.node_y[i] - w.net.node_y[i]) < 0.003, "node {i}: {} m", (again.node_x[i] - w.net.node_x[i]).hypot(again.node_y[i] - w.net.node_y[i]));
    }
}

/// T-308: what converting a real legacy-frame save to Mercator changes, line by line. Needs the
/// real save (trainworld/T-007-real-nyc.save, Anita's), so it runs only on request:
/// `cargo test --manifest-path sim/Cargo.toml --release t308 -- --ignored --nocapture`.
#[test]
#[ignore]
fn t308_real_save_conversion_report() {
    let path = std::env::var("T308_SAVE").unwrap_or_else(|_| concat!(env!("CARGO_MANIFEST_DIR"), "/../T-007-real-nyc.save").into());
    let bytes = std::fs::read(path).unwrap();
    let old = TrackWorld::new(super::save::decode(&bytes).unwrap());
    let merc_bytes = super::save::convert_frame(&bytes, FrameKind::Merc).unwrap();
    let new = TrackWorld::new(super::save::decode(&merc_bytes).unwrap());
    assert_eq!(&merc_bytes[..4], b"TWT6");
    let all = |w: &TrackWorld| (0..w.net.edge_count() as u32).filter(|&e| w.net.edge_ok(e)).collect::<Vec<_>>();
    let (eo, en) = (all(&old), all(&new));
    assert_eq!(eo, en);
    // Refusals the geometry would now give (the conversion itself never checks).
    let issues = |w: &TrackWorld| w.net.validate(&all(w), &[], &[]);
    let (io, inn) = (issues(&old), issues(&new));
    println!("issues: legacy {} merc {}", io.len(), inn.len());
    // Crossings carry positions that move with the frame: compare those by their edges only.
    let key = |i: &Issue| match i {
        Issue::Crossing { e1, e2, overlap, .. } => format!("Crossing {e1} {e2} {overlap}"),
        i => format!("{i:?}"),
    };
    let (ko, kn): (std::collections::BTreeSet<String>, std::collections::BTreeSet<String>) = (io.iter().map(&key).collect(), inn.iter().map(&key).collect());
    println!("issues new in Mercator: {}, gone: {}", kn.difference(&ko).count(), ko.difference(&kn).count());
    for k in kn.difference(&ko).take(20) {
        println!("  new   {k}");
    }
    for k in ko.difference(&kn).take(20) {
        println!("  gone  {k}");
    }
    let count = |s: &std::collections::BTreeSet<String>, what: &str| s.iter().filter(|k| k.contains(what)).count();
    println!("legacy: {} RadiusTooSmall, {} ArcDoesNotFit, {} crossings; merc: {}, {}, {}", count(&ko, "RadiusTooSmall"), count(&ko, "ArcDoesNotFit"), count(&ko, "Crossing"), count(&kn, "RadiusTooSmall"), count(&kn, "ArcDoesNotFit"), count(&kn, "Crossing"));
    // Per edge: length, cost, tightest curve (ground radius).
    let min_r = |w: &TrackWorld, e: u32| {
        w.net.edge_pieces(e).iter().filter(|p| p.k != 0.0).map(|p| (1.0 / p.k.abs()) / p.g).fold(f64::INFINITY, f64::min)
    };
    let (mut len_o, mut len_n, mut cost_o, mut cost_n) = (0.0, 0.0, 0.0, 0.0);
    let mut worst_r = (0.0f64, 0u32);
    let mut worst_edge = (0.0f64, 0u32);
    let mut tight = (f64::INFINITY, f64::INFINITY);
    let mut by_change: Vec<(f64, u32)> = vec![];
    for &e in &eo {
        let i = e as usize;
        len_o += old.net.edge_len[i];
        len_n += new.net.edge_len[i];
        cost_o += old.net.edge_cost[i];
        cost_n += new.net.edge_cost[i];
        let d = (new.net.edge_len[i] / old.net.edge_len[i] - 1.0).abs();
        if d > worst_edge.0 {
            worst_edge = (d, e);
        }
        by_change.push((d, e));
        let (ro, rn) = (min_r(&old, e), min_r(&new, e));
        if ro.is_finite() && rn.is_finite() {
            let d = (rn / ro - 1.0).abs();
            if d > worst_r.0 {
                worst_r = (d, e);
            }
            tight = (tight.0.min(ro), tight.1.min(rn));
        }
    }
    println!("edges {}: length {:.3} km -> {:.3} km ({:+.4}%), cost {:.3} -> {:.3} ({:+.4}%)", eo.len(), len_o / 1e3, len_n / 1e3, (len_n / len_o - 1.0) * 100.0, cost_o, cost_n, (cost_n / cost_o - 1.0) * 100.0);
    println!("worst edge length change {:.4}% (edge {}), worst tightest-radius change {:.4}% (edge {}), tightest radius anywhere {:.2} m -> {:.2} m", worst_edge.0 * 100.0, worst_edge.1, worst_r.0 * 100.0, worst_r.1, tight.0, tight.1);
    by_change.sort_by(|a, b| b.0.total_cmp(&a.0));
    for &(_, e) in by_change.iter().take(6) {
        let i = e as usize;
        println!("  edge {e}: {:.1} m -> {:.1} m ({:+.3}%)", old.net.edge_len[i], new.net.edge_len[i], (new.net.edge_len[i] / old.net.edge_len[i] - 1.0) * 100.0);
    }
    let over = by_change.iter().filter(|c| c.0 > 0.005).count();
    println!("edges whose length changes by more than 0.5%: {over} of {}", by_change.len());
    // Per line.
    struct Row {
        name: String,
        len_o: f64,
        len_n: f64,
        rt_o: [f64; 3],
        rt_n: [f64; 3],
        tr_o: [u32; 3],
        tr_n: [u32; 3],
        ok_o: bool,
        ok_n: bool,
    }
    let mut rows = vec![];
    for l in 0..old.net.line_count() {
        if !old.net.line_ok(l as u32) {
            continue;
        }
        let (a, b) = (&old.svc.lines[l], &new.svc.lines[l]);
        rows.push(Row { name: old.net.line_name[l].clone(), len_o: a.runs[0].len, len_n: b.runs[0].len, rt_o: a.round_trip, rt_n: b.round_trip, tr_o: a.trains, tr_n: b.trains, ok_o: a.ok, ok_n: b.ok });
    }
    println!("lines {}; not ok: legacy {}, merc {}", rows.len(), rows.iter().filter(|r| !r.ok_o).count(), rows.iter().filter(|r| !r.ok_n).count());
    let pct = |a: f64, b: f64| if a > 0.0 { (b / a - 1.0) * 100.0 } else { 0.0 };
    rows.sort_by(|a, b| pct(b.rt_o[0], b.rt_n[0]).abs().total_cmp(&pct(a.rt_o[0], a.rt_n[0]).abs()));
    println!("{:<28} {:>9} {:>9} {:>7} {:>10} {:>10} {:>7} trains(h/m/l) old -> new", "line", "len old m", "len new m", "len %", "rt old s", "rt new s", "rt %");
    for r in rows.iter().take(15) {
        println!("{:<28} {:>9.0} {:>9.0} {:>+7.3} {:>10.1} {:>10.1} {:>+7.3} {:?} -> {:?}", r.name, r.len_o, r.len_n, pct(r.len_o, r.len_n), r.rt_o[0], r.rt_n[0], pct(r.rt_o[0], r.rt_n[0]), r.tr_o, r.tr_n);
    }
    let over1 = rows.iter().filter(|r| (0..3).any(|k| pct(r.rt_o[k], r.rt_n[k]).abs() > 1.0)).count();
    let trains_changed = rows.iter().filter(|r| r.tr_o != r.tr_n).count();
    let tot = |f: &dyn Fn(&Row) -> f64| rows.iter().map(f).sum::<f64>();
    println!("lines with a run time change over 1%: {over1}; lines whose trains needed changed: {trains_changed}");
    println!("all lines: length {:+.4}%, round trip (high) {:+.4}%", pct(tot(&|r| r.len_o), tot(&|r| r.len_n)), pct(tot(&|r| r.rt_o[0]), tot(&|r| r.rt_n[0])));
    println!("trains needed, all lines, high/medium/low: {:?} -> {:?}", (0..3).map(|k| rows.iter().map(|r| r.tr_o[k]).sum::<u32>()).collect::<Vec<_>>(), (0..3).map(|k| rows.iter().map(|r| r.tr_n[k]).sum::<u32>()).collect::<Vec<_>>());
}
