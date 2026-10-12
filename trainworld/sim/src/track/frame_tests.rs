//! World frames through the track model (T-125): lengths and curves of long edges, the save
//! formats, converting a save, and the water mask in each frame. `frame.rs` tests the maths.

use super::cost::{GeoMask, RunMask, TransformedMask, WaterMask};
use super::frame::{great_circle, Frame, FrameKind};
use super::geom::{self, Pi};
use super::net::Network;
use super::tests::{edge, node, station, world};
use super::world::TrackWorld;
use std::sync::Arc;

fn world_in(kind: FrameKind) -> TrackWorld {
    TrackWorld::new(Network::new_in(kind, -73.985, 40.758, true))
}

const PLACES: [(&str, f64, f64); 4] = [
    ("Philadelphia", -75.1636, 39.9526),
    ("Boston", -71.0589, 42.3601),
    ("Washington", -77.0369, 38.9072),
    ("Pittsburgh", -79.9959, 40.4406),
];

/// A straight edge from New York to a place, in a frame: (its length in the plane, the great
/// circle between its ends).
fn long_edge(kind: FrameKind, to: (f64, f64)) -> (f64, f64) {
    let mut w = world_in(kind);
    let f = w.net.frame();
    let (x0, y0) = f.from_lonlat(-73.985, 40.758);
    let (x1, y1) = f.from_lonlat(to.0, to.1);
    let a = node(&mut w, x0, y0, 0);
    let b = node(&mut w, x1, y1, 0);
    let e = edge(&mut w, a, b, &[], 0).expect("a long edge is valid");
    (w.net.edge_len[e as usize], great_circle(-73.985, 40.758, to.0, to.1))
}

#[test]
fn a_long_edge_is_the_right_length_in_the_world_frame_and_not_in_the_legacy_one() {
    for (name, lon, lat) in PLACES {
        let (plane, ground) = long_edge(FrameKind::Lcc, (lon, lat));
        let (plane_old, _) = long_edge(FrameKind::Equirect, (lon, lat));
        let (err, err_old) = ((plane / ground - 1.0) * 100.0, (plane_old / ground - 1.0) * 100.0);
        println!("{name}: ground {:.1} km, world frame {err:+.3}%, legacy frame {err_old:+.3}%", ground / 1000.0);
        assert!(err.abs() < 0.15, "{name}: world frame {err}%");
        // The legacy frame is what it was: 1.2% at Philadelphia and worse further out.
        assert!(err_old.abs() > 0.2, "{name}: legacy {err_old}%");
    }
}

#[test]
fn arc_radius_and_curve_speed_are_right_in_the_world_frame() {
    // A 4 km radius arc turned through 90 degrees near Philadelphia, drawn east-west first and
    // north-south first: the ground radius is the plane radius over the local scale either way.
    let f = Frame::lcc(-73.985, 40.758);
    let (cx, cy) = f.from_lonlat(-75.1636, 39.9526);
    let k = f.scale_at(39.9526).0;
    for (dx, dy) in [(1.0, 0.0), (0.0, 1.0)] {
        let leg = 9000.0;
        let v = [
            Pi::new(cx, cy, 0.0, 0),
            Pi::new(cx + dx * leg, cy + dy * leg, 4000.0, 0),
            Pi::new(cx + dx * leg - dy * leg, cy + dy * leg + dx * leg, 0.0, 0),
        ];
        let fit = geom::fit(&v);
        let arc = fit.pieces.iter().find(|p| p.k != 0.0).unwrap();
        let pt = |s: f64| {
            let (x, y, _) = arc.at(arc.len * s);
            f.to_lonlat(x, y)
        };
        let (a, b, c) = (pt(0.0), pt(0.5), pt(1.0));
        let (ab, bc, ca) = (great_circle(a.0, a.1, b.0, b.1), great_circle(b.0, b.1, c.0, c.1), great_circle(c.0, c.1, a.0, a.1));
        let s = (ab + bc + ca) / 2.0;
        let area = (s * (s - ab) * (s - bc) * (s - ca)).sqrt();
        let ground_radius = ab * bc * ca / (4.0 * area);
        assert!((ground_radius * k / 4000.0 - 1.0).abs() < 2e-4, "{dx},{dy}: {ground_radius}");
        // the curve speed limit follows the ground radius to a tenth of a percent
        assert!((geom::curve_speed(ground_radius) / geom::curve_speed(4000.0) - 1.0).abs() < 1e-3);
    }
}

#[test]
fn world_frame_saves_are_twt5_and_legacy_ones_stay_twt4_byte_for_byte() {
    let build = |kind| {
        let mut w = world_in(kind);
        let a = node(&mut w, 0.0, 0.0, 0);
        let b = node(&mut w, 3000.0, 0.0, 0);
        let c = node(&mut w, 9000.0, 3000.0, 0);
        edge(&mut w, a, b, &[], 0).unwrap();
        edge(&mut w, b, c, &[(6000.0, 0.0)], 0).unwrap();
        station(&mut w, a, 200);
        station(&mut w, c, 200);
        super::save::encode(&w.net)
    };
    let (old, new) = (build(FrameKind::Equirect), build(FrameKind::Lcc));
    assert_eq!(&old[..4], b"TWT4");
    assert_eq!(&new[..4], b"TWT5");
    assert_eq!(old[4..], new[4..], "only the magic differs");
    let back = super::save::decode(&new).unwrap();
    assert_eq!(back.frame_kind, FrameKind::Lcc);
    assert_eq!(super::save::encode(&back), new);
    assert_eq!(super::save::decode(&old).unwrap().frame_kind, FrameKind::Equirect);
}

#[test]
fn converting_a_save_between_frames_keeps_every_place_on_the_ground() {
    // A network 100 km west and 90 km south of the origin, as a shared Northeast save holds Philadelphia.
    let mut w = world();
    let f0 = w.net.frame();
    let (px, py) = f0.from_lonlat(-75.1636, 39.9526);
    let a = node(&mut w, px, py, 0);
    let b = node(&mut w, px + 12000.0, py, 0);
    let c = node(&mut w, px + 30000.0, py, -1);
    edge(&mut w, a, b, &[(px + 6000.0, py)], 0).unwrap();
    edge(&mut w, b, c, &[], -1).unwrap();
    station(&mut w, a, 200);
    let legacy = super::save::encode(&w.net);
    let lcc = super::save::convert_frame(&legacy, FrameKind::Lcc).unwrap();
    assert_eq!(&lcc[..4], b"TWT5");
    let net = super::save::decode(&lcc).unwrap();
    let f1 = net.frame();
    let n0 = super::save::decode(&legacy).unwrap();
    for i in 0..3 {
        let (lon, lat) = f0.to_lonlat(n0.node_x[i], n0.node_y[i]);
        let (lon2, lat2) = f1.to_lonlat(net.node_x[i], net.node_y[i]);
        assert!(great_circle(lon, lat, lon2, lat2) < 0.002, "node {i}");
    }
    // The shape is kept: the PI of the first edge is where it was on the ground.
    let pi = net.edge_data(0).unwrap().pis[0];
    let (lon, lat) = f0.to_lonlat(px + 6000.0, py);
    let (lon2, lat2) = f1.to_lonlat(pi.x, pi.y);
    assert!(great_circle(lon, lat, lon2, lat2) < 0.002);
    assert_eq!(net.node_platform[0], 200);
    // And back again to a millimetre or two: positions are on the 1 mm grid both ways.
    let back = super::save::decode(&super::save::convert_frame(&lcc, FrameKind::Equirect).unwrap()).unwrap();
    assert_eq!(back.frame_kind, FrameKind::Equirect);
    for i in 0..3 {
        assert!((back.node_x[i] - n0.node_x[i]).hypot(back.node_y[i] - n0.node_y[i]) < 0.0025, "node {i} back");
    }
    // Converting to the frame a save is already in changes nothing.
    assert_eq!(super::save::convert_frame(&legacy, FrameKind::Equirect).unwrap(), legacy);
    assert_eq!(super::save::convert_frame(&lcc, FrameKind::Lcc).unwrap(), lcc);
    // Lengths follow the frame: the legacy plane draws Philadelphia's east-west metres about
    // 1.2% short, and the converted network measures the ground.
    let l_old = TrackWorld::new(n0).net.edge_len[1];
    let l_new = TrackWorld::new(net).net.edge_len[1];
    println!("edge {l_old:.1} m in the legacy frame, {l_new:.1} m in the world frame");
    assert!((l_new / l_old - 1.0).abs() > 0.0005 && (l_new / l_old - 1.0).abs() < 0.02);
}

#[test]
fn geo_mask_agrees_with_the_affine_mask_in_the_legacy_frame_and_follows_the_ground_in_the_world_frame() {
    // A pack at Philadelphia: 100 x 100 cells of 100 m, every third column water.
    let (w, h, cell) = (100u32, 100u32, 100.0);
    let mut row = vec![0u32];
    let mut xs = vec![];
    for _ in 0..h {
        for c in (0..w as u16).step_by(3) {
            xs.extend([c, c + 1]);
        }
        row.push(xs.len() as u32);
    }
    let mask = Arc::new(RunMask { x0: -5000.0, y0: -5000.0, cell, w, h, row, xs });
    let (plon, plat) = (-75.1636, 39.9526);
    let legacy = Frame::equirect(-73.985, 40.758);
    let world = Frame::lcc(-73.985, 40.758);
    let rad = std::f64::consts::PI / 180.0;
    let affine = TransformedMask {
        mask: mask.clone(),
        offset: [6371008.8 * (plon + 73.985) * rad * (40.758 * rad).cos(), 6371008.8 * (plat - 40.758) * rad],
        scale: [(40.758 * rad).cos() / (plat * rad).cos(), 1.0],
    };
    let geo = GeoMask::new(mask.clone(), legacy, plon, plat);
    let geo_world = GeoMask::new(mask.clone(), world, plon, plat);
    let pack = Frame::equirect(plon, plat);
    let (mut off_affine, mut off_truth, mut n, mut wet, mut worst) = (0, 0, 0, 0, 0.0f64);
    for i in 0..120 {
        for j in 0..120 {
            let (px, py) = (-4900.0 + i as f64 * 81.7 + 13.1, -4900.0 + j as f64 * 81.3 + 7.7);
            let (lon, lat) = pack.to_lonlat(px, py);
            let truth = mask.is_water(px, py);
            let (x, y) = legacy.from_lonlat(lon, lat);
            n += 1;
            wet += truth as i32;
            if affine.is_water(x, y) != geo.is_water(x, y) {
                off_affine += 1;
            }
            let (wx, wy) = world.from_lonlat(lon, lat);
            if geo.is_water(x, y) != truth || geo_world.is_water(wx, wy) != truth {
                off_truth += 1;
            }
            // the lattice interpolation against the exact map, metres in the pack's frame
            for (g, qx, qy) in [(&geo, x, y), (&geo_world, wx, wy)] {
                let (ax, ay) = g.to_pack(qx, qy).expect("inside the lattice");
                worst = worst.max((ax - px).hypot(ay - py));
            }
        }
    }
    assert!(wet > n / 4 && wet < n / 2, "{wet} of {n} wet");
    println!("lattice warp worst error {worst:.4} m; {off_affine} of {n} points differ from the affine mask, {off_truth} from the raster");
    assert!(worst < 0.1, "warp error {worst} m");
    // a point within the warp error of a cell edge can land in the neighbour cell, no more than that
    assert!(off_affine * 500 < n && off_truth * 500 < n);
    assert!(!geo.is_water(1e7, 1e7) && !geo_world.is_water(-1e7, 1e7));
}
