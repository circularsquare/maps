use super::*;
fn api() -> TrackApi { TrackApi::new(-73.985, 40.758, true) }
#[cfg(feature = "demand")]
fn free(x: f64, y: f64) -> [f64; 5] { [0.0, 0.0, x, y, 0.0] }
#[test]
fn pack_water_follows_the_frame_of_the_loaded_save() {
    // A pack at Philadelphia with one water cell of 100 m at its origin, in the Northeast.
    let mut a = api();
    let legacy_empty = a.save();
    a.clear_water();
    a.add_pack_water(100.0, 0.0, 0.0, 1, 1, &[0, 2], &[0, 1], -75.1636, 39.9526);
    let at = |a: &TrackApi, lon: f64, lat: f64| {
        let p = a.from_lonlat(lon, lat);
        a.w.net.water_mask.is_water(p[0], p[1])
    };
    // legacy frame: the Philadelphia cell is water, a point 1 km east is not
    assert_eq!(a.frame_kind(), 0);
    assert!(at(&a, -75.1636 + 0.0004, 39.9526 + 0.0004));
    assert!(!at(&a, -75.1636 + 0.012, 39.9526 + 0.0004));
    // a world-frame network: the same ground is water, through the other chart
    assert!(a.set_empty_frame(1));
    assert_eq!(a.frame_kind(), 1);
    assert!(at(&a, -75.1636 + 0.0004, 39.9526 + 0.0004));
    assert!(!at(&a, -75.1636 + 0.012, 39.9526 + 0.0004));
    let world_empty = a.save();
    assert_eq!(&world_empty[..4], b"TWT5");
    assert_eq!(&legacy_empty[..4], b"TWT4");
    // loading each kind of save rebuilds the mask for its frame
    assert!(a.load(&legacy_empty));
    assert_eq!(a.frame_kind(), 0);
    assert!(at(&a, -75.1636 + 0.0004, 39.9526 + 0.0004));
    assert!(!at(&a, -75.1636 + 0.012, 39.9526 + 0.0004));
    assert!(a.load(&world_empty));
    assert_eq!(a.frame_kind(), 1);
    assert!(at(&a, -75.1636 + 0.0004, 39.9526 + 0.0004));
    assert!(!at(&a, -75.1636 + 0.012, 39.9526 + 0.0004));
    // a frame cannot be switched under drawn track
    assert!(a.set_empty_frame(0));
    let n = a.new_node_id();
    assert_eq!(a.add_route(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3000.0, 0.0, 0.0], &[], 2, false), 0);
    let _ = n;
    assert!(!a.set_empty_frame(1));
    assert_eq!(TrackApi::convert_save_frame(&a.save(), 1)[..4], *b"TWT5");
    assert!(TrackApi::convert_save_frame(b"junk", 1).is_empty());
}

/// T-1039: a mask added under drawn track reworks only the track near it, to the same water
/// and prices as working out the whole network again.
#[test]
fn a_mask_added_under_track_matches_a_full_rebuild() {
    for world in [false, true] {
        let mut a = api();
        assert!(a.set_empty_frame(world as u8));
        let route = |a: &mut TrackApi, x0: f64, x1: f64| assert_eq!(a.add_route(&[0.0, 0.0, x0, 0.0, 0.0, 0.0, 0.0, x1, 0.0, 0.0], &[], 2, false), 0);
        route(&mut a, -3000.0, 3000.0);
        route(&mut a, 100_000.0, 106_000.0);
        let rows: Vec<u32> = (0..=20).map(|r| r * 2).collect();
        let xs: Vec<u16> = (0..20).flat_map(|_| [8u16, 12]).collect();
        let (lon, lat) = (a.w.net.origin_lon, a.w.net.origin_lat);
        let state = |a: &TrackApi| (0..a.w.net.edge_count() as u32).filter(|&e| a.w.net.edge_ok(e)).map(|e| (a.w.net.edge_cost[e as usize], a.w.net.edge_water(e).to_vec())).collect::<Vec<_>>();
        let dry = state(&a);
        // one far away first, then one under the first route
        a.add_pack_water(100.0, -1000.0, -1000.0, 20, 20, &rows, &xs, lon + 3.0, lat);
        assert_eq!(state(&a), dry);
        a.add_pack_water(100.0, -1000.0, -1000.0, 20, 20, &rows, &xs, lon, lat);
        let added = state(&a);
        assert!(added.iter().any(|(_, w)| !w.is_empty()), "the first route crosses water (world frame {world})");
        assert!(added.iter().zip(&dry).any(|(p, q)| p.0 > q.0));
        a.rebuild_water();
        assert_eq!(state(&a), added, "world frame {world}");
    }
}

#[test]
fn regional_water_union_survives_legacy_network_load() {
    let mut a = api();
    let legacy = a.save();
    a.set_water(25.0, 0.0, 0.0, 1, 1, &[0, 2], &[0, 1]);
    a.append_water(25.0, 0.0, 0.0, 1, 1, &[0, 2], &[0, 1], -100000.0, -90000.0, 0.98, 1.0);
    assert!(a.w.net.water_mask.is_water(12.0, 12.0));
    assert!(a.w.net.water_mask.is_water(-99988.0, -89988.0));
    assert!(!a.w.net.water_mask.is_water(-50000.0, -45000.0));
    assert!(a.load(&legacy));
    assert_eq!(a.save(), legacy, "adding region coverage changes no saved inputs");
    assert_eq!(a.w.net.origin_lon, -73.985);
    assert_eq!(a.w.net.origin_lat, 40.758);
    assert!(a.w.net.water_mask.is_water(-99988.0, -89988.0));
    a.set_water(25.0, 0.0, 0.0, 1, 1, &[0, 2], &[0, 1]);
    assert!(!a.w.net.water_mask.is_water(-99988.0, -89988.0), "legacy setter replaces coverage");
    a.clear_water();
    assert!(a.w.net.water_mask.is_empty());
}

#[test]
#[cfg(feature = "demand")]
#[ignore = "requires the generated NYC and Philadelphia city packs"]
fn northeast_pack_water_crossings() {
    // The same crossings, in the legacy Northeast frame and in the world frame (T-125).
    for world_frame in [false, true] {
        let mut a = api();
        assert!(a.set_empty_frame(world_frame as u8));
        for city in ["nyc", "phl"] {
            let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../data/packs");
            let head: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join(format!("{city}.json"))).unwrap()).unwrap();
            let bin = std::fs::read(dir.join(format!("{city}.bin"))).unwrap();
            let array = |name: &str, width: usize| {
                let item = head["arrays"].as_array().unwrap().iter().find(|v| v["name"] == name).unwrap();
                let start = item["offset"].as_u64().unwrap() as usize;
                &bin[start..start + item["count"].as_u64().unwrap() as usize * width]
            };
            let water = &head["water"];
            let m = RunMask::from_pack(water["cell_m"].as_f64().unwrap(),
                [water["origin_m"][0].as_f64().unwrap(), water["origin_m"][1].as_f64().unwrap()],
                [water["size"][0].as_u64().unwrap() as u32, water["size"][1].as_u64().unwrap() as u32],
                array("water_row", 4), array("water_x", 2));
            let lon = head["origin"]["lon"].as_f64().unwrap();
            let lat = head["origin"]["lat"].as_f64().unwrap();
            a.add_pack_water(m.cell, m.x0, m.y0, m.w, m.h, &m.row, &m.xs, lon, lat);
        }
        let world = |a: &TrackApi, lon: f64, lat: f64| { let p = a.from_lonlat(lon, lat); [p[0], p[1]] };
        for (name, lon, lat, water) in [
            ("Hudson", -74.018, 40.742, true), ("Times Square", -73.985, 40.758, false),
            ("Delaware at Philadelphia", -75.135, 39.950, true), ("City Hall", -75.1636, 39.9526, false),
            ("Delaware at Trenton", -74.7755, 40.2169, true), ("Trenton station", -74.754, 40.219, false),
        ] {
            let [x, y] = world(&a, lon, lat);
            assert_eq!(a.w.net.water_mask.is_water(x, y), water, "{name} (world frame {world_frame})");
        }
        let [x, y] = world(&a, -74.7755, 40.2169);
        let ends: Vec<f64> = free(x - 200.0, y).into_iter().chain(free(x + 200.0, y)).collect();
        assert_eq!(a.preview_route(&ends, &[], 2, false)[0], 0.0, "ground-level Delaware crossing refused");
        assert!(a.issue_kinds().contains("BridgeNoRoom"), "400 m is no room for the ramps");
    }
}
