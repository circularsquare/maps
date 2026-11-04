//! T-2414: bridges between close crossings, ends near water and islands.
use super::*;

fn api() -> TrackApi {
    TrackApi::new(-73.985, 40.758, true)
}
fn base(a: &TrackApi) -> (f64, f64) {
    let p = a.from_lonlat(-73.985, 40.758);
    (p[0], p[1])
}
/// Water where x (from `base`) lies in any of the bands.
fn water(a: &mut TrackApi, bands: &[(f64, f64)]) {
    let (bx, _) = base(a);
    let bands = bands.to_vec();
    a.w.net.set_water_mask(Box::new(move |x: f64, _y: f64| bands.iter().any(|b| x > bx + b.0 && x < bx + b.1)));
    a.w.reload();
}
fn free(a: &TrackApi, x: f64, y: f64) -> [f64; 5] {
    let (bx, by) = base(a);
    [0.0, 0.0, bx + x, by + y, 0.0]
}
fn node_end(n: u32) -> [f64; 5] {
    [1.0, n as f64, 0.0, 0.0, 0.0]
}
fn node_near(a: &TrackApi, x: f64, y: f64) -> u32 {
    let (bx, by) = base(a);
    (0..a.w.net.node_count() as u32).filter(|&n| a.w.net.node_ok(n)).min_by(|&p, &q| {
        let d = |n: u32| (a.w.net.node_x[n as usize] - bx - x).hypot(a.w.net.node_y[n as usize] - by - y);
        d(p).total_cmp(&d(q))
    }).unwrap()
}
fn alive(a: &TrackApi) -> Vec<u32> {
    (0..a.w.net.edge_count() as u32).filter(|&e| a.w.net.edge_ok(e)).collect()
}
fn problems(a: &TrackApi) -> Vec<Issue> {
    let nodes: Vec<u32> = (0..a.w.net.node_count() as u32).filter(|&n| a.w.net.node_ok(n)).collect();
    a.w.net.validate(&alive(a), &nodes, &[])
}
/// Every sample over water on every edge is a level up, but for slivers the edit's own reading
/// (every `WATER_STEP`) can step over.
fn all_held(a: &TrackApi) -> bool {
    alive(a).iter().all(|&e| {
        let net = &a.w.net;
        let (pieces, len) = (net.edge_pieces(e), net.edge_len[e as usize]);
        let (mut s, mut low) = (0.0, 0.0);
        while s <= len {
            let (x, y, _) = geom::pos_at(pieces, s);
            low = if net.water_mask.is_water(x, y) && geom::height_at(net.edge_vert(e), s) < LEVEL_H - 0.01 { low + 2.0 } else { 0.0 };
            if low > WATER_STEP {
                return false;
            }
            s += 2.0;
        }
        true
    })
}
fn route(a: &mut TrackApi, from: [f64; 5], to: [f64; 5]) -> Result<(), String> {
    route_n(a, from, to, 2)
}
fn route_n(a: &mut TrackApi, from: [f64; 5], to: [f64; 5], tracks: u8) -> Result<(), String> {
    let e: Vec<f64> = from.iter().chain(to.iter()).copied().collect();
    let pv = a.preview_route(&e, &[], tracks, false)[0];
    let r = a.add_route(&e, &[], tracks, false);
    if (pv == 1.0) != (r == 0) {
        return Err(format!("preview {pv} and edit {r} disagree: {}", a.issue_kinds()));
    }
    if r != 0 {
        return Err(a.issue_kinds());
    }
    if !problems(a).is_empty() {
        return Err(format!("built with problems {:?}", problems(a)));
    }
    if !all_held(a) {
        return Err("built low over water".into());
    }
    Ok(())
}

/// The reproduction cases: (name, result).
fn cases() -> Vec<(String, Result<(), String>)> {
    let mut out = vec![];
    let gaps = [60.0, 120.0, 200.0, 300.0, 420.0, 460.0, 520.0, 700.0];
    // A: one straight segment over two channels with land between
    for &g in &gaps {
        let mut a = api();
        water(&mut a, &[(1000.0, 1150.0), (1150.0 + g, 1300.0 + g)]);
        let (f, t) = (free(&a, 0.0, 0.0), free(&a, 3500.0, 0.0));
        out.push((format!("A straight, two channels {g} m apart"), route(&mut a, f, t)));
    }
    // B: the same on a curve drawn on from a track end
    for &from in &[900.0, 1100.0, 1300.0] {
        for &g in &gaps {
            for &(tx, ty) in &[(2600.0, 900.0), (2200.0, 1400.0)] {
                let mut a = api();
                let (f, t) = (free(&a, 0.0, 0.0), free(&a, 600.0, 0.0));
                route(&mut a, f, t).unwrap();
                water(&mut a, &[(from, from + 120.0), (from + 120.0 + g, from + 240.0 + g)]);
                a.set_route_tangent(true);
                let n = node_near(&a, 600.0, 0.0);
                let t = free(&a, tx, ty);
                out.push((format!("B curve to ({tx}, {ty}), channels at {from}, {g} m apart"), route(&mut a, node_end(n), t)));
            }
        }
    }
    // C: a click on an island between two channels, then on across the second
    for &w in &[80.0, 150.0, 250.0, 350.0, 430.0, 500.0] {
        let mut a = api();
        water(&mut a, &[(1000.0, 1200.0), (1200.0 + w, 1400.0 + w)]);
        let (f, t) = (free(&a, 0.0, 0.0), free(&a, 1200.0 + w / 2.0, 0.0));
        let r = route(&mut a, f, t);
        out.push((format!("C island {w} m wide, the click on it"), r.clone()));
        if r.is_ok() {
            let n = node_near(&a, 1200.0 + w / 2.0, 0.0);
            let t = free(&a, 3500.0, 0.0);
            out.push((format!("C island {w} m wide, on from the click"), route(&mut a, node_end(n), t)));
        }
    }
    // D: a click on land short of the water, then on across it, straight and at an angle
    for &d in &[30.0, 80.0, 150.0, 200.0, 250.0] {
        for &(tx, ty) in &[(3500.0, 0.0), (3000.0, 1600.0)] {
            let mut a = api();
            water(&mut a, &[(1500.0, 1800.0)]);
            let (f, t) = (free(&a, 0.0, 0.0), free(&a, 1500.0 - d, 0.0));
            route(&mut a, f, t).unwrap();
            a.set_route_tangent(true);
            let n = node_near(&a, 1500.0 - d, 0.0);
            let t = free(&a, tx, ty);
            out.push((format!("D click {d} m short of the water, on to ({tx}, {ty})"), route(&mut a, node_end(n), t)));
        }
    }
    // E: a segment across the water ending on land just past it
    for &d in &[30.0, 80.0, 150.0, 200.0] {
        let mut a = api();
        water(&mut a, &[(1500.0, 1800.0)]);
        let (f, t) = (free(&a, 0.0, 0.0), free(&a, 1800.0 + d, 0.0));
        out.push((format!("E across, ending {d} m past the water"), route(&mut a, f, t)));
        let mut q = api();
        water(&mut q, &[(1500.0, 1800.0)]);
        let (f, t) = (free(&q, 0.0, 0.0), free(&q, 1800.0 + d, 0.0));
        out.push((format!("E quad across, ending {d} m past the water"), route_n(&mut q, f, t, 4)));
    }
    // F: a station at a track end short of the water, track drawn on from it
    for &d in &[60.0, 120.0, 180.0, 230.0, 300.0, 400.0] {
        let mut a = api();
        let (f, t) = (free(&a, 0.0, 0.0), free(&a, 2000.0, 0.0));
        route(&mut a, f, t).unwrap();
        let n = node_near(&a, 2000.0, 0.0);
        assert_eq!(a.add_station(1, n, 0.0, 0.0, 100, "River".into()), 0, "{}", a.issue_kinds());
        water(&mut a, &[(2000.0 + d, 2200.0 + d)]);
        let n = node_near(&a, 2000.0, 0.0);
        let t = free(&a, 4000.0, 0.0);
        out.push((format!("F station {d} m short of the water"), route(&mut a, node_end(n), t)));
    }
    // G: a branch off the middle of track short of the water
    for &d in &[20.0, 300.0] {
        let mut a = api();
        let (f, t) = (free(&a, 1000.0 - d, -1500.0), free(&a, 1000.0 - d, 1500.0));
        route(&mut a, f, t).unwrap();
        water(&mut a, &[(1000.0, 1300.0)]);
        let e = alive(&a)[0];
        let (bx, by) = base(&a);
        let from = [2.0, e as f64, bx + 1000.0 - d, by, 0.0];
        let t = free(&a, 3000.0, 0.0);
        out.push((format!("G branch off track {d} m short of the water"), route(&mut a, from, t)));
    }
    // H: built track ending short of the water
    for &d in &[80.0, 300.0] {
        let mut a = api();
        let (f, t) = (free(&a, 0.0, 0.0), free(&a, 1000.0 - d, 0.0));
        route(&mut a, f, t).unwrap();
        assert_eq!(a.construct_all(), 0, "{}", a.issue_kinds());
        water(&mut a, &[(1000.0, 1300.0)]);
        let n = node_near(&a, 1000.0 - d, 0.0);
        let t = free(&a, 3000.0, 0.0);
        out.push((format!("H built track ending {d} m short of the water"), route(&mut a, node_end(n), t)));
    }
    out
}

/// Every case builds, but for water nearer than a ramp and its shore to a station, a junction or
/// built track, each refused naming that.
#[test]
fn close_crossings_islands_and_banks_build() {
    let res = cases();
    let refused: Vec<&(String, Result<(), String>)> = res.iter().filter(|r| r.1.is_err()).collect();
    for (name, r) in &res {
        println!("{name}: {}", match r { Ok(()) => "built".to_string(), Err(e) => e.clone() });
    }
    println!("T-2414 sweep: {} of {} refused", refused.len(), res.len());
    for (name, r) in &res {
        let short = |d: f64| ["F", "G", "H"].iter().any(|p| name.starts_with(p)) && name.contains(&format!(" {d} m short"));
        let want = if [60.0, 120.0, 180.0].iter().any(|&d| short(d)) {
            Some("BridgeNoRoomStation")
        } else if short(20.0) {
            Some("BridgeNoRoomJunction")
        } else if short(80.0) {
            Some("BridgeNoRoomBuilt")
        } else {
            None
        };
        match want {
            None => assert!(r.is_ok(), "{name}: {r:?}"),
            Some(k) => assert_eq!(r.as_ref().err().map(String::as_str), Some(k), "{name}"),
        }
    }
}

/// The same kinds of drawing over New York's real water (the city pack's mask, a world frame game
/// as a new game is): lines across rivers, channels and islands; each segment straight over all
/// of a line, a click on each island or strip of land narrower than 600 m then on, a click short
/// of each stretch of water then on (straight and turning), and a segment ending just past each.
/// Prints the refusals; `T2414_STRICT=1` fails on any.
#[test]
#[cfg(feature = "demand")]
#[ignore = "requires the generated NYC city pack"]
fn nyc_water_sweep() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../data/packs");
    let head: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("nyc.json")).unwrap()).unwrap();
    let bin = std::fs::read(dir.join("nyc.bin")).unwrap();
    let array = |name: &str, width: usize| {
        let item = head["arrays"].as_array().unwrap().iter().find(|v| v["name"] == name).unwrap();
        let start = item["offset"].as_u64().unwrap() as usize;
        bin[start..start + item["count"].as_u64().unwrap() as usize * width].to_vec()
    };
    let wj = &head["water"];
    let m = RunMask::from_pack(wj["cell_m"].as_f64().unwrap(), [wj["origin_m"][0].as_f64().unwrap(), wj["origin_m"][1].as_f64().unwrap()],
        [wj["size"][0].as_u64().unwrap() as u32, wj["size"][1].as_u64().unwrap() as u32], &array("water_row", 4), &array("water_x", 2));
    let (lon, lat) = (head["origin"]["lon"].as_f64().unwrap(), head["origin"]["lat"].as_f64().unwrap());
    let game = || {
        let mut a = api();
        assert!(a.set_empty_frame(1));
        a.add_pack_water(m.cell, m.x0, m.y0, m.w, m.h, &m.row, &m.xs, lon, lat);
        a
    };
    let lines: [(&str, [f64; 4]); 14] = [
        ("Roosevelt Island", [-73.9640, 40.7600, -73.9380, 40.7640]),
        ("Randalls and Wards Island", [-73.9400, 40.7960, -73.9050, 40.7850]),
        ("Bronx Kill", [-73.9230, 40.7880, -73.9150, 40.8080]),
        ("Newtown Creek", [-73.9460, 40.7300, -73.9330, 40.7450]),
        ("Gowanus Canal", [-73.9980, 40.6780, -73.9800, 40.6760]),
        ("Harlem River at 155th St", [-73.9400, 40.8300, -73.9220, 40.8300]),
        ("East River at Williamsburg", [-73.9800, 40.7130, -73.9580, 40.7130]),
        ("Flushing Creek", [-73.8480, 40.7600, -73.8250, 40.7600]),
        ("Jamaica Bay by Broad Channel", [-73.8200, 40.6500, -73.8150, 40.5880]),
        ("Mill Basin", [-73.9150, 40.6200, -73.8950, 40.6050]),
        ("Passaic and Hackensack", [-74.1450, 40.7350, -74.0650, 40.7350]),
        ("Spuyten Duyvil", [-73.9300, 40.8730, -73.9150, 40.8800]),
        ("Kill Van Kull", [-74.1250, 40.6500, -74.1250, 40.6330]),
        ("Hudson at Hoboken", [-74.0120, 40.7450, -74.0350, 40.7450]),
    ];
    let mut out: Vec<(String, Result<(), String>)> = vec![];
    let (mut gaps_seen, mut wet_seen) = (0, 0);
    for (name, l) in lines {
        let a = game();
        let (p, q) = (a.from_lonlat(l[0], l[1]), a.from_lonlat(l[2], l[3]));
        let wet = |x: f64, y: f64| a.w.net.water_mask.is_water(x, y);
        if wet(p[0], p[1]) || wet(q[0], q[1]) {
            println!("{name}: an end is in water, skipped");
            continue;
        }
        let k = crate::track::frame::merc_k(p[1]);
        let len = (q[0] - p[0]).hypot(q[1] - p[1]);
        let (ux, uy) = ((q[0] - p[0]) / len, (q[1] - p[1]) / len);
        let at = |s: f64| (p[0] + ux * s, p[1] + uy * s);
        // the water along the line, plane m
        let mut spans: Vec<[f64; 2]> = vec![];
        let mut s = 0.0;
        while s <= len {
            let (x, y) = at(s);
            if wet(x, y) {
                match spans.last_mut() {
                    Some(w) if s - w[1] < 6.0 => w[1] = s,
                    _ => spans.push([s, s]),
                }
            }
            s += 5.0;
        }
        println!("{name}: water at {:?} (ground m)", spans.iter().map(|w| [(w[0] / k).round(), (w[1] / k).round()]).collect::<Vec<_>>());
        if spans.is_empty() {
            continue;
        }
        let pt = |s: f64| { let (x, y) = at(s); [0.0, 0.0, x, y, 0.0] };
        let side = |s: f64, off: f64| { let (x, y) = at(s); [0.0, 0.0, x - uy * off * k, y + ux * off * k, 0.0] };
        let near = |a: &TrackApi, s: f64| { let (x, y) = at(s); (0..a.w.net.node_count() as u32).filter(|&n| a.w.net.node_ok(n)).min_by(|&m, &n| ((a.w.net.node_x[m as usize] - x).hypot(a.w.net.node_y[m as usize] - y)).total_cmp(&(a.w.net.node_x[n as usize] - x).hypot(a.w.net.node_y[n as usize] - y))).unwrap() };
        let mut b = game();
        b.set_route_tangent(true);
        out.push((format!("{name}: straight over all"), route(&mut b, pt(0.0), pt(len))));
        // islands and strips of land between two stretches of water
        for w in spans.windows(2) {
            let gap = w[1][0] - w[0][1];
            if gap / k > 600.0 {
                continue;
            }
            gaps_seen += 1;
            let mid = (w[0][1] + w[1][0]) / 2.0;
            let mut b = game();
            b.set_route_tangent(true);
            let r = route(&mut b, pt(0.0), pt(mid));
            out.push((format!("{name}: a click on land {:.0} m wide between water", gap / k), r.clone()));
            if r.is_ok() {
                let n = near(&b, mid);
                out.push((format!("{name}: on from land {:.0} m wide", gap / k), route(&mut b, node_end(n), pt(len))));
            }
        }
        // a click short of each stretch of water, then on; a segment ending just past each
        for w in &spans {
            wet_seen += 1;
            for d in [60.0, 120.0, 180.0] {
                let s0 = w[0] - d * k;
                let (x, y) = at(s0);
                if s0 <= 0.0 || wet(x, y) {
                    continue;
                }
                for (turn, to) in [("straight", pt(len)), ("turning", side(len, 600.0))] {
                    let mut b = game();
                    b.set_route_tangent(true);
                    if route(&mut b, pt(0.0), pt(s0)).is_err() {
                        continue;
                    }
                    let n = near(&b, s0);
                    out.push((format!("{name}: a click {d} m short of water at {:.0} m, on {turn}", w[0] / k), route(&mut b, node_end(n), to)));
                }
                let s1 = w[1] + d * k;
                let (x, y) = at(s1);
                if s1 < len && !wet(x, y) {
                    let mut b = game();
                    b.set_route_tangent(true);
                    out.push((format!("{name}: ending {d} m past water at {:.0} m", w[1] / k), route(&mut b, pt(0.0), pt(s1))));
                }
            }
        }
    }
    let refused: Vec<_> = out.iter().filter(|r| r.1.is_err()).collect();
    for (name, r) in &refused {
        println!("REFUSED {name}: {}", r.as_ref().unwrap_err());
    }
    println!("T-2414 NYC sweep: {} of {} refused ({gaps_seen} narrow land gaps, {wet_seen} stretches of water)", refused.len(), out.len());
    if std::env::var("T2414_STRICT").is_ok() {
        assert!(refused.is_empty());
    }
}

/// Land kept up between two channels is priced as viaduct: the segment costs what the same
/// segment drawn with that stretch at +1 does.
#[test]
fn land_kept_up_is_priced_as_viaduct() {
    let mut a = api();
    let bands = [(1000.0, 1150.0), (1450.0, 1600.0)];
    water(&mut a, &bands);
    let (f, t) = (free(&a, 0.0, 0.0), free(&a, 3500.0, 0.0));
    let e: Vec<f64> = f.iter().chain(t.iter()).copied().collect();
    assert_eq!(a.add_route(&e, &[], 2, false), 0, "{}", a.issue_kinds());
    let edge = alive(&a)[0];
    let kept = a.w.net.edge_cost[edge as usize];
    // the land between the channels is up
    assert!((geom::height_at(a.w.net.edge_vert(edge), 1300.0) - LEVEL_H).abs() < 1e-6);
    // the same shape and levels, drawn by hand as corners
    let v = a.w.net.edge_data(edge).unwrap().pis;
    let mut b = api();
    water(&mut b, &bands);
    let pis: Vec<f64> = v.iter().flat_map(|p| [p.x, p.y, p.radius, p.level as f64]).collect();
    assert_eq!(b.add_route(&e, &pis, 2, false), 0, "{}", b.issue_kinds());
    let drawn: f64 = alive(&b).iter().map(|&x| b.w.net.edge_cost[x as usize]).sum();
    assert!((kept - drawn).abs() < 1e-6 * kept.max(1.0), "kept up {kept}, drawn {drawn}");
}
