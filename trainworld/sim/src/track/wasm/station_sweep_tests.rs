//! T-1964: station placements across joints between pieces of track. A sweep asks the track model
//! for a 200 m station every `STEP` m along every piece, classifies each refusal by its reason and
//! by the joints the platform would span (walked through plain nodes, half a platform and 2 m each
//! way), and retries a refused spot at the nearest spot within half a platform that fits, as the
//! station tool slides it (T-1870). Two networks: the real save (its network bytes in `T1964_NET`,
//! `python trainworld/checks/save_net.py <save> <out>`) and a fresh network drawn as the track tool
//! draws, click by click. Prints counts per reason; the drawn one also asserts there are no
//! refusals over plain joints.
//!   cargo test --manifest-path sim/Cargo.toml --release station_sweep -- --ignored --nocapture
use super::*;
use std::collections::BTreeMap;

const PLATFORM: u16 = 200;

fn live(a: &TrackApi) -> Vec<u32> {
    (0..a.w.net.edge_count() as u32).filter(|&e| a.w.net.edge_ok(e)).collect()
}

/// What a joint between `cur` and the next piece at `node` is, for a platform reaching over it.
fn joint_kind(a: &TrackApi, node: u32, cur: u32) -> (&'static str, Option<(u32, u32)>) {
    let net = &a.w.net;
    let p = net.node_ports[node as usize];
    if net.node_platform[node as usize] > 0 {
        return ("station", None);
    }
    if p.n <= 1 {
        return ("end", None);
    }
    if p.n > 2 {
        return ("junction", None);
    }
    let Some((o, which)) = p.ends().iter().map(|&c| (c >> 1, c & 1)).find(|&(o, _)| o != cur) else { return ("loop", None) };
    let (d, od) = (net.edge_data(cur).unwrap(), net.edge_data(o).unwrap());
    let next = Some((o, net.end_node(o, 1 - which)));
    if d.built != od.built {
        return ("built|blueprint", next);
    }
    if d.off != od.off || d.crossover || od.crossover {
        return ("off|on or crossover", next);
    }
    if d.tracks != od.tracks {
        return ("track count", next);
    }
    if d.hsr != od.hsr {
        return ("high speed|not", next);
    }
    if !d.thru.is_empty() || !od.thru.is_empty() {
        return ("drawn through points", next);
    }
    let lvl = net.node_level[node as usize];
    let near_v = |e: u32| {
        let v = net.edge_vertices(e);
        if net.edge_a[e as usize] == node { v[1] } else { v[v.len() - 2] }
    };
    if near_v(cur).level != lvl || near_v(o).level != lvl {
        return ("level change", next);
    }
    let own = vertices_of(net, &d, &[]);
    let corner = if net.edge_a[cur as usize] == node { own[1] } else { own[own.len() - 2] };
    if a.free_joint(node, cur, corner, true).is_none() {
        if std::env::var("T1964_DEBUG").is_ok() {
            let ov = vertices_of(net, &od, &[]);
            let far = if which == 0 { ov[1] } else { ov[ov.len() - 2] };
            let (dx, dy) = (far.x - corner.x, far.y - corner.y);
            let l = dx.hypot(dy);
            let (jx, jy) = (net.node_x[node as usize] - corner.x, net.node_y[node as usize] - corner.y);
            eprintln!("corner at node {node}: t {:.3} off {:.3} m, pis {} / {}, near-far {l:.0} m", (jx * dx + jy * dy) / (l * l), (jx * dy - jy * dx).abs() / l, d.pis.len(), od.pis.len());
        }
        return ("corner", next);
    }
    ("plain", next)
}

/// The joints within `reach` either way of chainage `s` on edge `e`, by kind, and what ends the walk.
fn joints_under(a: &TrackApi, e: u32, s: f64, reach: f64) -> Vec<&'static str> {
    let net = &a.w.net;
    let mut out = vec![];
    for which in [0u32, 1] {
        let (mut cur, mut node) = (e, net.end_node(e, which));
        let mut gap = if which == 0 { s } else { net.edge_len[e as usize] - s };
        let mut seen = vec![e];
        while gap < reach {
            let (k, next) = joint_kind(a, node, cur);
            out.push(k);
            let Some((o, far)) = next else { break };
            if seen.contains(&o) {
                break;
            }
            seen.push(o);
            gap += net.edge_len[o as usize];
            cur = o;
            node = far;
        }
    }
    out
}

/// Spots along the track from (e, s), `d` m away either way through plain nodes: (edge, chainage).
fn along(a: &TrackApi, e: u32, s: f64, d: f64) -> Option<(u32, f64)> {
    let net = &a.w.net;
    let (mut cur, mut t) = (e, s + d);
    let mut seen = vec![e];
    loop {
        let len = net.edge_len[cur as usize];
        if (0.0..=len).contains(&t) {
            return Some((cur, t));
        }
        let which = if t < 0.0 { 0 } else { 1 };
        let over = if t < 0.0 { -t } else { t - len };
        let node = net.end_node(cur, which);
        let p = net.node_ports[node as usize];
        if p.n != 2 || net.node_platform[node as usize] > 0 {
            return None;
        }
        let (o, ow) = p.ends().iter().map(|&c| (c >> 1, c & 1)).find(|&(o, _)| o != cur)?;
        if seen.contains(&o) {
            return None;
        }
        seen.push(o);
        cur = o;
        t = if ow == 0 { over } else { net.edge_len[o as usize] - over };
    }
}

fn try_at(a: &mut TrackApi, e: u32, s: f64) -> Option<String> {
    let (x, y, _) = geom::pos_at(a.w.net.edge_pieces(e), s);
    (a.preview_station(2, e, x, y, PLATFORM) != 0).then(|| a.issue_kinds())
}

#[derive(Default)]
pub(super) struct Sweep {
    pub samples: usize,
    pub fits: usize,
    pub slid: usize,
    /// refusals at the spot itself, by reason and the joints the platform spans
    pub raw: BTreeMap<String, usize>,
    /// refusals left after the slide, the same way
    pub left: BTreeMap<String, usize>,
    pub examples: Vec<String>,
}

pub(super) fn sweep(a: &mut TrackApi, step: f64) -> Sweep {
    let mut out = Sweep::default();
    let reach = PLATFORM as f64 / 2.0 + 2.0;
    for e in live(a) {
        if !a.w.net.edge_ok(e) || a.w.net.edge_crossover[e as usize] {
            continue;
        }
        let len = a.w.net.edge_len[e as usize];
        let mut s = step / 2.0;
        while s < len {
            out.samples += 1;
            if out.samples % 2000 == 0 {
                eprintln!("{} samples, {} fit", out.samples, out.fits);
            }
            let Some(why) = try_at(a, e, s) else {
                out.fits += 1;
                s += step;
                continue;
            };
            let mut joints = joints_under(a, e, s, reach);
            joints.sort();
            joints.dedup();
            let key = format!("{} | {}", why.replace('\n', "+"), joints.join(","));
            *out.raw.entry(key.clone()).or_default() += 1;
            // the nearest spot within half a platform and 2 m that fits, for a refusal only for room
            // (the station tool's `onlyRoomIssues`)
            const ROOM: [&str; 5] = ["PlatformTooLong", "PlatformStation", "PlatformJunction", "PlatformJoint", "SplitPoint"];
            let mut slid = false;
            let mut d = if why.split('\n').all(|k| ROOM.contains(&k)) { 2.0 } else { f64::INFINITY };
            while d <= reach && !slid {
                for sign in [1.0, -1.0] {
                    if let Some((e2, s2)) = along(a, e, s, sign * d) {
                        if try_at(a, e2, s2).is_none() {
                            slid = true;
                            break;
                        }
                    }
                }
                d += 2.0;
            }
            if slid {
                out.slid += 1;
            } else {
                *out.left.entry(key.clone()).or_default() += 1;
                if out.examples.len() < 400 && joints.iter().any(|j| *j == "plain") {
                    let (x, y, _) = geom::pos_at(a.w.net.edge_pieces(e), s);
                    out.examples.push(format!("edge {e} s {s:.0} ({x:.0},{y:.0}): {key}"));
                }
            }
            s += step;
        }
    }
    out
}

pub(super) fn report(name: &str, r: &Sweep) {
    eprintln!("== {name}: {} samples, {} fit, {} refused ({} slid to a spot that fits)", r.samples, r.fits, r.samples - r.fits, r.slid);
    eprintln!("-- refused at the spot (reason | joints under the platform):");
    let mut v: Vec<_> = r.raw.iter().collect();
    v.sort_by(|p, q| q.1.cmp(p.1));
    for (k, n) in v {
        eprintln!("{n:>7}  {k}");
    }
    eprintln!("-- still refused after the slide:");
    let mut v: Vec<_> = r.left.iter().collect();
    v.sort_by(|p, q| q.1.cmp(p.1));
    for (k, n) in v {
        eprintln!("{n:>7}  {k}");
    }
    for x in r.examples.iter().take(30) {
        eprintln!("   {x}");
    }
}

/// Two straight pieces meeting at node (1000, 0) at a slight angle, the second ending at (2000, `y`).
fn kinked(y: f64) -> (TrackApi, [u32; 2]) {
    let mut a = TrackApi::new(-73.985, 40.758, true);
    a.set_cash(1e9);
    let n = [a.new_node_id(), a.new_node_id(), a.new_node_id()];
    for (k, x) in [0.0, 1000.0, 2000.0].into_iter().enumerate() {
        assert_eq!(a.set_node(n[k], x, if k == 2 { y } else { 0.0 }, 0, false), 0);
    }
    let e = [a.new_edge_id(), a.new_edge_id()];
    for k in 0..2 {
        let op = Op::Edge { id: e[k], data: Some(EdgeData { a: n[k], b: n[k + 1], tracks: 2, pis: vec![], built: false, off: false, thru: vec![], crossover: false, hsr: false, sep: false }) };
        let r = a.w.apply(op);
        assert!(r.is_ok(), "y {y}: {:?}", r.err());
    }
    let bad = a.w.net.validate(&e, &n, &[]);
    assert!(bad.is_empty(), "{bad:?}");
    (a, e)
}

/// T-1964: a platform over a joint where two pieces meet at a slight angle (as drawing and
/// splitting leave them, within the heading tolerance) joins them, as it does on the straight.
#[test]
fn a_platform_over_a_slightly_bent_joint_joins_the_pieces() {
    // 0.3, 0.75 and 1.45 m off the straight from (0, 0) to the far end: the last kept as a corner
    for y in [0.6, 1.5, 2.9] {
        for x in [960.0, 1000.0, 1040.0] {
            let (mut a, e) = kinked(y);
            let (s, _) = a.w.net.project(e[0], x, 0.0);
            let on = if s < a.w.net.edge_len[e[0] as usize] - 1.0 { e[0] } else { e[1] };
            let got = a.add_station(2, on, x, 0.0, 200, "P".into());
            assert_eq!(got, 0, "y {y} at {x}: {}", a.issues());
            assert_eq!(live(&a).len(), 2, "y {y} at {x}: one piece each side of the station");
            // the track moved by no more than the joint was off the straight
            for k in 0..=20 {
                let px = 100.0 * k as f64;
                let want = if px <= 1000.0 { 0.0 } else { y * (px - 1000.0) / 1000.0 };
                let off = live(&a).iter().map(|&e| a.w.net.project(e, px, want).1).fold(f64::INFINITY, f64::min);
                assert!(off < 1.0, "y {y} at {x}: track {off:.2} m off at x {px}");
            }
        }
    }
}

/// T-1964: a station within 5 m of a joint that stays (built beside blueprint) goes on the joint.
#[test]
fn a_station_by_a_joint_of_unlike_pieces_goes_on_it() {
    let (mut a, e) = kinked(0.0);
    assert_eq!(a.construct(&[e[0]], &[]), 0, "{}", a.issues());
    let joint = a.w.net.edge_b[e[0] as usize];
    for x in [996.0, 1003.0] {
        let on = if x < 1000.0 { e[0] } else { e[1] };
        assert_eq!(a.add_station(2, on, x, 0.0, 200, "P".into()), 0, "at {x}: {}", a.issues());
        assert!(a.w.net.node_platform[joint as usize] > 0, "at {x}: on the joint");
        assert_eq!(a.undo(), 0);
    }
    // further off, the platform reaches the joint
    assert_ne!(a.preview_station(2, e[1], 1030.0, 0.0, 200), 0);
    assert_eq!(a.issue_kinds(), "PlatformJoint");
}

/// T-1964: a split refused on a ramp says the platform would be on a ramp.
#[test]
fn a_station_on_a_ramp_says_so() {
    let mut a = TrackApi::new(-73.985, 40.758, true);
    a.set_cash(1e9);
    let mut ends = vec![0.0, 0.0, 0.0, 0.0, 0.0];
    ends.extend([0.0, 0.0, 2000.0, 0.0, -1.0]);
    assert_eq!(a.add_route(&ends, &[], 2, false), 0, "{}", a.issues());
    let e = live(&a)[0];
    let vert = a.w.net.edge_vert(e).to_vec();
    let mid = vert.windows(2).find(|w| w[0][1] != w[1][1]).map(|w| (w[0][0] + w[1][0]) / 2.0).expect("a ramp");
    let (x, y, _) = geom::pos_at(a.w.net.edge_pieces(e), mid);
    assert_ne!(a.preview_station(2, e, x, y, 200), 0);
    assert_eq!(a.issue_kinds(), "PlatformNotLevel");
}

#[test]
#[ignore]
fn station_sweep_real_save() {
    let raw = std::fs::read(std::env::var("T1964_NET").expect("T1964_NET: the save's network bytes")).unwrap();
    let mut a = TrackApi::new(-73.985, 40.758, true);
    assert!(a.load(&super::save::convert_frame(&raw, FrameKind::Merc).unwrap()));
    a.cash = 1e9;
    let step = std::env::var("T1964_STEP").ok().and_then(|s| s.parse().ok()).unwrap_or(50.0);
    let r = sweep(&mut a, step);
    report("real save", &r);
}

/// A network drawn click by click as the track tool draws (tangent arcs from each segment's end),
/// with level changes, built and blueprint pieces, single and double track, and branches.
pub(super) fn drawn() -> TrackApi {
    let mut a = TrackApi::new(-73.985, 40.758, true);
    a.set_cash(1e9);
    a.set_route_tangent(true);
    a.set_route_ramp(true);
    a.set_design_speed(160.0);
    let mut seed: u64 = 1964;
    let mut rnd = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed % 10_000) as f64 / 10_000.0
    };
    for route in 0..14 {
        let (x0, y0) = (route as f64 * 1700.0 - 12000.0, ((route * 7) % 5) as f64 * 2600.0 - 6000.0);
        let tracks = if route % 3 == 0 { 1 } else { 2 };
        let level: i8 = if route % 4 == 1 { -1 } else { 0 };
        let mut heading = rnd() * 6.28;
        let mut ends = vec![0.0, 0.0, x0, y0, level as f64];
        let first = 150.0 + 500.0 * rnd();
        ends.extend([0.0, 0.0, x0 + first * heading.cos(), y0 + first * heading.sin(), level as f64]);
        if a.add_route(&ends, &[], tracks, false) != 0 {
            continue;
        }
        let mut end = a.route_end();
        let mut lvl = level;
        for k in 0..9 {
            let turn = (rnd() - 0.5) * 0.9;
            let len = 120.0 + 700.0 * rnd();
            // now and then a change of level (the ramp lands in this segment or spreads), and the
            // odd segment straight on
            if k % 4 == 3 {
                lvl = if lvl == 0 { -1 } else { 0 };
            }
            let th = heading + if k % 3 == 2 { 0.0 } else { turn };
            let (x, y) = (a.w.net.node_x[end as usize] + len * th.cos(), a.w.net.node_y[end as usize] + len * th.sin());
            let ends = vec![1.0, end as f64, 0.0, 0.0, 0.0, 0.0, 0.0, x, y, lvl as f64];
            if a.add_route(&ends, &[], tracks, false) != 0 {
                continue;
            }
            heading = th + if k % 3 == 2 { 0.0 } else { turn };
            end = a.route_end();
            // a branch now and then, to make junctions
            if k == 5 && route % 2 == 0 {
                let bx = x + 600.0 * (heading + 1.0).cos();
                let by = y + 600.0 * (heading + 1.0).sin();
                let _ = a.add_route(&[1.0, end as f64, 0.0, 0.0, 0.0, 0.0, 0.0, bx, by, lvl as f64], &[], tracks, false);
            }
        }
        // the first half of every third route constructed: joints of built beside blueprint
        if route % 3 == 1 {
            let mine: Vec<u32> = live(&a).into_iter().filter(|&e| !a.w.net.edge_built[e as usize]).collect();
            let half: Vec<u32> = mine.iter().copied().take(mine.len() / 2).collect();
            let _ = a.construct(&half, &[]);
        }
    }
    a
}

#[test]
#[ignore]
fn station_sweep_drawn() {
    let mut a = drawn();
    eprintln!("drawn: {} pieces", live(&a).len());
    let mut kinds: BTreeMap<&str, usize> = BTreeMap::new();
    for e in live(&a) {
        let n = a.w.net.edge_b[e as usize];
        *kinds.entry(joint_kind(&a, n, e).0).or_default() += 1;
    }
    eprintln!("the pieces' far ends: {kinds:?}");
    let step = std::env::var("T1964_DSTEP").ok().and_then(|s| s.parse().ok()).unwrap_or(10.0);
    let r = sweep(&mut a, step);
    report("drawn", &r);
}
