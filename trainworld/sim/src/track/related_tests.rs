use super::*;

/// A route through `path` (node, x, y) on straight edges `edges[k]` (edge << 1 | dir) from node k
/// to node k + 1; `stops` are path positions.
fn route(line: u32, path: &[(u32, f64, f64)], edges: &[u32], stops: &[usize]) -> Route {
    let mut r = Route { line, ..Default::default() };
    let mut s = 0.0;
    for (i, &(n, x, y)) in path.iter().enumerate() {
        if i > 0 {
            let (_, px, py) = path[i - 1];
            s += (x - px).hypot(y - py);
        }
        r.pts.push([x, y]);
        r.ps.push(s);
        r.nodes.push(n);
        r.node_s.push(s);
        r.stop_at.push(stops.iter().position(|&k| k == i).map_or(-1, |k| k as i32));
    }
    r.segs = edges.to_vec();
    r
}

/// The local: stations 0..5 a km apart along y = 0, on edges 100..104, stopping everywhere.
fn local() -> Route {
    let path: Vec<(u32, f64, f64)> = (0..6).map(|i| (i, i as f64 * 1000.0, 0.0)).collect();
    route(0, &path, &[200, 202, 204, 206, 208], &[0, 1, 2, 3, 4, 5])
}

/// Each station is its own complex, except those `same` puts with another.
fn cx_with(same: &[(u32, u32)]) -> impl Fn(u32) -> u32 + '_ {
    move |n: u32| {
        if n >= 1000 {
            return NONE;
        }
        same.iter().find(|p| p.0 == n).map_or(n, |p| p.1)
    }
}

#[derive(Debug)]
struct G {
    line: u32,
    kind: u32,
    shared: bool,
    dir: f64,
    sl: [f64; 2],
    before: usize,
    after: usize,
    split: [f64; 3],
    meets: Vec<[f64; 5]>,
    stretches: Vec<[f64; 2]>,
    gaps: Vec<[f64; 5]>,
    own: Vec<[f64; 3]>,
}

fn decode(v: &[f64]) -> Vec<G> {
    let mut out = vec![];
    let mut i = 0;
    while i < v.len() {
        let h = &v[i..i + 17];
        let (nm, ns, ng, no) = (h[13] as usize, h[14] as usize, h[15] as usize, h[16] as usize);
        i += 17;
        let take = |i: &mut usize, n: usize, w: usize| -> Vec<Vec<f64>> {
            let r = (0..n).map(|k| v[*i + k * w..*i + (k + 1) * w].to_vec()).collect();
            *i += n * w;
            r
        };
        let meets = take(&mut i, nm, 5).into_iter().map(|x| [x[0], x[1], x[2], x[3], x[4]]).collect();
        let stretches = take(&mut i, ns, 2).into_iter().map(|x| [x[0], x[1]]).collect();
        let gaps = take(&mut i, ng, 5).into_iter().map(|x| [x[0], x[1], x[2], x[3], x[4]]).collect();
        let own = take(&mut i, no, 3).into_iter().map(|x| [x[0], x[1], x[2]]).collect();
        out.push(G { line: h[0] as u32, kind: h[1] as u32, shared: h[2] == 1.0, dir: h[3], sl: [h[4], h[5]], before: h[8] as usize, after: h[9] as usize, split: [h[10], h[11], h[12]], meets, stretches, gaps, own });
    }
    out
}

fn run(l: &Route, o: &[Route], same: &[(u32, u32)]) -> Vec<G> {
    decode(&related(l, o, &cx_with(same), &|_| 1.0))
}

#[test]
fn copied_express_shares_all_track() {
    let l = local();
    let path: Vec<(u32, f64, f64)> = (0..6).map(|i| (i, i as f64 * 1000.0, 0.0)).collect();
    let x = route(1, &path, &[200, 202, 204, 206, 208], &[0, 2, 5]);
    let g = run(&l, &[x.clone()], &[]);
    assert_eq!(g.len(), 1, "{g:?}");
    let g = &g[0];
    assert_eq!((g.line, g.kind, g.shared, g.dir), (1, 1, true, 1.0));
    assert_eq!(g.sl, [0.0, 5000.0]);
    assert_eq!(g.meets.len(), 6, "stations passed count as meetings");
    assert_eq!(g.meets.iter().map(|m| m[4] as i32).collect::<Vec<_>>(), vec![0, -1, 1, -1, -1, 2]);
    assert_eq!(g.meets.iter().map(|m| m[3] as i32).collect::<Vec<_>>(), vec![0, 1, 2, 3, 4, 5]);
    assert_eq!(g.stretches, vec![[0.0, 5000.0]]);
    assert!(g.gaps.is_empty() && g.own.is_empty());
    assert_eq!((g.before, g.after), (0, 0));
    // and the other way round: the local seen from the express is beside it too
    let g = run(&x, &[l], &[]);
    assert_eq!((g.len(), g[0].kind, g[0].shared), (1, 1, true));
}

#[test]
fn copied_line_running_the_other_way() {
    let l = local();
    let path: Vec<(u32, f64, f64)> = (0..6).rev().map(|i| (i, i as f64 * 1000.0, 0.0)).collect();
    let x = route(1, &path, &[209, 207, 205, 203, 201], &[0, 3, 5]);
    let g = run(&l, &[x], &[]);
    assert_eq!(g.len(), 1, "{g:?}");
    assert_eq!((g[0].kind, g[0].shared, g[0].dir), (1, true, -1.0));
    assert_eq!(g[0].sl, [0.0, 5000.0]);
}

#[test]
fn branch_on_shared_track() {
    let l = local();
    let mut path: Vec<(u32, f64, f64)> = (0..4).map(|i| (i, i as f64 * 1000.0, 0.0)).collect();
    path.extend([(10, 3700.0, 700.0), (11, 4400.0, 1400.0)]);
    let b = route(1, &path, &[200, 202, 204, 300, 302], &[0, 2, 3, 4, 5]);
    let g = run(&l, &[b], &[]);
    assert_eq!(g.len(), 1, "{g:?}");
    let g = &g[0];
    assert_eq!((g.kind, g.shared), (1, true));
    assert_eq!(g.sl, [0.0, 3000.0]);
    assert_eq!((g.before, g.after), (0, 2), "two stops of its own past where it leaves");
}

#[test]
fn parallel_line_on_its_own_track() {
    let l = local();
    let path: Vec<(u32, f64, f64)> = (0..6).map(|i| (20 + i, i as f64 * 1000.0, 60.0)).collect();
    let p = route(1, &path, &[400, 402, 404, 406, 408], &[0, 1, 2, 3, 4, 5]);
    let same: Vec<(u32, u32)> = (0..6).map(|i| (20 + i, i)).collect();
    let g = run(&l, &[p], &same);
    assert_eq!(g.len(), 1, "{g:?}");
    let g = &g[0];
    assert_eq!((g.kind, g.shared), (1, false));
    assert_eq!(g.meets.len(), 6);
    assert_eq!(g.gaps.len(), 5);
    assert!(g.gaps.iter().all(|x| (x[2] - 60.0).abs() < 1.0 && (x[3] - 1000.0).abs() < 1.0), "{:?}", g.gaps);
}

#[test]
fn crossing_once() {
    let l = local();
    let c = route(1, &[(30, 2000.0, -3000.0), (31, 2000.0, 50.0), (32, 2000.0, 3000.0)], &[500, 502], &[0, 1, 2]);
    let g = run(&l, &[c], &[(31, 2)]);
    assert_eq!(g.len(), 1, "{g:?}");
    assert_eq!((g[0].kind, g[0].shared), (2, false));
    assert_eq!(g[0].meets.len(), 1);
    assert_eq!((g[0].meets[0][3], g[0].meets[0][4]), (2.0, 1.0));
    assert_eq!((g[0].before, g[0].after), (1, 1));
}

/// O leaves L at station 1 and comes back at station 4, through its own stop 40 at (2500, `off`);
/// it also stops at 41 beyond, so not every stop of it is on L's route.
fn detour(off: f64) -> Route {
    route(1, &[(41, 1000.0, -2000.0), (21, 1000.0, 30.0), (40, 2500.0, off), (24, 4000.0, 30.0)], &[600, 602, 604], &[0, 1, 2, 3])
}

#[test]
fn meeting_twice_within_the_cutoff_is_beside() {
    let l = local();
    // 150 m off over a 3 km run: within 100 m + 3.5% of 3 km = 205 m
    let g = run(&l, &[detour(150.0)], &[(21, 1), (24, 4)]);
    assert_eq!(g.len(), 1, "{g:?}");
    let g = &g[0];
    assert_eq!((g.kind, g.shared), (1, false));
    assert_eq!(g.sl, [1000.0, 4000.0]);
    assert_eq!(g.gaps.len(), 1);
    assert!((g.gaps[0][2] - 150.0).abs() < 1.0, "{:?}", g.gaps);
    assert_eq!(g.own.len(), 1, "its own stop between");
    assert_eq!(g.own[0][0], 2.0);
    assert!((g.own[0][1] - 2500.0).abs() < 1.0, "placed at the nearest point of L: {:?}", g.own);
    assert_eq!(g.own[0][2], 40.0, "with its complex");
    assert_eq!(g.before, 1);
}

#[test]
fn touching_only_twice_gets_no_short_run_allowance() {
    let l = local();
    // 180 m over 3 km: within 205 m, but past 100 m + 2% of 3 km = 160 m for a line meeting twice
    let g = run(&l, &[detour(180.0)], &[(21, 1), (24, 4)]);
    assert_eq!(g.len(), 2, "{g:?}");
    assert!(g.iter().all(|x| x.kind == 2));
    assert!((g[1].split[0] - 180.0).abs() < 1.0);
}

#[test]
fn meeting_twice_beyond_the_cutoff_is_two_crossings() {
    let l = local();
    let g = run(&l, &[detour(800.0)], &[(21, 1), (24, 4)]);
    assert_eq!(g.len(), 2, "{g:?}");
    assert!(g.iter().all(|x| x.kind == 2));
    assert!(g[0].split[0].is_nan());
    assert!((g[1].split[0] - 800.0).abs() < 1.0, "the second names the gap: {:?}", g[1].split);
    assert!((g[1].split[2] - 3000.0).abs() < 1.0);
}

#[test]
fn a_line_with_every_stop_on_the_route_is_beside() {
    let l = local();
    let o = route(1, &[(21, 1000.0, 30.0), (40, 2500.0, 800.0), (24, 4000.0, 30.0)], &[602, 604], &[0, 2]);
    let g = run(&l, &[o], &[(21, 1), (24, 4)]);
    assert_eq!(g.len(), 1, "{g:?}");
    assert_eq!(g[0].kind, 1);
    assert!((g[0].gaps[0][2] - 800.0).abs() < 1.0);
}

#[test]
fn turning_back_along_the_line_splits() {
    let l = local();
    // a line that runs 1 -> 3 beside L and then comes back to 2 on its own track
    let o = route(1, &[(21, 1000.0, 30.0), (23, 3000.0, 30.0), (50, 3000.0, 300.0), (22, 2000.0, 300.0), (51, 2000.0, 900.0)], &[700, 702, 704, 706], &[0, 1, 3, 4]);
    let g = run(&l, &[o], &[(21, 1), (23, 3), (22, 2)]);
    assert_eq!(g.len(), 2, "{g:?}");
    assert_eq!(g[0].kind, 1);
    assert_eq!(g[1].kind, 2);
    assert!(g[1].split[0].is_nan(), "turned back rather than strayed");
}

#[test]
fn a_loop_meets_at_its_seam_once() {
    // L a loop round a square, its first station again at the end; O crosses at that station
    let path = [(0, 0.0, 0.0), (1, 2000.0, 0.0), (2, 2000.0, 2000.0), (3, 0.0, 2000.0), (0, 0.0, 0.0)];
    let l = route(0, &path, &[200, 202, 204, 206], &[0, 1, 2, 3, 4]);
    let o = route(1, &[(60, -2000.0, 1.0), (61, -30.0, 0.0), (62, -30.0, -2000.0)], &[800, 802], &[0, 1, 2]);
    let g = run(&l, &[o], &[(61, 0)]);
    assert_eq!(g.len(), 1, "{g:?}");
    assert_eq!(g[0].kind, 2);
}
