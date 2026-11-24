use super::*;

/// One line west to east, stops every `gap` m, each its own place, all bus.
fn line(n: usize, gap: f32) -> (Vec<f32>, Vec<u32>, Vec<u32>, Vec<f32>) {
    let xy: Vec<f32> = (0..n).flat_map(|i| [i as f32 * gap, 0.0]).collect();
    let stops: Vec<u32> = (0..n as u32).collect();
    let off = vec![0, n as u32];
    // run 0: 60 s a hop, run 1: 90 s
    let mut times = vec![0f32; TIMES_PER_STOP * n];
    for q in 0..PERIODS {
        for k in 0..n {
            if k + 1 < n {
                times[q * n + k] = 60.0;
            }
            if k > 0 {
                times[(PERIODS + q) * n + k] = 90.0;
            }
        }
    }
    (xy, off, stops, times)
}

fn lay(xy: &[f32], off: &[u32], stops: &[u32], times: &[f32], ok: bool) -> Layout {
    let n = xy.len() / 2;
    let map: Vec<u32> = (0..n as u32).collect();
    let (px, py): (Vec<f32>, Vec<f32>) = (xy.iter().step_by(2).copied().collect(), xy.iter().skip(1).step_by(2).copied().collect());
    let place_ok = vec![ok; n];
    layout(&Real { st_xy: xy, off, stops, times, st_map: &map, px: &px, py: &py, place_ok: &place_ok }, 300.0)
}

/// T-4094: stops 200 m apart pair up (the third is 400 m from the first); times keep their sum but the
/// end runs' halves; nothing merges where no place may.
#[test]
fn runs_along_a_line() {
    let (xy, off, stops, times) = line(6, 200.0);
    let l = lay(&xy, &off, &stops, &times, true);
    let f = l.fold.as_ref().expect("merged");
    assert_eq!(l.stops, vec![0, 1, 2], "{:?}", l.stops);
    assert_eq!(l.st_map, vec![0, 0, 1, 1, 2, 2]);
    assert_eq!(f.sizes(), (12, 6));
    // run 0: [0,1] [2,3] [4,5]: hops 30 + 60 + 30, 30 + 60 + 30, 0
    assert_eq!(&l.times[..3], &[120.0, 120.0, 0.0]);
    // run 1 (stop k to k - 1): 0 at the first
    assert_eq!(&l.times[PERIODS * 3..PERIODS * 3 + 3], &[0.0, 180.0, 180.0]);
    assert_eq!(l.stop_xy, vec![100.0, 0.0, 500.0, 0.0, 900.0, 0.0]);
    let none = lay(&xy, &off, &stops, &times, false);
    assert!(none.fold.is_none() && none.stops == stops);
}

/// T-4094: split figures keep their totals, the loads inside a run move between the load in and out
/// as the split boardings and alightings say, and the crowding totals fold back to the demand ones.
#[test]
fn out_and_back() {
    let (xy, off, stops, times) = line(6, 200.0);
    let mut l = lay(&xy, &off, &stops, &times, true);
    let f = l.fold.as_mut().unwrap();
    // cells: a big one by stop 1, a small one by stop 0
    f.weigh(&[200.0, 0.0], &[10.0, 10.0], &[900.0, 100.0]);
    // demand, run 0: 10 board at the first group, 4 more and 6 off at the second, the rest off at the last
    let (board, alight, seg) = (vec![10.0, 4.0, 0.0, 0.0, 0.0, 0.0], vec![0.0, 6.0, 8.0, 0.0, 0.0, 0.0], vec![10.0, 8.0, 0.0, 0.0, 0.0, 0.0]);
    let (b, a, s) = (f.out_split(&board), f.out_split(&alight), f.out_load(&seg));
    assert!((b.iter().sum::<f32>() - 14.0).abs() < 1e-5 && (a.iter().sum::<f32>() - 14.0).abs() < 1e-5);
    assert!(b[1] > b[0] && b[0] > 0.0, "stop 1 has the bigger catchment, stop 0 some: {b:?}");
    let mut load = 0f32;
    for r in 0..6 {
        load += b[r] - a[r];
        assert!((s[r] - load).abs() < 1e-4, "route node {r}: {s:?} against {load}");
    }
    let total: Vec<f32> = [s.clone(), b.clone(), a.clone()].concat();
    let back = f.in_total(&total);
    for (x, y) in back.iter().zip([seg, board, alight].concat()) {
        assert!((x - y).abs() < 1e-4, "{back:?}");
    }
    let pairs = f.out_pairs(&[1.0, 4.0, 3.0]);
    assert!((pairs.chunks(3).map(|t| t[2]).sum::<f32>() - 3.0).abs() < 1e-5, "{pairs:?}");
}

/// T-4094: a line that would be left with one stop keeps its places apart.
#[test]
fn never_one_stop() {
    let (xy, off, stops, times) = line(2, 150.0);
    assert!(lay(&xy, &off, &stops, &times, true).fold.is_none());
}

/// T-4094: a line that ends where it began (no ride on round) stays a line: its last hop and its reverse run's
/// first stay 0, though its first and last stops fold to one group.
#[test]
fn back_to_the_start_is_no_circle() {
    let xy: Vec<f32> = (0..4).flat_map(|i| [i as f32 * 200.0, 0.0]).collect();
    let stops = vec![0u32, 1, 2, 3, 0];
    let off = vec![0u32, 5];
    let n = 5;
    let mut times = vec![0f32; TIMES_PER_STOP * n];
    for q in 0..PERIODS {
        for k in 0..n - 1 {
            times[q * n + k] = 60.0;
            times[(PERIODS + q) * n + k + 1] = 60.0;
        }
    }
    let l = lay(&xy, &off, &stops, &times, true);
    assert!(l.fold.is_some());
    let m = l.stops.len();
    assert_eq!(l.stops.first(), l.stops.last(), "{:?}", l.stops);
    assert_eq!(l.times[m - 1], 0.0, "{:?}", &l.times[..m]);
    assert_eq!(l.times[PERIODS * m], 0.0, "{:?}", &l.times[PERIODS * m..PERIODS * m + m]);
}
