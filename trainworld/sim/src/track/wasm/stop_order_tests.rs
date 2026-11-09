use super::*;
use crate::track::tests::{edge, node, station, world};

fn api() -> TrackApi {
    TrackApi::new(-73.985, 40.758, true)
}

/// `stop_edit_tests::detour_fixture`: A - JA, a direct stretch to JB (10 km) and a long way round
/// through X1 and X2 (about 40 km), JB - B. Coming back from X1 to A a train reverses to go on to B.
fn detour() -> (TrackWorld, [u32; 4]) {
    let mut w = world();
    let a = node(&mut w, 0.0, 0.0, 0);
    let ja = node(&mut w, 1000.0, 0.0, 0);
    let jb = node(&mut w, 11000.0, 0.0, 0);
    let b = node(&mut w, 12000.0, 0.0, 0);
    let x1 = node(&mut w, 4000.0, 15000.0, 0);
    let x2 = node(&mut w, 8000.0, 15000.0, 0);
    edge(&mut w, a, ja, &[], 0).unwrap();
    edge(&mut w, ja, jb, &[], 0).unwrap();
    edge(&mut w, jb, b, &[], 0).unwrap();
    edge(&mut w, ja, x1, &[(2500.0, 0.0), (2500.0, 15000.0)], 0).unwrap();
    edge(&mut w, x1, x2, &[], 0).unwrap();
    edge(&mut w, x2, jb, &[(9500.0, 15000.0), (9500.0, 0.0)], 0).unwrap();
    for n in [a, b, x1, x2] {
        station(&mut w, n, 200);
    }
    (w, [a, x1, x2, b])
}

/// T-2895: an order the player set stays through later stop edits, a click fits a new stop in
/// without moving the others, Sort stops gives the order back to the game, and undo and a save
/// carry the bit.
#[test]
fn a_hand_order_is_kept_until_sorted() {
    let (w, [sa, x1, x2, sb]) = detour();
    let mut a = api();
    a.w = w;
    let id = a.new_line_id();
    assert_eq!(a.set_line(id, &[sa, sb], 6.0, 4.0, 2.0, 30.0, 180.0, 0, 0, "L".into()), 0, "{}", a.issues());
    assert!(!a.line_hand(id));
    assert!(!a.save().windows(4).any(|w| w == b"LHO1"), "no hand order: no block");
    // set_stops straightens this order (stop_edit_tests); order_stops keeps it
    assert_eq!(a.order_stops(id, &[x1, sa, sb]), 0, "{}", a.issues());
    assert_eq!(a.w.net.line_stops(id), &[x1, sa, sb]);
    assert!(a.line_hand(id));
    assert_eq!(a.line_reversals(id), [1.0], "trains reverse at A");
    // a click adds X2 where it costs least, the rest as they were, and nothing straightens them
    let stops: Vec<u32> = a.insert_stop(id, x2).iter().map(|&x| x as u32).collect();
    assert_eq!(a.set_stops(id, &stops), 0, "{}", a.issues());
    let got = a.w.net.line_stops(id).to_vec();
    assert_eq!(got.iter().filter(|&&s| s != x2).copied().collect::<Vec<_>>(), [x1, sa, sb], "{got:?}");
    assert!(a.line_hand(id));
    // a copy keeps the order and the bit
    let copy = a.new_line_id();
    assert_eq!(a.copy_line(id, copy, "L copy".into()), 0, "{}", a.issues());
    assert!(a.line_hand(copy));
    assert_eq!(a.undo(), 0);
    // the save carries it; without the block the line loads as any other
    let bytes = a.save();
    let at = bytes.windows(4).rposition(|w| w == b"LHO1").expect("LHO1 written");
    let mut b = api();
    assert!(b.load(&bytes));
    assert!(b.line_hand(0));
    assert_eq!(b.w.net.line_stops(0).len(), got.len());
    assert!(save::decode(&bytes[..at]).map(|n| n.line_hand.iter().all(|&h| !h)).unwrap());
    // Sort stops: the straight order, either way round, and the bit off
    assert_eq!(a.sort_stops(id), 0, "{}", a.issues());
    assert!(!a.line_hand(id));
    let sorted = a.w.net.line_stops(id).to_vec();
    let mut back = sorted.clone();
    back.reverse();
    assert!(sorted == [sa, x1, x2, sb] || back == [sa, x1, x2, sb], "{sorted:?}");
    assert!(a.line_reversals(id).is_empty());
    assert!(!a.save().windows(4).any(|w| w == b"LHO1"));
    // one undo each: the sort, the click, the order
    assert_eq!(a.undo(), 0);
    assert_eq!((a.w.net.line_stops(id).to_vec(), a.line_hand(id)), (got, true));
    assert_eq!(a.undo(), 0);
    assert_eq!(a.undo(), 0);
    assert_eq!((a.w.net.line_stops(id), a.line_hand(id)), (&[sa, sb][..], false));
}

/// T-2895: S0 - T - J - M along x, a branch J - B leaving J towards M, and two stations on track
/// of their own. From B a train reaches M only by turning at the junction, which no train does.
#[test]
fn an_order_with_no_route_names_its_hop() {
    let mut w = world();
    let s0 = node(&mut w, -3000.0, 0.0, 0);
    let t = node(&mut w, 0.0, 0.0, 0);
    let j = node(&mut w, 3000.0, 0.0, 0);
    let m = node(&mut w, 6000.0, 0.0, 0);
    let b = node(&mut w, 5000.0, 1500.0, 0);
    let z0 = node(&mut w, 0.0, 9000.0, 0);
    let z1 = node(&mut w, 3000.0, 9000.0, 0);
    edge(&mut w, s0, t, &[], 0).unwrap();
    edge(&mut w, t, j, &[], 0).unwrap();
    edge(&mut w, j, m, &[], 0).unwrap();
    edge(&mut w, j, b, &[(3400.0, 0.0)], 0).unwrap();
    edge(&mut w, z0, z1, &[], 0).unwrap();
    for n in [s0, t, m, b, z0, z1] {
        station(&mut w, n, 200);
    }
    let mut a = api();
    a.w = w;
    let id = a.new_line_id();
    assert_eq!(a.set_line(id, &[b, t, m], 6.0, 4.0, 2.0, 30.0, 180.0, 0, 0, "L".into()), 0, "{}", a.issues());
    assert_eq!(a.line_reversals(id), [1.0], "B, T, M reverses at T");
    assert!(a.order_fault(&[b, t, m]).is_empty());
    assert_eq!(a.order_fault(&[t, b, m]), [1.0, b as f64, m as f64]);
    assert_ne!(a.order_stops(id, &[t, b, m]), 0);
    assert_eq!((a.w.net.line_stops(id), a.line_hand(id)), (&[b, t, m][..], false), "a refused order changes nothing");
    assert_eq!(a.order_fault(&[s0, t, z1]), [0.0, t as f64, z1 as f64]);
    assert!(a.order_fault(&[s0, j]).is_empty(), "a junction is no stop: no hop to name");
}
