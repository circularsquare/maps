//! T-3565: where a line's one-way stations go in its stops. A one-way station (`Network::leg_stop`)
//! is served once a round trip, by the run whose track passes it, so its place comes from the
//! track, never from the order it was added in. An open line's stops are a skeleton (every other
//! stop, in the line's order) and a set of one-way stations: the skeleton is routed both ways and
//! each one-way station goes in the gap where the run that passes it does. The same stations make
//! the same line whatever order they were clicked in. Lines ordered by hand (T-2895) keep theirs.

use super::*;

/// `stops` split into the skeleton (in order) and the one-way stations, ends included; None for a
/// loop or a line with no one-way station.
pub(super) fn split(net: &Network, stops: &[u32]) -> Option<(Vec<u32>, Vec<u32>)> {
    if closed(stops) || !stops.iter().any(|&s| net.leg_heading(s).is_some()) {
        return None;
    }
    let (legs, skel): (Vec<u32>, Vec<u32>) = stops.iter().partition(|&&s| net.leg_stop(s));
    (!legs.is_empty()).then_some((skel, legs))
}

/// The line through `skel` (in order) and the one-way stations `legs` (any order), and its path:
/// each one-way station in the gap of the skeleton where the run that passes it does, run 0's in
/// its order, then run 1's in its order reversed (so run 1 meets them in its order). Those neither
/// run passes between two skeleton stops (with fewer than two, or past an end) go in where they
/// cost least (`best_slot`), in station order, tried with the skeleton both ways round and the
/// cheaper kept, so the line is the same whichever way round the skeleton came. None when that
/// leaves no route or no way back.
pub(super) fn flatten(net: &Network, skel: &[u32], legs: &[u32], memo: &mut HopMemo) -> Option<(Vec<u32>, Vec<u32>)> {
    if skel.windows(2).any(|w| w[0] == w[1]) {
        return None;
    }
    let rs: Vec<u32> = skel.iter().rev().copied().collect();
    let (a, left) = match place(net, skel, legs, memo) {
        Some((a, left)) => (Some(a), left),
        None => (None, true),
    };
    if !left || skel.len() < 2 {
        return a;
    }
    let b = place(net, &rs, legs, memo).map(|x| x.0);
    let key = |x: &(Vec<u32>, Vec<u32>)| (order_cost(net, &x.1), x.0.clone().min(x.0.iter().rev().copied().collect()));
    let take_b = match (&a, &b) {
        (Some(a), Some(b)) => {
            let (ka, kb) = (key(a), key(b));
            kb.0 < ka.0 - 0.001 || (kb.0 <= ka.0 + 0.001 && kb.1 < ka.1)
        }
        (None, b) => b.is_some(),
        _ => false,
    };
    if !take_b {
        return a;
    }
    // that line run the other way round, so the skeleton keeps its order
    let stops: Vec<u32> = b?.0.into_iter().rev().collect();
    let path = route_memo(net, &stops, memo)?;
    net.back_run_of(&stops, &path).is_some().then_some((stops, path))
}

/// `flatten` with the skeleton as given, and whether some one-way station went in by cost.
fn place(net: &Network, skel: &[u32], legs: &[u32], memo: &mut HopMemo) -> Option<((Vec<u32>, Vec<u32>), bool)> {
    let mut sorted = legs.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    let mut left: Vec<u32> = vec![];
    let (mut stops, mut path) = if skel.len() < 2 {
        left = sorted;
        (skel.to_vec(), route_memo(net, skel, memo)?)
    } else {
        let fwd = route_memo(net, skel, memo).filter(|p| !p.is_empty())?;
        let rs: Vec<u32> = skel.iter().rev().copied().collect();
        let back = route_memo(net, &rs, memo).filter(|p| !p.is_empty())?;
        let (fnodes, bnodes) = (nodes_of(net, &fwd), nodes_of(net, &back));
        let fat = super::super::net::match_stops(&fnodes, skel, &net.stop_reversals(&fwd))?;
        let bat = super::super::net::match_stops(&bnodes, &rs, &net.stop_reversals(&back))?;
        let m = skel.len();
        // per gap g (between skel[g] and skel[g + 1]): (run, place along that run's path, station)
        let mut gaps: Vec<Vec<(u8, usize, u32)>> = vec![vec![]; m - 1];
        for &s in &sorted {
            let on = |nodes: &[u32], at: &[usize]| (1..nodes.len() - 1).find(|&k| nodes[k] == s).and_then(|k| (0..at.len() - 1).find(|&g| at[g] < k && k < at[g + 1]).map(|g| (g, k)));
            if let Some((g, k)) = on(&fnodes, &fat) {
                gaps[g].push((0, k, s));
            } else if let Some((h, k)) = on(&bnodes, &bat) {
                gaps[m - 2 - h].push((1, k, s));
            } else {
                left.push(s);
            }
        }
        let mut stops = vec![skel[0]];
        for (g, gap) in gaps.iter_mut().enumerate() {
            gap.sort_by_key(|&(r, k, _)| (r, if r == 0 { k as i64 } else { -(k as i64) }));
            stops.extend(gap.iter().map(|x| x.2));
            stops.push(skel[g + 1]);
        }
        let path = if stops.len() == m { fwd } else { route_memo(net, &stops, memo)? };
        (stops, path)
    };
    let by_cost = !left.is_empty();
    for s in left {
        (stops, path) = best_slot(net, &stops, s, memo)?;
    }
    (stops.len() < 2 || net.back_run_of(&stops, &path).is_some()).then_some(((stops, path), by_cost))
}

/// `stops` with their one-way stations where `flatten` puts them, the skeleton kept in its order
/// unless it turns back and a straighter one exists (T-1899, `straighten`); None for a line with
/// none, or where `flatten` gives nothing.
pub(super) fn ordered(net: &Network, stops: &[u32], memo: &mut HopMemo) -> Option<(Vec<u32>, Vec<u32>)> {
    let (skel, legs) = split(net, stops)?;
    let skel = route_memo(net, &skel, memo).and_then(|p| straighten(net, &skel, &p)).map_or(skel, |s| s.0);
    flatten(net, &skel, &legs, memo)
}

/// A click adding `station` to an open line with `old` stops (`best_insert`): a one-way station
/// joins the set, any other goes into the skeleton where `best_slot` puts it; then `flatten`.
/// None when the line would have no one-way station, for `best_slot` over all the stops.
pub(super) fn insert(net: &Network, old: &[u32], station: u32, memo: &mut HopMemo) -> Option<(Vec<u32>, Vec<u32>)> {
    if old.contains(&station) || closed(old) || !old.iter().chain(std::iter::once(&station)).any(|&s| net.leg_heading(s).is_some()) {
        return None;
    }
    let (mut legs, skel): (Vec<u32>, Vec<u32>) = old.iter().partition(|&&s| net.leg_stop(s));
    let skel = if net.leg_stop(station) {
        legs.push(station);
        skel
    } else {
        best_slot(net, &skel, station, memo)?.0
    };
    if legs.is_empty() {
        return None;
    }
    flatten(net, &skel, &legs, memo)
}

/// `sort_stops` for a line with one-way stations: the skeleton put in again one stop at a time as
/// clicks would, then `flatten`. None for a line with none.
pub(super) fn sorted(net: &Network, stops: &[u32], memo: &mut HopMemo) -> Option<(Vec<u32>, Vec<u32>)> {
    let (skel, legs) = split(net, stops)?;
    let mut s: Vec<u32> = vec![];
    for &x in &skel {
        s = best_slot(net, &s, x, memo)?.0;
    }
    flatten(net, &s, &legs, memo)
}

/// Every open rail line of a loaded save, not ordered by hand, with its one-way stations where
/// `ordered` puts them: saved before the rule, a line could reach one only by turning back and
/// looping the pair. Whether any changed.
pub(super) fn straighten_legs(net: &mut Network) -> bool {
    let mut changed = false;
    let mut memo = HopMemo::new();
    for l in 0..net.line_count() as u32 {
        if !net.line_ok(l) || net.line_bus[l as usize].is_some() || net.line_ferry[l as usize].is_some() || net.line_hand[l as usize] {
            continue;
        }
        let Some((stops, path)) = ordered(net, net.line_stops(l), &mut memo) else { continue };
        if stops == net.line_stops(l) && path == net.line_path(l) {
            continue;
        }
        let mut d = net.line_data(l).unwrap();
        (d.stops, d.path) = (stops, path);
        changed |= net.set_line(l, Some(d)).is_ok();
    }
    changed
}
