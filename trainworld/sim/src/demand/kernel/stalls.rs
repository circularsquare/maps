use super::*;

// ---------------------------------------------------------------------------------------------
// 7b. Riders left behind by full trains (T-188)

/// The longest average extra wait, in real minutes, a boarding is given for full trains: a line
/// whose riders arrive faster than its trains take them all period long has a queue that never
/// clears, and its wait would grow without bound.
pub const LEFT_WAIT_CAP_MIN: f32 = 60.0;

/// What `left_behind` found over a period's network.
#[derive(Clone, Copy, Debug, Default)]
pub struct LeftStats {
    /// Riders who could not get on the first train to leave after they arrived (the boarding
    /// account's "left behind", `RunAccount::left_h`).
    pub left: f64,
    /// Extra minutes waited for full trains, all riders together.
    pub rider_min: f64,
    /// Route nodes where anyone is left behind.
    pub stops: usize,
}

/// The boarding account's rule (track/boarding.rs, docs/track-and-service.md 6.6) run on one
/// period's own trains, to price riders left behind into the next crowding round (T-188): per
/// route node, the average extra real minutes a rider boarding there waits because full trains
/// left them on the platform.
///
/// Each direction of each line runs its period's trains at an even headway; riders arrive at each
/// stop over the period by the account's shape for that period (scaled so the busiest hour is
/// the busiest-hour factor times the mean, so riders arrive fastest mid-period); each train lets
/// off the stop's share alighting (demand's alightings over the riders arriving there) and takes
/// whoever waits up to its crush load. Whoever does not fit waits a headway more for the next
/// train of the same line and direction. A queue still there when the period ends is cleared at
/// the period's average spare room (a queue that grows for half an hour and clears in half an
/// hour adds its riders' half hour). Trains leaving before the period carry riders arriving at
/// the period's opening rate, so they reach the later stops with a load. `board` and `alight`
/// are the period's riders per route node, every population together.
pub fn left_behind(g: &Graph, net: &Network, p: &Params, board: &[f32], alight: &[f32]) -> (Vec<f32>, LeftStats) {
    use crate::track::boarding::{shape_cum, shape_g, shape_strength, SHAPE};
    let nr = g.rn_line.len();
    let mut out = vec![0.0f32; nr];
    let mut stats = LeftStats::default();
    if board.len() != nr || alight.len() != nr || p.period_hours <= 0.0 {
        return (out, stats);
    }
    let shape = SHAPE[p.period.min(SHAPE.len() - 1)];
    let hours = p.period_hours as f64;
    let c = shape_strength(shape, hours, p.peak_hour_factor as f64);
    let len = hours * 3600.0;
    let rate0 = 1.0 + c * shape_g(shape, 0.0);
    let mut alpha: Vec<f64> = vec![];
    let mut off: Vec<f64> = vec![];
    let mut carry: Vec<f64> = vec![];
    let mut r0 = 0usize;
    for l in 0..net.n_lines() {
        let n = (net.line_off[l + 1] - net.line_off[l]) as usize;
        for _dir in 0..2 {
            let run = r0..r0 + n;
            r0 += n;
            // a stop with no train due (T-842) has none in `rn_tph`; the line's trains still run
            let tph = g.rn_tph[run.clone()].iter().fold(0.0f32, |a, &b| a.max(b)) as f64;
            let cap = net.line_crush[l] as f64;
            let b = &board[run.clone()];
            if n < 2 || tph <= 0.0 || cap <= 0.0 || b.iter().map(|&v| v as f64).sum::<f64>() < 1.0 {
                continue;
            }
            let a = &alight[run.clone()];
            // the share of the riders aboard leaving at each stop, from the period's totals
            alpha.clear();
            let mut on = 0.0f64;
            for k in 0..n {
                let off_k = a[k] as f64;
                alpha.push(if k + 1 == n { 1.0 } else if on > 1e-6 { (off_k / on).clamp(0.0, 1.0) } else { 0.0 });
                on = (on + b[k] as f64 - off_k).max(0.0);
            }
            // seconds from the first stop to each stop
            off.clear();
            let mut t = 0.0f64;
            for k in 0..n {
                off.push(t);
                t += g.rn_ride[run.start + k] as f64 * 60.0;
            }
            let h = 3600.0 / tph;
            // trains leaving the first stop from a run and a headway before the period to its end
            let i0 = -((off[n - 1] + h) / h).ceil() as i64;
            let i1 = (len / h).ceil() as i64;
            carry.clear();
            carry.resize((i1 - i0).max(0) as usize, 0.0);
            for k in 0..n {
                let bk = b[k] as f64;
                let cum = |t: f64| if t <= 0.0 { bk / len * rate0 * t } else if t >= len { bk } else { bk * shape_cum(shape, c, t / len) };
                let (mut q, mut q_end, mut rider_s, mut left) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
                let (mut room_sum, mut trains) = (0.0f64, 0usize);
                for (j, i) in (i0..i1).enumerate() {
                    let t = (i as f64 + 0.5) * h + off[k];
                    let arrived = (cum(t) - cum(t - h)).max(0.0);
                    let on = carry[j] * (1.0 - alpha[k]);
                    let room = (cap - on).max(0.0);
                    let waiting = q + arrived;
                    let got_on = waiting.min(room);
                    q = waiting - got_on;
                    carry[j] = if k + 1 < n { on + got_on } else { 0.0 };
                    if (0.0..len).contains(&t) {
                        rider_s += q * h;
                        left += arrived.min(q);
                        room_sum += room;
                        trains += 1;
                        q_end = q;
                    }
                }
                if bk < 1.0 || trains == 0 {
                    continue;
                }
                if q_end > 0.0 {
                    // what is left clears at the room the period's trains had to spare on average
                    let spare = room_sum / trains as f64 * tph / 3600.0 - bk / len;
                    rider_s += if spare > 0.0 { (q_end * q_end / (2.0 * spare)).min(q_end * len) } else { q_end * len };
                }
                if rider_s > 0.0 {
                    out[run.start + k] = ((rider_s / bk / 60.0) as f32).min(LEFT_WAIT_CAP_MIN);
                    stats.left += left;
                    stats.rider_min += rider_s / 60.0;
                    stats.stops += 1;
                }
            }
        }
    }
    (out, stats)
}

/// The costs a crowding round is solved with, from the frozen loads of every population together
/// (`total`): the crowding factor per segment from the segment loads, and, when `total` also
/// carries the boardings and alightings per route node after them (`3 x` route nodes), the extra
/// wait for riders left behind (`left_behind`, T-188; none with the segment loads alone or with
/// `Params::left_behind` off). None if `total` has another length or a bad number.
pub fn round_costs(g: &Graph, net: &Network, p: &Params, total: &[f32]) -> Option<(Vec<f32>, Vec<f32>)> {
    let nr = g.rn_line.len();
    if (total.len() != nr && total.len() != 3 * nr) || total.iter().any(|v| !v.is_finite() || *v < 0.0) {
        return None;
    }
    let mut loads = Loads::zero(g.n_st, nr);
    loads.seg.copy_from_slice(&total[..nr]);
    let (fac, _) = crowd_factors(g, net, &loads, p);
    let mut left = if total.len() == 3 * nr && p.left_behind { left_behind(g, net, p, &total[nr..2 * nr], &total[2 * nr..]).0 } else { vec![0.0; nr] };
    for ((l, s), b) in left.iter_mut().zip(&g.rn_span).zip(&g.rn_bus) {
        *l += s + b;
    }
    Some((fac, left))
}

/// A period's crowding rounds stop once one moves its summed segment loads by under this share,
/// rider weighted (T-434); the driver caps them at three.
pub const SETTLED: f64 = 0.01;

/// Whether a period's next crowding round is skipped (T-434): `seg`, the segment loads it would be
/// priced on, is within `SETTLED` of those the round before was priced on, kept in `seen`. The first
/// crowding round always runs, as `seen` starts empty. Each population of a cluster gets the
/// cluster's loads on its own route nodes (T-756), so it stops when those settle.
pub fn settled(seen: &mut Vec<f32>, seg: &[f32]) -> bool {
    if seen.len() == seg.len() {
        let (mut moved, mut base) = (0f64, 0f64);
        for (a, b) in seg.iter().zip(seen.iter()) {
            moved += (a - b).abs() as f64;
            base += *b as f64;
        }
        if moved < SETTLED * base {
            return true;
        }
    }
    seen.clear();
    seen.extend_from_slice(seg);
    false
}

// ---------------------------------------------------------------------------------------------
// 7c. The trains' room (T-1070), as a queue (T-1255)

/// What each route node keeps once the period's riders are held to their trains' room
/// (`capacity_fit`): the share of its boardings who get on, of its segment's riders, and of its
/// alightings (those of the segment before). All ones on a line within its room.
pub struct Fit {
    pub board: Vec<f32>,
    pub seg: Vec<f32>,
    pub alight: Vec<f32>,
    /// some route node keeps less than all
    pub any: bool,
}

/// Anita's cap on demand (T-1070), as a queue (T-1255): riders come to the platform and wait when
/// the riders ahead of them all get on within `WAIT_FOR_ROOM_S`, else go another way. Each
/// direction of each line runs its period's trains at an even headway, riders arriving at each
/// stop by the boarding account's shape (as `left_behind`); a train lets off the stop's share and
/// takes whoever waits up to its crush load, riders already aboard keeping their places. A rider
/// joins only while the queue is within this train's room and that of the trains due in the next
/// `WAIT_FOR_ROOM_S`, each with the same room. `total` is the final loads of every population
/// together (segment loads, boardings, alightings per route node, as `round_costs`), optionally
/// followed by the riders coming to each platform (`4 x` route nodes; at least the boardings: the
/// free-flow pass's, who would ride if the trains had room). A stop keeps its boardings up to those
/// the queue takes, so riders the rounds put there stay while the platform has room for them. None
/// if `total` has another length.
pub fn capacity_fit(g: &Graph, net: &Network, p: &Params, total: &[f32]) -> Option<Fit> {
    use crate::track::boarding::{shape_cum, shape_g, shape_strength, SHAPE, WAIT_FOR_ROOM_S};
    let nr = g.rn_line.len();
    if (total.len() != 3 * nr && total.len() != 4 * nr) || p.period_hours <= 0.0 {
        return None;
    }
    let (seg, board, alight) = (&total[..nr], &total[nr..2 * nr], &total[2 * nr..3 * nr]);
    let come = total.get(3 * nr..).filter(|c| c.len() == nr);
    let mut fit = Fit { board: vec![1.0; nr], seg: vec![1.0; nr], alight: vec![1.0; nr], any: false };
    let shape = SHAPE[p.period.min(SHAPE.len() - 1)];
    let hours = p.period_hours as f64;
    let c = shape_strength(shape, hours, p.peak_hour_factor as f64);
    let len = hours * 3600.0;
    let rate0 = 1.0 + c * shape_g(shape, 0.0);
    let (mut alpha, mut off, mut carry): (Vec<f64>, Vec<f64>, Vec<f64>) = (vec![], vec![], vec![]);
    let mut r0 = 0usize;
    for l in 0..net.n_lines() {
        let n = (net.line_off[l + 1] - net.line_off[l]) as usize;
        for _dir in 0..2 {
            let run = r0..r0 + n;
            r0 += n;
            let tph = g.rn_tph[run.clone()].iter().fold(0.0f32, |a, &b| a.max(b)) as f64;
            let cap = net.line_crush[l] as f64;
            let b = &board[run.clone()];
            if n < 2 || tph <= 0.0 || cap <= 0.0 || b.iter().map(|&v| v as f64).sum::<f64>() < 1.0 {
                continue;
            }
            let a = &alight[run.clone()];
            // the share of the riders aboard leaving at each stop, from the rounds' riders
            alpha.clear();
            let mut on = 0.0f64;
            for k in 0..n {
                let off_k = a[k] as f64;
                alpha.push(if k + 1 == n { 1.0 } else if on > 1e-6 { (off_k / on).clamp(0.0, 1.0) } else { 0.0 });
                on = (on + b[k] as f64 - off_k).max(0.0);
            }
            off.clear();
            let mut t = 0.0f64;
            for k in 0..n {
                off.push(t);
                t += g.rn_ride[run.start + k] as f64 * 60.0;
            }
            let h = 3600.0 / tph;
            // trains leaving the first stop from a run and a headway before the period to its end
            let i0 = -((off[n - 1] + h) / h).ceil() as i64;
            let i1 = (len / h).ceil() as i64;
            carry.clear();
            carry.resize((i1 - i0).max(0) as usize, 0.0);
            // this train and every one due within the wait
            let ahead = 1.0 + (WAIT_FOR_ROOM_S / h + 1e-9).floor();
            let mut kept = 1.0f32;
            for k in 0..n {
                let r = run.start + k;
                let bk = b[k].max(0.0) as f64;
                let ck = come.map_or(bk, |v| (v[r] as f64).max(bk));
                let cum = |t: f64| if t <= 0.0 { ck / len * rate0 * t } else if t >= len { ck } else { ck * shape_cum(shape, c, t / len) };
                let (mut q, mut came, mut joined) = (0.0f64, 0.0f64, 0.0f64);
                for (j, i) in (i0..i1).enumerate() {
                    let t = (i as f64 + 0.5) * h + off[k];
                    let arrived = (cum(t) - cum(t - h)).max(0.0);
                    let on = carry[j] * (1.0 - alpha[k]);
                    let room = (cap - on).max(0.0);
                    let join = arrived.min((room * ahead - q).max(0.0));
                    let got = (q + join).min(room);
                    q += join - got;
                    carry[j] = if k + 1 < n { on + got } else { 0.0 };
                    if (0.0..len).contains(&t) {
                        came += arrived;
                        joined += join;
                    }
                }
                let share = if came > 1e-9 && bk > 0.0 { (joined / came * ck / bk).min(1.0) as f32 } else { 1.0 };
                fit.alight[r] = kept;
                fit.board[r] = share;
                let stay = (seg[r] - b[k]).max(0.0) * kept;
                kept = if seg[r] > 0.0 { ((stay + b[k].max(0.0) * share) / seg[r]).min(1.0) } else { 1.0 };
                fit.seg[r] = kept;
                fit.any |= share < 1.0 || kept < 1.0;
            }
        }
    }
    Some(fit)
}

/// One population's loads held to the trains' room (`capacity_fit`), its transfers with them.
/// Returns the share of each station's entries (first boardings) that still ride, and of all its
/// rail trips. Riders turned away at a first boarding are that station's; riders turned away at a
/// change are taken off every entry alike, their first train's riders being in no table here.
pub fn cap_loads(g: &Graph, fit: &Fit, ix: &XferIndex, l: &mut Loads, xfer: &mut [f32]) -> (Vec<f32>, f64) {
    let s = g.n_st;
    let nr = g.rn_line.len();
    let mut xin = vec![0f32; nr];
    if xfer.len() == ix.len {
        for v in s..g.n {
            for e in g.adj_off[v] as usize..g.adj_off[v + 1] as usize {
                let slot = ix.at[e];
                if slot == NONE {
                    continue;
                }
                let to = g.adj_to[e] as usize;
                for (k, d) in (g.adj_off[to] as usize..g.adj_off[to + 1] as usize).enumerate() {
                    let w = g.adj_to[d] as usize - s;
                    let x = &mut xfer[slot as usize + k];
                    xin[w] += *x;
                    *x *= fit.board[w];
                }
            }
        }
    }
    let mut cut_first = vec![0f64; s];
    let mut cut_change = 0f64;
    let (mut ride0, mut ride1) = (0f64, 0f64);
    for r in 0..nr {
        let st = g.rn_station[r] as usize;
        let (b, a) = (l.rn_board[r], l.rn_alight[r]);
        let gone = (1.0 - fit.board[r]) * b;
        let changing = xin[r].min(b);
        cut_first[st] += ((1.0 - fit.board[r]) * (b - changing)) as f64;
        cut_change += ((1.0 - fit.board[r]) * changing) as f64;
        l.rn_board[r] = b - gone;
        l.board[st] = (l.board[st] - gone).max(0.0);
        // the fare curve's extra is counted on the km ridden (T-1516)
        if let Some(x) = l.rn_fare_x.get_mut(r) {
            *x *= fit.seg[r];
        }
        let left = (1.0 - fit.alight[r]) * a;
        l.rn_alight[r] = a - left;
        l.alight[st] = (l.alight[st] - left).max(0.0);
        ride0 += (l.seg[r] * g.rn_ride[r]) as f64;
        l.seg[r] *= fit.seg[r];
        ride1 += (l.seg[r] * g.rn_ride[r]) as f64;
    }
    if ride0 > 0.0 {
        l.pax_min *= ride1 / ride0;
    }
    let before: f64 = l.entry.iter().map(|&v| v as f64).sum();
    let mut keep = vec![1f32; s];
    let mut after = 0f64;
    for st in 0..s {
        let e = l.entry[st] as f64;
        let e1 = (e - cut_first[st]).max(0.0);
        if e > 0.0 {
            keep[st] = (e1 / e) as f32;
        }
        after += e1;
    }
    let spread = if after > 0.0 { ((after - cut_change) / after).max(0.0) } else { 1.0 };
    for st in 0..s {
        keep[st] *= spread as f32;
        l.entry[st] *= keep[st];
    }
    // the fare curve's extra goes with the trips (T-1143)
    for x in l.rn_fare_x.iter_mut() {
        *x *= spread as f32;
    }
    let all = if before > 0.0 { after * spread / before } else { 1.0 };
    for v in l.exit.iter_mut() {
        *v *= all as f32;
    }
    (keep, all)
}

/// `cap_loads`' route node shares on loads kept apart (the purposes, the market's commutes).
pub fn cap_part(fit: &Fit, seg: &mut [f32], board: &mut [f32], alight: &mut [f32]) {
    for (v, f) in seg.iter_mut().zip(&fit.seg) {
        *v *= f;
    }
    for (v, f) in board.iter_mut().zip(&fit.board) {
        *v *= f;
    }
    for (v, f) in alight.iter_mut().zip(&fit.alight) {
        *v *= f;
    }
}
