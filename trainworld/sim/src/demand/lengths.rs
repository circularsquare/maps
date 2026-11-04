//! T-2365: a zone's trips by straight-line length, by mode, for the city view's trip-length graph.
//!
//! Bins of `LEN_EDGES_KM`. Each population answers its solved periods' trips, rail, walking and
//! the trips no way serves per bin (`trip_lengths_day`), each with its km; the bins of one
//! population add up to its `summary` exactly, so the city's bins add up to its split.
//!
//! The commutes are counted between zones (`kernel` step 1), whose centres are 2 km apart or more:
//! each zone pair's trips are spread over the bins as the distances between points spread evenly
//! over the two zones' squares (`LenKernel`), so the curve is smooth below 2 km, and the zone's own
//! pair keeps the mean the kernel gives it (0.52 of its side). Everyday outings run between cells
//! and take their own distance; their rail goes by its subzone pair's mean distance.

use std::cell::RefCell;
use std::rc::Rc;

/// Upper edges of the bins, km; the last bin has none.
pub const LEN_EDGES_KM: [f32; N_LEN - 1] = [
    0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, // 0.25 km to 3 km
    3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, // 0.5 km to 10 km
    11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, // 1 km to 20 km
    22.5, 25.0, 27.5, 30.0, 32.5, 35.0, 37.5, 40.0, 42.5, 45.0, 47.5, 50.0, // 2.5 km to 50 km
    60.0, 70.0, 80.0, 90.0, 100.0, 125.0, 150.0, 200.0, 300.0,
];
pub const N_LEN: usize = 58;

/// The bin of a straight-line distance in km.
#[inline]
pub fn len_bin(km: f32) -> usize {
    LEN_EDGES_KM.partition_point(|&e| e <= km)
}

/// A bin's middle, km (the last bin's a third past its lower edge).
fn len_mid(b: usize) -> f32 {
    let lo = if b == 0 { 0.0 } else { LEN_EDGES_KM[b - 1] };
    match LEN_EDGES_KM.get(b) {
        Some(&hi) => 0.5 * (lo + hi),
        None => lo * 4.0 / 3.0,
    }
}

/// Trips and their km per bin: `N_LEN` trips, then `N_LEN` km.
pub type LenSums = Vec<f64>;

pub fn len_zero() -> LenSums {
    vec![0.0; 2 * N_LEN]
}

/// What a commute pass adds per bin (`ModeOut::len`): rail trips, the walkers rail takes and
/// (T-2214) the rail trips between two land parts, as `ModeOut::rail`, `rail_walk`, `rail_unr`.
#[derive(Clone, Debug, Default)]
pub struct LenOut {
    pub rail: LenSums,
    pub walk: LenSums,
    pub unr: LenSums,
}

impl LenOut {
    pub fn zero() -> LenOut {
        LenOut { rail: len_zero(), walk: len_zero(), unr: len_zero() }
    }
    /// Averaged in as a round's figures are (`w` the new round's weight).
    pub fn toward(&mut self, new: &LenOut, w: f64) {
        for (a, b) in [(&mut self.rail, &new.rail), (&mut self.walk, &new.walk), (&mut self.unr, &new.unr)] {
            if a.len() != b.len() {
                *a = b.clone();
                continue;
            }
            for (x, &y) in a.iter_mut().zip(b) {
                *x += (y - *x) * w;
            }
        }
    }
    pub fn scale(&mut self, rail: f64, walk: f64, unr: f64) {
        self.rail.iter_mut().for_each(|x| *x *= rail);
        self.walk.iter_mut().for_each(|x| *x *= walk);
        self.unr.iter_mut().for_each(|x| *x *= unr);
    }
}

/// Zone offsets (in half squares, each way) out to which a pair's trips are spread over the
/// distances between its squares' points; past them a pair goes to the two bins about its centres'
/// distance.
const SPREAD_HALF: i32 = 24;
/// Points per axis of the difference of two even spreads, for the spread's table.
const SPREAD_N: usize = 32;

/// Per zone offset and pair of sides (one or two squares, T-852), the share of a pair's trips in
/// each bin and their mean km there (`spread`). Worked out once per side length on a thread.
pub struct LenKernel {
    zone_km: f32,
    /// CSR over (sides, |dx|, |dy|) in half squares, `|dx|` >= `|dy|`: bin, share, share x km.
    off: Vec<u32>,
    ent: Vec<(u16, f32, f32)>,
}

thread_local! {
    static KERNEL: RefCell<Option<Rc<LenKernel>>> = const { RefCell::new(None) };
}

/// The spread for squares of `zone_m` metres, made on first use.
pub fn len_kernel(zone_m: f32) -> Rc<LenKernel> {
    KERNEL.with(|k| {
        let mut k = k.borrow_mut();
        match &*k {
            Some(t) if t.zone_km == zone_m / 1000.0 => t.clone(),
            _ => {
                let t = Rc::new(LenKernel::new(zone_m / 1000.0));
                *k = Some(t.clone());
                t
            }
        }
    })
}

impl LenKernel {
    fn idx(sides: usize, ax: i32, ay: i32) -> usize {
        let n = (SPREAD_HALF + 1) as usize;
        (sides * n + ax as usize) * n + ay as usize
    }

    fn new(zone_km: f32) -> LenKernel {
        let n = (SPREAD_HALF + 1) as usize;
        let mut off = vec![0u32; 3 * n * n + 1];
        let mut ent = vec![];
        let mut acc = vec![(0f64, 0f64); N_LEN];
        // the difference of two even spreads over sides a and b is a trapezoid: its points and weights
        let diff = |a: f32, b: f32| -> Vec<(f32, f32)> {
            let h = 0.5 * (a + b);
            (0..SPREAD_N)
                .map(|k| {
                    let x = -h + (k as f32 + 0.5) * (2.0 * h) / SPREAD_N as f32;
                    // density of U(a) - U(b) at x: overlap of [x - b/2, x + b/2] with [-a/2, a/2]
                    let w = ((x + 0.5 * b).min(0.5 * a) - (x - 0.5 * b).max(-0.5 * a)).max(0.0);
                    (x, w)
                })
                .collect()
        };
        for sides in 0..3 {
            let (a, b) = [(1.0, 1.0), (1.0, 2.0), (2.0, 2.0)][sides];
            let d = diff(a * zone_km, b * zone_km);
            let wsum: f32 = d.iter().map(|e| e.1).sum::<f32>().powi(2);
            for ax in 0..=SPREAD_HALF {
                for ay in 0..=SPREAD_HALF {
                    let i = Self::idx(sides, ax, ay);
                    if ay <= ax {
                        acc.iter_mut().for_each(|e| *e = (0.0, 0.0));
                        let (cx, cy) = (ax as f32 * 0.5 * zone_km, ay as f32 * 0.5 * zone_km);
                        for &(x, wx) in &d {
                            for &(y, wy) in &d {
                                let km = ((cx + x).powi(2) + (cy + y).powi(2)).sqrt();
                                let w = (wx * wy / wsum) as f64;
                                let e = &mut acc[len_bin(km)];
                                e.0 += w;
                                e.1 += w * km as f64;
                            }
                        }
                        for (bn, &(w, wk)) in acc.iter().enumerate() {
                            if w > 0.0 {
                                ent.push((bn as u16, w as f32, wk as f32));
                            }
                        }
                    }
                    off[i + 1] = ent.len() as u32;
                }
            }
        }
        LenKernel { zone_km, off, ent }
    }

    /// Add `x` trips between zones whose centres are (`hdx`, `hdy`) half squares apart, of sides
    /// `si` and `sj` squares, into `out` (`LenSums`).
    #[inline]
    pub fn add(&self, hdx: i32, hdy: i32, si: i32, sj: i32, x: f64, out: &mut [f64]) {
        let (ax, ay) = (hdx.abs(), hdy.abs());
        let (ax, ay) = if ax >= ay { (ax, ay) } else { (ay, ax) };
        if ax <= SPREAD_HALF {
            let sides = (si + sj - 2) as usize;
            let i = Self::idx(sides, ax, ay);
            for &(b, w, wk) in &self.ent[self.off[i] as usize..self.off[i + 1] as usize] {
                out[b as usize] += x * w as f64;
                out[N_LEN + b as usize] += x * wk as f64;
            }
            return;
        }
        let km = 0.5 * self.zone_km * ((ax * ax + ay * ay) as f32).sqrt();
        add_at(km, x, out);
    }
}

/// Add `x` trips of `km` into `out`, shared between the two bins whose middles are about it, so
/// that a curve of far pairs does not step where its pairs' distances cross an edge.
#[inline]
pub fn add_at(km: f32, x: f64, out: &mut [f64]) {
    let b = len_bin(km);
    let m = len_mid(b);
    let (b2, m2) = if km >= m { (b + 1, if b + 1 < N_LEN { len_mid(b + 1) } else { m }) } else { (b.wrapping_sub(1), if b > 0 { len_mid(b - 1) } else { m }) };
    let f = if b2 < N_LEN && m2 != m { ((km - m) / (m2 - m)).clamp(0.0, 1.0) as f64 } else { 0.0 };
    out[b] += x * (1.0 - f);
    out[N_LEN + b] += x * (1.0 - f) * km as f64;
    if f > 0.0 {
        out[b2] += x * f;
        out[N_LEN + b2] += x * f * km as f64;
    }
}

/// Add `x` trips of `km` into `out`, its own bin alone.
#[inline]
pub fn add_in(km: f32, x: f64, out: &mut [f64]) {
    let b = len_bin(km);
    out[b] += x;
    out[N_LEN + b] += x * km as f64;
}

/// Scale `v`'s trips and km so its trips add up to `total` (nothing where they add up to none).
pub fn scale_to(v: &mut [f64], total: f64) {
    let s: f64 = v[..N_LEN].iter().sum();
    if s > 0.0 {
        let f = total / s;
        v.iter_mut().for_each(|x| *x *= f);
    }
}

/// The bins' upper edges, km, for the app (the last bin has none).
#[wasm_bindgen::prelude::wasm_bindgen]
pub fn trip_length_edges() -> Vec<f32> {
    LEN_EDGES_KM.to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn edges_ascend_and_bins_find_them() {
        assert_eq!(LEN_EDGES_KM.len(), N_LEN - 1);
        assert!(LEN_EDGES_KM.windows(2).all(|w| w[0] < w[1]));
        assert_eq!(len_bin(0.0), 0);
        assert_eq!(len_bin(0.25), 1);
        assert_eq!(len_bin(9.99), 25);
        assert_eq!(len_bin(1000.0), N_LEN - 1);
    }

    #[test]
    fn the_spread_keeps_every_trip_and_the_own_pairs_mean() {
        let k = LenKernel::new(2.0);
        for (hdx, hdy, si, sj) in [(0, 0, 1, 1), (2, 0, 1, 1), (4, 2, 1, 1), (3, 1, 1, 2), (0, 0, 2, 2), (24, 24, 1, 1), (40, 6, 1, 1)] {
            let mut out = len_zero();
            k.add(hdx, hdy, si, sj, 10.0, &mut out);
            let t: f64 = out[..N_LEN].iter().sum();
            assert!((t - 10.0).abs() < 1e-4, "{hdx} {hdy}: {t}");
        }
        // a square's own pair: 0.5214 of its side, as `Gravity::dist_km` has it
        let mut out = len_zero();
        k.add(0, 0, 1, 1, 1.0, &mut out);
        let km: f64 = out[N_LEN..].iter().sum();
        assert!((km - 0.5214 * 2.0).abs() < 0.01, "{km}");
        // the next square's pair: about its centres' 2 km
        let mut out = len_zero();
        k.add(2, 0, 1, 1, 1.0, &mut out);
        let km: f64 = out[N_LEN..].iter().sum();
        assert!((km - 2.0).abs() < 0.2, "{km}");
        // past the spread, at the centres' distance
        let mut out = len_zero();
        k.add(60, 0, 1, 1, 1.0, &mut out);
        let km: f64 = out[N_LEN..].iter().sum();
        assert!((km - 60.0).abs() < 1e-3, "{km}");
    }
}
