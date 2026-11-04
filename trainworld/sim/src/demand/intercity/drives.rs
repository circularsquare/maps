//! T-279: routed drives between the market's places (pipeline/market_drives.py, docs/regional.md
//! 5.1): per pair of places on one landmass within 1,500 km, the routed drive's minutes and road km
//! as ratios to the rule's, and the ferry it takes over open water. A pair with no entry keeps the
//! rule; the market applies a pair's ratios to each of its zone pairs' own distances, times each
//! zone's own detour towards the other place (T-2424).
use super::{Cost, LONG_OCCUPANCY};
#[cfg(test)]
use super::FARE_MIN_PER_USD;
use std::collections::HashMap;

/// The car ferry OSM maps where a car takes one (T-2424; Anita 2026-11-04: "quite expensive"):
/// minutes and fare per car, each a base plus per km; the base is check-in, the wait for a sailing,
/// loading and unloading (notes/T-2424.md). pipeline/market_drives.py routes with the same numbers.
const FERRY_MIN: f32 = 90.;
const FERRY_KMH: f32 = 30.;
const FERRY_USD: f32 = 80.;
const FERRY_USD_PER_KM: f32 = 0.5;

/// A pair of places' drive against the rule: minutes and road km as ratios, the ferries' km and
/// how many.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Drive {
    pub rt: f32,
    pub rk: f32,
    pub ferry_km: f32,
    pub ferries: u8,
}
impl Drive {
    pub const RULE: Drive = Drive { rt: 1., rk: 1., ferry_km: 0., ferries: 0 };
    /// The ferries' minutes and each person's share of their fares.
    pub fn ferry_cost(&self) -> Cost {
        if self.ferries == 0 {
            return Cost::default();
        }
        let n = self.ferries as f32;
        Cost { min: n * FERRY_MIN + self.ferry_km * 60. / FERRY_KMH, usd: (n * FERRY_USD + self.ferry_km * FERRY_USD_PER_KM) / LONG_OCCUPANCY }
    }
}

/// The table by place: each place's pairs with later places (`off`, CSR), ascending.
#[derive(Default)]
pub(super) struct Drives {
    off: Vec<u32>,
    to: Vec<(u16, i8, i8)>,
    /// (place a < place b, ferry km, ferries), ascending
    ferry: Vec<(u32, u32, f32, u8)>,
    /// T-2424: a place's zones' detours towards another place, a group per pair: (place << 16 |
    /// other place, first entry, entries, dense), ascending. A dense group's entries are its zones
    /// in order (`zdense`), a sparse one's (zone, minutes, road km) by zone (`zsparse`); a zone not
    /// listed drives as its place's centre
    zgroups: Vec<(u32, u32, u16, bool)>,
    zdense: Vec<[i8; 2]>,
    zsparse: Vec<(u16, i8, i8)>,
}
fn take_bytes<'a>(b: &'a [u8], p: &mut usize, n: usize) -> Result<&'a [u8], String> {
    let s = b.get(*p..*p + n).ok_or_else(|| "invalid market drives".to_string())?;
    *p += n;
    Ok(s)
}
fn ratio(q: i8) -> f32 {
    (q as f32 / 64.).exp()
}
impl Drives {
    /// The table as `market_drives.py` writes it, its zone ids placed by `place_of`, `np` places.
    /// Ids no place has are skipped (a table built for more or fewer places than the registry's).
    pub fn decode(b: &[u8], place_of: &HashMap<&str, usize>, np: usize) -> Result<Drives, String> {
        let bad = || "invalid market drives".to_string();
        if np > 1 << 16 {
            return Err("market drives: more places than the table's u16 can name".into());
        }
        let mut p = 0usize;
        let mut take = |n: usize| take_bytes(b, &mut p, n);
        if take(4)? != b"TWD1" {
            return Err(bad());
        }
        let u32_of = |s: &[u8]| u32::from_le_bytes([s[0], s[1], s[2], s[3]]);
        let n = u32_of(take(4)?) as usize;
        let mut ids = Vec::with_capacity(n);
        for _ in 0..n {
            let len = take(1)?[0] as usize;
            let id = std::str::from_utf8(take(len)?).map_err(|_| bad())?;
            ids.push(place_of.get(id).copied().filter(|&q| q < np));
        }
        let counts: Vec<usize> = take(2 * n)?.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]]) as usize).collect();
        let mut all: Vec<(u32, u32, i8, i8)> = Vec::with_capacity(counts.iter().sum());
        let mut pair_places = Vec::with_capacity(all.capacity());
        for (a, &k) in counts.iter().enumerate() {
            for e in take(4 * k)?.chunks_exact(4) {
                let bi = u16::from_le_bytes([e[0], e[1]]) as usize;
                let pa = ids[a];
                let pb = *ids.get(bi).ok_or_else(bad)?;
                let pp = match (pa, pb) {
                    (Some(x), Some(y)) if x != y => Some((x.min(y) as u32, x.max(y) as u32)),
                    _ => None,
                };
                if let Some((x, y)) = pp {
                    all.push((x, y, e[2] as i8, e[3] as i8));
                }
                pair_places.push(pp);
            }
        }
        let nf = u32_of(take(4)?) as usize;
        let mut ferry = Vec::with_capacity(nf);
        for e in take(7 * nf)?.chunks_exact(7) {
            let i = u32_of(&e[0..4]) as usize;
            if let Some(&Some((x, y))) = pair_places.get(i) {
                ferry.push((x, y, u16::from_le_bytes([e[4], e[5]]) as f32, e[6]));
            }
        }
        // T-2424: after the ferries (a table from before has none), groups of (place, other place,
        // k): k (zone, minutes, road km), or with k's top bit set its low bits' zones in order, each
        // (minutes, road km)
        let (mut zgroups, mut zdense, mut zsparse) = (vec![], vec![], vec![]);
        if let Ok(g) = take(4) {
            for _ in 0..u32_of(g) {
                let h = take(6)?;
                let u16_at = |s: &[u8], i: usize| u16::from_le_bytes([s[i], s[i + 1]]);
                let (a, o, k) = (u16_at(h, 0) as usize, u16_at(h, 2) as usize, u16_at(h, 4));
                let (pa, po) = (*ids.get(a).ok_or_else(bad)?, *ids.get(o).ok_or_else(bad)?);
                let dense = k & 0x8000 != 0;
                let n = k & 0x7fff;
                let e = take(if dense { 2 } else { 4 } * n as usize)?;
                let (Some(x), Some(y)) = (pa, po) else { continue };
                if dense {
                    zgroups.push(((x as u32) << 16 | y as u32, zdense.len() as u32, n, true));
                    zdense.extend(e.chunks_exact(2).map(|c| [c[0] as i8, c[1] as i8]));
                } else {
                    zgroups.push(((x as u32) << 16 | y as u32, zsparse.len() as u32, n, false));
                    zsparse.extend(e.chunks_exact(4).map(|c| (u16_at(c, 0), c[2] as i8, c[3] as i8)));
                }
            }
        }
        all.sort_unstable_by_key(|e| (e.0, e.1));
        all.dedup_by_key(|e| (e.0, e.1));
        ferry.sort_unstable_by_key(|e| (e.0, e.1));
        zgroups.sort_unstable_by_key(|e| e.0);
        zgroups.dedup_by_key(|e| e.0);
        let mut off = vec![0u32; np + 1];
        for e in &all {
            off[e.0 as usize + 1] += 1;
        }
        for i in 0..np {
            off[i + 1] += off[i];
        }
        Ok(Drives { off, to: all.into_iter().map(|e| (e.1 as u16, e.2, e.3)).collect(), ferry, zgroups, zdense, zsparse })
    }
    /// T-2424: zone `k` of place `i`'s own drive towards place `j` over its place's (minutes, road
    /// km); (1, 1) where the table has none.
    pub fn zone(&self, i: usize, k: u32, j: usize) -> (f32, f32) {
        if i >= 1 << 16 || j >= 1 << 16 {
            return (1., 1.);
        }
        let Ok(g) = self.zgroups.binary_search_by_key(&((i as u32) << 16 | j as u32), |e| e.0) else { return (1., 1.) };
        let (_, s, n, dense) = self.zgroups[g];
        let q = if dense {
            (k < n as u32).then(|| self.zdense[(s + k) as usize])
        } else {
            let es = &self.zsparse[s as usize..(s + n as u32) as usize];
            es.binary_search_by_key(&k, |e| e.0 as u32).ok().map(|x| [es[x].1, es[x].2])
        };
        q.map_or((1., 1.), |[t, r]| (ratio(t), ratio(r)))
    }
    /// Pairs in the table.
    pub fn len(&self) -> usize {
        self.to.len()
    }
    /// Places i and j's drive; the rule's where the table has none.
    pub fn get(&self, i: usize, j: usize) -> Drive {
        let (a, b) = (i.min(j), i.max(j));
        if a == b || a + 1 >= self.off.len() {
            return Drive::RULE;
        }
        let s = &self.to[self.off[a] as usize..self.off[a + 1] as usize];
        let Ok(k) = s.binary_search_by_key(&(b as u16), |e| e.0) else { return Drive::RULE };
        let mut d = Drive { rt: ratio(s[k].1), rk: ratio(s[k].2), ferry_km: 0., ferries: 0 };
        if let Ok(f) = self.ferry.binary_search_by_key(&(a as u32, b as u32), |e| (e.0, e.1)) {
            d.ferry_km = self.ferry[f].2;
            d.ferries = self.ferry[f].3;
        }
        d
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A table of places "a", "b", "c", "x" (x unknown here): a-b, a-c with a ferry, b-x.
    fn bytes() -> Vec<u8> {
        let mut b = b"TWD1".to_vec();
        b.extend(4u32.to_le_bytes());
        for id in ["a", "b", "c", "x"] {
            b.push(1);
            b.extend(id.as_bytes());
        }
        for n in [2u16, 1, 0, 0] {
            b.extend(n.to_le_bytes());
        }
        for (to, lt, lk) in [(1u16, 64i8, -32i8), (2, 0, 0), (3, 10, 10)] {
            b.extend(to.to_le_bytes());
            b.extend([lt as u8, lk as u8]);
        }
        b.extend(1u32.to_le_bytes());
        b.extend(1u32.to_le_bytes());
        b.extend(80u16.to_le_bytes());
        b.push(1);
        b
    }

    #[test]
    fn decodes_by_place_both_ways() {
        let place_of: HashMap<&str, usize> = [("a", 2), ("b", 0), ("c", 1)].into_iter().collect();
        let d = Drives::decode(&bytes(), &place_of, 3).unwrap();
        assert_eq!(d.len(), 2);
        let ab = d.get(2, 0);
        assert_eq!(ab, d.get(0, 2));
        assert!((ab.rt - std::f32::consts::E).abs() < 1e-5 && (ab.rk - (-0.5f32).exp()).abs() < 1e-6);
        assert_eq!(ab.ferries, 0);
        let ac = d.get(1, 2);
        assert_eq!((ac.rt, ac.rk, ac.ferry_km, ac.ferries), (1., 1., 80., 1));
        assert_eq!(d.get(0, 1), Drive::RULE);
        assert_eq!(d.get(1, 1), Drive::RULE);
        assert_eq!(Drives::default().get(0, 1), Drive::RULE);
        assert!(Drives::decode(&bytes()[..20], &place_of, 3).is_err());
    }

    #[test]
    fn zone_detours_by_place_and_zone() {
        // the same table, then zone 3 of "a" towards "b" and zone 0 of "x" (unknown) towards "a",
        // and densely "c"'s three zones towards "a"
        let mut b = bytes();
        b.extend(3u32.to_le_bytes());
        for (a, o, zs) in [(0u16, 1u16, vec![(3u16, 64i8, -64i8)]), (3, 0, vec![(0, 10, 10)])] {
            for v in [a, o, zs.len() as u16] {
                b.extend(v.to_le_bytes());
            }
            for (k, t, r) in zs {
                b.extend(k.to_le_bytes());
                b.extend([t as u8, r as u8]);
            }
        }
        for v in [2u16, 0, 0x8000 | 3] {
            b.extend(v.to_le_bytes());
        }
        for q in [0i8, 0, 64, 0, 0, -64] {
            b.push(q as u8);
        }
        let place_of: HashMap<&str, usize> = [("a", 2), ("b", 0), ("c", 1)].into_iter().collect();
        let d = Drives::decode(&b, &place_of, 3).unwrap();
        assert_eq!(d.get(2, 0), Drives::decode(&bytes(), &place_of, 3).unwrap().get(2, 0));
        let (t, r) = d.zone(2, 3, 0);
        assert!((t - std::f32::consts::E).abs() < 1e-5 && (r - (-1f32).exp()).abs() < 1e-6);
        assert_eq!(d.zone(2, 3, 1), (1., 1.));
        assert_eq!(d.zone(2, 2, 0), (1., 1.));
        assert_eq!(d.zone(0, 3, 2), (1., 1.));
        assert_eq!(Drives::default().zone(0, 0, 1), (1., 1.));
        // "c" is place 1, "a" place 2
        assert_eq!(d.zone(1, 0, 2), (1., 1.));
        assert!((d.zone(1, 1, 2).0 - std::f32::consts::E).abs() < 1e-5 && d.zone(1, 1, 2).1 == 1.);
        assert!((d.zone(1, 2, 2).1 - (-1f32).exp()).abs() < 1e-6);
        assert_eq!(d.zone(1, 3, 2), (1., 1.));
        assert!(Drives::decode(&b[..b.len() - 1], &place_of, 3).is_err());
    }

    #[test]
    fn a_ferry_is_a_base_and_per_km() {
        let f = |km, n| Drive { ferry_km: km, ferries: n, ..Drive::RULE }.ferry_cost().at(FARE_MIN_PER_USD);
        assert_eq!(f(80., 0), 0.);
        let per = FERRY_MIN + FERRY_USD * FARE_MIN_PER_USD / LONG_OCCUPANCY;
        assert!((f(0., 1) - per).abs() < 1e-4);
        assert!((f(70., 1) - f(35., 1) - (35. * 60. / FERRY_KMH + 35. * FERRY_USD_PER_KM * FARE_MIN_PER_USD / LONG_OCCUPANCY)).abs() < 1e-3);
    }
}
