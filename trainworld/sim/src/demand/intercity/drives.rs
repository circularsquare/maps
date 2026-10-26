//! T-279: routed drives between the market's places (pipeline/market_drives.py, docs/regional.md
//! 5.1): per pair of places on one landmass within 1,500 km, the routed drive's minutes and road km
//! as ratios to the rule's, and the ferry it takes over open water. A pair with no entry keeps the
//! rule; the market applies a pair's ratios to each of its zone pairs' own distances.
use super::{FARE_MIN_PER_USD, LONG_OCCUPANCY};
use std::collections::HashMap;

/// The ferry a car takes over open water the landmass has no road across (Anita 2026-10-26):
/// minutes and fare per car, each a base plus per km. pipeline/market_drives.py routes with the same
/// numbers.
const FERRY_MIN: f32 = 30.;
const FERRY_KMH: f32 = 35.;
const FERRY_USD: f32 = 50.;
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
    /// The ferries' minutes and each person's share of their fares, perceived.
    pub fn ferry_min(&self) -> f32 {
        if self.ferries == 0 {
            return 0.;
        }
        let per = FERRY_MIN + FERRY_USD * FARE_MIN_PER_USD / LONG_OCCUPANCY;
        let per_km = 60. / FERRY_KMH + FERRY_USD_PER_KM * FARE_MIN_PER_USD / LONG_OCCUPANCY;
        self.ferries as f32 * per + self.ferry_km * per_km
    }
}

/// The table by place: each place's pairs with later places (`off`, CSR), ascending.
#[derive(Default)]
pub(super) struct Drives {
    off: Vec<u32>,
    to: Vec<(u16, i8, i8)>,
    /// (place a < place b, ferry km, ferries), ascending
    ferry: Vec<(u32, u32, f32, u8)>,
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
        all.sort_unstable_by_key(|e| (e.0, e.1));
        all.dedup_by_key(|e| (e.0, e.1));
        ferry.sort_unstable_by_key(|e| (e.0, e.1));
        let mut off = vec![0u32; np + 1];
        for e in &all {
            off[e.0 as usize + 1] += 1;
        }
        for i in 0..np {
            off[i + 1] += off[i];
        }
        Ok(Drives { off, to: all.into_iter().map(|e| (e.1 as u16, e.2, e.3)).collect(), ferry })
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
    fn a_ferry_is_a_base_and_per_km() {
        let f = |km, n| Drive { ferry_km: km, ferries: n, ..Drive::RULE }.ferry_min();
        assert_eq!(f(80., 0), 0.);
        let per = FERRY_MIN + FERRY_USD * FARE_MIN_PER_USD / LONG_OCCUPANCY;
        assert!((f(0., 1) - per).abs() < 1e-4);
        assert!((f(70., 1) - f(35., 1) - (60. + 35. * FERRY_USD_PER_KM * FARE_MIN_PER_USD / LONG_OCCUPANCY)).abs() < 1e-3);
    }
}
