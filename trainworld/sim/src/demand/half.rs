//! T-3160: a table of trip figures at 16 bits a number, for figures only a view reads (the commutes' station pairs,
//! `api::SubOut::od`): each a half float (5 exponent bits, 10 of mantissa) of the figure under one power-of-two scale
//! for the whole table, chosen so its largest number fits. A number reads back within 1 part in 2,048 of the f32 it was
//! made from (0 exactly; one under 2^-14 of the scale within 2^-25 of it), in half the memory.

/// Half floats under the table's scale, `2^shift`.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct HalfTable {
    bits: Vec<u16>,
    shift: i32,
}

/// The largest finite half float.
const HALF_MAX: f32 = 65504.0;

impl HalfTable {
    /// The table of the `n` numbers `at(0)`, `at(1)`, ...; `at` is called twice for each, once to find the largest,
    /// so no f32 copy of the table is held while it is made.
    pub(crate) fn build(n: usize, at: impl Fn(usize) -> f32) -> HalfTable {
        let mut big = 0f32;
        for i in 0..n {
            big = big.max(at(i).abs());
        }
        let shift = shift_for(big);
        let down = pow2(-shift);
        HalfTable { bits: (0..n).map(|i| to_half(at(i) * down)).collect(), shift }
    }

    #[cfg(test)]
    pub(crate) fn from_f32(v: &[f32]) -> HalfTable {
        Self::build(v.len(), |i| v[i])
    }

    pub(crate) fn len(&self) -> usize {
        self.bits.len()
    }

    #[inline]
    pub(crate) fn get(&self, i: usize) -> f32 {
        from_half(self.bits[i]) * pow2(self.shift)
    }

    /// Numbers `a..b`.
    pub(crate) fn range(&self, a: usize, b: usize) -> impl Iterator<Item = f32> + '_ {
        let up = pow2(self.shift);
        self.bits[a..b].iter().map(move |&h| from_half(h) * up)
    }

    /// `kernel::msa` on the numbers as read: `a += (n - a) * w` at the new pass's weight, over the numbers both have.
    pub(crate) fn msa(&mut self, new: &HalfTable, round: usize) {
        let w = 1.0 / (round as f32 + 1.0);
        let n = new.len().min(self.len());
        let old = &*self;
        let out = HalfTable::build(old.len(), |i| {
            let a = old.get(i);
            if i < n { a + (new.get(i) - a) * w } else { a }
        });
        *self = out;
    }

    #[cfg(test)]
    pub(crate) fn to_f32(&self) -> Vec<f32> {
        (0..self.len()).map(|i| self.get(i)).collect()
    }

    pub(crate) fn heap_bytes(&self) -> usize {
        self.bits.capacity() * std::mem::size_of::<u16>()
    }
}

/// `2^e` for `e` in -126..=127, exactly.
#[inline]
fn pow2(e: i32) -> f32 {
    f32::from_bits(((e + 127) as u32) << 23)
}

/// The least `shift` from 0 up that brings `big` to the largest half float or under.
fn shift_for(big: f32) -> i32 {
    let mut s = 0;
    while s < 120 && big * pow2(-s) > HALF_MAX {
        s += 1;
    }
    s
}

/// `x` as a half float, rounded to the nearest (ties to even); past the largest finite one, that one (NaN stays NaN).
fn to_half(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let a = b & 0x7fff_ffff;
    let mag = if a > 0x7f80_0000 {
        0x7e00
    } else if a >= 0x477f_f000 {
        0x7bff
    } else if a < 0x3880_0000 {
        // under 2^-14, a subnormal half: adding 0.5, whose last bit is 2^-24, rounds to the half's last bit
        ((f32::from_bits(a) + 0.5).to_bits() - 0x3f00_0000) as u16
    } else {
        let odd = (a >> 13) & 1;
        (a.wrapping_sub(112 << 23).wrapping_add(0xfff + odd) >> 13) as u16
    };
    sign | mag
}

/// A half float as an f32, exactly.
#[inline]
fn from_half(h: u16) -> f32 {
    let e = ((h >> 10) & 0x1f) as u32;
    let m = (h & 0x3ff) as u32;
    let mag = if e == 0 {
        m as f32 * (1.0 / 16_777_216.0)
    } else if e == 31 {
        if m == 0 { f32::INFINITY } else { f32::NAN }
    } else {
        f32::from_bits(((e + 112) << 23) | (m << 13))
    };
    if h & 0x8000 != 0 { -mag } else { mag }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A spread of figures from 1e-9 to 3e9, both signs, some 0.
    fn figures(n: usize, top: f32) -> Vec<f32> {
        let mut s = 0x2545_f491u32;
        (0..n)
            .map(|i| {
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                if i % 97 == 0 {
                    return 0.0;
                }
                let u = (s >> 8) as f32 / (1u32 << 24) as f32;
                let x = top * 10f32.powf(-12.0 * u);
                if i % 5 == 0 { -x } else { x }
            })
            .collect()
    }

    #[test]
    fn every_half_reads_back_to_itself() {
        for h in 0..=u16::MAX {
            if (h >> 10) & 0x1f == 31 {
                continue;
            }
            let x = from_half(h);
            assert_eq!(to_half(x), h, "half {h:#06x} ({x})");
        }
    }

    #[test]
    fn rounds_to_the_nearest_half() {
        for &x in &figures(20_000, 60_000.0) {
            let h = to_half(x);
            let y = from_half(h);
            let err = (y - x).abs();
            // no other finite half of the same sign is closer
            for d in [-1i32, 1] {
                let o = (h as i32 + d) as u16;
                if o & 0x7fff < 0x7c00 && (o ^ h) & 0x8000 == 0 {
                    assert!((from_half(o) - x).abs() >= err, "{x}: {y} against {}", from_half(o));
                }
            }
        }
    }

    #[test]
    fn a_table_reads_back_within_one_part_in_2048() {
        for top in [0.25f32, 60_000.0, 65_504.0, 70_000.0, 3e9] {
            let v = figures(50_000, top);
            let t = HalfTable::from_f32(&v);
            let scale = pow2(t.shift);
            assert!(t.shift == 0 || top > HALF_MAX, "{top}: shift {}", t.shift);
            for (i, (&x, y)) in v.iter().zip(t.to_f32()).enumerate() {
                let tol = (x.abs() / 2048.0).max(scale / 33_554_432.0);
                assert!((y - x).abs() <= tol, "{top}, {i}: {x} read back as {y}");
                if x == 0.0 {
                    assert_eq!(y, 0.0);
                }
                assert_eq!(x.is_sign_negative(), y.is_sign_negative(), "{x}");
            }
            assert_eq!(t.heap_bytes(), v.len() * 2);
        }
    }

    #[test]
    fn msa_follows_the_f32_average() {
        let a = figures(10_000, 4_000.0).iter().map(|x| x.abs()).collect::<Vec<_>>();
        let b = figures(10_000, 90_000.0).iter().rev().map(|x| x.abs()).collect::<Vec<_>>();
        let mut t = HalfTable::from_f32(&a);
        let mut want = a.clone();
        for round in 1..6 {
            let new = HalfTable::from_f32(&b);
            t.msa(&new, round);
            crate::demand::kernel::msa(&mut want, &b, round);
            for (i, (&x, y)) in want.iter().zip(t.to_f32()).enumerate() {
                // each round reads back within 1 part in 2,048; the averaging shrinks earlier rounds' errors
                assert!((y - x).abs() <= x.abs() / 600.0 + 1e-6, "round {round}, {i}: {x} against {y}");
            }
        }
    }

    #[test]
    fn a_view_sum_is_within_one_part_in_2048() {
        // a view reads a row times a weight per number, summed
        let v = figures(1454 * 64, 30_000.0).iter().map(|x| x.abs()).collect::<Vec<_>>();
        let t = HalfTable::from_f32(&v);
        for r in 0..64 {
            let w = |j: usize| 0.1 + (j % 13) as f32 / 13.0;
            let want: f64 = (0..1454).map(|j| (v[r * 1454 + j] * w(j)) as f64).sum();
            let got: f64 = t.range(r * 1454, (r + 1) * 1454).enumerate().map(|(j, x)| (x * w(j)) as f64).sum();
            assert!((got - want).abs() <= want / 2048.0, "row {r}: {want} against {got}");
        }
    }
}
