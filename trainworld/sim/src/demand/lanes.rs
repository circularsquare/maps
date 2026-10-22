//! Four f32 lanes for the commute pair loop (T-036): wasm simd128 when the build enables it,
//! SSE2 on x86_64, plain arrays elsewhere. Every operation is the IEEE lane-wise one (no fused
//! multiply-add), and `exp` and `ln` are `kernel::fast_exp` and `kernel::fast_ln` step for step,
//! so a lane gives the bits the scalar expression does.

pub use imp::F4;

const HALF_SQRT2: i32 = 0x3f35_04f3;
const E: [f32; 5] = [0.693_147_2, 0.240_226_5, 0.055_504_11, 0.009_618_129, 0.001_333_355_8];
const L: [f32; 6] = [1.000_004_68, -0.499_903_616, 0.332_567_778, -0.253_978_318, 0.220_702_996, -0.143_383_126];

impl std::ops::Add for F4 {
    type Output = F4;
    #[inline(always)]
    fn add(self, b: F4) -> F4 {
        self.add_(b)
    }
}
impl std::ops::Sub for F4 {
    type Output = F4;
    #[inline(always)]
    fn sub(self, b: F4) -> F4 {
        self.sub_(b)
    }
}
impl std::ops::Mul for F4 {
    type Output = F4;
    #[inline(always)]
    fn mul(self, b: F4) -> F4 {
        self.mul_(b)
    }
}
impl std::ops::Div for F4 {
    type Output = F4;
    #[inline(always)]
    fn div(self, b: F4) -> F4 {
        self.div_(b)
    }
}

impl F4 {
    /// `row[ix[l]]` per lane.
    #[inline(always)]
    pub fn gather(row: &[f32], ix: &[u32; 4]) -> F4 {
        F4::new([row[ix[0] as usize], row[ix[1] as usize], row[ix[2] as usize], row[ix[3] as usize]])
    }
    /// Four rows of four lanes as four columns.
    #[inline(always)]
    pub fn transpose(r: [F4; 4]) -> [F4; 4] {
        imp::transpose(r)
    }
    /// -x, as the scalar negation (sign bit flipped).
    #[inline(always)]
    pub fn neg(self) -> F4 {
        F4::splat(-0.0) - self
    }
    /// `kernel::fast_exp` per lane.
    #[inline(always)]
    pub fn exp(self) -> F4 {
        let t = self.max(F4::splat(-80.0)).min(F4::splat(80.0)) * F4::splat(std::f32::consts::LOG2_E);
        let (i, ii) = t.floor_i();
        let f = t - i;
        let c = |k: usize| F4::splat(E[k]);
        let p = F4::splat(1.0) + f * (c(0) + f * (c(1) + f * (c(2) + f * (c(3) + f * c(4)))));
        F4::exp2_int(ii) * p
    }
    /// `kernel::fast_ln` per lane.
    #[inline(always)]
    pub fn ln(self) -> F4 {
        let (e, t) = self.ln_split(HALF_SQRT2);
        let t = t - F4::splat(1.0);
        let c = |k: usize| F4::splat(L[k]);
        let q = c(0) + t * (c(1) + t * (c(2) + t * (c(3) + t * (c(4) + t * c(5)))));
        e * F4::splat(std::f32::consts::LN_2) + t * q
    }
}

/// Two `F4` side by side: the pair loop's chains (logsum, ln, exp, divisions) are long, and two
/// independent ones per step keep the processor busy where one leaves it waiting on latency.
#[derive(Clone, Copy)]
pub struct F8(F4, F4);

macro_rules! op8 {
    ($tr:ident, $f:ident) => {
        impl std::ops::$tr for F8 {
            type Output = F8;
            #[inline(always)]
            fn $f(self, b: F8) -> F8 {
                F8(std::ops::$tr::$f(self.0, b.0), std::ops::$tr::$f(self.1, b.1))
            }
        }
    };
}
op8!(Add, add);
op8!(Sub, sub);
op8!(Mul, mul);
op8!(Div, div);

impl F8 {
    #[inline(always)]
    pub fn new(a: [f32; 8]) -> F8 {
        F8(F4::new([a[0], a[1], a[2], a[3]]), F4::new([a[4], a[5], a[6], a[7]]))
    }
    #[inline(always)]
    pub fn splat(x: f32) -> F8 {
        F8(F4::splat(x), F4::splat(x))
    }
    #[inline(always)]
    pub fn to_array(self) -> [f32; 8] {
        let (a, b) = (self.0.to_array(), self.1.to_array());
        [a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3]]
    }
    #[inline(always)]
    pub fn gather(row: &[f32], ix: &[u32; 8]) -> F8 {
        F8(F4::gather(row, &[ix[0], ix[1], ix[2], ix[3]]), F4::gather(row, &[ix[4], ix[5], ix[6], ix[7]]))
    }
    /// Eight rows of four as four columns of eight.
    #[inline(always)]
    pub fn transpose(r: [[f32; 4]; 8]) -> [F8; 4] {
        let a = F4::transpose([F4::new(r[0]), F4::new(r[1]), F4::new(r[2]), F4::new(r[3])]);
        let b = F4::transpose([F4::new(r[4]), F4::new(r[5]), F4::new(r[6]), F4::new(r[7])]);
        [F8(a[0], b[0]), F8(a[1], b[1]), F8(a[2], b[2]), F8(a[3], b[3])]
    }
    /// Four columns of eight as eight rows of four: lane l of each, in order.
    #[inline(always)]
    pub fn by_lane(c: [F8; 4]) -> [F4; 8] {
        let a = F4::transpose([c[0].0, c[1].0, c[2].0, c[3].0]);
        let b = F4::transpose([c[0].1, c[1].1, c[2].1, c[3].1]);
        [a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3]]
    }
    #[inline(always)]
    pub fn min(self, b: F8) -> F8 {
        F8(self.0.min(b.0), self.1.min(b.1))
    }
    #[inline(always)]
    pub fn max(self, b: F8) -> F8 {
        F8(self.0.max(b.0), self.1.max(b.1))
    }
    #[inline(always)]
    pub fn sqrt(self) -> F8 {
        F8(self.0.sqrt(), self.1.sqrt())
    }
    #[inline(always)]
    pub fn neg(self) -> F8 {
        F8(self.0.neg(), self.1.neg())
    }
    #[inline(always)]
    pub fn exp(self) -> F8 {
        F8(self.0.exp(), self.1.exp())
    }
    #[inline(always)]
    pub fn ln(self) -> F8 {
        F8(self.0.ln(), self.1.ln())
    }
    /// Bit l set where lane l < b.
    #[inline(always)]
    pub fn lt_mask(self, b: F8) -> u32 {
        self.0.lt_mask(b.0) | self.1.lt_mask(b.1) << 4
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod imp {
    use core::arch::wasm32::*;

    #[derive(Clone, Copy)]
    pub struct F4(v128);

    #[inline(always)]
    pub fn transpose(r: [F4; 4]) -> [F4; 4] {
        let t0 = i32x4_shuffle::<0, 4, 1, 5>(r[0].0, r[1].0);
        let t1 = i32x4_shuffle::<0, 4, 1, 5>(r[2].0, r[3].0);
        let t2 = i32x4_shuffle::<2, 6, 3, 7>(r[0].0, r[1].0);
        let t3 = i32x4_shuffle::<2, 6, 3, 7>(r[2].0, r[3].0);
        [F4(i32x4_shuffle::<0, 1, 4, 5>(t0, t1)), F4(i32x4_shuffle::<2, 3, 6, 7>(t0, t1)), F4(i32x4_shuffle::<0, 1, 4, 5>(t2, t3)), F4(i32x4_shuffle::<2, 3, 6, 7>(t2, t3))]
    }

    impl F4 {
        #[inline(always)]
        pub fn new(a: [f32; 4]) -> F4 {
            F4(f32x4(a[0], a[1], a[2], a[3]))
        }
        #[inline(always)]
        pub fn splat(x: f32) -> F4 {
            F4(f32x4_splat(x))
        }
        #[inline(always)]
        pub fn to_array(self) -> [f32; 4] {
            [f32x4_extract_lane::<0>(self.0), f32x4_extract_lane::<1>(self.0), f32x4_extract_lane::<2>(self.0), f32x4_extract_lane::<3>(self.0)]
        }
        #[inline(always)]
        pub(super) fn add_(self, b: F4) -> F4 {
            F4(f32x4_add(self.0, b.0))
        }
        #[inline(always)]
        pub(super) fn sub_(self, b: F4) -> F4 {
            F4(f32x4_sub(self.0, b.0))
        }
        #[inline(always)]
        pub(super) fn mul_(self, b: F4) -> F4 {
            F4(f32x4_mul(self.0, b.0))
        }
        #[inline(always)]
        pub(super) fn div_(self, b: F4) -> F4 {
            F4(f32x4_div(self.0, b.0))
        }
        #[inline(always)]
        pub fn min(self, b: F4) -> F4 {
            F4(f32x4_min(self.0, b.0))
        }
        #[inline(always)]
        pub fn max(self, b: F4) -> F4 {
            F4(f32x4_max(self.0, b.0))
        }
        #[inline(always)]
        pub fn sqrt(self) -> F4 {
            F4(f32x4_sqrt(self.0))
        }
        /// Bit l set where lane l < b.
        #[inline(always)]
        pub fn lt_mask(self, b: F4) -> u32 {
            i32x4_bitmask(f32x4_lt(self.0, b.0)) as u32
        }
        /// floor, as a float and as an integer.
        #[inline(always)]
        pub(super) fn floor_i(self) -> (F4, v128) {
            let f = f32x4_floor(self.0);
            (F4(f), i32x4_trunc_sat_f32x4(f))
        }
        /// 2^i for integer lanes i (normal range).
        #[inline(always)]
        pub(super) fn exp2_int(i: v128) -> F4 {
            F4(i32x4_shl(i32x4_add(i, i32x4_splat(127)), 23))
        }
        /// The exponent and the folded mantissa of `fast_ln`.
        #[inline(always)]
        pub(super) fn ln_split(self, half: i32) -> (F4, F4) {
            let b = i32x4_sub(self.0, i32x4_splat(half));
            let e = f32x4_convert_i32x4(i32x4_shr(b, 23));
            let m = i32x4_add(v128_and(b, i32x4_splat(0x007f_ffff)), i32x4_splat(half));
            (F4(e), F4(m))
        }
    }
}

#[cfg(target_arch = "x86_64")]
mod imp {
    use core::arch::x86_64::*;

    #[derive(Clone, Copy)]
    pub struct F4(__m128);

    #[inline(always)]
    pub fn transpose(r: [F4; 4]) -> [F4; 4] {
        unsafe {
            let t0 = _mm_unpacklo_ps(r[0].0, r[1].0);
            let t1 = _mm_unpacklo_ps(r[2].0, r[3].0);
            let t2 = _mm_unpackhi_ps(r[0].0, r[1].0);
            let t3 = _mm_unpackhi_ps(r[2].0, r[3].0);
            [F4(_mm_movelh_ps(t0, t1)), F4(_mm_movehl_ps(t1, t0)), F4(_mm_movelh_ps(t2, t3)), F4(_mm_movehl_ps(t3, t2))]
        }
    }

    // SSE2 is part of x86_64, so these intrinsics are always available.
    impl F4 {
        #[inline(always)]
        pub fn new(a: [f32; 4]) -> F4 {
            unsafe { F4(_mm_loadu_ps(a.as_ptr())) }
        }
        #[inline(always)]
        pub fn splat(x: f32) -> F4 {
            unsafe { F4(_mm_set1_ps(x)) }
        }
        #[inline(always)]
        pub fn to_array(self) -> [f32; 4] {
            let mut a = [0f32; 4];
            unsafe { _mm_storeu_ps(a.as_mut_ptr(), self.0) };
            a
        }
        #[inline(always)]
        pub(super) fn add_(self, b: F4) -> F4 {
            unsafe { F4(_mm_add_ps(self.0, b.0)) }
        }
        #[inline(always)]
        pub(super) fn sub_(self, b: F4) -> F4 {
            unsafe { F4(_mm_sub_ps(self.0, b.0)) }
        }
        #[inline(always)]
        pub(super) fn mul_(self, b: F4) -> F4 {
            unsafe { F4(_mm_mul_ps(self.0, b.0)) }
        }
        #[inline(always)]
        pub(super) fn div_(self, b: F4) -> F4 {
            unsafe { F4(_mm_div_ps(self.0, b.0)) }
        }
        #[inline(always)]
        pub fn min(self, b: F4) -> F4 {
            unsafe { F4(_mm_min_ps(self.0, b.0)) }
        }
        #[inline(always)]
        pub fn max(self, b: F4) -> F4 {
            unsafe { F4(_mm_max_ps(self.0, b.0)) }
        }
        #[inline(always)]
        pub fn sqrt(self) -> F4 {
            unsafe { F4(_mm_sqrt_ps(self.0)) }
        }
        #[inline(always)]
        pub fn lt_mask(self, b: F4) -> u32 {
            unsafe { _mm_movemask_ps(_mm_cmplt_ps(self.0, b.0)) as u32 }
        }
        /// floor by truncation and a step down where that rounded up (exact for |x| < 2^31).
        #[inline(always)]
        pub(super) fn floor_i(self) -> (F4, __m128i) {
            unsafe {
                let ti = _mm_cvttps_epi32(self.0);
                let tf = _mm_cvtepi32_ps(ti);
                let up = _mm_cmpgt_ps(tf, self.0);
                let f = _mm_sub_ps(tf, _mm_and_ps(up, _mm_set1_ps(1.0)));
                (F4(f), _mm_add_epi32(ti, _mm_castps_si128(up)))
            }
        }
        #[inline(always)]
        pub(super) fn exp2_int(i: __m128i) -> F4 {
            unsafe { F4(_mm_castsi128_ps(_mm_slli_epi32(_mm_add_epi32(i, _mm_set1_epi32(127)), 23))) }
        }
        #[inline(always)]
        pub(super) fn ln_split(self, half: i32) -> (F4, F4) {
            unsafe {
                let b = _mm_sub_epi32(_mm_castps_si128(self.0), _mm_set1_epi32(half));
                let e = _mm_cvtepi32_ps(_mm_srai_epi32(b, 23));
                let m = _mm_add_epi32(_mm_and_si128(b, _mm_set1_epi32(0x007f_ffff)), _mm_set1_epi32(half));
                (F4(e), F4(_mm_castsi128_ps(m)))
            }
        }
    }
}

#[cfg(not(any(all(target_arch = "wasm32", target_feature = "simd128"), target_arch = "x86_64")))]
mod imp {
    #[derive(Clone, Copy)]
    pub struct F4([f32; 4]);

    #[inline(always)]
    pub fn transpose(r: [F4; 4]) -> [F4; 4] {
        std::array::from_fn(|c| F4(std::array::from_fn(|l| r[l].0[c])))
    }

    impl F4 {
        #[inline(always)]
        fn map(self, b: F4, f: impl Fn(f32, f32) -> f32) -> F4 {
            F4(std::array::from_fn(|l| f(self.0[l], b.0[l])))
        }
        #[inline(always)]
        pub fn new(a: [f32; 4]) -> F4 {
            F4(a)
        }
        #[inline(always)]
        pub fn splat(x: f32) -> F4 {
            F4([x; 4])
        }
        #[inline(always)]
        pub fn to_array(self) -> [f32; 4] {
            self.0
        }
        #[inline(always)]
        pub(super) fn add_(self, b: F4) -> F4 {
            self.map(b, |x, y| x + y)
        }
        #[inline(always)]
        pub(super) fn sub_(self, b: F4) -> F4 {
            self.map(b, |x, y| x - y)
        }
        #[inline(always)]
        pub(super) fn mul_(self, b: F4) -> F4 {
            self.map(b, |x, y| x * y)
        }
        #[inline(always)]
        pub(super) fn div_(self, b: F4) -> F4 {
            self.map(b, |x, y| x / y)
        }
        #[inline(always)]
        pub fn min(self, b: F4) -> F4 {
            self.map(b, f32::min)
        }
        #[inline(always)]
        pub fn max(self, b: F4) -> F4 {
            self.map(b, f32::max)
        }
        #[inline(always)]
        pub fn sqrt(self) -> F4 {
            F4(self.0.map(f32::sqrt))
        }
        #[inline(always)]
        pub fn lt_mask(self, b: F4) -> u32 {
            (0..4).fold(0, |m, l| m | ((self.0[l] < b.0[l]) as u32) << l)
        }
        #[inline(always)]
        pub(super) fn floor_i(self) -> (F4, [i32; 4]) {
            let f = self.0.map(f32::floor);
            (F4(f), f.map(|x| x as i32))
        }
        #[inline(always)]
        pub(super) fn exp2_int(i: [i32; 4]) -> F4 {
            F4(i.map(|i| f32::from_bits(((i + 127) << 23) as u32)))
        }
        #[inline(always)]
        pub(super) fn ln_split(self, half: i32) -> (F4, F4) {
            let b = self.0.map(|x| (x.to_bits() as i32).wrapping_sub(half));
            (F4(b.map(|b| (b >> 23) as f32)), F4(b.map(|b| f32::from_bits(((b & 0x007f_ffff) + half) as u32))))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::demand::kernel::{fast_exp, fast_ln};

    #[test]
    fn lanes_match_the_scalar_exp_and_ln_to_the_bit() {
        let mut x = -100.0f32;
        while x < 100.0 {
            let v = [x, -x, x * 0.37, x + 0.5];
            let e = F4::new(v).exp().to_array();
            for l in 0..4 {
                assert_eq!(e[l].to_bits(), fast_exp(v[l]).to_bits(), "exp {}", v[l]);
            }
            x += 0.013;
        }
        let mut y = 1e-30f32;
        while y < 1e6 {
            let v = [y, y * 1.5, y * 0.75, y * 3.1];
            let r = F4::new(v).ln().to_array();
            for l in 0..4 {
                assert_eq!(r[l].to_bits(), fast_ln(v[l]).to_bits(), "ln {}", v[l]);
            }
            y *= 1.0007;
        }
        let rows: [[f32; 4]; 8] = std::array::from_fn(|l| std::array::from_fn(|c| (10 * l + c) as f32));
        let cols = F8::transpose(rows);
        for (c, col) in cols.iter().enumerate() {
            assert_eq!(col.to_array(), std::array::from_fn(|l| rows[l][c]));
        }
        for (l, r) in F8::by_lane(cols).iter().enumerate() {
            assert_eq!(r.to_array(), rows[l]);
        }
        for v in [[0.0f32, -0.0, 1.0, -1.0], [79.9, 80.0, 80.1, -80.1]] {
            let e = F4::new(v).exp().to_array();
            for l in 0..4 {
                assert_eq!(e[l].to_bits(), fast_exp(v[l]).to_bits());
            }
        }
    }
}
