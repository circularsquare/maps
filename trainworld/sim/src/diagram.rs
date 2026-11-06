//! The network diagram's carry on the rubber sheet (app/src/diagram/sheetLines.ts, T-2589): the least squares over the
//! sheet's finest lattice, a 9-point matrix (slot (dj + 1) * 3 + di + 1 per node), solved by conjugate gradients with a
//! multigrid V-cycle, both axes at once (x and y per node, interleaved, one f64x2). Every float operation and its order
//! is the JavaScript's (`carryMatrix` aside, which stays there), so the result is the same to the bit: lane-wise IEEE
//! add, multiply and divide, never fused, and JavaScript's truthiness where it tests a number.

use wasm_bindgen::prelude::*;

pub use imp::F2;

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod imp {
    use core::arch::wasm32::*;

    #[derive(Clone, Copy)]
    pub struct F2(v128);
    impl F2 {
        #[inline(always)]
        pub fn splat(x: f64) -> F2 {
            F2(f64x2_splat(x))
        }
        #[inline(always)]
        pub fn new(a: f64, b: f64) -> F2 {
            F2(f64x2(a, b))
        }
        /// `s[i], s[i + 1]`; the caller keeps i + 1 in range
        #[inline(always)]
        pub unsafe fn ld(s: *const f64, i: usize) -> F2 {
            F2(v128_load(s.add(i) as *const v128))
        }
        #[inline(always)]
        pub unsafe fn st(self, s: *mut f64, i: usize) {
            v128_store(s.add(i) as *mut v128, self.0)
        }
        #[inline(always)]
        pub fn lo(self) -> f64 {
            f64x2_extract_lane::<0>(self.0)
        }
        #[inline(always)]
        pub fn hi(self) -> f64 {
            f64x2_extract_lane::<1>(self.0)
        }
        #[inline(always)]
        pub fn add(self, b: F2) -> F2 {
            F2(f64x2_add(self.0, b.0))
        }
        #[inline(always)]
        pub fn sub(self, b: F2) -> F2 {
            F2(f64x2_sub(self.0, b.0))
        }
        #[inline(always)]
        pub fn mul(self, b: F2) -> F2 {
            F2(f64x2_mul(self.0, b.0))
        }
        #[inline(always)]
        pub fn div(self, b: F2) -> F2 {
            F2(f64x2_div(self.0, b.0))
        }
    }
}

#[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
mod imp {
    #[derive(Clone, Copy)]
    pub struct F2(f64, f64);
    impl F2 {
        #[inline(always)]
        pub fn splat(x: f64) -> F2 {
            F2(x, x)
        }
        #[inline(always)]
        pub fn new(a: f64, b: f64) -> F2 {
            F2(a, b)
        }
        #[inline(always)]
        pub unsafe fn ld(s: *const f64, i: usize) -> F2 {
            F2(*s.add(i), *s.add(i + 1))
        }
        #[inline(always)]
        pub unsafe fn st(self, s: *mut f64, i: usize) {
            *s.add(i) = self.0;
            *s.add(i + 1) = self.1;
        }
        #[inline(always)]
        pub fn lo(self) -> f64 {
            self.0
        }
        #[inline(always)]
        pub fn hi(self) -> f64 {
            self.1
        }
        #[inline(always)]
        pub fn add(self, b: F2) -> F2 {
            F2(self.0 + b.0, self.1 + b.1)
        }
        #[inline(always)]
        pub fn sub(self, b: F2) -> F2 {
            F2(self.0 - b.0, self.1 - b.1)
        }
        #[inline(always)]
        pub fn mul(self, b: F2) -> F2 {
            F2(self.0 * b.0, self.1 * b.1)
        }
        #[inline(always)]
        pub fn div(self, b: F2) -> F2 {
            F2(self.0 / b.0, self.1 / b.1)
        }
    }
}


/// JavaScript's `!!v` for a number
#[inline(always)]
fn truthy(v: f64) -> bool {
    v != 0.0 && !v.is_nan()
}

/// s + a * x[i..i + 2]
#[inline(always)]
unsafe fn madd(s: F2, a: f64, x: *const f64, i: usize) -> F2 {
    s.add(F2::splat(a).mul(F2::ld(x, i)))
}
/// s - a * x[i..i + 2]
#[inline(always)]
unsafe fn msub(s: F2, a: f64, x: *const f64, i: usize) -> F2 {
    s.sub(F2::splat(a).mul(F2::ld(x, i)))
}

/// where each of a node's 9 slots is kept: among its 5 cross terms (true) or its 4 corner terms
const SLOT: [(bool, usize); 9] = [(false, 0), (true, 0), (false, 1), (true, 1), (true, 2), (true, 3), (false, 2), (true, 4), (false, 3)];

/// A level's matrix: per node its 5 cross terms (slots 1, 3, 4, 5, 7) and, apart, its 4 corner terms (0, 2, 6, 8),
/// which a node inside the lattice reads only where `corner` flags any (on the finest level only stations' nodes): the
/// lattice's passes read 40 bytes a node, not 72. `corner` as the JavaScript's `cornersOf` when the matrix was made.
#[derive(Default)]
struct Mat {
    nx: usize,
    ny: usize,
    s5: Vec<f64>,
    c4: Vec<f64>,
    corner: Vec<u8>,
}

impl Mat {
    fn new(nx: usize, ny: usize, a: &[f64]) -> Mat {
        let n = nx * ny;
        assert_eq!(a.len(), 9 * n);
        let (mut s5, mut c4) = (vec![0.0; 5 * n], vec![0.0; 4 * n]);
        for v in 0..n {
            for k in 0..9 {
                match SLOT[k] {
                    (true, s) => s5[5 * v + s] = a[9 * v + k],
                    (false, s) => c4[4 * v + s] = a[9 * v + k],
                }
            }
        }
        let mut m = Mat { nx, ny, s5, c4, corner: Vec::new() };
        m.corner = m.flags();
        m
    }
    #[inline(always)]
    fn get(&self, v: usize, k: usize) -> f64 {
        match SLOT[k] {
            (true, s) => self.s5[5 * v + s],
            (false, s) => self.c4[4 * v + s],
        }
    }
    fn set(&mut self, at: usize, x: f64) {
        let (v, k) = (at / 9, at % 9);
        match SLOT[k] {
            (true, s) => self.s5[5 * v + s] = x,
            (false, s) => self.c4[4 * v + s] = x,
        }
    }
    /// per node whether any of its corner terms is not 0, as they are now
    fn flags(&self) -> Vec<u8> {
        self.c4.chunks_exact(4).map(|c| (c[0] != 0.0 || c[1] != 0.0 || c[2] != 0.0 || c[3] != 0.0) as u8).collect()
    }
}

/// y = A x at node (i, j) on the lattice's edge, or from - A x; the terms in the JavaScript's order (itself, west, east,
/// the row below, the row above)
#[inline(always)]
unsafe fn mul_edge(m: &Mat, x: *const f64, y: *mut f64, i: usize, j: usize, from: Option<*const f64>) {
    let (nx, ny) = (m.nx, m.ny);
    let v = j * nx + i;
    let p = 2 * v;
    let q = 2 * nx;
    let mut s = F2::splat(m.get(v, 4)).mul(F2::ld(x, p));
    if i > 0 {
        s = madd(s, m.get(v, 3), x, p - 2);
    }
    if i + 1 < nx {
        s = madd(s, m.get(v, 5), x, p + 2);
    }
    if j > 0 {
        s = madd(s, m.get(v, 1), x, p - q);
        if i > 0 {
            s = madd(s, m.get(v, 0), x, p - q - 2);
        }
        if i + 1 < nx {
            s = madd(s, m.get(v, 2), x, p - q + 2);
        }
    }
    if j + 1 < ny {
        s = madd(s, m.get(v, 7), x, p + q);
        if i > 0 {
            s = madd(s, m.get(v, 6), x, p + q - 2);
        }
        if i + 1 < nx {
            s = madd(s, m.get(v, 8), x, p + q + 2);
        }
    }
    match from {
        Some(f) => F2::ld(f, p).sub(s).st(y, p),
        None => s.st(y, p),
    }
}

/// y = A x (or from - A x) along row j, node by node; with DOT, the sum of x y over the row's nodes added to `dot` in
/// their order
#[inline(always)]
unsafe fn mul_row<const DOT: bool>(m: &Mat, corner: *const u8, x: *const f64, y: *mut f64, j: usize, from: Option<*const f64>, dot: &mut F2) {
    let (nx, ny) = (m.nx, m.ny);
    let q = 2 * nx;
    if j == 0 || j + 1 == ny || nx < 3 {
        for i in 0..nx {
            mul_edge(m, x, y, i, j, from);
            if DOT {
                *dot = dot.add(F2::ld(x, 2 * (j * nx + i)).mul(F2::ld(y, 2 * (j * nx + i))));
            }
        }
        return;
    }
    mul_edge(m, x, y, 0, j, from);
    if DOT {
        *dot = dot.add(F2::ld(x, 2 * j * nx).mul(F2::ld(y, 2 * j * nx)));
    }
    let (s5, c4) = (m.s5.as_ptr(), m.c4.as_ptr());
    for v in j * nx + 1..j * nx + nx - 1 {
        let (b, p) = (5 * v, 2 * v);
        let (a1, a3, a4, a5, a7) = (*s5.add(b), *s5.add(b + 1), *s5.add(b + 2), *s5.add(b + 3), *s5.add(b + 4));
        let mut s = F2::splat(a4).mul(F2::ld(x, p));
        s = madd(s, a3, x, p - 2);
        s = madd(s, a5, x, p + 2);
        s = madd(s, a1, x, p - q);
        if *corner.add(v) != 0 {
            let c = 4 * v;
            let (a0, a2, a6, a8) = (*c4.add(c), *c4.add(c + 1), *c4.add(c + 2), *c4.add(c + 3));
            s = madd(s, a0, x, p - q - 2);
            s = madd(s, a2, x, p - q + 2);
            s = madd(s, a7, x, p + q);
            s = madd(s, a6, x, p + q - 2);
            s = madd(s, a8, x, p + q + 2);
        } else {
            s = madd(s, a7, x, p + q);
        }
        let out = match from {
            Some(f) => F2::ld(f, p).sub(s),
            None => s,
        };
        out.st(y, p);
        if DOT {
            *dot = dot.add(F2::ld(x, p).mul(out));
        }
    }
    let v = j * nx + nx - 1;
    mul_edge(m, x, y, nx - 1, j, from);
    if DOT {
        *dot = dot.add(F2::ld(x, 2 * v).mul(F2::ld(y, 2 * v)));
    }
}

fn check(m: &Mat, corner: &[u8], vecs: &[usize]) {
    let n = m.nx * m.ny;
    assert!(m.s5.len() == 5 * n && m.c4.len() == 4 * n && corner.len() >= n && vecs.iter().all(|&l| l >= 2 * n));
}

/// y = A x; with DOT, and the sum of x y in node order (as a pass over both after it would add it)
fn mul9<const DOT: bool>(m: &Mat, corner: &[u8], x: &[f64], y: &mut [f64]) -> F2 {
    check(m, corner, &[x.len(), y.len()]);
    let mut dot = F2::splat(0.0);
    for j in 0..m.ny {
        unsafe { mul_row::<DOT>(m, corner.as_ptr(), x.as_ptr(), y.as_mut_ptr(), j, None, &mut dot) };
    }
    dot
}

/// x at a node from its sum s and diagonal d (0 where d is not positive)
#[inline(always)]
unsafe fn gs_set(x: *mut f64, p: usize, s: F2, d: f64) {
    if d > 0.0 {
        s.div(F2::splat(d)).st(x, p)
    } else {
        F2::splat(0.0).st(x, p)
    }
}

/// one Gauss-Seidel step at node (i, j) on the lattice's edge
#[inline(always)]
unsafe fn gs_edge(m: &Mat, x: *mut f64, r: *const f64, i: usize, j: usize) {
    let (nx, ny) = (m.nx, m.ny);
    let v = j * nx + i;
    let p = 2 * v;
    let q = 2 * nx;
    let xc = x as *const f64;
    let mut s = F2::ld(r, p);
    if i > 0 {
        s = msub(s, m.get(v, 3), xc, p - 2);
    }
    if i + 1 < nx {
        s = msub(s, m.get(v, 5), xc, p + 2);
    }
    if j > 0 {
        s = msub(s, m.get(v, 1), xc, p - q);
        if i > 0 {
            s = msub(s, m.get(v, 0), xc, p - q - 2);
        }
        if i + 1 < nx {
            s = msub(s, m.get(v, 2), xc, p - q + 2);
        }
    }
    if j + 1 < ny {
        s = msub(s, m.get(v, 7), xc, p + q);
        if i > 0 {
            s = msub(s, m.get(v, 6), xc, p + q - 2);
        }
        if i + 1 < nx {
            s = msub(s, m.get(v, 8), xc, p + q + 2);
        }
    }
    gs_set(x, p, s, m.get(v, 4));
}

/// one Gauss-Seidel step at node v inside the lattice
#[inline(always)]
unsafe fn gs_inner(m: &Mat, corner: *const u8, x: *mut f64, r: *const f64, v: usize) {
    let (s5, c4) = (m.s5.as_ptr(), m.c4.as_ptr());
    let (b, p, q) = (5 * v, 2 * v, 2 * m.nx);
    let xc = x as *const f64;
    let (a1, a3, d, a5, a7) = (*s5.add(b), *s5.add(b + 1), *s5.add(b + 2), *s5.add(b + 3), *s5.add(b + 4));
    let mut s = F2::ld(r, p);
    s = msub(s, a3, xc, p - 2);
    s = msub(s, a5, xc, p + 2);
    s = msub(s, a1, xc, p - q);
    if *corner.add(v) != 0 {
        let c = 4 * v;
        let (a0, a2, a6, a8) = (*c4.add(c), *c4.add(c + 1), *c4.add(c + 2), *c4.add(c + 3));
        s = msub(s, a0, xc, p - q - 2);
        s = msub(s, a2, xc, p - q + 2);
        s = msub(s, a7, xc, p + q);
        s = msub(s, a6, xc, p + q - 2);
        s = msub(s, a8, xc, p + q + 2);
    } else {
        s = msub(s, a7, xc, p + q);
    }
    gs_set(x, p, s, d);
}

/// rows swept together in a Gauss-Seidel pass, each LAG nodes behind the one before it
const BAND: usize = 4;
const LAG: usize = 2;

/// The Gauss-Seidel sweep's rows jj0 .. jj0 + nb in its order (first to last, or back), each node by node in that
/// order. The rows go together as a wavefront, each LAG nodes behind the one before: a node then reads the values it
/// would in the plain order (its row's last node and the row before done, the row after not yet), so the same bits,
/// while the rows' steps, which do not depend on each other, overlap.
unsafe fn gs_band(m: &Mat, corner: *const u8, x: *mut f64, r: *const f64, fwd: bool, jj0: usize, nb: usize) {
    let (nx, ny) = (m.nx, m.ny);
    let mut rows = [0usize; BAND];
    let mut edge = [false; BAND];
    for k in 0..nb {
        let j = if fwd { jj0 + k } else { ny - 1 - jj0 - k };
        rows[k] = j;
        edge[k] = j == 0 || j + 1 == ny || nx < 3;
    }
    for t in 0..nx + LAG * (nb - 1) {
        for k in 0..nb {
            if t < LAG * k || t - LAG * k >= nx {
                continue;
            }
            let ii = t - LAG * k;
            let (i, j) = (if fwd { ii } else { nx - 1 - ii }, rows[k]);
            if edge[k] || i == 0 || i + 1 == nx {
                gs_edge(m, x, r, i, j);
            } else {
                gs_inner(m, corner, x, r, j * nx + i);
            }
        }
    }
}

/// a multigrid level: its matrix (level 0's is the solver's own), its corner flags as the multigrid was made, and but on
/// the coarsest each node's 4 prolongation weights (its parents follow from its place, see prolong_row; unused slots 0)
struct Level {
    mat: Mat,
    corner: Vec<u8>,
    x: Vec<f64>,
    b: Vec<f64>,
    r: Vec<f64>,
    pw: Vec<f64>,
}

/// Row j's residual r restricted onto the coarse right side cb: each node (where either axis is not 0) adds its weight
/// times its residual to each of its parents, in node order. A coarse node takes a row's terms from 3 nodes in a row
/// (2I - 1, 2I, 2I + 1), so its sum is kept in a register meanwhile: the same additions in the same order.
#[inline(always)]
unsafe fn restrict_row(cb: *mut f64, r: *const f64, pw: &[f64], nx: usize, j: usize) {
    let cx = (nx >> 1) + 1;
    let odd = j & 1 == 1;
    // the coarse row's node I (and with an odd row, the row above's)
    let (row, up) = (2 * (j >> 1) * cx, 2 * ((j >> 1) + 1) * cx);
    let mut acc = F2::ld(cb, row);
    let mut acc_n = if odd { F2::ld(cb, up) } else { F2::splat(0.0) };
    let w = pw.as_ptr();
    for i in 0..nx {
        let v = j * nx + i;
        let (rx, ry) = (*r.add(2 * v), *r.add(2 * v + 1));
        let live = truthy(rx) || truthy(ry);
        let rv = F2::ld(r, 2 * v);
        let s = 4 * v;
        if i & 1 == 0 {
            if live {
                acc = acc.add(F2::splat(*w.add(s)).mul(rv));
                if odd {
                    acc_n = acc_n.add(F2::splat(*w.add(s + 1)).mul(rv));
                }
            }
        } else {
            // parents I and I + 1 (slots: even row 0, 1; odd row SW 0, SE 1, NW 2, NE 3)
            let c = 2 * (i >> 1);
            if odd {
                if live {
                    acc = acc.add(F2::splat(*w.add(s)).mul(rv));
                    acc_n = acc_n.add(F2::splat(*w.add(s + 2)).mul(rv));
                }
                acc.st(cb, row + c);
                acc_n.st(cb, up + c);
                acc = F2::ld(cb, row + c + 2);
                acc_n = F2::ld(cb, up + c + 2);
                if live {
                    acc = acc.add(F2::splat(*w.add(s + 1)).mul(rv));
                    acc_n = acc_n.add(F2::splat(*w.add(s + 3)).mul(rv));
                }
            } else {
                if live {
                    acc = acc.add(F2::splat(*w.add(s)).mul(rv));
                }
                acc.st(cb, row + c);
                acc = F2::ld(cb, row + c + 2);
                if live {
                    acc = acc.add(F2::splat(*w.add(s + 1)).mul(rv));
                }
            }
        }
    }
    let c = 2 * ((nx - 1) >> 1) + if (nx - 1) & 1 == 1 { 2 } else { 0 };
    acc.st(cb, row + c);
    if odd {
        acc_n.st(cb, up + c);
    }
}

/// Row j of x plus the coarse x prolonged onto it: each node from its parents (by its place: 1 on both even, 4 on both odd
/// as SW, SE, NW, NE, else 2) in the order of their slots, from 0. Each node's own, so the even nodes go and then the odd.
#[inline(always)]
unsafe fn prolong_row(x: *mut f64, c: *const f64, pw: &[f64], nx: usize, j: usize) {
    let cx = (nx >> 1) + 1;
    let (row, up) = (2 * (j >> 1) * cx, 2 * ((j >> 1) + 1) * cx);
    let w = pw.as_ptr();
    let z = F2::splat(0.0);
    let odd = j & 1 == 1;
    let mut i = 0;
    while i < nx {
        let (v, ci) = (j * nx + i, 2 * (i >> 1));
        let s = if odd { madd(madd(z, *w.add(4 * v), c, row + ci), *w.add(4 * v + 1), c, up + ci) } else { madd(z, *w.add(4 * v), c, row + ci) };
        F2::ld(x, 2 * v).add(s).st(x, 2 * v);
        i += 2;
    }
    i = 1;
    while i < nx {
        let (v, ci) = (j * nx + i, 2 * (i >> 1));
        let s = if odd {
            madd(madd(madd(madd(z, *w.add(4 * v), c, row + ci), *w.add(4 * v + 1), c, row + ci + 2), *w.add(4 * v + 2), c, up + ci), *w.add(4 * v + 3), c, up + ci + 2)
        } else {
            madd(madd(z, *w.add(4 * v), c, row + ci), *w.add(4 * v + 1), c, row + ci + 2)
        };
        F2::ld(x, 2 * v).add(s).st(x, 2 * v);
        i += 2;
    }
}

/// The V-cycle's way down at a level: x from 0 by a Gauss-Seidel sweep forward on A x = b, its residual r, and r
/// restricted onto the coarse level's right side cb (from 0). Each row's residual and restriction come as soon as the
/// sweep has passed the row after it (the row's matrix still in cache); node for node the same as a pass of each after
/// the other, restricted in node order.
fn down(m: &Mat, corner: &[u8], x: &mut [f64], b: &[f64], r: &mut [f64], pw: &[f64], cb: &mut [f64]) {
    let (nx, ny, n) = (m.nx, m.ny, m.nx * m.ny);
    check(m, corner, &[x.len(), b.len(), r.len()]);
    assert!(pw.len() >= 4 * n && cb.len() >= 2 * ((nx >> 1) + 1) * ((ny >> 1) + 1));
    x.fill(0.0);
    cb.fill(0.0);
    let (xp, bp, rp, cp, cbp) = (x.as_mut_ptr(), b.as_ptr(), r.as_mut_ptr(), corner.as_ptr(), cb.as_mut_ptr());
    let mut done = 0;
    let mut jj0 = 0;
    let mut unused = F2::splat(0.0);
    unsafe {
        while jj0 < ny {
            let nb = BAND.min(ny - jj0);
            gs_band(m, cp, xp, bp, true, jj0, nb);
            jj0 += nb;
            let upto = if jj0 == ny { ny } else { jj0 - 1 };
            for j in done..upto {
                mul_row::<false>(m, cp, xp, rp, j, Some(bp), &mut unused);
                restrict_row(cbp, rp, pw, nx, j);
            }
            done = upto;
        }
    }
}

/// The V-cycle's way up at a level: the coarse level's x prolonged onto x, then a Gauss-Seidel sweep back. Rows are
/// prolonged just before the sweep reads them; each node's prolongation is its own, so the same as all first.
fn up(m: &Mat, corner: &[u8], x: &mut [f64], b: &[f64], pw: &[f64], cx: &[f64]) {
    let (nx, ny, n) = (m.nx, m.ny, m.nx * m.ny);
    check(m, corner, &[x.len(), b.len()]);
    assert!(pw.len() >= 4 * n && cx.len() >= 2 * ((nx >> 1) + 1) * ((ny >> 1) + 1));
    let (xp, bp, cp, cxp) = (x.as_mut_ptr(), b.as_ptr(), corner.as_ptr(), cx.as_ptr());
    // rows from `lo` up are prolonged
    let mut lo = ny;
    let mut jj0 = 0;
    unsafe {
        while jj0 < ny {
            let nb = BAND.min(ny - jj0);
            let need = (ny - jj0 - nb).saturating_sub(1);
            while lo > need {
                lo -= 1;
                prolong_row(xp, cxp, pw, nx, lo);
            }
            gs_band(m, cp, xp, bp, false, jj0, nb);
            jj0 += nb;
        }
    }
}

struct Multigrid {
    lev: Vec<Level>,
    /// the coarsest level's dense Cholesky factor, m by m
    m: usize,
    chol: Vec<f64>,
    yv: Vec<f64>,
}

fn multigrid(a0: &Mat) -> Multigrid {
    let mut lev: Vec<Level> = Vec::new();
    let mut next: Option<Mat> = None;
    loop {
        let own = next.take().unwrap_or_default();
        let a: &Mat = if lev.is_empty() { a0 } else { &own };
        let (nx, ny) = (a.nx, a.ny);
        let n = nx * ny;
        let corner = a.flags();
        let done = nx <= 5 && ny <= 5;
        let (cx, cy) = ((nx >> 1) + 1, (ny >> 1) + 1);
        let (mut c, mut pw) = (Vec::new(), Vec::new());
        if !done {
            c = vec![0.0f64; cx * cy * 9];
            // each node's coarse parents, for C below (the passes work them out from the node's place)
            let mut pc = vec![0u32; 4 * n];
            pw = vec![0.0f64; 4 * n];
            // the prolongation follows the matrix (black box multigrid, Dendy 1982), as the JavaScript's
            let at = |i: usize, j: usize, k: usize| a.get(j * nx + i, k);
            let line = |i: usize, j: usize, horiz: bool| -> (f64, f64) {
                if i >= nx || j >= ny {
                    return (0.0, 0.0);
                }
                let c = if horiz { at(i, j, 1) + at(i, j, 4) + at(i, j, 7) } else { at(i, j, 3) + at(i, j, 4) + at(i, j, 5) };
                let lo = if horiz { at(i, j, 0) + at(i, j, 3) + at(i, j, 6) } else { at(i, j, 0) + at(i, j, 1) + at(i, j, 2) };
                let hi = if horiz { at(i, j, 2) + at(i, j, 5) + at(i, j, 8) } else { at(i, j, 6) + at(i, j, 7) + at(i, j, 8) };
                if c > 1e-300 {
                    (-lo / c, -hi / c)
                } else {
                    (0.5, 0.5)
                }
            };
            for j in 0..ny {
                for i in 0..nx {
                    let v = j * nx + i;
                    let b = 4 * v;
                    let (oi, oj) = (i & 1, j & 1);
                    let base = ((j >> 1) * cx + (i >> 1)) as u32;
                    if oi == 0 && oj == 0 {
                        pc[b] = base;
                        pw[b] = 1.0;
                    } else if oi == 1 && oj == 0 {
                        let (w0, w1) = line(i, j, true);
                        (pc[b], pw[b], pc[b + 1], pw[b + 1]) = (base, w0, base + 1, w1);
                    } else if oi == 0 && oj == 1 {
                        let (w0, w1) = line(i, j, false);
                        (pc[b], pw[b], pc[b + 1], pw[b + 1]) = (base, w0, base + cx as u32, w1);
                    } else {
                        let c0 = at(i, j, 4);
                        let (w_s, w_n) = line(i - 1, j, false);
                        let (e_s, e_n) = line(i + 1, j, false);
                        let (s_w, s_e) = line(i, j - 1, true);
                        let (n_w, n_e) = line(i, j + 1, true);
                        let (aw, ae, as_, an) = (at(i, j, 3), at(i, j, 5), at(i, j, 1), at(i, j, 7));
                        let k = if c0 > 1e-300 { -1.0 / c0 } else { 0.0 };
                        (pc[b], pc[b + 1], pc[b + 2], pc[b + 3]) = (base, base + 1, base + cx as u32, base + cx as u32 + 1);
                        pw[b] = k * (at(i, j, 0) + aw * w_s + as_ * s_w);
                        pw[b + 1] = k * (at(i, j, 2) + ae * e_s + as_ * s_e);
                        pw[b + 2] = k * (at(i, j, 6) + aw * w_n + an * n_w);
                        pw[b + 3] = k * (at(i, j, 8) + ae * e_n + an * n_e);
                        if c0 <= 1e-300 {
                            for q in 0..4 {
                                pw[b + q] = 0.25;
                            }
                        }
                    }
                }
            }
            // C = P^T A P, summed in the JavaScript's order
            for v in 0..n {
                for k in 0..9 {
                    let av = a.get(v, k);
                    if !truthy(av) {
                        continue;
                    }
                    let w = v as isize + (k as isize / 3 - 1) * nx as isize + (k as isize % 3) - 1;
                    for p in 0..4 {
                        let wu = pw[4 * v + p] * av;
                        if !truthy(wu) {
                            continue;
                        }
                        let u = pc[4 * v + p] as isize;
                        let (ui, uj) = (u % cx as isize, u / cx as isize);
                        // a parent past the lattice's ends reads as undefined in the JavaScript: none
                        if w < 0 || w >= n as isize {
                            continue;
                        }
                        let w = w as usize;
                        for p2 in 0..4 {
                            let ww = pw[4 * w + p2];
                            if !truthy(ww) {
                                continue;
                            }
                            let u2 = pc[4 * w + p2] as isize;
                            let (i2, j2) = (u2 % cx as isize, u2 / cx as isize);
                            let at = u * 9 + (j2 - uj + 1) * 3 + (i2 - ui + 1);
                            if at >= 0 && (at as usize) < c.len() {
                                c[at as usize] += wu * ww;
                            }
                        }
                    }
                }
            }
        }
        lev.push(Level { mat: own, corner, x: vec![0.0; 2 * n], b: vec![0.0; 2 * n], r: vec![0.0; 2 * n], pw });
        if done {
            break;
        }
        next = Some(Mat::new(cx, cy, &c));
    }
    // the coarsest by a dense Cholesky factor
    let last = lev.len() - 1;
    let la: &Mat = if last == 0 { a0 } else { &lev[last].mat };
    let (lnx, lny) = (la.nx, la.ny);
    let m = lnx * lny;
    let mut mm = vec![0.0f64; m * m];
    for v in 0..m {
        for k in 0..9 {
            let i2 = (v % lnx) as isize + (k % 3) as isize - 1;
            let j2 = (v / lnx) as isize + (k / 3) as isize - 1;
            if i2 >= 0 && j2 >= 0 && (i2 as usize) < lnx && (j2 as usize) < lny {
                mm[v * m + j2 as usize * lnx + i2 as usize] += la.get(v, k);
            }
        }
    }
    let mut tr = 0.0;
    for i in 0..m {
        tr += mm[i * m + i];
    }
    for i in 0..m {
        mm[i * m + i] += (1e-9 * tr) / m as f64 + 1e-300;
    }
    for i in 0..m {
        for j in 0..=i {
            let mut s = mm[i * m + j];
            for k in 0..j {
                s -= mm[i * m + k] * mm[j * m + k];
            }
            // Math.max(s, 1e-300), NaN kept
            mm[i * m + j] = if i == j { (if s.is_nan() || s > 1e-300 { s } else { 1e-300 }).sqrt() } else { s / mm[j * m + j] };
        }
    }
    Multigrid { lev, m, chol: mm, yv: vec![0.0; m] }
}

impl Multigrid {
    fn dense(m: usize, chol: &[f64], yv: &mut [f64], b: &[f64], x: &mut [f64]) {
        for a in 0..2 {
            for i in 0..m {
                let mut s = b[2 * i + a];
                for k in 0..i {
                    s -= chol[i * m + k] * yv[k];
                }
                yv[i] = s / chol[i * m + i];
            }
            for i in (0..m).rev() {
                let mut s = yv[i];
                for k in i + 1..m {
                    s -= chol[k * m + i] * x[2 * k + a];
                }
                x[2 * i + a] = s / chol[i * m + i];
            }
        }
    }

    /// one V-cycle as a preconditioner, from r (level 0's right side; its matrix is `a0`): z is level 0's x after it
    fn apply(&mut self, a0: &Mat, r: &[f64]) {
        let last = self.lev.len() - 1;
        for d in 0..last {
            let (fine, coarse) = self.lev.split_at_mut(d + 1);
            let Level { mat, corner, x, b, r: fr, pw } = &mut fine[d];
            let (m, rhs): (&Mat, &[f64]) = if d == 0 { (a0, r) } else { (mat, b) };
            down(m, corner, x, rhs, fr, pw, &mut coarse[0].b);
        }
        {
            let l = &mut self.lev[last];
            let rhs: &[f64] = if last == 0 { r } else { &l.b };
            Multigrid::dense(self.m, &self.chol, &mut self.yv, rhs, &mut l.x);
        }
        for d in (0..last).rev() {
            let (fine, coarse) = self.lev.split_at_mut(d + 1);
            let Level { mat, corner, x, b, pw, .. } = &mut fine[d];
            let (m, rhs): (&Mat, &[f64]) = if d == 0 { (a0, r) } else { (mat, b) };
            up(m, corner, x, rhs, pw, &coarse[0].x);
        }
    }
}

fn dot(u: &[f64], w: &[f64]) -> F2 {
    assert_eq!(u.len(), w.len());
    let mut s = F2::splat(0.0);
    unsafe {
        for i in (0..u.len()).step_by(2) {
            s = s.add(F2::ld(u.as_ptr(), i).mul(F2::ld(w.as_ptr(), i)));
        }
    }
    s
}

/// The carry's solver: the matrix (the caller's carryMatrix, copied in), its multigrid, and conjugate gradients on it.
#[wasm_bindgen]
pub struct CarrySolver {
    /// the matrix, its corner flags as it first was (a station's reweighting only scales its stencil)
    a: Mat,
    mg: Option<Multigrid>,
}

#[wasm_bindgen]
impl CarrySolver {
    #[wasm_bindgen(constructor)]
    pub fn new(nx: u32, ny: u32, a: Vec<f64>) -> CarrySolver {
        CarrySolver { a: Mat::new(nx as usize, ny as usize, &a), mg: None }
    }

    /// the matrix's entries `at` (node times 9 plus slot) set to `v` (the caller changed its own copy)
    pub fn set(&mut self, at: &[u32], v: &[f64]) {
        for (&i, &x) in at.iter().zip(v) {
            self.a.set(i as usize, x);
        }
    }

    /// the multigrid made (again) from the matrix as it is now
    pub fn multigrid(&mut self) {
        self.mg = None;
        self.mg = Some(multigrid(&self.a));
    }

    /// conjugate gradients on A x = b from x for each axis on (`on_x`, `on_y`), preconditioned by the multigrid, to
    /// residual over right side under `tol` or `max_it` iterations; each axis's iterations. An axis settled or off is
    /// left as it is while the other goes on.
    pub fn solve(&mut self, b: &[f64], x: &mut [f64], tol: f64, max_it: u32, on_x: bool, on_y: bool) -> Vec<u32> {
        let m = x.len();
        assert!(m == 2 * self.a.nx * self.a.ny && b.len() == m);
        if self.mg.is_none() {
            self.multigrid();
        }
        let a = &self.a;
        let mg = self.mg.as_mut().unwrap();
        let (mut r, mut p, mut q) = (vec![0.0f64; m], vec![0.0f64; m], vec![0.0f64; m]);
        let mut it = [0u32, 0u32];
        let mut go = [on_x && max_it > 0, on_y && max_it > 0];
        mul9::<false>(a, &a.corner, x, &mut q);
        let mut bn2 = F2::splat(0.0);
        unsafe {
            for i in (0..m).step_by(2) {
                let bv = F2::ld(b.as_ptr(), i);
                bv.sub(F2::ld(q.as_ptr(), i)).st(r.as_mut_ptr(), i);
                bn2 = bn2.add(bv.mul(bv));
            }
        }
        let or1 = |s: f64| if truthy(s) { s } else { 1.0 };
        let bn = [or1(bn2.lo().sqrt()), or1(bn2.hi().sqrt())];
        mg.apply(a, &r);
        p.copy_from_slice(&mg.lev[0].x);
        let rz0 = dot(&r, &mg.lev[0].x);
        let mut rz = [rz0.lo(), rz0.hi()];
        while go[0] || go[1] {
            let pq2 = mul9::<true>(a, &a.corner, &p, &mut q);
            let pq = [pq2.lo(), pq2.hi()];
            // the axes' steps: both lanes at once where both go on, else the one alone (the lanes never mix)
            let mut al = [f64::NAN; 2];
            for ax in 0..2 {
                if !go[ax] {
                    continue;
                }
                if !(pq[ax] > 0.0) {
                    go[ax] = false;
                    continue;
                }
                al[ax] = rz[ax] / pq[ax];
            }
            let mut rn = [0.0f64; 2];
            if go[0] && go[1] {
                let alv = F2::new(al[0], al[1]);
                let mut s = F2::splat(0.0);
                unsafe {
                    let (xp, rp, pp, qp) = (x.as_mut_ptr(), r.as_mut_ptr(), p.as_ptr(), q.as_ptr());
                    for i in (0..m).step_by(2) {
                        F2::ld(xp, i).add(alv.mul(F2::ld(pp, i))).st(xp, i);
                        let rv = F2::ld(rp, i).sub(alv.mul(F2::ld(qp, i)));
                        rv.st(rp, i);
                        s = s.add(rv.mul(rv));
                    }
                }
                rn = [s.lo(), s.hi()];
            } else {
                for ax in 0..2 {
                    if !go[ax] {
                        continue;
                    }
                    let (alx, mut s) = (al[ax], 0.0);
                    for i in (ax..m).step_by(2) {
                        x[i] += alx * p[i];
                        r[i] -= alx * q[i];
                        s += r[i] * r[i];
                    }
                    rn[ax] = s;
                }
            }
            for ax in 0..2 {
                if go[ax] && rn[ax].sqrt() / bn[ax] < tol {
                    it[ax] += 1;
                    go[ax] = false;
                }
            }
            if !go[0] && !go[1] {
                break;
            }
            mg.apply(a, &r);
            let z = &mg.lev[0].x;
            let rz2v = dot(&r, z);
            let rz2 = [rz2v.lo(), rz2v.hi()];
            let mut be = [f64::NAN; 2];
            for ax in 0..2 {
                if go[ax] {
                    be[ax] = rz2[ax] / rz[ax];
                    rz[ax] = rz2[ax];
                }
            }
            if go[0] && go[1] {
                let bev = F2::new(be[0], be[1]);
                unsafe {
                    let (pp, zp) = (p.as_mut_ptr(), z.as_ptr());
                    for i in (0..m).step_by(2) {
                        F2::ld(zp, i).add(bev.mul(F2::ld(pp, i))).st(pp, i);
                    }
                }
            } else {
                for ax in 0..2 {
                    if go[ax] {
                        for i in (ax..m).step_by(2) {
                            p[i] = z[i] + be[ax] * p[i];
                        }
                    }
                }
            }
            for ax in 0..2 {
                if go[ax] {
                    it[ax] += 1;
                    if it[ax] >= max_it {
                        go[ax] = false;
                    }
                }
            }
        }
        vec![it[0], it[1]]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// a lattice of land and water links, with a few stations' corner terms, as carryMatrix makes it
    fn problem(nx: usize, ny: usize) -> (Vec<f64>, Vec<f64>) {
        let n = nx * ny;
        let mut a = vec![0.0; 9 * n];
        let lam = |i: usize, j: usize| if (i * 7 + j * 3) % 11 < 6 { 5.0 } else { 0.004 };
        for j in 0..ny {
            for i in 0..nx {
                let v = j * nx + i;
                if i + 1 < nx {
                    let l = lam(i, j);
                    a[v * 9 + 4] += l;
                    a[(v + 1) * 9 + 4] += l;
                    a[v * 9 + 5] -= l;
                    a[(v + 1) * 9 + 3] -= l;
                }
                if j + 1 < ny {
                    let l = lam(i + 1, j);
                    a[v * 9 + 4] += l;
                    a[(v + nx) * 9 + 4] += l;
                    a[v * 9 + 7] -= l;
                    a[(v + nx) * 9 + 1] -= l;
                }
            }
        }
        let mut b = vec![0.0; 2 * n];
        for s in 0..9 {
            let (i, j) = (3 + (s * 13) % (nx - 6), 3 + (s * 7) % (ny - 6));
            let nodes = [j * nx + i, j * nx + i + 1, (j + 1) * nx + i + 1];
            let w = [0.5, 0.3, 0.2];
            for p in 0..3 {
                for q in 0..3 {
                    let (u, t) = (nodes[p], nodes[q]);
                    let di = (t % nx) as isize - (u % nx) as isize;
                    let dj = (t / nx) as isize - (u / nx) as isize;
                    a[u * 9 + ((dj + 1) * 3 + di + 1) as usize] += w[p] * w[q];
                }
                b[2 * nodes[p]] += w[p] * (s as f64 - 4.0);
                b[2 * nodes[p] + 1] += w[p] * (s as f64 * 0.5 - 1.0);
            }
        }
        (a, b)
    }

    fn residual(nx: usize, ny: usize, a: &[f64], b: &[f64], x: &[f64], ax: usize) -> f64 {
        let (mut r, mut n) = (0.0, 0.0);
        for j in 0..ny {
            for i in 0..nx {
                let v = j * nx + i;
                let mut s = b[2 * v + ax];
                for k in 0..9 {
                    let (ii, jj) = (i as isize + (k % 3) as isize - 1, j as isize + (k / 3) as isize - 1);
                    if ii >= 0 && jj >= 0 && (ii as usize) < nx && (jj as usize) < ny {
                        s -= a[9 * v + k] * x[2 * (jj as usize * nx + ii as usize) + ax];
                    }
                }
                r += s * s;
                n += b[2 * v + ax] * b[2 * v + ax];
            }
        }
        (r / n).sqrt()
    }

    #[test]
    fn carry_solves_both_axes() {
        let (nx, ny) = (47, 38);
        let (a, b) = problem(nx, ny);
        let mut s = CarrySolver::new(nx as u32, ny as u32, a.clone());
        s.multigrid();
        let mut x = vec![0.0; b.len()];
        let it = s.solve(&b, &mut x, 1e-4, 60, true, true);
        assert!(it[0] > 0 && it[0] < 60 && it[1] > 0 && it[1] < 60, "{it:?}");
        assert!(residual(nx, ny, &a, &b, &x, 0) < 1e-4 && residual(nx, ny, &a, &b, &x, 1) < 1e-4);
    }

    #[test]
    fn carry_axes_do_not_mix() {
        // one axis alone gives the same bits as both at once (the JavaScript's own solve relies on it, T-2589)
        let (nx, ny) = (33, 29);
        let (a, b) = problem(nx, ny);
        let mut s = CarrySolver::new(nx as u32, ny as u32, a);
        s.multigrid();
        let mut both = vec![0.0; b.len()];
        s.solve(&b, &mut both, 1e-8, 60, true, true);
        let mut one = vec![0.0; b.len()];
        s.solve(&b, &mut one, 1e-8, 60, true, false);
        s.solve(&b, &mut one, 1e-8, 60, false, true);
        assert!(both.iter().zip(&one).all(|(p, q)| p.to_bits() == q.to_bits()));
    }

    #[test]
    fn carry_reweighted_and_remade() {
        let (nx, ny) = (40, 40);
        let (mut a, b) = problem(nx, ny);
        let mut s = CarrySolver::new(nx as u32, ny as u32, a.clone());
        s.multigrid();
        let mut x = vec![0.0; b.len()];
        s.solve(&b, &mut x, 1e-6, 60, true, true);
        let at = [(5 * nx + 5) * 9 + 4, (5 * nx + 5) * 9 + 5];
        a[at[0]] += 7.0;
        a[at[1]] -= 1.0;
        s.set(&[at[0] as u32, at[1] as u32], &[a[at[0]], a[at[1]]]);
        s.multigrid();
        s.solve(&b, &mut x, 1e-6, 60, true, true);
        assert!(residual(nx, ny, &a, &b, &x, 0) < 1e-6);
    }
}
