// Sparse least squares for the diagram layout (T-2160): a system of weighted linear terms gathered
// row by row, solved by LSQR (Paige and Saunders 1982, as scipy.sparse.linalg.lsqr without damping).

/** Rows of a sparse system A x = b, added one term at a time. */
export class Terms {
  private rowStart: number[] = [0];
  private col: number[] = [];
  private val: number[] = [];
  private rhs: number[] = [];
  constructor(readonly cols: number) {}
  /** one row: sum of coefs[k] * x[idx[k]] = target, all times w */
  add(idx: readonly number[], coefs: readonly number[], target: number, w: number): void {
    for (let k = 0; k < idx.length; k++) {
      this.col.push(idx[k]);
      this.val.push(coefs[k] * w);
    }
    this.rhs.push(target * w);
    this.rowStart.push(this.col.length);
  }
  get rows(): number { return this.rhs.length; }
  /** the least squares solution from `x0`, at most `iters` iterations */
  solve(x0: Float64Array, iters: number, atol = 1e-6, btol = 1e-6): Float64Array {
    return lsqr(Int32Array.from(this.rowStart), Int32Array.from(this.col), Float64Array.from(this.val), Float64Array.from(this.rhs), this.cols, x0, iters, atol, btol);
  }
}

function norm(v: Float64Array): number {
  let s = 0;
  for (let i = 0; i < v.length; i++) s += v[i] * v[i];
  return Math.sqrt(s);
}

/** A in CSR (rows from `start`), b, the start x0; returns x */
export function lsqr(start: Int32Array, col: Int32Array, val: Float64Array, b: Float64Array, n: number, x0: Float64Array, iters: number, atol: number, btol: number): Float64Array {
  const m = b.length;
  const Av = (v: Float64Array, out: Float64Array) => {
    for (let r = 0; r < m; r++) {
      let s = 0;
      for (let k = start[r]; k < start[r + 1]; k++) s += val[k] * v[col[k]];
      out[r] = s;
    }
  };
  const ATu = (u: Float64Array, out: Float64Array) => {
    out.fill(0);
    for (let r = 0; r < m; r++) {
      const ur = u[r];
      if (ur === 0) continue;
      for (let k = start[r]; k < start[r + 1]; k++) out[col[k]] += val[k] * ur;
    }
  };
  const x = Float64Array.from(x0);
  const bnorm = norm(b);
  const u = new Float64Array(m), v = new Float64Array(n), w = new Float64Array(n), tm = new Float64Array(m), tn = new Float64Array(n);
  Av(x, tm);
  for (let r = 0; r < m; r++) u[r] = b[r] - tm[r];
  let beta = norm(u);
  if (beta === 0) return x;
  for (let r = 0; r < m; r++) u[r] /= beta;
  ATu(u, v);
  let alpha = norm(v);
  if (alpha === 0) return x;
  for (let i = 0; i < n; i++) v[i] /= alpha, (w[i] = v[i]);
  // the iterations' A^T u gathered by column (the rows of each column in order, as the scatter above adds them); each
  // step's vector updates, norms and scalings in the loops that read them, in the same order
  const cs = new Int32Array(n + 1), at = new Int32Array(start[m]), av = new Float64Array(start[m]);
  for (let k = 0; k < start[m]; k++) cs[col[k] + 1]++;
  for (let i = 0; i < n; i++) cs[i + 1] += cs[i];
  {
    const fill = cs.slice(0, n);
    for (let r = 0; r < m; r++) for (let k = start[r]; k < start[r + 1]; k++) { const p = fill[col[k]]++; (at[p] = r), (av[p] = val[k]); }
  }
  let rhobar = alpha, phibar = beta, anorm = 0, xnorm = norm(x);
  for (let it = 0; it < iters; it++) {
    let ss = 0;
    for (let r = 0; r < m; r++) {
      let s = 0;
      for (let k = start[r]; k < start[r + 1]; k++) s += val[k] * v[col[k]];
      const ur = s - alpha * u[r];
      u[r] = ur;
      ss += ur * ur;
    }
    beta = Math.sqrt(ss);
    if (beta > 0) for (let r = 0; r < m; r++) u[r] /= beta;
    anorm = Math.sqrt(anorm * anorm + alpha * alpha + beta * beta);
    ss = 0;
    for (let i = 0; i < n; i++) {
      let s = 0;
      for (let k = cs[i]; k < cs[i + 1]; k++) s += av[k] * u[at[k]];
      const vi = s - beta * v[i];
      v[i] = vi;
      ss += vi * vi;
    }
    alpha = Math.sqrt(ss);
    const rho = Math.sqrt(rhobar * rhobar + beta * beta), c = rhobar / rho, s = beta / rho;
    const theta = s * alpha;
    rhobar = -c * alpha;
    const phi = c * phibar;
    phibar = s * phibar;
    const t1 = phi / rho, t2 = -theta / rho;
    let xs = 0;
    for (let i = 0; i < n; i++) {
      if (alpha > 0) v[i] /= alpha;
      x[i] += t1 * w[i];
      w[i] = v[i] + t2 * w[i];
      xs += x[i] * x[i];
    }
    xnorm = Math.sqrt(xs);
    const rnorm = phibar, arnorm = alpha * Math.abs(c) * phibar;
    if (rnorm <= btol * bnorm + atol * anorm * xnorm) break;
    if (anorm * rnorm > 0 && arnorm / (anorm * rnorm) <= atol) break;
  }
  return x;
}
