// The diagram's maths beyond + - * / and sqrt, from those alone (T-2255). Engines differ in the last bits
// of Math.atan2, Math.pow and the like (Node 22 and Chrome 156 do), and the layout grows such differences
// into stations hops apart; with these the layout of a graph is the same in every engine, so the Node
// checks see the player's drawing.

const PI = Math.PI, HALF_PI = PI / 2, SIXTH_PI = PI / 6;
const SQRT3 = Math.sqrt(3), TAN_PI_12 = 2 - SQRT3;
const LN2 = 0.6931471805599453;
const S = Math.SQRT1_2;

/** the eight octilinear unit directions, k * 45 degrees from east, exact */
export const DIRS8: readonly (readonly [number, number])[] = [[1, 0], [S, S], [0, 1], [-S, S], [-1, 0], [-S, -S], [0, -1], [S, -S]];

/** the length of (x, y): Math.hypot's last bit is the engine's own */
export const hyp = (x: number, y: number): number => Math.sqrt(x * x + y * y);

/** atan of t in [0, 1] */
function atan01(t: number): number {
  let off = 0;
  if (t > TAN_PI_12) {
    t = (t * SQRT3 - 1) / (t + SQRT3);
    off = SIXTH_PI;
  }
  // |t| <= tan(15 degrees): atan t = t - t^3 (1/3 - t^2/5 + ...), the series to t^31 within 1e-17
  const t2 = t * t;
  let s = 0;
  for (let k = 31; k >= 3; k -= 2) s = 1 / k - t2 * s;
  return off + t - t * t2 * s;
}

/** atan2(y, x), as Math.atan2 to within a few ulps */
export function atan2(y: number, x: number): number {
  if (y !== y || x !== x) return NaN;
  const ax = Math.abs(x), ay = Math.abs(y);
  if (ax === 0 && ay === 0) return x < 0 || Object.is(x, -0) ? (y < 0 || Object.is(y, -0) ? -PI : PI) : y;
  let a = ay <= ax ? atan01(ay / ax) : HALF_PI - atan01(ax / ay);
  if (x < 0) a = PI - a;
  return y < 0 || Object.is(y, -0) ? -a : a;
}

const F64 = new Float64Array(1), U32 = new Uint32Array(F64.buffer);
// little-endian on every engine the game runs in: the high word is the second
const HI = new Uint8Array(new Uint16Array([1]).buffer)[0] === 1 ? 1 : 0;

/** natural log of x > 0 */
function log(x: number): number {
  F64[0] = x;
  let e = ((U32[HI] >>> 20) & 0x7ff) - 1023;
  U32[HI] = (U32[HI] & 0x800fffff) | 0x3ff00000;
  let m = F64[0];
  if (m > S * 2) (m /= 2), e++;
  // log m = 2 atanh s, s = (m - 1) / (m + 1), |s| < 0.172: the series to s^29 is within 1e-18
  const s = (m - 1) / (m + 1), s2 = s * s;
  let p = 0;
  for (let k = 29; k >= 3; k -= 2) p = 1 / k + s2 * p;
  return e * LN2 + 2 * (s + s * s2 * p);
}

/** e to the x */
function exp(x: number): number {
  const k = Math.round(x / LN2), r = (x - k * LN2) / 16;
  // |r| < 0.022: Taylor to r^10 is within 1e-19; then squared four times
  let p = 1;
  for (let i = 10; i >= 1; i--) p = 1 + (r * p) / i;
  p *= p, p *= p, p *= p, p *= p;
  // 2 ^ k from its bits (|k| well inside the normal range here)
  U32[HI] = (k + 1023) << 20;
  U32[1 - HI] = 0;
  return p * F64[0];
}

/** x to the power a, x > 0 */
export function pow(x: number, a: number): number {
  return x === 1 || a === 0 ? 1 : exp(a * log(x));
}
