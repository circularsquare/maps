// Rings as flat lon, lat pairs (a Float64Array): the tests the real-network snap and its router share.
// No DOM, so a worker imports it too.

/** A ring's box: west, south, east, north. */
export function ringBox(r: ArrayLike<number>): [number, number, number, number] {
  let w = 180, s = 90, e = -180, n = -90;
  for (let k = 0; k < r.length; k += 2) (w = Math.min(w, r[k])), (e = Math.max(e, r[k])), (s = Math.min(s, r[k + 1])), (n = Math.max(n, r[k + 1]));
  return [w, s, e, n];
}

/** Whether (x, y) is inside the ring (even-odd). */
export function inRing(r: ArrayLike<number>, x: number, y: number): boolean {
  let inside = false;
  for (let i = 0, j = r.length - 2; i < r.length; j = i, i += 2) {
    const yi = r[i + 1], yj = r[j + 1];
    if (yi > y !== yj > y && x < ((r[j] - r[i]) * (y - yi)) / (yj - yi) + r[i]) inside = !inside;
  }
  return inside;
}

/** Whether an edge of ring r passes within d degrees of (x, y), longitude scaled to ground. */
export function nearRing(r: ArrayLike<number>, x: number, y: number, d: number): boolean {
  const k = Math.cos((y * Math.PI) / 180), d2 = d * d;
  for (let i = 0, j = r.length - 2; i < r.length; j = i, i += 2) {
    const ax = (r[j] - x) * k, ay = r[j + 1] - y, bx = (r[i] - x) * k, by = r[i + 1] - y;
    const dx = bx - ax, dy = by - ay, l2 = dx * dx + dy * dy;
    const t = l2 ? Math.max(0, Math.min(1, -(ax * dx + ay * dy) / l2)) : 0;
    if ((ax + t * dx) ** 2 + (ay + t * dy) ** 2 < d2) return true;
  }
  return false;
}
