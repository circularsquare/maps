// The automatic line colour pick, as functions of plain data (no world, so a Node check can bundle them).

import { colourDistanceFrom, LINE_PALETTE, rgb } from "./palette";

/** T-1894: the palette's first colours (red, blue, green, orange, purple, yellow) are the plain
 * ones: a city's first lines take only these, later ones favour them by this factor on distance. */
const BASIC = 6, BASIC_FIRST = 4, BASIC_BIAS = 1.3;

/** The palette colour farthest by `dist` from every colour in `used` (none used: any), ties at random. */
export function pickColour(used: readonly string[], cityLines = Infinity, dist: (a: string, b: string) => number = (a, b) => colourDistanceFrom(a)(b), rnd: () => number = Math.random): string {
  const first = cityLines < BASIC_FIRST;
  const score = LINE_PALETTE.map((c, i) => (first && i >= BASIC ? -Infinity : (used.length ? used.reduce((m, u) => Math.min(m, dist(u, c)), Infinity) : 1) * (i < BASIC ? BASIC_BIAS : 1)));
  const best = Math.max(...score);
  const pool = LINE_PALETTE.filter((_, i) => score[i] >= best - 1e-6);
  return pool[Math.floor(rnd() * pool.length)];
}

// CIEDE2000, the measure the palette's spacing is stated in (every palette pair is at least 19 apart).
const lin = (v: number) => { const c = v / 255; return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4; };
function lab(hex: string): [number, number, number] {
  const [r, g, b] = rgb(hex).map(lin);
  const f = (t: number) => (t > 216 / 24389 ? Math.cbrt(t) : ((24389 / 27) * t + 16) / 116);
  const fx = f((0.4124564 * r + 0.3575761 * g + 0.1804375 * b) / 0.95047), fy = f(0.2126729 * r + 0.7151522 * g + 0.072175 * b), fz = f((0.0193339 * r + 0.119192 * g + 0.9503041 * b) / 1.08883);
  return [116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)];
}
const rad = (d: number) => (d * Math.PI) / 180;
export function deltaE(a: string, b: string): number {
  const [L1, a1, b1] = lab(a), [L2, a2, b2] = lab(b);
  const Cb = (Math.hypot(a1, b1) + Math.hypot(a2, b2)) / 2;
  const G = 0.5 * (1 - Math.sqrt(Cb ** 7 / (Cb ** 7 + 25 ** 7)));
  const ap1 = (1 + G) * a1, ap2 = (1 + G) * a2, Cp1 = Math.hypot(ap1, b1), Cp2 = Math.hypot(ap2, b2);
  const hue = (y: number, x: number) => (x === 0 && y === 0 ? 0 : ((Math.atan2(y, x) * 180) / Math.PI + 360) % 360);
  const h1 = hue(b1, ap1), h2 = hue(b2, ap2);
  let dh = 0, hb = h1 + h2;
  if (Cp1 * Cp2 !== 0) {
    dh = h2 - h1;
    if (dh > 180) dh -= 360; else if (dh < -180) dh += 360;
    hb = Math.abs(h1 - h2) <= 180 ? (h1 + h2) / 2 : h1 + h2 < 360 ? (h1 + h2 + 360) / 2 : (h1 + h2 - 360) / 2;
  }
  const dL = L2 - L1, dC = Cp2 - Cp1, dH = 2 * Math.sqrt(Cp1 * Cp2) * Math.sin(rad(dh / 2));
  const Lb = (L1 + L2) / 2, Cpb = (Cp1 + Cp2) / 2;
  const T = 1 - 0.17 * Math.cos(rad(hb - 30)) + 0.24 * Math.cos(rad(2 * hb)) + 0.32 * Math.cos(rad(3 * hb + 6)) - 0.2 * Math.cos(rad(4 * hb - 63));
  const Rc = 2 * Math.sqrt(Cpb ** 7 / (Cpb ** 7 + 25 ** 7));
  const Sl = 1 + (0.015 * (Lb - 50) ** 2) / Math.sqrt(20 + (Lb - 50) ** 2), Sc = 1 + 0.045 * Cpb, Sh = 1 + 0.015 * Cpb * T;
  const Rt = -Math.sin(rad(2 * 30 * Math.exp(-(((hb - 275) / 25) ** 2)))) * Rc;
  return Math.sqrt((dL / Sl) ** 2 + (dC / Sc) ** 2 + (dH / Sh) ** 2 + Rt * (dC / Sc) * (dH / Sh));
}

/** T-2291: two colours closer than this (CIEDE2000, the palette's own spacing) read as one line. */
export const APART = 19;
/** T-2291: a stop of another line this near (m) one of the line's own puts that line beside it. */
export const BESIDE_M = 300;

export interface PlacedLine { id: string; colour: string; stops: readonly string[] }

/** T-2291: the colours of the lines beside `self`: those sharing a station or a track edge with it, or
 * with a stop within `BESIDE_M` of one of its stops. `edgeLines` lists, per track edge, the lines on it;
 * `at` a station's [lng, lat]. */
export function besideColours(self: PlacedLine, lines: readonly PlacedLine[], edgeLines: readonly (readonly string[])[], at: (station: string) => [number, number] | null): string[] {
  const near = new Set<string>();
  for (const e of edgeLines) if (e.includes(self.id)) for (const l of e) if (l !== self.id) near.add(l);
  const mine = new Set(self.stops);
  const pts = self.stops.map(at).filter((p): p is [number, number] => !!p);
  if (pts.length) {
    const kx = 111_320 * Math.cos((pts[0][1] * Math.PI) / 180);
    for (const l of lines) {
      if (l.id === self.id || near.has(l.id)) continue;
      if (l.stops.some((s) => mine.has(s))) { near.add(l.id); continue; }
      for (const s of l.stops) {
        const p = at(s);
        if (p && pts.some((q) => Math.hypot((p[0] - q[0]) * kx, (p[1] - q[1]) * 111_320) <= BESIDE_M)) { near.add(l.id); break; }
      }
    }
  }
  return lines.filter((l) => near.has(l.id)).map((l) => l.colour);
}

/** T-2291: the colour a new line should have among those beside it: `current` when it is at least `APART`
 * from all of them (or nothing is beside), else the palette colour farthest from them, unless none is
 * farther than `current` is. */
export function settledColour(current: string, beside: readonly string[], cityLines = Infinity, rnd: () => number = Math.random): string {
  if (!beside.length) return current;
  const gap = (c: string) => beside.reduce((m, u) => Math.min(m, deltaE(u, c)), Infinity);
  if (gap(current) >= APART) return current;
  const pick = pickColour(beside, cityLines, deltaE, rnd);
  return gap(pick) > gap(current) + 1e-6 ? pick : current;
}
