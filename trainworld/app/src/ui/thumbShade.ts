// A saved game's picture is drawn in the theme the game was saved in (workers/clock.worker.ts,
// `slotSummary`); the saves list shows it in the theme now set (T-1852). The picture is recoloured
// pixel by pixel: the land, the rule between panels and the grey track are known colours of each
// theme and are swapped; the lines' colours are the same in both themes and stay.

import { useEffect, useState } from "preact/hooks";
import { SHADES } from "../game/shadeTable";
import type { Shade } from "../game/ui";

type RGB = [number, number, number];
interface Pal { land: RGB; casing: RGB; track: RGB }

const rgb = (hex: string): RGB => [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16)) as RGB;
const setOf = (s: Shade): Pal => ({ land: rgb(SHADES[s].land), casing: rgb(SHADES[s].casing), track: rgb(SHADES[s].thumbTrack) });

/** Where `p` sits on the line from `a` to `b`, 0 to 1 (a little over is allowed), or -1 when it is off the line. */
function along(p: RGB, a: RGB, b: RGB, tol: number): number {
  const v = [b[0] - a[0], b[1] - a[1], b[2] - a[2]], w = [p[0] - a[0], p[1] - a[1], p[2] - a[2]];
  const vv = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
  if (vv < 400) return -1;
  const t = (w[0] * v[0] + w[1] * v[1] + w[2] * v[2]) / vv;
  if (t < -0.02 || t > 1.05) return -1;
  const e = Math.hypot(w[0] - t * v[0], w[1] - t * v[1], w[2] - t * v[2]);
  return e <= tol ? Math.min(1, Math.max(0, t)) : -1;
}

/** The pixels of a `w` wide picture drawn on `from`'s colours, recoloured to `to`'s. */
export function recolour(px: Uint8ClampedArray, w: number, from: Pal, to: Pal) {
  const n = px.length / 4, h = n / w;
  const dist = (p: RGB, q: RGB) => Math.max(Math.abs(p[0] - q[0]), Math.abs(p[1] - q[1]), Math.abs(p[2] - q[2]));
  const mix = (a: RGB, b: RGB, t: number): RGB => [0, 1, 2].map((k) => a[k] + t * (b[k] - a[k])) as RGB;
  // what each pixel is: land, or land mixed with the track grey or the casing (all swapped to the
  // new theme's), or anything else: a line's colour, which stays
  const out: (RGB | null)[] = new Array(n);
  const d: number[] = new Array(n);
  const at = (i: number): RGB => [px[i * 4], px[i * 4 + 1], px[i * 4 + 2]];
  for (let i = 0; i < n; i++) {
    const p = at(i);
    d[i] = dist(p, from.land);
    const t = d[i] <= 2 ? 0 : along(p, from.land, from.track, 7);
    if (d[i] <= 2) out[i] = to.land;
    else if (t >= 0) out[i] = mix(to.land, to.track, t);
    else {
      const u = along(p, from.land, from.casing, 3);
      out[i] = u >= 0 ? mix(to.land, to.casing, u) : dist(p, from.casing) <= 3 ? to.casing : null;
    }
  }
  const feat = out.map((o) => !o);
  // a line's edge is its colour blended into the land: the share of the colour is read against the
  // most distant line pixel beside it, and the rest is the new land
  for (let i = 0; i < n; i++) {
    if (!feat[i]) continue;
    const p = at(i), x = i % w, y = (i / w) | 0;
    let best = i;
    for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
      const X = x + dx, Y = y + dy;
      if (X < 0 || Y < 0 || X >= w || Y >= h) continue;
      const j = Y * w + X;
      if (feat[j] && d[j] > d[best]) best = j;
    }
    const c = at(best);
    const v = [c[0] - from.land[0], c[1] - from.land[1], c[2] - from.land[2]], q = [p[0] - from.land[0], p[1] - from.land[1], p[2] - from.land[2]];
    const a = Math.min(1, Math.max(0, (q[0] * v[0] + q[1] * v[1] + q[2] * v[2]) / (v[0] * v[0] + v[1] * v[1] + v[2] * v[2] || 1)));
    out[i] = a >= 0.98 ? p : mix(to.land, c, a);
  }
  for (let i = 0; i < n; i++) { const o = out[i]!; px[i * 4] = o[0]; px[i * 4 + 1] = o[1]; px[i * 4 + 2] = o[2]; }
}

const done = new Map<string, string>();

async function convert(src: string, from: Shade, to: Shade): Promise<string> {
  const img = new Image();
  img.src = src;
  await img.decode();
  const c = document.createElement("canvas");
  c.width = img.naturalWidth; c.height = img.naturalHeight;
  const g = c.getContext("2d", { willReadFrequently: true })!;
  g.drawImage(img, 0, 0);
  const data = g.getImageData(0, 0, c.width, c.height);
  recolour(data.data, c.width, setOf(from), setOf(to));
  g.putImageData(data, 0, 0);
  return c.toDataURL("image/png");
}

/** `src`, a picture saved in `from`, as it looks in `to`; undefined for the moment it takes to recolour. */
export function useThumbIn(src: string, from: Shade, to: Shade): string | undefined {
  const key = `${from}>${to}|${src.length}|${src.slice(-64)}`;
  const [out, setOut] = useState<{ key: string; url: string } | null>(null);
  useEffect(() => {
    if (from === to || done.has(key)) return;
    let live = true;
    convert(src, from, to).then((url) => { done.set(key, url); if (live) setOut({ key, url }); }, () => { if (live) setOut({ key, url: src }); });
    return () => { live = false; };
  }, [key]);
  if (from === to) return src;
  return done.get(key) ?? (out?.key === key ? out.url : undefined);
}
