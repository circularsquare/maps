// T-2887: a schematic as the frozen drawing format v1 (shonei-server's docs/trainworld-api.md, "The drawing"), and back
// into what `paint` and `paintNames` draw. Layout units are rescaled so a pitch is exactly 256; CSS px quantities are
// sixteenths of a px. No water, borders or city names: the format has none.

import type { Drawing } from "./draw";
import { levelAt, type LevelNames, type NamePlan } from "./names";

export interface SharedDrawing {
  v: 1; pitch: number; stroke: number;
  lines: { c: string; f?: 1; d: number[] }[];
  names: string[]; small: number[]; pills: number[]; passages: number[];
  fontPx: number; width: number[]; width2: number[]; levels: number[]; faint: number;
  at: { node: number[]; x: number[]; y: number[]; dir: number[]; slide: number[]; lift: number[]; wrap: number[] }[];
  box?: [number, number, number, number];
}

const PITCH = 256, PX = 16, MAX_PX = 1 << 16, MAX_COLOURS = 256, NAME_CHARS = 64;
/** the server's limits, a little under them */
const MAX_GZIP = 500 * 1024, MAX_RAW = 4 * 1024 * 1024 - 1024;
/** Which name levels go up, as a pitch in CSS px, most first (at most 8); a big network drops down the list until it fits. A
 * level's names stay clear at every zoom above it, so leaving levels out only means fewer names at some zooms. */
const LEVEL_TIERS = [[1.76, 2.95, 4.2, 5.9, 8.4, 11.8, 16.7, 28.1], [2.95, 4.2, 7.1, 11.8, 16.7, 28.1], [2.95, 4.2, 8.4, 16.7], [4.2, 11.8], []];

const hex6 = (c: string): string => {
  const s = c.trim().toLowerCase();
  if (/^#[0-9a-f]{6}$/.test(s)) return s;
  if (/^#[0-9a-f]{3}$/.test(s)) return "#" + [...s.slice(1)].map((x) => x + x).join("");
  return "#808080";
};
const rgbOf = (c: string) => [1, 3, 5].map((i) => parseInt(c.slice(i, i + 2), 16));

/** per line its #rrggbb; past MAX_COLOURS distinct, the rarest take the nearest kept colour */
function colours(raw: string[]): string[] {
  const counts = new Map<string, number>();
  const out = raw.map((c) => { const h = hex6(c); counts.set(h, (counts.get(h) ?? 0) + 1); return h; });
  if (counts.size <= MAX_COLOURS) return out;
  const kept = [...counts].sort((a, b) => b[1] - a[1]).slice(0, MAX_COLOURS).map(([c]) => c), set = new Set(kept), near = new Map<string, string>();
  return out.map((c) => {
    if (set.has(c)) return c;
    let best = near.get(c);
    if (!best) {
      const [r, g, b] = rgbOf(c);
      let bd = Infinity;
      for (const k of kept) {
        const [kr, kg, kb] = rgbOf(k), dd = (r - kr) ** 2 + (g - kg) ** 2 + (b - kb) ** 2;
        if (dd < bd) (bd = dd), (best = k);
      }
      near.set(c, best!);
    }
    return best!;
  });
}

/** the drawing and names in the format, with the name levels of `LEVEL_TIERS[tier]` */
export function encodeShared(d: Drawing, plan: NamePlan, tier: number): SharedDrawing {
  const k = PITCH / d.pitch, u = (v: number) => Math.round(v * k), px = (v: number) => Math.min(MAX_PX, Math.round(v * PX));
  const cs = colours(d.lines.map((l) => l.colour));
  const lines = d.lines.map((l, i): SharedDrawing["lines"][number] => {
    const c = l.cmds, p: number[] = [];
    for (let j = 0; j < c.length; ) {
      const n = c[j] === 2 ? 5 : 3;
      p.push(c[j]);
      for (let q = 1; q < n; q++) p.push(u(c[j + q]));
      j += n;
    }
    return l.fast ? { c: cs[i], f: 1, d: p } : { c: cs[i], d: p };
  });
  const pills: number[] = [], passages: number[] = [];
  d.pills.forEach((p, i) => pills.push(u(p[0]), u(p[1]), u(p[2]), u(p[3]), u(p[4]), d.pillOf[i]));
  d.passages.forEach((p, i) => passages.push(u(p[0]), u(p[1]), u(p[2]), u(p[3]), d.passageOf[i]));
  const picks = [...new Set(LEVEL_TIERS[tier].map((target) => levelAt(plan, target)).filter((li) => li >= 0))].sort((a, b) => a - b);
  return {
    v: 1, pitch: PITCH, stroke: u(d.stroke), lines,
    names: plan.text.map((s) => [...s].slice(0, NAME_CHARS).join("")),
    small: Array.from(d.small), pills, passages,
    fontPx: Math.min(48 * PX, Math.max(6 * PX, Math.round(plan.fontPx * PX))),
    width: Array.from(plan.width, px), width2: Array.from(plan.width2, px),
    levels: picks.map((li) => Math.max(1, px(plan.levels[li]))),
    faint: picks.filter((li) => li < plan.faint).length,
    at: picks.map((li) => {
      const L = plan.at[li];
      return { node: Array.from(L.node), x: Array.from(L.x, u), y: Array.from(L.y, u), dir: Array.from(L.dir), slide: Array.from(L.slide), lift: Array.from(L.lift, u), wrap: Array.from(L.wrap) };
    }),
  };
}

async function gzipJson(v: unknown): Promise<{ blob: Blob; raw: number }> {
  const raw = new TextEncoder().encode(JSON.stringify(v));
  const stream = new Blob([raw]).stream().pipeThrough(new CompressionStream("gzip"));
  return { blob: await new Response(stream).blob(), raw: raw.length };
}

/** the upload's gzipped JSON, with as many name levels as fit; null when not even the lines alone do */
export async function packShared(d: Drawing, plan: NamePlan): Promise<Blob | null> {
  for (let tier = 0; tier < LEVEL_TIERS.length; tier++) {
    const { blob, raw } = await gzipJson(encodeShared(d, plan, tier));
    if (blob.size <= MAX_GZIP && raw <= MAX_RAW) return blob;
  }
  return null;
}

export interface DecodedShare { d: Drawing; names: NamePlan }

/** what the painters draw, from a drawing as the server sent it */
export function decodeShared(j: SharedDrawing): DecodedShare {
  const pills: Drawing["pills"] = [], pillOf: number[] = [], passages: Drawing["passages"] = [], passageOf: number[] = [];
  for (let i = 0; i + 5 < j.pills.length; i += 6) pills.push([j.pills[i], j.pills[i + 1], j.pills[i + 2], j.pills[i + 3], j.pills[i + 4]]), pillOf.push(j.pills[i + 5]);
  for (let i = 0; i + 4 < j.passages.length; i += 5) passages.push([j.passages[i], j.passages[i + 1], j.passages[i + 2], j.passages[i + 3]]), passageOf.push(j.passages[i + 4]);
  const d: Drawing = {
    pitch: j.pitch, stroke: j.stroke, lineOf: [], ribbons: [], chainAt: [], hooks: [],
    lines: j.lines.map((l) => ({ colour: l.c, fast: l.f === 1, cmds: Float64Array.from(l.d) })),
    pills, pillOf, passages, passageOf,
    small: Uint8Array.from(j.small),
    box: j.box ?? [0, 0, 0, 0],
  };
  const at: LevelNames[] = j.at.map((L) => ({ node: Int32Array.from(L.node), x: Float64Array.from(L.x), y: Float64Array.from(L.y), dir: Uint8Array.from(L.dir), slide: Int8Array.from(L.slide), lift: Float64Array.from(L.lift), wrap: Uint8Array.from(L.wrap) }));
  const names: NamePlan = {
    levels: j.levels.map((v) => v / PX), at, faint: j.faint, text: j.names,
    width: Float32Array.from(j.width, (v) => v / PX), width2: Float32Array.from(j.width2, (v) => v / PX),
    fontPx: j.fontPx / PX, ms: 0,
  };
  return { d, names };
}

/** px per layout unit that shows a whole drawing in `w` by `h` CSS px, and its middle */
export function fitView(d: Drawing, w: number, h: number, marginPx = 10): { cx: number; cy: number; scale: number } {
  const [x0, y0, x1, y1] = d.box;
  const sx = Math.max(1, w - 2 * marginPx) / Math.max(1, x1 - x0), sy = Math.max(1, h - 2 * marginPx) / Math.max(1, y1 - y0);
  return { cx: (x0 + x1) / 2, cy: (y0 + y1) / 2, scale: Math.min(sx, sy) };
}
