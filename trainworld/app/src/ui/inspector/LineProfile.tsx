// T-2734: the line's cross-section at the foot of its view (docs/ui-inspectors.md "The cross section"):
// the whole line along its length on the ground's real heights, water in blue, the track coloured by
// level, other lines crossing it as dots and its stops; a close-up at one fixed scale follows the pointer.

import type { ComponentChildren } from "preact";
import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "preact/hooks";
import { t as tr, tNumber, tRich } from "../../i18n";
import { distance, levelLabel } from "../../game/format";
import { LEVEL_COLOURS } from "../../game/palette";
import { SHADES } from "../../game/shadeTable";
import { display, shade } from "../../game/ui";
import { linesByNum, stationsByNum, world } from "../../game/world";
import { groundAt, groundVersion, GROUND_MAX_Z, loadUnder, zoomFor } from "../../game/groundHeights";
import type { Level, LineInput } from "../../game/types";
import { useClockQuery } from "../useClockQuery";
import { Chip } from "../widgets";

/** the drawing's height a level, m (Anita, 2026-11-07); the sim's own is 8 m (`LEVEL_H`) */
const PROFILE_LEVEL_M = 5;
const SIM_LEVEL_M = 8;
const STRIP_H = 72, PAD = 7;
/** the close-up: this much line over LENS_W px, LENS_M_PX m of height a pixel, for every line */
const LENS_SPAN_M = 2000, LENS_W = 320, LENS_H = 120, LENS_M_PX = 1;
/** the least height range the overview spreads over its strip, m */
const MIN_RANGE_M = 40;
/** a station or crossing this near the pointer on the overview, px, takes the close-up's centre */
const SNAP_PX = 4;
/** crossings at one height this near each other along the line are one, m */
const MERGE_M = 60;
/** and the same line crossing again this near, at the same height, m: the real saves' parallel tracks
 * weave across each other */
const MERGE_SAME_M = 400;
/** the close-up's words name the crossings this near its middle, m */
const NEAR_M = 250;
/** water is drawn this deep: the tiles give its surface, never its bed, and the track's levels are from
 * the surface, so a tunnel at -1 reads as under the water */
const WATER_M = 4;
const WATER_BLUE = "#2f6fb8";
/** the overview's water band at least this thick, px */
const WATER_MIN_PX = 1.5;

interface Profile {
  s: Float64Array; lng: Float64Array; lat: Float64Array; h: Float64Array; len: number;
  stops: { s: number; node: number }[];
  crosses: { s: number; h: number; lines: number[] }[];
  /** stretches the water masks call water (1) or do not cover (-1), by distance along */
  wet: { a: number; b: number; kind: number }[];
}

function parse(d: Float64Array): Profile | null {
  if (d.length < 4) return null;
  const nP = d[0], nS = d[1], nC = d[2], nW = d[3];
  if (nP < 2) return null;
  const s = new Float64Array(nP), lng = new Float64Array(nP), lat = new Float64Array(nP), h = new Float64Array(nP);
  let k = 4;
  for (let i = 0; i < nP; i++) s[i] = d[k++], lng[i] = d[k++], lat[i] = d[k++], h[i] = d[k++];
  const stops = [];
  for (let i = 0; i < nS; i++, k += 2) stops.push({ s: d[k], node: d[k + 1] });
  const crosses = [];
  for (let i = 0; i < nC; i++) {
    const at = d[k], hh = d[k + 1], n = d[k + 2];
    crosses.push({ s: at, h: hh, lines: Array.from(d.subarray(k + 3, k + 3 + n)) });
    k += 3 + n;
  }
  const wet = [];
  for (let i = 0; i < nW; i++, k += 3) wet.push({ a: d[k], b: d[k + 1], kind: d[k + 2] });
  // a line's several tracks crossing at one place (each its own edge) are one crossing
  crosses.sort((a, b) => a.s - b.s);
  const merged: typeof crosses = [];
  for (const c of crosses) {
    const m = merged.find((o) => Math.abs(o.h - c.h) < SIM_LEVEL_M / 2
      && (Math.abs(o.s - c.s) < MERGE_M || (Math.abs(o.s - c.s) < MERGE_SAME_M && o.lines.some((id) => c.lines.includes(id)))));
    if (!m) merged.push(c);
    else for (const id of c.lines) if (!m.lines.includes(id)) m.lines.push(id);
  }
  return { s, lng, lat, h, len: s[nP - 1], stops, crosses: merged, wet };
}

/** The line at `at` m along: lng, lat, track height over the ground (sim m). */
function pointAt(p: Profile, at: number): [number, number, number] {
  const s = p.s;
  let lo = 0, hi = s.length - 1;
  if (at <= 0) return [p.lng[0], p.lat[0], p.h[0]];
  if (at >= s[hi]) return [p.lng[hi], p.lat[hi], p.h[hi]];
  while (hi - lo > 1) {
    const m = (lo + hi) >> 1;
    if (s[m] <= at) lo = m;
    else hi = m;
  }
  const t = s[hi] > s[lo] ? (at - s[lo]) / (s[hi] - s[lo]) : 0;
  return [p.lng[lo] + t * (p.lng[hi] - p.lng[lo]), p.lat[lo] + t * (p.lat[hi] - p.lat[lo]), p.h[lo] + t * (p.h[hi] - p.h[lo])];
}


const levelOf = (simH: number) => Math.max(-3, Math.min(3, Math.round(simH / SIM_LEVEL_M))) as Level;
const drawnH = (simH: number) => (simH / SIM_LEVEL_M) * PROFILE_LEVEL_M;

/** A height in the player's units, to the metre or foot. */
function heightText(m: number): string {
  return display.value.usUnits ? `${tNumber(Math.round(m * 3.28084))} ft` : `${tNumber(Math.round(m))} m`;
}

/** One place along the line: `top` the ground or the water's surface (the track's levels are from it),
 * `bed` the ground drawn under water (else `top`), undefined while no tile has it. */
interface Sample { s: number; top: number | undefined; bed: number | undefined; water: boolean; simH: number }

/** `n` samples from `a` to `b` m along, the ground at zoom `z`, else at `zBack` while z's tile comes. */
function samples(p: Profile, a: number, b: number, n: number, z: number, zBack = z): Sample[] {
  const out: Sample[] = [];
  for (let i = 0; i < n; i++) {
    const s = a + ((b - a) * i) / Math.max(1, n - 1);
    if (s < 0 || s > p.len) continue;
    out.push(sampleAt(p, s, z, zBack));
  }
  return out;
}

function sampleAt(p: Profile, s: number, z: number, zBack = z): Sample {
  const [lng, lat, h] = pointAt(p, s);
  const ground = (lng: number, lat: number) => groundAt(z, lng, lat) ?? (zBack !== z ? groundAt(zBack, lng, lat) : undefined);
  const raw = ground(lng, lat);
  const k = p.wet.findIndex((w) => s >= w.a && s <= w.b);
  const mask = k < 0 ? 0 : p.wet[k].kind;
  // the sea's tiles are its floor: below 0 with no mask to say is sea; a mask's water has its surface
  const water = mask === 1 || (mask === -1 && raw !== undefined && raw < 0);
  if (raw === undefined) return { s, top: undefined, bed: undefined, water, simH: h };
  if (!water) return { s, top: raw, bed: raw, water, simH: h };
  // a mask's water is flat: its lowest height along the stretch, since the cells at its banks read the land
  const top = mask === 1 ? stretchSurface(p, k, z, ground) ?? Math.max(0, raw) : 0;
  return { s, top, bed: top - WATER_M, water, simH: h };
}

/** samples a stretch of water takes for its surface */
const SURFACE_SAMPLES = 24;
const surfaces = new WeakMap<Profile, Map<string, number>>();
function stretchSurface(p: Profile, k: number, z: number, ground: (lng: number, lat: number) => number | undefined): number | undefined {
  let byKey = surfaces.get(p);
  if (!byKey) surfaces.set(p, (byKey = new Map()));
  const key = `${k}/${z}/${groundVersion.peek()}`;
  const got = byKey.get(key);
  if (got !== undefined) return got;
  const w = p.wet[k];
  let lo = Infinity;
  for (let i = 0; i <= SURFACE_SAMPLES; i++) {
    const [lng, lat] = pointAt(p, w.a + ((w.b - w.a) * i) / SURFACE_SAMPLES);
    const g = ground(lng, lat);
    if (g !== undefined) lo = Math.min(lo, g);
  }
  if (!Number.isFinite(lo)) return undefined;
  const v = Math.max(0, lo);
  byKey.set(key, v);
  return v;
}

/** The track as one polyline a run of one level: [level, points]. */
function trackRuns(ss: Sample[], x: (s: number) => number, y: (m: number) => number): [Level, string][] {
  const runs: [Level, string][] = [];
  let cur: Level | null = null, pts: string[] = [], last = "";
  for (const q of ss) {
    const lv = levelOf(q.simH), pt = `${x(q.s).toFixed(1)},${y((q.top ?? 0) + drawnH(q.simH)).toFixed(1)}`;
    if (lv !== cur) {
      if (cur !== null) runs.push([cur, pts.join(" ")]);
      cur = lv;
      pts = last ? [last] : [];
    }
    pts.push(pt);
    last = pt;
  }
  if (cur !== null) runs.push([cur, pts.join(" ")]);
  return runs;
}

/** The ground's y at a sample: the bed, at least `minPx` under the water's surface. */
const bedY = (q: Sample, y: (m: number) => number, minPx: number) =>
  q.water ? Math.max(y(q.bed ?? 0), y(q.top ?? 0) + minPx) : y(q.bed ?? 0);

/** The ground as a filled shape down to `bottom`, its top edge, and the water as shapes. */
function groundShape(ss: Sample[], x: (s: number) => number, y: (m: number) => number, bottom: number, minPx: number): [string, string, string, string] {
  if (!ss.length) return ["", "", "", ""];
  const top = ss.map((q) => `${x(q.s).toFixed(1)},${bedY(q, y, minPx).toFixed(1)}`);
  // water: each run of wet samples, its surface along and its bed back, half a step past each end
  let water = "", surface = "";
  for (let i = 0; i < ss.length; ) {
    if (!ss[i].water) { i++; continue; }
    let j = i;
    while (j + 1 < ss.length && ss[j + 1].water) j++;
    const half = ss.length > 1 ? (x(ss[Math.min(ss.length - 1, i + 1)].s) - x(ss[i].s)) / 2 : 0;
    const run = ss.slice(i, j + 1);
    const xa = (k: number) => Math.max(x(ss[0].s), Math.min(x(ss[ss.length - 1].s), x(run[k].s) + (k === 0 ? -half : k === run.length - 1 ? half : 0)));
    const surf = run.map((q, k) => `${xa(k).toFixed(1)},${y(q.top ?? 0).toFixed(1)}`);
    const bed = run.map((q, k) => `${xa(k).toFixed(1)},${bedY(q, y, minPx).toFixed(1)}`).reverse();
    water += `M${surf.join("L")}L${bed.join("L")}Z`;
    surface += `M${surf.join("L")}`;
    i = j + 1;
  }
  return [`M${x(ss[0].s).toFixed(1)},${bottom}L${top.join("L")}L${x(ss[ss.length - 1].s).toFixed(1)},${bottom}Z`, `M${top.join("L")}`, water, surface];
}

/** Sea level as a faint rule with its height written at the left, when `y0` is inside 0..h. */
function SeaLevel({ y0, w, h, ink }: { y0: number; w: number; h: number; ink: string }) {
  if (!(y0 >= 0 && y0 <= h)) return null;
  return <>
    <line x1={0} x2={w} y1={y0} y2={y0} stroke={ink} stroke-opacity={0.3} stroke-width={1} stroke-dasharray="3 3" />
    <text x={2} y={y0 > 12 ? y0 - 3 : y0 + 10} font-size="10" fill={ink} fill-opacity={0.6}>{heightText(0)}</text>
  </>;
}

function crossDir(dz: number): "up" | "down" | "level" {
  return dz > PROFILE_LEVEL_M / 2 ? "up" : dz < -PROFILE_LEVEL_M / 2 ? "down" : "level";
}

/** Resolves when the page is idle (or after `ms`), for work nobody is waiting on. */
const whenIdle = (ms: number) => new Promise<void>((r) => (typeof requestIdleCallback === "function" ? requestIdleCallback(() => r(), { timeout: ms }) : setTimeout(r, ms)));

export function LineProfile({ l, version }: { l: LineInput; version: number }) {
  // Anita, 2026-11-07: flavour at the foot of the view, so asked only once the page is idle, and the
  // worker answers it between its other messages
  const [idle, setIdle] = useState(false);
  useEffect(() => {
    let on = true;
    setIdle(false);
    void whenIdle(3000).then(() => on && setIdle(true));
    return () => { on = false; };
  }, [l.num]);
  const prof = useClockQuery(idle ? { what: "lineProfile", line: l.num } : null, [idle, l.num, version], parse);
  const box = useRef<HTMLDivElement>(null);
  const lens = useRef<HTMLDivElement>(null);
  const [w, setW] = useState(0);
  const [at, setAt] = useState<{ s: number; x: number } | null>(null);
  const [tilesIn, setTilesIn] = useState<Profile | null>(null);
  const gv = groundVersion.value;
  const water = SHADES[shade.value].water;
  // the map's water pushed toward blue, and its surface darker, so a thin band reads as water on the panel
  const waterFill = `color-mix(in srgb, ${water} 75%, ${WATER_BLUE})`, waterEdge = `color-mix(in srgb, ${water} 35%, ${WATER_BLUE})`; // i18n-ignore (CSS)

  useLayoutEffect(() => {
    const el = box.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setW(el.clientWidth));
    ro.observe(el);
    setW(el.clientWidth);
    return () => ro.disconnect();
  }, []);

  const midLat = prof ? prof.lat[prof.lat.length >> 1] : 0;
  const zOv = prof ? zoomFor(prof.len / 600, midLat) : GROUND_MAX_Z;
  useEffect(() => {
    if (!prof) return;
    let on = true;
    const ll: number[] = [];
    for (let i = 0; i < prof.s.length; i++) ll.push(prof.lng[i], prof.lat[i]);
    void loadUnder(zOv, ll).then(() => on && setTilesIn(prof));
    return () => { on = false; };
  }, [prof, zOv]);

  // the close-up is in the top layer, over the panel's clipping
  useLayoutEffect(() => {
    const el = lens.current, b = box.current;
    if (!el || !b) return;
    if (!at) {
      if (el.matches(":popover-open")) el.hidePopover();
      return;
    }
    if (!el.matches(":popover-open")) el.showPopover();
    const r = b.getBoundingClientRect(), lw = el.offsetWidth, lh = el.offsetHeight;
    const x = Math.max(4, Math.min(innerWidth - 4 - lw, r.left + at.x - lw / 2));
    const y = r.top - lh - 6 >= 4 ? r.top - lh - 6 : r.bottom + 6;
    el.style.transform = `translate(${Math.round(x)}px, ${Math.round(y)}px)`; // i18n-ignore (CSS)
  });

  const W = Math.max(60, w);
  const ready = !!prof && prof.len >= 1 && tilesIn === prof;
  const ov = useMemo(() => (ready ? samples(prof!, 0, prof!.len, Math.min(1200, W * 2), zOv) : []), [ready, prof, W, zOv, gv]);

  const strip = (inner: ComponentChildren, events = false) => (
    <>
      <div class="h">{tr("lineProfile.heading")}</div>
      <div ref={box} class="line-profile" style={{ position: "relative", height: `${STRIP_H}px`, margin: "4px 0 8px", cursor: events ? "crosshair" : undefined, touchAction: events ? "none" : undefined }}
        onPointerMove={events ? move : undefined} onPointerDown={events ? move : undefined} onPointerLeave={() => setAt(null)}>
        {inner}
      </div>
    </>
  );
  // a quiet empty strip until the line and its ground are in
  if (!ready || !prof) return strip(null);
  const p = prof;

  let lo = Infinity, hi = -Infinity;
  for (const q of ov) {
    const g = q.bed ?? 0, t = q.top ?? 0, tr2 = t + drawnH(q.simH);
    lo = Math.min(lo, g, tr2), hi = Math.max(hi, t, tr2);
  }
  if (!(hi - lo >= MIN_RANGE_M)) {
    const mid = Number.isFinite(hi + lo) ? (hi + lo) / 2 : MIN_RANGE_M / 2;
    lo = mid - MIN_RANGE_M / 2, hi = mid + MIN_RANGE_M / 2;
  }
  const innerH = STRIP_H - 2 * PAD;
  const mPerPxV = (hi - lo) / innerH;
  const xOv = (s: number) => (s / p.len) * W;
  const yOv = (m: number) => PAD + (hi - m) / mPerPxV;
  const [ovFill, ovEdge, ovWater, ovSurface] = groundShape(ov, xOv, yOv, STRIP_H, WATER_MIN_PX);
  const ownAt = (s: number) => pointAt(p, s)[2];
  const topOv = (s: number) => sampleAt(p, s, zOv).top ?? 0;

  const lineOf = linesByNum(world.value!);
  const stationOf = stationsByNum(world.value!);
  const stopName = (node: number) => stationOf.get(node)?.name ?? "";

  function move(e: PointerEvent) {
    const r = box.current!.getBoundingClientRect();
    const px = Math.max(0, Math.min(W, e.clientX - r.left));
    let s = (px / W) * p.len, x = px;
    let best = SNAP_PX;
    for (const m of [...p.stops, ...p.crosses]) {
      const d = Math.abs(xOv(m.s) - px);
      if (d < best) best = d, s = m.s, x = xOv(m.s);
    }
    setAt({ s, x });
  }

  return strip(<>
    <svg width={W} height={STRIP_H} style={{ display: "block", overflow: "visible", width: `${W}px`, height: `${STRIP_H}px`, stroke: "none", strokeWidth: 1 }}>
      <path d={ovFill} fill="var(--text)" fill-opacity={0.1} />
      <path d={ovWater} fill={waterFill} />
      <path d={ovSurface} fill="none" stroke={waterEdge} stroke-width={1} />
      <path d={ovEdge} fill="none" stroke="var(--text)" stroke-opacity={0.35} stroke-width={1} />
      <SeaLevel y0={yOv(0)} w={W} h={STRIP_H} ink="var(--text)" />
      {at && <rect x={xOv(at.s - LENS_SPAN_M / 2)} y={0} width={Math.max(1, (LENS_SPAN_M / p.len) * W)} height={STRIP_H} fill="var(--accent)" fill-opacity={0.15} />}
      {trackRuns(ov, xOv, yOv).map(([lv, pts]) => <polyline points={pts} fill="none" stroke={LEVEL_COLOURS[lv]} stroke-width={2} stroke-linejoin="round" />)}
      {p.crosses.map((c) => <circle cx={xOv(c.s)} cy={yOv(topOv(c.s) + drawnH(c.h))} r={2.5} fill={lineOf.get(c.lines[0])?.colour ?? "var(--text)"} />)}
      {p.stops.map((st) => <circle cx={xOv(st.s)} cy={yOv(topOv(st.s) + drawnH(ownAt(st.s)))} r={3} fill="#fff" stroke="var(--map-ink)" stroke-width={1.5} />)}
    </svg>
    <div ref={lens} popover="manual" class="tool-tag ui-tag" style={{ maxWidth: "none", background: "var(--map-land)" }}>
      {at && <Lens p={p} s={at.s} ownAt={ownAt} lineOf={lineOf} stopName={stopName} zBack={zOv} waterFill={waterFill} waterEdge={waterEdge} />}
    </div>
  </>, true);
}

function Lens({ p, s, ownAt, lineOf, stopName, zBack, waterFill, waterEdge }: { p: Profile; zBack: number; waterFill: string; waterEdge: string; s: number; ownAt: (s: number) => number; lineOf: ReadonlyMap<number, LineInput>; stopName: (n: number) => string }) {
  const a = s - LENS_SPAN_M / 2, b = s + LENS_SPAN_M / 2;
  const ss = samples(p, a, b, LENS_W + 1, GROUND_MAX_Z, zBack);
  const mid = sampleAt(p, s, GROUND_MAX_Z, zBack);
  const centre = (mid.top ?? 0) + drawnH(mid.simH);
  const x = (v: number) => ((v - a) / LENS_SPAN_M) * LENS_W;
  const y = (m: number) => LENS_H / 2 + (centre - m) / LENS_M_PX;
  const [fill, edge, water, surface] = groundShape(ss, x, y, LENS_H, 0);
  const topOf = (v: number) => sampleAt(p, v, GROUND_MAX_Z, zBack).top ?? 0;
  const stops = p.stops.filter((st) => st.s >= a && st.s <= b);
  const crosses = p.crosses.filter((c) => c.s >= a && c.s <= b);
  const lv = levelOf(mid.simH), above = drawnH(mid.simH);
  const from = p.stops.length ? stopName(p.stops[0].node) : "";
  const trackDir = lv > 0 ? (mid.water ? "over" : "up") : lv < 0 ? (mid.water ? "under" : "down") : "level";
  return (
    <div>
      <svg width={LENS_W} height={LENS_H} style={{ display: "block", margin: "3px 0", width: `${LENS_W}px`, height: `${LENS_H}px`, stroke: "none", strokeWidth: 1 }}>
        <defs><clipPath id="lp-clip"><rect width={LENS_W} height={LENS_H} /></clipPath></defs>
        <g clip-path="url(#lp-clip)">
          <path d={fill} fill="currentColor" fill-opacity={0.1} />
          <path d={water} fill={waterFill} />
          <path d={surface} fill="none" stroke={waterEdge} stroke-width={1.5} />
          <path d={edge} fill="none" stroke="currentColor" stroke-opacity={0.35} stroke-width={1} />
          <SeaLevel y0={y(0)} w={LENS_W} h={LENS_H} ink="currentColor" />
          <line x1={LENS_W / 2} x2={LENS_W / 2} y1={0} y2={LENS_H} stroke="currentColor" stroke-opacity={0.25} stroke-dasharray="2 2" />
          {trackRuns(ss, x, y).map(([l, pts]) => <polyline points={pts} fill="none" stroke={LEVEL_COLOURS[l]} stroke-width={3} stroke-linejoin="round" />)}
          {crosses.map((c) => <circle cx={x(c.s)} cy={y(topOf(c.s) + drawnH(c.h))} r={4} fill={lineOf.get(c.lines[0])?.colour ?? "currentColor"} stroke="var(--map-land)" stroke-width={1} />)}
          {stops.map((st) => {
            const sx = x(st.s), sy = y(topOf(st.s) + drawnH(ownAt(st.s)));
            return <>
              <circle cx={sx} cy={sy} r={4.5} fill="#fff" stroke="var(--map-ink)" stroke-width={2} />
              <text x={Math.max(4, Math.min(LENS_W - 4, sx))} y={Math.max(12, sy - 9)} text-anchor={sx < 40 ? "start" : sx > LENS_W - 40 ? "end" : "middle"} font-size="11" fill="currentColor">{stopName(st.node)}</text>
            </>;
          })}
        </g>
      </svg>
      {from && <div>{tr("lineProfile.along", { distance: distance(s), stop: from })}</div>}
      <div>{mid.top === undefined ? tr("lineProfile.groundUnknown")
        : mid.water ? tr("lineProfile.water", { sea: mid.top < 0.5, height: heightText(mid.top) })
        : tr("lineProfile.ground", { height: heightText(mid.top) })}</div>
      <div>{tr("lineProfile.track", { dir: trackDir, level: levelLabel(lv), height: heightText(Math.abs(above)) })}</div>
      {crosses.filter((c) => Math.abs(c.s - s) <= NEAR_M).slice(0, 3).map((c) => {
        const ls = c.lines.map((n) => lineOf.get(n)).filter((q): q is LineInput => !!q);
        const dz = drawnH(c.h - ownAt(c.s));
        const chips = <span class="lp-chips">{ls.slice(0, 4).map((q) => <Chip line={q} size="sm" plain />)}</span>;
        return <div>{tRich("lineProfile.crosses", { dir: crossDir(dz), count: ls.length, lines: chips, height: heightText(Math.abs(dz)) })}</div>;
      })}
    </div>
  );
}
