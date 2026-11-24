// T-4114, T-4145: what is drawn ahead of the network (game/pending.ts makes the model). A line whose edges change: until
// the network holding its edit is set, its own strokes are not drawn (the stroke shader's per-line texture) and these
// are: its segments on the edges it keeps, bands and all, and the track's segments in its colour on the edges it gained
// (centred: the network lays its band there). A deleted line is not drawn at all (its colour's alpha, as a line the
// player hid). An edge with a new look: its network strokes not drawn (the per-edge texture) and redrawn from the same
// segments with the look changed. Station boxes are drawn by the station shader from `standInStations`.

import { BAND_STRIDE } from "./lineBands";
import { F_BLUEPRINT, F_BUSWAY, F_FADED, F_HSR, F_OFF, F_SPLIT, F_TRAM, type NetworkBuffers, type Strokes } from "./network";
import { CHEVRON_STRIDE, oneWayTrackRuns, type OneWayEdge } from "./oneWayMarks";
import { platformPoints, platformShape, PLATFORM_HALF_M, PLATFORM_POINTS, ST_ONE, ST_ONE_MID, ST_ONE_NEG, ST_TRAM } from "./stationGeometry";
import type { EdgeLook, StationBox } from "../workers/protocol";

/** what a line's strokes show of it: whether its trains run, high speed or tram trains */
export interface AheadLook { running: boolean; fast: boolean; tram: boolean }
/** `edges`: the edges its strokes draw now, null when the line is gone; `armed`: the worker sent them, so the next
 * network set holds the edit (a guess is kept through networks until then); `at`: when, for a guess never answered */
export interface Ahead { edges: Set<number> | null; look: AheadLook }

/** T-4145: everything drawn ahead of the network: lines by id, edges hidden or drawn with a new look, stations hidden
 * (node ids) or drawn as boxes, line colours (0-255), and strokes to draw as they are (an undo's, from the network
 * before the edit) */
export interface AheadModel {
  lines: Map<number, Ahead>;
  hideEdges: Set<number>;
  edges: Map<number, EdgeLook>;
  hideStations: Set<number>;
  stations: StationBox[];
  colours: Map<number, [number, number, number]>;
  track: Strokes[];
  lineStrokes: Strokes[];
  /** lines whose network strokes are not drawn (an undo draws them from `lineStrokes`) */
  hideLines: Set<number>;
  rawStations: RawStations[];
}
export const noAhead = (): AheadModel => ({ lines: new Map(), hideEdges: new Set(), edges: new Map(), hideStations: new Set(), stations: [], colours: new Map(), track: [], lineStrokes: [], hideLines: new Set(), rawStations: [] });
export const aheadEmpty = (m: AheadModel) => !m.lines.size && !m.hideEdges.size && !m.edges.size && !m.hideStations.size && !m.stations.length && !m.colours.size && !m.track.length && !m.lineStrokes.length && !m.hideLines.size && !m.rawStations.length;

/** The lines' strokes drawn ahead of the network `n`. */
export function aheadStrokes(n: NetworkBuffers, ahead: ReadonlyMap<number, Ahead>): Strokes {
  const L = n.lines, T = n.track;
  const keep: number[] = [];
  const have = new Map<number, Set<number>>();
  for (let i = 0; i < L.count; i++) {
    const line = L.colour[i];
    if (!ahead.get(line)?.edges?.has(L.edge[i])) continue;
    keep.push(i);
    let s = have.get(line);
    if (!s) have.set(line, (s = new Set()));
    s.add(L.edge[i]);
  }
  // the edges a line gained, each with the lines that gained it
  const need = new Map<number, number[]>();
  for (const [line, a] of ahead) {
    if (!a.edges) continue;
    for (const e of a.edges) {
      if (have.get(line)?.has(e)) continue;
      const ls = need.get(e);
      if (ls) ls.push(line);
      else need.set(e, [line]);
    }
  }
  const add: number[] = [];
  if (need.size) for (let j = 0; j < T.count; j++) for (const line of need.get(T.edge[j]) ?? []) add.push(j, line);
  const m = keep.length + add.length / 2;
  const lane = new Uint8Array(2 * m).fill(255), band = new Int16Array(BAND_STRIDE * m);
  const out: Strokes = { seg: new Float32Array(4 * m), colour: new Uint32Array(m), edge: new Uint32Array(m), level: new Float32Array(m), flags: new Float32Array(m), dist: new Float32Array(m), lane, band, count: m };
  let k = 0;
  for (const i of keep) {
    out.seg.set(L.seg.subarray(4 * i, 4 * i + 4), 4 * k);
    out.colour[k] = L.colour[i];
    out.edge[k] = L.edge[i];
    out.level[k] = L.level[i];
    out.flags[k] = L.flags[i];
    if (L.lane) lane.set(L.lane.subarray(2 * i, 2 * i + 2), 2 * k);
    if (L.band) band.set(L.band.subarray(BAND_STRIDE * i, BAND_STRIDE * (i + 1)), BAND_STRIDE * k);
    k++;
  }
  for (let q = 0; q < add.length; q += 2) {
    const j = add[q], line = add[q + 1], look = ahead.get(line)!.look, f = T.flags[j];
    const unbuilt = (f & F_BLUEPRINT) !== 0;
    out.seg.set(T.seg.subarray(4 * j, 4 * j + 4), 4 * k);
    out.colour[k] = line;
    out.edge[k] = T.edge[j];
    out.level[k] = T.level[j];
    // as network.ts flags a line's own segment there
    out.flags[k] = (look.running && !unbuilt ? 0 : F_FADED | (unbuilt ? F_BLUEPRINT | (f & F_OFF) : 0)) | (look.fast ? f & F_HSR : 0) | (look.tram ? f & F_TRAM : 0) | (f & F_SPLIT);
    if (T.lane) lane.set(T.lane.subarray(2 * j, 2 * j + 2), 2 * k);
    k++;
  }
  return out;
}

/** The edges line `line`'s strokes draw in network `n`. */
export function drawnEdgesOf(n: NetworkBuffers, line: number): Set<number> {
  const L = n.lines, out = new Set<number>();
  for (let i = 0; i < L.count; i++) if (L.colour[i] === line) out.add(L.edge[i]);
  return out;
}

/** T-4114: the edges a line's strokes draw once its first or last stop is taken out (`stop` 0 or the last), from network
 * `n`: all but those drawn only on the stretch to that stop (run 0's numbering); null where the network cannot tell
 * (no stretches, a line drawn on no stretch there). */
export function edgesWithoutEnd(n: NetworkBuffers, line: number, stop: number, stops: number): Set<number> | null {
  const L = n.lines, cut = stop === 0 ? 0 : stops - 2;
  if (!L.stretch || stops < 3 || (stop !== 0 && stop !== stops - 1)) return null;
  const keep = new Set<number>(), end = new Set<number>();
  for (let i = 0; i < L.count; i++) {
    if (L.colour[i] !== line) continue;
    (L.stretch[i] === cut ? end : keep).add(L.edge[i]);
  }
  if (!end.size) return null;
  return keep;
}

/** T-4147: the edges a line's strokes draw once its stops before `lo` or from `hi` on are cut away (`endLineAt`: the
 * stretches lo..hi-1 stay), from network `n`; null where the network cannot tell (nothing cut, or nothing left). */
export function edgesInStretches(n: NetworkBuffers, line: number, lo: number, hi: number): Set<number> | null {
  const L = n.lines;
  if (!L.stretch) return null;
  const keep = new Set<number>();
  let cut = false;
  for (let i = 0; i < L.count; i++) {
    if (L.colour[i] !== line) continue;
    const s = L.stretch[i];
    if (s >= lo && s < hi) keep.add(L.edge[i]);
    else cut = true;
  }
  return cut && keep.size ? keep : null;
}

/** `levelCode`'s inverse halves: a segment's level code from its start and end levels */
const levelCode = (a: number, b: number) => (a + 4) * 32 * 256 + (b + 4) * 32;

/** T-4145: stroke `i` of `s` with an edge's look: flags, level (`k` of `m` segments along the edge) and track count */
function withLook(out: Strokes, k: number, look: EdgeLook, line: boolean, at: number, of: number) {
  let f = out.flags[k];
  if (look.built) f &= ~(F_BLUEPRINT | F_OFF | (line ? F_FADED : 0));
  if (look.off === true && f & F_BLUEPRINT) f |= F_OFF;
  if (look.off === false) f &= ~F_OFF;
  // (a line over the edge takes a tram's look but not high speed's, which is the line's own setting)
  if (look.kind !== undefined) {
    f &= ~(F_HSR | F_TRAM | F_BUSWAY);
    if (look.kind === 1 && !line) f |= F_HSR;
    if (look.kind >= 2) f |= F_TRAM | (look.kind === 4 && !line ? F_BUSWAY : 0);
  }
  out.flags[k] = f;
  if (look.level) {
    const [a, b] = look.level, t0 = of ? at / of : 0, t1 = of ? (at + 1) / of : 1;
    out.level[k] = levelCode(Math.round(a + (b - a) * t0), Math.round(a + (b - a) * t1));
  }
  if (look.tracks) out.lane![2 * k] = out.lane![2 * k + 1] = look.tracks === 2 ? 255 : 0;
}

/** T-4148: the one-way track chevrons drawn ahead of network `n`: its own without those on edges an edit sets one way
 * or clears (a chevron lies on its edge's stroke), and the chevrons of the edges set to run one way, laid as the network
 * lays them but clear only of stations and junctions. Null when no look changes a way. */
export function aheadChevrons(n: NetworkBuffers, edges: ReadonlyMap<number, EdgeLook>): Float32Array | null {
  const ways = new Map<number, 0 | 1 | 2 | 3>();
  for (const [e, look] of edges) if (look.oneWay !== undefined) ways.set(e, look.oneWay);
  if (!ways.size) return null;
  const T = n.track, idx = new Map<number, number[]>();
  for (let i = 0; i < T.count; i++) if (ways.has(T.edge[i])) (idx.get(T.edge[i]) ?? idx.set(T.edge[i], []).get(T.edge[i])!).push(i);
  const base = n.oneWayTrack ?? new Float32Array(0), keep: number[] = [];
  // a chevron is on a stroke when it is as near as the floats say
  const on = (x: number, y: number) => {
    for (const list of idx.values()) for (const i of list) {
      const x0 = T.seg[4 * i], y0 = T.seg[4 * i + 1], dx = T.seg[4 * i + 2] - x0, dy = T.seg[4 * i + 3] - y0, l2 = dx * dx + dy * dy;
      const t = l2 > 0 ? Math.max(0, Math.min(1, ((x - x0) * dx + (y - y0) * dy) / l2)) : 0;
      if (Math.hypot(x0 + t * dx - x, y0 + t * dy - y) <= 1e-4 * Math.sqrt(l2 || 1) + 1e-9) return true;
    }
    return false;
  };
  for (let k = 0; k + CHEVRON_STRIDE <= base.length; k += CHEVRON_STRIDE) if (!on(base[k], base[k + 1])) keep.push(k);
  const fresh: number[] = [], runs: OneWayEdge[] = [];
  for (const [e, way] of ways) if (way === 1 || way === 2) {
    const list = idx.get(e), ends = edges.get(e)?.ends;
    if (list?.length && ends) runs.push({ a: ends[0], b: ends[1], way, strokes: pick(T, list) });
  }
  if (runs.length) {
    const clear: [number, number][] = [];
    for (let i = 0; i < n.stationCount; i++) clear.push([n.stations[i * 3], n.stations[i * 3 + 1]]);
    for (let i = 0; i + 1 < n.junctions.length; i += 2) clear.push([n.junctions[i], n.junctions[i + 1]]);
    oneWayTrackRuns(runs, clear, fresh);
  }
  const out = new Float32Array(keep.length * CHEVRON_STRIDE + fresh.length);
  keep.forEach((k, j) => out.set(base.subarray(k, k + CHEVRON_STRIDE), j * CHEVRON_STRIDE));
  out.set(fresh, keep.length * CHEVRON_STRIDE);
  return out;
}

/** Strokes `idx` of `s` copied out (lane and band where it has them). */
function pick(s: Strokes, idx: number[]): Strokes {
  const m = idx.length, lane = new Uint8Array(2 * m).fill(255), band = new Int16Array(BAND_STRIDE * m);
  const out: Strokes = { seg: new Float32Array(4 * m), colour: new Uint32Array(m), edge: new Uint32Array(m), level: new Float32Array(m), flags: new Float32Array(m), dist: new Float32Array(m), lane, band, count: m };
  idx.forEach((i, k) => {
    out.seg.set(s.seg.subarray(4 * i, 4 * i + 4), 4 * k);
    out.colour[k] = s.colour[i];
    out.edge[k] = s.edge[i];
    out.level[k] = s.level[i];
    out.flags[k] = s.flags[i];
    out.dist[k] = s.dist[i];
    if (s.lane) lane.set(s.lane.subarray(2 * i, 2 * i + 2), 2 * k);
    if (s.band) band.set(s.band.subarray(BAND_STRIDE * i, BAND_STRIDE * (i + 1)), BAND_STRIDE * k);
  });
  return out;
}

/** Strokes joined into one set. */
export function joinStrokes(parts: Strokes[]): Strokes {
  const m = parts.reduce((a, p) => a + p.count, 0);
  const out: Strokes = { seg: new Float32Array(4 * m), colour: new Uint32Array(m), edge: new Uint32Array(m), level: new Float32Array(m), flags: new Float32Array(m), dist: new Float32Array(m), lane: new Uint8Array(2 * m).fill(255), band: new Int16Array(BAND_STRIDE * m), count: m };
  let k = 0;
  for (const p of parts) {
    out.seg.set(p.seg.subarray(0, 4 * p.count), 4 * k);
    out.colour.set(p.colour.subarray(0, p.count), k);
    out.edge.set(p.edge.subarray(0, p.count), k);
    out.level.set(p.level.subarray(0, p.count), k);
    out.flags.set(p.flags.subarray(0, p.count), k);
    out.dist.set(p.dist.subarray(0, p.count), k);
    if (p.lane) out.lane!.set(p.lane.subarray(0, 2 * p.count), 2 * k);
    if (p.band) out.band!.set(p.band.subarray(0, BAND_STRIDE * p.count), BAND_STRIDE * k);
    k += p.count;
  }
  return out;
}

/** T-4145: the line and track strokes drawn ahead of network `n`: the lines' (`aheadStrokes`), and the network's own
 * strokes on edges with a new look, redrawn with it (a line drawn ahead takes the look on its own strokes) */
export function aheadParts(n: NetworkBuffers, m: AheadModel): { lines: Strokes | null; track: Strokes | null } {
  const lineParts: Strokes[] = [], trackParts: Strokes[] = [...m.track];
  if (m.lines.size) lineParts.push(aheadStrokes(n, m.lines));
  if (m.edges.size) {
    for (const [s, isLine] of [[n.lines, true], [n.track, false]] as const) {
      const idx: number[] = [];
      for (let i = 0; i < s.count; i++) if (m.edges.has(s.edge[i]) && !(isLine && m.lines.has(s.colour[i]))) idx.push(i);
      if (idx.length) (isLine ? lineParts : trackParts).push(pick(s, idx));
    }
    // the look on every stroke drawn ahead, counted along each edge in the order the network lays them
    for (const [parts, isLine] of [[lineParts, true], [trackParts.slice(m.track.length), false]] as const)
      for (const p of parts) {
        const seen = new Map<string, number>(), of = new Map<string, number>();
        const key = (k: number) => p.colour[k] + ":" + p.edge[k];
        for (let k = 0; k < p.count; k++) if (m.edges.has(p.edge[k])) of.set(key(k), (of.get(key(k)) ?? 0) + 1);
        for (let k = 0; k < p.count; k++) {
          const look = m.edges.get(p.edge[k]);
          if (!look) continue;
          const at = seen.get(key(k)) ?? 0;
          seen.set(key(k), at + 1);
          withLook(p, k, look, isLine, at, of.get(key(k))!);
        }
      }
  }
  lineParts.push(...m.lineStrokes);
  const nonEmpty = (ps: Strokes[]) => ps.filter((p) => p.count);
  const l = nonEmpty(lineParts), t = nonEmpty(trackParts);
  return { lines: l.length ? (l.length === 1 ? l[0] : joinStrokes(l)) : null, track: t.length ? (t.length === 1 ? t[0] : joinStrokes(t)) : null };
}

/** T-4145: station boxes in the station buffers' form (network.ts's `stations`, `stationAxis`, `stationShape`,
 * `stationLine`), for a second draw of the station shader */
export function standInStations(boxes: readonly StationBox[]): RawStations {
  const m = boxes.length, P2 = 2 * PLATFORM_POINTS;
  const st = new Float32Array(Math.max(1, m) * 3), axis = new Float32Array(Math.max(1, m) * 2), shape = new Float32Array(Math.max(1, m) * 4), line = new Float32Array(Math.max(1, m) * P2);
  boxes.forEach((b, i) => {
    const sh = platformShape(b.ways, 0, b.plat / 2 * b.perM);
    // as network.ts flags a station: 2 blueprint, 4 a line through it
    st.set([b.lx, b.ly, (b.built ? 0 : 2) | 4 | (b.side ? ST_ONE | (b.side < 0 ? ST_ONE_NEG : 0) : 0) | (b.mid ? ST_ONE_MID : 0) | (b.tram ? ST_TRAM : 0)], i * 3);
    axis.set([sh.ax, sh.ay], i * 2);
    shape.set([sh.mx, sh.my, sh.phi, PLATFORM_HALF_M * b.perM], i * 4);
    line.set(platformPoints(sh), i * P2);
  });
  return { st, axis, shape, line, count: m };
}

/** T-4145: stations in the buffers' form, drawn as they are (an undo's, from the network before the edit) */
export interface RawStations { st: Float32Array; axis: Float32Array; shape: Float32Array; line: Float32Array; count: number }

/** Stations `idx` of network `n` in the buffers' form. */
function stationsOf(n: NetworkBuffers, idx: number[]): RawStations {
  const P2 = 2 * PLATFORM_POINTS, m = idx.length;
  const out: RawStations = { st: new Float32Array(m * 3), axis: new Float32Array(m * 2), shape: new Float32Array(m * 4), line: new Float32Array(m * P2), count: m };
  idx.forEach((i, k) => {
    out.st.set(n.stations.subarray(i * 3, i * 3 + 3), k * 3);
    if (n.stationAxis) out.axis.set(n.stationAxis.subarray(i * 2, i * 2 + 2), k * 2);
    if (n.stationShape) out.shape.set(n.stationShape.subarray(i * 4, i * 4 + 4), k * 4);
    if (n.stationLine && n.stationLine.length >= (i + 1) * P2) out.line.set(n.stationLine.subarray(i * P2, (i + 1) * P2), k * P2);
  });
  return out;
}

/** Station sets joined (boxes first). */
export function joinStations(parts: RawStations[]): RawStations {
  const P2 = 2 * PLATFORM_POINTS, m = parts.reduce((a, p) => a + p.count, 0);
  const out: RawStations = { st: new Float32Array(Math.max(1, m) * 3), axis: new Float32Array(Math.max(1, m) * 2), shape: new Float32Array(Math.max(1, m) * 4), line: new Float32Array(Math.max(1, m) * P2), count: m };
  let k = 0;
  for (const p of parts) {
    out.st.set(p.st.subarray(0, p.count * 3), k * 3);
    out.axis.set(p.axis.subarray(0, p.count * 2), k * 2);
    out.shape.set(p.shape.subarray(0, p.count * 4), k * 4);
    out.line.set(p.line.subarray(0, p.count * P2), k * P2);
    k += p.count;
  }
  return out;
}

/** A number per key from what each stroke of `s` draws (position, flags, level, lanes), for telling two networks apart. */
const sigCache = new WeakMap<Strokes, Map<number, number>>();
function signatures(s: Strokes, keys: ArrayLike<number>): Map<number, number> {
  const kept = sigCache.get(s);
  if (kept) return kept;
  const out = new Map<number, number>(), seg = s.seg, fl = s.flags, lv = s.level, ed = s.edge, ln = s.lane;
  // (a key's strokes mostly come together: its hash is carried along and put in the map when the key changes)
  let k = -1, h = 0;
  for (let i = 0; i < s.count; i++) {
    const ki = keys[i];
    if (ki !== k) {
      if (k >= 0) out.set(k, h);
      k = ki;
      h = out.get(k) ?? 17;
    }
    const o = 4 * i;
    h = (Math.imul(h, 31) + ((seg[o] * 16) | 0)) | 0;
    h = (Math.imul(h, 31) + ((seg[o + 1] * 16) | 0)) | 0;
    h = (Math.imul(h, 31) + ((seg[o + 2] * 16) | 0)) | 0;
    h = (Math.imul(h, 31) + ((seg[o + 3] * 16) | 0)) | 0;
    h = (Math.imul(h, 31) + fl[i] * 7 + lv[i] + ed[i] * 3 + (ln ? ln[2 * i] + ln[2 * i + 1] * 256 : 0)) | 0;
  }
  if (k >= 0) out.set(k, h);
  sigCache.set(s, out);
  return out;
}
/** T-4145: the undo's comparisons of network `n` worked out ahead (an idle moment after it is drawn) */
export function warmUndo(n: NetworkBuffers) {
  signatures(n.track, n.track.edge);
  signatures(n.lines, n.lines.colour);
}
const changedKeys = (a: Map<number, number>, b: Map<number, number>) => {
  const out = new Set<number>();
  for (const [k, h] of a) if (b.get(k) !== h) out.add(k);
  for (const k of b.keys()) if (!a.has(k)) out.add(k);
  return out;
};

/** T-4145: an undo of the newest edit, drawn at its click from the network before that edit (`was`) over the one drawn
 * now (`n`): what differs is hidden in `n` and drawn from `was` (track, line strokes, stations, line colours). */
export function undoAhead(was: NetworkBuffers, n: NetworkBuffers): { hideEdges: number[]; hideLines: number[]; hideStations: number[]; track: Strokes[]; lineStrokes: Strokes[]; stations: RawStations | null; colours: Map<number, [number, number, number]> } {
  const edges = changedKeys(signatures(was.track, was.track.edge), signatures(n.track, n.track.edge));
  const lines = changedKeys(signatures(was.lines, was.lines.colour), signatures(n.lines, n.lines.colour));
  const trackIdx: number[] = [], lineIdx: number[] = [];
  for (let i = 0; i < was.track.count; i++) if (edges.has(was.track.edge[i])) trackIdx.push(i);
  for (let i = 0; i < was.lines.count; i++) if (lines.has(was.lines.colour[i]) || edges.has(was.lines.edge[i])) lineIdx.push(i);
  // lines kept whole on an edge hidden for its new track: drawn from `was` too (lineIdx has them), so hidden in `n` only there
  const stSig = (m: NetworkBuffers, i: number) => [m.stations[i * 3], m.stations[i * 3 + 1], m.stations[i * 3 + 2], m.stationShape?.[i * 4 + 3] ?? 0, m.stationAxis?.[i * 2] ?? 0, m.stationAxis?.[i * 2 + 1] ?? 0].join(",");
  const wasSt = new Map(was.stationIds.map((id, i) => [id, i])), nowSt = new Map(n.stationIds.map((id, i) => [id, i]));
  const hideStations: number[] = [], drawSt: number[] = [];
  for (const [id, i] of nowSt) {
    const j = wasSt.get(id);
    if (j === undefined || stSig(was, j) !== stSig(n, i)) hideStations.push(Number(id));
  }
  for (const [id, j] of wasSt) {
    const i = nowSt.get(id);
    if (i === undefined || stSig(was, j) !== stSig(n, i)) drawSt.push(j);
  }
  const colours = new Map<number, [number, number, number]>();
  for (let l = 0; l * 4 + 3 < was.lineColours.length && l * 4 + 3 < n.lineColours.length; l++)
    if ([0, 1, 2].some((k) => was.lineColours[l * 4 + k] !== n.lineColours[l * 4 + k])) colours.set(l, [was.lineColours[l * 4], was.lineColours[l * 4 + 1], was.lineColours[l * 4 + 2]]);
  return {
    hideEdges: [...edges], hideLines: [...lines], hideStations,
    track: trackIdx.length ? [pick(was.track, trackIdx)] : [], lineStrokes: lineIdx.length ? [pick(was.lines, lineIdx)] : [],
    stations: drawSt.length ? stationsOf(was, drawSt) : null, colours,
  };
}
