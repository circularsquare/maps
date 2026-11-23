// T-4114: a line's strokes drawn ahead of the network. A stop click or a deleted line changes which edges a line's
// strokes draw; the worker says so right after the edit is applied (`linePath`), well before the network with it is
// built, and the line tool guesses the plain cases at the click. Until that network is set the line's own strokes are
// not drawn (the stroke shader's `u_hide_lines`) and these are: its segments on the edges it keeps, bands and all, and
// the track's segments in its colour on the edges it gained (centred: the network lays its band there). A deleted line
// is not drawn at all (its colour's alpha, as a line the player hid).

import { BAND_STRIDE } from "./lineBands";
import { F_BLUEPRINT, F_FADED, F_HSR, F_OFF, F_SPLIT, F_TRAM, type NetworkBuffers, type Strokes } from "./network";

/** what a line's strokes show of it: whether its trains run, high speed or tram trains */
export interface AheadLook { running: boolean; fast: boolean; tram: boolean }
/** `edges`: the edges its strokes draw now, null when the line is gone; `armed`: the worker sent them, so the next
 * network set holds the edit (a guess is kept through networks until then); `at`: when, for a guess never answered */
export interface Ahead { edges: Set<number> | null; armed: boolean; look: AheadLook; at: number }

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
