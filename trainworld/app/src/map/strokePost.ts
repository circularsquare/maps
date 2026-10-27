// The track and line strokes as the clock worker posts them (T-1381). Each edit used to lay both
// stroke sets end to end into new arrays (about 70 MB on Japan) to transfer them whole. Now a set
// is posted as one of three:
//   "whole": new arrays, with room after the last segment for later edits;
//   "same":  no arrays, the segments are exactly the last network's;
//   "patch": the segments that did not move, kept where they are in the page's arrays, and only the
//            ranges of new ones (`ranges`: start, count pairs), which the page writes over what was
//            there. A segment moving (one inserted or removed before it) makes the set whole.
// The page has to hold the whole arrays (picking, the pick grids, the renderer all read them), so
// posting only the new runs and rebuilding the rest there would move the copy to the main thread.
// A patch writes only where the last network has segments that the new one does not, and the new
// network's arrays are views of the same buffers: so the last network is not to be read once its
// successor is merged (`planStrokes`).

import type { Strokes } from "./network";

export type StrokeKind = "track" | "lines";
export type StrokeMode = "whole" | "same" | "patch";

/** values per segment of each array of a set */
const WIDTH = { seg: 4, colour: 1, edge: 1, level: 1, flags: 1, dist: 1, lane: 2, stretch: 1, speed: 1, trainSpeed: 1 } as const;
type Field = keyof typeof WIDTH;
type TypedArray = Float32Array | Uint32Array | Uint8Array;
const FIELDS: Record<StrokeKind, readonly Field[]> = {
  track: ["seg", "colour", "edge", "level", "flags", "dist", "lane", "speed", "trainSpeed"],
  lines: ["seg", "colour", "edge", "level", "flags", "dist", "lane", "stretch"],
};
const make = (f: Field, n: number): TypedArray => (f === "colour" || f === "edge" ? new Uint32Array(n) : f === "lane" ? new Uint8Array(n) : new Float32Array(n));

/** One edge's or line's piece of a set: its strokes, and for track the speed limit and fastest train
 * speed of each of its segments. */
export interface StrokePiece {
  strokes: Strokes;
  speed?: Float32Array;
  train?: Float32Array;
}
const arrayOf = (p: StrokePiece, f: Field): TypedArray | undefined => (f === "speed" ? p.speed : f === "trainSpeed" ? p.train : p.strokes[f]);

/** Arrays for `cap` segments, the first `count` of them in the Strokes returned: views of the
 * arrays, so those on the page keep the room for later edits. */
function allocate(kind: StrokeKind, count: number, cap: number): { s: Strokes; whole: Record<string, TypedArray> } {
  const whole: Record<string, TypedArray> = {}, view: Record<string, TypedArray> = {};
  for (const f of FIELDS[kind]) {
    const a = make(f, cap * WIDTH[f]);
    whole[f] = a;
    view[f] = cap === count ? a : a.subarray(0, count * WIDTH[f]);
  }
  return { s: { ...view, count } as unknown as Strokes, whole };
}

/** One piece's segments into `out` at segment `to`. */
function put(kind: StrokeKind, out: Record<string, TypedArray>, p: StrokePiece, to: number) {
  const n = p.strokes.count;
  for (const f of FIELDS[kind]) {
    const a = arrayOf(p, f);
    if (a) out[f].set(a, to * WIDTH[f]);
    else if (f === "lane") out[f].fill(255, to * 2, (to + n) * 2);
  }
}

/** The pieces laid end to end, into arrays of exactly their total size. */
export function strokesOfPieces(kind: StrokeKind, pieces: readonly StrokePiece[]): Strokes {
  let n = 0;
  for (const p of pieces) n += p.strokes.count;
  const { s, whole } = allocate(kind, n, n);
  let at = 0;
  for (const p of pieces) if (p.strokes.count) put(kind, whole, p, (at += p.strokes.count) - p.strokes.count);
  return s;
}

/** Strokes with a count and no segments, for a set that is not posted whole. */
function bare(kind: StrokeKind, count: number): Strokes {
  const s: Record<string, unknown> = { count };
  for (const f of FIELDS[kind]) s[f] = make(f, 0);
  return s as unknown as Strokes;
}

/** Segments of room left after the last in a whole post: an eighth, and at least this many. */
const MIN_ROOM = 2048;

export interface StrokePost {
  mode: StrokeMode;
  /** what to post: the arrays whole, none, or (patch) those of the new segments laid end to end, `count` the set's */
  strokes: Strokes;
  buffers: ArrayBufferLike[];
  /** patch: start, count pairs of the new segments */
  ranges: Int32Array;
}

/** The worker's side: what the page holds, and what to post of a set so that it ends with the pieces. */
export class StrokeSender {
  /** segments the page's arrays of each set have room for; 0 until a set is posted whole */
  private room: Record<StrokeKind, number> = { track: 0, lines: 0 };
  forget() { this.room = { track: 0, lines: 0 }; }

  /**
   * `runs`, `oldCount`: how the set differs from the one last posted (`NetPatch`: start, count, old
   * start or -1), or null for none (a whole network).
   */
  post(kind: StrokeKind, pieces: readonly StrokePiece[], count: number, runs: Int32Array | null, oldCount: number): StrokePost {
    const none = new Int32Array(0);
    if (runs && this.room[kind] > 0) {
      if (count === oldCount && (count === 0 || (runs.length === 3 && runs[0] === 0 && runs[1] === count && runs[2] === 0)))
        return { mode: "same", strokes: bare(kind, count), buffers: [], ranges: none };
      // every segment kept is where it was; the rest is new
      let kept = 0, fresh = 0, ok = count <= this.room[kind];
      for (let k = 0; k < runs.length && ok; k += 3) {
        if (runs[k + 2] < 0) fresh += runs[k + 1];
        else if (runs[k + 2] === runs[k]) kept += runs[k + 1];
        else ok = false;
      }
      if (ok && kept > 0) {
        const ranges: number[] = [];
        for (let k = 0; k < runs.length; k += 3) if (runs[k + 2] < 0) ranges.push(runs[k], runs[k + 1]);
        const { s, whole } = allocate(kind, fresh, fresh);
        let at = 0, r = 0, to = 0;
        for (const p of pieces) {
          const n = p.strokes.count;
          if (!n) continue;
          while (runs[r * 3] + runs[r * 3 + 1] <= at) r++;
          if (runs[r * 3 + 2] < 0) put(kind, whole, p, (to += n) - n);
          at += n;
        }
        s.count = count;
        return { mode: "patch", strokes: s, buffers: FIELDS[kind].map((f) => whole[f].buffer), ranges: Int32Array.from(ranges) };
      }
    }
    const cap = count + Math.max(count >> 3, MIN_ROOM);
    const { s, whole } = allocate(kind, count, cap);
    let at = 0;
    for (const p of pieces) if (p.strokes.count) put(kind, whole, p, (at += p.strokes.count) - p.strokes.count);
    this.room[kind] = cap;
    return { mode: "whole", strokes: s, buffers: FIELDS[kind].map((f) => whole[f].buffer), ranges: none };
  }
}

/** A set made whole from the one before it (`planStrokes`): `write` puts a patch's new segments into
 * the page's arrays, and is called only once every set of the network can be merged. */
export interface StrokeMerge {
  strokes: Strokes;
  write: () => void;
}

/**
 * The page's side: the set made whole from the one before it, `prev`, and what was posted, `msg` in
 * `mode` (`ranges`: see `StrokePost`). "patch" is views of `prev`'s own buffers, longer or shorter,
 * with the new segments written by `write`. Null when it cannot be (the page then asks for the whole
 * state): nothing has been written then.
 */
export function planStrokes(kind: StrokeKind, prev: Strokes | undefined, msg: Strokes, mode: StrokeMode, ranges: Int32Array): StrokeMerge | null {
  if (mode === "whole") return { strokes: msg, write() {} };
  if (!prev) return null;
  if (mode === "same") return msg.count === prev.count ? { strokes: { ...prev }, write() {} } : null;
  const count = msg.count;
  let fresh = 0, reach = 0;
  for (let k = 0; k + 1 < ranges.length; k += 2) {
    if (ranges[k] < reach || ranges[k + 1] <= 0 || ranges[k] + ranges[k + 1] > count) return null;
    reach = ranges[k] + ranges[k + 1];
    fresh += ranges[k + 1];
  }
  // segments past the page's own are all new
  if (count > prev.count && !(ranges.length && ranges[ranges.length - 2] <= prev.count && reach === count)) return null;
  const out: Record<string, unknown> = { count };
  const writes: (() => void)[] = [];
  for (const f of FIELDS[kind]) {
    const a = prev[f] as TypedArray | undefined, tail = msg[f] as TypedArray | undefined, w = WIDTH[f];
    if (!a || !tail || a.byteOffset !== 0 || a.length !== prev.count * w || tail.length !== fresh * w || a.buffer.byteLength < count * w * a.BYTES_PER_ELEMENT) return null;
    const v = new (a.constructor as new (b: ArrayBufferLike, o: number, n: number) => TypedArray)(a.buffer, 0, count * w);
    out[f] = v;
    writes.push(() => {
      for (let k = 0, from = 0; k < ranges.length; from += ranges[k + 1], k += 2) v.set(tail.subarray(from * w, (from + ranges[k + 1]) * w), ranges[k] * w);
    });
  }
  return { strokes: out as unknown as Strokes, write: () => writes.forEach((x) => x()) };
}

/** `planStrokes`, then its write. */
export function mergeStrokes(kind: StrokeKind, prev: Strokes | undefined, msg: Strokes, mode: StrokeMode, ranges = new Int32Array(0)): Strokes | null {
  const m = planStrokes(kind, prev, msg, mode, ranges);
  m?.write();
  return m?.strokes ?? null;
}
