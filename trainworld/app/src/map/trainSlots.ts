// The trains' tables on the GPU in slots of their own (T-4274). Each line's run holds a slot in the per-sample tables
// (its samples, where its track bends, its stretches, lanes, fill and levels) and one in the bend points, each line one
// in the phase tables, so an edit writes only the lines it changed and a deleted line's slot takes the next line that
// fits. The shaders reach a slot only through the index tables (a line row's first sample, a profile's first phase, a
// sample's first bend point), which are the values rewritten here; the network's arrays stay packed for every reader
// on the CPU. Nothing here is drawn in order (unlike the strokes, strokeSlots.ts), so a slot goes wherever it fits.

import { META_PER_LINE, PROFILES_PER_LINE } from "./profiles";
import type { NetworkRenderer } from "./renderer";
import type { TexKind } from "./rendererStrokes";

/**
 * A table's slots, made for one network and never changed after: per key (`sampleRows`' or a line id) its slot and
 * where its values lie in that network's packed arrays.
 */
export interface SlotLayout {
  at: Map<number, number>;
  /** texels its slot holds, at least `count` (a line that shrank keeps its slot) */
  size: Map<number, number>;
  first: Map<number, number>;
  count: Map<number, number>;
  /** free texels below `end`: start, count pairs in order */
  free: number[];
  end: number;
  /** texels the tables under it are made to hold */
  cap: number;
  /** the layout laid out afresh this one comes from: a table last written under another is written whole */
  base: number;
  /** every slot is where the packed arrays have its values (laid out afresh from arrays with no gaps) */
  packed: boolean;
}

/** `?slotRoom=tiny` (checks): the tables keep 8 texels of room past a fresh layout, so a small game's edits lay out afresh */
export const trainRoom = { tiny: false };
if (typeof location !== "undefined" && new URLSearchParams(location.search).get("slotRoom") === "tiny") trainRoom.tiny = true;

/** The room a line's phase tables keep in their slot: a stop more or less changes their length on most edits. */
export const PHASE_ROOM = 1 / 8;

/** A bend point's slot is drawn only below this (`bendAt` holds it times 32 plus a count in a float, exact below 2^24). */
export const BEND_LIMIT = 1 << 19;

let bases = 0;

/**
 * Slots for `rows` (key, packed first, count triples, each count above 0): the last layout's kept while they still hold
 * their key, the rest where they fit best among the free ones (a deleted line's slot takes the next line its size), else
 * after the last; laid out afresh in packed order once they outgrow the tables or pass `limit` texels. A slot given
 * out holds `room` more than its count (a share), so a line that grows a little keeps it. The last layout itself while
 * no row changed.
 */
export function planSlots(prev: SlotLayout | null, rows: readonly number[], width: number, limit = Infinity, room = 0): SlotLayout {
  if (prev && sameRows(prev, rows)) return prev;
  if (prev) {
    const next = moveSlots(prev, rows, room);
    if (next.end <= prev.cap && next.end <= limit) return next;
  }
  return afresh(rows, width, room);
}

function sameRows(L: SlotLayout, rows: readonly number[]): boolean {
  if (rows.length / 3 !== L.at.size) return false;
  for (let i = 0; i < rows.length; i += 3) if (L.first.get(rows[i]) !== rows[i + 1] || L.count.get(rows[i]) !== rows[i + 2]) return false;
  return true;
}

function moveSlots(prev: SlotLayout, rows: readonly number[], room: number): SlotLayout {
  const first = new Map<number, number>(), count = new Map<number, number>();
  for (let i = 0; i < rows.length; i += 3) first.set(rows[i], rows[i + 1]), count.set(rows[i], rows[i + 2]);
  const at = new Map<number, number>(), size = new Map<number, number>(), free = prev.free.slice();
  let end = prev.end;
  // (the slots let go first, so a line that moves may take another's)
  for (const [k, a] of prev.at) {
    const s = prev.size.get(k)!, c = count.get(k);
    if (c !== undefined && c <= s) at.set(k, a), size.set(k, s);
    else end = release(free, a, s, end);
  }
  for (let i = 0; i < rows.length; i += 3) {
    const k = rows[i], c = rows[i + 2] + Math.floor(rows[i + 2] * room);
    if (at.has(k)) continue;
    let best = -1;
    for (let j = 0; j < free.length; j += 2) if (free[j + 1] >= c && (best < 0 || free[j + 1] < free[best + 1])) best = j;
    if (best < 0) {
      at.set(k, end);
      end += c;
    } else {
      at.set(k, free[best]);
      if (free[best + 1] === c) free.splice(best, 2);
      else (free[best] += c), (free[best + 1] -= c);
    }
    size.set(k, c);
  }
  return { at, size, first, count, free, end, cap: prev.cap, base: prev.base, packed: false };
}

/** texels `a` to `a + s` let go into `free`, joined with the free runs beside them; the new `end` */
function release(free: number[], a: number, s: number, end: number): number {
  let j = 0;
  while (j < free.length && free[j] < a) j += 2;
  free.splice(j, 0, a, s);
  if (j + 2 < free.length && free[j] + free[j + 1] === free[j + 2]) (free[j + 1] += free[j + 3]), free.splice(j + 2, 2);
  if (j > 0 && free[j - 2] + free[j - 1] === free[j]) (free[j - 1] += free[j + 1]), free.splice(j, 2), (j -= 2);
  if (free[j] + free[j + 1] === end) (end = free[j]), free.splice(j, 2);
  return end;
}

function afresh(rows: readonly number[], width: number, room: number): SlotLayout {
  const order: number[] = [];
  for (let i = 0; i < rows.length; i += 3) order.push(i);
  order.sort((x, y) => rows[x + 1] - rows[y + 1]);
  const at = new Map<number, number>(), size = new Map<number, number>(), first = new Map<number, number>(), count = new Map<number, number>();
  let end = 0, packed = true, most = 0;
  for (const i of order) {
    const k = rows[i], f = rows[i + 1], c = rows[i + 2], s = c + Math.floor(c * room);
    if (f !== end || s !== c) packed = false;
    at.set(k, end), size.set(k, s), first.set(k, f), count.set(k, c);
    end += s;
    most = Math.max(most, s);
  }
  // (an eighth, and room for the largest line to move past the last: on her Asia save one holds a fifth of the phases)
  const cap = end + (trainRoom.tiny ? 8 : Math.max(4 * width, end >> 3, most + (most >> 3)));
  return { at, size, first, count, free: [], end, cap, base: ++bases, packed };
}

/** Keys of a line table's rows: a line's own run 0 its id times 2, its return run's own way back (T-3250, the rows past
 * `base`) that plus 1. */
export const rowKey = (r: number, base: number) => (base > 0 && r >= base ? (r - base) * 2 + 1 : r * 2);

/** The per-sample tables' rows (key, first sample, count) of a line table (`base`: `backBaseOf`). */
export function sampleRows(table: ArrayLike<number>, base: number): number[] {
  const out: number[] = [];
  for (let r = 0; r * 4 + 1 < table.length; r++) {
    const c = table[r * 4 + 1];
    if (c > 0) out.push(rowKey(r, base), table[r * 4], c);
  }
  return out;
}

/** The line table as drawn: each row's first sample its slot's (0 for a row of no samples). */
export function slottedLineTable(table: Float32Array, base: number, L: SlotLayout): Float32Array {
  const out = table.slice();
  for (let r = 0; r * 4 + 1 < table.length; r++) out[r * 4] = table[r * 4 + 1] > 0 ? L.at.get(rowKey(r, base)) ?? 0 : 0;
  return out;
}

/** The bend points' rows of `sampleRows` `rows` (key, first point, count): a run's points lie together, in its
 * samples' order (clock worker `bendTables`). */
export function bendRows(sampleBend: ArrayLike<number>, rows: readonly number[]): number[] {
  const out: number[] = [];
  for (let i = 0; i < rows.length; i += 3) {
    let a = rows[i + 1], z = rows[i + 1] + rows[i + 2] - 1;
    while (a <= z && !(sampleBend[a] > 0)) a++;
    if (a > z) continue;
    while (!(sampleBend[z] > 0)) z--;
    const lo = Math.floor(sampleBend[a] / 32), hi = Math.floor(sampleBend[z] / 32) + (sampleBend[z] % 32);
    out.push(rows[i], lo, hi - lo);
  }
  return out;
}

/** The phase tables' rows (line id, first phase, count): each line's profiles' phases lie together (`profile_meta`). */
export function phaseRows(meta: ArrayLike<number>): number[] {
  const out: number[] = [];
  for (let l = 0; (l + 1) * META_PER_LINE <= meta.length; l++) {
    let lo = Infinity, hi = -Infinity;
    for (let o = l * META_PER_LINE; o < (l + 1) * META_PER_LINE; o += 4) {
      if (meta[o + 1] > 0) (lo = Math.min(lo, meta[o])), (hi = Math.max(hi, meta[o] + meta[o + 1]));
    }
    if (hi > lo) out.push(l, lo, hi - lo);
  }
  return out;
}

/** The profile meta as drawn: each profile's first phase in its line's slot (0 for a profile of no phases). */
export function slottedMeta(meta: Float32Array, L: SlotLayout): Float32Array {
  const out = meta.slice();
  for (let l = 0; (l + 1) * META_PER_LINE <= meta.length; l++) {
    const a = L.at.get(l), f = L.first.get(l) ?? 0;
    for (let o = l * META_PER_LINE; o < (l + 1) * META_PER_LINE; o += 4) out[o] = meta[o + 1] > 0 && a !== undefined ? a + meta[o] - f : 0;
  }
  return out;
}

/** `?patch=check`: whether every line's phases are laid out from its first profile's first phase, as the readers that
 * slice a line's phases take them (stateRows `splitPhases`, terminalBays.ts) */
export function phasesFromFirst(meta: ArrayLike<number>, rows: readonly number[]): boolean {
  for (let i = 0; i < rows.length; i += 3) {
    const m = rows[i] * META_PER_LINE;
    let n = 0;
    for (let p = 0; p < PROFILES_PER_LINE; p++) n += meta[m + p * 4 + 1];
    if (meta[m] !== rows[i + 1] || n !== rows[i + 2]) return false;
  }
  return true;
}

/** A train table on the GPU in a layout's slots, and what it was last written from. */
export interface SlotTex {
  tex: WebGLTexture;
  kind: TexKind;
  rows: number;
  layout: SlotLayout;
  srcs: readonly (ArrayLike<number> | undefined)[];
  /** per key, the shift its values were written with (`SlotSrc.shift`) */
  shift: Map<number, number> | null;
  /** keys whose slot holds something other than their values in `srcs` (`blankSlots`) */
  dirty: Set<number> | null;
  /** the keys the last write wrote, or "all" */
  wrote: Set<number> | "all";
  /** T-4280: the last write compared values to find what changed (no list could say) */
  compared?: boolean;
}

/** T-4280: the keys whose values may differ from the sources `from` (the network before's own arrays), as the worker names
 * them (`ClockState.touched`, `NetPatch.lanes`); used where a table was last written from exactly those arrays */
export interface SlotNames {
  keys: ReadonlySet<number>;
  from: readonly (ArrayLike<number> | undefined)[];
}

/** `?trainNames=drop` (checks): each list drops its first key, which `?patch=check` must find drawn wrong */
export const trainNamesTest = { drop: typeof location !== "undefined" && new URLSearchParams(location.search).get("trainNames") === "drop" };

/** A train table's values: its sources, in packed sample (or point, or phase) order, `per` numbers a texel each. */
export interface SlotSrc {
  srcs: readonly (Float32Array | Int32Array | undefined)[];
  per: readonly number[];
  /** a key's `count` texels from packed texel `first` into `out` from 0; absent: the one source as it is */
  pack?: (out: Float32Array, first: number, count: number) => void;
  /** `bendAt`: per key, how many bend points its codes move by (its points' slot less their packed first) */
  shift?: (key: number) => number;
}

const compsOf = (kind: TexKind) => (kind === "r32f" ? 1 : kind === "rg32f" ? 2 : 4);
const FORMATS = { r32f: [0x822e, 0x1903], rg32f: [0x8230, 0x8227], rgba32f: [0x8814, 0x1908], rgba8: [0x8058, 0x1908] } as const;
const words = (v: ArrayLike<number>) => { const a = v as Float32Array; return new Uint32Array(a.buffer, a.byteOffset, a.length); };

/** a `bendAt` code moved `shift` points along, 0 where it would pass what a float holds (drawn straight) */
const shifted = (code: number, shift: number) => { if (!(code > 0)) return 0; const v = code + shift * 32; return v < 1 << 24 ? v : 0; };

let scratch = new Float32Array(0);

/** Key `key`'s texels in layout `L` from `src`: a view of its source where it is the values as they are, else made in
 * a scratch array (used until the next call). */
function stage(src: SlotSrc, L: SlotLayout, key: number, comps: number, shift: number): Float32Array {
  const f = L.first.get(key)!, c = L.count.get(key)!, s0 = src.srcs[0];
  if (!src.pack && !src.shift && s0 instanceof Float32Array && src.per[0] === comps && (f + c) * comps <= s0.length) return s0.subarray(f * comps, (f + c) * comps);
  if (scratch.length < c * comps) scratch = new Float32Array(Math.max(c * comps, scratch.length * 2));
  const out = scratch.subarray(0, c * comps);
  if (src.pack) src.pack(out, f, c);
  else if (src.shift) for (let i = 0; i < c; i++) out[i] = shifted(s0?.[f + i] ?? 0, shift);
  else { out.fill(0); if (s0) for (let i = 0; i < c * comps && (f * comps + i) < s0.length; i++) out[i] = s0[f * comps + i]; }
  return out;
}

/** whether key `key`'s values differ between `k`'s last write and `src` under `L` (`xs`, `ys`: the sources' words) */
function differs(k: SlotTex, src: SlotSrc, L: SlotLayout, key: number, shift: number, xs: (Uint32Array | null)[], ys: (Uint32Array | null)[]): boolean {
  const c = L.count.get(key)!, fo = k.layout.first.get(key)!, fn = L.first.get(key)!;
  const so = src.shift ? k.shift?.get(key) ?? 0 : 0;
  if (src.shift && so !== shift) {
    const a = k.srcs[0], b = src.srcs[0];
    if (!a || !b) return a !== b;
    for (let i = 0; i < c; i++) if (shifted(a[fo + i], so) !== shifted(b[fn + i], shift)) return true;
    return false;
  }
  for (let j = 0; j < src.srcs.length; j++) {
    const x = xs[j], y = ys[j];
    if (!x || !y) { if (!x !== !y) return true; continue; }
    if (k.srcs[j] === src.srcs[j] && fo === fn) continue;
    const p = src.per[j], lo = fo * p, ln = fn * p, n = c * p;
    if (lo + n > x.length || ln + n > y.length) return true;
    for (let i = 0; i < n; i++) if (x[lo + i] !== y[ln + i]) return true;
  }
  return false;
}

/** texels `a` to `a + c` of the bound texture (`width` a row) from `data` at `o` */
function upload(gl: WebGL2RenderingContext, kind: TexKind, width: number, a: number, c: number, data: Float32Array, o: number) {
  const fmt = FORMATS[kind][1], comps = compsOf(kind);
  let y = Math.floor(a / width);
  const x0 = a % width;
  if (x0 && c) {
    const w = Math.min(c, width - x0);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, x0, y, w, 1, fmt, gl.FLOAT, data, o);
    (o += w * comps), (c -= w), y++;
  }
  const h = Math.floor(c / width);
  if (h) {
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, y, width, h, fmt, gl.FLOAT, data, o);
    (o += h * width * comps), (c -= h * width), (y += h);
  }
  if (c) gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, y, c, 1, fmt, gl.FLOAT, data, o);
}

/**
 * Train table `k` (null: none yet) holding `src` in layout `L`'s slots: a key's slot written where it moved since `k`'s
 * last write, was blanked, or its values may differ from that write's: those `named` lists, where `k` was last written
 * from its `from`, else those found differing by comparing (`SlotTex.compared`); the whole table written
 * into a new texture when `k` cannot take `L` (too small, or last written under another fresh layout), or with
 * `?perfOff=keepGl`.
 */
export function putSlots(rnd: NetworkRenderer, k: SlotTex | null, kind: TexKind, width: number, L: SlotLayout, src: SlotSrc, named?: SlotNames | null): SlotTex {
  const gl = rnd.gl, comps = compsOf(kind);
  if (k && rnd.keepGl && k.kind === kind && k.layout === L && !k.dirty && k.srcs.length === src.srcs.length && k.srcs.every((s, i) => s === src.srcs[i])) {
    k.wrote = new Set();
    k.compared = false;
    return k;
  }
  const rows = Math.max(1, Math.ceil(L.cap / width));
  const fits = !!k && rnd.keepGl && k.kind === kind && k.rows >= rows;
  const whole = !fits || k!.layout.base !== L.base;
  if (!fits) {
    if (k) gl.deleteTexture(k.tex);
    const tex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, FORMATS[kind][0], width, rows, 0, FORMATS[kind][1], gl.FLOAT, null);
    rnd.gpuBytes.alloc += width * rows * comps * 4;
    k = { tex, kind, rows, layout: L, srcs: [], shift: null, dirty: null, wrote: "all" };
  }
  const t = k!;
  gl.bindTexture(gl.TEXTURE_2D, t.tex);
  gl.pixelStorei(gl.UNPACK_ALIGNMENT, 4);
  const shifts = src.shift ? new Map<number, number>() : null;
  if (whole) {
    // (one upload of every slot: the source itself where it is the table as laid out)
    const s0 = src.srcs[0];
    let all: Float32Array;
    if (L.packed && !src.pack && !src.shift && s0 instanceof Float32Array && src.per[0] === comps && s0.length >= L.end * comps) all = s0;
    else {
      all = new Float32Array(L.end * comps);
      for (const [key, a] of L.at) {
        const s = src.shift ? src.shift(key) : 0;
        shifts?.set(key, s);
        all.set(stage(src, L, key, comps, s), a * comps);
      }
    }
    upload(gl, kind, width, 0, L.end, all, 0);
    t.wrote = "all";
    t.compared = false;
  } else {
    const wrote = new Set<number>();
    const listed = named && named.from.length === t.srcs.length && named.from.every((v, i) => v === t.srcs[i]) ? named.keys : null;
    t.compared = !listed;
    const xs = listed ? [] : t.srcs.map((v) => (v ? words(v) : null)), ys = listed ? [] : src.srcs.map((v) => (v ? words(v) : null));
    for (const [key, a] of L.at) {
      const s = src.shift ? src.shift(key) : 0;
      shifts?.set(key, s);
      if (t.layout.at.get(key) === a && t.layout.count.get(key) === L.count.get(key) && !t.dirty?.has(key) && (listed ? !listed.has(key) : !differs(t, src, L, key, s, xs, ys))) continue;
      upload(gl, kind, width, a, L.count.get(key)!, stage(src, L, key, comps, s), 0);
      wrote.add(key);
    }
    t.wrote = wrote;
  }
  t.layout = L;
  t.srcs = src.srcs;
  t.shift = shifts;
  t.dirty = null;
  return t;
}

/** Keys `keys`' slots of table `k` (laid out by `L`, as its last write) filled with `value` until its next write. */
export function blankSlots(rnd: NetworkRenderer, k: SlotTex, width: number, L: SlotLayout, keys: Iterable<number>, value: number) {
  const gl = rnd.gl, comps = compsOf(k.kind);
  gl.bindTexture(gl.TEXTURE_2D, k.tex);
  gl.pixelStorei(gl.UNPACK_ALIGNMENT, 4);
  k.dirty ??= new Set();
  for (const key of keys) {
    const a = L.at.get(key), c = L.count.get(key);
    if (a === undefined || c === undefined) continue;
    if (scratch.length < c * comps) scratch = new Float32Array(Math.max(c * comps, scratch.length * 2));
    upload(gl, k.kind, width, a, c, scratch.subarray(0, c * comps).fill(value), 0);
    k.dirty.add(key);
  }
}

/** `?patch=check`: table `k` read back, each key's slot against its values in `src` (bit for bit); the keys that differ. */
export function checkSlots(rnd: NetworkRenderer, k: SlotTex, width: number, L: SlotLayout, src: SlotSrc): number[] {
  const gl = rnd.gl, comps = compsOf(k.kind), rows = Math.max(1, Math.ceil(L.end / width));
  const fb = gl.createFramebuffer(), prev = gl.getParameter(gl.READ_FRAMEBUFFER_BINDING);
  gl.bindFramebuffer(gl.READ_FRAMEBUFFER, fb);
  gl.framebufferTexture2D(gl.READ_FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, k.tex, 0);
  const bad: number[] = [];
  if (gl.checkFramebufferStatus(gl.READ_FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) bad.push(-1);
  else {
    const got = new Float32Array(width * rows * 4);
    gl.readPixels(0, 0, width, rows, gl.RGBA, gl.FLOAT, got);
    const g = new Uint32Array(got.buffer);
    for (const [key, a] of L.at) {
      const want = words(stage(src, L, key, comps, src.shift ? src.shift(key) : 0)), c = L.count.get(key)!;
      let off = false;
      for (let t = 0; t < c && !off; t++) for (let j = 0; j < comps; j++) if (g[(a + t) * 4 + j] !== want[t * comps + j]) { off = true; break; }
      if (off) bad.push(key);
    }
  }
  gl.bindFramebuffer(gl.READ_FRAMEBUFFER, prev);
  gl.deleteFramebuffer(fb);
  return bad;
}
