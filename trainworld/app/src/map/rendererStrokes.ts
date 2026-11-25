// The renderer's stroke buffers and their level of detail, moved out of renderer.ts (T-2374).

import { type Project } from "./stationGeometry";
import { pxPerUnitAt } from "./geo";
import { chunkBoxes, mergeRanges, significance, simplifiedLater, visibleRanges, type ChunkBoxes } from "./strokeLod";
import { StrokeGather } from "./strokeGather";
import { type Strokes } from "./network";
import { BAND_STRIDE } from "./lineBands";
import { shownSpeed, StrokeVao, plainWmul, fullAttr, STROKE_ATTRIBS, LOD_TOP_ZOOM, GATHER_MAX, cullLevel, StrokeLodState, StrokeBand, LOD_TOL_PX, CULL_ZOOM, CULL_MARGIN_PX, RANGE_GAP } from "./rendererParts";
import type { NetworkRenderer } from "./renderer";

export type TexKind = "r32f" | "rg32f" | "rgba32f" | "rgba8";
const compsOf = (kind: TexKind) => (kind === "r32f" ? 1 : kind === "rg32f" ? 2 : 4);

export function texture(rnd: NetworkRenderer, data: ArrayBufferView, kind: TexKind, width: number, list = rnd.res): WebGLTexture {
  const gl = rnd.gl;
  const comps = kind === "r32f" ? 1 : kind === "rg32f" ? 2 : 4;
  const texels = Math.max(1, Math.ceil((data as Float32Array).length / comps));
  const height = Math.max(1, Math.ceil(texels / width));
  const padded = kind === "rgba8" ? new Uint8Array(width * height * 4) : new Float32Array(width * height * comps);
  padded.set(data as any);
  const tex = gl.createTexture()!;
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.pixelStorei(gl.UNPACK_ALIGNMENT, 4);
  if (kind === "r32f") gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, width, height, 0, gl.RED, gl.FLOAT, padded);
  else if (kind === "rg32f") gl.texImage2D(gl.TEXTURE_2D, 0, gl.RG32F, width, height, 0, gl.RG, gl.FLOAT, padded);
  else if (kind === "rgba32f") gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, width, height, 0, gl.RGBA, gl.FLOAT, padded);
  else gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, width, height, 0, gl.RGBA, gl.UNSIGNED_BYTE, padded);
  list.push({ del: () => gl.deleteTexture(tex) });
  return tex;
}

/** T-4262: a network texture kept from one network to the next, with room for more rows than it holds, and the
 * array last put in it (never written after: a later put compares against it). Texels past that array are 0. */
export interface KeptTex { tex: WebGLTexture; kind: TexKind; width: number; rows: number; data: ArrayBufferView }

const GL_FORMAT = { r32f: [0x822e, 0x1903, 0x1406], rg32f: [0x8230, 0x8227, 0x1406], rgba32f: [0x8814, 0x1908, 0x1406], rgba8: [0x8058, 0x1908, 0x1401] } as const;

/** a view's 32-bit words, or null where its bytes do not line up with them */
const wordsOf = (v: ArrayBufferView) => (v.byteOffset % 4 === 0 && v.byteLength % 4 === 0 ? new Uint32Array(v.buffer, v.byteOffset, v.byteLength / 4) : null);

/**
 * T-4262: the rows (start, count pairs, of `rowWords` 32-bit words each) where `b` differs from `a`, a word past
 * either's end read as 0; every row of either when their words cannot be compared.
 */
export function changedRows(a: ArrayBufferView, b: ArrayBufferView, rowWords: number): number[] {
  const x = wordsOf(a), y = wordsOf(b), out: number[] = [];
  const rows = Math.ceil(Math.max(a.byteLength, b.byteLength) / 4 / rowWords);
  if (!x || !y) return rows ? [0, rows] : [];
  const nx = x.length, ny = y.length;
  for (let r = 0; r < rows; r++) {
    const s = r * rowWords, e = s + rowWords;
    let differs = false;
    const both = Math.min(e, nx, ny);
    for (let i = s; i < both; i++) if (x[i] !== y[i]) { differs = true; break; }
    if (!differs) for (let i = Math.max(s, both); i < Math.min(e, Math.max(nx, ny)); i++) if ((i < nx ? x[i] : 0) !== (i < ny ? y[i] : 0)) { differs = true; break; }
    if (!differs) continue;
    const k = out.length;
    if (k && out[k - 2] + out[k - 1] === r) out[k - 1]++;
    else out.push(r, 1);
  }
  return out;
}

/**
 * Put `data` in network texture `k` (null: none yet), returning the texture that holds it: `k` itself when its room
 * and layout fit, with only the rows that differ from what it held written (`rows`, start and count pairs, when
 * the caller knows them; `data` the same array as last time: nothing changed), else a new one with room to grow.
 */
export function keepTexture(rnd: NetworkRenderer, k: KeptTex | null, data: ArrayBufferView, kind: TexKind, width: number, rows?: readonly number[]): KeptTex {
  const gl = rnd.gl, comps = compsOf(kind), len = (data as Float32Array).length;
  const need = Math.max(1, Math.ceil(Math.max(1, Math.ceil(len / comps)) / width));
  const [ifmt, fmt, type] = GL_FORMAT[kind];
  let spans: readonly number[];
  if (k && k.kind === kind && k.width === width && need <= k.rows && rnd.keepGl) {
    if (data === k.data && !rows) return k;
    // (a row of an rgba8 texture is `width` words, of a float one `width * comps`)
    spans = rows ?? changedRows(k.data, data, kind === "rgba8" ? width : width * comps);
  } else {
    const room = need + Math.max(4, need >> 3);
    const tex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, ifmt, width, room, 0, fmt, type, null);
    rnd.gpuBytes.alloc += width * room * comps * (kind === "rgba8" ? 1 : 4);
    k = { tex, kind, width, rows: room, data };
    spans = [0, need];
  }
  const rowLen = width * comps;
  gl.bindTexture(gl.TEXTURE_2D, k.tex);
  gl.pixelStorei(gl.UNPACK_ALIGNMENT, 4);
  const Ctor = (kind === "rgba8" ? Uint8Array : Float32Array) as unknown as new (n: number) => Float32Array;
  for (let s = 0; s + 1 < spans.length; s += 2) {
    const r0 = spans[s], r1 = Math.min(k.rows, spans[s] + spans[s + 1]);
    // whole rows of `data`, then its last row and any rows past it padded with 0
    const whole = Math.max(r0, Math.min(r1, Math.floor(len / rowLen)));
    if (whole > r0) gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, r0, width, whole - r0, fmt, type, data, r0 * rowLen);
    if (r1 > whole) {
      const pad = new Ctor((r1 - whole) * rowLen), from = whole * rowLen;
      if (from < len) pad.set((data as Float32Array).subarray(from, Math.min(len, r1 * rowLen)));
      gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, whole, width, r1 - whole, fmt, type, pad, 0);
    }
  }
  k.data = data;
  return k;
}

export function buffer(rnd: NetworkRenderer, data: ArrayBufferView, list = rnd.res): WebGLBuffer {
  const gl = rnd.gl;
  const b = gl.createBuffer()!;
  gl.bindBuffer(gl.ARRAY_BUFFER, b);
  gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
  // left bound, WebGL refuses it as a transform feedback target (the probe)
  gl.bindBuffer(gl.ARRAY_BUFFER, null);
  list.push({ del: () => gl.deleteBuffer(b) });
  return b;
}

export function attrib(rnd: NetworkRenderer, prog: WebGLProgram, name: string, buf: WebGLBuffer, size: number, type: number, divisor: number, integer = false) {
  const gl = rnd.gl;
  const loc = gl.getAttribLocation(prog, name);
  if (loc < 0) return;
  gl.bindBuffer(gl.ARRAY_BUFFER, buf);
  gl.enableVertexAttribArray(loc);
  if (integer) gl.vertexAttribIPointer(loc, size, type, 0, 0);
  else gl.vertexAttribPointer(loc, size, type, false, 0, 0);
  gl.vertexAttribDivisor(loc, divisor);
}

/** A stroke set's VAO. `from` (a network's strokes, T-1028): the strokes drawn before and the runs
 * (start, count, old start or -1, `NetPatch`) that make these from them: unchanged runs are copied
 * on the GPU from the old buffers, only the new segments uploaded. T-4262: where no run moved, the new
 * segments are written into the old buffers and the old VAO is returned; else the copies go into `spare`'s
 * buffers where they have room (a set no longer drawn, its VAO gone: those not taken are deleted). `more`: the
 * `a_wmul` values when worked out already, and the `a_full` values, or "later" for an `a_full` written by its
 * caller (T-2403); `fullKept`: the old buffers' `a_full` holds these values on the runs copied; `room` (a
 * network's set): new buffers made with room to grow, for a later set to copy into. */
export function strokes(rnd: NetworkRenderer, s: Strokes, list = rnd.res, from?: { old: StrokeVao; runs: Int32Array; spare?: StrokeVao | null } | null, more?: { wmul?: Float32Array | null; full?: Float32Array | "later"; fullKept?: boolean; room?: boolean }): StrokeVao {
  const gl = rnd.gl;
  const n = s.count;
  const patch = from && n && from.old.attribs?.length === STROKE_ATTRIBS.length ? from : null;
  const sub = <T extends Float32Array | Uint32Array | Uint8Array | Int16Array>(a: T, per: number) => (start: number, count: number) => a.subarray(start * per, (start + count) * per);
  const speed = s.speed, lane = s.lane, band = s.band;
  const speedOf = speed ? (start: number, count: number) => shownSpeed(speed.subarray(start, start + count), s.trainSpeed?.length === speed.length ? s.trainSpeed.subarray(start, start + count) : undefined) : (_: number, count: number) => new Float32Array(count).fill(160);
  /** per buffer: its values for segments (start, count) and its bytes a segment; the attributes in `STROKE_ATTRIBS`
   * order, then `a_wmul` (WMUL) and `a_full` (FULL) */
  const given = more?.full instanceof Float32Array ? more.full : null;
  const wmulPlain = !more?.wmul;
  const valuesOf: [(start: number, count: number) => ArrayBufferView, number][] = [
    [sub(s.seg, 4), 16], [sub(s.colour, 1), 4], [sub(s.edge, 1), 4], [sub(s.level, 1), 4], [sub(s.flags, 1), 4], [speedOf, 4],
    [lane ? sub(lane, 2) : (_, count) => new Uint8Array(count * 2).fill(255), 2],
    [band ? sub(band, BAND_STRIDE) : (_, count) => new Int16Array(count * BAND_STRIDE), BAND_STRIDE * 2],
    [more?.wmul ? sub(more.wmul, 3) : (_, count) => plainWmul(count), 12],
    [given ? sub(given, 2) : (start, count) => fullAttr(s, count, undefined, start), 8],
  ];
  /** the old `a_full` holds the values of the runs copied: the track's (geometry only) or a given one's that says so */
  const fullCopies = !more?.full || (!!given && !!more.fullKept);
  const bufOf = (v: StrokeVao, i: number) => (i < WMUL ? v.attribs![i].buf : i === WMUL ? v.wmulBuf : v.fullBuf);
  if (patch && rnd.keepGl && n === patch.old.count && !more?.wmul && inPlace(patch.runs)) {
    const v = patch.old, r = patch.runs;
    const put = (i: number, at: number, data: ArrayBufferView) => {
      gl.bindBuffer(gl.COPY_WRITE_BUFFER, bufOf(v, i));
      gl.bufferSubData(gl.COPY_WRITE_BUFFER, at * valuesOf[i][1], data);
    };
    for (let k = 0; k < r.length; k += 3) {
      if (r[k + 2] >= 0) continue;
      const a = r[k], c = r[k + 1];
      for (let i = 0; i < FULL; i++) put(i, a, valuesOf[i][0](a, c));
      if (fullCopies && more?.full !== "later") put(FULL, a, valuesOf[FULL][0](a, c));
      v.wmulHeld?.set(plainWmul(c), a * 3);
    }
    if (given && !fullCopies) put(FULL, 0, given.subarray(0, n * 2));
    gl.bindBuffer(gl.COPY_WRITE_BUFFER, null);
    return v;
  }
  const vao = gl.createVertexArray()!;
  list.push({ del: () => gl.deleteVertexArray(vao) });
  gl.bindVertexArray(vao);
  const p = rnd.strokeProg;
  const spare = rnd.keepGl && from?.spare?.caps ? from.spare : null, taken: boolean[] = [];
  const caps: number[] = [];
  /** buffer `i` for `n` segments, its contents to be written: a spare one with room, or a new one */
  const target = (i: number): WebGLBuffer => {
    const bytes = valuesOf[i][1], need = n * bytes;
    const sb = spare ? bufOf(spare, i) : null;
    if (sb && spare!.caps![i] >= need) {
      taken[i] = true;
      caps[i] = spare!.caps![i];
      list.push({ del: () => gl.deleteBuffer(sb) });
      gl.bindBuffer(gl.COPY_WRITE_BUFFER, sb);
      return sb;
    }
    const buf = gl.createBuffer()!;
    list.push({ del: () => gl.deleteBuffer(buf) });
    caps[i] = more?.room && rnd.keepGl ? (n + Math.max(n >> 3, 256)) * bytes : need;
    gl.bindBuffer(gl.COPY_WRITE_BUFFER, buf);
    gl.bufferData(gl.COPY_WRITE_BUFFER, caps[i], gl.STATIC_DRAW);
    return buf;
  };
  /** buffer `i`: unchanged runs copied from the old set's on the GPU, the rest uploaded; `whole` all uploaded */
  const make = (i: number, whole = false): WebGLBuffer => {
    const [get, bytes] = valuesOf[i];
    if (!n) {
      const empty = EMPTY[i]();
      caps[i] = empty.byteLength;
      return buffer(rnd, empty, list);
    }
    if (!patch || whole) {
      if (!more?.room || !rnd.keepGl) {
        caps[i] = n * bytes;
        return buffer(rnd, i === FULL && given ? given.subarray(0, n * 2) : get(0, n), list);
      }
      const buf = target(i);
      gl.bufferSubData(gl.COPY_WRITE_BUFFER, 0, i === FULL && given ? given.subarray(0, n * 2) : get(0, n));
      gl.bindBuffer(gl.COPY_WRITE_BUFFER, null);
      return buf;
    }
    const buf = target(i);
    gl.bindBuffer(gl.COPY_READ_BUFFER, bufOf(patch.old, i));
    const r = patch.runs;
    for (let k = 0; k < r.length; k += 3) {
      if (r[k + 2] >= 0) gl.copyBufferSubData(gl.COPY_READ_BUFFER, gl.COPY_WRITE_BUFFER, r[k + 2] * bytes, r[k] * bytes, r[k + 1] * bytes);
      else gl.bufferSubData(gl.COPY_WRITE_BUFFER, r[k] * bytes, get(r[k], r[k + 1]));
    }
    gl.bindBuffer(gl.COPY_READ_BUFFER, null);
    gl.bindBuffer(gl.COPY_WRITE_BUFFER, null);
    return buf;
  };
  const attribs: { buf: WebGLBuffer; bytes: number }[] = [];
  const attr = (i: number) => {
    const buf = make(i);
    attribs.push({ buf, bytes: valuesOf[i][1] });
    return buf;
  };
  attrib(rnd, p, "a_seg", attr(0), 4, gl.FLOAT, 1);
  attrib(rnd, p, "a_colour", attr(1), 1, gl.UNSIGNED_INT, 1, true);
  attrib(rnd, p, "a_edge", attr(2), 1, gl.UNSIGNED_INT, 1, true);
  attrib(rnd, p, "a_level", attr(3), 1, gl.FLOAT, 1);
  attrib(rnd, p, "a_flags", attr(4), 1, gl.FLOAT, 1);
  attrib(rnd, p, "a_speed", attr(5), 1, gl.FLOAT, 1);
  // T-4262: the widths of the runs kept copied from the old set (plain, or the demand view's it held), the new
  // segments' plain, rather than all uploaded again
  const wmulFrom = patch && rnd.keepGl && wmulPlain ? (patch.old.wmulPlain ? "plain" : patch.old.wmulHeld ? "held" : null) : null;
  const wmulBuf = make(WMUL, !wmulFrom);
  let wmulHeld: Float32Array | null = null;
  if (wmulFrom === "held") {
    const old = patch!.old.wmulHeld!, r = patch!.runs;
    wmulHeld = new Float32Array(n * 3);
    for (let k = 0; k < r.length; k += 3) wmulHeld.set(r[k + 2] >= 0 ? old.subarray(r[k + 2] * 3, (r[k + 2] + r[k + 1]) * 3) : plainWmul(r[k + 1]), r[k] * 3);
  }
  attrib(rnd, p, "a_wmul", wmulBuf, 3, gl.FLOAT, 1);
  let fullBuf: WebGLBuffer;
  if (more?.full === "later") {
    fullBuf = gl.createBuffer()!;
    list.push({ del: () => gl.deleteBuffer(fullBuf) });
    gl.bindBuffer(gl.ARRAY_BUFFER, fullBuf);
    gl.bufferData(gl.ARRAY_BUFFER, Math.max(1, n) * 8, gl.STATIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
    caps[FULL] = Math.max(1, n) * 8;
  } else fullBuf = make(FULL, !fullCopies);
  attrib(rnd, p, "a_full", fullBuf, 2, gl.FLOAT, 1);
  const laneBuf = attr(6);
  const loc = gl.getAttribLocation(p, "a_lane");
  if (loc >= 0) {
    gl.bindBuffer(gl.ARRAY_BUFFER, laneBuf);
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, 2, gl.UNSIGNED_BYTE, true, 0, 0);
    gl.vertexAttribDivisor(loc, 1);
  }
  // lines side by side (T-1799): none on track and the other stroke sets
  const bandBuf = attr(7);
  // the shifts and widths, then the metres to the ribbon's middle (T-2132), from one buffer
  for (const [name, size, off] of [["a_band", 4, 0], ["a_band_m", 2, 8]] as const) {
    const bloc = gl.getAttribLocation(p, name);
    if (bloc < 0) continue;
    gl.bindBuffer(gl.ARRAY_BUFFER, bandBuf);
    gl.enableVertexAttribArray(bloc);
    gl.vertexAttribPointer(bloc, size, gl.SHORT, true, BAND_STRIDE * 2, off);
    gl.vertexAttribDivisor(bloc, 1);
  }
  gl.bindVertexArray(null);
  gl.bindBuffer(gl.ARRAY_BUFFER, null);
  if (spare) for (let i = 0; i <= FULL; i++) if (!taken[i]) gl.deleteBuffer(bufOf(spare, i));
  return { vao, count: n, wmulBuf, fullBuf, attribs, res: list, wmulPlain: wmulFrom !== "held" && wmulPlain, wmulHeld, caps };
}

/** two sets of culling boxes alike, bit for bit */
function sameBoxes(a: ChunkBoxes, b: ChunkBoxes): boolean {
  if (a.length !== b.length) return false;
  for (let l = 0; l < a.length; l++) {
    const x = new Uint32Array(a[l].buffer, a[l].byteOffset, a[l].length), y = new Uint32Array(b[l].buffer, b[l].byteOffset, b[l].length);
    if (x.length !== y.length) return false;
    for (let i = 0; i < x.length; i++) if (x[i] !== y[i]) return false;
  }
  return true;
}

/** a stroke set's buffers after its attributes: `a_wmul`'s and `a_full`'s (`StrokeVao.caps`) */
const WMUL = STROKE_ATTRIBS.length, FULL = WMUL + 1;
/** each buffer of a set of no segments */
const EMPTY: (() => ArrayBufferView)[] = [() => new Float32Array(4), () => new Uint32Array(1), () => new Uint32Array(1), () => new Float32Array(1), () => new Float32Array(1), () => new Float32Array(1).fill(160), () => new Uint8Array(2).fill(255), () => new Int16Array(BAND_STRIDE), () => plainWmul(1), () => new Float32Array(2)];

/** T-4262: a network's stroke set no longer drawn kept as the next one's spare buffers (`strokes`): its VAO deleted */
export function spareOf(rnd: NetworkRenderer, v: StrokeVao): StrokeVao | null {
  rnd.gl.deleteVertexArray(v.vao);
  v.wmulHeld = null;
  if (v.caps?.length === FULL + 1 && rnd.keepGl) return v;
  for (let i = 0; i <= FULL; i++) rnd.gl.deleteBuffer(i < WMUL ? v.attribs![i].buf : i === WMUL ? v.wmulBuf : v.fullBuf);
  return null;
}

/** T-4262: spare buffers deleted */
export function dropSpare(rnd: NetworkRenderer, v: StrokeVao | null) {
  if (!v) return;
  for (let i = 0; i <= FULL; i++) rnd.gl.deleteBuffer(i < WMUL ? v.attribs![i].buf : i === WMUL ? v.wmulBuf : v.fullBuf);
}

/** T-4262: every run of a patch (start, count, old start or -1) that keeps segments keeps them where they were */
export function inPlace(runs: Int32Array): boolean {
  for (let k = 0; k < runs.length; k += 3) if (runs[k + 2] >= 0 && runs[k + 2] !== runs[k]) return false;
  return true;
}

/** T-4262: whether a patch's runs make every segment of the old set again, where it was */
export function unchangedRuns(runs: Int32Array, count: number): boolean {
  return count === 0 ? runs.length === 0 : runs.length === 3 && runs[0] === 0 && runs[1] === count && runs[2] === 0;
}

export function checkStrokes(rnd: NetworkRenderer, name: string, v: StrokeVao | null, s: Strokes) {
  if (!v?.attribs || !s.count) return;
  const gl = rnd.gl, n = s.count;
  const want: ArrayBufferView[] = [s.seg, s.colour, s.edge, s.level, s.flags, s.speed ? shownSpeed(s.speed, s.trainSpeed) : new Float32Array(n).fill(160), s.lane ?? new Uint8Array(n * 2).fill(255), s.band ?? new Int16Array(n * BAND_STRIDE)];
  const bad: string[] = [];
  v.attribs.forEach(({ buf, bytes }, i) => {
    const got = new Uint8Array(n * bytes), w = want[i];
    gl.bindBuffer(gl.COPY_READ_BUFFER, buf);
    gl.getBufferSubData(gl.COPY_READ_BUFFER, 0, got);
    const x = new Uint8Array(w.buffer, w.byteOffset, w.byteLength);
    if (x.length !== got.length || x.some((b, k) => b !== got[k])) bad.push(STROKE_ATTRIBS[i]);
  });
  gl.bindBuffer(gl.COPY_READ_BUFFER, null);
  if (bad.length) console.error(`T-1028 check: the ${name} strokes patched on the GPU differ from the network's in ${bad.join(", ")}`);
}

/** `?patch=check` (T-2403): a set's `a_full` patched by runs against one worked out whole (the track's), or against
 * `want` (T-4262: the lines', with their `a_wmul` against `wmul`) */
export function checkFull(rnd: NetworkRenderer, v: StrokeVao | null, s: Strokes, want = fullAttr(s, s.count), wmul?: Float32Array) {
  if (!v || !s.count) return;
  const gl = rnd.gl;
  const differs = (buf: WebGLBuffer, w: Float32Array) => {
    const got = new Float32Array(s.count * (w.length / Math.max(1, s.count)));
    gl.bindBuffer(gl.COPY_READ_BUFFER, buf);
    gl.getBufferSubData(gl.COPY_READ_BUFFER, 0, got);
    gl.bindBuffer(gl.COPY_READ_BUFFER, null);
    const x = new Uint32Array(w.buffer, w.byteOffset, got.length), y = new Uint32Array(got.buffer);
    return x.some((b, k) => b !== y[k]);
  };
  if (differs(v.fullBuf, want.subarray(0, s.count * 2))) console.error(`T-2403 check: the ${wmul ? "lines'" : "track's"} a_full patched by runs differs from a whole one`);
  if (wmul && differs(v.wmulBuf, wmul.subarray(0, s.count * 3))) console.error("T-4262 check: the lines' a_wmul after a network differs from the plain widths");
}

/** `?patch=check` (T-4262): each network texture kept from one network to the next read back and compared with the
 * array last put in it (0 past its end). Needs float colour buffers to read the float ones. */
export function checkTextures(rnd: NetworkRenderer): number {
  const gl = rnd.gl;
  const floats = !!gl.getExtension("EXT_color_buffer_float");
  const fb = gl.createFramebuffer(), prev = gl.getParameter(gl.READ_FRAMEBUFFER_BINDING);
  const bad: string[] = [];
  let read = 0;
  for (const [name, k] of Object.entries(rnd.keptTex)) {
    if (!k) continue;
    if (k.kind !== "rgba8" && !floats) { bad.push(`${name} (no float colour buffers to read it)`); continue; }
    gl.bindFramebuffer(gl.READ_FRAMEBUFFER, fb);
    gl.framebufferTexture2D(gl.READ_FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, k.tex, 0);
    if (gl.checkFramebufferStatus(gl.READ_FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) { bad.push(`${name} (not readable)`); continue; }
    read++;
    const comps = compsOf(k.kind), data = k.data as Float32Array, len = data.length;
    const rows = Math.min(k.rows, Math.ceil(Math.max(1, Math.ceil(len / comps)) / k.width) + 1);
    const byte = k.kind === "rgba8";
    const got = byte ? new Uint8Array(k.width * rows * 4) : new Float32Array(k.width * rows * 4);
    gl.readPixels(0, 0, k.width, rows, gl.RGBA, byte ? gl.UNSIGNED_BYTE : gl.FLOAT, got);
    // (floats compared bit for bit)
    const want = new Float32Array(1), w = new Uint32Array(want.buffer), g = byte ? got : new Uint32Array(got.buffer);
    let off = -1;
    for (let t = 0; t < k.width * rows && off < 0; t++) for (let c = 0; c < comps; c++) {
      const i = t * comps + c, v = i < len ? data[i] : 0;
      want[0] = v;
      if ((byte ? v : w[0]) !== g[t * 4 + c]) { off = i; break; }
    }
    if (off >= 0) bad.push(`${name} (from value ${off})`);
  }
  gl.bindFramebuffer(gl.READ_FRAMEBUFFER, prev);
  gl.deleteFramebuffer(fb);
  if (bad.length) console.error(`T-4262 check: the kept textures differ from what was put in them: ${bad.join(", ")}`);
  return read;
}

/** T-1639: what to draw of network stroke set `s` (made from `src`) this frame: its simplified copy
 * for the zoom band when that has fewer segments, and the instance ranges near the view (null: all). */
export function pickStrokes(rnd: NetworkRenderer, s: StrokeVao | null, src: Strokes, lines: boolean, zoom: number, dpr: number, grow: number, project: Project, w: number, h: number): { vao: StrokeVao; ranges: number[] | null } | null {
  if (!s || !s.count) return null;
  let st = rnd.lodOf.get(s);
  if (!st) rnd.lodOf.set(s, (st = { boxes: null, speed: undefined, sig: null, sigAt: 0, working: false, bands: new Map(), making: null }));
  let vao = s, boxes = st.boxes, used: StrokeBand | null = null;
  const bz = Math.ceil(zoom + Math.log2(dpr) - 1e-6);
  if (rnd.strokeLod && grow === 0 && bz < LOD_TOP_ZOOM && (st.sig && st.sigAt >= src.count || workLod(rnd, s, st, src))) {
    // the full set is drawn until the band is made (T-2403)
    const band = st.bands.get(bz) ?? workBand(rnd, s, st, src, bz, lines);
    if (band && band.vao.count < s.count) {
      if (lines && band.v !== rnd.perSegV) {
        band.v = rnd.perSegV;
        band.wm = bandWmul(rnd, band.first);
        band.fu = bandFull(rnd, band.strokes, band.first, 0, band.first.length, rnd.linesFull);
        rewrite(rnd, band.vao.wmulBuf, band.wm ?? plainWmul(band.first.length));
        rewrite(rnd, band.vao.fullBuf, band.fu);
      }
      vao = band.vao;
      boxes = band.boxes;
      used = band;
    }
  }
  if (!rnd.strokeCull || zoom < CULL_ZOOM) return { vao, ranges: null };
  if (vao === s && !st.boxes) {
    st.boxes = chunkBoxes(src.seg, src.count, s.boxesFrom);
    if (s.boxesFrom && rnd.checkPatch && !sameBoxes(st.boxes, chunkBoxes(src.seg, src.count))) console.error("T-4262 check: the culling boxes kept from the last set differ from whole ones");
    s.boxesFrom = undefined;
  }
  if (vao === s) boxes = st.boxes;
  const ranges = visibleRanges(boxes!, vao.count, project, w, h, CULL_MARGIN_PX, cullLevel(zoom));
  if (!ranges) return { vao, ranges: null };
  let total = 0;
  for (let i = 1; i < ranges.length; i += 2) total += ranges[i];
  if (!total) return { vao, ranges };
  if (!rnd.strokeGather || total > GATHER_MAX) return { vao, ranges: mergeRanges(ranges, RANGE_GAP) };
  let g = rnd.gathers.get(s);
  if (!g) rnd.gathers.set(s, (g = new StrokeGather(rnd.gl, (name) => rnd.strokeLoc[name])));
  const b = used?.strokes ?? src;
  g.fill({ seg: b.seg, colour: b.colour, edge: b.edge, level: b.level, flags: b.flags, speed: shownOf(rnd, b), lane: b.lane, band: b.band, wmul: lines ? (used ? used.wm : rnd.linesWmul) : null, full: lines ? (used ? used.fu : rnd.linesFull) : null }, ranges, total);
  return { vao: g, ranges: null };
}

/** a stroke set's speeds as its buffer holds them, kept per set */
export function shownOf(rnd: NetworkRenderer, s: Strokes): Float32Array {
  let v = rnd.speedShown.get(s);
  if (!v) rnd.speedShown.set(s, (v = s.speed ? shownSpeed(s.speed, s.trainSpeed?.length === s.speed.length ? s.trainSpeed : undefined) : new Float32Array(s.count).fill(160)));
  return v;
}

/** A line band's `a_wmul`: a merged segment's width and shift at its start are its first segment's,
 * at its end its last's; null for plain widths. */
export function bandWmul(rnd: NetworkRenderer, f: Uint32Array): Float32Array | null {
  const wm = rnd.linesWmul;
  if (!wm) return null;
  const m = f.length, all = wm.length / 3, out = new Float32Array(m * 3);
  for (let k = 0; k < m; k++) {
    const i = f[k], last = (k + 1 < m ? f[k + 1] : all) - 1;
    out[k * 3] = wm[i * 3];
    out[k * 3 + 1] = wm[i * 3 + 1];
    out[k * 3 + 2] = wm[last * 3 + 2];
  }
  return out;
}

/** A band's `a_full` for merged segments `from` to `to` into `out`, the traffic marks from `fu`
 * (the full set's) when given; `out` made when absent. */
export function bandFull(_rnd: NetworkRenderer, b: Strokes, f: Uint32Array, from: number, to: number, fu: Float32Array | null, out?: Float32Array): Float32Array {
  const part = fullAttr(b, to - from, fu ? (k) => fu[f[k] * 2] : undefined, from);
  if (!out) {
    if (from === 0 && to === f.length) return part;
    out = new Float32Array(Math.max(1, f.length) * 2);
  }
  out.set(part.subarray(0, (to - from) * 2), from * 2);
  return out;
}

/** T-2403: a zoom band's simplified copy made in 8 ms slices between frames after the significance
 * (`workLod`), one band at a time a set; null until it is made. `fullAttr` starts a run at each
 * slice, so the slices of `a_full` end only where a run of the band starts. */
export function workBand(rnd: NetworkRenderer, s: StrokeVao, st: StrokeLodState, src: Strokes, bz: number, lines: boolean): StrokeBand | null {
  if (st.making !== null) return null;
  st.making = bz;
  const t0 = performance.now();
  const job = simplifiedLater(src, st.sig!, LOD_TOL_PX / pxPerUnitAt(bz), st.speed ?? null);
  const m = job.strokes.count, b = job.strokes, f = job.first;
  const SLICE = 16384;
  let phase = 0, at = 0, boxes: ChunkBoxes | null = null, v = -1, wm: Float32Array | null = null, fuSrc: Float32Array | null = null, full: Float32Array | null = null;
  // where a run of the band's own starts (`fullAttr`): slices of `a_full` end only there
  const runEnd = (k: number) => {
    while (k < m && k > 0 && b.colour[k] === b.colour[k - 1] && b.edge[k] === b.edge[k - 1] && b.seg[k * 4] === b.seg[k * 4 - 2] && b.seg[k * 4 + 1] === b.seg[k * 4 - 1]) k++;
    return Math.min(k, m);
  };
  const step = (): boolean => {
    if (phase === 0) {
      job.fill(at, at + SLICE);
      at += SLICE;
      if (at >= m) phase = 1, at = 0;
    } else if (phase === 1) {
      boxes = chunkBoxes(b.seg, m);
      if (lines) { v = rnd.perSegV; wm = bandWmul(rnd, f); fuSrc = rnd.linesFull; }
      full = new Float32Array(Math.max(1, m) * 2);
      phase = 2;
    } else if (phase === 2) {
      const to = runEnd(Math.min(m, at + SLICE));
      bandFull(rnd, b, f, at, to, lines ? fuSrc : null, full!);
      at = to;
      if (at >= m) phase = 3;
    } else {
      const res: { del: () => void }[] = [];
      const vao = strokes(rnd, b, res, null, { wmul: wm, full: full! });
      st.bands.set(bz, { vao, strokes: b, first: f, boxes: boxes!, v, res, wm: lines ? wm : null, fu: lines ? full : null });
      st.making = null;
      return true;
    }
    return false;
  };
  const run = () => {
    if (rnd.lodOf.get(s) !== st) return;
    const t1 = performance.now();
    let done = false;
    while (!done && performance.now() - t1 < 8) done = step();
    rnd.lodMs += performance.now() - t1;
    if (!done) setTimeout(run, 0);
  };
  rnd.lodMs += performance.now() - t0;
  setTimeout(run, 0);
  return null;
}

/** T-1639: the runs' significance worked out in 8 ms slices between frames, so an edit's first
 * draw (the full set, as before) is not held up by it (~80 ms a set on the all-Japan save); false
 * until it is done. */
export function workLod(rnd: NetworkRenderer, s: StrokeVao, st: StrokeLodState, src: Strokes): false {
  if (st.working) return false;
  st.working = true;
  const step = () => {
    if (rnd.lodOf.get(s) !== st) return;
    const t0 = performance.now();
    if (st.speed === undefined) st.speed = src.speed ? shownSpeed(src.speed, src.trainSpeed) : null;
    st.sig ??= new Float32Array(src.count);
    st.sigAt = significance(src, st.speed, st.sig, st.sigAt, t0 + 8);
    rnd.lodMs += performance.now() - t0;
    // the picture drawn meanwhile is the same, so nothing is redrawn when done: the next
    // stroke redraw takes the copy
    if (st.sigAt < src.count) setTimeout(step, 0);
    else st.working = false;
  };
  setTimeout(step, 0);
  return false;
}

/** Point stroke set `v`'s attributes (its VAO bound) at instance `start`: WebGL2 draws have no
 * first instance. */
export function bindAt(rnd: NetworkRenderer, v: StrokeVao, start: number) {
  const gl = rnd.gl, a = v.attribs!;
  const set = (name: string, buf: WebGLBuffer, size: number, type: number, bytes: number, int = false, norm = false, stride = 0, off = 0) => {
    const loc = rnd.strokeLoc[name];
    if (loc < 0) return;
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    if (int) gl.vertexAttribIPointer(loc, size, type, stride, start * bytes + off);
    else gl.vertexAttribPointer(loc, size, type, norm, stride, start * bytes + off);
  };
  set("a_seg", a[0].buf, 4, gl.FLOAT, 16);
  set("a_colour", a[1].buf, 1, gl.UNSIGNED_INT, 4, true);
  set("a_edge", a[2].buf, 1, gl.UNSIGNED_INT, 4, true);
  set("a_level", a[3].buf, 1, gl.FLOAT, 4);
  set("a_flags", a[4].buf, 1, gl.FLOAT, 4);
  set("a_speed", a[5].buf, 1, gl.FLOAT, 4);
  set("a_lane", a[6].buf, 2, gl.UNSIGNED_BYTE, 2, false, true);
  set("a_band", a[7].buf, 4, gl.SHORT, BAND_STRIDE * 2, false, true, BAND_STRIDE * 2);
  set("a_band_m", a[7].buf, 2, gl.SHORT, BAND_STRIDE * 2, false, true, BAND_STRIDE * 2, 8);
  set("a_wmul", v.wmulBuf, 3, gl.FLOAT, 12);
  set("a_full", v.fullBuf, 2, gl.FLOAT, 8);
  gl.bindBuffer(gl.ARRAY_BUFFER, null);
}

/** T-4262: stroke set `v`'s simplified copies, culling boxes and gathered instances dropped */
export function dropLodOf(rnd: NetworkRenderer, v: StrokeVao) {
  const st = rnd.lodOf.get(v);
  if (st) for (const b of st.bands.values()) for (const r of b.res) r.del();
  rnd.lodOf.delete(v);
  rnd.gathers.get(v)?.dispose();
  rnd.gathers.delete(v);
}

/** T-4262: row spans (start, count pairs, each list in order) joined into one list in order */
export function joinRows(lists: readonly (readonly number[])[]): number[] {
  const all: [number, number][] = [];
  for (const l of lists) for (let i = 0; i + 1 < l.length; i += 2) all.push([l[i], l[i] + l[i + 1]]);
  all.sort((a, b) => a[0] - b[0]);
  const out: number[] = [];
  for (const [s, e] of all) {
    const k = out.length;
    if (k && out[k - 2] + out[k - 1] >= s) out[k - 1] = Math.max(out[k - 2] + out[k - 1], e) - out[k - 2];
    else out.push(s, e - s);
  }
  return out;
}

/** Rewrite a dynamic buffer in place (sizes never change between `setNetwork`s). */
export function rewrite(rnd: NetworkRenderer, buf: WebGLBuffer, data: Float32Array) {
  const gl = rnd.gl;
  gl.bindBuffer(gl.ARRAY_BUFFER, buf);
  gl.bufferSubData(gl.ARRAY_BUFFER, 0, data);
  gl.bindBuffer(gl.ARRAY_BUFFER, null);
}
