// The renderer's stroke buffers and their level of detail, moved out of renderer.ts (T-2374).

import { type Project } from "./stationGeometry";
import { pxPerUnitAt } from "./geo";
import { chunkBoxes, mergeRanges, significance, simplifiedLater, visibleRanges, type ChunkBoxes } from "./strokeLod";
import { StrokeGather } from "./strokeGather";
import { type Strokes } from "./network";
import { BAND_STRIDE } from "./lineBands";
import { shownSpeed, StrokeVao, plainWmul, fullAttr, STROKE_ATTRIBS, LOD_TOP_ZOOM, GATHER_MAX, cullLevel, StrokeLodState, StrokeBand, LOD_TOL_PX, CULL_ZOOM, CULL_MARGIN_PX, RANGE_GAP } from "./rendererParts";
import type { NetworkRenderer } from "./renderer";

export function texture(rnd: NetworkRenderer, data: ArrayBufferView, kind: "r32f" | "rg32f" | "rgba32f" | "rgba8", width: number, list = rnd.res): WebGLTexture {
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
 * on the GPU from the old buffers, only the new segments uploaded. `more`: the `a_wmul` and `a_full`
 * values when worked out already, or "later" for an `a_full` written by its caller (T-2403). */
export function strokes(rnd: NetworkRenderer, s: Strokes, list = rnd.res, from?: { old: StrokeVao; runs: Int32Array } | null, more?: { wmul?: Float32Array | null; full?: Float32Array | "later" }): StrokeVao {
  const gl = rnd.gl;
  const vao = gl.createVertexArray()!;
  list.push({ del: () => gl.deleteVertexArray(vao) });
  gl.bindVertexArray(vao);
  const p = rnd.strokeProg;
  const n = s.count;
  const attribs: { buf: WebGLBuffer; bytes: number }[] = [];
  const patch = from && n && from.old.attribs?.length === STROKE_ATTRIBS.length ? from : null;
  /** an attribute's buffer: `get(start, count)` its values for those segments, `per` a segment */
  const make = (get: (start: number, count: number) => ArrayBufferView, per: number, bpe: number, empty: ArrayBufferView): WebGLBuffer => {
    const bytes = per * bpe;
    let buf: WebGLBuffer;
    if (!n) buf = buffer(rnd, empty, list);
    else if (!patch) buf = buffer(rnd, get(0, n), list);
    else {
      buf = gl.createBuffer()!;
      list.push({ del: () => gl.deleteBuffer(buf) });
      gl.bindBuffer(gl.COPY_WRITE_BUFFER, buf);
      gl.bufferData(gl.COPY_WRITE_BUFFER, n * bytes, gl.STATIC_DRAW);
      gl.bindBuffer(gl.COPY_READ_BUFFER, patch.old.attribs![attribs.length].buf);
      const r = patch.runs;
      for (let k = 0; k < r.length; k += 3) {
        if (r[k + 2] >= 0) gl.copyBufferSubData(gl.COPY_READ_BUFFER, gl.COPY_WRITE_BUFFER, r[k + 2] * bytes, r[k] * bytes, r[k + 1] * bytes);
        else gl.bufferSubData(gl.COPY_WRITE_BUFFER, r[k] * bytes, get(r[k], r[k + 1]));
      }
      gl.bindBuffer(gl.COPY_READ_BUFFER, null);
      gl.bindBuffer(gl.COPY_WRITE_BUFFER, null);
    }
    attribs.push({ buf, bytes });
    return buf;
  };
  const sub = <T extends Float32Array | Uint32Array | Uint8Array | Int16Array>(a: T, per: number) => (start: number, count: number) => a.subarray(start * per, (start + count) * per);
  const speed = s.speed;
  const speedOf = speed ? (start: number, count: number) => shownSpeed(speed.subarray(start, start + count), s.trainSpeed?.length === speed.length ? s.trainSpeed.subarray(start, start + count) : undefined) : (_: number, count: number) => new Float32Array(count).fill(160);
  attrib(rnd, p, "a_seg", make(sub(s.seg, 4), 4, 4, new Float32Array(4)), 4, gl.FLOAT, 1);
  attrib(rnd, p, "a_colour", make(sub(s.colour, 1), 1, 4, new Uint32Array(1)), 1, gl.UNSIGNED_INT, 1, true);
  attrib(rnd, p, "a_edge", make(sub(s.edge, 1), 1, 4, new Uint32Array(1)), 1, gl.UNSIGNED_INT, 1, true);
  attrib(rnd, p, "a_level", make(sub(s.level, 1), 1, 4, new Float32Array(1)), 1, gl.FLOAT, 1);
  attrib(rnd, p, "a_flags", make(sub(s.flags, 1), 1, 4, new Float32Array(1)), 1, gl.FLOAT, 1);
  attrib(rnd, p, "a_speed", make(speedOf, 1, 4, new Float32Array(1).fill(160)), 1, gl.FLOAT, 1);
  const wmulBuf = buffer(rnd, more?.wmul ?? plainWmul(Math.max(1, n)), list);
  attrib(rnd, p, "a_wmul", wmulBuf, 3, gl.FLOAT, 1);
  let fullBuf: WebGLBuffer;
  if (more?.full instanceof Float32Array) fullBuf = buffer(rnd, more.full, list);
  else if (more?.full === "later") {
    fullBuf = gl.createBuffer()!;
    list.push({ del: () => gl.deleteBuffer(fullBuf) });
    gl.bindBuffer(gl.ARRAY_BUFFER, fullBuf);
    gl.bufferData(gl.ARRAY_BUFFER, Math.max(1, n) * 8, gl.STATIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
  } else if (patch) {
    // as the attributes: a run's values depend on its own segments alone (`fullAttr`)
    fullBuf = gl.createBuffer()!;
    list.push({ del: () => gl.deleteBuffer(fullBuf) });
    gl.bindBuffer(gl.COPY_WRITE_BUFFER, fullBuf);
    gl.bufferData(gl.COPY_WRITE_BUFFER, n * 8, gl.STATIC_DRAW);
    gl.bindBuffer(gl.COPY_READ_BUFFER, patch.old.fullBuf);
    const r = patch.runs;
    for (let k = 0; k < r.length; k += 3) {
      if (r[k + 2] >= 0) gl.copyBufferSubData(gl.COPY_READ_BUFFER, gl.COPY_WRITE_BUFFER, r[k + 2] * 8, r[k] * 8, r[k + 1] * 8);
      else gl.bufferSubData(gl.COPY_WRITE_BUFFER, r[k] * 8, fullAttr(s, r[k + 1], undefined, r[k]));
    }
    gl.bindBuffer(gl.COPY_READ_BUFFER, null);
    gl.bindBuffer(gl.COPY_WRITE_BUFFER, null);
  } else fullBuf = buffer(rnd, fullAttr(s, n), list);
  attrib(rnd, p, "a_full", fullBuf, 2, gl.FLOAT, 1);
  const lane = s.lane;
  const laneBuf = make(lane ? sub(lane, 2) : (_, count) => new Uint8Array(count * 2).fill(255), 2, 1, new Uint8Array(2).fill(255));
  const loc = gl.getAttribLocation(p, "a_lane");
  if (loc >= 0) {
    gl.bindBuffer(gl.ARRAY_BUFFER, laneBuf);
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, 2, gl.UNSIGNED_BYTE, true, 0, 0);
    gl.vertexAttribDivisor(loc, 1);
  }
  // lines side by side (T-1799): none on track and the other stroke sets
  const band = s.band;
  const bandBuf = make(band ? sub(band, BAND_STRIDE) : (_, count) => new Int16Array(count * BAND_STRIDE), BAND_STRIDE, 2, new Int16Array(BAND_STRIDE));
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
  return { vao, count: n, wmulBuf, fullBuf, attribs, res: list };
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

/** `?patch=check` (T-2403): the track's `a_full` patched by runs against one worked out whole */
export function checkFull(rnd: NetworkRenderer, v: StrokeVao | null, s: Strokes) {
  if (!v || !s.count) return;
  const gl = rnd.gl, want = fullAttr(s, s.count), got = new Float32Array(want.length);
  gl.bindBuffer(gl.COPY_READ_BUFFER, v.fullBuf);
  gl.getBufferSubData(gl.COPY_READ_BUFFER, 0, got);
  gl.bindBuffer(gl.COPY_READ_BUFFER, null);
  const x = new Uint32Array(want.buffer), y = new Uint32Array(got.buffer);
  if (x.some((b, k) => b !== y[k])) console.error("T-2403 check: the track's a_full patched by runs differs from a whole one");
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
  if (vao === s) boxes = st.boxes ??= chunkBoxes(src.seg, src.count);
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

export function dropLod(rnd: NetworkRenderer) {
  for (const st of rnd.lodOf.values()) for (const b of st.bands.values()) for (const r of b.res) r.del();
  rnd.lodOf.clear();
  for (const g of rnd.gathers.values()) g.dispose();
  rnd.gathers.clear();
}

/** Rewrite a dynamic buffer in place (sizes never change between `setNetwork`s). */
export function rewrite(rnd: NetworkRenderer, buf: WebGLBuffer, data: Float32Array) {
  const gl = rnd.gl;
  gl.bindBuffer(gl.ARRAY_BUFFER, buf);
  gl.bufferSubData(gl.ARRAY_BUFFER, 0, data);
  gl.bindBuffer(gl.ARRAY_BUFFER, null);
}
