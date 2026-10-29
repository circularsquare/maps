// A stroke set's visible instances gathered into one small buffer set (T-1644, notes/T-1644.md). Drawing
// a set by ranges (strokeLod.ts) rebinds its nine attribute buffers for every range, in every pass, and
// zoomed in that is hundreds of ranges a pass. Gathered, a pass is one draw of a few thousand instances.

/** one stroke set's per-instance values, as its attribute buffers hold them; `wmul` and `full` null: all 1 and all 0 */
export interface StrokeCpu {
  seg: Float32Array;
  colour: Uint32Array;
  edge: Uint32Array;
  level: Float32Array;
  flags: Float32Array;
  speed: Float32Array;
  lane: Uint8Array | undefined;
  wmul: Float32Array | null;
  full: Float32Array | null;
}

const MIN_CAP = 4096;

export class StrokeGather {
  readonly vao: WebGLVertexArrayObject;
  /** instances in the buffers now */
  count = 0;
  readonly wmulBuf: WebGLBuffer;
  readonly fullBuf: WebGLBuffer;
  private bufs: WebGLBuffer[] = [];
  private cap = 0;
  private seg = new Float32Array(0);
  private colour = new Uint32Array(0);
  private edge = new Uint32Array(0);
  private level = new Float32Array(0);
  private flags = new Float32Array(0);
  private speed = new Float32Array(0);
  private lane = new Uint8Array(0);
  private wmul = new Float32Array(0);
  private full = new Float32Array(0);

  /** `loc`: the stroke program's attribute location by name (negative: unused) */
  constructor(private gl: WebGL2RenderingContext, loc: (name: string) => number) {
    this.vao = gl.createVertexArray()!;
    gl.bindVertexArray(this.vao);
    const attrib = (name: string, size: number, type: number, integer = false, norm = false) => {
      const l = loc(name), b = gl.createBuffer()!;
      this.bufs.push(b);
      if (l < 0) return b;
      gl.bindBuffer(gl.ARRAY_BUFFER, b);
      gl.enableVertexAttribArray(l);
      if (integer) gl.vertexAttribIPointer(l, size, type, 0, 0);
      else gl.vertexAttribPointer(l, size, type, norm, 0, 0);
      gl.vertexAttribDivisor(l, 1);
      return b;
    };
    attrib("a_seg", 4, gl.FLOAT);
    attrib("a_colour", 1, gl.UNSIGNED_INT, true);
    attrib("a_edge", 1, gl.UNSIGNED_INT, true);
    attrib("a_level", 1, gl.FLOAT);
    attrib("a_flags", 1, gl.FLOAT);
    attrib("a_speed", 1, gl.FLOAT);
    attrib("a_lane", 2, gl.UNSIGNED_BYTE, false, true);
    this.wmulBuf = attrib("a_wmul", 1, gl.FLOAT);
    this.fullBuf = attrib("a_full", 1, gl.FLOAT);
    gl.bindVertexArray(null);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
  }

  private grow(n: number) {
    const gl = this.gl;
    let cap = Math.max(MIN_CAP, this.cap);
    while (cap < n) cap *= 2;
    this.cap = cap;
    this.seg = new Float32Array(cap * 4);
    this.colour = new Uint32Array(cap);
    this.edge = new Uint32Array(cap);
    this.level = new Float32Array(cap);
    this.flags = new Float32Array(cap);
    this.speed = new Float32Array(cap);
    this.lane = new Uint8Array(cap * 2);
    this.wmul = new Float32Array(cap);
    this.full = new Float32Array(cap);
    const bytes = [16, 4, 4, 4, 4, 4, 2, 4, 4];
    this.bufs.forEach((b, i) => {
      gl.bindBuffer(gl.ARRAY_BUFFER, b);
      gl.bufferData(gl.ARRAY_BUFFER, cap * bytes[i], gl.DYNAMIC_DRAW);
    });
  }

  /** Gather the instances of `ranges` (start, count, ...; `total` of them) from `cpu` and upload them */
  fill(cpu: StrokeCpu, ranges: number[], total: number) {
    const gl = this.gl;
    if (total > this.cap) this.grow(total);
    const { seg, colour, edge, level, flags, speed, lane, wmul, full } = this;
    let at = 0;
    for (let r = 0; r < ranges.length; r += 2) {
      const start = ranges[r], n = ranges[r + 1];
      copy(seg, at * 4, cpu.seg, start * 4, n * 4);
      copy(colour, at, cpu.colour, start, n);
      copy(edge, at, cpu.edge, start, n);
      copy(level, at, cpu.level, start, n);
      copy(flags, at, cpu.flags, start, n);
      copy(speed, at, cpu.speed, start, n);
      if (cpu.lane) copy(lane, at * 2, cpu.lane, start * 2, n * 2);
      if (cpu.wmul) copy(wmul, at, cpu.wmul, start, n);
      if (cpu.full) copy(full, at, cpu.full, start, n);
      at += n;
    }
    if (!cpu.lane) lane.fill(255, 0, total * 2);
    if (!cpu.wmul) wmul.fill(1, 0, total);
    if (!cpu.full) full.fill(0, 0, total);
    const data: ArrayBufferView[] = [seg, colour, edge, level, flags, speed, lane, wmul, full];
    const per = [4, 1, 1, 1, 1, 1, 2, 1, 1];
    this.bufs.forEach((b, i) => {
      gl.bindBuffer(gl.ARRAY_BUFFER, b);
      gl.bufferSubData(gl.ARRAY_BUFFER, 0, data[i], 0, total * per[i]);
    });
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
    this.count = total;
  }

  dispose() {
    const gl = this.gl;
    gl.deleteVertexArray(this.vao);
    for (const b of this.bufs) gl.deleteBuffer(b);
    this.bufs = [];
    this.count = 0;
  }
}

type Typed = Float32Array | Uint32Array | Uint8Array;

/** `n` elements from `src` at `from` to `dst` at `at`; short runs by hand, since a subarray costs more than they do */
function copy(dst: Typed, at: number, src: Typed, from: number, n: number) {
  if (n > 48) (dst as Float32Array).set((src as Float32Array).subarray(from, from + n), at);
  else for (let i = 0; i < n; i++) dst[at + i] = src[from + i];
}
