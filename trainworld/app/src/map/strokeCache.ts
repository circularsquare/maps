// The overlay's slow-changing passes kept in textures between frames (T-1638, notes/T-1618.md): the
// track and line strokes are millions of instances, and while the clock runs only the trains move.
// Each layer is drawn into an offscreen target with the overlay canvas's own multisampling and
// stencil, resolved into a texture, and copied onto the canvas each frame with the same
// premultiplied blend the strokes use, so the picture is the one drawn directly.

import { compile } from "./gl";

const VS = `#version 300 es
void main() {
  vec2 p = vec2(float((gl_VertexID & 1) << 2) - 1.0, float((gl_VertexID & 2) << 1) - 1.0);
  gl_Position = vec4(p, 0.0, 1.0);
}`;
const FS = `#version 300 es
precision highp float;
uniform sampler2D u_tex;
out vec4 o;
void main() {
  o = texelFetch(u_tex, ivec2(gl_FragCoord.xy), 0);
}`;
/** a texture unit none of the renderer's passes use */
const UNIT = 12;

export class StrokeCache {
  private prog: WebGLProgram;
  private uTex: WebGLUniformLocation | null;
  private samples: number;
  /** where layers are drawn: multisampled renderbuffers, or (without MSAA) each layer's texture */
  private drawFb: WebGLFramebuffer[] = [];
  private resolveFb: WebGLFramebuffer[] = [];
  private tex: WebGLTexture[] = [];
  private rbs: WebGLRenderbuffer[] = [];
  private w = 0;
  private h = 0;
  /** what the textures were drawn with; NaN-free numbers compared one by one */
  private key: number[] = [];
  private keyObj: unknown = null;
  /** times the layers were drawn (checks read it) */
  redraws = 0;

  constructor(private gl: WebGL2RenderingContext, private layers: number) {
    this.prog = compile(gl, VS, FS);
    this.uTex = gl.getUniformLocation(this.prog, "u_tex");
    this.samples = Math.min(gl.getParameter(gl.SAMPLES) ?? 0, gl.getParameter(gl.MAX_SAMPLES) ?? 0);
  }

  /** The layers need drawing again: drawn with another `key` (numbers) or `obj` (by identity), at
   * another size, or never. Takes the new key. */
  stale(w: number, h: number, key: number[], obj: unknown): boolean {
    const k = this.key;
    let same = w === this.w && h === this.h && obj === this.keyObj && k.length === key.length;
    for (let i = 0; same && i < key.length; i++) same = k[i] === key[i];
    if (same) return false;
    if (w !== this.w || h !== this.h) this.allocate(w, h);
    this.key = key;
    this.keyObj = obj;
    return true;
  }

  /** Forget what was drawn: the next `stale` is true. */
  clear() {
    this.key = [];
  }

  private allocate(w: number, h: number) {
    const gl = this.gl;
    this.free();
    this.w = w;
    this.h = h;
    const rb = (format: number) => {
      const r = gl.createRenderbuffer()!;
      gl.bindRenderbuffer(gl.RENDERBUFFER, r);
      if (this.samples > 1) gl.renderbufferStorageMultisample(gl.RENDERBUFFER, this.samples, format, w, h);
      else gl.renderbufferStorage(gl.RENDERBUFFER, format, w, h);
      this.rbs.push(r);
      return r;
    };
    const depth = rb(gl.DEPTH24_STENCIL8);
    const colour = this.samples > 1 ? rb(gl.RGBA8) : null;
    gl.bindRenderbuffer(gl.RENDERBUFFER, null);
    if (colour) {
      const fb = gl.createFramebuffer()!;
      gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
      gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.RENDERBUFFER, colour);
      gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_STENCIL_ATTACHMENT, gl.RENDERBUFFER, depth);
      this.drawFb.push(fb);
    }
    for (let i = 0; i < this.layers; i++) {
      const t = gl.createTexture()!;
      gl.bindTexture(gl.TEXTURE_2D, t);
      gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RGBA8, w, h);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      this.tex.push(t);
      const fb = gl.createFramebuffer()!;
      gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, t, 0);
      if (!colour) {
        gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_STENCIL_ATTACHMENT, gl.RENDERBUFFER, depth);
        this.drawFb.push(fb);
      }
      this.resolveFb.push(fb);
    }
    gl.bindTexture(gl.TEXTURE_2D, null);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  private free() {
    const gl = this.gl;
    for (const f of new Set([...this.drawFb, ...this.resolveFb])) gl.deleteFramebuffer(f);
    for (const t of this.tex) gl.deleteTexture(t);
    for (const r of this.rbs) gl.deleteRenderbuffer(r);
    this.drawFb = [];
    this.resolveFb = [];
    this.tex = [];
    this.rbs = [];
    this.w = this.h = 0;
  }

  /** Draw layer `i` from here until `end`: its target bound and cleared, the viewport its size. */
  begin(i: number) {
    const gl = this.gl;
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.drawFb[this.samples > 1 ? 0 : i]);
    gl.viewport(0, 0, this.w, this.h);
    gl.clearColor(0, 0, 0, 0);
    gl.clearStencil(0);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.STENCIL_BUFFER_BIT);
    this.redraws += i === 0 ? 1 : 0;
  }

  /** Layer `i` is drawn: resolved into its texture, and the canvas bound again with `vp` as viewport. */
  end(i: number, vp: readonly [number, number]) {
    const gl = this.gl;
    if (this.samples > 1) {
      gl.bindFramebuffer(gl.READ_FRAMEBUFFER, this.drawFb[0]);
      gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, this.resolveFb[i]);
      gl.blitFramebuffer(0, 0, this.w, this.h, 0, 0, this.w, this.h, gl.COLOR_BUFFER_BIT, gl.NEAREST);
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, vp[0], vp[1]);
  }

  /** Layer `i` over what is drawn, as the strokes blend (premultiplied); stencil and blend as set. */
  draw(i: number) {
    const gl = this.gl;
    gl.useProgram(this.prog);
    gl.activeTexture(gl.TEXTURE0 + UNIT);
    gl.bindTexture(gl.TEXTURE_2D, this.tex[i]);
    gl.uniform1i(this.uTex, UNIT);
    gl.bindVertexArray(null);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    gl.activeTexture(gl.TEXTURE0);
  }
}
