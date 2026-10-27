// The synthesised building voice (T-1413), made with WebAudio and no files: "bung" (a plucked string);
// and `sampleVoice`, which plays a recorded sound (T-1474). Pure functions of a context, so the same
// code plays in the game and renders to WAV offline (checks/t1413-render.mjs). `level` is the master
// level the sound sits under (buildSound.ts).

/** A sine or triangle that jumps up and dies away exponentially, its pitch gliding down. */
function thunk(ctx: BaseAudioContext, dest: AudioNode, t: number, o: {
  type: OscillatorType; f: number; fTo: number; glide: number; peak: number; decay: number;
}) {
  const osc = ctx.createOscillator(), g = ctx.createGain();
  osc.type = o.type;
  osc.frequency.setValueAtTime(o.f, t);
  osc.frequency.exponentialRampToValueAtTime(o.fTo, t + o.glide);
  g.gain.setValueAtTime(0.0001, t);
  g.gain.exponentialRampToValueAtTime(o.peak, t + 0.002);
  g.gain.exponentialRampToValueAtTime(0.0001, t + o.decay);
  osc.connect(g).connect(dest);
  osc.start(t);
  osc.stop(t + o.decay + 0.02);
}

/** A recorded sound (samples.ts) from `t`, scaled so its loudest sample is `peak`. */
export function sampleVoice(ctx: BaseAudioContext, dest: AudioNode, t: number, s: { buf: AudioBuffer; peak: number }, peak: number) {
  const src = ctx.createBufferSource(), g = ctx.createGain();
  src.buffer = s.buf;
  g.gain.value = peak / s.peak;
  src.connect(g).connect(dest);
  src.start(t);
}

/** The bung's string: Karplus-Strong at `f0`, a noise burst in a loop of one period that each pass
 * averages with its neighbour (so the highs die first), run to 0.4 s, under an envelope that is
 * silent by 0.38 s. Peak 1. */
function pluck(ctx: BaseAudioContext, f0: number): AudioBuffer {
  const sr = ctx.sampleRate, n = Math.round(sr * 0.4), p = Math.max(2, Math.round(sr / f0));
  const out = ctx.createBuffer(1, n, sr), y = out.getChannelData(0);
  const line = new Float32Array(p);
  let s = 777;
  for (let i = 0; i < p; i++) {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    line[i] = s / 2147483648 - 1;
  }
  // a softer pluck: the first pass of the excitation is smoothed twice
  for (let r = 0; r < 2; r++) { let prev = line[p - 1]; for (let i = 0; i < p; i++) { const c = line[i]; line[i] = 0.5 * (c + prev); prev = c; } }
  const decay = 0.995;
  let peak = 0;
  for (let i = 0; i < n; i++) {
    const j = i % p, nx = line[(j + 1) % p];
    y[i] = line[j];
    line[j] = decay * 0.5 * (line[j] + nx);
    const t = i / sr;
    y[i] *= Math.exp(-t / 0.11) * (t > 0.34 ? Math.max(0, (0.38 - t) / 0.04) : 1);
    peak = Math.max(peak, Math.abs(y[i]));
  }
  for (let i = 0; i < n; i++) y[i] /= peak;
  return out;
}

const pluckCache = new WeakMap<BaseAudioContext, AudioBuffer>();
/** The bung's pitch, A3. */
const BUNG_HZ = 220;

/** "bung", a station placed or an upgrade built: one plucked string at A3 (220 Hz, Karplus-Strong)
 * whose pitch starts 5% sharp and settles in 60 ms, through a 2.4 kHz low-pass for warmth, with a
 * sine body thump at 110 Hz that drops a few percent as it dies; the whole is gone by 0.4 s. */
export function bung(ctx: BaseAudioContext, dest: AudioNode, t: number, level: number) {
  let buf = pluckCache.get(ctx);
  if (!buf) pluckCache.set(ctx, buf = pluck(ctx, BUNG_HZ));
  const src = ctx.createBufferSource(), lp = ctx.createBiquadFilter(), g = ctx.createGain();
  src.buffer = buf;
  src.playbackRate.setValueAtTime(1.05, t);
  src.playbackRate.exponentialRampToValueAtTime(1, t + 0.06);
  lp.type = "lowpass";
  lp.frequency.value = 2400;
  lp.Q.value = 0.5;
  g.gain.value = 0.4 * level;
  src.connect(lp).connect(g).connect(dest);
  src.start(t);
  thunk(ctx, dest, t, { type: "sine", f: 115, fTo: 108, glide: 0.1, peak: 0.18 * level, decay: 0.16 });
}
