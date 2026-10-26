// The three building voices (T-1413), made with WebAudio and no files: "ke-cha" (a tick and a clack),
// "bung" (a plucked string) and "tsh" (a burst of filtered noise). Pure functions of a context, so
// the same code plays in the game and renders to WAV offline (checks/t1413-render.mjs).
// `level` is the master level the sound sits under (buildSound.ts).

export type Rnd = () => number;

const semis = (n: number) => 2 ** (n / 12);

const noiseCache = new WeakMap<BaseAudioContext, AudioBuffer>();
/** A quarter second of white noise, the same every time. */
function noise(ctx: BaseAudioContext): AudioBuffer {
  let b = noiseCache.get(ctx);
  if (!b) {
    b = ctx.createBuffer(1, Math.ceil(ctx.sampleRate * 0.25), ctx.sampleRate);
    const d = b.getChannelData(0);
    let s = 12345;
    for (let i = 0; i < d.length; i++) {
      s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
      d[i] = s / 2147483648 - 1;
    }
    noiseCache.set(ctx, b);
  }
  return b;
}

/** Noise through one filter and a gain that jumps up and dies away exponentially. */
function burst(ctx: BaseAudioContext, dest: AudioNode, t: number, o: {
  type: BiquadFilterType; f: number; fTo?: number; q: number; peak: number; attack: number; decay: number;
}) {
  const src = ctx.createBufferSource(), flt = ctx.createBiquadFilter(), g = ctx.createGain();
  src.buffer = noise(ctx);
  flt.type = o.type;
  flt.Q.value = o.q;
  flt.frequency.setValueAtTime(o.f, t);
  if (o.fTo) flt.frequency.exponentialRampToValueAtTime(o.fTo, t + o.decay);
  g.gain.setValueAtTime(0.0001, t);
  g.gain.exponentialRampToValueAtTime(o.peak, t + o.attack);
  g.gain.exponentialRampToValueAtTime(0.0001, t + o.decay);
  src.connect(flt).connect(g).connect(dest);
  src.start(t);
  src.stop(t + o.decay + 0.02);
}

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

/** Gap between the tick and the clack of "ke-cha", seconds. */
export const KECHA_GAP_S = 0.075;

/** "ke-cha", a blueprint built: a short bright tick (a burst of noise above 4 kHz, 14 ms, with a
 * 3.4 kHz click in it), then 75 ms on a lower, longer clack (noise around 1.1 kHz and a triangle
 * thunk dropping from 520 to 300 Hz, dying by 120 ms). The pair shifts up or down half a semitone at
 * random. */
export function keCha(ctx: BaseAudioContext, dest: AudioNode, t: number, level: number, rnd: Rnd = Math.random) {
  const k = semis((rnd() - 0.5));
  burst(ctx, dest, t, { type: "highpass", f: 4200 * k, q: 0.7, peak: 0.36 * level, attack: 0.001, decay: 0.014 });
  thunk(ctx, dest, t, { type: "sine", f: 3400 * k, fTo: 2800 * k, glide: 0.012, peak: 0.16 * level, decay: 0.016 });
  const c = t + KECHA_GAP_S;
  burst(ctx, dest, c, { type: "bandpass", f: 1100 * k, q: 1.2, peak: 0.45 * level, attack: 0.002, decay: 0.12 });
  thunk(ctx, dest, c, { type: "triangle", f: 520 * k, fTo: 300 * k, glide: 0.05, peak: 0.3 * level, decay: 0.12 });
}

/** "tsh", track placed: noise through a band that falls from 6 kHz to 3 kHz, 1.5 ms attack and a
 * quick fall to silence at 120 ms (40 dB down by about 65 ms). */
export function tsh(ctx: BaseAudioContext, dest: AudioNode, t: number, level: number) {
  burst(ctx, dest, t, { type: "bandpass", f: 6000, fTo: 3000, q: 0.8, peak: 0.34 * level, attack: 0.0015, decay: 0.12 });
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
