// The game's one audio context (T-686, T-1282), shared by the station sounds and the building sounds.
// Browsers keep audio silent until the page has had a click or a key, so it is made at the first
// sound after one, and nothing plays before it.

let ctx: AudioContext | null = null;
let out: DynamicsCompressorNode | null = null;
let armed = false;
let failed = false;

if (typeof window !== "undefined") {
  const arm = () => {
    window.removeEventListener("pointerdown", arm, true);
    window.removeEventListener("keydown", arm, true);
    armed = true;
  };
  window.addEventListener("pointerdown", arm, true);
  window.addEventListener("keydown", arm, true);
}

/** The context and its output (a compressor), running; null before the page's first gesture, where
 * audio is missing, or while the context is still resuming. Made at the first sound, not on the press:
 * opening the audio device took 60-210 ms, a freeze on the page's first click (T-1162). */
export function audioOut(): { ctx: AudioContext; out: AudioNode } | null {
  if (!ctx && !failed && armed) {
    try {
      const C = window.AudioContext ?? (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
      if (!C) failed = true;
      else {
        ctx = new C();
        out = ctx.createDynamicsCompressor();
        out.connect(ctx.destination);
      }
    } catch {
      ctx = null;
      failed = true;
    }
  }
  if (!ctx || !out) return null;
  if (ctx.state === "suspended") void ctx.resume();
  return ctx.state === "running" ? { ctx, out } : null;
}
