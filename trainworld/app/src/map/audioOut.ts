// The game's one audio context (T-686, T-1282), shared by the station sounds and the building sounds.
// Browsers keep audio silent until the page has had a click or a key, so nothing plays before one.

import { display } from "../game/ui";
import { live } from "./live";

let ctx: AudioContext | null = null;
let out: DynamicsCompressorNode | null = null;
let failed = false;

/** Opening the audio device holds the page 60-210 ms (444 at 4x CPU throttling): not on the press
 * itself (T-1162), nor at the first sound, which came mid-drag (T-1207), but once the first click or
 * key is over, with no button held and the camera still, when the page is next idle. */
function open() {
  if (ctx || failed) return;
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

if (typeof window !== "undefined") {
  let held = 0, timer = 0;
  const idle = (f: () => void) => (typeof requestIdleCallback === "function" ? requestIdleCallback(f, { timeout: 3000 }) : setTimeout(f, 200));
  const later = () => {
    window.clearTimeout(timer);
    timer = window.setTimeout(() => idle(() => {
      if (ctx || failed || !display.peek().sound) return;
      // a drag or a camera move started meanwhile: wait for it to end
      if (held > 0 || live.map?.isMoving()) return later();
      open();
      if (ctx) for (const [type, f] of listeners) window.removeEventListener(type, f, true);
    }), 250);
  };
  const listeners: [string, (e: Event) => void][] = [
    ["pointerdown", () => { held++; window.clearTimeout(timer); }],
    ["pointerup", () => { held = Math.max(0, held - 1); later(); }],
    ["pointercancel", () => { held = Math.max(0, held - 1); later(); }],
    ["keyup", () => later()],
  ];
  for (const [type, f] of listeners) window.addEventListener(type, f, true);
}

/** The context and its output (a compressor), running; null before it is opened (above), where audio
 * is missing, or while the context is still resuming. */
export function audioOut(): { ctx: AudioContext; out: AudioNode } | null {
  if (!ctx || !out) return null;
  if (ctx.state === "suspended") void ctx.resume();
  return ctx.state === "running" ? { ctx, out } : null;
}
