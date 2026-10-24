// T-1200: the busy tag (ui/BusyTag.tsx). Work on the main thread that will freeze the page for a
// while is announced first: its words are shown, a frame is let paint them, then the work runs. Only
// work expected to take over BUSY_MS on this device is held back; how long a kind of work takes per
// unit is learnt from its last runs here, so a phone shows the tag where a fast desktop never does.

import { signal } from "@preact/signals";

/** Work expected to take longer than this, ms, shows the tag first. */
export const BUSY_MS = 150;

/** The words the tag shows, null when nothing long is running. */
export const busyText = signal<string | null>(null);

/** ms per unit of work by kind, measured on this device (half old, half new each run). */
const rates = new Map<string, number>();
/** shown texts by token, the newest shown */
const shown = new Map<number, string>();
let nextToken = 1;

/** How long work of `kind` over `units` should take here, ms; `guess` ms a unit until it has run once. */
export function expectedMs(kind: string, units: number, guess: number): number {
  return (rates.get(kind) ?? guess) * units;
}

/** Record that work of `kind` over `units` took `ms`. */
export function learnBusy(kind: string, units: number, ms: number) {
  if (!(units > 0) || !(ms >= 0)) return;
  const r = ms / units, old = rates.get(kind);
  rates.set(kind, old === undefined ? r : (old + r) / 2);
}

/** Show `text`; the token hides it again. */
export function showBusy(text: string): number {
  const t = nextToken++;
  shown.set(t, text);
  busyText.value = text;
  return t;
}

export function hideBusy(token: number) {
  if (!shown.delete(token)) return;
  busyText.value = shown.size ? [...shown.values()].at(-1)! : null;
}

/** `f` once the frame after now has been painted (a task after the next animation frame), or after
 * 500 ms at most should no frame come; a hidden page draws none, so its work goes at once. */
export function afterPaint(f: () => void) {
  let done = false;
  const go = () => { if (!done) { done = true; f(); } };
  if (typeof document !== "undefined" && document.visibilityState === "hidden") { setTimeout(go, 0); return; }
  requestAnimationFrame(() => { const ch = new MessageChannel(); ch.port1.onmessage = go; ch.port2.postMessage(0); });
  setTimeout(go, 500);
}

/**
 * Run `work` of `kind` over `units`: at once when it should be quick here, else with `text` shown
 * and painted first. `guess` is ms a unit before the kind has run on this device.
 */
export function runBusy(kind: string, text: string, units: number, guess: number, work: () => void) {
  const timed = () => { const t0 = performance.now(); try { work(); } finally { learnBusy(kind, units, performance.now() - t0); } };
  if (expectedMs(kind, units, guess) <= BUSY_MS) { timed(); return; }
  const token = showBusy(text);
  afterPaint(() => { try { timed(); } finally { hideBusy(token); } });
}
