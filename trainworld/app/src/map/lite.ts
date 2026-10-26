// The lighter map (T-1207): fewer basemap layers (map/basemap.ts `setBasemapLite`) and a pixel ratio
// of at most 1, for machines where panning runs slow. Settings has it on, off, or on by itself
// ("auto", the default): on once the median gap between MapLibre frames drawn while the camera moves,
// over the last SAMPLE such frames, is over SLOW_FRAME_MS. Measured afresh each page load; once on
// it stays on until the page is reloaded, so the map does not switch back and forth.

import { computed, effect, signal } from "@preact/signals";
import type { Map as MlMap } from "maplibre-gl";
import { clock } from "../game/clock";
import { display } from "../game/ui";
import { setBasemapLite } from "./basemap";
import type { Overlay } from "./overlay";

/** Under 25 frames a second while the camera moves: New York at 4x CPU throttling pans at about 34
 * (29 ms), at 6x about 16-24 (notes/T-1207.md). */
export const SLOW_FRAME_MS = 40;
const SAMPLE = 40;
/** A longer gap is a pause in the gesture (the button held, the mouse still), not a slow frame. */
const PAUSE_MS = 250;

/** Auto has found the frames slow this page load. */
export const liteDetected = signal(false);
export const liteOn = computed(() => display.value.lighterMap === "on" || (display.value.lighterMap === "auto" && liteDetected.value));

const median = (a: number[]) => { const s = [...a].sort((x, y) => x - y); return s[s.length >> 1]; };

/** Call once the map's style has loaded. */
export function startLite(map: MlMap, overlay: Overlay) {
  const gaps: number[] = [];
  overlay.onMovingFrame = (gap) => {
    // frames while a game loads are slow for other reasons
    if (liteDetected.peek() || display.peek().lighterMap !== "auto" || clock.held.peek() || gap > PAUSE_MS) return;
    gaps.push(gap);
    if (gaps.length > SAMPLE) gaps.shift();
    if (gaps.length === SAMPLE && median(gaps) > SLOW_FRAME_MS) liteDetected.value = true;
  };
  let ratio: number | null = null;
  effect(() => {
    const on = liteOn.value;
    setBasemapLite(map, on);
    const want = on ? Math.min(1, window.devicePixelRatio || 1) : null;
    if (want !== ratio) {
      ratio = want;
      // null goes back to the device's own ratio
      map.setPixelRatio(want as number);
    }
  });
}
