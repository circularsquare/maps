// The lighter map (T-1207): fewer basemap layers (map/basemap.ts `setBasemapLite`) and a pixel ratio
// of at most 1. Settings, Map detail: Low is on, High is off (the default). There was an Auto that turned
// Low on once frames ran slow; it is gone (T-2840): it tripped on a slow zoom just after a load, stayed on
// for the page, and Low saves no resident memory (notes/T-2840.md).

import { computed, effect } from "@preact/signals";
import type { Map as MlMap } from "maplibre-gl";
import { display } from "../game/ui";
import { setBasemapLite } from "./basemap";
import { cappedPixelRatio } from "../workers/deviceMemory";

export const liteOn = computed(() => display.value.lighterMap === "on");

/** Call once the map's style has loaded. */
export function startLite(map: MlMap) {
  // T-2794: a small device's capped ratio is its normal one
  const normal = cappedPixelRatio();
  let ratio: number | null = normal;
  effect(() => {
    const on = liteOn.value;
    setBasemapLite(map, on);
    const want = on ? Math.min(1, window.devicePixelRatio || 1) : normal;
    if (want !== ratio) {
      ratio = want;
      // null goes back to the device's own ratio
      map.setPixelRatio(want as number);
    }
  });
}
