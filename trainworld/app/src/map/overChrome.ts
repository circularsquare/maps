// T-4009: the bars, buttons and panels are drawn inside the map's container, so the pointer moving
// onto one is no `mouseleave` of the container and the map's hover tags and tool ghosts stayed up. One
// place turns that into the leave every map hover already listens for, so none needs its own check.

import type { Map as MlMap } from "maplibre-gl";

/** interface the player operates, inside the container */
const CHROME = "button, input, select, label, a, [data-tip], .tool-bar, .tool-bar-pop, .blueprint-bar, .map-top-left, .map-sky, .map-zoom, .realm-buttons, .maplibregl-control-container";
/** map-owned marks and handles above the canvas, whose own tags show on purpose */
const MAP_OWNED = ".cap-mark, .touch-handle, .bp-box, .tool-marks, .stop-marks, .rev-marks, .cap-marks, .stn-labels, .demand-box, .airport-marks, .attraction-marks, .intercity-labels, .tool-tag";

export function startOverChrome(map: MlMap) {
  const box = map.getContainer();
  let over = false;
  box.addEventListener("mousemove", (e) => {
    const t = e.target as Element | null;
    const chrome = !!t?.closest?.(CHROME) && !t.closest(MAP_OWNED);
    if (chrome === over) return;
    over = chrome;
    if (chrome) box.dispatchEvent(new MouseEvent("mouseleave", { clientX: e.clientX, clientY: e.clientY }));
  }, true);
  box.addEventListener("mouseleave", (e) => { if (e.isTrusted) over = false; });
}
