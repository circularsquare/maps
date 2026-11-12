// T-3268: the plus and minus buttons at the map's right edge, one zoom step a click with the map's own easing.
// Settings > Map hides them (display.zoomButtons).
import { h, render } from "preact";
import type { Map as MapLibre } from "maplibre-gl";
import { display } from "../game/ui";
import { t } from "../i18n";
import { Icon } from "./widgets";

function ZoomButtons({ map }: { map: MapLibre }) {
  if (!display.value.zoomButtons) return null;
  return (
    <div class="map-zoom">
      <button class="btn icon" aria-label={t("mapZoom.in")} onClick={() => map.zoomIn()}><Icon.plus /></button>
      <button class="btn icon" aria-label={t("mapZoom.out")} onClick={() => map.zoomOut()}><Icon.minus /></button>
    </div>
  );
}

/** Mount the buttons in the map's container. */
export function startZoomButtons(map: MapLibre) {
  const root = document.createElement("div");
  map.getContainer().appendChild(root);
  render(h(ZoomButtons, { map }), root);
}
