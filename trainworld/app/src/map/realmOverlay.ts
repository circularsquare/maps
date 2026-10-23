// T-975: the realms not shown (game/realms.ts), drawn as one dark fill over their land, under labels
// and the network and taking no clicks, with one button each at its label point that shows it. A
// button whose label point is off screen sits inside the on-screen part of its realm's box (as the
// loading bars do, T-939); buttons that would overlap give way to the realm with more zones; zoomed
// out to the world the buttons are the eye alone, the name in the hover.

import { effect } from "@preact/signals";
import { h, render } from "preact";
import type { FeatureCollection, MultiPolygon } from "geojson";
import type { GeoJSONSource, Map as MlMap } from "maplibre-gl";
import { choosing } from "../game/loading";
import { followStations, inactiveRealms, loadRealmLand, realmName, showRealm, type RealmLand } from "../game/realms";
import { SHADES } from "../game/shadeTable";
import { shade } from "../game/ui";
import { Icon } from "../ui/widgets";

const FILL = "tw-realms-inactive";
/** how far a button keeps from the map's edges, px */
const EDGE = 48;
/** below this zoom a button is its icon alone */
const ICON_ZOOM = 3;
const GAP = 6;

export function startRealmOverlay(map: MlMap): () => void {
  const stopStations = followStations();
  void loadRealmLand();
  const layer = document.createElement("div");
  layer.className = "realm-buttons";
  map.getContainer().appendChild(layer);
  const buttons = new Map<string, { el: HTMLDivElement; compact: boolean | null; w: number; h: number }>();
  let shown: RealmLand[] = [];

  const data = (): FeatureCollection<MultiPolygon> => ({ type: "FeatureCollection", features: shown.map((l) => l.feature) });

  const button = (l: RealmLand, compact: boolean) => {
    let b = buttons.get(l.id);
    if (!b) {
      const el = document.createElement("div");
      el.className = "realm-button";
      layer.appendChild(el);
      buttons.set(l.id, (b = { el, compact: null, w: 0, h: 0 }));
    }
    if (b.compact !== compact) {
      const name = realmName(l.id);
      render(h("button", {
        type: "button", class: `btn ${compact ? "icon" : "text"}`, "aria-label": `Show ${name}`,
        "data-tip": compact ? `${name}\nClick to show its cities` : "Click to show its cities",
        onClick: () => showRealm(l.id),
      }, h(Icon.eye, null), compact ? null : h("span", null, name)), b.el);
      b.compact = compact;
      b.w = 0;
    }
    // measured once drawn (zero while the layer is hidden behind the start menu)
    if (!b.w) { b.el.style.display = ""; b.w = b.el.offsetWidth; b.h = b.el.offsetHeight; }
    return b;
  };

  const place = () => {
    const c = map.getContainer(), W = c.clientWidth, H = c.clientHeight;
    const compact = map.getZoom() < ICON_ZOOM;
    const used: [number, number, number, number][] = [];
    // the realm with more zones first, so it keeps its place when two would overlap
    for (const l of [...shown].sort((a, b) => b.zones - a.zones)) {
      const b = button(l, compact);
      const [x0, y0, x1, y1] = l.box;
      const corners = [[x0, y0], [x1, y0], [x0, y1], [x1, y1]].map(([x, y]) => map.project([x, y]));
      const left = Math.max(EDGE, Math.min(...corners.map((p) => p.x)));
      const right = Math.min(W - EDGE, Math.max(...corners.map((p) => p.x)));
      const top = Math.max(EDGE, Math.min(...corners.map((p) => p.y)));
      const bottom = Math.min(H - EDGE, Math.max(...corners.map((p) => p.y)));
      const p = map.project(l.label);
      // centred on the point, the whole button kept on the map
      const x = Math.max(GAP, Math.min(W - b.w - GAP, Math.min(right, Math.max(left, p.x)) - b.w / 2));
      const y = Math.max(GAP, Math.min(H - b.h - GAP, Math.min(bottom, Math.max(top, p.y)) - b.h / 2));
      const box: [number, number, number, number] = [x - GAP, y - GAP, x + b.w + GAP, y + b.h + GAP];
      if (left > right || top > bottom || used.some((u) => box[0] < u[2] && box[2] > u[0] && box[1] < u[3] && box[3] > u[1])) {
        b.el.style.display = "none";
        continue;
      }
      used.push(box);
      b.el.style.display = "";
      b.el.style.transform = `translate(${Math.round(x)}px, ${Math.round(y)}px)`;
    }
  };

  const paint = () => {
    const c = SHADES[shade.peek()];
    if (!map.getLayer(FILL)) return;
    map.setPaintProperty(FILL, "fill-color", c.inactive);
    map.setPaintProperty(FILL, "fill-opacity", c.inactiveOpacity);
  };
  const visibility = () => {
    const v = choosing.peek() ? "none" : "visible";
    if (map.getLayer(FILL)) map.setLayoutProperty(FILL, "visibility", v);
    layer.style.display = choosing.peek() ? "none" : "";
  };
  const add = () => {
    if (!map.getSource(FILL)) map.addSource(FILL, { type: "geojson", data: data() });
    if (!map.getLayer(FILL)) {
      const before = map.getStyle().layers.find((l) => l.type === "symbol")?.id;
      map.addLayer({ id: FILL, type: "fill", source: FILL, paint: { "fill-antialias": false } }, before);
    }
    paint();
    visibility();
  };
  map.on("style.load", add);
  if (map.isStyleLoaded()) add();
  map.on("move", place);
  map.on("resize", place);

  const disposeRealms = effect(() => {
    shown = inactiveRealms.value;
    (map.getSource(FILL) as GeoJSONSource | undefined)?.setData(data());
    const keep = new Set(shown.map((l) => l.id));
    for (const [id, b] of buttons) if (!keep.has(id)) { render(null, b.el); b.el.remove(); buttons.delete(id); }
    place();
  });
  const disposeShade = effect(() => { void shade.value; paint(); });
  const disposeChoosing = effect(() => { void choosing.value; visibility(); });
  return () => {
    stopStations();
    disposeRealms();
    disposeShade();
    disposeChoosing();
    map.off("style.load", add);
    map.off("move", place);
    map.off("resize", place);
    for (const b of buttons.values()) render(null, b.el);
    layer.remove();
    if (map.getLayer(FILL)) map.removeLayer(FILL);
    if (map.getSource(FILL)) map.removeSource(FILL);
  };
}
