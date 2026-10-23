// T-975, T-1006: the realms not shown (game/realms.ts). One shade covers the world, land and water
// alike, with the shown realms' zones cut out along their coarse outlines (the lines every other zone
// outline follows), drawn by its own layer (shadeLayer.ts, T-1025), under labels and the network and
// taking no clicks; where hidden realms meet there
// is no edge. Each hidden realm has a floating button at its point that shows it. A button whose point
// is off screen sits inside the on-screen part of its realm's box (as the loading bars do, T-939);
// buttons that would overlap give way to the realm with more zones.

import { effect } from "@preact/signals";
import { h, render } from "preact";
import type { MultiPolygon } from "geojson";
import type { Map as MlMap } from "maplibre-gl";
import { CITIES } from "../game/cities";
import { choosing } from "../game/loading";
import { followStations, inactiveRealms, loadRealms, realmName, shownRealms, showRealm, zoneOutlines, zoneShown, type RealmSpot } from "../game/realms";
import { SHADES } from "../game/shadeTable";
import { shade } from "../game/ui";
import { Icon } from "../ui/widgets";
import { outsideZones } from "./boundaryGeometry";
import { ShadeLayer } from "./shadeLayer";

const FILL = "tw-realms-inactive";
/** how far a button keeps from the map's edges, px */
const EDGE = 48;
const GAP = 6;

export function startRealmOverlay(map: MlMap): () => void {
  const stopStations = followStations();
  void loadRealms();
  const layer = document.createElement("div");
  layer.className = "realm-buttons";
  map.getContainer().appendChild(layer);
  const buttons = new Map<string, { el: HTMLDivElement; w: number; h: number }>();
  let hidden: RealmSpot[] = [];
  const shadeLayer = new ShadeLayer(FILL);

  const button = (l: RealmSpot) => {
    let b = buttons.get(l.id);
    if (!b) {
      const el = document.createElement("div");
      el.className = "realm-button";
      layer.appendChild(el);
      const name = realmName(l.id);
      render(h("button", {
        type: "button", "aria-label": `Show ${name}`, "data-tip": `${name}\nClick to show the cities there`,
        onClick: () => showRealm(l.id),
      }, h(Icon.expand, null)), el);
      buttons.set(l.id, (b = { el, w: 0, h: 0 }));
    }
    // measured once drawn (zero while the layer is hidden behind the start menu)
    if (!b.w) { b.el.style.display = ""; b.w = b.el.offsetWidth; b.h = b.el.offsetHeight; }
    return b;
  };

  const place = () => {
    const c = map.getContainer(), W = c.clientWidth, H = c.clientHeight;
    const used: [number, number, number, number][] = [];
    // the realm with more zones first, so it keeps its place when two would overlap
    for (const l of [...hidden].sort((a, b) => b.zones - a.zones)) {
      const b = button(l);
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
    shadeLayer.setColor(c.inactive, c.inactiveOpacity);
  };
  const visibility = () => {
    shadeLayer.visible = !choosing.peek();
    map.triggerRepaint();
    layer.style.display = choosing.peek() ? "none" : "";
  };
  const add = () => {
    if (!map.getLayer(FILL)) {
      const before = map.getStyle().layers.find((l) => l.type === "symbol")?.id;
      map.addLayer(shadeLayer, before);
    }
    paint();
    visibility();
  };
  map.on("style.load", add);
  if (map.isStyleLoaded()) add();
  map.on("move", place);
  map.on("resize", place);

  const disposeCut = effect(() => {
    const shown = shownRealms.value, outlines = zoneOutlines.value;
    if (!shown || !outlines) shadeLayer.setTriangles(new Float64Array(0));
    else {
      const zones: MultiPolygon[] = [];
      for (const c of CITIES) {
        const g = outlines.get(c.id);
        if (g && zoneShown(c.id)) zones.push(g);
      }
      shadeLayer.setTriangles(outsideZones(zones));
    }
  });
  const disposeButtons = effect(() => {
    hidden = inactiveRealms.value;
    const keep = new Set(hidden.map((l) => l.id));
    for (const [id, b] of buttons) if (!keep.has(id)) { render(null, b.el); b.el.remove(); buttons.delete(id); }
    place();
  });
  const disposeShade = effect(() => { void shade.value; paint(); });
  const disposeChoosing = effect(() => { void choosing.value; visibility(); place(); });
  return () => {
    stopStations();
    disposeCut();
    disposeButtons();
    disposeShade();
    disposeChoosing();
    map.off("style.load", add);
    map.off("move", place);
    map.off("resize", place);
    for (const b of buttons.values()) render(null, b.el);
    layer.remove();
    if (map.getLayer(FILL)) map.removeLayer(FILL);
  };
}
