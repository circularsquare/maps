// T-975, T-1006: the realms not shown (game/realms.ts). One shade covers the world, land and water
// alike, with the shown realms' zones cut out along their coarse outlines (the lines every other zone
// outline follows), drawn by its own layer (shadeLayer.ts, T-1025), under labels and the network and
// taking no clicks; where hidden realms meet there is no edge. Each hidden realm has a floating
// circle at its point (an unopened zone's dot, a little bigger, T-1046) that shows it, and a plain
// click on the realm with the select tool shows it too (T-1049). A circle stays at its point and
// leaves the screen with it, never pinned at the edge (T-1142); circles that would overlap give way
// to the realm with more zones. The hidden land has a hover saying a click loads it, and a view
// resting at AUTO_SHOW_ZOOM or closer with its centre on a hidden realm's land shows that realm, as a
// click would (T-1151). Zoomed out past the world (CITY_MARKS_MIN_ZOOM) there are no circles, no
// hover and no click.

import { effect } from "@preact/signals";
import { h, render } from "preact";
import type { MultiPolygon } from "geojson";
import type { Map as MlMap } from "maplibre-gl";
import { CITIES } from "../game/cities";
import { linking } from "../game/links";
import { mapBlank } from "../game/loading";
import { followStations, hiddenRealmAt, inactiveRealms, loadRealms, realmName, shownRealms, showRealm, zoneOutlines, zoneShown, type RealmSpot } from "../game/realms";
import { SHADES } from "../game/shadeTable";
import { selection, shade, tool } from "../game/ui";
import type { ShadeIn, ShadeOut } from "../workers/shade.worker";
import { outsideZones } from "./boundaryGeometry";
import { firstSymbolId } from "./basemap";
import { takeOutlineText } from "./coarseOutlines";
import { CITY_MARKS_MIN_ZOOM } from "./intercityView";
import { ShadeLayer } from "./shadeLayer";
import { hiddenOnGlobe } from "./live";

const FILL = "tw-realms-inactive";
/** the room kept round a circle when two would overlap, px */
const GAP = 6;
/** a circle's diameter, CSS px (style.css .realm-button button) */
const SIZE = 16;
/** T-1151: a view this close, resting this long with its centre on a hidden realm's land, shows it */
export const AUTO_SHOW_ZOOM = 10;
const AUTO_SHOW_REST_MS = 400;

const marksOffAt = (map: MlMap) => mapBlank.peek() || map.getZoom() < CITY_MARKS_MIN_ZOOM;

/** The hover on a hidden realm's land with the select tool (map/trackHover.ts asks when nothing of
 * the network is under the pointer): its name, and that a click loads it. */
export function hiddenRealmTip(map: MlMap, px: number, py: number): string[] | null {
  if (marksOffAt(map) || tool.peek() !== "select" || linking.peek() !== null) return null;
  const at = map.unproject([px, py]);
  const id = hiddenRealmAt(at.lng, at.lat);
  return id ? [realmName(id), "Click to load"] : null;
}

export function startRealmOverlay(map: MlMap): () => void {
  const stopStations = followStations();
  void loadRealms();
  const layer = document.createElement("div");
  layer.className = "realm-buttons";
  map.getContainer().appendChild(layer);
  const buttons = new Map<string, { el: HTMLDivElement; w: number; h: number }>();
  let hidden: RealmSpot[] = [];
  const shadeLayer = new ShadeLayer(FILL);
  const marksOff = () => marksOffAt(map);

  const button = (l: RealmSpot) => {
    let b = buttons.get(l.id);
    if (!b) {
      const el = document.createElement("div");
      el.className = "realm-button";
      layer.appendChild(el);
      const name = realmName(l.id);
      render(h("button", {
        type: "button", "aria-label": `Show ${name}`, "data-tip": `${name}\nClick the shaded area to show the cities there`,
        onClick: () => showRealm(l.id),
      }), el);
      buttons.set(l.id, (b = { el, w: SIZE, h: SIZE }));
    }
    return b;
  };

  const place = () => {
    layer.style.display = marksOff() ? "none" : "";
    if (marksOff()) return;
    const c = map.getContainer(), W = c.clientWidth, H = c.clientHeight;
    const used: [number, number, number, number][] = [];
    // the realm with more zones first, so it keeps its place when two would overlap
    for (const l of [...hidden].sort((a, b) => b.zones - a.zones)) {
      const b = button(l);
      const p = map.project(l.label);
      // T-1142: centred on the point, moving with the map and leaving the screen with it
      const x = p.x - b.w / 2, y = p.y - b.h / 2;
      const box: [number, number, number, number] = [x - GAP, y - GAP, x + b.w + GAP, y + b.h + GAP];
      if (x + b.w <= 0 || y + b.h <= 0 || x >= W || y >= H || hiddenOnGlobe(l.label[0], l.label[1]) || used.some((u) => box[0] < u[2] && box[2] > u[0] && box[1] < u[3] && box[3] > u[1])) {
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
    shadeLayer.visible = !mapBlank.peek();
    map.triggerRepaint();
  };
  const add = () => {
    if (!map.getLayer(FILL)) {
      map.addLayer(shadeLayer, firstSymbolId(map));
    }
    paint();
    visibility();
  };
  map.on("style.load", add);
  if (map.isStyleLoaded()) add();
  map.on("move", place);
  map.on("resize", place);

  // A plain click (the map fires none after a drag) with the select tool on a hidden realm's outline.
  // It runs after the pick's own handler: a click that selected something is that action's, not ours.
  const onClick = (e: { lngLat: { lng: number; lat: number }; originalEvent: MouseEvent }) => {
    if (marksOff() || tool.peek() !== "select" || linking.peek() !== null || e.originalEvent.shiftKey) return;
    const at = e.lngLat;
    setTimeout(() => {
      if (selection.peek()) return;
      const id = hiddenRealmAt(at.lng, at.lat);
      if (id) showRealm(id);
    }, 0);
  };
  map.on("click", onClick);

  // on a move's end only: a page opened without a view in its address starts over New York
  // (mapView.ts), which is no reason to load the realm there
  let restTimer = 0;
  const autoShow = () => {
    if (marksOff() || map.getZoom() < AUTO_SHOW_ZOOM) return;
    const c = map.getCenter();
    const id = hiddenRealmAt(c.lng, c.lat);
    if (id) showRealm(id);
  };
  const onRest = () => {
    clearTimeout(restTimer);
    restTimer = window.setTimeout(autoShow, AUTO_SHOW_REST_MS);
  };
  map.on("moveend", onRest);

  const cutter = shadeCutter((tri) => shadeLayer.setTriangles(tri));
  const disposeCut = effect(() => {
    const shown = shownRealms.value, outlines = zoneOutlines.value;
    if (!shown || !outlines) {
      cutter.cancel();
      shadeLayer.setTriangles(new Float64Array(0));
    } else cutter.cut(CITIES.filter((c) => outlines.has(c.id) && zoneShown(c.id)).map((c) => c.id), outlines);
  });
  const disposeButtons = effect(() => {
    hidden = inactiveRealms.value;
    const keep = new Set(hidden.map((l) => l.id));
    for (const [id, b] of buttons) if (!keep.has(id)) { render(null, b.el); b.el.remove(); buttons.delete(id); }
    place();
  });
  const disposeShade = effect(() => { void shade.value; paint(); });
  const disposeBlank = effect(() => { void mapBlank.value; visibility(); place(); });
  return () => {
    stopStations();
    disposeCut();
    cutter.stop();
    disposeButtons();
    disposeShade();
    disposeBlank();
    map.off("style.load", add);
    map.off("move", place);
    map.off("resize", place);
    map.off("click", onClick);
    map.off("moveend", onRest);
    clearTimeout(restTimer);
    for (const b of buttons.values()) render(null, b.el);
    layer.remove();
    if (map.getLayer(FILL)) map.removeLayer(FILL);
  };
}

/** T-1026: the shade's cut in a worker, one at a time, only the newest asked for applied; the shade
 * keeps its triangles until then. On the main thread when the worker cannot run. */
function shadeCutter(apply: (tri: Float64Array) => void) {
  let worker: Worker | null | undefined;
  let gen = 0, busy = false;
  let queued: string[] | null = null;
  let last: { ids: string[]; outlines: ReadonlyMap<string, MultiPolygon> } | null = null;
  const onMain = () => {
    worker?.terminate();
    worker = null;
    busy = false;
    queued = null;
    if (last) apply(outsideZones(last.ids.map((id) => last!.outlines.get(id)!)));
  };
  const send = (ids: string[]) => {
    busy = true;
    worker!.postMessage({ kind: "cut", gen, ids } satisfies ShadeIn);
  };
  const start = () => {
    const text = takeOutlineText();
    if (!text) return null;
    try {
      const w = new Worker(new URL("../workers/shade.worker.ts", import.meta.url), { type: "module" });
      w.onmessage = (e: MessageEvent<ShadeOut>) => {
        busy = false;
        if (queued) { const q = queued; queued = null; send(q); }
        else if (e.data.gen === gen) apply(e.data.tri);
      };
      w.onerror = (e) => { console.warn("Shade worker failed", e.message); onMain(); };
      w.postMessage({ kind: "outlines", text } satisfies ShadeIn);
      return w;
    } catch (error) {
      console.warn("Shade worker unavailable", error);
      return null;
    }
  };
  return {
    cut(ids: string[], outlines: ReadonlyMap<string, MultiPolygon>) {
      gen++;
      last = { ids, outlines };
      if (worker === undefined) worker = start();
      if (!worker) return onMain();
      if (busy) queued = ids;
      else send(ids);
    },
    cancel() { gen++; queued = null; last = null; },
    stop() { worker?.terminate(); worker = null; },
  };
}
