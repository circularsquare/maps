// The cities this game has not opened (T-729, notes/T-728.md): one grey bubble each at its registry
// centre, at every zoom, sized as the zoomed-out city bubbles are (`overviewRadius`, by the
// pipeline's `localTrips`). The hover lightens it (map/hovered.ts) and says what it is; a click adds
// the city to the game's opened list (game/openedZones.ts), whose loading then opens it bare. Hidden
// while a bubble or city selection shows its far ends, which draw partner bubbles at the same centres.

import { effect } from "@preact/signals";
import type { Map as MlMap } from "maplibre-gl";
import { CITIES, type CityId } from "../game/cities";
import { count } from "../game/format";
import { addOpened, openedZones } from "../game/openedZones";
import { BUBBLE_ALPHA, rgb } from "../game/palette";
import { shadeColours } from "../game/shades";
import { selection, shade } from "../game/ui";
import { DISC } from "../workers/commutersProtocol";
import { anchorVersion } from "./anchorVersion";
import { addFinder, discAt, hideHover, pointerAt } from "./demandHover";
import type { DemandOverlay } from "./demandOverlay";
import { anchor, toLocal, type Anchor } from "./geo";
import { overviewRadius } from "./intercityView";

/** Over the commuter discs (10), since a grey bubble takes the click first. */
const FINDER_ORDER = 11;

let marks: { discs: Float32Array; anchor: Anchor; ids: CityId[] } | null = null;
let overlay: DemandOverlay | null = null;
let mapRef: MlMap | null = null;

export function startGreyCities(view: DemandOverlay, map: MlMap) {
  overlay = view;
  mapRef = map;
  const render = () => {
    const o = openedZones.peek();
    const kind = selection.peek()?.kind;
    if (o === null || o === "all" || kind === "commuters" || kind === "city") {
      marks = null;
      view.setDiscs("grey", null);
      return;
    }
    const zoom = map.getZoom(), a = anchor();
    const [r, g, b] = rgb(shadeColours().greyCity).map((v) => v / 255);
    // biggest first, so the small ones draw on top
    const cities = CITIES.filter((c) => !o.has(c.id) && (c.localTrips ?? 0) > 0).sort((p, q) => q.localTrips! - p.localTrips!);
    const discs = new Float32Array(cities.length * DISC);
    cities.forEach((c, i) => {
      const [x, y] = toLocal(c.center[0], c.center[1], a);
      discs.set([x, y, overviewRadius(c.localTrips!, zoom), r, g, b, BUBBLE_ALPHA, 1], i * DISC);
    });
    marks = { discs, anchor: a, ids: cities.map((c) => c.id) };
    view.setDiscs("grey", discs, a);
  };
  effect(() => {
    openedZones.value;
    selection.value;
    shade.value;
    anchorVersion.value; // the discs are measured from the draw anchor (T-317)
    render();
  });
  map.on("zoom", render);

  addFinder(FINDER_ORDER, (p) => {
    if (!marks) return null;
    const i = discAt(marks.discs, p, marks.anchor, view.bubbleSize);
    if (i < 0) return null;
    const c = CITIES.find((x) => x.id === marks!.ids[i])!;
    return [c.name,`${count(c.localTrips ?? 0)} trips a day within ${c.name}`, "Click to open"];
  });
}

/** The grey city under a map point (CSS px), or null. */
export function greyCityAt(px: number, py: number): CityId | null {
  if (!marks || !overlay || !mapRef) return null;
  const i = discAt(marks.discs, pointerAt(mapRef, px, py), marks.anchor, overlay.bubbleSize);
  return i < 0 ? null : marks.ids[i];
}

/** A click on a grey city opens it, the camera staying where it is (Anita, 2026-10-19); true if one
 * was there. */
export function openGreyCityAt(px: number, py: number): boolean {
  const id = greyCityAt(px, py);
  if (!id) return false;
  addOpened([id]);
  hideHover();
  return true;
}
