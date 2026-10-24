// A mark on each attraction (T-263, docs/ui-map.md "Attractions"), with a hover of its name, its
// visits a day and how many come by train. The bigger ones show first as the map zooms in; with the
// purpose filter on Attractions they show from further out. The share by train is the attractions
// purpose's legs at the attraction's cell (the cell the demand layer gave it, the nearest within
// 600 m), so two attractions in one cell share it.

import { effect } from "@preact/signals";
import type { Map as MlMap } from "maplibre-gl";
import { activeCity, cityById, getWorldFrame, widenBox, worldFrameEpoch, zoneBox, type CityId } from "../game/cities";
import { loadAttractions, visitsOn, type Attraction } from "../game/attractions";
import { cellModes, cityCells, demandView, type DemandView } from "../game/demand";
import { count } from "../game/format";
import type { DemandDay } from "../game/days";
import { nearbyZones } from "../game/zones";
import { purposeFilter, shownDay, shownDayPhrase } from "../game/ui";
import { addFinder } from "./demandHover";
import { modeBar } from "./modeBar";
import { cityOverview } from "./intercityView";
import { hiddenOnGlobe } from "./live";
import type { Overlay } from "./overlay";

/** A classical front: a pediment on three columns. */
const MARK_PATH = "M8 1 15 5H1ZM2.5 6h2.2v6H2.5ZM6.9 6h2.2v6H6.9ZM11.3 6h2.2v6h-2.2ZM1 13h14v2H1Z";
const SIZE = 12;
const HOVER_PX = 8;
/** As the sim's `NEAR_CELL_M` (sim/src/demand/trip.rs). */
const NEAR_CELL_M = 600;
/** The zoom from which attractions show, and from which with the purpose filter on Attractions. */
const MIN_ZOOM = 12;
const MIN_ZOOM_FILTERED = 10;
/** The smallest attraction shown at a zoom (visitors a year): the big ones first. */
const smallestAt = (zoom: number) => (zoom >= 14 ? 0 : zoom >= 13 ? 200_000 : zoom >= 12 ? 500_000 : zoom >= 11 ? 1_000_000 : 2_000_000);

interface Modes { train: number; walk: number; drive: number }

export function startAttractionMarks(map: MlMap, overlay: Overlay) {
  const box = document.createElement("div");
  box.className = "attraction-marks";
  map.getContainer().appendChild(box);
  let all: Attraction[] = [];
  void loadAttractions().then((a) => { all = a; place(); });
  /** each zone's attractions with the cell the demand layer gave them, -1 none (not in the zone) */
  const cellsOf = new Map<CityId, Map<Attraction, number>>();
  /** each zone's attractions' legs by mode on a day, and the result they were read from */
  const modesOf = new Map<string, { at: DemandView; modes: Map<Attraction, Modes> }>();
  const asked = new Set<string>();
  let marks: HTMLDivElement[] = [];
  let shown: Attraction[] = [];

  /** The nearest cell within reach of each attraction in the zone's box, by a grid of cells. */
  const placeIn = (zone: CityId, xy: Float32Array, n: number): Map<Attraction, number> => {
    const frame = getWorldFrame(), grid = new Map<string, number[]>();
    for (let c = 0; c < n; c++) {
      const k = `${Math.floor(xy[2 * c] / NEAR_CELL_M)},${Math.floor(xy[2 * c + 1] / NEAR_CELL_M)}`;
      (grid.get(k) ?? grid.set(k, []).get(k)!).push(c);
    }
    const [w, s, e, nn] = widenBox(zoneBox(cityById(zone)), 1000), out = new Map<Attraction, number>();
    for (const a of all) {
      if (a.lng < w || a.lng > e || a.lat < s || a.lat > nn) continue;
      const [x, y] = frame.fromLonLat(a.lng, a.lat), gx = Math.floor(x / NEAR_CELL_M), gy = Math.floor(y / NEAR_CELL_M);
      let best = -1, bd = NEAR_CELL_M;
      for (let dx = -1; dx <= 1; dx++) for (let dy = -1; dy <= 1; dy++) {
        for (const c of grid.get(`${gx + dx},${gy + dy}`) ?? []) {
          const d = Math.hypot(xy[2 * c] - x, xy[2 * c + 1] - y);
          if (d <= bd) { bd = d; best = c; }
        }
      }
      if (best >= 0) out.set(a, best);
    }
    return out;
  };

  const load = async (zone: CityId, focus: boolean, day: DemandDay) => {
    const key = `${zone}:${day}`, at = demandView.peek();
    if (!at || !all.length || modesOf.get(key)?.at === at || asked.has(key)) return;
    asked.add(key);
    const [cells, modes] = await Promise.all([cityCells(focus ? undefined : zone), cellModes("attractions", focus ? undefined : zone)]);
    asked.delete(key);
    if (!cells || !modes || modes.day !== day) return;
    if (!cellsOf.has(zone)) cellsOf.set(zone, placeIn(zone, cells.xy, cells.n));
    const out = new Map<Attraction, Modes>();
    for (const [a, c] of cellsOf.get(zone)!) out.set(a, { train: modes.workRail[c], walk: modes.workWalk[c], drive: modes.workDrive[c] });
    modesOf.set(key, { at, modes: out });
    if (demandView.peek() !== at) ensure();
  };

  const minZoom = () => (purposeFilter.peek() === "attractions" ? MIN_ZOOM_FILTERED : MIN_ZOOM);
  const ensure = () => {
    if (map.getZoom() < minZoom() || cityOverview.peek() || !demandView.peek()) return;
    const focus = activeCity.peek(), day = shownDay.peek();
    void load(focus, true, day);
    for (const id of nearbyZones.peek()) if (id !== focus) void load(id, false, day);
  };
  /** The modes of an attraction from the zones on show, on the day shown. */
  const modesAt = (a: Attraction): Modes | null => {
    const day = shownDay.peek();
    for (const id of [activeCity.peek(), ...nearbyZones.peek()]) {
      const m = modesOf.get(`${id}:${day}`)?.modes.get(a);
      if (m) return m;
    }
    return null;
  };

  let drawn = 0;
  const placed: string[] = [];
  const place = () => {
    shown = [];
    const z = map.getZoom();
    if (all.length && z >= minZoom() && !cityOverview.peek()) {
      const b = map.getBounds(), smallest = smallestAt(z);
      for (const a of all) {
        if (a.visitors < smallest) break;
        if (a.lng >= b.getWest() && a.lng <= b.getEast() && a.lat >= b.getSouth() && a.lat <= b.getNorth()) shown.push(a);
      }
    }
    if (!shown.length && !drawn) return;
    const w = overlay.cssW, h = overlay.cssH;
    let k = 0;
    for (const a of shown) {
      const { x, y } = map.project([a.lng, a.lat]);
      if (x < -SIZE || y < -SIZE || x > w + SIZE || y > h + SIZE || hiddenOnGlobe(a.lng, a.lat)) continue;
      let m = marks[k];
      if (!m) {
        m = marks[k] = document.createElement("div");
        m.className = "attraction-mark";
        m.innerHTML = `<svg viewBox="0 0 16 16"><path d="${MARK_PATH}" /></svg>`;
        box.appendChild(m);
      }
      if (k >= drawn) m.style.display = "";
      const t = `translate(${Math.round(x)}px, ${Math.round(y)}px)`;
      if (placed[k] !== t) m.style.transform = placed[k] = t;
      k++;
    }
    for (let i = k; i < drawn; i++) marks[i].style.display = "none";
    drawn = k;
  };
  overlay.afterMapFrame.push(place);
  map.on("moveend", ensure);

  let epoch = -1;
  effect(() => {
    purposeFilter.value;
    cityOverview.value;
    nearbyZones.value;
    activeCity.value;
    demandView.value;
    shownDay.value;
    if (worldFrameEpoch.value !== epoch) {
      epoch = worldFrameEpoch.peek();
      cellsOf.clear();
      modesOf.clear();
      asked.clear();
    }
    ensure();
    place();
  });

  const escape = (s: string) => s.replace(/[&<>"]/g, (c) => `&#${c.charCodeAt(0)};`);
  // the hover tag, above the bubbles' (10) and the airports' (11)
  addFinder(12, (p) => {
    if (!shown.length) return null;
    const at = map.project([p.lng, p.lat]);
    for (const a of shown) {
      const m = map.project([a.lng, a.lat]);
      if (Math.hypot(m.x - at.x, m.y - at.y) > HOVER_PX || hiddenOnGlobe(a.lng, a.lat)) continue;
      const lines = [escape(a.name), `About ${count(visitsOn(a.visitors, shownDay.peek()))} visits ${shownDayPhrase.peek()}`];
      const modes = modesAt(a), legs = modes ? modes.train + modes.walk + modes.drive : 0;
      if (modes && legs > 0) lines.push(`${Math.round((100 * modes.train) / legs)}% by train`, modeBar(modes.train, modes.walk, modes.drive));
      return lines;
    }
    return null;
  });
}
