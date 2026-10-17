// A plane on each airport, and under All purposes the airport's own bubble (T-448, T-649). Destination
// bubbles have no kind of their own, so the airports are found from the airport purpose's legs: the
// cells that receive any are the terminals. Their legs do not move with the network (only the mode split does), so each zone is
// asked once per kind of day, when its Destinations bubbles first show. Nearby zones' airports
// get their badge with their bubbles. Other POI kinds could be marked the same way: any purpose
// whose destinations are a few fixed sites (the medical purpose, universities under school) would
// need only its own cluster size and glyph.

import { effect } from "@preact/signals";
import type { Map as MlMap } from "maplibre-gl";
import { activeCity, getWorldFrame, worldFrameEpoch, type CityId } from "../game/cities";
import { cellModes, cityCells, demandView } from "../game/demand";
import { count } from "../game/format";
import type { DemandDay } from "../game/days";
import { nearbyZones } from "../game/zones";
import { display, purposeFilter, selection, shownDay, shownDayPhrase } from "../game/ui";
import { addFinder } from "./demandHover";
import { modeBar } from "./modeBar";
import { cityOverview } from "./intercityView";
import { BUBBLE_ALPHA, modeMix } from "../game/palette";
import { BUBBLE_DENSITY, DISC } from "../workers/commutersProtocol";
import type { DemandOverlay } from "./demandOverlay";
import { anchor, EARTH_C, kAtY, toLocal } from "./geo";
import { PLANE_PATH } from "./glyphs";
import type { Overlay } from "./overlay";

interface Airport {
  lng: number;
  lat: number;
  train: number;
  walk: number;
  drive: number;
  /** its bubble's radius in local units before the Bubble size setting; 0 when only the plane is drawn */
  r: number;
}

/** Terminal cells this close (m) share a badge; Heathrow's T5 and T2/3 are further apart and get one each. */
const JOIN_M = 2500;
/** An airport smaller than this (trip legs a day) gets no badge. */
const MIN_LEGS = 500;
/** The plane's size, px, and how near the pointer must be for the hover when there is no bubble. */
const SIZE = 14;
const HOVER_PX = 12;

export function startAirportMarks(map: MlMap, overlay: Overlay, view: DemandOverlay) {
  const box = document.createElement("div");
  box.className = "airport-marks";
  map.getContainer().appendChild(box);
  /** the airports of a zone and kind of day, once asked for */
  const known = new Map<string, Airport[]>();
  const asked = new Set<string>();
  const tries = new Map<string, number>();
  let marks: HTMLDivElement[] = [];
  let shown: Airport[] = [];
  let drawn = false;

  /** Terminals' cells into airports: greedy, the busiest cell first. An airport sits at its busiest
   * cell, inside the biggest bubble at every zoom (the legs' mean can fall between terminals). */
  const cluster = (lngs: number[], lats: number[], t: number[], w: number[], d: number[]): Airport[] => {
    const order = lngs.map((_, i) => i).sort((a, b) => t[b] + w[b] + d[b] - (t[a] + w[a] + d[a]));
    const used = new Uint8Array(lngs.length);
    const out: Airport[] = [];
    for (const i of order) {
      if (used[i]) continue;
      const kx = 111320 * Math.cos((lats[i] * Math.PI) / 180);
      let sw = 0, tt = 0, ww = 0, dd = 0;
      for (const j of order) {
        if (used[j] || Math.hypot((lngs[j] - lngs[i]) * kx, (lats[j] - lats[i]) * 110540) > JOIN_M) continue;
        used[j] = 1;
        const all = t[j] + w[j] + d[j];
        sw += all;
        tt += t[j];
        ww += w[j];
        dd += d[j];
      }
      if (sw >= MIN_LEGS) out.push({ lng: lngs[i], lat: lats[i], train: tt, walk: ww, drive: dd, r: 0 });
    }
    return out;
  };

  const load = async (zone: CityId, focus: boolean, day: DemandDay) => {
    const key = `${zone}:${day}`;
    if (known.has(key) || asked.has(key)) return;
    asked.add(key);
    const [cells, modes] = await Promise.all([cityCells(focus ? undefined : zone), cellModes("airport", focus ? undefined : zone)]);
    asked.delete(key);
    if (!cells || !modes || modes.day !== day) {
      // a result landed while it was asked for, or the zone is not in the solve yet: ask again
      if ((tries.set(key, (tries.get(key) ?? 0) + 1).get(key) ?? 0) < 6) setTimeout(ensure, 1500);
      return;
    }
    const frame = getWorldFrame();
    const lngs: number[] = [], lats: number[] = [], t: number[] = [], w: number[] = [], d: number[] = [];
    for (let c = 0; c < cells.n; c++) {
      if (modes.workRail[c] + modes.workWalk[c] + modes.workDrive[c] <= 0) continue;
      const [lng, lat] = frame.toLonLat(cells.xy[c * 2], cells.xy[c * 2 + 1]);
      lngs.push(lng);
      lats.push(lat);
      t.push(modes.workRail[c]);
      w.push(modes.workWalk[c]);
      d.push(modes.workDrive[c]);
    }
    known.set(key, cluster(lngs, lats, t, w, d));
    refresh();
  };

  /** Which airports to badge now: Destinations bubbles are on and show the airport's trips. */
  const wanted = () => {
    const sel = selection.peek()?.kind;
    const p = purposeFilter.peek();
    return display.peek().commuters && display.peek().commuterEnd === "work" && (p === "all" || p === "airport")
      && !cityOverview.peek() && sel !== "commuters";
  };

  let epoch = -1;
  /** Ask for the airports of every zone on show that is not known yet. */
  const ensure = () => {
    if (!wanted() || !demandView.peek()) return;
    const focus = activeCity.peek(), day = shownDay.peek();
    void load(focus, true, day);
    for (const id of nearbyZones.peek()) if (id !== focus) void load(id, false, day);
  };
  const refresh = () => {
    shown = [];
    if (wanted() && demandView.peek()) {
      const focus = activeCity.peek(), day = shownDay.peek();
      for (const id of [focus, ...nearbyZones.peek()]) shown.push(...(known.get(`${id}:${day}`) ?? []));
    }
    // under All purposes the airport is a bubble of its own on the destination bubbles; under Airport
    // the destination bubbles already are the airports' trips, so only the plane is added
    if (shown.length && purposeFilter.peek() === "all") {
      const a = anchor(), discs = new Float32Array(shown.length * DISC);
      shown.forEach((p, i) => {
        const [x, y] = toLocal(p.lng, p.lat, a);
        const c = modeMix(p.train, p.walk, p.drive);
        const r = Math.sqrt((p.train + p.walk + p.drive) / BUBBLE_DENSITY.work / Math.PI) * kAtY(y, a);
        discs.set([x, y, r, c[0] / 255, c[1] / 255, c[2] / 255, BUBBLE_ALPHA, 0], i * DISC);
        p.r = r;
      });
      view.setDiscs("airport", discs, a);
      drawn = true;
    } else {
      if (drawn) view.setDiscs("airport", null);
      drawn = false;
      for (const p of shown) p.r = 0;
    }
    place();
  };

  const place = () => {
    const w = map.getContainer().clientWidth, h = map.getContainer().clientHeight;
    let k = 0;
    for (const a of shown) {
      const { x, y } = map.project([a.lng, a.lat]);
      if (x < -SIZE || y < -SIZE || x > w + SIZE || y > h + SIZE) continue;
      let m = marks[k];
      if (!m) {
        m = marks[k] = document.createElement("div");
        m.className = "airport-mark";
        m.innerHTML = `<svg viewBox="0 0 16 16"><path d="${PLANE_PATH}" /></svg>`;
        box.appendChild(m);
      }
      m.style.display = "";
      m.style.transform = `translate(${Math.round(x)}px, ${Math.round(y)}px)`;
      k++;
    }
    for (let i = k; i < marks.length; i++) marks[i].style.display = "none";
  };
  overlay.afterMapFrame.push(place);

  effect(() => {
    display.value;
    purposeFilter.value;
    selection.value;
    cityOverview.value;
    nearbyZones.value;
    activeCity.value;
    demandView.value;
    shownDay.value;
    if (worldFrameEpoch.value !== epoch) {
      epoch = worldFrameEpoch.peek();
      known.clear();
      asked.clear();
      tries.clear();
    }
    ensure();
    refresh();
  });

  // the hover tag, above the bubbles' own (order 10)
  addFinder(11, (p) => {
    if (!shown.length) return null;
    const at = map.project([p.lng, p.lat]);
    const pxPerUnit = (512 * 2 ** map.getZoom()) / EARTH_C;
    for (const a of shown) {
      const m = map.project([a.lng, a.lat]);
      if (Math.hypot(m.x - at.x, m.y - at.y) > Math.max(HOVER_PX, a.r * view.bubbleSize * pxPerUnit)) continue;
      const all = a.train + a.walk + a.drive;
      return [
        `${count(all)} airport trips ${shownDayPhrase.peek()} here`,
        modeBar(a.train, a.walk, a.drive),
      ];
    }
    return null;
  });
}
