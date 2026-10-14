// Longer trips on the map and in the inspectors (T-118, T-120, T-186, SPEC 8). The commuter views
// show where people go to work; this shows where the same places send the market's other, longer
// trips (visits, shopping, business; sim/src/demand/intercity.rs), which a commuter bubble cannot:
// a Philadelphia bubble's longer trips end in New York, far off the screen.
//
// - A bubble selection (a `commuters` selection) asks `intercitySelection(end, cells)`: how many
//   trips start in those cells and how many end there, by far zone and by place, all modes and by
//   train. The map shows the far end of the selection's own end (homes: where the trips go; jobs:
//   where they come from).
// - A selected station asks `intercityStation(id)`: the longer-trip rail riders starting and ending
//   there. The map shows the ones starting there (their home end is the station).
// - Zoomed far out (OVERVIEW_ZOOM and below) with Homes or Destinations on and nothing of the
//   above selected, every city is one bubble of its longer trips, the focused city too, in place
//   of the focused city's own bubbles (T-186). A click on one selects the city (a `city`
//   selection): it is drawn outlined, with one bubble per other city for the trips between them.
//   These come from the market's corridors (`intercityDemand`), and catch up over a game day after
//   a new result like the commuter bubbles (T-156, map/catchUp.ts).
//
// Up close every positive far 6 km zone outside the focused market is shown on its own relative
// trip scale. Same-city longer trips of a selection remain in the inspector only (T-139). Regional
// zooms aggregate a selection's trips to one labeled circle per other market (T-135, T-136). Area
// represents total travellers, without a minimum radius or additive size offset. All use the same
// opacity and size slider, coloured by `modeMix(rail, 0, everything else)`: blue train, red the
// rest. They are always drawn (the setting that hid them went with T-222), except under a local
// purpose filter; the numbers stay in the inspectors.
// Asked for, not pushed, and asked again whenever `demandView` changes, keeping the last answer
// meanwhile.

import { effect, signal } from "@preact/signals";
import type { Map as MlMap } from "maplibre-gl";
import { activeCity, cityById, CITIES } from "../game/cities";
import { clock } from "../game/clock";
import { packToLngLat } from "../game/coords";
import { cityDemandViews, demandView, intercityDemand, INTERCITY_PLACES, intercitySelection, intercityStation, type IntercitySelection, type IntercityStation, type PlaceFlows } from "../game/demand";
import { selectedScope } from "../game/complexView";
import { count } from "../game/format";
import { BUBBLE_ALPHA, modeMix } from "../game/palette";
import { display, purposeFilter, selection } from "../game/ui";
import type { Selection } from "../game/types";
import { DISC } from "../workers/commutersProtocol";
import { CATCH_STEP, caughtUp, tripsDone } from "./catchUp";
import { addFinder, discAt, pointerLocal } from "./demandHover";
import type { DemandOverlay } from "./demandOverlay";
import { EARTH_C, HOME_ANCHOR, MERC_K, toLocal, toLngLat } from "./geo";

/** A bubble selection's longer trips, for the commuters inspector; null while unknown. */
export const intercityPick = signal<{ sel: Selection; data: IntercitySelection; own: number } | null>(null);
/** A selected station's, for the station inspector. */
export const intercityStationView = signal<IntercityStation | null>(null);

const marketOf = (id: string) => cityById(id).market;

/** Regional views show the complete place totals, rather than scattered endpoint specks. */
export const CITY_BUBBLE_ZOOM = 9.5;
/** Keep longer-trip bubbles subordinate to the local demand view at every zoom. */
const INTERCITY_RADIUS_SCALE = 0.5;
/**
 * At this zoom and below every city is one longer-trips bubble (T-186): the zoom at which the
 * metropolitan outlines are fully drawn, a New York region about half a 1,400 px map wide with
 * Philadelphia and Hartford on screen. The focused city's own bubbles are 6 km apart there.
 */
export const OVERVIEW_ZOOM = 8.5;
/** The busiest city's bubble in the overview, and the selected city's, px radius before the size
 * slider. Larger than a selection's 19 px city circles: here nothing local is drawn beside it. */
const OVERVIEW_PX = 28;

/** The city bubbles are drawn in place of the focused city's own: zoomed far out, or a city
 * selected (T-186). The commuter view hides its bubbles and leaves clicks and hover to these. */
export const cityOverview = signal(false);

interface Marks {
  discs: Float32Array;
  /** per disc, in disc order: the hover tag's lines */
  tags: string[][];
  /** per disc, the city it stands for, in the city view (T-186) */
  cities?: string[];
}

const PLACES = INTERCITY_PLACES.length;

/** Longer trips a day between every two places, both directions, all modes then by train:
 * [trips, rail] at (a * PLACES + b) * 2, the same for (b, a). */
function pairs(corridors: { from: number; to: number; trips: number; rail: number }[]): Float64Array {
  const m = new Float64Array(PLACES * PLACES * 2);
  for (const c of corridors) {
    if (c.from === c.to || c.from >= PLACES || c.to >= PLACES) continue;
    for (const [a, b] of [[c.from, c.to], [c.to, c.from]]) {
      m[(a * PLACES + b) * 2] += c.trips;
      m[(a * PLACES + b) * 2 + 1] += c.rail;
    }
  }
  return m;
}

export function startIntercityView(view: DemandOverlay, map: MlMap) {
  let marks: Marks | null = null;
  /** what is on show, redrawn when the setting changes */
  let current: { flows: PlaceFlows; verb: string; railOnly: boolean; own: number } | null = null;
  const labels = document.createElement("div");
  labels.className = "intercity-labels";
  map.getContainer().appendChild(labels);
  const placeLabels = () => {
    const used: { x: number; y: number; w: number; h: number }[] = [];
    Array.from(labels.children).forEach((child, i) => {
      const el = child as HTMLElement;
      if (!marks) return;
      const p = map.project(toLngLat(marks.discs[i * DISC], marks.discs[i * DISC + 1], HOME_ANCHOR));
      const r = marks.discs[i * DISC + 2] * (512 * 2 ** map.getZoom() / EARTH_C) * view.bubbleSize;
      const w = el.offsetWidth, h = el.offsetHeight;
      const positions = [[p.x + r + 5, p.y - h / 2], [p.x - w - r - 5, p.y - h / 2], [p.x - w / 2, p.y + r + 5], [p.x - w / 2, p.y - h - r - 5]];
      const at = positions.find(([x, y]) => x >= 0 && y >= 0 && x + w <= map.getContainer().clientWidth && y + h <= map.getContainer().clientHeight && !used.some(a => x < a.x + a.w + 4 && x + w + 4 > a.x && y < a.y + a.h + 4 && y + h + 4 > a.y));
      el.style.visibility = at ? "" : "hidden";
      if (at) {
        el.style.transform = `translate(${Math.round(at[0])}px, ${Math.round(at[1])}px)`;
        used.push({x: at[0], y: at[1], w, h});
      }
    });
  };
  const label = (text: string) => {
    const el = document.createElement("div");
    el.className = "intercity-label";
    el.textContent = text;
    labels.appendChild(el);
  };

  // ---- the market's corridors, catching up over a game day (T-156) ----
  let market: Float64Array | null = null;
  let marketVersion = -1;
  /** the pairs drawn when the newest result landed, and when (catchUp.ts); null: caught up */
  let marketFrom: Float64Array | null = null, marketStart = 0;
  /** the share last drawn */
  let marketF = 1;
  const marketShare = () => (marketFrom ? caughtUp(marketStart, clock.now()) : 1);
  const drawnPairs = (f: number): Float64Array | null => {
    if (!market || !marketFrom || f >= 1) return market;
    const out = new Float64Array(market.length);
    for (let i = 0; i < out.length; i++) out[i] = marketFrom[i] + (market[i] - marketFrom[i]) * f;
    return out;
  };

  /** What the city view draws: the overview, a selected city, or neither. */
  const cityMode = (): "overview" | "selected" | null => {
    const sel = selection.peek();
    if (sel?.kind === "city") return "selected";
    if (display.peek().commuters && map.getZoom() <= OVERVIEW_ZOOM && sel?.kind !== "commuters" && sel?.kind !== "station") return "overview";
    return null;
  };

  /** One bubble per city, or a selected city and one bubble per other city (T-186). */
  const renderCities = (selected: string | null) => {
    const f = marketShare();
    marketF = f;
    if (f >= 1) marketFrom = null;
    const m = drawnPairs(f);
    if (!m) {
      view.setDiscs("inter", null);
      return;
    }
    const pair = (a: number, b: number) => ({ all: m[(a * PLACES + b) * 2], rail: m[(a * PLACES + b) * 2 + 1] });
    const total = (a: number) => {
      let all = 0, rail = 0;
      for (let b = 0; b < PLACES; b++) if (b !== a) (all += m[(a * PLACES + b) * 2]), (rail += m[(a * PLACES + b) * 2 + 1]);
      return { all, rail };
    };
    const own = selected ? cityById(selected) : null;
    // a city outside the regional market (market -1: Toronto, T-277) has no longer trips to show
    if (own && own.market < 0) {
      view.setDiscs("inter", null);
      return;
    }
    // T-276: zoomed out, a zone is one bubble as if it were one place: its local trips (the city
    // view's, or with no rail the pipeline's count for a zone not opened) and the regional
    // layer's trips to and from it (commutes between zones and longer trips)
    const local = cityDemandViews.peek();
    const points = CITIES.filter((c) => c.market >= 0).map((c) => {
      const isOwn = c === own;
      let t = !own || isOwn ? total(c.market) : pair(own.market, c.market);
      if (!own) {
        const v = local.get(c.id);
        t = { all: t.all + (v ? v.tripsPerDay : c.localTrips ?? 0), rail: t.rail + (v ? v.railTripsPerDay : 0) };
      }
      const tag = !own
        ? [`${count(t.all)} trips a day in and from ${c.name}`, `${count(t.rail)} by train`]
        : isOwn
          ? [`${count(t.all)} trips a day between ${c.name} and other places`, `${count(t.rail)} by train`]
          : [`${count(t.all)} trips a day between ${own.name} and ${c.name}`, `${count(t.rail)} by train`];
      return { ...t, city: c, isOwn, tag };
    }).filter((p) => p.all > 0 || p.isOwn);
    points.sort((a, b) => b.all - a.all);
    // one scale: the busiest zone on screen in the overview (T-276: with local trips counted New
    // York's bubble would dwarf every other zone's), the selected zone when one is
    const bounds = map.getBounds();
    const onScreen = points.filter((p) => bounds.contains(p.city.center));
    const max = (own ? points.find((p) => p.isOwn)?.all : (onScreen[0] ?? points[0])?.all) || 1;
    const pxPerUnit = 512 * 2 ** map.getZoom() / EARTH_C;
    const discs = new Float32Array(points.length * DISC);
    points.forEach((p, i) => {
      const [x, y] = toLocal(p.city.center[0], p.city.center[1], HOME_ANCHOR);
      const c = modeMix(p.rail, 0, Math.max(0, p.all - p.rail));
      // a zone off screen bigger than the scale is drawn at the scale (T-276)
      const r = OVERVIEW_PX * Math.sqrt(Math.min(1, p.all / max)) / pxPerUnit;
      discs.set([x, y, r, c[0] / 255, c[1] / 255, c[2] / 255, BUBBLE_ALPHA, p.isOwn ? 2 : 1], i * DISC);
      label(`${p.city.name} · ${count(p.all)} trips/day`);
    });
    marks = { discs, tags: points.map((p) => p.tag), cities: points.map((p) => p.city.id) };
    view.setDiscs("inter", discs);
    placeLabels();
  };

  const render = () => {
    marks = null;
    labels.replaceChildren();
    // the regional market has no purposes: its bubbles show for all purposes only (T-158, T-305);
    // the inspector keeps the numbers
    const filter = purposeFilter.peek();
    // (always drawn otherwise: the setting to hide them went with T-222)
    const shown = filter === "all";
    const mode = cityMode();
    // a selected city always takes the place of the city's own bubbles; zoomed far out only
    // when the overview has something to draw
    cityOverview.value = mode === "selected" || (mode === "overview" && shown);
    if (mode && shown) {
      const sel = selection.peek();
      renderCities(sel?.kind === "city" ? sel.city : null);
      return;
    }
    if (!current || !shown || mode) {
      view.setDiscs("inter", null);
      return;
    }
    const regional = map.getZoom() <= CITY_BUBBLE_ZOOM;
    const points = regional ? current.flows.places.filter(p => p.all > 0 && p.place !== current!.own).map(p => {
      const city = CITIES.find(c => c.market === p.place);
      const ll = city?.center ?? packToLngLat(p.x, p.y);
      return { ...p, ll, name: INTERCITY_PLACES[p.place] };
    }) : current.flows.zones.filter(z => z.all > 0 && z.place !== current!.own).map(z => ({...z, ll: packToLngLat(z.x, z.y), name: ""}));
    points.sort((a, b) => b.all - a.all);
    const { verb, railOnly } = current;
    const max = points[0]?.all || 1;
    const pxPerUnit = 512 * 2 ** map.getZoom() / EARTH_C;
    // biggest first, so the small ones draw on top of the big ones
    const n = points.length;
    const discs = new Float32Array(n * DISC);
    const tags: string[][] = [];
    points.forEach((z, i) => {
      const [lng, lat] = z.ll;
      const [x, y] = toLocal(lng, lat, HOME_ANCHOR);
      const c = modeMix(z.rail, 0, Math.max(0, z.all - z.rail));
      // Area, not radius, represents total travellers. Before the common radius scale, the
      // biggest regional circle is 38px and detailed zones scale against their busiest zone.
      // No minimum distorts smaller counts.
      const unscaledRadius = regional ? 38 * Math.sqrt(z.all / max) / pxPerUnit
        : Math.sqrt(0.6 * 6000 ** 2 / Math.PI * z.all / max) * MERC_K;
      const r = unscaledRadius * INTERCITY_RADIUS_SCALE;
      discs.set([x, y, r, c[0] / 255, c[1] / 255, c[2] / 255, BUBBLE_ALPHA, 1], i * DISC);
      const place = z.name || "here";
      tags.push(railOnly ? [`${count(z.rail)} riders a day on inter-region trips ${verb} ${place}`] : [`${count(z.all)} inter-region trips a day ${verb} ${place}`, `${count(z.rail)} by train`]);
      if (regional) label(`${z.name} · ${count(z.all)} ${railOnly ? "riders" : "trips"}/day`);
    });
    marks = { discs, tags };
    view.setDiscs("inter", discs);
    placeLabels();
  };
  effect(() => {
    display.value.bubbleSize;
    display.value.commuters;
    purposeFilter.value;
    render();
  });
  map.on("zoom", render);
  // the overview is scaled to the zones on screen (T-276)
  map.on("moveend", () => { if (cityOverview.peek()) render(); });
  map.on("move", placeLabels);
  map.on("resize", placeLabels);

  // a new market result: from a new network, the city bubbles catch up from what they show now;
  // a refined round of the same carries on (T-156)
  effect(() => {
    const r = intercityDemand.value;
    if (!r) return;
    const next = pairs(r.corridors);
    if (r.version !== marketVersion) {
      const fresh = market && marketVersion >= 0 && !clock.held.peek();
      // where the bubbles are now, drawn or not
      marketFrom = fresh ? drawnPairs(marketShare()) : null;
      marketStart = tripsDone(clock.now());
      marketF = fresh ? 0 : 1;
      marketVersion = r.version;
    }
    market = next;
    if (cityMode()) render();
  });
  // on the game clock's minutes, redraw the city bubbles when the share has moved a step
  effect(() => {
    clock.minute.value;
    if (!marketFrom || !cityMode() || !cityOverview.peek()) return;
    const f = marketShare();
    if (f >= 1 || f - marketF >= CATCH_STEP) render();
  });

  // under the commuter discs (order 10): where both are under the pointer, the commuters answer
  addFinder(5, (x, y) => {
    if (!marks) return null;
    const i = discAt(marks.discs, x, y, view.bubbleSize);
    return i < 0 ? null : marks.tags[i];
  });

  // ---- what is selected ----
  let seen: Selection = null;
  let seenScope = "";
  let ask = 0;
  const show = (flows: PlaceFlows, verb: string, railOnly: boolean, own: number) => {
    current = { flows, verb, railOnly, own };
    render();
  };
  const clear = () => {
    current = null;
    intercityPick.value = null;
    intercityStationView.value = null;
    render();
  };
  effect(() => {
    const sel = selection.value;
    const dv = demandView.value;
    const focus = activeCity.value;
    // a station's, or its complex's in complex view (T-146)
    const scope = selectedScope.value;
    if ((sel?.kind !== "commuters" && sel?.kind !== "station") || !dv) {
      ask++;
      seen = null;
      clear();
      return;
    }
    if (seen !== sel || seenScope !== scope) {
      // a different selection: nothing of the last one's answer stays
      seen = sel;
      seenScope = scope;
      clear();
    }
    const mine = ++ask;
    void (async () => {
      if (sel.kind === "commuters") {
        const own = marketOf(focus);
        const r = await intercitySelection(sel.end, Uint32Array.from(sel.cells));
        // null: the workers moved on, and the next demandView asks again
        if (mine !== ask || !r || selection.peek() !== sel) return;
        intercityPick.value = { sel, data: r, own };
        // homes: where their trips go; jobs: where the trips to them come from
        show(sel.end === "home" ? r.starts : r.ends, sel.end === "home" ? "to" : "from", false, own);
      } else {
        const r = await intercityStation(sel.station, (scope || sel.station).split(","));
        if (mine !== ask || !r || selection.peek() !== sel) return;
        intercityStationView.value = r;
        show(r.starts, "to", true, marketOf(focus));
      }
    })();
  });

  /** A click with the select tool on a city bubble: that city selected (the selected one's own
   * bubble keeps the selection); undefined when no city bubble is there. */
  const click = (px: number, py: number): Selection | undefined => {
    if (!marks?.cities) return undefined;
    const [x, y] = pointerLocal(map, px, py);
    const i = discAt(marks.discs, x, y, view.bubbleSize);
    if (i < 0) return undefined;
    const sel = selection.peek();
    return sel?.kind === "city" && sel.city === marks.cities[i] ? sel : { kind: "city", city: marks.cities[i] };
  };
  return { click };
}
