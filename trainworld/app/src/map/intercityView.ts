// Longer trips on the map and in the inspectors (T-118, T-120, T-186, SPEC 8). The commuter views
// show where people go to work; this shows where the same places send the market's other, longer
// trips (visits, shopping, business; sim/src/demand/intercity.rs), which a commuter bubble cannot:
// a Philadelphia bubble's longer trips end in New York, far off the screen.
//
// - A bubble selection (a `commuters` selection) asks `intercitySelection(end, cells)`: how many
//   trips start in those cells and how many end there, by far zone and by place, all modes and by
//   train. The map shows the far end of the selection's own end (homes: where the trips go; jobs:
//   where they come from).
// - Zoomed far out (OVERVIEW_ZOOM and below) with Homes or Destinations on and no bubble
//   selection, every city is one bubble of its longer trips, the focused city too, in place
//   of the focused city's own bubbles (T-186). A click on one selects the city (a `city`
//   selection): it is drawn outlined, with one bubble per other city for the trips between them.
//   These come from the market's corridors (`intercityDemand`), each new result shown as it lands.
//
// Up close every far 6 km zone outside the focused market with a trip or more is shown, and those of
// the focused market from OWN_FAR_M out (T-4140). Regional zooms aggregate a
// selection's trips to one labeled circle per other market (T-135, T-136). Both are drawn as the
// selection's in-city far end is (T-1246): its area per trip, no outline, one colour by
// `bubbleColour(rail, walk, everything else)`. Under a local purpose filter only that purpose's
// everyday outings across a zone line are drawn (T-1229, `IntercitySelection.byPurpose`); the
// numbers stay in the inspectors.
// Asked for, not pushed, and asked again whenever `demandView` changes, keeping the last answer
// meanwhile.

import { t as tr, tSeries } from "../i18n";
import { tPlace } from "../i18n/places";
import { computed, effect, signal, untracked } from "@preact/signals";
import { mapBlank, zoneLoads } from "../game/loading";
import type { Map as MlMap } from "maplibre-gl";
import { activeCity, cityById, CITIES, type City } from "../game/cities";
import { shownRealms, zoneShown } from "../game/realms";
import { groundScale, lngLatToPack, packToLngLat } from "../game/coords";
import { askMarketTrip, marketKinds, openFarTrip, tripOpenedFrom } from "../game/trip";
import { cityDemandViews, demandView, intercityDemand, INTERCITY_PLACES, intercitySelection, rowsLanded, rowsOf, type IntercitySelection, type PlaceFlows } from "../game/demand";
import { builtCities } from "../game/worldTotals";
import { marketPairs, Z, zoneOf, zoneRowOf, type MarketPairs } from "../game/corridorFigures";
import { count } from "../game/format";
import { BUBBLE_ALPHA, bubbleColour, purposeColour3, purposeSchemeOn, rgb, setBubbleColours, setBubbleShade, START_TIER_FILL, UNBUILT_FILL } from "../game/palette";
import { openedZones } from "../game/openedZones";
import { lineSetUp, startTier, type StartTier } from "../game/startDifficulty";
import { PURPOSE_CLASS, purposeWords, type PurposeKey } from "../game/purposes";
import { display, purposeFilter, selection, mapShade, shownDayPhrase } from "../game/ui";
import type { CommutersOneWay, Selection } from "../game/types";
import { DISC, RES_AREA_M2 } from "../workers/commutersProtocol";
import { anchorVersion } from "./anchorVersion";
import { localModes, modeBar, purposeLines } from "./modeBar";
import { zoneTransit } from "../game/transitModes";
import { addFinder, discAt, pointerAt, refreshHover } from "./demandHover";
import { BUBBLE_BASE, bubbleLevel, LINE, splitEdge, type DemandOverlay } from "./demandOverlay";
import { labelFont, type LabelFont } from "./labelLayout";
import { anchor, kAtLat, pxPerUnitAt, toLocal, toLngLat, type Anchor } from "./geo";
import { hiddenOnGlobe } from "./live";
import { perfOff } from "../perfFlags";

/** A bubble selection's longer trips, for the commuters inspector; null while unknown. */
export const intercityPick = signal<{ sel: Selection; data: IntercitySelection; own: number } | null>(null);

const marketOf = (id: string) => cityById(id).place;

/** Two directions' longer trips as one (T-1903: a Total selection's starts and ends), by place and zone. */
function mergePlaces(a: PlaceFlows, b: PlaceFlows): PlaceFlows {
  const opt = (x?: number, y?: number) => (x === undefined && y === undefined ? undefined : (x ?? 0) + (y ?? 0));
  // the far ends' mean position, weighted by trips (NaN with none)
  const mean = (x: number, n: number, y: number, m: number) => (n > 0 && m > 0 ? (x * n + y * m) / (n + m) : n > 0 ? x : y);
  const places = new Map<number, PlaceFlows["places"][number]>();
  for (const p of [...a.places, ...b.places]) {
    const q = places.get(p.place);
    places.set(p.place, !q ? { ...p } : { ...q, all: q.all + p.all, rail: q.rail + p.rail, air: q.air + p.air, car: q.car + p.car, walk: opt(q.walk, p.walk),
      x: mean(q.x, q.all, p.x, p.all), y: mean(q.y, q.all, p.y, p.all) });
  }
  const zones = new Map<number, PlaceFlows["zones"][number]>();
  for (const z of [...a.zones, ...b.zones]) {
    const q = zones.get(z.gid);
    zones.set(z.gid, !q ? { ...z } : { ...q, all: q.all + z.all, rail: q.rail + z.rail, walk: opt(q.walk, z.walk) });
  }
  return { all: a.all + b.all, rail: a.rail + b.rail, air: a.air + b.air, car: a.car + b.car, walk: opt(a.walk, b.walk), elsewhere: a.elsewhere + b.elsewhere,
    places: [...places.values()].sort((p, q) => q.all - p.all), zones: [...zones.values()].sort((p, q) => q.all - p.all) };
}
/** the zones still loading (game/loading.ts `zoneLoads`), changing only when the set does */
const loadingZones = computed(() => [...zoneLoads.value.keys()].sort().join(","));

/** Regional views show the complete place totals, rather than scattered endpoint specks. */
export const CITY_BUBBLE_ZOOM = 9.5;
/** T-1246: a selection's far end in another zone is drawn on its in-city far end's area scale,
 * this many times the radius (Anita's figure): at 1 a far zone's few hundred trips beside a
 * selection's own tens of thousands are a pixel. */
const FAR_RADIUS = 2;
/** T-1246: the selection's in-city far end as drawn (map/commuterView.ts): square metres of bubble
 * per trip at the level on screen; 0 while none is drawn. */
export const farAreaPerTrip = signal(0);
/** T-1246: no bubble for a count its label would round to 0 (`count`). */
const shownCount = (n: number) => n >= 0.5;
/** T-2770: with a city selected, a pair of cities under this many trips a day is left out (bubble, line and list rows) */
export const MIN_PAIR_TRIPS = 10;
/**
 * At this zoom and below every city is one longer-trips bubble (T-186): the zoom at which the
 * metropolitan outlines are fully drawn, a New York region about half a 1,400 px map wide with
 * Philadelphia and Hartford on screen. The focused city's own bubbles are 6 km apart there.
 */
export const OVERVIEW_ZOOM = 8.5;
/** The overview's scale (T-525, T-530): trips a day per pixel of bubble area at `SCALE_ZOOM`, before
 * the size slider; it depends on the zoom only, never on what is on screen. New York's 71M is a 34 px
 * radius there (the Northeast on screen), a 1.6M prefecture 5 px. */
const TRIPS_PER_PX2 = 60000;
/** The same for a selected zone's inter-region trips only, which are far fewer: New York's own 1.6M
 * is a 33 px radius at `SCALE_ZOOM`, Philadelphia's 796k 24 px, a 15k partner 3 px. */
const SELECTED_TRIPS_PER_PX2 = 1400;
/** Bubbles' screen radius doubles every 1 / ZOOM_GROWTH zoom levels in (T-530): far less than the
 * map itself grows. */
const SCALE_ZOOM = 6.5;
const ZOOM_GROWTH = 0.35;
/** The lines from a selected zone to the others (T-520). */
const ZONE_LINE_ALPHA = 0.8;
/** The zones' pointers are this many times the neighbourhoods' length (T-739). */
const ZONE_STUB = 2;
/** Intercity bubbles drawn as a red dot with a blue dot inside, the blue's area the train share
 * (T-901); false brings back the solid disc coloured by the mix. */
const SPLIT_BUBBLES = true;
/** A bubble's edge value: its outline (0 none, 1 plain, 2 the selected one's), and under
 * `SPLIT_BUBBLES` the train share; a bubble nobody travels in stays a plain grey disc. */
const edgeOf = (outline: 0 | 1 | 2, rail: number, all: number) => (SPLIT_BUBBLES && all > 0 ? splitEdge(outline, rail / all) : outline);
/** An unopened zone's dot: a 1 px grey outline (the disc shader's 1.1-1.5 band, `UNBUILT_RIM`). */
const UNBUILT_EDGE = 1.25;
const UNBUILT_GROW = 1.25;
/** Most place labels on screen at once, the biggest first (T-493). */
const MAX_LABELS = 8;
/** T-4140: a selection's far 6 km zones in its own zone are drawn from this far from its busiest cell, metres:
 * where the local everyday outings have tapered out (the market's onset, twice ONSET_KM in sim intercity.rs). */
const OWN_FAR_M = 50_000;
/** The least space kept between two labels, CSS px sideways and up and down (T-1050). */
const LABEL_GAP_X = 28;
const LABEL_GAP_Y = 16;
/** No two labelled bubbles are closer than this, px: a label for one more in a cluster adds clutter. */
const LABEL_SEPARATION = 130;
/** Below this zoom (the world, a hemisphere at most) no city is drawn unasked, and the realms'
 * circles stay off (T-1046): the overview's bubbles clutter a map of the whole world. */
export const CITY_MARKS_MIN_ZOOM = 2.5;

/** T-1133: the least distance, CSS px, between an unopened zone's dot and any bigger zone's mark. */
const DOT_GAP = 20;

/** T-1133: leave out, in place, each unopened zone's dot within `DOT_GAP` of a bigger zone's mark
 * (`points` biggest first), so dense realms show their biggest cities zoomed out. */
function spaceDots<P extends { city: { center: [number, number] }; unbuilt: boolean }>(points: P[], zoom: number) {
  const W = 512 * 2 ** zoom, cell = new Map<number, [number, number][]>();
  const key = (i: number, j: number) => i * 1e6 + j;
  let n = 0;
  for (const p of points) {
    const [lon, lat] = p.city.center, s = Math.sin((Math.max(-85, Math.min(85, lat)) * Math.PI) / 180);
    const x = ((lon + 180) / 360) * W, y = (0.5 - Math.log((1 + s) / (1 - s)) / (4 * Math.PI)) * W;
    const i = Math.floor(x / DOT_GAP), j = Math.floor(y / DOT_GAP);
    let near = false;
    if (p.unbuilt) {
      for (let di = -1; di <= 1 && !near; di++) for (let dj = -1; dj <= 1 && !near; dj++) {
        near = (cell.get(key(i + di, j + dj)) ?? []).some(([u, v]) => (u - x) ** 2 + (v - y) ** 2 < DOT_GAP ** 2);
      }
    }
    if (near) continue;
    const k = key(i, j);
    (cell.get(k) ?? cell.set(k, []).get(k)!).push([x, y]);
    points[n++] = p;
  }
  points.length = n;
}

/** The city bubbles are drawn in place of the focused city's own: zoomed far out, or a city
 * selected (T-186). The commuter view hides its bubbles and leaves clicks and hover to these. */
export const cityOverview = signal(false);
/** The map is zoomed out to one bubble per zone (`OVERVIEW_ZOOM`), whatever the demand buttons say;
 * the buttons work there even over dead space (T-801). */
export const zoomedOut = signal(false);

/** A zone of the overview (`renderCities`): its trips, and how it is drawn and tagged. */
interface CityPoint { all: number; rail: number; air: number; walk: number; none: number; city: City; isOwn: boolean; tag: () => string[]; unbuilt: boolean; tier: StartTier | null }

interface Marks {
  discs: Float32Array;
  /** the draw anchor the discs are relative to (T-317) */
  anchor: Anchor;
  /** per disc, in disc order: the hover tag's lines, or what writes them */
  tags: (string[] | (() => string[]))[];
  /** per disc, the city it stands for, in the city view (T-186) */
  cities?: string[];
  /** the disc whose label is placed first (the selected city's) */
  first?: number;
  /** per disc, a bubble selection's far end it stands for (T-866: a click opens the trip
   * explorer): its place, its 6 km zone (-1 the whole place) and where it is, world metres */
  far?: { place: number; gid: number; at: [number, number] }[];
}

/** T-2136: how many zones' rows are asked ahead (game/demand.ts keeps 48 places' or sets') */
const PREFETCH_PLACES = 16;
/** the city view's market figures before any result (`MarketPairs.totals` all zero) */
let noPairs: Float64Array | null = null;

const discLngLats = new WeakMap<Marks, Float64Array>();
function lngLats(m: Marks): Float64Array {
  let a = discLngLats.get(m);
  if (a) return a;
  const n = m.discs.length / DISC;
  a = new Float64Array(2 * n);
  for (let i = 0; i < n; i++) {
    const ll = toLngLat(m.discs[i * DISC], m.discs[i * DISC + 1], m.anchor);
    a[2 * i] = ll[0]; a[2 * i + 1] = ll[1];
  }
  discLngLats.set(m, a);
  return a;
}

/** [west, south, east, north] of the discs whose labels can still be on the page (a screen's width
 * beyond each side, plus the biggest disc's radius), when the map is flat and north-up; else null. */
function flatBox(map: MlMap, W: number, H: number, m: Marks, k: number): [number, number, number, number] | null {
  if (map.getBearing() !== 0 || map.getPitch() !== 0 || (map.getProjection()?.type ?? "mercator") !== "mercator") return null;
  let rMax = 0;
  for (let i = 0; i < m.discs.length / DISC; i++) rMax = Math.max(rMax, m.discs[i * DISC + 2]);
  rMax *= k;
  const nw = map.unproject([-W - rMax, -H - rMax]), se = map.unproject([2 * W + rMax, 2 * H + rMax]);
  // a view that wraps past the antimeridian, or out of the poles, is left to the full pass
  if (nw.lng < -180 || se.lng > 180 || !(se.lat > -85) || !(nw.lat < 85)) return null;
  return [nw.lng, se.lat, se.lng, nw.lat];
}

export function startIntercityView(view: DemandOverlay, map: MlMap) {
  let marks: Marks | null = null;
  /** what is on show, redrawn when the setting changes */
  /** `parts`: a Total selection's two directions, trips starting here (home) and ending here (work) */
  let current: { flows: PlaceFlows; verb: string; own: number; by?: IntercitySelection["byPurpose"]; end?: "home" | "work"; parts?: { home: PlaceFlows; work: PlaceFlows } } | null = null;
  const labels = document.createElement("div");
  labels.className = "intercity-labels";
  map.getContainer().appendChild(labels);
  /** each disc's label, in disc order; an element only for each label placed (T-1026: the overview
   * has one per zone, and making them all took 10-20 ms a render) */
  let texts: (string | (() => string))[] = [];
  const placeLabels = () => {
    const used: { x: number; y: number; w: number; h: number; px: number; py: number }[] = [];
    if (marks) {
      // discs run biggest first, so the cap keeps the biggest on screen (T-493)
      const order = texts.map((_, i) => i);
      if (marks.first !== undefined && marks.first > 0) order.unshift(...order.splice(marks.first, 1));
      // T-871: a label off screen is passed over unmeasured, since each measure after a write lays
      // the page out again
      const W = map.getContainer().clientWidth, H = map.getContainer().clientHeight;
      let font: LabelFont | null | undefined;
      const lls = lngLats(marks), k = pxPerUnitAt(map.getZoom()) * view.bubbleSize;
      // T-2749: on the flat north-up map the discs beyond the margin below are found from their
      // longitude and latitude, before any is projected
      const box = flatBox(map, W, H, marks, k);
      for (const i of order) {
        if (used.length >= MAX_LABELS) break;
        const ll: [number, number] = [lls[2 * i], lls[2 * i + 1]];
        if (box && (ll[0] < box[0] || ll[0] > box[2] || ll[1] < box[1] || ll[1] > box[3])) continue;
        if (hiddenOnGlobe(ll[0], ll[1])) continue;
        const p = map.project(ll);
        const r = marks.discs[i * DISC + 2] * k;
        if (used.some((a) => Math.hypot(a.px - p.x, a.py - p.y) < LABEL_SEPARATION) || p.x < -r - W || p.x > W + r + W || p.y < -r - H || p.y > H + r + H) continue;
        let el = labels.children[used.length] as HTMLElement | undefined;
        if (!el) {
          el = document.createElement("div");
          el.className = "intercity-label";
          labels.appendChild(el);
        }
        const tx = texts[i], text = typeof tx === "function" ? (texts[i] = tx()) : tx;
        if (el.textContent !== text) el.textContent = text;
        const f = (font ??= labelFont(labels));
        // the canvas gives the size without the layout that a read after a write forces (T-2749); 2 px of padding each side
        const w = f ? Math.round(f.ctx.measureText(text.trim()).width) + 4 : el.offsetWidth, h = f ? f.h : el.offsetHeight;
        const positions = [[p.x + r + 5, p.y - h / 2], [p.x - w - r - 5, p.y - h / 2], [p.x - w / 2, p.y + r + 5], [p.x - w / 2, p.y - h - r - 5]];
        const at = positions.find(([x, y]) => x >= 0 && y >= 0 && x + w <= W && y + h <= H && !used.some(a => x < a.x + a.w + LABEL_GAP_X && x + w + LABEL_GAP_X > a.x && y < a.y + a.h + LABEL_GAP_Y && y + h + LABEL_GAP_Y > a.y));
        if (at) {
          el.style.transform = `translate(${Math.round(at[0])}px, ${Math.round(at[1])}px)`;
          used.push({ x: at[0], y: at[1], w, h, px: p.x, py: p.y });
        }
      }
    }
    while (labels.children.length > used.length) labels.lastChild!.remove();
  };
  const label = (text: string | (() => string)) => { texts.push(text); };
  /** T-2748: the overview's points, kept while what they are made from is the same (`inputs`), and
   * `spaced` their dots left after `spaceDots` at zoom `zq` tenths */
  let cityPoints: { inputs: unknown[]; all: CityPoint[]; zq: number; spaced: CityPoint[] } | null = null;

  // ---- the market's corridors ----
  let market: MarketPairs | null = null;
  // T-4606: a zone pair's three kinds of trips as the trip explorer counts them (the workers' answer
  // for the pair, asked when it is first hovered); null: not answerable, so the hover keeps the market's two kinds
  let pairKinds: { src: MarketPairs | null; got: Map<string, ReturnType<typeof marketKinds> | null | "asking"> } = { src: null, got: new Map() };
  const pairKindsOf = (a: number, b: number) => {
    if (pairKinds.src !== market) pairKinds = { src: market, got: new Map() };
    const key = `${a}:${b}`, had = pairKinds.got.get(key);
    if (had !== undefined) return had;
    const mine = pairKinds;
    mine.got.set(key, "asking");
    void askMarketTrip("home", new Uint32Array(0), b, -1, { place: a, gid: -1 }).then((ps) => {
      mine.got.set(key, ps ? marketKinds(ps, 2) : null);
      if (pairKinds === mine) refreshHover();
    });
    return "asking" as const;
  };
  /** a selected zone's rows were asked for and are not in yet */
  let rowWanted = false;

  /** What the city view draws: the overview, a selected city, or neither. */
  const cityMode = (): "overview" | "selected" | null => {
    const sel = selection.peek();
    const out = map.getZoom() <= OVERVIEW_ZOOM;
    // T-885: zoomed in, a selected city stays selected under the local bubbles, so a click on one
    // explores the trips between them
    if (sel?.kind === "city") return out ? "selected" : null;
    if (display.peek().commuters && out && sel?.kind !== "commuters") return "overview";
    return null;
  };

  /** One bubble per city, or a selected city and one bubble per other city (T-186). */
  const renderCities = (selected: string | null) => {
    const a = anchor();
    // T-801: before any market result (a start on the world map) the overview is the local trips
    // alone, and stays while a zone is selected, since its pairs need a solve
    const pairsIn = market;
    if (!pairsIn) selected = null;
    const own = selected ? cityById(selected) : null;
    const totals = pairsIn?.totals ?? (noPairs ??= new Float64Array(Z * 3));
    const ownZone = own ? zoneOf(own.place) : -1;
    // T-2136: the selected zone's rows come from a worker; drawn again as they land (`rowWanted`)
    const row = pairsIn && ownZone >= 0 ? zoneRowOf(pairsIn, ownZone) : null;
    rowWanted = !!pairsIn && ownZone >= 0 && !row;
    const at = (m: Float64Array | null, z: number) => (!m || z < 0 ? { all: 0, rail: 0, air: 0 } : { all: m[3 * z], rail: m[3 * z + 1], air: m[3 * z + 2] });
    // (only ever the selected zone's pairs)
    const pair = (a: number, b: number) => at(zoneOf(a) === ownZone ? row : null, zoneOf(b));
    const total = (a: number) => at(totals, zoneOf(a));
    // T-3503: a zone's everyday outings across its lines on foot
    const interWalk = (a: number) => (pairsIn && zoneOf(a) >= 0 ? pairsIn.walk[zoneOf(a)] : 0);
    // T-276: zoomed out, a zone is one bubble as if it were one place: its local trips (the city
    // view's, or with no rail the pipeline's count for a zone not opened) and the regional
    // layer's trips to and from it (commutes between zones and longer trips)
    const local = cityDemandViews.peek();
    const opened = openedZones.peek();
    const tiered = !lineSetUp.peek();
    const loads = zoneLoads.peek();
    const day = shownDayPhrase.peek();
    // T-2748: a zoom step reuses the points and, within a tenth of a zoom level, their spacing; only the discs change
    const inputs = [pairsIn, own, row, local, opened, tiered, loads, day, shownRealms.peek()];
    const zoom = map.getZoom();
    const zq = Math.round(zoom * 10);
    if (!cityPoints || perfOff("overviewCache") || inputs.some((x, k) => x !== cityPoints!.inputs[k])) {
      cityPoints = { inputs, all: CITIES.flatMap((c) => {
      const isOwn = c === own;
      let t = { ...(!own || isOwn ? total(c.place) : pair(own.place, c.place)), walk: 0, none: 0 };
      if (!own) {
        // T-576: a zone's local walking is green; a zone not opened has no walk split (its walk is
        // inside `all`, so it draws as the rest)
        const v = local.get(c.id), lm = localModes(v);
        t = { all: t.all + (v ? v.tripsPerDay : c.localTrips ?? 0), rail: t.rail + (v ? v.railTripsPerDay : 0), air: t.air, walk: (lm?.walk ?? 0) + interWalk(c.place), none: lm?.none ?? 0 };
      }
      // T-421: a zone still loading has its bar instead
      // T-975: the overview leaves out the realms not shown; a selection's far ends draw everywhere
      if (!isOwn && !((own ? t.all >= MIN_PAIR_TRIPS : t.all > 0) && (own ? true : zoneShown(c.id) && !loads.has(c.id)))) return [];
      const unbuilt = !own && opened !== null && !opened.has(c.id);
      // T-2690: until a line is set up an unopened zone's dot is coloured by its start tier
      const tier = unbuilt && tiered ? startTier(c.id) : null;
      // T-494: a zone's local and inter-region trips are two figures, each with a mode bar; written
      // when hovered (T-2748: writing every zone's on each zoom step was most of a step)
      const pairTag = (a: City, b: City, old: { all: number }, oldBar: string) => {
        const k = pairKindsOf(a.place, b.place);
        const between = (n: number) => tr("intercityView.between", { count: count(n), when: day, a: tPlace(a.id, a.name), b: tPlace(b.id, b.name) });
        if (k === "asking") return [tr("cityInspector.calculating"), tr("intercityView.explore")];
        if (!k) return [between(old.all), oldBar, tr("intercityView.explore")];
        const kinds = ([["inter", k.long], ["commute", k.commute], ["everyday", k.everyday]] as [string, number][]).filter(([, n]) => n >= 0.5)
          .map(([kind, amount]) => tr("intercityView.kind", { kind, amount: count(amount) }));
        return [between(k.total), modeBar(k.rail, k.walk, Math.max(0, k.total - k.rail - k.walk), 0, 1, null), tSeries(kinds), tr("intercityView.explore")];
      };
      const tag = () => {
        const v = local.get(c.id);
        const localAll = v ? v.tripsPerDay : c.localTrips ?? 0;
        const localLine = tr("intercityView.within", { count: count(localAll), when: day, name: tPlace(c.id, c.name) });
        const lm = localModes(v);
        const localBar = lm ? modeBar(lm.train, lm.walk, lm.drive, lm.none, 1, zoneTransit(v, c.id)) : "";
        const inter = !own || isOwn ? total(c.place) : t;
        const interLine = tr("intercityView.interCity", { count: count(inter.all), when: day, name: tPlace(c.id, c.name) });
        const iw = !own || isOwn ? interWalk(c.place) : 0;
        // T-3695: a pair of zones has no split of its own by kind of line, so its bar keeps one transit part
        const interBar = modeBar(inter.rail, iw, Math.max(0, inter.all - inter.rail - iw), 0, 1, !own || isOwn ? zoneTransit(demandView.peek(), c.id, true) : null);
        // T-3074: before any market result the inter-city figure is unknown, not 0: leave its row out
        // T-3658: a 0 just means it is not calculated yet: leave the row out then too
        const known = pairsIn && Math.round(inter.all) > 0;
        return !own
          ? [localLine, localBar, ...(known ? [interLine, interBar] : []), ...(tier ? [tr("intercityView.startTier", { tier })] : [])]
          : isOwn
            ? (known ? [interLine, interBar, `(${localLine})`] : [localLine, localBar])
            : pairTag(own, c, t, interBar);
      };
      return [{ ...t, city: c, isOwn, tag, unbuilt, tier }];
    }).sort((a, b) => b.all - a.all), zq: NaN, spaced: [] };
    }
    const cp = cityPoints;
    if (cp.zq !== zq) {
      cp.zq = zq;
      cp.spaced = cp.all.slice();
      if (!own) spaceDots(cp.spaced, zoom);
    }
    const points = cp.spaced;
    const pxPerUnit = pxPerUnitAt(zoom);
    const zoomScale = 2 ** (ZOOM_GROWTH * (zoom - SCALE_ZOOM));
    const discs = new Float32Array(points.length * DISC);
    const dot = rgb(UNBUILT_FILL);
    const tierDot = { easy: rgb(START_TIER_FILL.easy), medium: rgb(START_TIER_FILL.medium), hard: rgb(START_TIER_FILL.hard) };
    const size = display.peek().bubbleSize * BUBBLE_BASE;
    points.forEach((p, i) => {
      const [x, y] = toLocal(p.city.center[0], p.city.center[1], a);
      const c = p.unbuilt ? (p.tier ? tierDot[p.tier] : dot) : bubbleColour(p.rail, p.walk, Math.max(0, p.all - p.rail - p.walk - p.none), p.none);
      let r = Math.sqrt(p.all / (own ? SELECTED_TRIPS_PER_PX2 : TRIPS_PER_PX2)) * zoomScale / pxPerUnit;
      // T-976: a zone not opened is a dot of half the radius it would have, in whole CSS px, at least 2
      // so its 1 px outline leaves some fill; then a quarter bigger (T-1886: beside the opened zones' bubbles it read small)
      if (p.unbuilt) r = (UNBUILT_GROW * Math.max(2, Math.floor((r * pxPerUnit * size) / 2))) / (pxPerUnit * size);
      discs.set([x, y, r, c[0] / 255, c[1] / 255, c[2] / 255, p.unbuilt ? 1 : BUBBLE_ALPHA, p.unbuilt ? UNBUILT_EDGE : edgeOf(p.isOwn ? 2 : 1, p.rail, p.all)], i * DISC);
      label(() => tr("intercityView.label", { name: tPlace(p.city.id, p.city.name), count: count(p.all) }));
    });
    const first = points.findIndex((p) => p.isOwn);
    marks = { discs, anchor: a, tags: points.map((p) => p.tag), cities: points.map((p) => p.city.id), first: first >= 0 ? first : undefined };
    view.setDiscs("inter", discs, a);
    // a thin line from the selected zone to every other, so the selection shows from afar (T-520)
    if (own && first >= 0) {
      const lines = new Float32Array((points.length - 1) * LINE);
      const o = first * DISC;
      for (let i = 0, k = 0; i < points.length; i++) {
        const s = i * DISC;
        if (i !== first) lines.set([discs[s], discs[s + 1], discs[s + 2], discs[o], discs[o + 1], discs[o + 2], ZONE_STUB, ZONE_LINE_ALPHA, discs[s + 3], discs[s + 4], discs[s + 5]], k++ * LINE);
      }
      view.setLines("inter", lines, a);
    }
  };

  /** The marks and labels of what is on show; the labels follow the marks (`placeLabels`). */
  const render = () => {
    build();
    placeLabels();
  };
  const build = () => {
    marks = null;
    texts = [];
    view.setLines("inter", null);
    // T-1006, T-1051: a blank map behind the start menu and while a game loads; the overview's place keeps the local bubbles off too
    if (mapBlank.peek()) {
      cityOverview.value = true;
      view.setDiscs("inter", null);
      return;
    }
    // the regional market has no purposes: its bubbles show for all purposes only (T-158, T-305);
    // the inspector keeps the numbers
    const filter = purposeFilter.peek();
    // (always drawn otherwise: the setting to hide them went with T-222)
    const shown = filter === "all";
    const mode = cityMode();
    // a selected city always takes the place of the city's own bubbles; zoomed far out only
    // when the overview has something to draw
    cityOverview.value = mode === "selected" || (mode === "overview" && shown);
    // the world view: nothing, and the local bubbles stay off (`cityOverview`); a selected city stays
    if (mode === "overview" && map.getZoom() < CITY_MARKS_MIN_ZOOM) {
      view.setDiscs("inter", null);
      return;
    }
    if (mode && shown) {
      const sel = selection.peek();
      renderCities(sel?.kind === "city" ? sel.city : null);
      return;
    }
    // T-1229: a local purpose filter draws that purpose's own everyday outings across a zone line
    // as the far end (the longer trips stay under all purposes)
    const flows = !current ? undefined : shown ? current.flows : current.by?.get(filter as PurposeKey)?.[current.end === "work" ? "ends" : "starts"];
    if (!current || !flows || mode) {
      view.setDiscs("inter", null);
      return;
    }
    const regional = map.getZoom() <= CITY_BUBBLE_ZOOM;
    const a = anchor();
    // T-4140: the zone's own far parts are drawn too, but only past where its local far end reaches (T-139's overlap)
    const sel = selection.peek();
    const near = sel?.kind === "commuters" && nearPointOf ? nearPointOf(sel) : null;
    const ownFar = (z: { x: number; y: number }) => !!near && Math.hypot(z.x - near[0], z.y - near[1]) / groundScale((z.y + near[1]) / 2) >= OWN_FAR_M;
    const points = regional ? flows.places.filter(p => shownCount(p.all) && p.place !== current!.own).map(p => {
      const city = CITIES.find(c => c.place === p.place);
      const ll = city?.center ?? packToLngLat(p.x, p.y);
      return { ...p, ll, name: INTERCITY_PLACES[p.place], gid: -1 };
    }) : flows.zones.filter(z => shownCount(z.all) && (z.place !== current!.own || ownFar(z))).map(z => ({...z, ll: packToLngLat(z.x, z.y), name: ""}));
    points.sort((a, b) => b.all - a.all);
    const { verb } = current;
    const trips = shown ? tr("intercityView.interCityTrips") : tr("intercityView.purposeTrips", { purpose: purposeWords(filter as PurposeKey) });
    // T-1246: the in-city far end's scale (its busiest bubble about its hexagon's area), so a far
    // zone's bubble reads against them; with none drawn yet, the busiest of these takes that size
    const perTrip = farAreaPerTrip.peek() || (0.6 * RES_AREA_M2[bubbleLevel(map.getZoom())]) / (points[0]?.all || 1);
    // biggest first, so the small ones draw on top of the big ones
    const n = points.length;
    const discs = new Float32Array(n * DISC);
    const tags: string[][] = [];
    // T-2892, Purpose scheme: school, and errands and leisure, among these trips from `current.by`, and the longer trips
    // the rest; commutes are not in this far end. With no `by`, mode colours and a mode bar.
    const byPurpose = purposeSchemeOn() && (!shown || !!current.by);
    const schoolAt = new Map<string, number>(), errandsAt = new Map<string, number>();
    if (byPurpose && shown) {
      const dir = current.end === "work" ? "ends" : "starts";
      for (const [k, into] of [["school", schoolAt], ["shopping", errandsAt], ["social", errandsAt]] as const) {
        const part = current.by?.get(k)?.[dir];
        if (!part) continue;
        for (const q of regional ? part.places : part.zones) {
          const key = `${q.place}:${"gid" in q ? q.gid : -1}`;
          into.set(key, (into.get(key) ?? 0) + q.all);
        }
      }
    }
    const cls = PURPOSE_CLASS[filter];
    points.forEach((z, i) => {
      const [lng, lat] = z.ll;
      const [x, y] = toLocal(lng, lat, a);
      const walk = z.walk ?? 0, rest = Math.max(0, z.all - z.rail - walk);
      const at = `${z.place}:${z.gid}`;
      const w = shown ? Math.min(z.all, schoolAt.get(at) ?? 0) : cls === "work" ? z.all : 0;
      const e = shown ? Math.min(z.all - w, errandsAt.get(at) ?? 0) : cls === "errands" ? z.all : 0;
      const c = byPurpose ? purposeColour3(w, z.all - w - e, e) : bubbleColour(z.rail, walk, rest);
      const r = FAR_RADIUS * Math.sqrt((perTrip * z.all) / Math.PI) * kAtLat(lat);
      discs.set([x, y, r, c[0] / 255, c[1] / 255, c[2] / 255, BUBBLE_ALPHA, 0], i * DISC);
      const place = z.name || tr("intercityView.here");
      const split = byPurpose ? purposeLines({ work: w, other: z.all - w - e, errands: e }) : [modeBar(z.rail, walk, rest)];
      tags.push([tr("intercityView.farTag", { verb, count: count(z.all), trips, when: shownDayPhrase.peek(), place }), ...split, tr("intercityView.explore")]);
      if (regional) label(tr("intercityView.label", { name: z.name, count: count(z.all) }));
    });
    const far = points.map((z) => { const p = lngLatToPack(z.ll[0], z.ll[1]); return { place: z.place, gid: z.gid, at: [p.x, p.y] as [number, number] }; });
    marks = { discs, anchor: a, tags, far };
    view.setDiscs("inter", discs, a);
  };
  effect(() => {
    display.value.bubbleSize;
    display.value.commuters;
    mapShade.value;
    openedZones.value; // T-976: a zone opening turns its dot into a bubble
    lineSetUp.value; // T-2690: the dots' start tier colours go with the first line set up
    setBubbleShade(mapShade.value);
    setBubbleColours(display.value.bubbleColours, display.value.rampFrom, display.value.rampTo);
    purposeFilter.value;
    anchorVersion.value; // the discs are measured from the draw anchor (T-317)
    loadingZones.value;
    shownRealms.value;
    mapBlank.value;
    farAreaPerTrip.value;
    render();
  });
  map.on("zoom", render);
  const zoomed = () => { const z = map.getZoom() <= OVERVIEW_ZOOM; if (z !== zoomedOut.peek()) zoomedOut.value = z; };
  map.on("zoom", zoomed);
  zoomed();
  map.on("move", placeLabels);
  map.on("resize", placeLabels);

  effect(() => {
    const r = intercityDemand.value;
    if (!r) return;
    market = marketPairs(r.corridors, r.everyday);
    if (cityMode()) untracked(render);
    // T-2136: the zone in view's rows and the built zones', asked as a market's book comes (kept by its
    // key, so once a market), since a question waits for a worker's next break while trips are calculated
    const c = r.corridors;
    untracked(() => {
      const ids = [activeCity.peek(), ...builtCities().keys()];
      const places = [...new Set(ids.map((id) => cityById(id)?.place ?? -1).filter((p) => p >= 0))];
      for (const p of places.slice(0, PREFETCH_PLACES)) rowsOf(c, p);
    });
  });
  effect(() => {
    void rowsLanded.value;
    if (rowWanted && cityMode()) untracked(render);
  });

  // under the commuter discs (order 10): where both are under the pointer, the commuters answer
  addFinder(5, (p) => {
    if (!marks) return null;
    const i = discAt(marks.discs, p, marks.anchor, view.bubbleSize);
    const tag = i < 0 ? null : marks.tags[i];
    return typeof tag === "function" ? tag() : tag;
  });

  // ---- what is selected ----
  let seen: Selection = null;
  let ask = 0;
  const show = (flows: PlaceFlows, verb: string, own: number, by?: IntercitySelection["byPurpose"], end?: "home" | "work", parts?: { home: PlaceFlows; work: PlaceFlows }) => {
    current = { flows, verb, own, by, end, parts };
    render();
  };
  const clear = () => {
    current = null;
    intercityPick.value = null;
    render();
  };
  effect(() => {
    const sel = selection.value;
    const dv = demandView.value;
    const focus = activeCity.value;
    if (sel?.kind !== "commuters" || !dv) {
      ask++;
      seen = null;
      clear();
      return;
    }
    if (seen !== sel) {
      // a different selection: nothing of the last one's answer stays
      seen = sel;
      clear();
    }
    const mine = ++ask;
    void (async () => {
      const own = marketOf(focus);
      const cells = Uint32Array.from(sel.cells);
      // T-2746: a selection gone by the time its far end is on its way asks the workers nothing
      const still = () => selection.peek() === sel;
      if (sel.end === "both") {
        // T-1903: Total draws the trips starting here and those ending here as one far end
        const [h, w] = await Promise.all([intercitySelection("home", cells, still), intercitySelection("work", cells, still)]);
        if (mine !== ask || !h || !w || selection.peek() !== sel) return;
        const by = new Map<PurposeKey, { starts: PlaceFlows; ends: PlaceFlows }>();
        for (const k of new Set([...(h.byPurpose?.keys() ?? []), ...(w.byPurpose?.keys() ?? [])])) {
          const s = h.byPurpose?.get(k)?.starts, e = w.byPurpose?.get(k)?.ends;
          const m = s && e ? mergePlaces(s, e) : (s ?? e)!;
          by.set(k, { starts: m, ends: m });
        }
        const all = mergePlaces(h.starts, w.ends);
        intercityPick.value = { sel, data: { ...h, starts: all, ends: all, byPurpose: by }, own };
        show(all, "both", own, by, "home", { home: h.starts, work: w.ends });
        return;
      }
      const r = await intercitySelection(sel.end, cells, still);
      // null: the workers moved on, and the next demandView asks again
      if (mine !== ask || !r || selection.peek() !== sel) return;
      intercityPick.value = { sel, data: r, own };
      // homes: where their trips go; jobs: where the trips to them come from
      show(sel.end === "home" ? r.starts : r.ends, sel.end === "home" ? "to" : "from", own, r.byPurpose, sel.end);
    })();
  });

  /** A click with the select tool on a city bubble: that city selected (the selected one's own
   * bubble keeps the selection); undefined when no city bubble is there. */
  const click = (px: number, py: number): Selection | undefined => {
    const sel = selection.peek();
    // T-866: with a bubble selection, a far place or 6 km zone opens the trip explorer between them
    if (marks?.far && sel?.kind === "commuters" && nearPointOf) {
      const i = discAt(marks.discs, pointerAt(map, px, py), marks.anchor, view.bubbleSize);
      const f = i >= 0 ? marks.far[i] : undefined;
      if (!f) return undefined;
      const one = oneWay(sel, [f]);
      const near = nearPointOf(one);
      if (!near) return undefined;
      return openFarTrip(one, sel, near, f.at, { place: f.place, gid: f.gid });
    }
    if (!marks?.cities) return undefined;
    const i = discAt(marks.discs, pointerAt(map, px, py), marks.anchor, view.bubbleSize);
    if (i < 0) return undefined;
    if (sel?.kind === "city" && sel.city === marks.cities[i]) return sel;
    // T-885: with a city selected, another city's bubble opens the trip explorer between the two
    if (sel?.kind === "city") return placeTrip(sel, [cityById(marks.cities[i]).place]) ?? sel;
    return { kind: "city", city: marks.cities[i] };
  };
  /** T-885: the market's trips between a selected city and other places, as one far end. */
  const placeTrip = (sel: Extract<Selection, { kind: "city" }>, places: number[]): Selection | null => {
    const own = cityById(sel.city);
    const fars = places.filter((p) => p !== own.place).map((place) => ({ place, gid: -1 }));
    if (!fars.length) return null;
    const centre = (place: number): [number, number] => {
      const ll = CITIES.find((c) => c.place === place)?.center ?? own.center;
      const p = lngLatToPack(ll[0], ll[1]);
      return [p.x, p.y];
    };
    const trip: Selection = { kind: "trip", from: centre(own.place), to: centre(fars[0].place), far: fars[0], near: { place: own.place, gid: -1 } };
    tripOpenedFrom.value = { trip, from: sel, places: { end: "home", near: new Uint32Array(0), far: new Uint32Array(0), ...(fars.length > 1 ? { fars } : {}) } };
    return trip;
  };
  /** T-881, T-885: the inter-region bubbles whose centres are in a box (screen px): a bubble
   * selection's far places or 6 km zones, or the city view's other cities, biggest first. */
  const inBox = (x0: number, y0: number, x1: number, y1: number): { place: number; gid: number }[] => {
    if (!marks) return [];
    const out: { place: number; gid: number }[] = [];
    for (let i = 0; i < marks.discs.length / DISC; i++) {
      const ll = toLngLat(marks.discs[i * DISC], marks.discs[i * DISC + 1], marks.anchor);
      if (hiddenOnGlobe(ll[0], ll[1])) continue;
      const p = map.project(ll);
      if (p.x < Math.min(x0, x1) || p.x > Math.max(x0, x1) || p.y < Math.min(y0, y1) || p.y > Math.max(y0, y1)) continue;
      if (marks.far) out.push({ place: marks.far[i].place, gid: marks.far[i].gid });
      else if (marks.cities) out.push({ place: cityById(marks.cities[i]).place, gid: -1 });
    }
    return out;
  };
  interBox = (x0, y0, x1, y1) => {
    const sel = selection.peek();
    const ends = inBox(x0, y0, x1, y1);
    if (!ends.length) return null;
    if (sel?.kind === "city") return placeTrip(sel, ends.map((e) => e.place));
    if (sel?.kind !== "commuters" || !marks?.far || !nearPointOf) return null;
    const one = oneWay(sel, ends);
    const near = nearPointOf(one);
    if (!near) return null;
    const first = marks.far.find((f) => f.place === ends[0].place && f.gid === ends[0].gid)!;
    return openFarTrip(one, sel, near, first.at, ends[0], ends);
  };
  /** T-1903: a Total selection opens the trip explorer, which counts one way, as homes or
   * destinations: whichever has more of its trips with `fars` */
  const oneWay = (sel: Extract<Selection, { kind: "commuters" }>, fars: { place: number; gid: number }[]): CommutersOneWay => {
    if (sel.end !== "both") return { ...sel, end: sel.end };
    const n = (pf: PlaceFlows) => fars.reduce((s, f) => s + ((f.gid >= 0 ? pf.zones.find((z) => z.gid === f.gid)?.all : pf.places.find((q) => q.place === f.place)?.all) ?? 0), 0);
    const p = current?.parts;
    return { ...sel, end: p && n(p.work) > n(p.home) ? "work" : "home" };
  };
  return { click };
}

/** T-881, T-885: a Shift box's inter-region bubbles (screen px) as a trip, from the commuter
 * view's gesture; null when the box holds none. Set by `startIntercityView`. */
export let interBox: (x0: number, y0: number, x1: number, y1: number) => Selection | null = () => null;
/** The commuter view's `nearPoint`, for `interBox`. */
let nearPointOf: ((sel: Extract<Selection, { kind: "commuters" }>) => [number, number] | null) | null = null;
export const setNearPoint = (f: typeof nearPointOf) => (nearPointOf = f);
