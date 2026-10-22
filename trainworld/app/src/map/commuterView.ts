// Commuter bubbles (T-097, SPEC 8), Anita's Subway Builder style view: where commuters live (or
// work), as bubbles sized by people and coloured by how they travel, aggregated to H3 parents by
// zoom. A click on a bubble, or a long press and a drag over several, selects them (a
// `commuters` selection, so Escape and an empty click clear it): the map then shows the selected
// bubbles outlined and where those people work (or live) as bubbles, and the inspector lists them.
//
// Data is asked for, not pushed: `cellModes()` when the demand result changes (a solve's first
// estimate and its refined result, not the crowding rounds between), `flows(end, cells)` when the
// selection or the demand result changes. Whole-city aggregation runs in
// workers/commuters.worker.ts; the selection's few thousand cells are summed here. The bubbles
// show each new result as soon as it lands (T-906).

import { computed, effect, signal } from "@preact/signals";
import type { Map as MlMap } from "maplibre-gl";
import { activeCity, cityById, getWorldFrame, worldFrameEpoch, type CityId } from "../game/cities";
import { lngLatToPack } from "../game/coords";
import { nearbyZones } from "../game/zones";
import { cellModes, cityCells, demandUpdating, demandView, flows, type CellModes, type CityCells, type DemandView, type FarEnd, type Flows, type PurposeTrips } from "../game/demand";
import { count, wholeDistance } from "../game/format";
import { BUBBLE_ALPHA, bubbleSchemeKey, setBubbleColours } from "../game/palette";
import { dealFlows } from "../game/farChunks";
import { GROUP_SAMPLE_SIDE, marketGroup, tripOpenedFrom, type TripPlaces } from "../game/trip";
import { compass, stationNameNear } from "../game/places";
import { purposeWords, type PurposeFilter } from "../game/purposes";
import type { Selection } from "../game/types";
import { display, purposeFilter, selection, shownDayPhrase, tool } from "../game/ui";
import { world } from "../game/world";
import { BUBBLE_DENSITY, DISC, RES_AREA_M2, type Bubbles, type FromCommuters, type Moved, type Ready, type Summed, type ToCommuters } from "../workers/commutersProtocol";
import { aggregate, discs, recolour, type Groups } from "./bubbles";
import { cityOverview, interBox, setNearPoint } from "./intercityView";
import { anchorVersion } from "./anchorVersion";
import { modeBar } from "./modeBar";
import { addFinder, discAt, pointerAt, pointerLocal, type Pointer } from "./demandHover";
import { bubbleLevel, LINE, type DemandOverlay } from "./demandOverlay";
import { anchor, anchorShift, kAtY, toLocal, type Anchor } from "./geo";


export interface Place {
  name: string;
  riders: number;
}
/** A bubble selection as the inspector shows it. `far` is null while its far end is being asked
 * for,. */
export const commuterPick = signal<{ sel: Selection; totals: [number, number, number]; purposes?: PurposeTrips[]; far: { totals: [number, number, number]; places: Place[] } | null;
  /** the map's purpose filter the totals and far end count (T-158) */
  purpose?: PurposeFilter;
  /** the selection's centre, longitude and latitude (T-323) */
  at?: [number, number] } | null>(null);

/** "trips", or the purpose filter's trips (T-158), for the hover tags. */
const tripWords = () => {
  const p = purposeFilter.peek();
  return p === "all" ? "trips" : `${purposeWords(p)} trips`;
};

/** With a bubble selection, a short faint line from each far-end bubble toward the selection
 * (T-520, Anita: to try and maybe undo; false turns them off). */
const LINES_TO_SELECTION = true;
const STUB_ALPHA = 0.8;
/** T-865: a group of bubbles is named by the station nearest its centre within this, m. */
const GROUP_NAME_M = 3000;

/** One stub per far disc (`LINE` floats, relative to `farA`), pointing at the selection's centre:
 * its discs' centres weighted by area. A single selected disc's edge stops the stub. */
function linesToward(near: Float32Array | null, nearA: Anchor, far: Float32Array, farA: Anchor): Float32Array | null {
  if (!near || !near.length) return null;
  let sx = 0, sy = 0, sw = 0;
  for (let i = 0; i < near.length; i += DISC) {
    const w = near[i + 2] ** 2;
    (sx += near[i] * w), (sy += near[i + 1] * w), (sw += w);
  }
  if (!(sw > 0)) return null;
  const [dx, dy] = anchorShift(nearA, farA);
  const tx = sx / sw + dx, ty = sy / sw + dy, tr = near.length === DISC ? near[2] : 0;
  const n = far.length / DISC, out = new Float32Array(n * LINE);
  for (let i = 0; i < n; i++) out.set([far[i * DISC], far[i * DISC + 1], far[i * DISC + 2], tx, ty, tr, 1, STUB_ALPHA, far[i * DISC + 3], far[i * DISC + 4], far[i * DISC + 5]], i * LINE);
  return out;
}

/** A selection's far end under the purpose filter (T-158); null: none of that purpose. */
function farOf(f: Flows, purpose: PurposeFilter): FarEnd | null {
  return purpose === "all" ? f : f.byPurpose?.get(purpose) ?? null;
}


export const commuterStats = { nearbyMs: 0, nearbyBuilds: 0, nearbyShown: {} as Record<string, number>, cellsMs: 0, workerMs: 0, roundTripMs: 0, builds: 0, uploadMs: 0, selectMs: 0, flowsMs: 0, farMs: 0, farCells: 0, sumMs: 0,
  /** T-285: performance.now() when a selection's far end was asked for, its answer came, the
   * bubble worker's sums came and the far end was handed to the overlay */
  pickAt: 0, flowsAt: 0, summedAt: 0, farAt: 0,
  /** ?debug=1 (T-317): the city's bubbles on show, with the anchor their positions are relative to */
  probe: null as null | (() => { anchor: Anchor; levels: { discs: Float32Array; group: Uint32Array }[] } | null),
  farProbe: null as null | (() => { anchor: Anchor; discs: Float32Array; group: Uint32Array; train: Float32Array; walk: Float32Array; drive: Float32Array } | null) };


export function startCommuterView(map: MlMap, view: DemandOverlay) {
  let focusSeen = activeCity.peek(), focusEpoch = 0, frameSeen = worldFrameEpoch.peek();
  let city: Ready | null = null;
  let worker: Worker | null = null;
  let workerReady: Promise<boolean> | null = null;
  /** each cell's world metres (the trip explorer's points, T-208) */
  let cellsWorld: Float32Array | null = null;
  /** a far end as drawn: grouped into at least `FAR_CHUNK` trips a bubble (T-909) */
  const dealtOf = (f: Flows, purpose: PurposeFilter) => (cellsWorld ? dealFlows(f, purpose, cellsWorld) : f);
  let nextId = 1;
  /** replies of every bubble worker (the focused zone's and the nearby zones'), by message id */
  const pending = new Map<number, (b: any) => void>();
  // Positions are measured from the draw anchor (map/geo.ts, T-317). A worker is told the anchor in
  // force with its cells and again whenever it has changed (`sync`, before anything is asked of it),
  // and every answer says which anchor it is relative to. Its cells come back each time (`Moved`),
  // for the selection's discs made here.
  const readies = new WeakMap<Worker, Ready>();
  const told = new WeakMap<Worker, number>();
  const sync = (w: Worker) => {
    const a = anchor();
    if (told.get(w) === a.version) return;
    told.set(w, a.version);
    const id = nextId++;
    pending.set(id, (m: Moved) => {
      const r = readies.get(w);
      // the cells are updated in place: the focused zone's and a nearby zone's both hold the one object
      if (!r || m.anchor.version < r.anchor.version) return;
      r.cellX = m.cellX;
      r.cellY = m.cellY;
      r.anchor = m.anchor;
      if (w === worker) drawPick();
    });
    const msg: ToCommuters = { kind: "anchor", id, anchor: a };
    w.postMessage(msg);
  };
  /** A bubble worker for one zone's cells: the focused zone's, or a nearby zone's (T-276). */
  const spawn = (cells: CityCells): { worker: Worker; ready: Promise<Ready> } => {
    const w = new Worker(new URL("../workers/commuters.worker.ts", import.meta.url), { type: "module" });
    const ready = new Promise<Ready>((done) => {
      w.onmessage = (e: MessageEvent<FromCommuters>) => {
        const m = e.data;
        if (m.kind === "ready") {
          readies.set(w, m);
          done(m);
        } else {
          pending.get(m.id)?.(m);
          pending.delete(m.id);
        }
      };
    });
    const f = getWorldFrame();
    const a = anchor();
    const msg: ToCommuters = { kind: "cells", xy: cells.xy.slice(), h3: cells.h3.slice(), frame: { kind: f.kind, lon: f.lon, lat: f.lat }, anchor: a };
    told.set(w, a.version);
    w.postMessage(msg, [msg.xy.buffer, msg.h3.buffer]);
    return { worker: w, ready };
  };
  const startWorker = (): Promise<boolean> =>
    (workerReady ??= (async () => {
      const epoch = focusEpoch;
      const cells = await cityCells();
      if (epoch !== focusEpoch) return false;
      if (!cells) {
        workerReady = null;
        return false;
      }
      const s = spawn(cells);
      worker = s.worker;
      cellsWorld = cells.xy;
      const ready = await s.ready;
      if (epoch !== focusEpoch) return false;
      city = ready;
      commuterStats.cellsMs = ready.ms;
      return true;
    })());

  // ---- the city's bubbles ----
  let shown: { dv: DemandView; end: "home" | "work"; purpose: PurposeFilter; bubbles: Bubbles; modes: CellModes } | null = null;
  let built: { dv: DemandView; purpose: PurposeFilter; modes: CellModes; ends: Map<string, Bubbles> } | null = null;
  commuterStats.probe = () => (shown ? { anchor: shown.bubbles.anchor, levels: shown.bubbles.levels.map((l) => ({ discs: l.discs, group: l.group })) } : null);
  let wanted = 0;

  // ---- nearby zones' bubbles (T-276, docs/zones.md 6) ----
  // Up to three zones beside the focused one (game/zones.ts `nearbyZones`) show their bubbles too,
  // with the same look, end, purpose filter and day, so a zone line is not a cliff. Each has a
  // bubble worker of its own, built as the focused zone's are, and nothing else: no selection,
  // far end or inspector. A click on one focuses
  // its zone, which takes over its worker and bubbles.
  /** The worker builds the mode mix; bubbles are painted in the chosen scheme here (T-804). */
  const painted = (b: Bubbles): Bubbles => {
    const key = bubbleSchemeKey();
    if ((b.coloured ?? "mode") !== key) {
      recolour(b.levels);
      b.coloured = key;
    }
    return b;
  };
  type ZoneShown = { dv: DemandView; end: "home" | "work"; purpose: PurposeFilter; bubbles: Bubbles; modes: CellModes };
  const zones = new Map<CityId, { worker: Worker | null; city: Ready | null; cellsWorld: Float32Array | null; start?: Promise<boolean> | null; ask: number; shown: ZoneShown | null }>();
  const dropZone = (id: CityId) => {
    zones.get(id)?.worker?.terminate();
    zones.delete(id);
    view.setNearbyBubbles(id, null);
  };

  effect(() => {
    const focus = activeCity.value, frameNow = worldFrameEpoch.value;
    if (focus !== focusSeen || frameNow !== frameSeen) {
      // T-276: across a zone line the new focus's bubbles are usually built already as a nearby
      // zone's, and the old focus's become a nearby zone's: their workers and bubbles change hands
      const sameFrame = frameNow === frameSeen;
      if (!sameFrame) for (const id of [...zones.keys()]) dropZone(id);
      const adopted = zones.get(focus);
      if (adopted) {
        zones.delete(focus);
        view.setNearbyBubbles(focus, null);
      }
      const old = focusSeen, was = shown;
      if (sameFrame && worker && city && cellsWorld && !zones.has(old)) {
        const keep = was ? { dv: was.dv, end: was.end, purpose: was.purpose, bubbles: was.bubbles, modes: was.modes } : null;
        zones.set(old, { worker, city, cellsWorld, start: Promise.resolve(true), ask: 0, shown: keep });
        if (keep) view.setNearbyBubbles(old, painted(keep.bubbles).levels.map((l) => l.discs), keep.bubbles.anchor);
      } else worker?.terminate();
      focusSeen = focus; frameSeen = frameNow; focusEpoch++; wanted++;
      worker = null; workerReady = null; city = null; cellsWorld = null;
      shown = built = null;
      near = far = null; nearDiscs = farDiscs = null; farCells = null;
      view.setBubbles([]); view.setDiscs("near", null); view.setDiscs("far", null); view.setLines("far", null);
      if (adopted?.worker && adopted.city && adopted.cellsWorld) {
        worker = adopted.worker; city = adopted.city; cellsWorld = adopted.cellsWorld;
        workerReady = Promise.resolve(true);
        const a = adopted.shown;
        if (a) {
          built = { dv: a.dv, purpose: a.purpose, modes: a.modes, ends: new Map([[a.end, a.bubbles]]) };
          showBubbles(a);
        }
      } else adopted?.worker?.terminate();
      commuterPick.value = null;
      if (selection.peek()?.kind === "commuters") selection.value = null;
    }
    anchorVersion.value; // the bubbles are measured from the draw anchor (T-317): again when it moves
    const d = display.value;
    const dv = demandView.value;
    const updating = demandUpdating.value;
    const sel = selection.value;
    // the city's bubbles give way to a selection's own, and to the one bubble per city zoomed far
    // out (T-186)
    view.setBubblesVisible(d.commuters && sel?.kind !== "commuters" && !cityOverview.value);
    // one purpose's legs only under the map's purpose filter (T-158)
    const purpose = purposeFilter.value;
    if (shown && shown.purpose !== purpose) {
      // another purpose's bubbles would mislead while this one's are asked for: none meanwhile
      shown = null;
      view.setBubbles([]);
      drawPick();
    }
    if (!d.commuters || !dv) return;
    const end = d.commuterEnd;
    // a solve's first estimate, and its refined result once the crowding rounds are done
    if (shown && shown.dv.version === dv.version && shown.dv.day === dv.day && shown.end === end && shown.purpose === purpose && (shown.dv === dv || updating) && shown.bubbles.anchor.version === anchor().version) return;
    const ask = ++wanted;
    void (async () => {
      const same = built?.dv === dv && built.purpose === purpose;
      let modes = same ? built!.modes : null;
      if (!modes) {
        if (!(await startWorker())) return;
        modes = await cellModes(purpose);
        // null: the workers moved on; the next demandView asks again and the last bubbles stay
        if (!modes || ask !== wanted) return;
        built = { dv, purpose, modes, ends: new Map() };
      }
      let b = built?.modes === modes ? built.ends.get(end) : undefined;
      if (b && b.anchor.version !== anchor().version) b = undefined;
      if (!b) {
        if (!(await startWorker())) return;
        const t0 = performance.now();
        b = await buildOn(worker!, end, endCells(modes, end));
        commuterStats.workerMs = b.ms;
        commuterStats.roundTripMs = performance.now() - t0;
        commuterStats.builds++;
        if (built?.modes === modes && b.anchor.version === anchor().version) built.ends.set(end, b);
      }
      if (ask !== wanted) return;
      showBubbles({ dv, end, purpose, bubbles: b, modes });
      if (selection.peek()?.kind === "commuters") refreshPick();
    })();
  });

  const MODE_KEYS = ["homeRail", "homeWalk", "homeDrive", "workRail", "workWalk", "workDrive"] as const;
  /** One end's legs per cell (rail, walking, driving), copies to transfer to a worker. */
  const endCells = (m: CellModes, end: "home" | "work") => {
    const o = end === "home" ? 0 : 3;
    return MODE_KEYS.slice(o, o + 3).map((k) => m[k].slice());
  };
  /** Bubbles of one end from legs per cell (rail, walking, driving; transferred), in worker `w`. */
  const buildOn = (w: Worker, end: "home" | "work", cells: Float32Array[]) => {
    const id = nextId++;
    sync(w);
    return new Promise<Bubbles>((done) => {
      pending.set(id, done);
      const msg: ToCommuters = { kind: "build", id, end, train: cells[0], walk: cells[1], drive: cells[2] };
      w.postMessage(msg, [msg.train.buffer, msg.walk.buffer, msg.drive.buffer]);
    });
  };

  // a nearby zone's bubbles, rebuilt when the result, the end, the purpose filter or the day changes
  effect(() => {
    const ids = nearbyZones.value;
    const focus = activeCity.value;
    const d = display.value;
    const dv = demandView.value;
    const updating = demandUpdating.value;
    const purpose = purposeFilter.value;
    anchorVersion.value; // again at a new draw anchor (T-317)
    // the focused zone's own entry is left for the focus change to take over (above)
    for (const id of [...zones.keys()]) if (!ids.includes(id) && id !== focus) dropZone(id);
    if (!d.commuters || !dv) return;
    const end = d.commuterEnd;
    for (const id of ids) {
      if (id === focus) continue;
      let z = zones.get(id);
      if (!z) zones.set(id, (z = { worker: null, city: null, cellsWorld: null, ask: 0, shown: null }));
      const s = z.shown;
      if (s && s.dv.version === dv.version && s.dv.day === dv.day && s.end === end && s.purpose === purpose && (s.dv === dv || updating) && s.bubbles.anchor.version === anchor().version) continue;
      // another purpose's or end's bubbles would mislead: none meanwhile (a new result keeps the last)
      if (s && (s.purpose !== purpose || s.end !== end)) {
        z.shown = null;
        view.setNearbyBubbles(id, null);
      }
      const entry = z, ask = ++entry.ask;
      const live = () => zones.get(id) === entry && entry.ask === ask;
      void (async () => {
        const t0 = performance.now();
        // its pack may still be loading (workers/demandClient.ts opens nearby zones); one worker a zone
        entry.start ??= (async () => {
          const cells = await cityCells(id);
          if (!cells || zones.get(id) !== entry) return (entry.start = null), false;
          const sp = spawn(cells);
          entry.worker = sp.worker;
          entry.cellsWorld = cells.xy;
          entry.city = await sp.ready;
          return true;
        })();
        if (!(await entry.start) || !live() || !entry.city) return;
        // null: the zone is not in the solve shown yet (opened since); the next result asks again
        const modes = await cellModes(purpose, id);
        if (!modes || !live() || !entry.worker) return;
        const b = await buildOn(entry.worker, end, endCells(modes, end));
        if (!live()) return;
        entry.shown = { dv, end, purpose, bubbles: b, modes };
        view.setNearbyBubbles(id, painted(b).levels.map((l) => l.discs), b.anchor);
        commuterStats.nearbyMs = performance.now() - t0;
        commuterStats.nearbyBuilds++;
        commuterStats.nearbyShown[id] = performance.now();
      })();
    }
  });
  // a new draw anchor goes to every worker at once, so their cells are measured from it for the
  // selection's discs even while no bubbles are being built
  effect(() => {
    anchorVersion.value;
    if (worker) sync(worker);
    for (const z of zones.values()) if (z.worker) sync(z.worker);
  });
  const showBubbles = (s: NonNullable<typeof shown>) => {
    shown = s;
    view.setBubbles(painted(s.bubbles).levels.map((l) => l.discs), s.bubbles.anchor);
    commuterStats.uploadMs = view.uploadMs;
  };

  // the size slider in the settings pane scales every commuter bubble on the GPU (T-099)
  effect(() => view.setBubbleSize(display.value.bubbleSize));
  effect(() => view.setBubbleOpacity(display.value.bubbleOpacity));
  effect(() => view.setBubbleDetail(display.value.bubbleDetail));
  // a new colour scheme (T-804): the bubbles on show are painted again without a rebuild
  const colourKey = computed(() => `${display.value.bubbleColours}${display.value.rampFrom}${display.value.rampTo}`);
  effect(() => {
    colourKey.value;
    const d = display.peek();
    setBubbleColours(d.bubbleColours, d.rampFrom, d.rampTo);
    for (const [id, z] of zones) if (z.shown) view.setNearbyBubbles(id, painted(z.shown.bubbles).levels.map((l) => l.discs), z.shown.bubbles.anchor);
    if (!shown) return;
    view.setBubbles(painted(shown.bubbles).levels.map((l) => l.discs), shown.bubbles.anchor);
    drawPick();
  });

  // ---- hover ----
  const endWord = (end: "home" | "work") => (end === "home" ? "from here" : "to here");
  const bubbleTag = (end: "home" | "work", t: number, w: number, d: number, ...rest: string[]) => [`${count(t + w + d)} ${tripWords()} ${shownDayPhrase.peek()} ${endWord(end)}`, modeBar(t, w, d), ...rest];
  addFinder(10, (p) => {
    if (!shown || !display.peek().commuters || cityOverview.peek()) return null;
    const sel = selection.peek();
    if (sel?.kind === "commuters") {
      let i = discAt(nearDiscs, p, nearAnchor, view.bubbleSize);
      if (i >= 0 && near) return bubbleTag(sel.end, near.train[i], near.walk[i], near.drive[i]);
      i = discAt(farDiscs, p, farAnchor, view.bubbleSize);
      if (i >= 0 && far) return bubbleTag(sel.end === "home" ? "work" : "home", far.train[i], far.walk[i], far.drive[i], "Click to explore this trip");
      return null;
    }
    const L = shown.bubbles.levels[bubbleLevel(map.getZoom())];
    const i = discAt(L.discs, p, shown.bubbles.anchor, view.bubbleSize);
    if (i < 0) return null;
    return bubbleTag(shown.end, L.train[i], L.walk[i], L.drive[i], ...cityHint());
  });
  /** T-885: with a city selected, a bubble in another place opens the trips between them. */
  const cityHint = (): string[] => {
    const sel = selection.peek();
    if (sel?.kind !== "city") return [];
    const c = cityById(sel.city);
    return c.place === cityById(activeCity.peek()).place ? [] : [`Click to explore trips with ${c.name}`];
  };
  /** A nearby zone's bubble under the point (T-276): drawn under the focused zone's, shown with them. */
  const nearbyAt = (p: Pointer): { id: CityId; s: ZoneShown; i: number } | null => {
    const sel = selection.peek()?.kind;
    if (!display.peek().commuters || cityOverview.peek() || sel === "commuters") return null;
    const level = bubbleLevel(map.getZoom());
    for (const [id, z] of zones) {
      if (!z.shown || id === activeCity.peek()) continue;
      const i = discAt(z.shown.bubbles.levels[level].discs, p, z.shown.bubbles.anchor, view.bubbleSize);
      if (i >= 0) return { id, s: z.shown, i };
    }
    return null;
  };
  addFinder(9, (p) => {
    const hit = nearbyAt(p);
    if (!hit) return null;
    const L = hit.s.bubbles.levels[bubbleLevel(map.getZoom())], i = hit.i;
    return bubbleTag(hit.s.end, L.train[i], L.walk[i], L.drive[i], ...cityHint());
  });

  // ---- selecting bubbles ----
  /** The city's bubbles at the zoom on screen whose disc contains (or, for a box, whose centre is
   * in) the given area: their groups. */
  const groupsAt = (p: Pointer): number[] => {
    if (!shown) return [];
    const L = shown.bubbles.levels[bubbleLevel(map.getZoom())];
    const i = discAt(L.discs, p, shown.bubbles.anchor, view.bubbleSize);
    return i < 0 ? [] : [L.group[i]];
  };
  const groupsIn = (p0: Pointer, p1: Pointer): number[] => {
    if (!shown) return [];
    const L = shown.bubbles.levels[bubbleLevel(map.getZoom())];
    const [x0, y0] = pointerLocal(p0, shown.bubbles.anchor), [x1, y1] = pointerLocal(p1, shown.bubbles.anchor);
    const out: number[] = [];
    for (let i = 0; i < L.group.length; i++) {
      const x = L.discs[i * 8], y = L.discs[i * 8 + 1];
      if (x >= Math.min(x0, x1) && x <= Math.max(x0, x1) && y >= Math.min(y0, y1) && y <= Math.max(y0, y1)) out.push(L.group[i]);
    }
    return out;
  };
  const selectGroups = (groups: number[], add: boolean): Extract<Selection, { kind: "commuters" }> | null => {
    if (!shown || !city || !groups.length) return null;
    const g = city.levels[bubbleLevel(map.getZoom())];
    const cells = new Set<number>();
    const cur = selection.peek();
    if (add && cur?.kind === "commuters" && cur.end === shown.end) for (const c of cur.cells) cells.add(c);
    for (const q of groups) for (let k = g.off[q]; k < g.off[q + 1]; k++) cells.add(g.members[k]);
    const prevAreas = add && cur?.kind === "commuters" ? cur.areas : 0;
    return { kind: "commuters", end: shown.end, cells: [...cells], areas: prevAreas + groups.length };
  };

  /**
   * The trip explorer for a selection and one of its far-end bubbles (T-208): from the selection's
   * busiest cell (most legs at its own end, under the purpose filter) to the far bubble's busiest
   * cell for this selection (most of its trips with the selection), homes to destinations whichever
   * end is selected. One cell a side because the explorer reads one place pair; the busiest is
   * where most of the trips the bubbles count start or end, where a bubble's drawn centre (their
   * weighted mean) can fall in a park or the water with nobody.
   */
  const farTrip = (sel: Extract<Selection, { kind: "commuters" }>, idx: number[]): Selection | null => {
    const fe = lastFlows?.sel === sel && shown ? farOf(dealtOf(lastFlows.f, shown.purpose), shown.purpose) : null;
    if (!fe || !far || !city || !cellsWorld || !shown || !idx.length || idx.some((i) => i >= far!.group.length)) return null;
    const g = city.levels[bubbleLevel(map.getZoom())];
    const qs = new Set(idx.map((i) => far!.group[i]));
    // each end's bubbles at the zoom on screen: their trips, and their busiest cell
    type Side = Map<number, { n: number; best: number; bestN: number }>;
    const tally = (side: Side, q: number, c: number, n: number) => {
      const s = side.get(q) ?? { n: 0, best: -1, bestN: 0 };
      s.n += n;
      if (n > s.bestN) (s.bestN = n), (s.best = c);
      side.set(q, s);
    };
    const farSide: Side = new Map(), nearSide: Side = new Map();
    for (let k = 0; k < fe.cells.length; k++) {
      const c = fe.cells[k], q = g.groupOf[c];
      if (qs.has(q)) tally(farSide, q, c, fe.farRail[k] + fe.farWalk[k] + fe.farDrive[k]);
    }
    const m = shown.modes;
    const [mt, mw, md] = sel.end === "home" ? [m.homeRail, m.homeWalk, m.homeDrive] : [m.workRail, m.workWalk, m.workDrive];
    for (const c of sel.cells) tally(nearSide, g.groupOf[c], c, mt[c] + mw[c] + md[c]);
    const busiest = (side: Side) => [...side.values()].filter((s) => s.best >= 0).sort((a, b) => b.n - a.n);
    const farB = busiest(farSide), nearB = busiest(nearSide);
    const bestOf = (b: ReturnType<typeof busiest>) => b.reduce((x, s) => (s.bestN > x.bestN ? s : x), b[0]);
    if (!farB.length || !nearB.length) return null;
    const to = bestOf(farB).best, own = bestOf(nearB).best;
    const xy = cellsWorld;
    const at = (c: number): [number, number] => [xy[c * 2], xy[c * 2 + 1]];
    const pairOf = (n: number, f: number): { from: [number, number]; to: [number, number] } => (sel.end === "home" ? { from: at(n), to: at(f) } : { from: at(f), to: at(n) });
    const trip: Selection = { kind: "trip", ...pairOf(own, to) };
    // T-586: the explorer counts the trips between the selection and every cell of the far bubbles,
    // as their hovers do
    const farCells: number[] = [];
    for (const q of qs) for (let k = g.off[q]; k < g.off[q + 1]; k++) farCells.push(g.members[k]);
    const places: TripPlaces = { end: sel.end, near: Uint32Array.from(sel.cells), far: Uint32Array.from(farCells) };
    // T-865: several bubbles at an end: the busiest few a side, each to each, weighted by the near
    // bubble's trip legs times the far bubble's trips with the selection
    if (nearB.length > 1 || farB.length > 1) {
      /** the station nearest the trip-weighted centre of `cells`, if near enough to name them by */
      const centre = (cells: Iterable<number>, weight: (c: number) => number) => {
        let sx = 0, sy = 0, sw = 0;
        for (const c of cells) {
          const w = weight(c);
          (sx += xy[c * 2] * w), (sy += xy[c * 2 + 1] * w), (sw += w);
        }
        return sw > 0 ? stationNameNear(sx / sw, sy / sw, GROUP_NAME_M) : "";
      };
      const farN = new Map<number, number>();
      for (let k = 0; k < fe.cells.length; k++) if (qs.has(g.groupOf[fe.cells[k]])) farN.set(fe.cells[k], fe.farRail[k] + fe.farWalk[k] + fe.farDrive[k]);
      const samples = nearB.slice(0, GROUP_SAMPLE_SIDE).flatMap((a) => farB.slice(0, GROUP_SAMPLE_SIDE).map((b) => ({ ...pairOf(a.best, b.best), w: a.n * b.n })));
      places.group = {
        near: nearB.length, far: farB.length, samples,
        nearName: centre(sel.cells, (c) => mt[c] + mw[c] + md[c]),
        farName: centre(farN.keys(), (c) => farN.get(c) ?? 0),
      };
    }
    tripOpenedFrom.value = { trip, from: sel, places };
    return trip;
  };

  /** A click in the select tool: a bubble selection, or undefined to let the network have it. */
  const click = (px: number, py: number, add: boolean): Selection | undefined => {
    if (!display.peek().commuters || !shown || cityOverview.peek()) return undefined;
    const p = pointerAt(map, px, py);
    const sel = selection.peek();
    // A selection replaces the city's bubbles. Its near/far discs remain above the network,
    // but clicking them must not select an invisible city bubble underneath.
    if (sel?.kind === "commuters") {
      if (discAt(nearDiscs, p, nearAnchor, view.bubbleSize) >= 0) return sel;
      // a far-end bubble opens the trip explorer for the pair (T-208)
      const i = discAt(farDiscs, p, farAnchor, view.bubbleSize);
      return i >= 0 ? farTrip(sel, [i]) ?? sel : undefined;
    }
    const s = selectGroups(groupsAt(p), add && sel?.kind !== "city");
    if (s) return cityTrip(s) ?? s;
    // a nearby zone's bubble (T-276): its zone takes the focus, with this worker and these bubbles
    // (the focus change above), and the bubble is selected as one of the focused zone's
    const hit = nearbyAt(p);
    if (!hit) return undefined;
    activeCity.value = hit.id;
    const n = activeCity.peek() === hit.id && focusSeen === hit.id ? selectGroups(groupsAt(p), false) : null;
    return n ? cityTrip(n) ?? n : undefined;
  };

  /** T-885: with a city selected (zoomed in, under the local bubbles), bubbles picked in another
   * place open the trip explorer between them and that city; null otherwise. */
  const cityTrip = (s: Extract<Selection, { kind: "commuters" }>): Selection | null => {
    const sel = selection.peek();
    if (sel?.kind !== "city") return null;
    const c = cityById(sel.city), own = cityById(activeCity.peek());
    const near = nearPoint(s);
    if (c.place === own.place || !near) return null;
    const p = lngLatToPack(c.center[0], c.center[1]);
    const far = { place: c.place, gid: -1 }, at: [number, number] = [p.x, p.y];
    const trip: Selection = s.end === "home" ? { kind: "trip", from: near, to: at, far } : { kind: "trip", from: at, to: near, far };
    tripOpenedFrom.value = { trip, from: sel, places: { end: s.end, near: Uint32Array.from(s.cells), far: new Uint32Array(0), ...marketGroup(s, near) } };
    return trip;
  };

  /** T-865: the far-end discs whose centres are in a box: their indices. */
  const farIn = (p0: Pointer, p1: Pointer): number[] => {
    if (!farDiscs) return [];
    const [x0, y0] = pointerLocal(p0, farAnchor), [x1, y1] = pointerLocal(p1, farAnchor);
    const out: number[] = [];
    for (let i = 0; i < farDiscs.length / DISC; i++) {
      const x = farDiscs[i * DISC], y = farDiscs[i * DISC + 1];
      if (x >= Math.min(x0, x1) && x <= Math.max(x0, x1) && y >= Math.min(y0, y1) && y <= Math.max(y0, y1)) out.push(i);
    }
    return out;
  };

  // A box: after a long press on a bubble, or at once with Shift held (T-865). On a city bubble or
  // a selected one it selects the bubbles whose centres are in it; with Shift and a selection,
  // started anywhere else, it takes the far-end bubbles in it and opens the trip explorer for them.
  const box = document.createElement("div");
  box.className = "demand-box";
  box.style.display = "none";
  map.getContainer().appendChild(box);
  let press: { x: number; y: number; timer: number; on: boolean; add: boolean; far: boolean } | null = null;
  const LONG_MS = 300, STILL_PX = 6, RING_PX = 16;
  const canvas = map.getCanvasContainer();
  const rel = (e: MouseEvent) => {
    const r = map.getContainer().getBoundingClientRect();
    return [e.clientX - r.left, e.clientY - r.top] as const;
  };
  const boxOn = (p: NonNullable<typeof press>) => {
    p.on = true;
    map.dragPan.disable();
    map.getCanvas().style.cursor = "crosshair";
    box.style.display = "";
    // a long press arms with a ring at the pointer (T-908), until the drag draws the box
    const r = p.add ? 0 : RING_PX / 2;
    box.classList.toggle("armed", !p.add);
    Object.assign(box.style, { left: p.x - r + "px", top: p.y - r + "px", width: 2 * r + "px", height: 2 * r + "px" });
  };
  const boxOff = () => {
    box.style.display = "none";
    box.classList.remove("armed");
    map.dragPan.enable();
    map.getCanvas().style.cursor = "";
  };
  // T-885: a selected city's view (zoomed out, no local bubbles) takes a Shift box over its other cities
  const boxes = () => tool.peek() === "select" && display.peek().commuters && ((!!shown && !cityOverview.peek()) || selection.peek()?.kind === "city");
  // MapLibre's own Shift-drag zoom (main.ts keeps it on in the select tool) is off from before its
  // handlers see the press (they listen on the canvas container too, registered first) until the
  // button is up
  let zoomOff = false;
  map.getContainer().addEventListener("mousedown", (e) => {
    if (e.button !== 0 || !e.shiftKey || !canvas.contains(e.target as Node) || !boxes()) return;
    map.boxZoom.disable();
    zoomOff = true;
  }, true);
  canvas.addEventListener("mousedown", (e) => {
    if (e.button !== 0 || !boxes()) return;
    const [x, y] = rel(e);
    const sel = selection.peek();
    const at = pointerAt(map, x, y);
    const onNear = cityOverview.peek() ? false : sel?.kind === "commuters" ? discAt(nearDiscs, at, nearAnchor, view.bubbleSize) >= 0 : groupsAt(at).length > 0;
    const farBox = e.shiftKey && (sel?.kind === "commuters" || (sel?.kind === "city" && cityOverview.peek())) && !onNear;
    if (cityOverview.peek() && !farBox) return;
    // Without Shift a box starts only on a visible city or selected bubble: a long press on a
    // train, station, empty map or a far disc keeps its ordinary interaction.
    if (!onNear && !e.shiftKey) return;
    const p = { x, y, on: false, add: e.shiftKey, far: farBox, timer: 0 };
    press = p;
    if (e.shiftKey) boxOn(p);
    else p.timer = window.setTimeout(() => boxOn(p), LONG_MS);
  });
  // Escape drops a box being drawn and nothing changes (before main.ts's Escape clears the selection)
  window.addEventListener("keydown", (e) => {
    if (e.key !== "Escape" || !press) return;
    clearTimeout(press.timer);
    if (press.on) boxOff();
    press = null;
    e.stopImmediatePropagation();
  }, true);
  // While a long press waits, MapLibre does not see the pointer's small moves (its drag pan listens
  // on the document, capture phase, and would start after 3 px, T-908); past STILL_PX the press is
  // a pan, and MapLibre's first move carries the whole distance from the press.
  window.addEventListener("mousemove", (e) => {
    const p = press;
    if (!p || p.on) return;
    const [x, y] = rel(e);
    if (Math.hypot(x - p.x, y - p.y) > STILL_PX) {
      clearTimeout(p.timer);
      press = null;
    } else e.stopPropagation();
  }, true);
  window.addEventListener("mousemove", (e) => {
    const p = press;
    if (!p?.on) return;
    const [x, y] = rel(e);
    if (box.classList.contains("armed") && Math.hypot(x - p.x, y - p.y) <= STILL_PX) return;
    box.classList.remove("armed");
    Object.assign(box.style, { left: Math.min(x, p.x) + "px", top: Math.min(y, p.y) + "px", width: Math.abs(x - p.x) + "px", height: Math.abs(y - p.y) + "px" });
  });
  window.addEventListener("mouseup", (e) => {
    if (zoomOff && tool.peek() === "select") map.boxZoom.enable();
    zoomOff = false;
    const p = press;
    press = null;
    if (!p) return;
    clearTimeout(p.timer);
    if (!p.on) return;
    boxOff();
    const [x, y] = rel(e);
    if (Math.hypot(x - p.x, y - p.y) <= STILL_PX) return; // a press in place: the click decides
    const p0 = pointerAt(map, p.x, p.y), p1 = pointerAt(map, x, y);
    const sel = selection.peek();
    if (p.far) {
      // the inter-region bubbles in the box, else the local far ones (T-881; the two models' trips
      // are not added together, and a box reaching past the zone's edge is aimed at the others)
      const t = interBox(p.x, p.y, x, y) ?? (sel?.kind === "commuters" ? farTrip(sel, farIn(p0, p1)) : null);
      if (t) selection.value = t;
      return;
    }
    const s = selectGroups(groupsIn(p0, p1), p.add);
    if (s) selection.value = cityTrip(s) ?? s;
  });

  // ---- the selection on the map and in the inspector ----
  let near: Groups | null = null, far: Groups | null = null;
  let nearDiscs: Float32Array | null = null, farDiscs: Float32Array | null = null;
  /** the anchors the near and far discs are relative to (T-317) */
  let nearAnchor: Anchor = anchor(), farAnchor: Anchor = anchor();
  /** the far end summed into every level (in the worker: up to ~80k cells), relative to `farAnchor` */
  let farCells: Groups[] | null = null;
  let flowAsk = 0;
  let lastFlowKey: unknown = null;
  /** the selection's last far-end answer, every purpose in it (T-158) */
  let lastFlows: { sel: Selection; dv: DemandView | null; f: Flows } | null = null;

  /** Near and far discs at the level on screen. */
  const drawPick = () => {
    const sel = selection.peek();
    if (!city || !shown || sel?.kind !== "commuters") {
      near = far = nearDiscs = farDiscs = null;
      view.setDiscs("near", null);
      view.setDiscs("far", null);
      view.setLines("far", null);
      return;
    }
    const level = bubbleLevel(map.getZoom());
    const g = city.levels[level];
    const m = shown.modes;
    const [mt, mw, md] = sel.end === "home" ? [m.homeRail, m.homeWalk, m.homeDrive] : [m.workRail, m.workWalk, m.workDrive];
    const t = Float32Array.from(sel.cells, (c) => mt[c]), w = Float32Array.from(sel.cells, (c) => mw[c]), d = Float32Array.from(sel.cells, (c) => md[c]);
    near = aggregate(g, city.cellX, city.cellY, sel.cells, t, w, d);
    nearAnchor = city.anchor;
    nearDiscs = discs(near, 1 / BUBBLE_DENSITY[sel.end], BUBBLE_ALPHA, 2, nearAnchor);
    view.setDiscs("near", nearDiscs, nearAnchor);
    if (farCells) {
      far = farCells[level];
      // sized among themselves: the biggest about the area of a bubble's hexagon
      let max = 0;
      for (let i = 0; i < far.group.length; i++) max = Math.max(max, far.train[i] + far.walk[i] + far.drive[i]);
      farDiscs = discs(far, max > 0 ? (0.6 * RES_AREA_M2[level]) / max : 0, BUBBLE_ALPHA, 0, farAnchor);
      view.setDiscs("far", farDiscs, farAnchor);
    } else {
      far = farDiscs = null;
      view.setDiscs("far", null);
    }
    view.setLines("far", LINES_TO_SELECTION && farDiscs ? linesToward(nearDiscs, nearAnchor, farDiscs, farAnchor) : null, farAnchor);
  };
  view.onLevel = () => drawPick();
  // ?debug=1 (T-586's check): the selection's far-end bubbles as drawn, with the anchor
  commuterStats.farProbe = () => (far && farDiscs ? { anchor: farAnchor, discs: farDiscs, group: far.group, train: far.train, walk: far.walk, drive: far.drive } : null);

  /** The far end's places, named by the station nearest them. */
  const places = (fc: Groups[], A: Anchor): Place[] => {
    if (!city) return [];
    const a = fc[2]; // resolution 7, about 5 km²: places a player knows
    const st = (world.peek()?.save.stations ?? []).map((s) => ({ name: s.name, at: toLocal(s.lng, s.lat, A) }));
    const sel = selection.peek();
    // everything in the units of `A`, the anchor the far end was summed at
    const [sx, sy] = anchorShift(city.anchor, A);
    let cx = 0, cy = 0, n = 0;
    if (sel?.kind === "commuters") for (const c of sel.cells.slice(0, 2000)) (cx += city.cellX[c] + sx), (cy += city.cellY[c] + sy), n++;
    cx /= n || 1;
    cy /= n || 1;
    const byName = new Map<string, number>();
    // every area, not the biggest few: a row totals all the areas named for it (T-465)
    for (let i = 0; i < a.group.length; i++) {
      if (a.train[i] + a.walk[i] + a.drive[i] <= 0) continue;
      let best = Infinity, name = "";
      for (const s of st) {
        const dd = Math.hypot(s.at[0] - a.x[i], s.at[1] - a.y[i]);
        if (dd < best) (best = dd), (name = s.name);
      }
      const k = kAtY(a.y[i], A);
      if (best > 3000 * k) name = `${wholeDistance(Math.hypot(a.x[i] - cx, a.y[i] - cy) / k)} ${compass(a.x[i] - cx, -(a.y[i] - cy))}`;
      byName.set(name, (byName.get(name) ?? 0) + a.train[i] + a.walk[i] + a.drive[i]);
    }
    return [...byName].map(([name, riders]) => ({ name, riders })).sort((p, q) => q.riders - p.riders);
  };

  const refreshPick = () => {
    const sel = selection.peek();
    if (sel?.kind !== "commuters" || !shown) return;
    const t0 = performance.now();
    const m = shown.modes;
    const [mt, mw, md] = sel.end === "home" ? [m.homeRail, m.homeWalk, m.homeDrive] : [m.workRail, m.workWalk, m.workDrive];
    const totals: [number, number, number] = [0, 0, 0];
    for (const c of sel.cells) (totals[0] += mt[c]), (totals[1] += mw[c]), (totals[2] += md[c]);
    // T-323: the selection's centre (longitude, latitude), its cells weighted by their trip legs, for
    // the panel's closest stations
    let at: [number, number] | undefined;
    if (cellsWorld) {
      let sx = 0, sy = 0, sw = 0;
      for (const c of sel.cells) {
        const wt = mt[c] + mw[c] + md[c];
        sx += cellsWorld[c * 2] * wt; sy += cellsWorld[c * 2 + 1] * wt; sw += wt;
      }
      if (sw > 0) at = getWorldFrame().toLonLat(sx / sw, sy / sw);
    }
    const prev = commuterPick.peek();
    const purpose = shown.purpose;
    const keepFar = prev?.sel === sel && prev.purpose === purpose ? prev.far : null;
    commuterPick.value = { sel, totals, far: keepFar, purposes: prev?.sel === sel ? prev.purposes : undefined, purpose, at };
    drawPick();
    commuterStats.selectMs = performance.now() - t0;
    // the far end: asked again when the selection or the demand result changes, and summed
    // again for another purpose filter from the answer in hand (T-158)
    const dv = demandView.peek();
    const key = [sel, dv, purpose, anchor().version]; // the far end is summed again at a new draw anchor (T-317)
    if ((lastFlowKey as unknown[] | null)?.every((v, i) => v === key[i])) return;
    lastFlowKey = key;
    const ask = ++flowAsk;
    const t1 = performance.now();
    commuterStats.pickAt = t1;
    const answer = lastFlows?.sel === sel && lastFlows.dv === dv ? Promise.resolve(lastFlows.f) : flows(sel.end, Uint32Array.from(sel.cells));
    void answer.then(async (f) => {
      // null: the workers moved on; the next demand result asks again
      if (ask !== flowAsk || !f || selection.peek() !== sel || !worker) return;
      lastFlows = { sel, dv, f };
      commuterStats.flowsAt = performance.now();
      commuterStats.flowsMs = commuterStats.flowsAt - t1;
      const fe = farOf(dealtOf(f, purpose), purpose);
      commuterStats.farCells = fe?.cells.length ?? 0;
      const ft: [number, number, number] = fe ? [fe.rail - fe.cutRail, fe.walk - fe.cutWalk, fe.drive - fe.cutDrive] : [0, 0, 0];
      // summed into the bubble levels in the worker, not here
      const id = nextId++;
      const summed = await new Promise<Summed>((done) => {
        pending.set(id, done);
        const none = new Float32Array(0);
        const msg: ToCommuters = { kind: "sum", id, cells: fe?.cells ?? new Uint32Array(0), train: fe?.farRail ?? none, walk: fe?.farWalk ?? none, drive: fe?.farDrive ?? none };
        sync(worker!);
        worker!.postMessage(msg);
      });
      if (ask !== flowAsk || selection.peek() !== sel) return;
      const t2 = performance.now();
      commuterStats.summedAt = t2;
      farCells = summed.levels;
      farAnchor = summed.anchor;
      commuterPick.value = { sel, totals, purposes: f.purposes, far: { totals: ft, places: places(farCells, farAnchor) }, purpose, at };
      drawPick();
      commuterStats.farAt = performance.now();
      commuterStats.farMs = commuterStats.farAt - t2;
      commuterStats.sumMs = summed.ms;
    });
  };
  effect(() => {
    const sel = selection.value;
    demandView.value;
    anchorVersion.value;
    if (sel?.kind !== "commuters") {
      flowAsk++;
      farCells = null;
      lastFlowKey = null;
      lastFlows = null;
      commuterPick.value = null;
      drawPick();
      return;
    }
    if (commuterPick.peek()?.sel !== sel) {
      farCells = null;
      lastFlowKey = null;
    }
    refreshPick();
  });
  // a new end in the settings drops a selection of the other end
  effect(() => {
    const d = display.value;
    const sel = selection.peek();
    if (sel?.kind === "commuters" && (!d.commuters || sel.end !== d.commuterEnd)) selection.value = null;
  });

  /** T-866: a bubble selection's busiest cell (as `farTrip` takes its own end), world metres. */
  const nearPoint = (sel: Extract<Selection, { kind: "commuters" }>): [number, number] | null => {
    if (!shown || !cellsWorld) return null;
    const m = shown.modes;
    const [mt, mw, md] = sel.end === "home" ? [m.homeRail, m.homeWalk, m.homeDrive] : [m.workRail, m.workWalk, m.workDrive];
    let own = -1, ownN = -1;
    for (const c of sel.cells) {
      const n = mt[c] + mw[c] + md[c];
      if (n > ownN) (ownN = n), (own = c);
    }
    return own < 0 ? null : [cellsWorld[own * 2], cellsWorld[own * 2 + 1]];
  };

  setNearPoint(nearPoint);
  return { click, nearPoint };
}
