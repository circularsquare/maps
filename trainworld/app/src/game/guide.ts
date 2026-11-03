// The tutorial's two cards (T-1822, T-2123; notes/T-1822.md): the first builds a running line, the
// second reads demand. Steps are checked from the world and the page whenever they change. A step
// ticks once and stays ticked (undo never unticks it); the counters keep following the world.
// Progress is kept in the save through the clock worker (`SaveHeader.guide`), the browser remembers
// only that the first card was finished once.

import { computed, effect, signal } from "@preact/signals";
import { activeCity, cityById, viewedZone } from "./cities";
import { clock } from "./clock";
import { AVERAGE_DAY } from "./days";
import { cityDemandViews, demandByDay } from "./demand";
import { moneyState } from "./money";
import { STAR_SHARE } from "./points";
import { display, guideFinished, selection, tab, type MapRow } from "./ui";
import { trainsShown, world } from "./world";
import { zoomedOut } from "../map/intercityView";
import type { WorldView } from "./types";
import { saveGuide } from "../workers/clockClient";
import type { GuideSave } from "../workers/saveFile";

export type GuideId = "zoom" | "track" | "stations" | "construct" | "line" | "stops" | "trains" | "crowd" | "levels"
  | "play" | "homes" | "bubble" | "trip" | "star";

/** Each step's bit in `GuideSave.done`: the first six are T-1822's, so older saves keep their ticks. */
export const GUIDE_BITS: Record<GuideId, number> = {
  track: 0, stations: 1, construct: 2, line: 3, stops: 4, trains: 5,
  zoom: 6, crowd: 7, levels: 8, homes: 9, bubble: 10, trip: 11, star: 12, play: 13,
};

export interface GuideStep {
  id: GuideId;
  words: string;
  /** a counter: what there is now and the goal; none for a step with nothing to count */
  count?: { have: number; goal: number; unit?: "km" | "%" };
  met: boolean;
}

/** track to lay, m, and its stations apart from each other, m (step 1, 2) */
export const GUIDE_TRACK_M = 5000, GUIDE_STATIONS = 3, GUIDE_APART_M = 500;
const WORDS: Record<Exclude<GuideId, "star">, string> = {
  zoom: "Zoom into a city",
  track: "Place 5km of track",
  stations: "Place stations on the track (space them ~1km apart)",
  construct: "Construct your blueprints",
  line: "Create a line",
  stops: "Assign stops for the line by clicking them on the map",
  trains: "Add trains to the line",
  crowd: "Hover the dot by High to see if there are enough trains",
  levels: "Add trains at high, medium and low demand",
  play: "Unpause the game",
  homes: "Show Homes on the map",
  bubble: "Click a red bubble to see where its trips go",
  trip: "Click one of its destinations to see that trip",
};

/** a crowding dot in the line panel has been hovered in this page (step "crowd") */
const crowdSeen = signal(false);

/** How many stations stand at least GUIDE_APART_M from each other (taken in order, a station within
 * reach of one already taken counts as part of it), so three side by side count as one. */
function spaced(list: readonly { x: number; y: number }[]): number {
  const kept: { x: number; y: number }[] = [];
  for (const s of list) if (kept.every((k) => Math.hypot(k.x - s.x, k.y - s.y) >= GUIDE_APART_M)) kept.push(s);
  return kept.length;
}

/** The first card's steps as the world has them now. `trains`: a line's trains per demand level. */
function buildSteps(w: WorldView, g: GuideSave, trains: (line: string) => readonly number[] | null): GuideStep[] {
  let laid = 0, built = 0;
  for (const e of w.edges) {
    if (e.off || e.crossover) continue;
    laid += e.lengthM;
    if (e.built) built += e.lengthM;
  }
  const stations = w.save.stations.filter((s) => !s.off);
  const stationsOn = spaced(stations), stationsBuilt = stations.filter((s) => s.built);
  const builtCount = spaced(stationsBuilt);
  const lines = w.save.lines;
  const mostStops = lines.reduce((m, l) => Math.max(m, l.stops.length), 0);
  const stopGoal = Math.max(1, Math.min(3, stationsBuilt.length));
  const anyTrain = lines.some((l) => Math.max(0, ...(trains(l.id) ?? [])) >= 1);
  // levels with a train on the rail line with most (bus lines start with buses at every level)
  const levels = lines.reduce((m, l) => (w.lineStats[l.id]?.bus ? m : Math.max(m, (trains(l.id) ?? []).filter((n) => n >= 1).length)), 0);
  const km = (m: number) => Math.min(m, GUIDE_TRACK_M) / 1000;
  const steps: GuideStep[] = [];
  if (g.zoom) steps.push({ id: "zoom", words: WORDS.zoom, met: !zoomedOut.value && !!viewedZone.value && cityDemandViews.value.has(activeCity.value) });
  steps.push(
    { id: "track", words: WORDS.track, count: { have: km(laid), goal: GUIDE_TRACK_M / 1000, unit: "km" }, met: laid >= GUIDE_TRACK_M },
    { id: "stations", words: WORDS.stations, count: { have: Math.min(stationsOn, GUIDE_STATIONS), goal: GUIDE_STATIONS }, met: stationsOn >= GUIDE_STATIONS },
    { id: "construct", words: WORDS.construct, met: built >= GUIDE_TRACK_M && builtCount >= GUIDE_STATIONS },
    { id: "line", words: WORDS.line, met: lines.length > 0 },
    { id: "stops", words: WORDS.stops, count: { have: Math.min(mostStops, stopGoal), goal: stopGoal }, met: mostStops >= stopGoal },
    { id: "trains", words: WORDS.trains, count: { have: anyTrain ? 1 : 0, goal: 1 }, met: anyTrain },
    { id: "crowd", words: WORDS.crowd, met: crowdSeen.value },
    { id: "levels", words: WORDS.levels, count: { have: levels, goal: 3 }, met: levels >= 3 },
    // T-2142: a new game starts paused
    { id: "play", words: WORDS.play, met: clock.running.value },
  );
  return steps;
}

/** The second card's steps: reading demand (T-2123). */
function demandSteps(): GuideStep[] {
  const d = display.value, s = selection.value, m = moneyState.value;
  const city = activeCity.value;
  const share = demandByDay.value[AVERAGE_DAY]?.cities.get(city)?.split.train ?? 0;
  const starred = Object.keys(m?.stars ?? {}).length > 0;
  return [
    { id: "homes", words: WORDS.homes, met: d.commuters && d.commuterEnd === "home" },
    { id: "bubble", words: WORDS.bubble, met: s?.kind === "commuters" },
    { id: "trip", words: WORDS.trip, met: s?.kind === "trip" && s.to !== null },
    { id: "star", words: `Reach ${STAR_SHARE}% train share in ${cityById(city)?.name ?? "a city"}`, count: { have: starred ? STAR_SHARE : Math.min(share, STAR_SHARE), goal: STAR_SHARE, unit: "%" }, met: starred },
  ];
}

/** The steps of the card in hand. */
export function guideSteps(w: WorldView, g: GuideSave, trains: (line: string) => readonly number[] | null = trainsShown): GuideStep[] {
  return g.card === 2 ? demandSteps() : buildSteps(w, g, trains);
}

/** Which steps are ticked (met now, or ticked before), and the step in hand: the first unticked, -1
 * once all are. */
export function guideTicks(steps: readonly GuideStep[], g: GuideSave): { done: boolean[]; now: number } {
  const done = steps.map((s) => s.met || (g.done & (1 << GUIDE_BITS[s.id])) !== 0);
  return { done, now: done.indexOf(false) };
}

/** Whether the card is on show for the game in hand: it has progress, is not a sandbox, and was not
 * closed with the x or finished with Done. */
export function guideShown(g: GuideSave | undefined, sandbox: boolean): boolean {
  return !!g && !sandbox && !g.hidden && !g.closed;
}

/** T-2123: while the first card is up, the map's view rows (demand bubbles, station ridership, track
 * colour) are folded, unless the player opened one in this page (`guideOpened`). */
export const guideFolding = computed(() => {
  const m = moneyState.value, g = m?.guide;
  return !!m && guideShown(g, m.mode === "sandbox") && g!.card !== 2;
});
export const guideOpened = signal<MapRow[]>([]);

/** T-2092: this page opens a new game, so its card (if it has one) starts the dock on the Build tab,
 * where step 1's Track tool is. A saved game keeps the tab it had. Set by main.ts from the game choice. */
let newGameOpening = false;
export function guideNewGame(on: boolean) {
  newGameOpening = on;
}

/** Tick steps as they are met, and remember in this browser that the first card was finished. */
export function startGuide() {
  addEventListener("pointerover", (e) => {
    if (!crowdSeen.peek() && (e.target as Element | null)?.closest?.(".crowd-dot")) crowdSeen.value = true;
  }, { capture: true, passive: true });
  let sent = -1;
  effect(() => {
    const m = moneyState.value, w = world.value;
    const g = m?.guide;
    if (newGameOpening && m) {
      newGameOpening = false;
      if (g && guideShown(g, m.mode === "sandbox")) tab.value = "build";
    }
    if (!g || !w || m.mode === "sandbox") return;
    const steps = guideSteps(w, g);
    let mask = g.done;
    steps.forEach((s) => { if (s.met) mask |= 1 << GUIDE_BITS[s.id]; });
    const all = steps.every((s) => (mask & (1 << GUIDE_BITS[s.id])) !== 0);
    if (g.card !== 2 && all && !guideFinished.peek()) guideFinished.value = true;
    if (mask !== g.done && mask !== sent) {
      sent = mask;
      saveGuide({ ...g, done: mask });
    }
  });
}
