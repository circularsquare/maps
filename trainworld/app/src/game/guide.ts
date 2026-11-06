// The tutorial's two cards (T-1822, T-2123, T-2263; notes/T-1822.md): the first reads demand and
// lays a line, the second runs it. Steps are checked from the world and the page whenever they
// change. A step ticks once and stays ticked (undo never unticks it); the counters keep following the
// world. Progress is kept in the save through the clock worker (`SaveHeader.guide`), the browser
// remembers only that the first card was finished once.

import { computed, effect, signal } from "@preact/signals";
import { activeCity, cityById, viewedZone } from "./cities";
import { clock } from "./clock";
import { AVERAGE_DAY } from "./days";
import { cityDemandViews, demandByDay } from "./demand";
import { moneyState } from "./money";
import { display, guideFinished, guideVerbose, selection, tab, type MapRow } from "./ui";
import { trainsShown, world } from "./world";
import { zoomedOut } from "../map/intercityView";
import type { WorldView } from "./types";
import { saveGuide } from "../workers/clockClient";
import { t } from "../i18n";
import type { GuideSave } from "../workers/saveFile";

export type GuideId = "zoom" | "track" | "stations" | "construct" | "line" | "stops" | "trains" | "crowd" | "levels"
  | "play" | "homes" | "bubble" | "trip" | "star" | "split" | "traffic";

/** Each step's bit in `GuideSave.done`: the first six are T-1822's, so older saves keep their ticks. */
export const GUIDE_BITS: Record<GuideId, number> = {
  track: 0, stations: 1, construct: 2, line: 3, stops: 4, trains: 5,
  zoom: 6, crowd: 7, levels: 8, homes: 9, bubble: 10, trip: 11, star: 12, play: 13, split: 14, traffic: 15,
};

export interface GuideStep {
  id: GuideId;
  words: string;
  /** T-2263: a line under the step while it is the step in hand */
  note?: string;
  /** a counter: what there is now and the goal; none for a step with nothing to count */
  count?: { have: number; goal: number; unit?: "km" | "%" };
  met: boolean;
}

/** track to lay, m, and its stations apart from each other, m (step 1, 2) */
export const GUIDE_TRACK_M = 5000, GUIDE_STATIONS = 3, GUIDE_APART_M = 500;
/** the last step's train share, % (Anita, T-2263: the region star's 10% was too far for a tutorial) */
export const GUIDE_SHARE = 2;
const WORDS: Record<Exclude<GuideId, "star">, string> = {
  get zoom() { return t("guide.zoom"); },
  get split() { return t("guide.split"); },
  get homes() { return t("guide.homes"); },
  get bubble() { return t("guide.bubble"); },
  get trip() { return t("guide.trip"); },
  get track() { return t("guide.track"); },
  get stations() { return t("guide.stations"); },
  get construct() { return t("guide.construct"); },
  get line() { return t("guide.line"); },
  get stops() { return t("guide.stops"); },
  get trains() { return t("guide.trains"); },
  get play() { return t("guide.play"); },
  get crowd() { return t("guide.crowd"); },
  get levels() { return t("guide.levels"); },
  get traffic() { return t("guide.traffic"); },
};
const NOTES: Partial<Record<GuideId, string>> = {
  get split() { return t("guide.splitNote"); },
  get homes() { return t("guide.homesNote"); },
  get crowd() { return t("guide.crowdNote"); },
  get levels() { return t("guide.levelsNote"); },
  get traffic() { return t("guide.trafficNote"); },
};
const step = (id: Exclude<GuideId, "star">, met: boolean, count?: GuideStep["count"]): GuideStep => ({ id, words: WORDS[id], note: NOTES[id], count, met });

/** a crowding dot in the line panel, or the top bar's mode split, has been hovered in this page
 * (steps "crowd", "split") */
const crowdSeen = signal(false), splitSeen = signal(false);
/** the top bar's mode split (ui/TopBar.tsx), the "split" step's target */
export const GUIDE_SPLIT = "#topbar .split-row";

/** How many stations stand at least GUIDE_APART_M from each other (taken in order, a station within
 * reach of one already taken counts as part of it), so three side by side count as one. */
function spaced(list: readonly { x: number; y: number }[]): number {
  const kept: { x: number; y: number }[] = [];
  for (const s of list) if (kept.every((k) => Math.hypot(k.x - s.x, k.y - s.y) >= GUIDE_APART_M)) kept.push(s);
  return kept.length;
}

/** The first card's steps as the world has them now (T-2263: reading demand, then laying a line). */
function firstSteps(w: WorldView, g: GuideSave): GuideStep[] {
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
  const d = display.value, s = selection.value;
  const km = (m: number) => Math.min(m, GUIDE_TRACK_M) / 1000;
  const steps: GuideStep[] = [];
  if (g.zoom) steps.push(step("zoom", !zoomedOut.value && !!viewedZone.value && cityDemandViews.value.has(activeCity.value)));
  steps.push(
    step("split", splitSeen.value),
    step("homes", d.commuters && d.commuterEnd === "home"),
    step("bubble", s?.kind === "commuters"),
    step("trip", s?.kind === "trip" && s.to !== null),
    step("track", laid >= GUIDE_TRACK_M, { have: km(laid), goal: GUIDE_TRACK_M / 1000, unit: "km" }),
    step("stations", stationsOn >= GUIDE_STATIONS, { have: Math.min(stationsOn, GUIDE_STATIONS), goal: GUIDE_STATIONS }),
    step("construct", built >= GUIDE_TRACK_M && builtCount >= GUIDE_STATIONS),
    step("line", lines.length > 0),
    step("stops", mostStops >= stopGoal, { have: Math.min(mostStops, stopGoal), goal: stopGoal }),
  );
  return steps;
}

/** The second card's steps: running the line. `trains`: a line's trains per demand level. */
function secondSteps(w: WorldView, trains: (line: string) => readonly number[] | null): GuideStep[] {
  const lines = w.save.lines;
  const anyTrain = lines.some((l) => Math.max(0, ...(trains(l.id) ?? [])) >= 1);
  // levels with a train on the rail line with most (bus lines start with buses at every level)
  const levels = lines.reduce((m, l) => (w.lineStats[l.id]?.bus ? m : Math.max(m, (trains(l.id) ?? []).filter((n) => n >= 1).length)), 0);
  const city = activeCity.value;
  const share = demandByDay.value[AVERAGE_DAY]?.cities.get(city)?.split.train ?? 0;
  const starred = Object.keys(moneyState.value?.stars ?? {}).length > 0;
  return [
    step("trains", anyTrain, { have: anyTrain ? 1 : 0, goal: 1 }),
    // T-2142: a new game starts paused; T-2263: unpaused before the crowding is read
    step("play", clock.running.value),
    step("crowd", crowdSeen.value),
    step("levels", levels >= 3, { have: levels, goal: 3 }),
    step("traffic", display.value.trackColour === "traffic"),
    { id: "star", words: t("guide.star", { share: GUIDE_SHARE, city: cityById(city)?.name ?? t("guide.aCity") }), count: { have: starred ? GUIDE_SHARE : Math.min(share, GUIDE_SHARE), goal: GUIDE_SHARE, unit: "%" }, met: starred || share >= GUIDE_SHARE },
  ];
}

/** The steps of the card in hand. */
export function guideSteps(w: WorldView, g: GuideSave, trains: (line: string) => readonly number[] | null = trainsShown): GuideStep[] {
  return g.card === 2 ? secondSteps(w, trains) : firstSteps(w, g);
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

/** T-2123: while the first card is up, the map's station ridership and track colour rows are folded,
 * unless the player opened one in this page (`guideOpened`); T-2263: the demand bubbles' row stays
 * open, since the card starts with it. */
export const guideFolding = computed((): readonly MapRow[] => {
  const m = moneyState.value, g = m?.guide;
  return !!m && guideShown(g, m.mode === "sandbox") && g!.card !== 2 ? ["stations", "track"] : [];
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
    const t = e.target as Element | null;
    if (!crowdSeen.peek() && t?.closest?.(".crowd-dot")) crowdSeen.value = true;
    if (!splitSeen.peek() && t?.closest?.(GUIDE_SPLIT)) splitSeen.value = true;
  }, { capture: true, passive: true });
  let sent = -1;
  effect(() => {
    const m = moneyState.value, w = world.value;
    const g = m?.guide;
    guideVerbose.value = !!m && guideShown(g, m.mode === "sandbox");
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
