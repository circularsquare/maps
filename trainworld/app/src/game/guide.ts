// The Getting started card's steps (T-1822, notes/T-1822.md): six steps checked from the world
// whenever it or the card's progress changes. A step ticks once and stays ticked (undo never
// unticks it); the counters keep following the world. Progress is kept in the save through the clock
// worker (`SaveHeader.guide`), the browser remembers only that the guide was finished once.

import { effect } from "@preact/signals";
import { moneyState } from "./money";
import { guideFinished } from "./ui";
import { trainsShown, world } from "./world";
import type { WorldView } from "./types";
import { saveGuide } from "../workers/clockClient";
import type { GuideSave } from "../workers/saveFile";

export interface GuideStep {
  words: string;
  /** a counter: what there is now, the goal, and its hover; none for a step with nothing to count */
  count?: { have: number; goal: number; km?: true };
  met: boolean;
}

/** track to lay, m, and its stations apart from each other, m (step 1, 2) */
export const GUIDE_TRACK_M = 5000, GUIDE_STATIONS = 3, GUIDE_APART_M = 500;
const WORDS = [
  "Place 5km of track",
  "Place stations on the track (space them ~1km apart)",
  "Construct your blueprints",
  "Create a line",
  "Assign stops for the line",
  "Add trains to the line",
];

/** How many stations stand at least GUIDE_APART_M from each other (taken in order, a station within
 * reach of one already taken counts as part of it), so three side by side count as one. */
function spaced(list: readonly { x: number; y: number }[]): number {
  const kept: { x: number; y: number }[] = [];
  for (const s of list) if (kept.every((k) => Math.hypot(k.x - s.x, k.y - s.y) >= GUIDE_APART_M)) kept.push(s);
  return kept.length;
}

/** The six steps as the world has them now. `trains`: a line's trains per demand level. */
export function guideSteps(w: WorldView, trains: (line: string) => readonly number[] | null = trainsShown): GuideStep[] {
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
  const km = (m: number) => Math.min(m, GUIDE_TRACK_M) / 1000;
  return [
    { words: WORDS[0], count: { have: km(laid), goal: GUIDE_TRACK_M / 1000, km: true }, met: laid >= GUIDE_TRACK_M },
    { words: WORDS[1], count: { have: Math.min(stationsOn, GUIDE_STATIONS), goal: GUIDE_STATIONS }, met: stationsOn >= GUIDE_STATIONS },
    { words: WORDS[2], met: built >= GUIDE_TRACK_M && builtCount >= GUIDE_STATIONS },
    { words: WORDS[3], met: lines.length > 0 },
    { words: WORDS[4], count: { have: Math.min(mostStops, stopGoal), goal: stopGoal }, met: mostStops >= stopGoal },
    { words: WORDS[5], count: { have: anyTrain ? 1 : 0, goal: 1 }, met: anyTrain },
  ];
}

/** Which steps are ticked (met now, or ticked before), and the step in hand: the first unticked, -1
 * once all are. */
export function guideTicks(steps: readonly GuideStep[], g: GuideSave): { done: boolean[]; now: number } {
  const done = steps.map((s, i) => s.met || (g.done & (1 << i)) !== 0);
  return { done, now: done.indexOf(false) };
}

/** Whether the card is on show for the game in hand: it has progress, is not a sandbox, and was not
 * closed with the x or finished with Done. */
export function guideShown(g: GuideSave | undefined, sandbox: boolean): boolean {
  return !!g && !sandbox && !g.hidden && !g.closed;
}

/** Tick steps as they are met, and remember in this browser that the guide was finished. */
export function startGuide() {
  let sent = -1;
  effect(() => {
    const m = moneyState.value, w = world.value;
    const g = m?.guide;
    if (!g || !w || m.mode === "sandbox") return;
    const steps = guideSteps(w);
    let mask = g.done;
    steps.forEach((s, i) => { if (s.met) mask |= 1 << i; });
    if (mask === (1 << steps.length) - 1 && !guideFinished.peek()) guideFinished.value = true;
    if (mask !== g.done && mask !== sent) {
      sent = mask;
      saveGuide({ ...g, done: mask });
    }
  });
}
