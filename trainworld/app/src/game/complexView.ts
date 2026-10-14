// The world's station complexes (T-146, game/complexes.ts) and the complex view: a selected station
// is shown as its complex unless opened on its own (T-249). The demand circles take a complex as
// one by their own setting (`display.complexView`, demandViews.ts).

import { computed, effect, signal } from "@preact/signals";
import { buildComplexes, membersOf, type Complexes } from "./complexes";
import { demandView, stationRidersFor } from "./demand";
import type { StationId } from "./types";
import { selection } from "./ui";
import { world } from "./world";
import { groundScale } from "./coords";
import { getWorldFrame } from "./cities";

let lastKey = "";
let last: Complexes = { of: new Map(), members: [] };
/** The world's complexes; the same object while no station moved, appeared or changed length and
 * no link (T-197) was made or undone. */
export const complexes = computed<Complexes>(() => {
  const st = world.value?.save.stations ?? [];
  const links = world.value?.save.links ?? [];
  const k = getWorldFrame().kind + st.map((s) => `${s.id}:${s.x}:${s.y}:${s.heading}:${s.length}`).join("|") + "/" + links.join("|");
  if (k !== lastKey) {
    lastKey = k;
    last = buildComplexes(st, links, groundScale);
  }
  return last;
});

/** Groups with each one's busiest station (boardings plus alightings a day, every purpose; its first
 * station before demand is known). */
function withLeads(c: Complexes) {
  const dv = demandView.value;
  const riders = (id: StationId) => {
    const d = stationRidersFor(dv, id);
    return d ? d.boardings + d.alightings : 0;
  };
  return { ...c, lead: c.members.map((m) => m.reduce((best, id) => (riders(id) > riders(best) ? id : best), m[0])) };
}

/** The complexes with each one's busiest station. */
export const complexLeads = computed(() => withLeads(complexes.value));

let nearKey = "";
let near: Complexes = { of: new Map(), members: [] };
/** The stations the distance rule groups, the player's links left out (T-225, Anita): zoomed out
 * each group is one dot, standing for its busiest station when clicked (T-217), and each link a thin
 * line between two dots. A complex linked across hundreds of metres was one dot covering all of it
 * (5 Av/42 St with Grand Central, about 240 px at zoom 14.2). */
export const dotGroups = computed(() => {
  const st = world.value?.save.stations ?? [];
  const k = getWorldFrame().kind + st.map((s) => `${s.id}:${s.x}:${s.y}:${s.heading}:${s.length}`).join("|");
  if (k !== nearKey) {
    nearKey = k;
    near = buildComplexes(st, [], groundScale);
  }
  return withLeads(near);
});

/** The stations in `id`'s complex, itself included (just [id] for a station on its own). */
export const complexOf = (id: StationId): StationId[] => membersOf(complexes.value, id);

/** A station opened on its own from its complex's station list (T-246, Anita: "clicking a station
 * should take us to that specific station view"). Any other selection, a new click on the same
 * station included (every selection is a new object), clears it: a click on a station always
 * selects its whole complex (T-249, Anita: "in general we should just pick the complex"). Set it
 * after the selection (`openAlone`). */
export const singleStation = signal<StationId | null>(null);
effect(() => {
  void selection.value;
  if (singleStation.peek() !== null) singleStation.value = null;
});

/** Select station `id` on its own, in station view (T-246). */
export function openAlone(id: StationId) {
  selection.value = { kind: "station", station: id };
  singleStation.value = id;
}

/** Whether selected station `id` is shown as its complex: always, unless opened on its own (T-249;
 * `display.complexView` is now only the demand circles' "One circle per station complex"). */
export function inComplexView(id: StationId): boolean {
  return singleStation.value !== id;
}

/** What a selected station's numbers cover: its complex in complex view, else the station. */
export function stationScope(id: StationId): StationId[] {
  return inComplexView(id) ? complexOf(id) : [id];
}

/** The selected station's scope as ids joined by commas ("" when no station is selected): a
 * string, so effects that read it run again only when it changes, not on every setting. */
export const selectedScope = computed(() => {
  const s = selection.value;
  return s?.kind === "station" ? stationScope(s.station).join(",") : "";
});
