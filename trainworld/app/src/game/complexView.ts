// The world's station complexes (T-146, game/complexes.ts) and the complex view: a selected station
// is shown as its complex unless opened on its own (T-249). The ridership circles take a complex as
// one (map/stationCircles.ts). Zoomed out, the distance groups whose dots are joined by connectors
// (T-313).

import { computed, effect, signal } from "@preact/signals";
import { buildComplexes, membersOf, type Complexes } from "./complexes";
import { ComplexBuilder } from "./complexBuilder";
import { PATCH_MODE } from "./patchMode";
import type { StationId, StationInput } from "./types";
import { selection } from "./ui";
import { world } from "./world";
import { groundScale } from "./coords";
import { getWorldFrame } from "./cities";

/** The two builders keep their complexes up to date from the stations that changed (T-1380). */
const builder = new ComplexBuilder();
const nearBuilder = new ComplexBuilder();
const same = (a: Complexes, b: Complexes) => a.members.length === b.members.length && a.of.size === b.of.size
  && [...b.of].every(([id, c]) => a.of.get(id) === c) && b.members.every((m, i) => m.join() === a.members[i].join());

/** `?patch=0`: built whole when the stations' key changes */
let wholeKey = "";
let whole: Complexes = { of: new Map(), members: [] };
let nearWholeKey = "";
let nearWhole: Complexes = { of: new Map(), members: [] };
const keyOf = (st: readonly StationInput[]) => getWorldFrame().kind + st.map((s) => `${s.id}:${s.x}:${s.y}:${s.heading}:${s.length}`).join("|");

/** The world's complexes; the same object while no station moved, appeared or changed length and
 * no link (T-197) was made or undone. */
export const complexes = computed<Complexes>(() => {
  const st = world.value?.save.stations ?? [];
  const links = world.value?.save.links ?? [];
  if (PATCH_MODE === "0") {
    const k = keyOf(st) + "/" + links.join("|");
    if (k !== wholeKey) { wholeKey = k; whole = buildComplexes(st, links, groundScale); }
    return whole;
  }
  const out = builder.update(st, links, groundScale, getWorldFrame().kind);
  if (PATCH_MODE === "check" && !same(out, buildComplexes(st, links, groundScale))) console.error("T-1380 check: the complexes patched from changes differ from a whole build");
  return out;
});

/** The stations the distance rule groups, the player's links left out: zoomed out each station is
 * its own dot, and each group's dots are joined by the shortest set of connectors, each link by one
 * more between its two stations' dots (T-313, map/renderer.ts `syncGroups`). The same object while
 * no station moved, appeared or changed length. */
export const dotGroups = computed(() => {
  // with no links the two are the same grouping: built once (T-496)
  if (!(world.value?.save.links ?? []).length) return complexes.value;
  const st = world.value?.save.stations ?? [];
  if (PATCH_MODE === "0") {
    const k = keyOf(st);
    if (k !== nearWholeKey) { nearWholeKey = k; nearWhole = buildComplexes(st, [], groundScale); }
    return nearWhole;
  }
  const out = nearBuilder.update(st, [], groundScale, getWorldFrame().kind);
  if (PATCH_MODE === "check" && !same(out, buildComplexes(st, [], groundScale))) console.error("T-1380 check: the distance groups patched from changes differ from a whole build");
  return out;
});

/** The stations in `id`'s complex, itself included (just [id] for a station on its own). */
export const complexOf = (id: StationId): StationId[] => membersOf(complexes.value, id);

/** A station and the stations of its complex are one place to a line's carried figures (T-1419). */
export const stopKey = (id: StationId): string => {
  const c = complexes.peek().of.get(id);
  return c === undefined ? id : `c${c}`;
};

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

/** Whether selected station `id` is shown as its complex: always, unless opened on its own (T-249). */
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
