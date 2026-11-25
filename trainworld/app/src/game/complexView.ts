// The world's station complexes (T-146, game/complexes.ts) and the complex view: a selected station
// is shown as its complex unless opened on its own (T-249). The ridership circles take a complex as
// one (map/stationCircles.ts). Zoomed out, the distance groups whose dots are joined by connectors
// (T-313).

import { computed, effect, signal } from "@preact/signals";
import { NO_PAIRS, apartKeys, buildComplexes, membersOf, type Complexes } from "./complexes";
import { ComplexBuilder, complexesDiffer } from "./complexBuilder";
import { withBuildingLinks } from "./buildings";
import { PATCH_MODE } from "./patchMode";
import type { BuildingInput, StationId, StationInput, WorldView } from "./types";
import { selection } from "./ui";
import { world } from "./world";
import { worldDiff } from "./worldDiff";
import { groundScale } from "./coords";
import { getWorldFrame } from "./cities";

/** `?patch=0`: built whole when the stations' key changes */
let wholeKey = "";
let whole: Complexes = { of: new Map(), members: new Map() };
let nearWholeKey = "";
let nearWhole: Complexes = { of: new Map(), members: new Map() };
const keyOf = (st: readonly StationInput[]) => getWorldFrame().kind + st.map((s) => `${s.id}:${s.x}:${s.y}:${s.heading}:${s.length}:${s.line ? Array.from(s.line).join(";") : ""}`).join("|");

let apartKey = "", apartSet: ReadonlySet<string> = NO_PAIRS;
/** T-4048: the pairs the player separated, as `pairKey`s; the same set while they stay the same */
function apartNow(): ReadonlySet<string> {
  const ap = world.value?.save.apart ?? [], k = ap.join("|");
  if (k !== apartKey) { apartKey = k; apartSet = ap.length ? apartKeys(ap) : NO_PAIRS; }
  return apartSet;
}

const NO_LINKS: readonly (readonly [StationId, StationId])[] = [];
const NO_BUILDINGS: readonly BuildingInput[] = [];

/** T-1380, T-4300: a builder kept up to date from the world it last read: by the stations the world diff names when there
 * is one (the steps' ids, else a walk of both lists), else from every station */
class WorldComplexes {
  private b = new ComplexBuilder();
  private from: WorldView | null = null;
  get(w: WorldView | null, links: readonly (readonly [StationId, StationId])[], apart: ReadonlySet<string>, buildings: readonly BuildingInput[]): Complexes {
    const st = w?.save.stations ?? [], kind = getWorldFrame().kind;
    const d = w && this.from && this.b.ready(kind) ? worldDiff(this.from, w) : null;
    const out = d
      ? this.b.patch(Array.from(d.stations, ([id, [, now]]) => [id, now ?? null] as const), links, groundScale, kind, apart, buildings)
      : this.b.update(st, links, groundScale, kind, apart, buildings);
    this.from = w;
    if (PATCH_MODE === "check") {
      const bad = complexesDiffer(out, st, links, groundScale, apart, buildings);
      if (bad) console.error(`T-1380 check: the ${links.length || buildings.length ? "complexes" : "distance groups"} patched from changes differ from a whole build (${bad})`);
    }
    return out;
  }
}
const built = new WorldComplexes(), near = new WorldComplexes();

/** The world's complexes; the same object while no station moved, appeared or changed length and
 * no link (T-197) was made or undone, nor stations separated (T-4048), nor a station building (T-2217) drawn or moved. */
export const complexes = computed<Complexes>(() => {
  const w = world.value, st = w?.save.stations ?? [];
  const links = w?.save.links ?? NO_LINKS, buildings = w?.save.buildings ?? NO_BUILDINGS;
  const apart = apartNow();
  if (PATCH_MODE === "0") {
    // T-2217: a station building joins its members as a link would
    const all = withBuildingLinks(links, buildings, st, groundScale);
    const k = keyOf(st) + "/" + all.join("|") + "/" + apartKey;
    if (k !== wholeKey) { wholeKey = k; whole = buildComplexes(st, all, groundScale, apart); }
    return whole;
  }
  return built.get(w, links, apart, buildings);
});

/** The stations the distance rule groups, the player's links and buildings left out: zoomed out each
 * station is its own dot, and each group's dots are joined by the shortest set of connectors, each
 * link by one more between its two stations' dots (T-313, map/renderer.ts `syncGroups`). The same
 * object while no station moved, appeared or changed length. */
export const dotGroups = computed(() => {
  // with no links or buildings the two are the same grouping: built once (T-496)
  if (!(world.value?.save.links ?? []).length && !world.value?.save.buildings?.length) return complexes.value;
  const st = world.value?.save.stations ?? [];
  const apart = apartNow();
  if (PATCH_MODE === "0") {
    const k = keyOf(st) + "/" + apartKey;
    if (k !== nearWholeKey) { nearWholeKey = k; nearWhole = buildComplexes(st, [], groundScale, apart); }
    return nearWhole;
  }
  return near.get(world.value, NO_LINKS, apart, NO_BUILDINGS);
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
 * station included (every selection is a new object), clears it: a click on a station selects its
 * whole complex (T-249), unless that complex is already selected, when the click opens the station
 * alone (map/main.ts `selectAt`, T-3539). Set it after the selection (`openAlone`). */
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
