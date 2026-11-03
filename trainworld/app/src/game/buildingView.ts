// The station buildings as the panels and the map read them (T-2218): each building's members, the
// building of a complex, the building under a point. The geometry is game/buildings.ts.

import { computed, signal } from "@preact/signals";
import { buildingMembers, insideOutline, type BuildingInput } from "./buildings";
import { groundScale } from "./coords";
import type { StationId } from "./types";
import { world } from "./world";

/** What the station building tool (map/buildingDraw.ts) is doing, for the bar at the map's bottom left
 * and the drawing: null while it is off. `id` the building being reshaped, null for a new one. */
export const buildingDraft = signal<{ id: number | null; corners: number; closed: boolean } | null>(null);

/** Each building's members by its id (stations whose platform touches it, any level). */
export const buildingMembersById = computed(() => {
  const w = world.value, bs = w?.save.buildings ?? [];
  if (!w || !bs.length) return new Map<number, StationId[]>();
  const m = buildingMembers(bs, w.save.stations, groundScale);
  return new Map(bs.map((b, i) => [b.id, m[i]]));
});

/** The building any of `stations` belongs to (the first standing), or null. */
export function buildingOf(stations: readonly StationId[]): BuildingInput | null {
  const inside = new Set(stations);
  for (const b of world.value?.save.buildings ?? []) {
    if ((buildingMembersById.value.get(b.id) ?? []).some((id) => inside.has(id))) return b;
  }
  return null;
}

/** The building whose floor holds a pack point, or null. */
export function buildingAtPack(x: number, y: number): BuildingInput | null {
  for (const b of world.peek()?.save.buildings ?? []) if (insideOutline(b.corners, x, y)) return b;
  return null;
}

export function buildingById(id: number): BuildingInput | null {
  return world.peek()?.save.buildings?.find((b) => b.id === id) ?? null;
}
