// T-1223: which reached zones demand holds fine and which coarse (notes/T-1206.md sections 3-4). A
// coarse zone is the same model at a lower sample density (`DemandApi::at_level`, T-1222); a zone goes
// fine when the player looks at it, clicks in it or builds in it, and coarse again once left alone.
// Plain functions of their inputs, so a check can run them without a page.

import { cityById, isCity, type CityId } from "../game/cities";
import { deviceGB } from "./deviceMemory";

export const LOD_FINE = 0;
export const LOD_COARSE = 1;

/** the zone in view counts as looked at once it has been in view this long (a pan refines nothing) */
export const VIEW_SETTLE_MS = 2000;
/** a fine zone in none of the first three priorities this long goes coarse, whatever the memory */
export const LEFT_ALONE_MS = 10 * 60_000;
/** an edit keeps its zone in the third priority this long */
export const EDIT_KEEP_MS = 15 * 60_000;
/** a zone made fine stays fine this long, so looking between two cities does not flap */
export const FINE_HOLD_MS = 5 * 60_000;

/** `?lod=fine` every zone fine, `?lod=coarse` every zone coarse (checks); else by the rules here */
export type LodMode = "auto" | "fine" | "coarse";
export function lodMode(): LodMode {
  const q = new URLSearchParams(location.search).get("lod");
  return q === "fine" || q === "coarse" ? q : "auto";
}

/** T-1197's rate (memoryCheck.ts): a solving worker's GB per million local trips a day of the zones
 * it holds fine; a coarse zone takes COARSE_SHARE of it (T-1222: 0.68 of fine's held memory over 17
 * Chinese zones, 0.75 for New York) */
export const GB_PER_MTRIPS = 0.0012;
export const COARSE_SHARE = 0.7;
export function zoneGB(id: string, coarse: boolean): number {
  const t = isCity(id) ? cityById(id).localTrips ?? 0 : 0;
  return (GB_PER_MTRIPS * t) / 1e6 * (coarse ? COARSE_SHARE : 1);
}

/** The zones' memory a solving worker would hold with `fine` fine and the rest of `zones` coarse. */
export function zonesGB(zones: Iterable<CityId>, fine: ReadonlySet<CityId>): number {
  let gb = 0;
  for (const id of zones) gb += zoneGB(id, !fine.has(id));
  return gb;
}

/** The zones' memory each solving worker may hold (notes/T-1206.md section 3): about 1.5 GB on a
 * device reporting 8 GB (`deviceGB`, which never reports more), less below, as much as 8 GB where it
 * is not reported (desktop Firefox and Safari) */
export function lodBudgetGB(): number {
  return 1.5 * Math.min(8, deviceGB() ?? 8) / 8;
}

/** T-2005, T-2153: the zones the memory drops spare for being built in, as builds, latest first: the
 * zones of the last two builds however long ago (`built`: an order of when each zone last had a
 * station placed or a stop added; zones of one edit share a time). Only zones in `among` count (a
 * zone no station reaches any more holds nothing to keep). */
export function builtKeep(built: ReadonlyMap<CityId, number>, among: ReadonlySet<CityId>): CityId[][] {
  const rows = [...built].filter(([id]) => among.has(id));
  const times = [...new Set(rows.map(([, t]) => t))].sort((a, b) => b - a).slice(0, 2);
  return times.map((t) => rows.filter(([, r]) => r === t).map(([id]) => id));
}

export interface LodInput {
  /** every zone the solves cover (reached), in any order */
  zones: readonly CityId[];
  /** priorities 1-3, highest first: the zone in view (settled), the selected zones, zones edited
   * in the last EDIT_KEEP_MS; always fine */
  held: readonly CityId[];
  /** priority 4: other zones on screen */
  near: readonly CityId[];
  /** the fine set now */
  fine: ReadonlySet<CityId>;
  /** when each zone was last in priorities 1-3, and when it was made fine (ms, any clock) */
  touched: ReadonlyMap<CityId, number>;
  madeFine: ReadonlyMap<CityId, number>;
  now: number;
  /** the zones' memory allowed (GB, as `zonesGB` counts it) */
  budget: number;
  /** `builtKeep`'s builds, latest first: fine zones that stay fine however long left alone, and go
   * last over the budget, the latest build's after the other's */
  keep?: readonly (readonly CityId[])[];
}

/**
 * The next fine set: priorities 1-3 always; a fine zone stays fine unless it has been in none of
 * them for LEFT_ALONE_MS; then, while the zones' memory is over the budget, the lowest priority
 * fine zone goes (an off-screen one before an on-screen one, the least recently looked at first),
 * never one in 1-3 nor one made fine within FINE_HOLD_MS; the `keep` zones neither go for being left
 * alone nor before every other droppable zone, the latest build's last of all. No zone becomes fine here except by
 * priorities 1-3: a coarse zone is refined by looking at it.
 */
export function planFine(inp: LodInput): Set<CityId> {
  const zones = new Set(inp.zones);
  const held = new Set(inp.held.filter((id) => zones.has(id)));
  const recent = (id: CityId) => inp.now - (inp.madeFine.get(id) ?? -Infinity) < FINE_HOLD_MS;
  const next = new Set(held);
  // a kept zone's rank: higher goes later
  const keep = new Map<CityId, number>();
  (inp.keep ?? []).forEach((ids, i, all) => { for (const id of ids) if (!keep.has(id)) keep.set(id, all.length - i); });
  for (const id of inp.fine) {
    if (!zones.has(id) || next.has(id)) continue;
    if (keep.has(id) || inp.now - (inp.touched.get(id) ?? -Infinity) < LEFT_ALONE_MS || recent(id)) next.add(id);
  }
  const near = new Set(inp.near);
  const drops = [...next].filter((id) => !held.has(id) && !recent(id))
    .sort((a, b) => (keep.get(a) ?? 0) - (keep.get(b) ?? 0) || Number(near.has(a)) - Number(near.has(b)) || (inp.touched.get(a) ?? -Infinity) - (inp.touched.get(b) ?? -Infinity));
  let gb = zonesGB(zones, next);
  for (const id of drops) {
    if (gb <= inp.budget) break;
    next.delete(id);
    gb -= zoneGB(id, false) - zoneGB(id, true);
  }
  return next;
}

/** A game's first fine set: `saved` (the save's `demandFine`) where there is one; else every zone
 * fine that fits the budget, in priority order (`order`: the zone in view first, then the zones on
 * screen, then the nearest). */
export function firstFine(zones: readonly CityId[], order: readonly CityId[], saved: readonly string[] | null, budget: number): Set<CityId> {
  const set = new Set(zones);
  if (saved) return new Set(saved.filter((id): id is CityId => set.has(id as CityId)));
  const out = new Set<CityId>();
  let gb = zonesGB(zones, out);
  for (const id of order) {
    if (!set.has(id) || out.has(id)) continue;
    const more = zoneGB(id, false) - zoneGB(id, true);
    if (out.size && gb + more > budget) continue;
    out.add(id);
    gb += more;
  }
  return out;
}
