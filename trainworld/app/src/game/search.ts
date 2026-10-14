// Search (T-148): stations, lines and the registry's cities by name, for the search box at the
// map's upper right (ui/Search.tsx). Stations sharing a name within one station complex
// (game/complexes.ts) are one result, so "Times Sq" with three platforms is found once. Towns and
// neighbourhoods come from each city's OSM place list (T-177, game/placeList.ts), once the box has
// loaded them; addresses are not searched (notes/T-148.md).

import { CITIES, type City } from "./cities";
import { zoneAt } from "./zones";
import { complexes } from "./complexView";
import { lineDemand, stationDemand } from "./demand";
import { places, type Place } from "./placeList";
import { compass } from "./places";
import type { LineInput, StationId } from "./types";
import { world } from "./world";

export type SearchHit =
  | {
      kind: "station";
      name: string;
      /** the stations found: one, or the same-named members of a complex */
      ids: StationId[];
      /** the one to select: the member with the most lines */
      target: StationId;
      lng: number;
      lat: number;
      lines: LineInput[];
      built: boolean;
      /** the registry city it is nearest */
      city: string;
      /** what tells it apart from a same-named result (T-183): its city, its own lines, or the
       * way it lies from the others; none when no other result shares its name */
      tag?: string;
    }
  | { kind: "line"; name: string; line: LineInput }
  | { kind: "city"; name: string; city: City }
  /** a town or neighbourhood from a place list; `tag` as for stations (its city, or a direction) */
  | { kind: "place"; name: string; place: Place; lng: number; lat: number; city: string; tag?: string };

/** Lower case, accents and punctuation dropped, spaces collapsed. */
export function normalise(s: string): string {
  return s
    .normalize("NFD")
    .replace(/[̀-ͯ]/g, "")
    .toLowerCase()
    .replace(/&/g, " and ")
    .replace(/[^\p{L}\p{N}]+/gu, " ")
    .trim();
}

/** How well `name` matches the query `q` (both normalised): 0 exact, 1 starts with it, 2 a word
 * starts with it, 3 contains it, 4 contains it ignoring spaces, 5 its words are clipped forms of
 * the name's (looseWords); -1 no match. */
export function matchScore(name: string, q: string): number {
  if (!q) return -1;
  if (name === q) return 0;
  if (name.startsWith(q)) return 1;
  if ((" " + name).includes(" " + q)) return 2;
  if (name.includes(q)) return 3;
  if (name.replace(/ /g, "").includes(q.replace(/ /g, ""))) return 4;
  if (looseWords(name, q)) return 5;
  return -1;
}

/** Nicknames made by clipping a word and adding an ending (T-177: "south philly" for South
 * Philadelphia): the query's words match consecutive words of the name, each either starting it
 * or sharing its first letters bar the query word's last two, at least four and not the whole
 * name word (so "southie" does not find every South Street). */
function looseWords(name: string, q: string): boolean {
  const qw = q.split(" ");
  if (!name.includes(qw[0].slice(0, 4))) return false; // the common case, cheaply
  const nw = name.split(" ");
  const like = (n: string, w: string) => {
    if (n.startsWith(w)) return true;
    let k = 0;
    while (k < n.length && k < w.length && n[k] === w[k]) k++;
    return k >= Math.max(4, w.length - 2) && k < n.length;
  };
  for (let i = 0; i + qw.length <= nw.length; i++) if (qw.every((w, j) => like(nw[i + j], w))) return true;
  return false;
}

/** Equal matches: cities first (few, and the biggest places), then lines, then stations, then
 * towns and neighbourhoods. */
const KIND_ORDER = { city: 0, line: 1, station: 2, place: 3 } as const;
/** Equal place matches: bigger kinds first, then by population. */
const PLACE_WEIGHT: Record<string, number> = {
  city: 1e8, town: 1e7, borough: 1e7, suburb: 5e6, quarter: 4e6, village: 3e6, neighbourhood: 2e6, hamlet: 0,
};
export const MAX_HITS = 8;

/** The best matches for `query`, best first. */
export function search(query: string): SearchHit[] {
  const q = normalise(query);
  if (!q) return [];
  const scored: { hit: SearchHit; score: number; weight: number }[] = [];
  const w = world.peek();
  if (w) {
    const linesAt = new Map<StationId, LineInput[]>();
    for (const l of w.save.lines) for (const s of new Set(l.stops)) linesAt.set(s, [...(linesAt.get(s) ?? []), l]);
    // stations, grouped by complex and name
    const cx = complexes.peek();
    const groups = new Map<string, typeof w.save.stations>();
    for (const s of w.save.stations) {
      const score = matchScore(normalise(s.name), q);
      if (score < 0) continue;
      const k = `${cx.of.get(s.id) ?? s.id}|${normalise(s.name)}`;
      groups.set(k, [...(groups.get(k) ?? []), s]);
    }
    for (const g of groups.values()) {
      const lines = [...new Map(g.flatMap((s) => linesAt.get(s.id) ?? []).map((l) => [l.id, l])).values()];
      const target = [...g].sort((a, b) => (linesAt.get(b.id)?.length ?? 0) - (linesAt.get(a.id)?.length ?? 0) || Number(b.built) - Number(a.built))[0];
      const lng = g.reduce((t, s) => t + s.lng, 0) / g.length, lat = g.reduce((t, s) => t + s.lat, 0) / g.length;
      const boardings = g.reduce((t, s) => t + (stationDemand(s.id)?.boardings ?? 0), 0);
      scored.push({
        hit: { kind: "station", name: target.name, ids: g.map((s) => s.id), target: target.id, lng, lat, lines, built: g.some((s) => s.built), city: cityById(zoneAt(lng, lat)) },
        score: matchScore(normalise(target.name), q),
        weight: lines.length * 1e9 + boardings,
      });
    }
    for (const l of w.save.lines) {
      const byName = matchScore(normalise(l.name), q);
      const score = normalise(l.letter) === q ? 0 : byName;
      if (score < 0) continue;
      scored.push({ hit: { kind: "line", name: l.name, line: l }, score, weight: lineDemand(l.id)?.ridersPerDay ?? 0 });
    }
  }
  CITIES.forEach((c, i) => {
    const score = matchScore(normalise(c.name), q);
    if (score >= 0) scored.push({ hit: { kind: "city", name: c.name, city: c }, score, weight: -i });
  });
  for (const list of places.peek().values())
    for (const p of list) {
      let score = -1;
      for (const k of p.keys) {
        const s = matchScore(k, q);
        if (s >= 0 && (score < 0 || s < score)) score = s;
      }
      if (score < 0 || isRegistryCity(p)) continue;
      scored.push({
        hit: { kind: "place", name: p.name, place: p, lng: p.lng, lat: p.lat, city: cityById(p.city) },
        score, weight: (PLACE_WEIGHT[p.kind] ?? 0) + p.pop,
      });
    }
  scored.sort((a, b) => a.score - b.score || KIND_ORDER[a.hit.kind] - KIND_ORDER[b.hit.kind] || b.weight - a.weight
    || numeric(a.hit.name, b.hit.name));
  const hits = scored.slice(0, MAX_HITS).map((s) => s.hit);
  tagTwins(hits);
  return hits;
}

const cityById = (id: string) => CITIES.find((c) => c.id === id)?.name ?? "";
/** Names in natural order; one collator, since localeCompare with options builds one per call (T-348). */
const numeric = new Intl.Collator(undefined, { numeric: true }).compare;

type StationHit = Extract<SearchHit, { kind: "station" }>;

/** Tags for station results that share a name (T-183), so the rows can be told apart. One in
 * another city gets the city. In the same city (two complexes more than 50 m apart, like New
 * York's Times Sq-42 St and its 8 Av platforms) a result gets the lines only it serves ("subway
 * A, C, E"), or, when it has none of its own, the way it lies from the others ("northwest"). */
export function tagTwins(hits: SearchHit[]) {
  const stations = hits.filter((h): h is StationHit => h.kind === "station");
  for (const hit of stations) {
    const twins = stations.filter((o) => o !== hit && normalise(o.name) === normalise(hit.name));
    if (!twins.length) continue;
    const near = twins.filter((o) => o.city === hit.city);
    if (!near.length) {
      hit.tag = hit.city;
      continue;
    }
    const theirs = new Set(near.flatMap((o) => o.lines.map((l) => l.name)));
    const own = [...new Set(hit.lines.map((l) => l.name))].filter((n) => !theirs.has(n));
    hit.tag = own.length ? lineList(own) : direction(hit, near);
  }
  // towns and neighbourhoods (T-177): the city, or within one city the way it lies from the others
  const found = hits.filter((h): h is PlaceHit => h.kind === "place");
  for (const hit of found) {
    const twins = found.filter((o) => o !== hit && normalise(o.name) === normalise(hit.name));
    if (!twins.length) continue;
    const near = twins.filter((o) => o.city === hit.city);
    const way = near.length ? direction(hit, near) : undefined;
    // "New York, northeast" when other cities have one too, so the direction is not read as theirs
    hit.tag = !way ? hit.city : near.length < twins.length ? `${hit.city}, ${way}` : way;
  }
}

type PlaceHit = Extract<SearchHit, { kind: "place" }>;

/** A place list's own entry for a registry city ("Washington" near Washington, DC's centre),
 * which the city's own result already covers. */
function isRegistryCity(p: Place): boolean {
  if (p.kind !== "city" && p.kind !== "town") return false;
  const name = p.keys[0];
  return CITIES.some((c) => normalise(c.name).startsWith(name)
    && Math.hypot((p.lng - c.center[0]) * Math.cos((p.lat * Math.PI) / 180), p.lat - c.center[1]) < 0.15);
}

/** "subway A, C, E": line names in order, a first word they all share written once, three at
 * most ("subway 1, 2, 3 and 7 more"). */
export function lineList(names: string[]): string {
  const sorted = [...names].sort(numeric);
  const first = sorted[0].split(" ")[0];
  const shared = sorted.every((n) => n.startsWith(first + " ") && n.length > first.length + 1);
  const parts = shared ? sorted.map((n) => n.slice(first.length + 1)) : sorted;
  const shown = parts.slice(0, 3).join(", ") + (parts.length > 3 ? ` and ${parts.length - 3} more` : "");
  return shared ? `${first} ${shown}` : shown;
}

/** The compass direction from the others' middle to `hit`, or nothing when they share a spot. */
function direction(hit: { lng: number; lat: number }, others: { lng: number; lat: number }[]): string | undefined {
  const lng = others.reduce((t, o) => t + o.lng, 0) / others.length, lat = others.reduce((t, o) => t + o.lat, 0) / others.length;
  const dx = (hit.lng - lng) * Math.cos((lat * Math.PI) / 180), dy = hit.lat - lat;
  return Math.hypot(dx, dy) > 1e-6 ? compass(dx, dy) : undefined;
}
