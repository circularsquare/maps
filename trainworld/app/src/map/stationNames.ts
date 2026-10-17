// A station's suggested names (T-065, T-534): the nearest cross street on the basemap, a small
// neighbourhood from the city's place list (game/placeList.ts), and the city or town. A new
// station takes the first there is; the station inspector's cycle button steps through them.
// Worked out when asked, never stored.

import type { Geometry } from "geojson";
import type { Map as MlMap } from "maplibre-gl";
import { compass } from "../game/places";
import { hoodsNear, placesNear, townsNear, type NearPlace } from "../game/placeList";
import { toLngLat, toLocal } from "./geo";
import type { NetworkBuffers } from "./network";

/** From this zoom the basemap's tiles carry street names. */
export const STREET_NAME_ZOOM = 13;
const STREET_M = 400;
/** A street within this angle of the track's direction runs along it, and never names a station on
 * it (DECISIONS 2026-10-16). */
const PARALLEL_DEG = 30;
/** A taken nearest place gives way to an untaken one at most this much further away. */
const PLACE_SLACK_M = 800;

export interface Suggestions {
  street: string | null;
  hood: string | null;
  town: string | null;
  /** today's fallbacks after those (any place on the list, then the tiles) */
  near: NearPlace[];
}

/** The track's direction at a local point, as a unit vector east and north, or null if no track
 * is within a few metres. */
export function trackDirection(net: NetworkBuffers | null, lx: number, ly: number): [number, number] | null {
  if (!net) return null;
  let best = 30, seg: [number, number, number, number] | null = null;
  for (const pts of net.edgePts.values())
    for (let k = 0; k + 5 < pts.length; k += 3) {
      const ax = pts[k], ay = pts[k + 1], bx = pts[k + 3], by = pts[k + 4];
      const dx = bx - ax, dy = by - ay, l2 = dx * dx + dy * dy;
      if (!l2) continue;
      const t = Math.max(0, Math.min(1, ((lx - ax) * dx + (ly - ay) * dy) / l2));
      const d = Math.hypot(ax + t * dx - lx, ay + t * dy - ly);
      if (d < best) (best = d), (seg = [ax, ay, bx, by]);
    }
  if (!seg) return null;
  const [a0, a1] = toLngLat(seg[0], seg[1], net.anchor), [b0, b1] = toLngLat(seg[2], seg[3], net.anchor);
  const ex = (b0 - a0) * Math.cos((a1 * Math.PI) / 180), ny = b1 - a1, l = Math.hypot(ex, ny);
  return l ? [ex / l, ny / l] : null;
}

/** The named features of a basemap source layer in the loaded tiles. */
function features(map: MlMap, sourceLayer: string): { geometry: Geometry; properties: Record<string, any> }[] {
  const layer = map.getStyle()?.layers.find((l) => (l as any)["source-layer"] === sourceLayer) as any;
  if (!layer) return [];
  try {
    return map.querySourceFeatures(layer.source, { sourceLayer }) as any;
  } catch {
    return [];
  }
}

const nameOf = (p: Record<string, any>) => (p?.["name:latin"] || p?.name_en || p?.name) as string | undefined;

/** A name that ends in "station" ("Light Rail Station", "Penn Sta.", "Stn") is a stop's own label
 * in OSM, and never a suggestion, unless it is a settlement's name (Huntington Station, a real
 * place: `kind` city, town, village or hamlet; T-568). */
const SETTLEMENT = new Set(["city", "town", "village", "hamlet"]);
export const endsInStation = (n: string, kind?: string) => /\b(station|sta|stn)\.?\s*$/i.test(n) && !(kind && SETTLEMENT.has(kind));

/** Metres from (lng, lat) to a geometry, and the direction of the line there (unit, east and
 * north; null for a point). */
function nearestOn(g: Geometry, lng: number, lat: number): { d: number; dir: [number, number] | null } {
  const kx = 111320 * Math.cos((lat * Math.PI) / 180), ky = 110540;
  const lines: number[][][] = g.type === "LineString" ? [g.coordinates] : g.type === "MultiLineString" ? g.coordinates : g.type === "Point" ? [[g.coordinates]] : [];
  let best = Infinity, dir: [number, number] | null = null;
  for (const l of lines) {
    const pts = l.map((c) => [(c[0] - lng) * kx, (c[1] - lat) * ky]);
    if (pts.length === 1) {
      const d = Math.hypot(pts[0][0], pts[0][1]);
      if (d < best) (best = d), (dir = null);
    }
    for (let i = 0; i + 1 < pts.length; i++) {
      const [ax, ay] = pts[i], [bx, by] = pts[i + 1];
      const dx = bx - ax, dy = by - ay, l2 = dx * dx + dy * dy;
      const t = l2 ? Math.max(0, Math.min(1, (-ax * dx - ay * dy) / l2)) : 0;
      const d = Math.hypot(ax + t * dx, ay + t * dy);
      if (d < best) (best = d), (dir = l2 ? [dx / Math.sqrt(l2), dy / Math.sqrt(l2)] : null);
    }
  }
  return { d: best, dir };
}

/** The nearest named street within STREET_M that crosses the track (more than PARALLEL_DEG off
 * its direction where it comes nearest), shortened; any street when the direction is unknown. */
function crossStreet(map: MlMap, lng: number, lat: number, track: [number, number] | null, taken: Set<string>): string | null {
  if (map.getZoom() < STREET_NAME_ZOOM) return null; // further out the tiles hold only the bigger roads
  const best = new Map<string, { d: number; dir: [number, number] | null }>();
  for (const f of features(map, "transportation_name")) {
    const raw = nameOf(f.properties);
    if (!raw || f.properties?.class === "motorway" || endsInStation(raw)) continue;
    const name = shorten(raw), n = nearestOn(f.geometry, lng, lat);
    if (n.d <= STREET_M && n.d < (best.get(name)?.d ?? Infinity)) best.set(name, n);
  }
  const sinMax = Math.sin((PARALLEL_DEG * Math.PI) / 180);
  return [...best]
    .filter(([, n]) => !track || !n.dir || Math.abs(track[0] * n.dir[1] - track[1] * n.dir[0]) > sinMax)
    .sort((a, b) => a[1].d - b[1].d)
    .map((e) => e[0])
    .find((n) => !taken.has(n)) ?? null;
}

/** A place on the loaded tiles within 1.2 km, for where the place list has nothing. */
function tilePlace(map: MlMap, lng: number, lat: number, taken: Set<string>): string | null {
  const kinds = ["neighbourhood", "quarter", "suburb", "village", "hamlet", "town"];
  const best = new Map<string, number>();
  for (const f of features(map, "place")) {
    const raw = nameOf(f.properties);
    if (!raw || !kinds.includes(f.properties?.class) || endsInStation(raw, f.properties?.class)) continue;
    const name = shorten(raw), d = nearestOn(f.geometry, lng, lat).d;
    if (d <= 1200 && d < (best.get(name) ?? Infinity)) best.set(name, d);
  }
  return [...best].sort((a, b) => a[1] - b[1]).map((e) => e[0]).find((n) => !taken.has(n)) ?? null;
}

/** The suggestions for a station at (lng, lat) whose track runs along `track`; `taken` holds the
 * other stations' names, which each suggestion avoids. */
export async function suggestNames(map: MlMap, lng: number, lat: number, track: [number, number] | null, taken: () => Set<string>): Promise<Suggestions> {
  const street = crossStreet(map, lng, lat, track, taken());
  const [listed, listedTowns] = await Promise.all([placesNear(lng, lat).catch(() => []), townsNear(lng, lat).catch(() => [])]);
  const keep = (n: NearPlace) => !endsInStation(n.place.name, n.place.kind);
  const near = listed.filter(keep), towns = listedTowns.filter(keep);
  const t = taken(); // a station added meanwhile counts
  return { street, hood: placeName(hoodsNear(near), t), town: placeName(towns, t), near };
}

/** A new station's name: the cross street, else a small neighbourhood, else the nearest place on
 * the list, else a place on the tiles, else "Station N". */
export async function newStationName(map: MlMap, lng: number, lat: number, track: [number, number] | null, taken: () => Set<string>): Promise<string> {
  const s = await suggestNames(map, lng, lat, track, taken);
  if (s.street) return s.street;
  if (s.hood) return s.hood;
  const t = taken();
  const listed = placeName(s.near, t) ?? tilePlace(map, lng, lat, t);
  if (listed) return listed;
  for (let k = 1; ; k++) if (!t.has(`Station ${k}`)) return `Station ${k}`;
}

/** The suggestion after `current` for a station at (lng, lat), wrapping round (the first when
 * `current` is none of them); null when there is nothing else to offer. */
export async function nextSuggestion(map: MlMap, lng: number, lat: number, current: string, taken: () => Set<string>, net: NetworkBuffers | null): Promise<string | null> {
  const [lx, ly] = toLocal(lng, lat, net?.anchor);
  const s = await suggestNames(map, lng, lat, trackDirection(net, lx, ly), taken);
  const list = [...new Set([s.street, s.hood, s.town].filter((n): n is string => !!n))];
  if (!list.length) return null;
  const next = list[(list.indexOf(current) + 1) % list.length];
  return next === current ? null : next;
}

/**
 * A name from places near a point (nearest or best first): the first, or if that name is taken
 * another within PLACE_SLACK_M further, or the first with the way the station lies from it
 * ("North Williamsburg"), or with a number ("Williamsburg 2"). Null when no place is near.
 */
export function placeName(near: NearPlace[], taken: Set<string>): string | null {
  if (!near.length) return null;
  const first = near[0];
  const free = near.find((n) => n.d <= first.d + PLACE_SLACK_M && !taken.has(n.place.name));
  if (free) return free.place.name;
  if (first.d > 150) {
    const way = compass(first.dx, first.dy);
    const named = `${way[0].toUpperCase()}${way.slice(1)} ${first.place.name}`;
    if (!taken.has(named)) return named;
  }
  for (let k = 2; ; k++) if (!taken.has(`${first.place.name} ${k}`)) return `${first.place.name} ${k}`;
}

/** "West 42nd Street" -> "W 42nd St". */
function shorten(n: string): string {
  return n
    .replace(/\bWest\b/g, "W")
    .replace(/\bEast\b/g, "E")
    .replace(/\bNorth\b/g, "N")
    .replace(/\bSouth\b/g, "S")
    .replace(/\bStreet\b/g, "St")
    .replace(/\bAvenue\b/g, "Av")
    .replace(/\bBoulevard\b/g, "Blvd")
    .replace(/\bRoad\b/g, "Rd")
    .replace(/\bParkway\b/g, "Pkwy")
    .replace(/\bPlace\b/g, "Pl")
    .replace(/\bExpressway\b/g, "Expy");
}
