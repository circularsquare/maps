// World metres (the track model's frame, game/frame.ts) to and from lng/lat. Which frame it is
// comes with each loaded game: the clock worker reports it (`ClockClient`), and `setWorldFrame`
// (game/cities.ts) holds it. Old saves are in the legacy equirectangular frame or the Northeast
// conic, new games in the Mercator frame of the whole Earth (T-306, docs/world.md).
//
// In the Mercator frame a world metre is not a ground metre: it is `groundScale(y)` world metres
// to a ground metre (1.32 at New York, 1.61 at London). A metre figure compared with world
// coordinates goes through `groundScale` or `groundDist`. Lengths from the track model (`lengthM`,
// chainage) are already ground metres.

import { getWorldFrame } from "./cities";
import { mercK } from "./frame";


export function lngLatToPack(lng: number, lat: number): { x: number; y: number } {
  const [x, y] = getWorldFrame().fromLonLat(lng, lat);
  return { x, y };
}

export function packToLngLat(x: number, y: number): [number, number] {
  return getWorldFrame().toLonLat(x, y);
}

/** World metres per ground metre at world `y`: 1 except in the Mercator frame. */
export function groundScale(y: number): number {
  return getWorldFrame().kind === "merc" ? mercK(y) : 1;
}

/** Ground metres between two nearby world points (within a few km; exact enough to 100 km). */
export function groundDist(ax: number, ay: number, bx: number, by: number): number {
  return Math.hypot(ax - bx, ay - by) / groundScale((ay + by) / 2);
}
