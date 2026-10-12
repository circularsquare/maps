// World metres (the track model's frame, game/frame.ts) to and from lng/lat. Which frame it is
// comes with each loaded game: the clock worker reports it (`ClockClient`), and `setWorldFrame`
// (game/cities.ts) holds it. Old saves are in the legacy equirectangular frame, new games in the
// Lambert conformal one.

import { getWorldFrame } from "./cities";


export function lngLatToPack(lng: number, lat: number): { x: number; y: number } {
  const [x, y] = getWorldFrame().fromLonLat(lng, lat);
  return { x, y };
}

export function packToLngLat(x: number, y: number): [number, number] {
  return getWorldFrame().toLonLat(x, y);
}
