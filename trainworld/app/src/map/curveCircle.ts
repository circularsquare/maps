// The circle a segment being drawn curves along (T-418), from a route preview's `arc`.

const EARTH_R = 6371008.8;

/**
 * Where to draw a preview's circle: `arc` is centre x, y (local units), ground radius m, 0 (the
 * stride of `verts`, so the anchor shift treats it alike). Screen centre and radius in px, the
 * radius measured east of the centre. Null without an arc.
 */
export function circleOnScreen(
  arc: ArrayLike<number> | undefined,
  screen: (lx: number, ly: number) => [number, number],
  toLngLat: (lx: number, ly: number) => [number, number],
  toLocal: (lng: number, lat: number) => [number, number],
): { x: number; y: number; r: number } | null {
  if (!arc || arc.length < 3 || !(arc[2] > 0)) return null;
  const [x, y] = screen(arc[0], arc[1]);
  const [lng, lat] = toLngLat(arc[0], arc[1]);
  const dLng = ((arc[2] / (EARTH_R * Math.cos((lat * Math.PI) / 180))) * 180) / Math.PI;
  const [ex, ey] = screen(...toLocal(lng + dLng, lat));
  return { x, y, r: Math.hypot(ex - x, ey - y) };
}
