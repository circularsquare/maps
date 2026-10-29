// Moving what is drawn over the map in screen space (tool marks, ghosts, rings, tags at a place,
// markers and names) to where its place is on the screen (T-1777). Every placer that projects a
// map point goes through `placeAt`.
//
// A point with no place on the screen projects to NaN (the far side of the globe, `mercToCss`). The
// browser refuses a NaN transform and keeps the element where it last was, so it stays put on the
// screen while the map moves under it; NaN also passes every "is it off the screen" comparison. Here
// such an element is hidden (visibility, so the callers' own display switching is untouched) until
// its point is on the globe's near side again.

type Placed = HTMLElement | SVGElement;

/**
 * Move `el` to (x, y) CSS px in the map container, plus (dx, dy), rounded to whole px unless
 * `round` is false, writing only what changed. Hidden while (x, y) has no place on the screen.
 * Whether it shows.
 */
export function placeAt(el: Placed, x: number, y: number, dx = 0, dy = 0, round = true): boolean {
  const ok = Number.isFinite(x) && Number.isFinite(y);
  const vis = ok ? "" : "hidden";
  if (el.style.visibility !== vis) el.style.visibility = vis;
  if (!ok) return false;
  const px = round ? Math.round(x + dx) : x + dx, py = round ? Math.round(y + dy) : y + dy;
  const t = `translate(${px}px, ${py}px)`;
  if (el.style.transform !== t) el.style.transform = t;
  return true;
}

/** (x, y) is within `pad` px of a `w` by `h` screen; false for a point with no place on it (NaN). */
export function onScreen(x: number, y: number, w: number, h: number, pad = 0): boolean {
  return x >= -pad && y >= -pad && x <= w + pad && y <= h + pad;
}
