// The circle a segment being drawn curves along (T-418), from a route preview's `arc`, and the way
// the segment arrives at its end (T-1895).

import { EARTH_R } from "../game/frame";
import { placeAt } from "./screenPlace";

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

/**
 * Where a previewed segment ends and the way it arrives there, a unit vector (local units), for
 * the line on from its end (T-1895). `verts` is the preview's vertex rows (x, y, radius, speed),
 * the last one its end; `pts` its ghost (x, y, height triples; a quad's other pair after the drawn
 * one). The last step of the ghost gives the way; where the segment ends on its circle `arc`, the
 * circle's own tangent there, since a step along a curve turns less than the curve. Null when
 * there is too little to go on.
 */
export function endHeading(
  verts: ArrayLike<number>,
  pts: ArrayLike<number>,
  arc: ArrayLike<number> | undefined,
  toLngLat: (lx: number, ly: number) => [number, number],
  toLocal: (lng: number, lat: number) => [number, number],
): { x: number; y: number; ux: number; uy: number } | null {
  if (verts.length < 8 || pts.length < 6) return null;
  const x = verts[verts.length - 4], y = verts[verts.length - 3];
  let k = -1;
  for (let i = 3; i + 2 < pts.length; i += 3) {
    if (Math.hypot(pts[i] - x, pts[i + 1] - y) < 0.5) { k = i; break; }
  }
  if (k < 3) return null;
  let ux = pts[k] - pts[k - 3], uy = pts[k + 1] - pts[k - 2];
  const l = Math.hypot(ux, uy);
  if (!(l > 1e-6)) return null;
  ux /= l; uy /= l;
  if (arc && arc.length >= 3 && arc[2] > 0) {
    const [lng, lat] = toLngLat(arc[0], arc[1]);
    const dLng = ((arc[2] / (EARTH_R * Math.cos((lat * Math.PI) / 180))) * 180) / Math.PI;
    const [ex, ey] = toLocal(lng + dLng, lat);
    const r = Math.hypot(ex - arc[0], ey - arc[1]);
    const rx = x - arc[0], ry = y - arc[1], d = Math.hypot(rx, ry);
    if (r > 0 && Math.abs(d - r) < 0.005 * r) {
      let tx = -ry / d, ty = rx / d;
      if (tx * ux + ty * uy < 0) tx = -tx, ty = -ty;
      ux = tx; uy = ty;
    }
  }
  return { x, y, ux, uy };
}

/** how far the line on from a segment's end reaches, px (T-1895) */
const TANGENT_PX = 120;

/** T-1895, T-2765: draw (or hide, `h` null) the line on from a segment's end in the 1 px SVG `svg`
 * (a `line` child; the track tool's tangent and the node drag's share it). */
export function drawTangent(
  svg: SVGSVGElement,
  h: { x: number; y: number; ux: number; uy: number } | null,
  ok: boolean,
  screen: (lx: number, ly: number) => [number, number],
) {
  if (!h) { svg.style.display = "none"; return; }
  const [x0, y0] = screen(h.x, h.y);
  const [x1, y1] = screen(h.x + h.ux, h.y + h.uy);
  const l = Math.hypot(x1 - x0, y1 - y0);
  if (!(l > 0)) { svg.style.display = "none"; return; }
  svg.style.display = "";
  svg.setAttribute("class", "curve-circle" + (ok ? "" : " bad"));
  placeAt(svg, x0, y0, 0, 0, false);
  const line = svg.firstElementChild!;
  line.setAttribute("x2", String(((x1 - x0) / l) * TANGENT_PX));
  line.setAttribute("y2", String(((y1 - y0) / l) * TANGENT_PX));
}
