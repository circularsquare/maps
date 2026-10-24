// Shared visibility and breathing room for network annotations, and the pieces their tags share.
import type { Map as MlMap } from "maplibre-gl";

export const MIN_LABEL_ZOOM = 12;
export function labelPadding(zoom: number): [number, number] {
  return zoom < 14 ? [24, 14] : [10, 6];
}

/** px from the pointer to a tag beside it (the hover tags of the track, the bubbles and the rest
 * tag, which sits flush against them). */
export const TAG_OFFSET_PX = 14;

/** A tag's lines: one div each for a string, an element as it is. */
export function tagLines(el: HTMLElement, lines: readonly (string | HTMLElement)[]) {
  el.replaceChildren(...lines.map((s) => (typeof s === "string" ? Object.assign(document.createElement("div"), { textContent: s }) : s)));
}

/** Show `tip` (filled with `lines`) 4 px below `el`, or to its right, in `container`'s px. */
export function tipBeside(tip: HTMLElement, el: HTMLElement, container: HTMLElement, lines: readonly string[], side: "below" | "right") {
  tagLines(tip, lines);
  const r = el.getBoundingClientRect(), c = container.getBoundingClientRect();
  const [x, y] = side === "below" ? [r.left - c.left, r.bottom - c.top + 4] : [r.right - c.left + 4, r.top - c.top];
  tip.style.transform = `translate(${Math.round(x)}px, ${Math.round(y)}px)`;
  tip.style.display = "";
}

const BIN = 64;
/** Boxes (left, top, right, bottom) filed by the 64 px grid cells they touch, so a new box is tested
 * only against its neighbours (T-689: against every one placed took most of a zoom frame). */
export class BoxGrid {
  private cells = new Map<number, number[][]>();

  private each(r: number[], f: (cell: number) => boolean | void): boolean {
    for (let bx = Math.floor(r[0] / BIN); bx <= Math.floor(r[2] / BIN); bx++)
      for (let by = Math.floor(r[1] / BIN); by <= Math.floor(r[3] / BIN); by++) if (f(bx * 4096 + by)) return true;
    return false;
  }

  /** Whether `r` overlaps a box added. */
  hits(r: number[]): boolean {
    return this.each(r, (k) => this.cells.get(k)?.some((t) => r[0] < t[2] && r[2] > t[0] && r[1] < t[3] && r[3] > t[1]));
  }

  add(r: number[]) {
    this.each(r, (k) => {
      const b = this.cells.get(k);
      if (b) b.push(r);
      else this.cells.set(k, [r]);
    });
  }
}

/** The part of a map box (west, south, east, north in degrees) that is on screen in a `w` by `h`
 * map, kept `edge` px inside it, as px: left, right, top, bottom. Nothing of it is on screen when
 * `left > right` or `top > bottom`. */
export function screenBounds(map: MlMap, box: readonly [number, number, number, number], w: number, h: number, edge: number) {
  const [x0, y0, x1, y1] = box;
  const corners = [[x0, y0], [x1, y0], [x0, y1], [x1, y1]].map(([x, y]) => map.project([x, y]));
  return {
    left: Math.max(edge, Math.min(...corners.map((p) => p.x))),
    right: Math.min(w - edge, Math.max(...corners.map((p) => p.x))),
    top: Math.max(edge, Math.min(...corners.map((p) => p.y))),
    bottom: Math.min(h - edge, Math.max(...corners.map((p) => p.y))),
  };
}
