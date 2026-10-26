// Point markers on the map (T-416). While the track or station tool is on, a round marker at each
// loose end (a node with one piece of track and no station), blueprint or built: where a line is
// continued or a station goes at the end of one. With the select tool, a square marker at every
// blueprint point that can be dragged (map/nodeDrag.ts): track ends, corners and junctions. They
// take no presses; the tools and the node drag work on the nodes underneath as before.

import type { Map as MlMap } from "maplibre-gl";
import { display, tool } from "../game/ui";
import { world } from "../game/world";
import { effect } from "@preact/signals";
import { toLocal } from "./geo";
import { live } from "./live";
import { M_JUNCTION, MARK_STRIDE } from "./markData";
import type { Overlay } from "./overlay";

type Kind = "loose" | "drag";

/** below this zoom the markers would only crowd the map */
const MIN_ZOOM = 9;
const MAX_MARKS = 400;
/** a point's reach for a click, px: nodeDrag's, so a press that drags and one that clicks agree */
const HIT_PX = 9;
/** a badge and the node it sits on agree to within this, local units (the cells are looked up with their neighbours) */
const BADGE_CELL = 3;

export class EndMarkers {
  private box: HTMLDivElement;
  private marks: HTMLDivElement[] = [];
  private pts: { lx: number; ly: number }[] = [];
  private kind: Kind | null = null;
  private version = -1;

  constructor(private map: MlMap, private overlay: Overlay) {
    this.box = document.createElement("div");
    this.box.className = "tool-marks";
    map.getContainer().appendChild(this.box);
    overlay.afterMapFrame.push(() => this.place());
    effect(() => {
      tool.value;
      world.value;
      this.place();
    });
  }

  private wanted(): Kind | null {
    const t = tool.peek();
    return t === "track" || t === "station" ? "loose" : t === "select" ? "drag" : null;
  }

  /** The points to mark, found again only when the tool or the world changes. */
  private refresh(kind: Kind) {
    const w = world.peek();
    const v = w?.version ?? -1;
    if (kind === this.kind && v === this.version) return;
    this.kind = kind;
    this.version = v;
    this.pts = [];
    if (!w) return;
    for (const n of w.nodes.values()) {
      if (n.platform > 0 || n.bus) continue;
      if (kind === "loose" ? n.ports !== 1 : n.builtPorts !== 0) continue;
      const [lx, ly] = toLocal(n.lng, n.lat);
      this.pts.push({ lx, ly });
    }
  }

  /** Whether a select tool's square is drawn within `HIT_PX` of a screen point: drawn over the demand
   * bubbles, so a click there acts on the point, not the bubble (T-486). */
  at(px: number, py: number): boolean {
    if (this.wanted() !== "drag" || !this.shown()) return false;
    this.refresh("drag");
    for (const p of this.pts) {
      const [x, y] = this.overlay.toScreen(p.lx, p.ly);
      if ((x - px) ** 2 + (y - py) ** 2 < HIT_PX * HIT_PX) return true;
    }
    return false;
  }

  private shown(): boolean {
    return this.map.getZoom() >= MIN_ZOOM && !live.nodeDrag?.dragging;
  }

  /** T-663: a blueprint junction's fork badge is its own handle, and the square would cover it. The
   * badges' places in cells of BADGE_CELL, found again when the network's marks are replaced. */
  private badges: { marks: Float32Array | undefined; cells: Set<string> } = { marks: undefined, cells: new Set() };
  private badged(lx: number, ly: number): boolean {
    const marks = this.overlay.renderer.net?.marks;
    if (!marks || !display.peek().flatMarks) return false;
    if (marks !== this.badges.marks) {
      const cells = new Set<string>();
      for (let i = 0; i < marks.length; i += MARK_STRIDE) {
        if (marks[i + 2] === M_JUNCTION && marks[i + 4] > 0.5) cells.add(`${Math.round(marks[i] / BADGE_CELL)},${Math.round(marks[i + 1] / BADGE_CELL)}`);
      }
      this.badges = { marks, cells };
    }
    const cx = Math.round(lx / BADGE_CELL), cy = Math.round(ly / BADGE_CELL);
    for (let dx = -1; dx <= 1; dx++) for (let dy = -1; dy <= 1; dy++) if (this.badges.cells.has(`${cx + dx},${cy + dy}`)) return true;
    return false;
  }

  place() {
    const kind = this.wanted();
    let k = 0;
    if (kind && this.shown()) {
      this.refresh(kind);
      const w = this.overlay.cssW, h = this.overlay.cssH;
      for (const p of this.pts) {
        const [x, y] = this.overlay.toScreen(p.lx, p.ly);
        if (x < -8 || y < -8 || x > w + 8 || y > h + 8) continue;
        if (kind === "drag" && this.badged(p.lx, p.ly)) continue;
        if (k >= MAX_MARKS) break;
        let m = this.marks[k];
        if (!m) {
          m = this.marks[k] = document.createElement("div");
          this.box.appendChild(m);
        }
        const cls = kind === "loose" ? "end-mark" : "pi-mark";
        if (m.className !== cls) m.className = cls;
        m.style.display = "";
        m.style.transform = `translate(${Math.round(x)}px, ${Math.round(y)}px)`;
        k++;
      }
    }
    for (let i = k; i < this.marks.length; i++) this.marks[i].style.display = "none";
  }
}
