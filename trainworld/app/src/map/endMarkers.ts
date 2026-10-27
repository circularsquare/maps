// Point markers on the map (T-416). While the track or station tool is on, a ring at each loose end
// (a node with one piece of track and no station), blueprint or built, near the pointer: where a
// line is continued or a station goes at the end of one. With the select tool, a square marker at
// every blueprint point that can be dragged (map/nodeDrag.ts): track ends, corners and junctions.
// They take no presses; the tools and the node drag work on the nodes underneath as before.

import type { Map as MlMap } from "maplibre-gl";
import { display, tool } from "../game/ui";
import { world } from "../game/world";
import { effect } from "@preact/signals";
import { anchor, toLocal } from "./geo";
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
/** T-1485: loose ends show within this of the pointer, CSS px; every one in a city at once, many on
 * faint unserved or switched-off track, read as stray stations */
const NEAR_PX = 160;
/** a screen with no hovering pointer (touch) shows them all */
const NO_HOVER = typeof matchMedia === "function" && matchMedia("(hover: none)").matches;

export class EndMarkers {
  private box: HTMLDivElement;
  private marks: HTMLDivElement[] = [];
  private pts: { lx: number; ly: number }[] = [];
  private kind: Kind | null = null;
  private version = -1;
  private net: unknown = null;
  /** the pointer over the map, CSS px, or null when it is off the map */
  private pointer: [number, number] | null = null;
  /** the map's box on the page, read again after a resize */
  private rect: DOMRect | null = null;
  private anchorV = -1;

  constructor(private map: MlMap, private overlay: Overlay) {
    this.box = document.createElement("div");
    this.box.className = "tool-marks";
    map.getContainer().appendChild(this.box);
    overlay.afterMapFrame.push(() => this.place());
    // T-1522: from the window, not MapLibre's mousemove, which stops while the map is dragged and over
    // anything laid on the map (the view tabs, a card), and left the rings round where it last was
    addEventListener("pointermove", (e) => {
      const on = e.target instanceof Node && map.getCanvasContainer().contains(e.target);
      if (on) {
        this.rect ??= map.getContainer().getBoundingClientRect();
        this.pointer = [e.clientX - this.rect.left, e.clientY - this.rect.top];
      } else if (this.pointer) this.pointer = null;
      else return;
      if (this.wanted() === "loose") this.place();
    }, { passive: true });
    map.on("resize", () => (this.rect = null));
    addEventListener("resize", () => (this.rect = null));
    map.getContainer().addEventListener("mouseleave", () => {
      this.pointer = null;
      this.place();
    });
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
    const w = world.peek(), net = this.overlay.renderer.net;
    const v = w?.version ?? -1, av = anchor().version;
    // the points are in local units, so they go stale when the draw anchor moves
    if (kind === this.kind && v === this.version && net === this.net && av === this.anchorV) return;
    this.kind = kind;
    this.version = v;
    this.net = net;
    this.anchorV = av;
    this.pts = [];
    if (!w) return;
    // only where the network draws the track: a mark is never left on its own (T-1485)
    const drawnAt = new Set<number>();
    for (const e of w.edges) if (net?.edgePts.has(e.id)) drawnAt.add(e.a).add(e.b);
    for (const n of w.nodes.values()) {
      if (n.platform > 0 || n.bus || !drawnAt.has(n.id)) continue;
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
      const near = kind === "loose" && !NO_HOVER ? this.pointer : undefined;
      for (const p of this.pts) {
        if (near === null) break;
        const [x, y] = this.overlay.toScreen(p.lx, p.ly);
        if (x < -8 || y < -8 || x > w + 8 || y > h + 8) continue;
        if (near && (x - near[0]) ** 2 + (y - near[1]) ** 2 > NEAR_PX * NEAR_PX) continue;
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
