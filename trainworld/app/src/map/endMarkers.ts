// Point markers on the map (T-416). While the track or station tool is on, a ring at each loose end
// (a node with one piece of track and no station), built or an active blueprint, near the pointer: where a
// line is continued or a station goes at the end of one. With the select tool, a square marker at
// every blueprint point that can be dragged (map/nodeDrag.ts): track ends, corners and junctions.
// The track tool has both (T-1811): the squares, a blueprint loose end's in the accent, and the
// rings near the pointer at built loose ends. With either, a square at both ends of a blueprint
// platform (T-1869, map/dragHandles.ts).
// They take no presses; the tools and the node drag work on the nodes underneath.

import type { Map as MlMap } from "maplibre-gl";
import type { TrackEdge, TrackNode, WorldView } from "../game/types";
import { checkPatches, worldDiff, type WorldDiff } from "../game/worldDiff";
import type { NetworkBuffers } from "./network";
import { display, tool } from "../game/ui";
import { world } from "../game/world";
import { effect } from "@preact/signals";
import { anchor, toLocal } from "./geo";
import { live } from "./live";
import { hidden, onBlueprintsHidden } from "./visibility";
import { M_JUNCTION, MARK_STRIDE } from "./markData";
import { MARKS_ZOOM, NODE_HIT_PX, stationEnds } from "./dragHandles";
import type { Overlay } from "./overlay";
import { marksLayer, onScreen, placeAt } from "./screenPlace";

type Kind = "loose" | "drag" | "track";
type Pt = { lx: number; ly: number; cls: Cls; stub?: Float32Array };
/** past this many changed nodes and edges the points are found again everywhere */
const PATCH_MAX = 2000;
type Cls = "end-mark" | "pi-mark" | "pi-mark loose";
const HOT = " hot";

/** below this zoom the markers would only crowd the map */
const MIN_ZOOM = MARKS_ZOOM;
const MAX_MARKS = 400;
/** a badge and the node it sits on agree to within this, local units (the cells are looked up with their neighbours) */
const BADGE_CELL = 3;
/** T-1485: loose ends show within this of the pointer, CSS px; every one in a city at once, many on
 * faint unserved or switched-off track, read as stray stations */
const NEAR_PX = 160;
/** T-1534: a loose end on a stub shorter than this on screen has no ring: the track cannot be seen */
const MIN_STUB_PX = 4;
/** a screen with no hovering pointer (touch) shows them all */
const NO_HOVER = typeof matchMedia === "function" && matchMedia("(hover: none)").matches;

export class EndMarkers {
  private box: HTMLDivElement;
  private marks: HTMLDivElement[] = [];
  /** `stub`: a loose end's track, local units; the ring is left off while it is under `MIN_STUB_PX` long */
  private pts: Pt[] = [];
  private scratch = [0, 0, 0, 0];
  private kind: Kind | null = null;
  private version = -1;
  private net: unknown = null;
  /** the pointer over the map, CSS px, or null when it is off the map */
  private pointer: [number, number] | null = null;
  /** the map's box on the page, read again after a resize */
  private rect: DOMRect | null = null;
  private anchorV = -1;
  private hiddenV = -1;

  constructor(private map: MlMap, private overlay: Overlay) {
    this.box = marksLayer(map);
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
      const k = this.wanted();
      if (k) this.place();
    }, { passive: true });
    map.on("resize", () => (this.rect = null));
    addEventListener("resize", () => (this.rect = null));
    map.getContainer().addEventListener("mouseleave", () => {
      this.pointer = null;
      this.place();
    });
    onBlueprintsHidden(() => this.place());
    effect(() => {
      tool.value;
      world.value;
      this.place();
    });
  }

  /** The platforms' ends, px (T-1869). */
  private handles(): { x: number; y: number; cls: Cls }[] {
    return stationEnds(this.overlay).map((e) => ({ x: e.x, y: e.y, cls: "pi-mark" as Cls }));
  }

  private wanted(): Kind | null {
    const t = tool.peek();
    return t === "track" ? "track" : t === "station" ? "loose" : t === "select" ? "drag" : null;
  }

  /** The points to mark, found again only when the tool or the world changes: at the nodes a new world changed and the
   * ends of the edges it changed (T-4150), everywhere for a new tool, draw anchor or visibility. */
  private refresh(kind: Kind) {
    const w = world.peek(), net = this.overlay.renderer.net;
    const v = w?.version ?? -1, av = anchor().version;
    // the points are in local units, so they go stale when the draw anchor moves
    if (kind === this.kind && v === this.version && net === this.net && av === this.anchorV && hidden.v === this.hiddenV) return;
    const netMoved = net !== this.net;
    // a network built for an older world than this one has not drawn its edits yet (a state's world comes before its
    // network, a staged state's network later still): what they changed is looked at again with the next network
    const netHolds = this.overlay.renderer.netVersion >= v;
    const d = w && kind === this.kind && av === this.anchorV && hidden.v === this.hiddenV && !(netMoved && this.wholeStale) ? worldDiff(this.from, w) : null;
    this.kind = kind;
    this.version = v;
    this.net = net;
    this.anchorV = av;
    this.hiddenV = hidden.v;
    this.from = w;
    if (!w) {
      this.pts = [];
      this.byNode.clear();
      this.incident.clear();
      this.edgeBy.clear();
      this.stale.clear();
      this.wholeStale = false;
      return;
    }
    if (d && d.nodes.size + d.edges.size + this.stale.size <= PATCH_MAX) this.patchPts(kind, w, net, d, netMoved, netHolds);
    else {
      this.allPts(kind, w, net);
      this.stale.clear();
      this.wholeStale = !netHolds;
    }
    // in the network's node order, as found whole
    this.pts = [...this.byNode].sort((a, b) => a[0] - b[0]).map(([, p]) => p);
    if (checkPatches() && d) {
      const got = this.pts.map((p) => `${p.lx},${p.ly},${p.cls},${p.stub?.length ?? -1}`).join("|");
      this.allPts(kind, w, net);
      const whole = [...this.byNode].sort((a, b) => a[0] - b[0]).map(([, p]) => `${p.lx},${p.ly},${p.cls},${p.stub?.length ?? -1}`).join("|");
      if (got !== whole) console.error("T-4150 check: the end markers patched differ from a whole build");
    }
  }

  /** T-4150: the world the points were found for, each marked node's point, the edges at each node (by id) and the edges */
  private from: WorldView | null = null;
  private byNode = new Map<number, Pt>();
  private incident = new Map<number, number[]>();
  private edgeBy = new Map<number, TrackEdge>();
  /** the nodes worlds changed since the drawn network was last one built for the world in hand: a world comes before its
   * network, and a point is only where the network draws track, so they are looked at again with each network until one
   * holds them */
  private stale = new Set<number>();
  /** the points were all found on a network not built for their world (none drawn yet, a load's): found again on the next */
  private wholeStale = false;

  private allPts(kind: Kind, w: WorldView, net: NetworkBuffers | null) {
    this.byNode.clear();
    this.incident.clear();
    this.edgeBy.clear();
    for (const e of w.edges) this.addEdge(e);
    for (const n of w.nodes.values()) {
      const p = this.ptAt(kind, n, net);
      if (p) this.byNode.set(n.id, p);
    }
  }

  private patchPts(kind: Kind, w: WorldView, net: NetworkBuffers | null, d: WorldDiff, netMoved: boolean, netHolds: boolean) {
    const touched = new Set<number>(d.nodes.keys());
    for (const [id, [a, b]] of d.edges) {
      if (a) {
        for (const at of [a.a, a.b]) {
          const v = this.incident.get(at), i = v ? v.indexOf(id) : -1;
          if (i >= 0) v!.splice(i, 1);
          touched.add(at);
        }
        this.edgeBy.delete(id);
      }
      if (b) {
        this.addEdge(b);
        touched.add(b.a).add(b.b);
      }
    }
    for (const id of touched) this.stale.add(id);
    for (const id of netMoved ? this.stale : touched) {
      const n = w.nodes.get(id), p = n ? this.ptAt(kind, n, net) : null;
      if (p) this.byNode.set(id, p);
      else this.byNode.delete(id);
    }
    if (netMoved && netHolds) this.stale.clear();
  }

  private addEdge(e: TrackEdge) {
    this.edgeBy.set(e.id, e);
    for (const at of e.a === e.b ? [e.a] : [e.a, e.b]) {
      const v = this.incident.get(at);
      if (v) v.push(e.id);
      else this.incident.set(at, [e.id]);
    }
  }

  /** A node's point, if it has one: only where the network draws its track, so a mark is never left on its own (T-1485) */
  private ptAt(kind: Kind, n: TrackNode, net: NetworkBuffers | null): Pt | null {
    if (n.platform > 0 || n.bus) return null;
    // (the last drawn edge in the network's order, as a pass over the edges would leave it)
    let e: TrackEdge | undefined;
    for (const id of this.incident.get(n.id) ?? []) if (net?.edgePts.has(id) && (!e || id > e.id)) e = this.edgeBy.get(id);
    if (!e) return null;
    // Settings' Show blueprints off: no square on a blueprint-only point, no ring at a blueprint end
    const square = kind !== "loose" && n.builtPorts === 0 && !hidden.blueprints;
    let loose = kind !== "drag" && n.ports === 1;
    // T-1534: not the end of a blueprint that is switched off, which is drawn grey and faint
    if (loose && !e.built && (e.off || hidden.blueprints)) loose = false;
    if (!square && !loose) return null;
    const [lx, ly] = toLocal(n.lng, n.lat);
    if (square) return { lx, ly, cls: loose ? "pi-mark loose" : "pi-mark" };
    return { lx, ly, cls: "end-mark", stub: net?.edgePts.get(e.id) };
  }

  /** Whether a stub of track is `MIN_STUB_PX` or more long on the screen now. */
  private visible(stub: Float32Array): boolean {
    const o = this.scratch;
    let len = 0;
    this.overlay.toScreenInto(stub[0], stub[1], o, 0);
    for (let k = 3; k < stub.length; k += 3) {
      this.overlay.toScreenInto(stub[k], stub[k + 1], o, 2);
      len += Math.hypot(o[2] - o[0], o[3] - o[1]);
      if (len >= MIN_STUB_PX) return true;
      o[0] = o[2];
      o[1] = o[3];
    }
    return false;
  }

  /** Whether a select tool's square is drawn within `NODE_HIT_PX` of a screen point: drawn over the demand
   * bubbles, so a click there acts on the point, not the bubble (T-486). */
  at(px: number, py: number): boolean {
    if (this.wanted() !== "drag" || !this.shown()) return false;
    this.refresh("drag");
    for (const p of this.pts) {
      const [x, y] = this.overlay.toScreen(p.lx, p.ly);
      if ((x - px) ** 2 + (y - py) ** 2 < NODE_HIT_PX * NODE_HIT_PX) return true;
    }
    return !hidden.blueprints && this.handles().some((h) => (h.x - px) ** 2 + (h.y - py) ** 2 < NODE_HIT_PX * NODE_HIT_PX);
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
      const near = kind !== "drag" && !NO_HOVER ? this.pointer : undefined;
      // T-3897: the square under the pointer lights up, the one a press would drag
      const hs = kind !== "loose" && !hidden.blueprints ? this.handles() : [];
      const hot = NO_HOVER ? null : this.hotAt(kind, hs);
      for (const p of this.pts) {
        const ring = p.cls === "end-mark";
        if (ring && near === null) continue;
        const [x, y] = this.overlay.toScreen(p.lx, p.ly);
        if (!onScreen(x, y, w, h, 8)) continue;
        if (ring && near && (x - near[0]) ** 2 + (y - near[1]) ** 2 > NEAR_PX * NEAR_PX) continue;
        if (!ring && this.badged(p.lx, p.ly)) continue;
        if (p.stub && !this.visible(p.stub)) continue;
        if (k >= MAX_MARKS) break;
        this.mark(k++, ring || p !== hot ? p.cls : ((p.cls + HOT) as Cls), x, y);
      }
      if (kind !== "loose")
        for (const p of hs) {
          if (!onScreen(p.x, p.y, w, h, 8)) continue;
          if (k >= MAX_MARKS) break;
          this.mark(k++, p === hot ? ((p.cls + HOT) as Cls) : p.cls, p.x, p.y);
        }
    }
    for (let i = k; i < this.marks.length; i++) this.marks[i].style.display = "none";
  }

  /** The square nearest the pointer within `NODE_HIT_PX`, as the node drag finds it. */
  private hotAt(kind: Kind, hs: { x: number; y: number }[]): object | null {
    const q = this.pointer;
    if (!q || kind === "loose") return null;
    let best = NODE_HIT_PX * NODE_HIT_PX, hot: object | null = null;
    for (const p of [...this.pts, ...hs]) {
      if ("cls" in p && p.cls === "end-mark") continue;
      const [x, y] = "lx" in p ? this.overlay.toScreen(p.lx, p.ly) : [p.x, p.y];
      const d = (x - q[0]) ** 2 + (y - q[1]) ** 2;
      if (d < best) [best, hot] = [d, p];
    }
    return hot;
  }

  private mark(k: number, cls: Cls, x: number, y: number) {
    let m = this.marks[k];
    if (!m) {
      m = this.marks[k] = document.createElement("div");
      this.box.appendChild(m);
    }
    if (m.className !== cls) m.className = cls;
    m.style.display = "";
    placeAt(m, x, y);
  }
}
