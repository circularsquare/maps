// The blueprint tool (T-333): a drag draws a box on the map, and the blueprint track it touches and
// the blueprint stations inside it are picked (game/blueprintPick.ts). Shift adds to what is picked.
// A click picks the piece under the pointer (shift adds or removes it), a click on nothing clears.
//
// Why a tool: a plain drag on the map is the map's pan and shift+drag is MapLibre's zoom box, so a
// box needs a mode of its own. While the tool is on the left button draws the box; the right or
// middle button drags the map instead (the wheel and the arrow keys work as always). A finger pans
// as always; a tap adds or takes out a piece, a hold then drag adds a box (T-1260).
// Construct and the quote leave out switched-off pieces; they are drawn grey (renderer.ts).

import type { Map as MlMap } from "maplibre-gl";
import { bpPicked, onlyOneKind, removePicked, switchPicked, unionPicked, type BpPicked } from "../game/blueprintPick";
import { lngLatToPack } from "../game/coords";
import { describeIssues } from "../game/issues";
import { tool } from "../game/ui";
import { touchInput } from "./touch";
import { edit, say, world } from "../game/world";
import type { ClockClient } from "../workers/clockClient";
import { effect } from "@preact/signals";
import { toLocal } from "./geo";
import { strokesFromEdges } from "./network";
import { previewStrokes, setGrabbing } from "./nodeDrag";
import type { Overlay } from "./overlay";
import { topSeg } from "./pick";
import { STATIONS_ZOOM } from "./stationGeometry";

/** A click, not a drag: the pointer moved less than this (px). */
const CLICK_PX = 4;
/** how near a picked node a press grabs the pick (px), as for one node (map/nodeDrag.ts) */
const NODE_PX = 9;
/** T-1260: a finger held this long without moving more than HOLD_SLOP_PX draws a box (ms) */
const HOLD_MS = 450;
const HOLD_SLOP_PX = 8;

type Rect = [number, number, number, number];

/** Whether the segment (ax, ay)-(bx, by) touches the rectangle (Liang-Barsky clipping). */
export function segmentTouches(r: Rect, ax: number, ay: number, bx: number, by: number): boolean {
  let t0 = 0, t1 = 1;
  const dx = bx - ax, dy = by - ay;
  const clip = (p: number, q: number) => {
    if (p === 0) return q >= 0;
    const t = q / p;
    if (p < 0) {
      if (t > t1) return false;
      if (t > t0) t0 = t;
    } else {
      if (t < t0) return false;
      if (t < t1) t1 = t;
    }
    return true;
  };
  return clip(-dx, ax - r[0]) && clip(dx, r[2] - ax) && clip(-dy, ay - r[1]) && clip(dy, r[3] - ay);
}

export class BlueprintBox {
  private box: HTMLDivElement;
  private drag: { x0: number; y0: number; x: number; y: number; add: boolean; base: BpPicked; pan: boolean; moved: boolean } | null = null;
  private frame = 0;
  /** T-475: a press on a picked node or picked track moves every movable node of the pick */
  private move: { x0: number; y0: number; nodes: number[]; at: { x: number; y: number }; dx: number; dy: number; moved: boolean; ok: boolean; issues: string[] } | null = null;

  constructor(private map: MlMap, private overlay: Overlay, private client: ClockClient) {
    this.box = document.createElement("div");
    this.box.className = "bp-box";
    this.box.style.display = "none";
    map.getContainer().appendChild(this.box);
    const host = map.getCanvasContainer();
    host.addEventListener("pointerdown", (e) => this.onDown(e as PointerEvent));
    host.addEventListener("contextmenu", (e) => { if (tool.peek() === "blueprint") e.preventDefault(); });
    addEventListener("pointermove", (e) => this.onMove(e));
    // a finger's box or pick drag over, the map pans again (T-1260)
    addEventListener("pointerup", (e) => void this.onUp(e).finally(() => {
      if (e.pointerType === "touch" && touchInput.peek() && !this.drag && !this.move) this.map.dragPan.enable();
    }));
    // the left button is the box's while the tool is on; the map's own drag would pan under it. A
    // finger pans as everywhere, and draws a box after a hold (T-1260)
    effect(() => {
      if (tool.value === "blueprint" && !touchInput.value) map.dragPan.disable();
      else map.dragPan.enable();
      if (tool.value !== "blueprint") this.cancel();
    });
    host.addEventListener("pointercancel", (e) => this.endTouch(e));
    // the highlight follows the pick and the world (a new network has new buffers)
    effect(() => {
      bpPicked.value;
      world.value;
      this.refreshHighlight();
    });
  }

  get dragging() {
    return (this.drag !== null && !this.drag.pan) || this.move !== null;
  }

  /** The pick's nodes that can move (every edge there blueprint, no crossover, station not
   * constructed), when a press at (x, y) px is on one of them or on picked track; else null. */
  private grabAt(x: number, y: number): number[] | null {
    const w = world.peek(), p = bpPicked.peek();
    if (!w || (!p.edges.length && !p.stations.length)) return null;
    const ids = new Set(p.stations);
    for (const e of w.edges) if (p.edges.includes(e.id)) ids.add(e.a), ids.add(e.b);
    const fixed = new Set<number>();
    for (const e of w.edges) if (e.crossover) fixed.add(e.a), fixed.add(e.b);
    const built = new Set(w.save.stations.filter((s) => s.built).map((s) => s.num));
    const nodes: number[] = [];
    let near = false;
    for (const id of ids) {
      const n = w.nodes.get(id);
      if (!n || n.builtPorts > 0 || fixed.has(id) || built.has(id)) continue;
      nodes.push(id);
      const [sx, sy] = this.overlay.toScreen(...toLocal(n.lng, n.lat));
      if ((sx - x) ** 2 + (sy - y) ** 2 < NODE_PX * NODE_PX) near = true;
    }
    if (!nodes.length) return null;
    if (near) return nodes;
    const under = this.under(x, y).edges[0];
    return under !== undefined && p.edges.includes(under) ? nodes : null;
  }

  /** Esc while dragging the pick: it goes back, nothing changes. */
  private cancelMove(): boolean {
    if (!this.move) return false;
    this.move = null;
    this.client.dropPreview();
    this.endMove();
    return true;
  }

  private endMove() {
    setGrabbing(false);
    this.overlay.renderer.setPreview(null);
    this.overlay.invalidate();
  }

  /** Esc while dragging a box: the box goes away and the pick goes back. */
  cancel(): boolean {
    if (this.cancelMove()) return true;
    const d = this.drag;
    if (!d) return false;
    this.drag = null;
    this.box.style.display = "none";
    if (!d.pan) bpPicked.value = d.base;
    return true;
  }

  /** The picked track as an ink band under the track (renderer `setHighlight`); call after the
   * network buffers were replaced. */
  refreshHighlight() {
    const n = this.overlay.renderer.net;
    const p = bpPicked.peek();
    if (!n || !p.edges.length) {
      this.overlay.renderer.setHighlight(null);
    } else {
      const pts: Float32Array[] = [];
      for (const e of p.edges) {
        const a = n.edgePts.get(e);
        if (a) pts.push(a);
      }
      this.overlay.renderer.setHighlight(strokesFromEdges(pts));
    }
    this.overlay.invalidate();
  }

  private point(e: PointerEvent): [number, number] {
    const r = this.map.getCanvasContainer().getBoundingClientRect();
    return [e.clientX - r.left, e.clientY - r.top];
  }

  /** T-1260, a finger: the press waiting to become a box (held still HOLD_MS) or a tap, and the fingers down */
  private hold: { x0: number; y0: number; timer: number } | null = null;
  private fingers = new Set<number>();

  /** A finger on the map with the tool on: a tap adds or takes out the piece under it, a hold then
   * drag adds what a box touches, a press on the pick drags it, anything else pans or zooms. */
  private touchDown(e: PointerEvent) {
    this.fingers.add(e.pointerId);
    if (this.fingers.size > 1) return void this.dropHold(); // a pinch
    if (this.drag || this.move) return;
    const [x, y] = this.point(e);
    const grab = this.grabAt(x, y);
    if (grab) {
      this.map.dragPan.disable();
      const ll = this.map.unproject([x, y]);
      this.move = { x0: x, y0: y, nodes: grab, at: lngLatToPack(ll.lng, ll.lat), dx: 0, dy: 0, moved: false, ok: true, issues: [] };
      return;
    }
    this.hold = { x0: x, y0: y, timer: setTimeout(() => {
      const h = this.hold;
      if (!h) return;
      this.hold = null;
      this.map.dragPan.disable();
      this.drag = { x0: h.x0, y0: h.y0, x: h.x0, y: h.y0, add: true, base: bpPicked.peek(), pan: false, moved: false };
    }, HOLD_MS) as unknown as number };
  }

  private dropHold() {
    if (this.hold) clearTimeout(this.hold.timer);
    this.hold = null;
  }

  /** A finger lifted or lost: a tap if it was still waiting, and the map pans again. */
  private endTouch(e: PointerEvent): boolean {
    if (e.pointerType !== "touch") return false;
    this.fingers.delete(e.pointerId);
    const h = this.hold;
    if (h && e.type === "pointerup") {
      this.dropHold();
      const [x, y] = this.point(e);
      const hit = this.under(x, y);
      if (hit.edges.length || hit.stations.length) this.toggle(hit);
      return true;
    }
    this.dropHold();
    if (e.type === "pointercancel") this.cancel();
    return false;
  }

  private onDown(e: PointerEvent) {
    if (tool.peek() !== "blueprint") return;
    if (e.pointerType === "touch") return this.touchDown(e);
    if (this.drag || this.move) return;
    const [x, y] = this.point(e);
    const grab = e.button === 0 && !e.shiftKey ? this.grabAt(x, y) : null;
    if (grab) {
      e.preventDefault();
      const ll = this.map.unproject([x, y]);
      this.move = { x0: x, y0: y, nodes: grab, at: lngLatToPack(ll.lng, ll.lat), dx: 0, dy: 0, moved: false, ok: true, issues: [] };
    } else if (e.button === 0) {
      e.preventDefault();
      this.drag = { x0: x, y0: y, x, y, add: e.shiftKey, base: e.shiftKey ? bpPicked.peek() : { edges: [], stations: [] }, pan: false, moved: false };
    } else if (e.button === 1 || e.button === 2) {
      e.preventDefault();
      this.drag = { x0: x, y0: y, x, y, add: false, base: bpPicked.peek(), pan: true, moved: false };
    }
  }

  private onMove(e: PointerEvent) {
    const h = this.hold;
    if (h && e.pointerType === "touch") {
      const [x, y] = this.point(e);
      if (Math.hypot(x - h.x0, y - h.y0) > HOLD_SLOP_PX) this.dropHold(); // a pan
      return;
    }
    const m = this.move;
    if (m) return this.moveTo(m, ...this.point(e));
    const d = this.drag;
    if (!d) {
      // the grab cursor over what a press would drag (T-474)
      if (tool.peek() === "blueprint" && e.buttons === 0 && e.target === this.map.getCanvas()) {
        const c = this.grabAt(...this.point(e)) ? "grab" : "crosshair";
        if (this.map.getCanvas().style.cursor !== c) this.map.getCanvas().style.cursor = c;
      }
      return;
    }
    const [x, y] = this.point(e);
    if (d.pan) {
      this.map.panBy([d.x - x, d.y - y], { animate: false });
      d.x = x;
      d.y = y;
      return;
    }
    d.x = x;
    d.y = y;
    if (!d.moved && Math.hypot(x - d.x0, y - d.y0) < CLICK_PX) return;
    d.moved = true;
    const [l, t, r, b] = [Math.min(d.x0, x), Math.min(d.y0, y), Math.max(d.x0, x), Math.max(d.y0, y)];
    Object.assign(this.box.style, { display: "", left: `${l}px`, top: `${t}px`, width: `${r - l}px`, height: `${b - t}px` });
    // the pick follows the box, at most once a frame
    if (!this.frame)
      this.frame = requestAnimationFrame(() => {
        this.frame = 0;
        if (this.drag && !this.drag.pan && this.drag.moved) bpPicked.value = unionPicked(this.drag.base, this.within([l, t, r, b]));
      });
  }

  private moveTo(m: NonNullable<BlueprintBox["move"]>, x: number, y: number) {
    if (!m.moved && Math.hypot(x - m.x0, y - m.y0) < CLICK_PX) return;
    if (!m.moved) setGrabbing(true);
    m.moved = true;
    const ll = this.map.unproject([x, y]);
    const p = lngLatToPack(ll.lng, ll.lat);
    m.dx = p.x - m.at.x;
    m.dy = p.y - m.at.y;
    this.client.previewNodes(m.nodes, m.dx, m.dy, (a) => {
      if (this.move !== m) return;
      m.ok = a.ok;
      m.issues = a.issues;
      this.overlay.renderer.setPreview(previewStrokes(a));
      this.overlay.invalidate();
    });
  }

  private async onUp(e: PointerEvent) {
    if (this.endTouch(e)) return;
    const m = this.move;
    if (m) {
      this.move = null;
      if (!m.moved) {
        // a press without a drag is a click, as without the pick under it (a finger's adds or takes out)
        const add = e.pointerType === "touch";
        this.drag = { x0: m.x0, y0: m.y0, x: m.x0, y: m.y0, add, base: add ? bpPicked.peek() : { edges: [], stations: [] }, pan: false, moved: false };
      } else {
        this.client.dropPreview();
        const a = await edit({ op: "moveNodes", nodes: m.nodes, dx: m.dx, dy: m.dy });
        if (!a.ok && !a.issues.length) say(m.issues.length ? describeIssues(m.issues) : "This cannot be moved there.", true);
        this.endMove();
        return;
      }
    }
    const d = this.drag;
    if (!d) return;
    this.drag = null;
    this.box.style.display = "none";
    if (d.pan) return;
    if (this.frame) cancelAnimationFrame(this.frame);
    this.frame = 0;
    const [x, y] = this.point(e);
    if (d.moved) {
      bpPicked.value = unionPicked(d.base, this.within([Math.min(d.x0, x), Math.min(d.y0, y), Math.max(d.x0, x), Math.max(d.y0, y)]));
      return;
    }
    // a click: the piece under the pointer
    const hit = this.under(x, y);
    if (!hit.edges.length && !hit.stations.length) {
      if (!d.add) bpPicked.value = { edges: [], stations: [] };
      return;
    }
    if (!d.add) {
      bpPicked.value = hit;
      return;
    }
    this.toggle(hit);
  }

  /** The piece hit added to the pick, or taken out if it is in it. */
  private toggle(hit: BpPicked) {
    const cur = bpPicked.peek();
    const has = hit.edges.length ? cur.edges.includes(hit.edges[0]) : cur.stations.includes(hit.stations[0]);
    bpPicked.value = has
      ? { edges: cur.edges.filter((v) => !hit.edges.includes(v)), stations: cur.stations.filter((v) => !hit.stations.includes(v)), built: cur.built }
      : unionPicked(cur, hit);
  }

  /** The track a box touches and the stations (drawn, so zoomed in enough) inside it: the
   * blueprint ones if there are any, else the constructed ones (T-829). */
  private within(r: Rect): BpPicked {
    const n = this.overlay.renderer.net, w = world.peek();
    const out: BpPicked = { edges: [], stations: [] };
    if (!n || !w) return out;
    const p = new Float64Array(4);
    for (const [id, pts] of n.edgePts) {
      for (let k = 0; k + 5 < pts.length; k += 3) {
        this.overlay.toScreenInto(pts[k], pts[k + 1], p, 0);
        this.overlay.toScreenInto(pts[k + 3], pts[k + 4], p, 2);
        if (segmentTouches(r, p[0], p[1], p[2], p[3])) {
          out.edges.push(id);
          break;
        }
      }
    }
    if (this.map.getZoom() >= STATIONS_ZOOM) {
      for (let i = 0; i < n.stationCount; i++) {
        this.overlay.toScreenInto(n.stations[i * 3], n.stations[i * 3 + 1], p, 0);
        if (p[0] >= r[0] && p[0] <= r[2] && p[1] >= r[1] && p[1] <= r[3]) out.stations.push(Number(n.stationIds[i]));
      }
    }
    return onlyOneKind(out);
  }

  /** The station or track piece under a point (px), a station first, blueprint before constructed. */
  private under(x: number, y: number): BpPicked {
    const n = this.overlay.renderer.net, w = world.peek();
    const none: BpPicked = { edges: [], stations: [] };
    if (!n || !w) return none;
    const hits: BpPicked[] = [];
    if (this.map.getZoom() >= STATIONS_ZOOM) {
      const i = this.overlay.renderer.stationAt(x, y, (a, b) => this.overlay.toScreen(a, b), 7);
      if (i >= 0) hits.push({ edges: [], stations: [Number(n.stationIds[i])] });
    }
    const [ti] = topSeg(this.overlay, n.track.seg, n.track.count, x, y, 7 * 7 + 1);
    if (ti >= 0) hits.push({ edges: [n.track.edge[ti]], stations: [] });
    const bp = hits.map(onlyOneKind);
    return bp.find((h) => !h.built && (h.edges.length || h.stations.length)) ?? bp[0] ?? none;
  }

  /** The D and E keys: switch the pick off or on; Delete removes it (T-829). True if used. */
  key(e: KeyboardEvent): boolean {
    if (tool.peek() !== "blueprint" || e.ctrlKey || e.metaKey || e.altKey) return false;
    if (e.key === "Delete") {
      void removePicked();
      return true;
    }
    const k = e.key.toLowerCase();
    if (k !== "d" && k !== "e") return false;
    void switchPicked(k === "d");
    return true;
  }
}
