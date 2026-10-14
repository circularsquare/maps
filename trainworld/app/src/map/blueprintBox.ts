// The blueprint tool (T-333): a drag draws a box on the map, and the blueprint track it touches and
// the blueprint stations inside it are picked (game/blueprintPick.ts). Shift adds to what is picked.
// A click picks the piece under the pointer (shift adds or removes it), a click on nothing clears.
//
// Why a tool: a plain drag on the map is the map's pan and shift+drag is MapLibre's zoom box, so a
// box needs a mode of its own. While the tool is on the left button draws the box; the right or
// middle button drags the map instead (the wheel and the arrow keys work as always).
// Construct and the quote leave out switched-off pieces; they are drawn grey (renderer.ts).

import type { Map as MlMap } from "maplibre-gl";
import { bpPicked, switchPicked, unionPicked, type BpPicked } from "../game/blueprintPick";
import { tool } from "../game/ui";
import { world } from "../game/world";
import { effect } from "@preact/signals";
import { strokesFromEdges } from "./network";
import type { Overlay } from "./overlay";
import { topSeg } from "./pick";
import { STATIONS_ZOOM } from "./stationGeometry";

/** A click, not a drag: the pointer moved less than this (px). */
const CLICK_PX = 4;

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

  constructor(private map: MlMap, private overlay: Overlay) {
    this.box = document.createElement("div");
    this.box.className = "bp-box";
    this.box.style.display = "none";
    map.getContainer().appendChild(this.box);
    const host = map.getCanvasContainer();
    host.addEventListener("pointerdown", (e) => this.onDown(e as PointerEvent));
    host.addEventListener("contextmenu", (e) => { if (tool.peek() === "blueprint") e.preventDefault(); });
    addEventListener("pointermove", (e) => this.onMove(e));
    addEventListener("pointerup", (e) => this.onUp(e));
    // the left button is the box's while the tool is on; the map's own drag would pan under it
    effect(() => {
      if (tool.value === "blueprint") map.dragPan.disable();
      else {
        map.dragPan.enable();
        this.cancel();
      }
    });
    // the highlight follows the pick and the world (a new network has new buffers)
    effect(() => {
      bpPicked.value;
      world.value;
      this.refreshHighlight();
    });
  }

  get dragging() {
    return this.drag !== null && !this.drag.pan;
  }

  /** Esc while dragging a box: the box goes away and the pick goes back. */
  cancel(): boolean {
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

  private onDown(e: PointerEvent) {
    if (tool.peek() !== "blueprint" || this.drag) return;
    const [x, y] = this.point(e);
    if (e.button === 0) {
      e.preventDefault();
      this.drag = { x0: x, y0: y, x, y, add: e.shiftKey, base: e.shiftKey ? bpPicked.peek() : { edges: [], stations: [] }, pan: false, moved: false };
    } else if (e.button === 1 || e.button === 2) {
      e.preventDefault();
      this.drag = { x0: x, y0: y, x, y, add: false, base: bpPicked.peek(), pan: true, moved: false };
    }
  }

  private onMove(e: PointerEvent) {
    const d = this.drag;
    if (!d) return;
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

  private onUp(e: PointerEvent) {
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
    const cur = bpPicked.peek();
    if (!hit.edges.length && !hit.stations.length) {
      if (!d.add) bpPicked.value = { edges: [], stations: [] };
      return;
    }
    if (!d.add) {
      bpPicked.value = hit;
      return;
    }
    const has = hit.edges.length ? cur.edges.includes(hit.edges[0]) : cur.stations.includes(hit.stations[0]);
    bpPicked.value = has
      ? { edges: cur.edges.filter((v) => !hit.edges.includes(v)), stations: cur.stations.filter((v) => !hit.stations.includes(v)) }
      : unionPicked(cur, hit);
  }

  /** Blueprint track a box touches and blueprint stations (drawn, so zoomed in enough) inside it. */
  private within(r: Rect): BpPicked {
    const n = this.overlay.renderer.net, w = world.peek();
    const out: BpPicked = { edges: [], stations: [] };
    if (!n || !w) return out;
    const blueprint = new Set(w.edges.filter((e) => !e.built).map((e) => e.id));
    const p = new Float64Array(4);
    for (const [id, pts] of n.edgePts) {
      if (!blueprint.has(id)) continue;
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
      const unbuilt = new Set(w.save.stations.filter((s) => !s.built).map((s) => String(s.num)));
      for (let i = 0; i < n.stationCount; i++) {
        if (!unbuilt.has(n.stationIds[i])) continue;
        this.overlay.toScreenInto(n.stations[i * 3], n.stations[i * 3 + 1], p, 0);
        if (p[0] >= r[0] && p[0] <= r[2] && p[1] >= r[1] && p[1] <= r[3]) out.stations.push(Number(n.stationIds[i]));
      }
    }
    return out;
  }

  /** The blueprint station or track piece under a point (px), a station first. */
  private under(x: number, y: number): BpPicked {
    const n = this.overlay.renderer.net, w = world.peek();
    const none: BpPicked = { edges: [], stations: [] };
    if (!n || !w) return none;
    if (this.map.getZoom() >= STATIONS_ZOOM) {
      const i = this.overlay.renderer.stationAt(x, y, (a, b) => this.overlay.toScreen(a, b), 7);
      if (i >= 0) {
        const num = Number(n.stationIds[i]);
        if (w.save.stations.some((s) => s.num === num && !s.built)) return { edges: [], stations: [num] };
      }
    }
    const [ti] = topSeg(this.overlay, n.track.seg, n.track.count, x, y, 7 * 7 + 1);
    if (ti >= 0) {
      const id = n.track.edge[ti];
      if (w.edges.some((e) => e.id === id && !e.built)) return { edges: [id], stations: [] };
    }
    return none;
  }

  /** The D and E keys: switch the pick off or on. True if used. */
  key(e: KeyboardEvent): boolean {
    if (tool.peek() !== "blueprint" || e.ctrlKey || e.metaKey || e.altKey) return false;
    const k = e.key.toLowerCase();
    if (k !== "d" && k !== "e") return false;
    void switchPicked(k === "d");
    return true;
  }
}
