// Dragging blueprint nodes (T-093): with the select or the track tool, press on a track end, a junction or a
// station and drag it. Only a blueprint node moves: every edge there is blueprint and its station,
// if any, is not constructed. The blueprint track meeting there follows (the track model's
// `move_node`: its corners stay, a junction or station keeps its heading). While dragging, the
// clock worker previews every edge there as a trial edit, with each curve's radius and speed or
// why it cannot be built; letting go applies it as one undoable edit, Escape cancels. A press
// that does not move is an ordinary click (selection, map/pick.ts; with the track tool, track is
// placed or branched from the node as usual).
//
// T-1869: two more handles drag the same way (map/dragHandles.ts). Either end of a blueprint
// station's platform moves the station as its node does, the end held under the pointer. The
// middle of a blueprint curve moves the curve's PI so that the curve runs through the pointer (its
// radius rule unchanged, SPEC 6.1): the PI goes where the last preview's curve middle missed the
// pointer by, scaled by how far the middle moved for the last PI step, and steps again until it
// is within MID_TOL_M, so the curve settles under a still pointer.
//
// The press is caught on the map container before MapLibre sees it, so the map does not pan.
// Constructed nodes are left to the map (a drag over a station pans as usual), except the one
// selected, where a drag says why it stays put.

import { t } from "../i18n";
import type { Map as MlMap } from "maplibre-gl";
import { lngLatToPack } from "../game/coords";
import { describeIssues } from "../game/issues";
import { designSpeed, selection, tool } from "../game/ui";
import { edit, say, world } from "../game/world";
import type { ClockClient, PreviewAnswer } from "../workers/clockClient";
import { queryClock } from "../workers/clockClient";
import type { PiInput } from "../workers/protocol";
import type { Level } from "../game/types";
import { MARKS_ZOOM, NODE_HIT_PX, stationEnds, visibleCurves, type CurveHandle } from "./dragHandles";
import { toLngLat, toLocal } from "./geo";
import { live } from "./live";
import { previewStrokes, strokesFromPreview } from "./network";
import type { Overlay } from "./overlay";
import { marksLayer, placeAt } from "./screenPlace";
import { curveLabels, CurveLabelPool } from "./trackTagRows";

/** a press becomes a drag once the pointer has moved this far, px (MapLibre's click tolerance) */
const DRAG_PX = 3;
/** a curve's middle this near the pointer has settled, m */
const MID_TOL_M = 0.3;
/** at most this many PI steps after the pointer last moved */
const MAX_STEPS = 10;

interface Hit {
  /** the node moved; -1 for a curve */
  node: number;
  movable: boolean;
  /** a platform's end (T-1869), px: the node keeps its offset from it */
  end?: { x: number; y: number };
  /** a curve's middle (T-1869) */
  curve?: CurveHandle;
}

type P = { x: number; y: number };

/** A curve being bent (T-1869): its edge's PIs, and the PI and curve middle each preview had. */
interface Bend {
  edge: number;
  i: number;
  tracks: number;
  hsr: boolean;
  pis: PiInput[] | null;
  /** the pointer, pack metres */
  q: P;
  /** the PI and its curve's middle: at the press, then as the last two previews had them */
  last: { p: P; m: P } | null;
  prev: { p: P; m: P } | null;
  /** how far the PI moves for a metre of the middle's */
  gain: number;
  /** the PI of the preview shown */
  shown: P | null;
  steps: number;
}

/** The grabbing cursor everywhere while something is dragged (T-474), so it holds off the handle. */
export function setGrabbing(on: boolean) {
  document.documentElement.classList.toggle("grabbing", on);
}

/** A drag over: the grabbing cursor and the preview strokes go. */
export function endDrag(overlay: Overlay) {
  setGrabbing(false);
  overlay.renderer.setPreview(null);
  overlay.invalidate();
}

export class NodeDrag {
  private box: HTMLDivElement;
  private ring: HTMLDivElement;
  private tag: HTMLDivElement;
  private labels: CurveLabelPool;
  private press: (Hit & { px: number; py: number; told?: boolean; pis?: Promise<Float64Array | null> }) | null = null;
  /** `x, y`: where the node goes, pack metres (node + `ox, oy` from the pointer); `lx, ly`: the pointer, local */
  private drag: { node: number; x: number; y: number; lx: number; ly: number; ox: number; oy: number; bend?: Bend } | null = null;
  private preview: PreviewAnswer | null = null;

  constructor(private map: MlMap, private overlay: Overlay, private client: ClockClient) {
    this.box = marksLayer(map);
    this.labels = new CurveLabelPool(this.box);
    this.ring = document.createElement("div");
    this.ring.className = "snap-ring";
    this.tag = document.createElement("div");
    this.tag.className = "tool-tag bad";
    this.box.append(this.ring, this.tag);
    overlay.afterMapFrame.push(() => this.place());
    // capture: before MapLibre's own handlers on the canvas
    map.getContainer().addEventListener("mousedown", (e) => this.onDown(e), true);
    addEventListener("mousemove", (e) => this.onMove(e));
    addEventListener("mouseup", () => void this.onUp());
    this.place();
  }

  get dragging() {
    return this.drag !== null;
  }

  /** The node under a screen point (px from the map's top left), if any. */
  hit(px: number, py: number): Hit | null {
    const w = world.peek();
    const t = tool.peek();
    if (!w || (t !== "select" && t !== "track")) return null;
    let best = NODE_HIT_PX * NODE_HIT_PX, hit: Hit | null = null;
    for (const n of w.nodes.values()) {
      if (n.bus || n.port) continue; // a bus stop or a port (T-2194) is no node of the track (T-1464)
      const [x, y] = this.overlay.toScreen(...toLocal(n.lng, n.lat));
      const d = (x - px) ** 2 + (y - py) ** 2;
      if (d < best) {
        best = d;
        const st = n.platform > 0 ? w.save.stations.find((s) => s.num === n.id) : undefined;
        hit = { node: n.id, movable: n.builtPorts === 0 && !st?.built };
      }
    }
    if (hit || this.map.getZoom() < MARKS_ZOOM) return hit;
    // T-1869: a platform's end, then a curve's middle
    best = NODE_HIT_PX * NODE_HIT_PX;
    for (const e of stationEnds(this.overlay)) {
      const d = (e.x - px) ** 2 + (e.y - py) ** 2;
      if (d < best) [best, hit] = [d, { node: e.node, movable: true, end: { x: e.x, y: e.y } }];
    }
    if (hit) return hit;
    for (const c of visibleCurves(this.overlay)) {
      const d = (c.x - px) ** 2 + (c.y - py) ** 2;
      if (d < best) [best, hit] = [d, { node: -1, movable: true, curve: c }];
    }
    return hit;
  }

  /** The node is what is selected (a station, a junction, or an end of the selected track). */
  private selected(node: number): boolean {
    const s = selection.peek();
    if (s?.kind === "station") return s.station === String(node);
    if (s?.kind === "junction") return s.node === node;
    if (s?.kind === "track") {
      const e = world.peek()?.edges.find((x) => x.id === s.edge);
      return !!e && (e.a === node || e.b === node);
    }
    return false;
  }

  private onDown(e: MouseEvent) {
    if (e.button !== 0 || e.target !== this.map.getCanvas() || e.shiftKey || e.ctrlKey || e.metaKey || e.altKey) return;
    const r = this.map.getContainer().getBoundingClientRect();
    const px = e.clientX - r.left, py = e.clientY - r.top;
    const h = this.hit(px, py);
    if (!h || (!h.movable && !this.selected(h.node))) return;
    // ours, over a demand bubble too (T-474): the map must not pan; a press without a drag still reaches MapLibre as a click
    this.map.dragPan.disable();
    this.press = { ...h, px, py };
    // the curve's PIs are read while the pointer has yet to move
    if (h.curve) this.press.pis = queryClock({ what: "edgePis", edge: h.curve.edge });
  }

  private packAt(px: number, py: number): P {
    const ll = this.map.unproject([px, py]);
    return lngLatToPack(ll.lng, ll.lat);
  }

  private onMove(e: MouseEvent) {
    const p = this.press;
    if (!p) return;
    const r = this.map.getContainer().getBoundingClientRect();
    const px = e.clientX - r.left, py = e.clientY - r.top;
    if (!this.drag) {
      if (p.told || Math.hypot(px - p.px, py - p.py) < DRAG_PX) return;
      if (!p.movable) {
        // the press stays ours until let go (no pan), said once
        say(t("nodeDrag.builtFixed"), true);
        p.told = true;
        return;
      }
      let ox = 0, oy = 0;
      if (p.end) {
        const n = world.peek()?.nodes.get(p.node), at = this.packAt(p.end.x, p.end.y);
        if (n) [ox, oy] = [n.x - at.x, n.y - at.y];
      }
      this.drag = { node: p.node, x: 0, y: 0, lx: 0, ly: 0, ox, oy, bend: p.curve ? this.startBend(p.curve, p.pis!) : undefined };
      setGrabbing(true);
      live.edgeEditor?.place(); // its squares hide while a node moves
      this.map.triggerRepaint(); // and map/endMarkers.ts' marks, placed on map frames, which a paused game has none of
    }
    const ll = this.map.unproject([px, py]);
    const at = lngLatToPack(ll.lng, ll.lat);
    const [lx, ly] = toLocal(ll.lng, ll.lat);
    const d = this.drag!;
    Object.assign(d, { x: at.x + d.ox, y: at.y + d.oy, lx, ly });
    if (d.bend) {
      d.bend.q = at;
      d.bend.steps = 0;
      this.step(d, d.bend);
    } else
      this.client.previewNode(d.node, d.x, d.y, (a) => {
        if (this.drag !== d) return;
        this.preview = a;
        this.overlay.renderer.setPreview(previewStrokes(a));
        this.overlay.invalidate();
        this.place();
      });
    this.place();
  }

  private startBend(c: CurveHandle, pis: Promise<Float64Array | null>): Bend {
    const e = world.peek()?.edges.find((x) => x.id === c.edge);
    const b: Bend = { edge: c.edge, i: c.i, tracks: e?.tracks ?? 2, hsr: !!e?.hsr, pis: null, q: { x: 0, y: 0 }, last: null, prev: null, gain: 1.5, shown: null, steps: 0 };
    void pis.then((data) => {
      const d = this.drag;
      if (d?.bend !== b || !data) return;
      const k = 1 + c.i * 8;
      if (k + 8 > data.length) return;
      const list: PiInput[] = [];
      for (let j = 1; j + 8 <= data.length; j += 8) list.push({ x: data[j], y: data[j + 1], radius: data[j + 2], level: data[j + 3] as Level });
      b.pis = list;
      b.last = { p: { x: data[k], y: data[k + 1] }, m: { x: data[k + 6], y: data[k + 7] } };
      this.step(d, b);
    });
    return b;
  }

  /** The next PI for a bent curve: where the last preview's curve middle missed the pointer by,
   * scaled by the gain; previewed, and stepped again from its answer until it settles. */
  private step(d: NonNullable<NodeDrag["drag"]>, b: Bend) {
    const ref = b.last;
    if (!b.pis || !ref) return;
    const p = { x: ref.p.x + b.gain * (b.q.x - ref.m.x), y: ref.p.y + b.gain * (b.q.y - ref.m.y) };
    const pis = this.bentPis(b, p);
    this.client.previewEdge(b.edge, pis, (a) => {
      if (this.drag !== d) return;
      this.preview = a;
      b.shown = p;
      const v = (b.i + 1) * 4;
      if (v + 2 <= a.verts.length) {
        const m = lngLatToPack(...toLngLat(a.verts[v], a.verts[v + 1]));
        b.prev = b.last;
        b.last = { p, m };
        const pv = b.prev!, dp = Math.hypot(p.x - pv.p.x, p.y - pv.p.y), dm = Math.hypot(m.x - pv.m.x, m.y - pv.m.y);
        // a curve between two set ends moves about half as far as its PI; one with long legs as far
        if (dp > 0.5) b.gain = Math.min(3, Math.max(1, dm > 1e-6 ? dp / dm : 3));
        if (Math.hypot(b.q.x - m.x, b.q.y - m.y) > MID_TOL_M && b.steps++ < MAX_STEPS) this.step(d, b);
      }
      this.overlay.renderer.setPreview(a.pts.length >= 6 ? strokesFromPreview(a.pts, !a.ok, undefined, false, a.marks, b.tracks === 1 ? 0 : 255, b.hsr) : null);
      this.overlay.invalidate();
      this.place();
    });
  }

  private bentPis(b: Bend, p: P): PiInput[] {
    return b.pis!.map((q, k) => (k === b.i ? { ...q, x: p.x, y: p.y } : q));
  }

  private async onUp() {
    const d = this.drag;
    this.press = null;
    this.drag = null;
    this.map.dragPan.enable();
    if (!d) return;
    this.client.dropPreview();
    const b = d.bend;
    if (b) {
      // the shape last shown is the one made
      if (b.pis && b.shown) await edit({ op: "edgePis", edge: b.edge, pis: this.bentPis(b, b.shown), design: designSpeed.peek() });
      this.clear();
      return;
    }
    const a = await edit({ op: "moveNode", node: d.node, x: d.x, y: d.y, design: designSpeed.peek() });
    if (!a.ok && !a.issues.length) say(t("nodeDrag.cannotMoveThis"), true);
    this.clear();
    if (a.ok) live.tools?.nodeMoved(d.node);
  }

  /** Escape: drop the drag, nothing changes. */
  cancel(): boolean {
    if (!this.drag && !this.press) return false;
    const was = this.drag !== null;
    this.drag = null;
    this.press = null;
    this.map.dragPan.enable();
    this.client.dropPreview();
    this.clear();
    return was;
  }

  private clear() {
    endDrag(this.overlay);
    this.preview = null;
    this.place();
    live.edgeEditor?.place();
    this.map.triggerRepaint();
  }

  /** The ring at the point held, the curve labels and the reason it cannot be built. */
  place() {
    const d = this.drag;
    const pv = d ? this.preview : null;
    if (d) {
      const [x, y] = this.overlay.toScreen(d.lx, d.ly);
      this.ring.style.display = "";
      placeAt(this.ring, x, y);
      if (pv && !pv.ok) {
        this.tag.style.display = "";
        this.tag.textContent = describeIssues(pv.issues) || t("nodeDrag.cannotMove");
        placeAt(this.tag, x, y, 14, -30);
      } else this.tag.style.display = "none";
    } else {
      this.ring.style.display = "none";
      this.tag.style.display = "none";
    }
    this.labels.place(pv ? curveLabels(pv.verts) : [], (x, y) => this.overlay.toScreen(x, y), this.overlay.cssW, this.overlay.cssH);
  }
}
