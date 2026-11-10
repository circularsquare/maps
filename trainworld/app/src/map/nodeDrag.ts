// Dragging blueprint nodes (T-093): with the select or the track tool, press on a track end, a junction or a
// station and drag it. Only a blueprint node moves: every edge there is blueprint and its station,
// if any, is not constructed. The blueprint track meeting there follows (the track model's
// `move_node`: its corners stay, a junction or station keeps its heading). While dragging, the
// clock worker previews every edge there as a trial edit, with each curve's radius and speed or
// why it cannot be built; letting go applies it as one undoable edit, Escape cancels. A press
// that does not move is an ordinary click (selection, map/pick.ts; with the track tool, track is
// placed or branched from the node as usual).
//
// T-1869: either end of a blueprint station's platform (map/dragHandles.ts) moves the station as
// its node does, the end held under the pointer.
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
import { MARKS_ZOOM, NODE_HIT_PX, stationEnds } from "./dragHandles";
import { drawTangent, endHeading } from "./curveCircle";
import { toLngLat, toLocal } from "./geo";
import { live } from "./live";
import { previewStrokes } from "./network";
import type { Overlay } from "./overlay";
import { marksLayer, placeAt } from "./screenPlace";
import { curveLabels, CurveLabelPool } from "./trackTagRows";

/** a press becomes a drag once the pointer has moved this far, px (MapLibre's click tolerance) */
const DRAG_PX = 3;

interface Hit {
  node: number;
  movable: boolean;
  /** a platform's end (T-1869), px: the node keeps its offset from it */
  end?: { x: number; y: number };
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
  private tangent: SVGSVGElement;
  private labels: CurveLabelPool;
  private press: (Hit & { px: number; py: number; told?: boolean }) | null = null;
  /** `x, y`: where the node goes, pack metres (node + `ox, oy` from the pointer); `lx, ly`: the pointer, local */
  private drag: { node: number; x: number; y: number; lx: number; ly: number; ox: number; oy: number } | null = null;
  private preview: PreviewAnswer | null = null;

  constructor(private map: MlMap, private overlay: Overlay, private client: ClockClient) {
    this.box = marksLayer(map);
    this.labels = new CurveLabelPool(this.box);
    this.ring = document.createElement("div");
    this.ring.className = "snap-ring";
    this.tag = document.createElement("div");
    this.tag.className = "tool-tag bad";
    this.box.append(this.ring, this.tag);
    const NS = "http://www.w3.org/2000/svg";
    this.tangent = document.createElementNS(NS, "svg");
    this.tangent.setAttribute("width", "1");
    this.tangent.setAttribute("height", "1");
    this.tangent.appendChild(document.createElementNS(NS, "line"));
    this.tangent.style.display = "none";
    this.box.insertBefore(this.tangent, this.box.firstChild);
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
    // T-1869: a platform's end
    best = NODE_HIT_PX * NODE_HIT_PX;
    for (const e of stationEnds(this.overlay)) {
      const d = (e.x - px) ** 2 + (e.y - py) ** 2;
      if (d < best) [best, hit] = [d, { node: e.node, movable: true, end: { x: e.x, y: e.y } }];
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
  }

  private packAt(px: number, py: number): { x: number; y: number } {
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
      this.drag = { node: p.node, x: 0, y: 0, lx: 0, ly: 0, ox, oy };
      setGrabbing(true);
      live.edgeEditor?.place(); // its squares hide while a node moves
      this.map.triggerRepaint(); // and map/endMarkers.ts' marks, placed on map frames, which a paused game has none of
    }
    const ll = this.map.unproject([px, py]);
    const at = lngLatToPack(ll.lng, ll.lat);
    const [lx, ly] = toLocal(ll.lng, ll.lat);
    const d = this.drag!;
    Object.assign(d, { x: at.x + d.ox, y: at.y + d.oy, lx, ly });
    this.client.previewNode(d.node, d.x, d.y, (a) => {
      if (this.drag !== d) return;
      this.preview = a;
      this.overlay.renderer.setPreview(previewStrokes(a));
      this.overlay.invalidate();
      this.place();
    });
    this.place();
  }

  private async onUp() {
    const d = this.drag;
    this.press = null;
    this.drag = null;
    this.map.dragPan.enable();
    if (!d) return;
    this.client.dropPreview();
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
    this.placeTangent(d, pv);
    this.labels.place(pv ? curveLabels(pv.verts) : [], (x, y) => this.overlay.toScreen(x, y), this.overlay.cssW, this.overlay.cssH);
  }

  /** T-2765: dragging a loose end (the only track at the node), the line on from where the track arrives
   * at it, as the track tool draws while placing (map/curveCircle.ts). */
  private placeTangent(d: NodeDrag["drag"], pv: PreviewAnswer | null) {
    let h = null;
    const w = world.peek();
    if (d && pv && w) {
      const es = w.edges.filter((e) => e.a === d.node || e.b === d.node);
      if (es.length === 1 && pv.verts.length >= 8 && pv.pts.length >= 6) {
        if (es[0].b === d.node) h = endHeading(pv.verts, pv.pts, pv.arc, toLngLat, toLocal);
        else {
          // the node is the edge's first end: read the preview from the other way round
          const v: number[] = [], p: number[] = [];
          for (let i = pv.verts.length - 4; i >= 0; i -= 4) v.push(pv.verts[i], pv.verts[i + 1], pv.verts[i + 2], pv.verts[i + 3]);
          for (let i = pv.pts.length - 3; i >= 0; i -= 3) p.push(pv.pts[i], pv.pts[i + 1], pv.pts[i + 2]);
          h = endHeading(v, p, undefined, toLngLat, toLocal);
        }
      }
    }
    drawTangent(this.tangent, h, !!pv?.ok, (x, y) => this.overlay.toScreen(x, y));
  }
}
