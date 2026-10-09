// Dragging blueprint nodes (T-093): with the select tool, press on a track end, a junction or a
// station and drag it. Only a blueprint node moves: every edge there is blueprint and its station,
// if any, is not constructed. The blueprint track meeting there follows (the track model's
// `move_node`: its corners stay, a junction or station keeps its heading). While dragging, the
// clock worker previews every edge there as a trial edit, with each curve's radius and speed or
// why it cannot be built; letting go applies it as one undoable edit, Escape cancels. A press
// that does not move is an ordinary click (selection, map/pick.ts).
//
// The press is caught on the map container before MapLibre sees it, so the map does not pan.
// Constructed nodes are left to the map (a drag over a station pans as usual), except the one
// selected, where a drag says why it stays put.

import type { Map as MlMap } from "maplibre-gl";
import { lngLatToPack } from "../game/coords";
import { describeIssues } from "../game/issues";
import { selection, tool } from "../game/ui";
import { edit, say, world } from "../game/world";
import type { ClockClient, PreviewAnswer } from "../workers/clockClient";
import { toLocal } from "./geo";
import { live } from "./live";
import { strokesFromPreview, type Strokes } from "./network";
import type { Overlay } from "./overlay";
import { curveLabels } from "./tools";

const HIT_PX = 9;
/** a press becomes a drag once the pointer has moved this far, px (MapLibre's click tolerance) */
const DRAG_PX = 3;

interface Hit {
  node: number;
  movable: boolean;
}

export class NodeDrag {
  private box: HTMLDivElement;
  private ring: HTMLDivElement;
  private tag: HTMLDivElement;
  private labels: HTMLDivElement[] = [];
  private press: (Hit & { px: number; py: number; told?: boolean }) | null = null;
  private drag: { node: number; x: number; y: number; lx: number; ly: number } | null = null;
  private preview: PreviewAnswer | null = null;

  constructor(private map: MlMap, private overlay: Overlay, private client: ClockClient) {
    this.box = document.createElement("div");
    this.box.className = "tool-marks";
    map.getContainer().appendChild(this.box);
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
    if (!w || tool.peek() !== "select") return null;
    let best = HIT_PX * HIT_PX, hit: Hit | null = null;
    for (const n of w.nodes.values()) {
      const [x, y] = this.overlay.toScreen(...toLocal(n.lng, n.lat));
      const d = (x - px) ** 2 + (y - py) ** 2;
      if (d < best) {
        best = d;
        const st = n.platform > 0 ? w.save.stations.find((s) => s.num === n.id) : undefined;
        hit = { node: n.id, movable: n.builtPorts === 0 && !st?.built };
      }
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
    // ours: the map must not pan; a press without a drag still reaches MapLibre as a click
    this.map.dragPan.disable();
    this.press = { ...h, px, py };
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
        say("Constructed track and stations cannot be moved.", true);
        p.told = true;
        return;
      }
      this.drag = { node: p.node, x: 0, y: 0, lx: 0, ly: 0 };
      live.edgeEditor?.place(); // its squares hide while a node moves
    }
    const ll = this.map.unproject([px, py]);
    const at = lngLatToPack(ll.lng, ll.lat);
    const [lx, ly] = toLocal(ll.lng, ll.lat);
    const d = this.drag!;
    Object.assign(d, { x: at.x, y: at.y, lx, ly });
    this.client.previewNode(d.node, at.x, at.y, (a) => {
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
    const a = await edit({ op: "moveNode", node: d.node, x: d.x, y: d.y });
    if (!a.ok && !a.issues.length) say("This cannot be moved there.", true);
    this.clear();
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
    this.preview = null;
    this.overlay.renderer.setPreview(null);
    this.overlay.invalidate();
    this.place();
    live.edgeEditor?.place();
  }

  /** The ring at the node's new place, the curve labels and the reason it cannot be built. */
  place() {
    const d = this.drag;
    const pv = d ? this.preview : null;
    if (d) {
      const [x, y] = this.overlay.toScreen(d.lx, d.ly);
      this.ring.style.display = "";
      this.ring.style.transform = `translate(${Math.round(x)}px, ${Math.round(y)}px)`;
      if (pv && !pv.ok) {
        this.tag.style.display = "";
        this.tag.textContent = describeIssues(pv.issues) || "Cannot be built like this.";
        this.tag.style.transform = `translate(${Math.round(x + 14)}px, ${Math.round(y - 30)}px)`;
      } else this.tag.style.display = "none";
    } else {
      this.ring.style.display = "none";
      this.tag.style.display = "none";
    }
    const labels = pv ? curveLabels(pv.verts) : [];
    labels.forEach((c, i) => {
      let l = this.labels[i];
      if (!l) {
        l = this.labels[i] = document.createElement("div");
        l.className = "pi-label";
        this.box.appendChild(l);
      }
      const [x, y] = this.overlay.toScreen(c.x, c.y);
      l.style.display = "";
      l.textContent = c.text;
      l.style.transform = `translate(${Math.round(x + 8)}px, ${Math.round(y + 6)}px)`;
    });
    for (let i = labels.length; i < this.labels.length; i++) this.labels[i].style.display = "none";
  }
}

/** One stroke set from a preview of several alignments (`parts`), red when it cannot be built. */
function previewStrokes(a: PreviewAnswer): Strokes | null {
  const parts = a.parts ?? new Uint32Array([0, a.pts.length / 3]);
  const all: Strokes[] = [];
  for (let k = 0; k + 1 < parts.length; k++) {
    const pts = a.pts.subarray(parts[k] * 3, parts[k + 1] * 3);
    if (pts.length >= 6) all.push(strokesFromPreview(pts, !a.ok));
  }
  if (!all.length) return null;
  if (all.length === 1) return all[0];
  const n = all.reduce((s, x) => s + x.count, 0);
  const out: Strokes = {
    seg: new Float32Array(n * 4), colour: new Uint32Array(n), edge: new Uint32Array(n), level: new Float32Array(n),
    flags: new Float32Array(n), dist: new Float32Array(n), offset: new Float32Array(n), count: n,
  };
  let o = 0;
  for (const s of all) {
    out.seg.set(s.seg.subarray(0, s.count * 4), o * 4);
    out.colour.set(s.colour.subarray(0, s.count), o);
    out.edge.set(s.edge.subarray(0, s.count), o);
    out.level.set(s.level.subarray(0, s.count), o);
    out.flags.set(s.flags.subarray(0, s.count), o);
    out.dist.set(s.dist.subarray(0, s.count), o);
    if (s.offset) out.offset!.set(s.offset.subarray(0, s.count), o);
    o += s.count;
  }
  return out;
}
