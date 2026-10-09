// Reshaping blueprint track (T-064, T-092): with a stretch of blueprint track selected, its points
// of intersection (PIs, the corners the track turns at) show as squares that can be dragged, each
// curve labelled with its radius and speed. While dragging, the clock worker previews the new
// shape as a trial edit (the same check an edit gets, one request in flight); letting go commits
// it as one undoable edit, Escape puts the square back. The track inspector sets a curve's radius
// (`setPis`) and the stretch's level (`edgeLevel`). Constructed track cannot be reshaped (SPEC
// 6.4). Track drawn while the track ran through the clicked points (T-079) shows its clicks as
// the squares; its first reshape makes them corners like on any other track.

import type { Map as MlMap } from "maplibre-gl";
import { lngLatToPack, packToLngLat } from "../game/coords";
import { describeIssues } from "../game/issues";
import { selection, tool } from "../game/ui";
import { edit, world } from "../game/world";
import type { ClockClient, PreviewAnswer } from "../workers/clockClient";
import { queryClock } from "../workers/clockClient";
import type { PiInput } from "../workers/protocol";
import type { Level } from "../game/types";
import { toLocal } from "./geo";
import { live } from "./live";
import { strokesFromPreview } from "./network";
import { curveLabels } from "./tools";
import type { Overlay } from "./overlay";

/** A PI of a stretch, pack metres. */
export interface EdgePi {
  x: number;
  y: number;
  /** set radius m, 0 = auto (SPEC 6.1) */
  radius: number;
  level: number;
  /** the radius the curve has (0 = no curve) and its speed limit */
  used: number;
  kmh: number;
  /** the middle of its curve, pack metres */
  mx: number;
  my: number;
}

export interface EdgePis {
  /** drawn while the track ran through the clicks (T-079): the PIs are those clicks, and the
   * curves are what the first reshape will make */
  legacy: boolean;
  pis: EdgePi[];
}

/** `TrackApi.edge_pis` in objects. */
export function parseEdgePis(d: Float64Array): EdgePis {
  const pis: EdgePi[] = [];
  for (let i = 1; i + 8 <= d.length; i += 8)
    pis.push({ x: d[i], y: d[i + 1], radius: d[i + 2], level: d[i + 3], used: d[i + 4], kmh: d[i + 5], mx: d[i + 6], my: d[i + 7] });
  return { legacy: d[0] === 1, pis };
}

const toInput = (p: EdgePi): PiInput => ({ x: p.x, y: p.y, level: p.level as Level, radius: p.radius });

export class EdgeEditor {
  private box: HTMLDivElement;
  private marks: HTMLDivElement[] = [];
  private labels: HTMLDivElement[] = [];
  private tag: HTMLDivElement;
  private edge = -1;
  private version = -1;
  private shape: EdgePis = { legacy: false, pis: [] };
  private drag: { i: number; moved: boolean; from: EdgePi } | null = null;
  private preview: PreviewAnswer | null = null;

  constructor(private map: MlMap, private overlay: Overlay, private client: ClockClient) {
    this.box = document.createElement("div");
    this.box.className = "tool-marks";
    map.getContainer().appendChild(this.box);
    this.tag = document.createElement("div");
    this.tag.className = "tool-tag";
    this.tag.style.display = "none";
    this.box.appendChild(this.tag);
    overlay.afterMapFrame.push(() => this.place());
    addEventListener("pointermove", (e) => this.onMove(e));
    addEventListener("pointerup", () => void this.onUp());
  }

  /** The blueprint edge being edited, or -1: a selected blueprint stretch with the select tool. */
  private target(): number {
    const s = selection.peek();
    if (tool.peek() !== "select" || s?.kind !== "track") return -1;
    const e = world.peek()?.edges.find((x) => x.id === s.edge);
    return e && !e.built ? e.id : -1;
  }

  /** Call when the selection, tool or world changes. */
  async refresh() {
    const e = this.target();
    const v = world.peek()?.version ?? -1;
    if (e === this.edge && v === this.version) return;
    this.edge = e;
    this.version = v;
    if (this.drag) return;
    this.shape = e >= 0 ? parseEdgePis((await queryClock({ what: "edgePis", edge: e })) ?? new Float64Array(0)) : { legacy: false, pis: [] };
    this.place();
  }

  /** New PIs for a stretch (the inspector's radius stepper): one undoable edit. */
  async setPis(edge: number, pis: EdgePi[]) {
    await edit({ op: "edgePis", edge, pis: pis.map(toInput) });
  }

  get dragging() {
    return this.drag !== null;
  }

  /** Escape while dragging: the square goes back, nothing changes. */
  cancel(): boolean {
    const d = this.drag;
    if (!d) return false;
    this.drag = null;
    this.shape.pis[d.i] = d.from;
    this.client.dropPreview();
    this.endPreview();
    return true;
  }

  private local(p: { x: number; y: number }): [number, number] {
    const [lng, lat] = packToLngLat(p.x, p.y);
    return toLocal(lng, lat);
  }

  private onDown(i: number, e: PointerEvent) {
    e.preventDefault();
    e.stopPropagation();
    this.drag = { i, moved: false, from: { ...this.shape.pis[i] } };
    this.map.dragPan.disable();
  }

  private onMove(e: PointerEvent) {
    if (!this.drag || this.edge < 0) return;
    const r = this.map.getContainer().getBoundingClientRect();
    const ll = this.map.unproject([e.clientX - r.left, e.clientY - r.top]);
    const p = lngLatToPack(ll.lng, ll.lat);
    const pis = this.shape.pis;
    pis[this.drag.i] = { ...pis[this.drag.i], x: p.x, y: p.y };
    this.drag.moved = true;
    const edge = this.edge;
    this.client.previewEdge(edge, pis.map(toInput), (a) => {
      if (!this.drag || this.edge !== edge) return;
      this.preview = a;
      this.overlay.renderer.setPreview(a.pts.length >= 6 ? strokesFromPreview(a.pts, !a.ok) : null);
      this.overlay.invalidate();
      this.place();
    });
    this.place();
  }

  private async onUp() {
    const d = this.drag;
    if (!d) return;
    this.drag = null;
    const edge = this.edge;
    if (d.moved && edge >= 0) await edit({ op: "edgePis", edge, pis: this.shape.pis.map(toInput) });
    this.endPreview();
  }

  private endPreview() {
    this.map.dragPan.enable();
    this.preview = null;
    this.overlay.renderer.setPreview(null);
    this.overlay.invalidate();
    this.version = -1; // reload the PIs as the worker has them (refused or not)
    void this.refresh();
  }

  private el(list: HTMLDivElement[], i: number, cls: string): HTMLDivElement {
    let m = list[i];
    if (!m) {
      m = list[i] = document.createElement("div");
      this.box.appendChild(m);
      if (list === this.marks) m.addEventListener("pointerdown", (e) => this.onDown(list.indexOf(m!), e));
    }
    m.className = cls;
    m.style.display = "";
    return m;
  }

  /** Squares and curve labels to the camera. */
  place() {
    // hidden while a node of the stretch is being dragged (map/nodeDrag.ts): they would be stale
    const on = this.edge >= 0 && this.target() === this.edge && !live.nodeDrag?.dragging;
    const pis = on ? this.shape.pis : [];
    for (let i = 0; i < pis.length; i++) {
      const [x, y] = this.overlay.toScreen(...this.local(pis[i]));
      const m = this.el(this.marks, i, "pi-mark drag");
      m.title = "Drag to move this corner";
      m.style.transform = `translate(${Math.round(x)}px, ${Math.round(y)}px)`;
    }
    for (let i = pis.length; i < this.marks.length; i++) this.marks[i].style.display = "none";
    // curve labels: from the preview while dragging (local units), else the stretch's own (a
    // T-079 stretch shows none until it is reshaped: its curves are not the ones listed)
    const pv = this.drag ? this.preview : null;
    const labels: { at: [number, number]; text: string }[] = !on
      ? []
      : pv
        ? curveLabels(pv.verts).map((c) => ({ at: [c.x, c.y] as [number, number], text: c.text }))
        : this.shape.legacy
          ? []
          : this.shape.pis
              .filter((c) => c.used > 0 && c.used < 5000)
              .map((c) => ({ at: this.local({ x: c.mx, y: c.my }), text: `${Math.round(c.used).toLocaleString("en-US")} m, ${Math.round(c.kmh)} km/h` }));
    labels.forEach((c, i) => {
      const [x, y] = this.overlay.toScreen(...c.at);
      const l = this.el(this.labels, i, "pi-label");
      l.textContent = c.text;
      l.style.transform = `translate(${Math.round(x + 8)}px, ${Math.round(y + 6)}px)`;
    });
    for (let i = labels.length; i < this.labels.length; i++) this.labels[i].style.display = "none";
    if (pv && !pv.ok && this.drag) {
      const [x, y] = this.overlay.toScreen(...this.local(pis[this.drag.i]));
      this.tag.className = "tool-tag bad";
      this.tag.textContent = describeIssues(pv.issues) || "Cannot be built like this.";
      this.tag.style.display = "";
      this.tag.style.transform = `translate(${Math.round(x + 14)}px, ${Math.round(y - 30)}px)`;
    } else this.tag.style.display = "none";
  }
}
