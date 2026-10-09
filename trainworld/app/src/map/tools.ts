// The map tools (T-023, T-024): what a click does while drawing track, placing stations or adding
// stops to a line. Selecting (picking) is map/pick.ts; main.ts routes clicks here when a tool is
// on.
//
// Track: click where the track turns (T-023, T-092): each click is a point of intersection (PI)
// and the track cuts the corner there with a circular curve, auto radius (SPEC 6.1), so a new
// click leaves the track before the previous click's curve as it was. The first click may start
// on a track end or on track (a new junction, flat, T-080), and a click on track or a track end
// finishes the route there. A click on the last point (or a double click), or Enter, finishes at
// that point. Backspace takes the last point back, Escape drops the route. Each point takes the
// level selected when it was placed. While drawing, the worker previews the route (one request in
// flight): its cost, whether it can be built and why not, and the radius and speed of every curve.
//
// Delete (T-096): hovering shows what a click removes (a station or flyover ringed, a stretch of
// track drawn red, its name by the cursor); a click removes blueprint at once and asks first for
// anything constructed (game/remove.ts).
//
// Everything drawn is blueprint until constructed (SPEC 6.4).

import type { Geometry } from "geojson";
import type { Map as MlMap, MapMouseEvent } from "maplibre-gl";
import { lngLatToPack } from "../game/coords";
import { describeIssues } from "../game/issues";
import { km, money, moneyShort } from "../game/format";
import { ask, buildLevel, buildPlatform, lineDraft, selection, singleTrack, tool } from "../game/ui";
import { describeRemovable, removeThing, type Removable } from "../game/remove";
import { edit, nextFreeColour, nextLineName, say, world } from "../game/world";
import type { PiInput, RouteEnd } from "../workers/protocol";
import type { ClockClient, PreviewAnswer } from "../workers/clockClient";
import { toLngLat, toLocal } from "./geo";
import { strokesFromPreview } from "./network";
import type { Overlay } from "./overlay";

const SNAP_NODE_PX = 12;
const SNAP_EDGE_PX = 10;

type Snap =
  | { kind: "node"; node: number; lx: number; ly: number }
  | { kind: "edge"; edge: number; lx: number; ly: number }
  | { kind: "free"; lx: number; ly: number };

interface Placed {
  end: RouteEnd;
  lx: number;
  ly: number;
}

export class Tools {
  private start: Placed | null = null;
  private pis: { pi: PiInput; lx: number; ly: number }[] = [];
  private hover: Snap | null = null;
  /** the delete tool: what a click would remove, and the pointer (map px) */
  private delHover: (Removable & { lx: number; ly: number }) | null = null;
  private delAt: [number, number] = [0, 0];
  private lastPreview: PreviewAnswer | null = null;
  private box: HTMLDivElement;
  private marks: HTMLDivElement[] = [];
  private tag: HTMLDivElement;
  private ring: HTMLDivElement;

  constructor(private map: MlMap, private overlay: Overlay, private client: ClockClient) {
    this.box = document.createElement("div");
    this.box.className = "tool-marks";
    map.getContainer().appendChild(this.box);
    this.tag = document.createElement("div");
    this.tag.className = "tool-tag";
    this.ring = document.createElement("div");
    this.ring.className = "snap-ring";
    this.box.append(this.tag, this.ring);
    overlay.afterMapFrame.push(() => this.place());
  }

  get drawing() {
    return this.start !== null;
  }

  // ---------------------------------------------------------------- snapping

  private screen(lx: number, ly: number) {
    return this.overlay.toScreen(lx, ly);
  }

  /** What is under the pointer for building: a node, a point on track, or open ground. */
  private snap(px: number, py: number, lngLat: { lng: number; lat: number }): Snap {
    const w = world.peek();
    const n = this.overlay.renderer.net;
    const [fx, fy] = toLocal(lngLat.lng, lngLat.lat);
    if (!w || !n) return { kind: "free", lx: fx, ly: fy };
    let best = SNAP_NODE_PX * SNAP_NODE_PX, hit: Snap | null = null;
    for (const node of w.nodes.values()) {
      const [lx, ly] = toLocal(node.lng, node.lat);
      const [x, y] = this.screen(lx, ly);
      const d = (x - px) ** 2 + (y - py) ** 2;
      if (d < best) {
        best = d;
        hit = { kind: "node", node: node.id, lx, ly };
      }
    }
    if (hit) return hit;
    best = SNAP_EDGE_PX * SNAP_EDGE_PX;
    for (const [edge, pts] of n.edgePts) {
      for (let k = 0; k + 5 < pts.length; k += 3) {
        const [x0, y0] = this.screen(pts[k], pts[k + 1]);
        const [x1, y1] = this.screen(pts[k + 3], pts[k + 4]);
        const dx = x1 - x0, dy = y1 - y0, l2 = dx * dx + dy * dy;
        const t = l2 ? Math.max(0, Math.min(1, ((px - x0) * dx + (py - y0) * dy) / l2)) : 0;
        const d = (x0 + t * dx - px) ** 2 + (y0 + t * dy - py) ** 2;
        if (d < best) {
          best = d;
          hit = { kind: "edge", edge, lx: pts[k] + (pts[k + 3] - pts[k]) * t, ly: pts[k + 1] + (pts[k + 4] - pts[k + 1]) * t };
        }
      }
    }
    return hit ?? { kind: "free", lx: fx, ly: fy };
  }

  private pack(lx: number, ly: number) {
    const [lng, lat] = toLngLat(lx, ly);
    return lngLatToPack(lng, lat);
  }

  private endOf(s: Snap): RouteEnd {
    if (s.kind === "node") return { kind: "node", node: s.node };
    const p = this.pack(s.lx, s.ly);
    if (s.kind === "edge") return { kind: "edge", edge: s.edge, x: p.x, y: p.y };
    return { kind: "free", x: p.x, y: p.y, level: buildLevel.peek() };
  }

  // ---------------------------------------------------------------- events

  move(e: MapMouseEvent) {
    const t = tool.peek();
    if (t === "select") return;
    if (t === "delete") return this.deleteHover(e.point.x, e.point.y);
    this.hover = t === "line" ? null : this.snap(e.point.x, e.point.y, e.lngLat);
    if (t === "track" && this.start) this.requestPreview();
    this.place();
  }

  click(e: MapMouseEvent) {
    const t = tool.peek();
    if (t === "track") this.trackClick(e);
    else if (t === "station") void this.stationClick(e);
    else if (t === "line") this.lineClick(e);
    else if (t === "delete") this.deleteClick(e);
  }

  /** Keys while a tool is on; true if used. */
  key(e: KeyboardEvent): boolean {
    const t = tool.peek();
    if (t === "select") return false;
    if (e.key === "Escape") {
      if (t === "track" && this.start) this.reset();
      else this.exit();
      return true;
    }
    if (t === "track" && this.start) {
      if (e.key === "Backspace") {
        if (this.pis.length) this.pis.pop();
        else this.reset();
        this.requestPreview();
        this.place();
        return true;
      }
      if (e.key === "Enter") {
        this.finishAtLast();
        return true;
      }
    }
    if (t === "line" && e.key === "Enter") {
      this.exit();
      return true;
    }
    return false;
  }

  /** Leave the current tool (back to selecting). */
  exit() {
    const d = lineDraft.peek();
    if (tool.peek() === "line" && d.line !== null) selection.value = { kind: "line", line: String(d.line) };
    this.reset();
    lineDraft.value = { line: null, stops: [] };
    ask.value = null;
    tool.value = "select";
  }

  /** Drop the route being drawn (and what the delete tool points at). */
  reset() {
    this.start = null;
    this.pis = [];
    this.delHover = null;
    this.lastPreview = null;
    this.overlay.renderer.setPreview(null);
    this.overlay.invalidate();
    this.place();
  }

  // ---------------------------------------------------------------- track

  private trackClick(e: MapMouseEvent) {
    const s = this.snap(e.point.x, e.point.y, e.lngLat);
    if (!this.start) {
      this.start = { end: this.endOf(s), lx: s.lx, ly: s.ly };
      this.requestPreview();
      this.place();
      return;
    }
    if (s.kind !== "free") {
      void this.finish(this.endOf(s));
      return;
    }
    // A click on the last point (the second click of a double click lands here too) finishes.
    const last = this.pis[this.pis.length - 1] ?? this.start;
    const [lx, ly] = this.screen(last.lx, last.ly);
    if (Math.hypot(lx - e.point.x, ly - e.point.y) < 6) {
      this.finishAtLast();
      return;
    }
    const p = this.pack(s.lx, s.ly);
    this.pis.push({ pi: { x: p.x, y: p.y, level: buildLevel.peek() }, lx: s.lx, ly: s.ly });
    this.requestPreview();
    this.place();
  }

  private finishAtLast() {
    const last = this.pis.pop();
    if (!last) return;
    void this.finish({ kind: "free", x: last.pi.x, y: last.pi.y, level: last.pi.level });
  }

  private async finish(to: RouteEnd) {
    if (!this.start) return;
    const a = await edit({ op: "route", from: this.start.end, to, pis: this.pis.map((p) => p.pi), single: singleTrack.peek() });
    if (a.ok) {
      const p = this.lastPreview;
      say(p ? `Blueprint added: ${(p.lengthM / 1000).toFixed(2)} km, ${money(p.cost * 1e6)} to construct.` : "Blueprint added.");
      this.reset();
    }
  }

  private requestPreview() {
    if (!this.start) return;
    const h = this.hover;
    if (!h) return;
    // Nothing to show while the pointer is on the last point placed.
    const last = this.pis[this.pis.length - 1] ?? this.start;
    const [ax, ay] = this.screen(last.lx, last.ly), [bx, by] = this.screen(h.lx, h.ly);
    if (Math.hypot(ax - bx, ay - by) < 3) return;
    const to = this.endOf(h);
    this.client.preview({ from: this.start.end, to, pis: this.pis.map((p) => p.pi), single: singleTrack.peek() }, (a) => {
      if (!this.start) return;
      this.lastPreview = a;
      this.overlay.renderer.setPreview(a.pts.length >= 6 ? strokesFromPreview(a.pts, !a.ok) : null);
      this.overlay.invalidate();
      this.place();
    });
  }

  // ---------------------------------------------------------------- stations

  private async stationClick(e: MapMouseEvent) {
    const s = this.snap(e.point.x, e.point.y, e.lngLat);
    if (s.kind === "free") {
      say("Click on track to put a station there.");
      return;
    }
    const name = this.nameFor(e.point.x, e.point.y);
    const at = s.kind === "node" ? { kind: "node" as const, node: s.node } : { kind: "edge" as const, edge: s.edge, ...this.pack(s.lx, s.ly) };
    const a = await edit({ op: "addStation", at, platform: buildPlatform.peek(), name });
    if (a.ok) say(`${name} added as a blueprint station.`);
  }

  /**
   * A name for a new station: the nearest named street within 400 m on the basemap's loaded
   * tiles (shortened New York style), else the nearest neighbourhood, else "Station N". Names
   * already in use are skipped, so a second station on Broadway takes its cross street.
   */
  private nameFor(px: number, py: number): string {
    const taken = new Set(world.peek()?.save.stations.map((s) => s.name) ?? []);
    const at = this.map.unproject([px, py]);
    const kx = 111320 * Math.cos((at.lat * Math.PI) / 180), ky = 110540;
    const dist = (g: Geometry): number => {
      const lines: number[][][] = g.type === "LineString" ? [g.coordinates] : g.type === "MultiLineString" ? g.coordinates : g.type === "Point" ? [[g.coordinates]] : [];
      let best = Infinity;
      for (const l of lines) {
        const pts = l.map((c) => [(c[0] - at.lng) * kx, (c[1] - at.lat) * ky]);
        if (pts.length === 1) best = Math.min(best, Math.hypot(pts[0][0], pts[0][1]));
        for (let i = 0; i + 1 < pts.length; i++) {
          const [ax, ay] = pts[i], [bx, by] = pts[i + 1];
          const dx = bx - ax, dy = by - ay, l2 = dx * dx + dy * dy;
          const t = l2 ? Math.max(0, Math.min(1, (-ax * dx - ay * dy) / l2)) : 0;
          best = Math.min(best, Math.hypot(ax + t * dx, ay + t * dy));
        }
      }
      return best;
    };
    const pickFrom = (sourceLayer: string, maxM: number, ok: (p: Record<string, any>) => boolean): string | null => {
      const layer = this.map.getStyle()?.layers.find((l) => (l as any)["source-layer"] === sourceLayer) as any;
      if (!layer) return null;
      let feats: { geometry: Geometry; properties: Record<string, any> }[] = [];
      try {
        feats = this.map.querySourceFeatures(layer.source, { sourceLayer }) as any;
      } catch {
        return null;
      }
      const best = new Map<string, number>();
      for (const f of feats) {
        const raw = (f.properties?.["name:latin"] || f.properties?.name_en || f.properties?.name) as string | undefined;
        if (!raw || !ok(f.properties)) continue;
        const name = shorten(raw);
        const d = dist(f.geometry);
        if (d <= maxM && d < (best.get(name) ?? Infinity)) best.set(name, d);
      }
      return [...best].sort((a, b) => a[1] - b[1]).map((e) => e[0]).find((n) => !taken.has(n)) ?? null;
    };
    const street = pickFrom("transportation_name", 400, (p) => p.class !== "motorway");
    if (street) return street;
    const place = pickFrom("place", 1200, (p) => ["neighbourhood", "quarter", "suburb", "village", "hamlet", "town"].includes(p.class));
    if (place) return place;
    for (let k = 1; ; k++) if (!taken.has(`Station ${k}`)) return `Station ${k}`;
  }

  // ---------------------------------------------------------------- delete (T-096)

  /** What a click with the delete tool would remove: a station, a flyover, else a stretch of
   * track (lines are removed from the Lines tab). */
  private deletePick(px: number, py: number): (Removable & { lx: number; ly: number }) | null {
    const w = world.peek();
    const n = this.overlay.renderer.net;
    if (!w || !n) return null;
    const near = (lx: number, ly: number, slop: number) => {
      const [x, y] = this.screen(lx, ly);
      const d = Math.hypot(x - px, y - py);
      return d < slop ? d : Infinity;
    };
    let best = Infinity, hit: (Removable & { lx: number; ly: number }) | null = null;
    for (let i = 0; i < n.stationCount; i++) {
      const lx = n.stations[i * 3], ly = n.stations[i * 3 + 1];
      const d = near(lx, ly, 10);
      if (d < best) [best, hit] = [d, { kind: "station", node: Number(n.stationIds[i]), lx, ly }];
    }
    if (hit) return hit;
    for (const node of w.nodes.values()) {
      if (!node.flying || node.ports < 3) continue;
      const [lx, ly] = toLocal(node.lng, node.lat);
      const d = near(lx, ly, 10);
      if (d < best) [best, hit] = [d, { kind: "flyover", node: node.id, lx, ly }];
    }
    if (hit) return hit;
    const s = this.snap(px, py, this.map.unproject([px, py]));
    if (s.kind === "edge") return { kind: "track", edge: s.edge, lx: s.lx, ly: s.ly };
    if (s.kind === "node") {
      // a plain node (track end, junction): the stretch nearest to the pointer there
      let e = -1, bd = Infinity;
      for (const [edge, pts] of n.edgePts)
        for (let k = 0; k < pts.length; k += 3) {
          const d = near(pts[k], pts[k + 1], 14);
          if (d < bd) [bd, e] = [d, edge];
        }
      if (e >= 0) return { kind: "track", edge: e, lx: s.lx, ly: s.ly };
    }
    return null;
  }

  private deleteHover(px: number, py: number) {
    const h = this.deletePick(px, py);
    const same = (a: Removable | null, b: Removable | null) => JSON.stringify(a && { ...a, lx: 0, ly: 0 }) === JSON.stringify(b && { ...b, lx: 0, ly: 0 });
    const changed = !same(h, this.delHover);
    this.delHover = h;
    this.delAt = [px, py];
    if (changed) {
      // the stretch that would go, drawn red over itself
      const pts = h?.kind === "track" ? this.overlay.renderer.net?.edgePts.get(h.edge) : undefined;
      this.overlay.renderer.setPreview(pts && pts.length >= 6 ? strokesFromPreview(pts, true) : null);
      this.overlay.invalidate();
    }
    this.map.getCanvas().style.cursor = h ? "pointer" : "crosshair";
    this.place();
  }

  private deleteClick(e: MapMouseEvent) {
    const h = this.deletePick(e.point.x, e.point.y);
    if (!h) {
      say("Click track, a station or a flyover to remove it.");
      return;
    }
    void removeThing(h).then(() => this.deleteHover(e.point.x, e.point.y));
  }

  // ---------------------------------------------------------------- lines

  private lineClick(e: MapMouseEvent) {
    const n = this.overlay.renderer.net;
    if (!n) return;
    let best = 12 * 12, hit = -1;
    for (let i = 0; i < n.stationCount; i++) {
      const [x, y] = this.screen(n.stations[i * 3], n.stations[i * 3 + 1]);
      const d = (x - e.point.x) ** 2 + (y - e.point.y) ** 2;
      if (d < best) {
        best = d;
        hit = Number(n.stationIds[i]);
      }
    }
    if (hit < 0) {
      say("Click a station to add it to the line. Press Enter when the line is done.");
      return;
    }
    void this.addStop(hit);
  }

  private async addStop(station: number) {
    const d = lineDraft.peek();
    if (d.stops[d.stops.length - 1] === station) {
      this.exit();
      return;
    }
    const stops = [...d.stops, station];
    if (d.line === null) {
      if (stops.length < 2) {
        lineDraft.value = { line: null, stops };
        return;
      }
      const a = await edit({ op: "addLine", stops, name: nextLineName(), colour: nextFreeColour() });
      if (a.ok && a.line !== undefined) {
        lineDraft.value = { line: a.line, stops };
        selection.value = { kind: "line", line: String(a.line) };
      }
      return;
    }
    const a = await edit({ op: "setStops", line: d.line, stops });
    if (a.ok) lineDraft.value = { line: d.line, stops };
  }

  // ---------------------------------------------------------------- markers

  private el(i: number, cls: string): HTMLDivElement {
    let m = this.marks[i];
    if (!m) {
      m = this.marks[i] = document.createElement("div");
      this.box.appendChild(m);
    }
    m.className = cls;
    m.textContent = "";
    m.style.display = "";
    return m;
  }

  /** Move the DOM markers to the camera (after each MapLibre frame and each change). */
  place() {
    let k = 0;
    const put = (m: HTMLDivElement, lx: number, ly: number, dx = 0, dy = 0) => {
      const [x, y] = this.screen(lx, ly);
      m.style.transform = `translate(${Math.round(x + dx)}px, ${Math.round(y + dy)}px)`;
    };
    const t = tool.peek();
    if (t === "track" && this.start) {
      put(this.el(k++, "pi-mark"), this.start.lx, this.start.ly);
      for (const p of this.pis) put(this.el(k++, "pi-mark"), p.lx, p.ly);
      const pv = this.lastPreview;
      if (pv) {
        for (const c of curveLabels(pv.verts)) {
          const m = this.el(k++, "pi-label");
          m.textContent = c.text;
          put(m, c.x, c.y, 8, 6);
        }
      }
    }
    for (let i = k; i < this.marks.length; i++) this.marks[i].style.display = "none";

    // the delete tool: a ring on the station or flyover, and what a click removes by the cursor
    if (t === "delete") {
      const d = this.delHover, what = d && describeRemovable(d);
      if (d && d.kind !== "track") {
        this.ring.style.display = "";
        put(this.ring, d.lx, d.ly);
      } else this.ring.style.display = "none";
      if (what) {
        this.tag.style.display = "";
        this.tag.className = "tool-tag";
        this.tag.textContent = `Remove ${what.what}`;
        this.tag.style.transform = `translate(${Math.round(this.delAt[0] + 14)}px, ${Math.round(this.delAt[1] + 12)}px)`;
      } else this.tag.style.display = "none";
      return;
    }
    // the snap ring and the cursor tag
    const h = this.hover;
    if ((t === "track" || t === "station") && h && h.kind !== "free") {
      this.ring.style.display = "";
      put(this.ring, h.lx, h.ly);
    } else this.ring.style.display = "none";
    const pv = this.lastPreview;
    if (t === "track" && this.start && h && pv) {
      this.tag.style.display = "";
      this.tag.className = "tool-tag" + (pv.ok ? "" : " bad");
      this.tag.textContent = pv.ok
        ? `${km(pv.lengthM)}, cost ${pv.mult.toFixed(2)}x, ${moneyShort(pv.cost * 1e6)}`
        : describeIssues(pv.issues) || "Cannot be built here.";
      put(this.tag, h.lx, h.ly, 14, 12);
    } else this.tag.style.display = "none";
  }
}

/**
 * One label per curve from a preview's vertices (x, y at the middle of each curve, radius, km/h):
 * a curve fitted as several PIs (one turning more than a right angle) has the same radius at
 * consecutive vertices and gets one label, at its middle one. Curves too gentle to slow a train
 * are not labelled.
 */
export function curveLabels(verts: Float64Array): { x: number; y: number; text: string }[] {
  const out: { x: number; y: number; text: string }[] = [];
  const nv = verts.length / 4;
  for (let i = 1; i < nv - 1; ) {
    const r = verts[i * 4 + 2];
    let j = i + 1;
    while (j < nv - 1 && r > 0 && Math.abs(verts[j * 4 + 2] - r) < 0.5) j++;
    if (r > 0 && r < 5000) {
      const m = (i + j - 1) >> 1;
      out.push({ x: verts[m * 4], y: verts[m * 4 + 1], text: `${Math.round(r).toLocaleString("en-US")} m, ${Math.round(verts[i * 4 + 3])} km/h` });
    }
    i = j;
  }
  return out;
}

/** "West 42nd Street" -> "W 42nd St". */
function shorten(n: string): string {
  return n
    .replace(/\bWest\b/g, "W")
    .replace(/\bEast\b/g, "E")
    .replace(/\bNorth\b/g, "N")
    .replace(/\bSouth\b/g, "S")
    .replace(/\bStreet\b/g, "St")
    .replace(/\bAvenue\b/g, "Av")
    .replace(/\bBoulevard\b/g, "Blvd")
    .replace(/\bRoad\b/g, "Rd")
    .replace(/\bParkway\b/g, "Pkwy")
    .replace(/\bPlace\b/g, "Pl")
    .replace(/\bExpressway\b/g, "Expy");
}

