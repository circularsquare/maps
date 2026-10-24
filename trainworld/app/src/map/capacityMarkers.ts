// Capacity markers (SPEC 6.2, T-031): a tag on every resource at 75% and up for the demand level
// in force, with the delay per train, coloured busy (75-90%), near full (90-100%) or overloaded
// (over 100%, or full with some line's trains waiting for room, T-517). Nothing shows below 75% or zoom 12. Like station names, tags leave room around
// one another; the worst delays win collisions. DOM tags move on MapLibre's frames.
// A junction's tag is a link to the junction's inspector, where a flyover can be built (T-080).
// Hovering a tag says what it means (T-099). The tags sit under the demand views' canvas.
// Station names are placed first and tags keep off them (T-184): a tag takes the first of six
// spots around its resource (above right, above left, below right, below left, right, left)
// clear of every shown name and every tag already placed, and hides only when none is.

import { effect } from "@preact/signals";
import { CAPACITY_KINDS } from "../game/capacityKinds";
import { duration } from "../game/format";
import type { Selection, WorldView } from "../game/types";
import { selection, tool } from "../game/ui";
import { hoverAt } from "./demandHover";
import { hoverHeld } from "../ui/blurHold";
import { toLocal } from "./geo";
import type { Overlay } from "./overlay";
import { perfOff, perfTry } from "../perfFlags";
import type { Map as MlMap } from "maplibre-gl";
import { BoxGrid, MIN_LABEL_ZOOM, labelPadding, tipBeside } from "./labelLayout";
import type { StationLabels } from "./stationLabels";

/** ?perfOff=markers (T-045, measurement only). */
const NO_MARKERS = perfOff("markers");
/** px between a tag's corner and its resource, and kept clear around a station name (T-184). */
const GAP_X = 6, GAP_Y = 5, NAME_CLEAR = 3;

const DEMAND = ["high", "medium", "low"];

export class CapacityMarkers {
  /** Demand discs draw above these DOM tags; wired to the same select handler as the map. */
  demandClick: ((px: number, py: number, add: boolean) => Selection | undefined) | null = null;
  /** The select tool's click at a map point (px), for a click on a tag that opens nothing of its
   * own: the station, train or track under the tag is selected as if the tag were not there (T-256). */
  selectClick: ((px: number, py: number, add: boolean) => void) | null = null;
  private box: HTMLDivElement;
  private items: { el: HTMLDivElement; x: number; y: number; w: number; h: number; shown?: boolean; transform?: string }[] = [];
  private visible = true;
  private placedMatrix: number[] | null = null;
  private placedSize = "";
  private placedLabels = -1;
  private key = "";

  private tip: HTMLDivElement;

  /** `labels` must be constructed first: its frame hook then runs before this one, so a pan's
   * label offset is current when the tags are placed. */
  constructor(private container: HTMLElement, private overlay: Overlay, private map: MlMap, private labels: StationLabels | null = null) {
    this.box = document.createElement("div");
    this.box.className = "cap-marks";
    container.appendChild(this.box);
    this.tip = document.createElement("div");
    this.tip.className = "tool-tag demand-tag";
    this.tip.style.display = "none";
    container.appendChild(this.tip);
    // T-224: with any tool on, the tags take no pointer events, so every press, move and double
    // click reaches the map (a tag can sit on a station or track the tool wants)
    effect(() => {
      const pass = tool.value !== "select";
      this.box.classList.toggle("pass", pass);
      if (pass) this.tip.style.display = "none";
    });
    overlay.afterMapFrame.push(() => this.place());
    if (labels) labels.onPlace = () => this.place();
    const remeasure = () => {
      for (const it of this.items) it.w = 0;
      this.placedMatrix = null;
      this.place();
    };
    document.fonts.ready.then(remeasure);
    document.addEventListener("tw-fontchange", remeasure);
    map.on("movestart", () => (this.tip.style.display = "none"));
  }

  set(w: WorldView, level: number) {
    // Keep the worst bottlenecks when several compete for the same screen space.
    const sorted = [...(w.capacity[level] ?? [])].sort((a, b) => b.delayS - a.delayS || b.rho - a.rho || a.kind - b.kind || a.x - b.x || a.y - b.y);
    const junctions = sorted.map((m) => (m.kind === 4 ? nearestJunction(w, m.x, m.y) : null));
    // most edits leave the tags as they were: keep them then, no rebuild and no layout (T-496)
    const key = level + "|" + sorted.map((m, i) => `${m.kind},${m.rho},${m.delayS},${m.tph},${m.held},${m.x},${m.y},${junctions[i]}`).join("|");
    if (key === this.key) return;
    this.key = key;
    this.box.textContent = "";
    this.tip.style.display = "none";
    this.placedMatrix = null;
    this.items = sorted.map((m, i) => {
      const el = document.createElement("div");
      el.className = "cap-mark " + (m.rho > 1 || m.held ? "over" : m.rho >= 0.9 ? "near" : "busy");
      if (perfTry("labelLayers")) el.style.willChange = "transform"; // T-045 measurement
      const d = Math.round(m.delayS);
      el.textContent = d >= 2 ? `+${Math.floor(d / 60)}:${String(d % 60).padStart(2, "0")}` : `${Math.round(m.rho * 100)}%`;
      // the tooltip (T-099): what the tag means, in plain words
      const text = CAPACITY_KINDS[m.kind] ?? CAPACITY_KINDS[0];
      const tip = [`${text.name} at ${Math.round(m.rho * 100)}% of capacity`];
      // T-517: its trains an hour against the most it takes
      if (m.rho > 0) tip.push(`${Math.round(m.tph)} trains an hour${text.ways} against a limit of ${Math.round(m.tph / m.rho)}`);
      tip.push(text.cause, `Helps: ${text.help.charAt(0).toLowerCase()}${text.help.slice(1)}`);
      if (m.held) tip.push("The line's extra trains are held out of service at its ends");
      if (d >= 2) tip.push(`Each train waits ${duration(d)} here at ${DEMAND[level] ?? "this"} demand`);
      else if (!m.held) tip.push("Trains do not wait here yet");
      let junction: number | null = null;
      if (m.kind === 4) {
        const node = junctions[i];
        if (node !== null) {
          junction = node;
          el.classList.add("link");
          tip.push("Click to see the junction and its flyover");
        }
      }
      const point = (e: MouseEvent) => {
        const r = this.container.getBoundingClientRect();
        return [e.clientX - r.left, e.clientY - r.top] as const;
      };
      el.onclick = (e) => {
        if (tool.peek() !== "select") return;
        const [x, y] = point(e);
        const above = this.demandClick?.(x, y, e.shiftKey);
        if (above !== undefined) {
          selection.value = above;
          e.stopPropagation();
        } else if (junction !== null) {
          selection.value = { kind: "junction", node: junction };
          e.stopPropagation();
        } else if (this.selectClick) {
          // T-256: otherwise what is drawn under the tag (its station, a train, the track)
          this.selectClick(x, y, e.shiftKey);
          e.stopPropagation();
        }
      };
      const hover = (e: MouseEvent) => {
        const [x, y] = point(e);
        if (hoverAt(this.map, x, y)) this.tip.style.display = "none";
        else tipBeside(this.tip, el, this.container, tip, "below");
      };
      el.onmouseenter = hover;
      el.onmousemove = hover;
      el.onmouseleave = () => hoverHeld() || (this.tip.style.display = "none");
      this.box.appendChild(el);
      return { el, x: m.x, y: m.y, w: 0, h: 0 };
    });
    this.place();
  }

  setVisible(on: boolean) {
    this.visible = on;
    this.placedMatrix = null;
    this.place();
  }

  place() {
    const zoom = this.map.getZoom();
    const show = this.visible && zoom >= MIN_LABEL_ZOOM && !NO_MARKERS;
    this.box.style.display = show ? "" : "none";
    if (!show) {
      this.tip.style.display = "none";
      this.placedMatrix = null;
      return;
    }
    const m = this.overlay.camMatrix;
    if (!m) return;
    const W = this.overlay.cssW, H = this.overlay.cssH;
    const size = `${W},${H},${zoom < 14}`;
    const labelsAt = this.labels?.version ?? 0;
    if (size === this.placedSize && labelsAt === this.placedLabels && this.placedMatrix?.every((v, i) => v === m[i])) return;
    this.placedMatrix = Array.from(m);
    this.placedSize = size;
    this.placedLabels = labelsAt;
    const [padX, padY] = labelPadding(zoom);
    // Shown station names on or near the screen, with a little clearance (T-184).
    const names = (this.labels?.rects() ?? [])
      .filter((n) => n[2] > -64 && n[0] < W + 64 && n[3] > -64 && n[1] < H + 64)
      .map((n) => [n[0] - NAME_CLEAR, n[1] - NAME_CLEAR, n[2] + NAME_CLEAR, n[3] + NAME_CLEAR]);
    const nameGrid = new BoxGrid(), takenGrid = new BoxGrid();
    for (const n of names) nameGrid.add(n);
    // Batch measurements before writes, and repeat only when tags or fonts change.
    for (const it of this.items) if (!it.w) {
      it.w = it.el.offsetWidth;
      it.h = it.el.offsetHeight;
    }
    for (const it of this.items) {
      const [x, y] = this.overlay.toScreen(it.x, it.y);
      const sx = Math.round(x), sy = Math.round(y), w = it.w, h = it.h;
      // The tag's top left at each spot around the resource, in order of preference.
      const spots: [number, number][] = [
        [sx + GAP_X, sy - GAP_Y - h], [sx - GAP_X - w, sy - GAP_Y - h],
        [sx + GAP_X, sy + GAP_Y], [sx - GAP_X - w, sy + GAP_Y],
        [sx + GAP_X + 2, sy - Math.round(h / 2)], [sx - GAP_X - 2 - w, sy - Math.round(h / 2)],
      ];
      let at: [number, number] | null = null;
      for (const [tx, ty] of spots) {
        if (tx + w < 0 || tx > W || ty + h < 0 || ty > H) continue;
        if (nameGrid.hits([tx, ty, tx + w, ty + h])) continue;
        const r = [tx - padX, ty - padY, tx + w + padX, ty + h + padY];
        if (takenGrid.hits(r)) continue;
        takenGrid.add(r);
        at = [tx, ty];
        break;
      }
      // only what changed is written (T-689)
      if (it.shown !== !!at) it.el.style.visibility = at ? "" : "hidden";
      it.shown = !!at;
      const transform = at ? `translate(${at[0]}px, ${at[1]}px)` : it.transform;
      if (transform !== it.transform) it.el.style.transform = it.transform = transform!;
    }
  }
}

/** The junction a marker at (x, y) (local units) belongs to: the nearest node where three or
 * more tracks meet. */
function nearestJunction(w: WorldView, x: number, y: number): number | null {
  let best = Infinity, id: number | null = null;
  for (const n of w.nodes.values()) {
    if (n.ports < 3) continue;
    const [lx, ly] = toLocal(n.lng, n.lat);
    const d = (lx - x) ** 2 + (ly - y) ** 2;
    if (d < best) {
      best = d;
      id = n.id;
    }
  }
  return id;
}
