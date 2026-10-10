// Capacity markers (SPEC 6.2, T-031): a tag on every resource at 75% and up for the demand level
// in force, with the delay per train, coloured busy (75-90%), near full (90-100%) or overloaded
// (over 100%). Nothing shows below 75% or zoom 12. Like station names, tags leave room around
// one another; the worst delays win collisions. DOM tags move on MapLibre's frames.
// A junction's tag is a link to the junction's inspector, where a flyover can be built (T-080).
// Hovering a tag says what it means (T-099). The tags sit under the demand views' canvas.

import { duration } from "../game/format";
import type { WorldView } from "../game/types";
import { selection } from "../game/ui";
import { toLocal } from "./geo";
import type { Overlay } from "./overlay";
import { perfOff, perfTry } from "../perfFlags";
import type { Map as MlMap } from "maplibre-gl";
import { MIN_LABEL_ZOOM, labelPadding } from "./labelLayout";

/** ?perfOff=markers (T-045, measurement only). */
const NO_MARKERS = perfOff("markers");

const KIND = ["Track", "Single track", "Platform", "Terminus", "Junction", "Flat crossing"];
const DEMAND = ["high", "medium", "low"];

export class CapacityMarkers {
  private box: HTMLDivElement;
  private items: { el: HTMLDivElement; x: number; y: number; w: number; h: number }[] = [];
  private visible = true;
  private placedMatrix: number[] | null = null;
  private placedSize = "";

  private tip: HTMLDivElement;

  constructor(private container: HTMLElement, private overlay: Overlay, private map: MlMap) {
    this.box = document.createElement("div");
    this.box.className = "cap-marks";
    container.appendChild(this.box);
    this.tip = document.createElement("div");
    this.tip.className = "tool-tag demand-tag";
    this.tip.style.display = "none";
    container.appendChild(this.tip);
    overlay.afterMapFrame.push(() => this.place());
    document.fonts.ready.then(() => {
      for (const it of this.items) it.w = 0;
      this.placedMatrix = null;
      this.place();
    });
    map.on("movestart", () => (this.tip.style.display = "none"));
  }

  /** The tooltip by a tag, in the map's top layer (over the demand views too). */
  private showTip(el: HTMLElement, lines: string[]) {
    this.tip.replaceChildren(...lines.map((s) => Object.assign(document.createElement("div"), { textContent: s })));
    const r = el.getBoundingClientRect(), c = this.container.getBoundingClientRect();
    this.tip.style.transform = `translate(${Math.round(r.left - c.left)}px, ${Math.round(r.bottom - c.top + 4)}px)`;
    this.tip.style.display = "";
  }

  set(w: WorldView, level: number) {
    this.box.textContent = "";
    this.tip.style.display = "none";
    this.placedMatrix = null;
    // Keep the worst bottlenecks when several compete for the same screen space.
    this.items = [...(w.capacity[level] ?? [])].sort((a, b) => b.delayS - a.delayS || b.rho - a.rho || a.kind - b.kind || a.x - b.x || a.y - b.y).map((m) => {
      const el = document.createElement("div");
      el.className = "cap-mark " + (m.rho > 1 ? "over" : m.rho >= 0.9 ? "near" : "busy");
      if (perfTry("labelLayers")) el.style.willChange = "transform"; // T-045 measurement
      const d = Math.round(m.delayS);
      el.textContent = d >= 2 ? `+${Math.floor(d / 60)}:${String(d % 60).padStart(2, "0")}` : `${Math.round(m.rho * 100)}%`;
      // the tooltip (T-099): what the tag means, in plain words
      const tip = [`${KIND[m.kind] ?? "Track"} at ${Math.round(m.rho * 100)}% of capacity`, d >= 2 ? `Each train waits ${duration(d)} here at ${DEMAND[level] ?? "this"} demand` : "Trains do not wait here yet"];
      if (m.kind === 4) {
        const node = nearestJunction(w, m.x, m.y);
        if (node !== null) {
          el.classList.add("link");
          tip.push("Click to see the junction and its flyover");
          el.onclick = () => (selection.value = { kind: "junction", node });
        }
      }
      el.onmouseenter = () => this.showTip(el, tip);
      el.onmouseleave = () => (this.tip.style.display = "none");
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
    const W = this.overlay.canvas.clientWidth, H = this.overlay.canvas.clientHeight;
    const size = `${W},${H},${zoom < 14}`;
    if (size === this.placedSize && this.placedMatrix?.every((v, i) => v === m[i])) return;
    this.placedMatrix = Array.from(m);
    this.placedSize = size;
    const [padX, padY] = labelPadding(zoom);
    const taken: [number, number, number, number][] = [];
    // Batch measurements before writes, and repeat only when tags or fonts change.
    for (const it of this.items) if (!it.w) {
      it.w = it.el.offsetWidth;
      it.h = it.el.offsetHeight;
    }
    for (const it of this.items) {
      const [x, y] = this.overlay.toScreen(it.x, it.y);
      const sx = Math.round(x), sy = Math.round(y);
      // Match the CSS offset above and right of the resource.
      const r: [number, number, number, number] = [sx + 6 - padX, sy - 22 - padY, sx + 6 + it.w + padX, sy - 22 + it.h + padY];
      const off = sx + 6 + it.w < 0 || sx + 6 > W || sy - 22 + it.h < 0 || sy - 22 > H;
      const hit = taken.some(t => r[0] < t[2] && r[2] > t[0] && r[1] < t[3] && r[3] > t[1]);
      it.el.style.visibility = off || hit ? "hidden" : "";
      if (off || hit) continue;
      taken.push(r);
      it.el.style.transform = `translate(${sx}px, ${sy}px)`;
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
