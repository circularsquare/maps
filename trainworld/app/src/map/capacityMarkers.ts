// Capacity markers (SPEC 6.2, T-031): a tag on every resource at 75% and up for the demand level
// in force, with the delay per train, coloured busy (75-90%), near full (90-100%) or overloaded
// (over 100%). Nothing shows below 75%. DOM tags moved on MapLibre's frames, like station names.
// A junction's tag is a link to the junction's inspector, where a flyover can be built (T-080).
// Hovering a tag says what it means (T-099). The tags sit under the demand views' canvas.

import { duration } from "../game/format";
import type { WorldView } from "../game/types";
import { selection } from "../game/ui";
import { toLocal } from "./geo";
import type { Overlay } from "./overlay";
import { perfOff, perfTry } from "../perfFlags";

/** ?perfOff=markers (T-045, measurement only). */
const NO_MARKERS = perfOff("markers");

const KIND = ["Track", "Single track", "Platform", "Terminus", "Junction", "Flat crossing"];
const DEMAND = ["high", "medium", "low"];

export class CapacityMarkers {
  private box: HTMLDivElement;
  private items: { el: HTMLDivElement; x: number; y: number }[] = [];
  private visible = true;

  private tip: HTMLDivElement;

  constructor(private container: HTMLElement, private overlay: Overlay) {
    this.box = document.createElement("div");
    this.box.className = "cap-marks";
    container.appendChild(this.box);
    this.tip = document.createElement("div");
    this.tip.className = "tool-tag demand-tag";
    this.tip.style.display = "none";
    container.appendChild(this.tip);
    overlay.afterMapFrame.push(() => this.place());
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
    this.items = (w.capacity[level] ?? []).map((m) => {
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
      return { el, x: m.x, y: m.y };
    });
    this.place();
  }

  setVisible(on: boolean) {
    this.visible = on;
    this.box.style.display = on ? "" : "none";
    if (on) this.place();
  }

  place() {
    if (!this.visible || !this.overlay.camMatrix || NO_MARKERS) return;
    for (const it of this.items) {
      const [x, y] = this.overlay.toScreen(it.x, it.y);
      it.el.style.transform = `translate(${Math.round(x)}px, ${Math.round(y)}px)`;
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
