// Reversal icons (T-797, docs/ui-map.md): a large U-turn sign over every station where a line's
// trains reverse partway, the same size at every zoom so an unintended reversal shows from far
// out. It floats just above the station, clear of the stop editing marker on it. Hovering it names
// the lines that reverse there; a click selects the first of them (not while stops are edited).

import { effect } from "@preact/signals";
import type { Map as MlMap } from "maplibre-gl";
import type { WorldView } from "../game/types";
import { selection, stopEditing, tool } from "../game/ui";
import { world } from "../game/world";
import { toLocal } from "./geo";
import type { Overlay } from "./overlay";

/** the most trains an hour reversing at one side of a station (sim `REVERSAL_LIMIT_S`) */
const REVERSALS_AN_HOUR = 12;

const ICON = `<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M7.5 19.5V10a4.75 4.75 0 0 1 9.5 0v7" fill="none" stroke="currentColor" stroke-width="2.8" stroke-linecap="round"/><path d="M12.8 13.4l4.2 4.3 4.2-4.3" fill="none" stroke="currentColor" stroke-width="2.8" stroke-linecap="round" stroke-linejoin="round"/></svg>`;

interface Item {
  el: HTMLDivElement;
  lx: number;
  ly: number;
  shown: boolean;
  transform: string;
}

/** Where lines reverse: per station, its name, place and the lines (id, name), from the world. */
export function reversalsOf(w: WorldView): { station: string; name: string; lng: number; lat: number; lines: { id: string; name: string }[] }[] {
  const stations = new Map(w.save.stations.map((s) => [s.id, s]));
  const out = new Map<string, { station: string; name: string; lng: number; lat: number; lines: { id: string; name: string }[] }>();
  for (const l of w.save.lines) {
    for (const k of w.lineStats[l.id]?.reversals ?? []) {
      const s = stations.get(l.stops[k]);
      if (!s) continue;
      const at = out.get(s.id) ?? { station: s.id, name: s.name, lng: s.lng, lat: s.lat, lines: [] };
      if (!at.lines.some((x) => x.id === l.id)) at.lines.push({ id: l.id, name: l.name });
      out.set(s.id, at);
    }
  }
  return [...out.values()];
}

export class ReversalMarks {
  private box: HTMLDivElement;
  private tip: HTMLDivElement;
  private items: Item[] = [];
  private key = "";

  constructor(private map: MlMap, private overlay: Overlay) {
    const c = map.getContainer();
    this.box = document.createElement("div");
    this.box.className = "rev-marks";
    this.tip = document.createElement("div");
    this.tip.className = "tool-tag rev-tag";
    this.tip.style.display = "none";
    c.append(this.box, this.tip);
    overlay.afterMapFrame.push(() => this.place());
    map.on("movestart", () => (this.tip.style.display = "none"));
    effect(() => {
      // with a build tool on, presses on an icon reach the map underneath (as the capacity tags)
      this.box.classList.toggle("pass", tool.value !== "select");
    });
    effect(() => {
      const w = world.value;
      if (w) this.set(w);
    });
  }

  private set(w: WorldView) {
    const at = reversalsOf(w);
    const key = at.map((r) => `${r.station}:${r.lines.map((l) => l.id + "=" + l.name).join(",")}`).join("|");
    if (key === this.key) return;
    this.key = key;
    this.box.textContent = "";
    this.tip.style.display = "none";
    this.items = at.map((r) => {
      const el = document.createElement("div");
      el.className = "rev-mark";
      el.innerHTML = ICON;
      const words = [
        r.name,
        ...r.lines.map((l) => `${l.name} reverses here`),
        `${REVERSALS_AN_HOUR} trains an hour at most reverse here from each side`,
      ];
      el.onmouseenter = el.onmousemove = () => this.showTip(el, words);
      el.onmouseleave = () => (this.tip.style.display = "none");
      el.onclick = (e) => {
        e.stopPropagation();
        if (stopEditing.peek() || tool.peek() !== "select") return;
        selection.value = { kind: "line", line: r.lines[0].id };
      };
      this.box.appendChild(el);
      const [lx, ly] = toLocal(r.lng, r.lat);
      return { el, lx, ly, shown: true, transform: "" };
    });
    this.place();
  }

  private showTip(el: HTMLElement, lines: string[]) {
    this.tip.replaceChildren(...lines.map((s) => Object.assign(document.createElement("div"), { textContent: s })));
    const r = el.getBoundingClientRect(), c = this.map.getContainer().getBoundingClientRect();
    this.tip.style.transform = `translate(${Math.round(r.right - c.left + 4)}px, ${Math.round(r.top - c.top)}px)`;
    this.tip.style.display = "";
  }

  place() {
    if (!this.items.length || !this.overlay.camMatrix) return;
    const W = this.overlay.cssW, H = this.overlay.cssH;
    for (const it of this.items) {
      const [x, y] = this.overlay.toScreen(it.lx, it.ly);
      const on = x > -40 && y > -60 && x < W + 40 && y < H + 20;
      if (on !== it.shown) {
        it.el.style.display = on ? "" : "none";
        it.shown = on;
      }
      if (!on) continue;
      const t = `translate(${Math.round(x)}px, ${Math.round(y)}px)`;
      if (t !== it.transform) it.el.style.transform = it.transform = t;
    }
  }
}
