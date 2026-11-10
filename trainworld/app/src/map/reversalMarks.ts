// Reversal icons (T-797, docs/ui-map.md): a U-turn sign over every station or bare track end
// where a line's trains reverse partway, in the built flat junction badge's style and size (T-1772),
// larger with a placing tool on. It floats just above the station, clear of the stop editing
// marker on it, and hides with the stations below STATIONS_ZOOM. Hovering it names the lines that
// reverse there; a click selects the first of them (not while stops are edited).

import { t } from "../i18n";
import { effect } from "@preact/signals";
import type { Map as MlMap } from "maplibre-gl";
import type { WorldView } from "../game/types";
import { selection, stopEditing, tool } from "../game/ui";
import { world } from "../game/world";
import { anchorVersion } from "./anchorVersion";
import { anchor, toLocal } from "./geo";
import { tipBeside } from "./labelLayout";
import type { Overlay } from "./overlay";
import { quietBadgeRadius, STATIONS_ZOOM } from "./stationGeometry";

const PLACING = new Set<string>(["track", "station", "line", "blueprint", "bus", "ferry", "building"]);
/** how much larger the icon is with a placing tool on (T-1581) */
const PLACING_SCALE = 1.6;

const ICON =`<svg viewBox="-4 -4 32 32" aria-hidden="true"><path d="M7.5 19.5V10a4.75 4.75 0 0 1 9.5 0v7" fill="none" stroke="currentColor" stroke-width="3.2" stroke-linecap="round"/><path d="M12.8 13.4l4.2 4.3 4.2-4.3" fill="none" stroke="currentColor" stroke-width="3.2" stroke-linecap="round" stroke-linejoin="round"/></svg>`;

interface Item {
  el: HTMLDivElement;
  lx: number;
  ly: number;
  shown: boolean;
  transform: string;
}

type Reversal = { station: string; name: string; lng: number; lat: number; lines: { id: string; name: string }[] };

/** Where lines reverse: per station or bare track end (T-797 B3, keyed `end:<node>`), its name,
 * place and the lines (id, name), from the world. */
export function reversalsOf(w: WorldView): Reversal[] {
  const stations = new Map(w.save.stations.map((s) => [s.id, s]));
  const out = new Map<string, Reversal>();
  const add = (key: string, name: string, lng: number, lat: number, l: { id: string; name: string }) => {
    const at = out.get(key) ?? { station: key, name, lng, lat, lines: [] };
    if (!at.lines.some((x) => x.id === l.id)) at.lines.push({ id: l.id, name: l.name });
    out.set(key, at);
  };
  for (const l of w.save.lines) {
    // a bus turns at its ends: no reversal to mark
    if (w.lineStats[l.id]?.bus) continue;
    for (const k of w.lineStats[l.id]?.reversals ?? []) {
      const s = stations.get(l.stops[k]);
      if (s) add(s.id, s.name, s.lng, s.lat, l);
    }
    for (const n of w.lineStats[l.id]?.endTurns ?? []) {
      const node = w.nodes.get(n);
      if (node) add(`end:${n}`, "Track end", node.lng, node.lat, l);
    }
  }
  return [...out.values()];
}

export class ReversalMarks {
  private box: HTMLDivElement;
  private tip: HTMLDivElement;
  private items: Item[] = [];
  private key = "";
  private on = true;
  private big = false;
  private size = "";

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
      const pass = tool.value !== "select";
      this.box.classList.toggle("pass", pass);
      // larger while a placing tool is on (T-1581)
      this.big = PLACING.has(tool.value);
      if (pass) this.tip.style.display = "none";
      this.place();
    });
    effect(() => {
      anchorVersion.value;
      const w = world.value;
      if (w) this.set(w);
    });
  }

  private set(w: WorldView) {
    const at = reversalsOf(w);
    // the icons' places are in local units: a new draw anchor places them again (T-1772)
    const key = anchor().version + "#" + at.map((r) => `${r.station}:${r.lines.map((l) => l.id + "=" + l.name).join(",")}`).join("|");
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
        ...r.lines.map((l) => t("reversalMarks.reverses", { name: l.name })),
      ];
      el.onmouseenter = el.onmousemove = () => tipBeside(this.tip, el, this.map.getContainer(), words, "right");
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

  place() {
    const zoom = this.map.getZoom(), on = zoom >= STATIONS_ZOOM;
    if (on !== this.on) {
      this.on = on;
      this.box.style.display = on ? "" : "none";
      if (!on) this.tip.style.display = "none";
    }
    if (!on || !this.items.length || !this.overlay.camMatrix) return;
    const size = `${Math.round(2 * quietBadgeRadius(zoom) * (this.big ? PLACING_SCALE : 1) * 2) / 2}px`;
    if (size !== this.size) this.box.style.setProperty("--rs", (this.size = size));
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
