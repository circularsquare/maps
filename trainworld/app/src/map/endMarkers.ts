// Point markers on the map (T-416). While the track or station tool is on, a round marker at each
// loose end (a node with one piece of track and no station), blueprint or built: where a line is
// continued or a station goes at the end of one. With the select tool, a square marker at every
// blueprint point that can be dragged (map/nodeDrag.ts): track ends, corners and junctions. They
// take no presses; the tools and the node drag work on the nodes underneath as before.

import type { Map as MlMap } from "maplibre-gl";
import { tool } from "../game/ui";
import { world } from "../game/world";
import { effect } from "@preact/signals";
import { toLocal } from "./geo";
import { live } from "./live";
import type { Overlay } from "./overlay";

type Kind = "loose" | "drag";

/** below this zoom the markers would only crowd the map */
const MIN_ZOOM = 9;
const MAX_MARKS = 400;

export class EndMarkers {
  private box: HTMLDivElement;
  private marks: HTMLDivElement[] = [];
  private pts: { lx: number; ly: number }[] = [];
  private kind: Kind | null = null;
  private version = -1;

  constructor(private map: MlMap, private overlay: Overlay) {
    this.box = document.createElement("div");
    this.box.className = "tool-marks";
    map.getContainer().appendChild(this.box);
    overlay.afterMapFrame.push(() => this.place());
    effect(() => {
      tool.value;
      world.value;
      this.place();
    });
  }

  private wanted(): Kind | null {
    const t = tool.peek();
    return t === "track" || t === "station" ? "loose" : t === "select" ? "drag" : null;
  }

  /** The points to mark, found again only when the tool or the world changes. */
  private refresh(kind: Kind) {
    const w = world.peek();
    const v = w?.version ?? -1;
    if (kind === this.kind && v === this.version) return;
    this.kind = kind;
    this.version = v;
    this.pts = [];
    if (!w) return;
    for (const n of w.nodes.values()) {
      if (n.platform > 0) continue;
      if (kind === "loose" ? n.ports !== 1 : n.builtPorts !== 0) continue;
      const [lx, ly] = toLocal(n.lng, n.lat);
      this.pts.push({ lx, ly });
    }
  }

  place() {
    const kind = this.wanted();
    let k = 0;
    if (kind && this.map.getZoom() >= MIN_ZOOM && !live.nodeDrag?.dragging) {
      this.refresh(kind);
      const el = this.map.getContainer();
      const w = el.clientWidth, h = el.clientHeight;
      for (const p of this.pts) {
        const [x, y] = this.overlay.toScreen(p.lx, p.ly);
        if (x < -8 || y < -8 || x > w + 8 || y > h + 8) continue;
        if (k >= MAX_MARKS) break;
        let m = this.marks[k];
        if (!m) {
          m = this.marks[k] = document.createElement("div");
          this.box.appendChild(m);
        }
        const cls = kind === "loose" ? "end-mark" : "pi-mark";
        if (m.className !== cls) m.className = cls;
        m.style.display = "";
        m.style.transform = `translate(${Math.round(x)}px, ${Math.round(y)}px)`;
        k++;
      }
    }
    for (let i = k; i < this.marks.length; i++) this.marks[i].style.display = "none";
  }
}
