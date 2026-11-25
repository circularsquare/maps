// T-4048 (Anita, 11-24): while a passage is being placed (game/links.ts `linking`), a small button on each passage
// and connection drawn (a bridge between two platforms, from STATION_BOX_ZOOM where the bridges are drawn, and a BRT
// station's dashed join) removes it at a click (`removeJoin`). Each sits at its join's middle, else nudged along it or
// to one side of its middle, clear of the other buttons, the stations' names and the capacity dots; one that fits
// nowhere is left out. The bridges are read from
// the network the renderer draws, so a button is always on a drawn join.

import type { Map as MlMap } from "maplibre-gl";
import { effect } from "@preact/signals";
import { h, render } from "preact";
import { t } from "../i18n";
import { drawnJoins, linking, removeJoin } from "../game/links";
import { world } from "../game/world";
import { dotGroups } from "../game/complexView";
import { brtJoins } from "../game/busLinks";
import type { StationId } from "../game/types";
import { Icon } from "../ui/widgets";
import { anchor, toLocal } from "./geo";
import { BoxGrid } from "./labelLayout";
import { STATION_BOX_ZOOM } from "./stationGeometry";
import { marksLayer, onScreen, placeAt } from "./screenPlace";
import type { Overlay } from "./overlay";

/** a button's side, CSS px (`.join-x`) */
const SIDE = 18;
/** room kept round a button, px */
const PAD = 2;
/** where along its join a button may sit, tried in turn; then beside its middle, either side */
const ALONG = [0.5, 0.35, 0.65, 0.2, 0.8, 0.1, 0.9];
const BESIDE = [1, -1];

interface Join {
  a: StationId;
  b: StationId;
  /** its two ends, local units */
  ends: [number, number, number, number];
}

export class JoinButtons {
  private box: HTMLDivElement;
  private pool: HTMLButtonElement[] = [];
  private shown: Join[] = [];
  private joins: Join[] = [];
  /** every join drawn, those of hidden stations (levels, blueprints) too: the sides of a cut (`cutPairs`) */
  private all: [StationId, StationId][] = [];
  private seen: unknown[] = [];
  private scratch = [0, 0, 0, 0];
  /** what the buttons keep clear of besides each other, as boxes in CSS px: the stations' names
   * (map/stationLabels.ts `shownBoxes`) and the capacity dots */
  nameBoxes: () => number[][] = () => [];

  constructor(private map: MlMap, private overlay: Overlay) {
    this.box = marksLayer(map);
    this.box.classList.add("join-buttons");
    overlay.afterMapFrame.push(() => this.place());
    drawnJoins.list = () => {
      this.refresh();
      return this.all;
    };
    effect(() => {
      void linking.value;
      void world.value;
      this.place();
    });
  }

  /** The joins drawn now, found again when the network, the world, the draw anchor or what is hidden changes. */
  private refresh() {
    const rnd = this.overlay.renderer, net = rnd.net, w = world.peek(), groups = dotGroups.peek();
    const now = [net, w, groups, anchor().version, rnd.stationHid];
    if (now.every((v, i) => v === this.seen[i])) return;
    this.seen = now;
    this.joins = [];
    this.all = [];
    if (!net || !w) return;
    const add = (a: StationId, b: StationId, hidden: boolean, ends: () => Join["ends"]) => {
      this.all.push([a, b]);
      if (!hidden) this.joins.push({ a, b, ends: ends() });
    };
    const idx = net.stationLinkIdx, ends = net.stationLinks;
    for (let k = 0; idx && ends && k < idx.length / 2; k++) {
      const i = idx[k * 2], j = idx[k * 2 + 1];
      add(net.stationIds[i], net.stationIds[j], rnd.stationHidden(i) || rnd.stationHidden(j), () => [ends[k * 4], ends[k * 4 + 1], ends[k * 4 + 2], ends[k * 4 + 3]]);
    }
    const brt = brtJoins(w, groups.members.values());
    if (!brt.length) return;
    const at = new Map(net.stationIds.map((id, i) => [id, i]));
    const hidden = (id: StationId) => {
      const i = at.get(id);
      return i !== undefined && rnd.stationHidden(i);
    };
    for (const [x, y] of brt) {
      add(x.id, y.id, hidden(x.id) || hidden(y.id), () => {
        const [ax, ay] = toLocal(x.lng, x.lat), [bx, by] = toLocal(y.lng, y.lat);
        return [ax, ay, bx, by];
      });
    }
  }

  place() {
    let k = 0;
    if (linking.peek() !== null && this.map.getZoom() >= STATION_BOX_ZOOM) {
      this.refresh();
      const W = this.overlay.cssW, H = this.overlay.cssH, half = SIDE / 2 + PAD, p = this.scratch;
      const taken = new BoxGrid();
      for (const r of this.nameBoxes()) taken.add(r);
      for (const j of this.joins) {
        this.overlay.toScreenInto(j.ends[0], j.ends[1], p, 0);
        this.overlay.toScreenInto(j.ends[2], j.ends[3], p, 2);
        const dx = p[2] - p[0], dy = p[3] - p[1], len = Math.hypot(dx, dy);
        // the join's normal on the screen (straight down for one too short to have a direction)
        const [nx, ny] = len > 1 ? [-dy / len, dx / len] : [0, 1];
        const spots = [...ALONG.map((f) => [p[0] + dx * f, p[1] + dy * f]), ...BESIDE.map((s) => [p[0] + dx / 2 + nx * s * SIDE, p[1] + dy / 2 + ny * s * SIDE])];
        for (const [x, y] of spots) {
          if (!onScreen(x, y, W, H, -half)) continue;
          const r = [x - half, y - half, x + half, y + half];
          if (taken.hits(r)) continue;
          taken.add(r);
          this.shown[k] = j;
          placeAt(this.button(k++), x, y);
          break;
        }
      }
    }
    this.shown.length = k;
    for (let i = k; i < this.pool.length; i++) if (this.pool[i].style.display !== "none") this.pool[i].style.display = "none";
  }

  private button(k: number): HTMLButtonElement {
    let el = this.pool[k];
    if (!el) {
      el = this.pool[k] = document.createElement("button");
      el.className = "btn icon join-x";
      el.setAttribute("aria-label", t("stationInspector.removeLink"));
      render(h(Icon.cross, {}), el);
      // the press is the button's own: the map never sees it, so the passage being placed goes on
      for (const type of ["pointerdown", "mousedown", "dblclick"]) el.addEventListener(type, (e) => e.stopPropagation());
      el.addEventListener("click", (e) => {
        e.stopPropagation();
        const j = this.shown[k];
        if (j) void removeJoin(j.a, j.b);
      });
      this.box.appendChild(el);
    }
    if (el.style.display) el.style.display = "";
    return el;
  }
}
