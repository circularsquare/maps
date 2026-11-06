// Capacity markers (SPEC 6.2, T-031, T-1772): a round dot on every resource at 75% and up for the
// demand level in force, a gauge ringed yellow at 75% to red when overloaded (over 100%, or full
// with some line's trains waiting for room, T-517), its needle at the utilisation (T-1893).
// Nothing shows below 75% or zoom 12. Each dot sits on its resource's anchor; dots keep a little
// room between them and the worst delays win. Station names draw over them and never move them.
// A junction's dot is a link to the junction's inspector, where a flyover can be built (T-080).
// Hovering a dot says what it means (T-099). The dots sit under the demand views' canvas.

import { t, tNumber } from "../i18n";
import { effect } from "@preact/signals";
import { CAPACITY_KINDS } from "../game/capacityKinds";
import { duration } from "../game/format";
import type { Selection, WorldView } from "../game/types";
import { selection, tool } from "../game/ui";
import { hoverAt } from "./demandHover";
import { hoverHeld } from "../ui/blurHold";
import { toLocal } from "./geo";
import type { Overlay } from "./overlay";
import { perfOff } from "../perfFlags";
import type { Map as MlMap } from "maplibre-gl";
import { BoxGrid, MIN_LABEL_ZOOM, tipBeside } from "./labelLayout";
import { PlaceThrottle } from "./moving";

/** ?perfOff=markers (T-045, measurement only). */
const NO_MARKERS = perfOff("markers");
/** the dot's diameter, CSS px */
const DOT = 16;
/** room kept clear round a shown dot, CSS px, zoomed out and from zoom 14 */
const PAD_FAR = 6, PAD_NEAR = 3;
const DEMAND = ["high", "medium", "low"];

/** The dot's colour by utilisation: yellow at 75%, orange at 90%, red at 100% and over. */
const RAMP: [number, [number, number, number]][] = [[0.75, [232, 180, 37]], [0.9, [229, 122, 36]], [1, [204, 58, 44]]];
function dotColour(rho: number, over: boolean): string {
  if (over || rho >= 1) return rgbText(RAMP[2][1]);
  if (rho <= RAMP[0][0]) return rgbText(RAMP[0][1]);
  const i = rho < RAMP[1][0] ? 1 : 2;
  const [r0, a] = RAMP[i - 1], [r1, b] = RAMP[i], t = (rho - r0) / (r1 - r0);
  return rgbText([0, 1, 2].map((k) => a[k] + (b[k] - a[k]) * t) as [number, number, number]);
}
const rgbText = (c: [number, number, number]) => `rgb(${c.map(Math.round).join(",")})`;

const HAND_INK = "#262626";
const n2 = (v: number) => +v.toFixed(2);

/** T-1893: the dot as SVG, a gauge: a white disc ringed in the utilisation's colour, a three
 * quarter dial inside and a needle turned to the utilisation (full scale at 100%, so it points
 * lower right when full or over). */
export function capacityDot(rho: number, over: boolean): string {
  const col = dotColour(rho, over), r = 3.4;
  const a = (-135 + 270 * (over ? 1 : Math.min(1, rho))) * Math.PI / 180;
  const x = n2(r * Math.sin(a)), y = n2(-r * Math.cos(a));
  return `<svg viewBox="-8 -8 16 16" aria-hidden="true"><circle r="7.25" fill="#fff" stroke="var(--map-casing, #fff)" stroke-width="1.5"/>`
    + `<circle r="6.2" fill="none" stroke="${col}" stroke-width="2"/>`
    + `<path d="M-2.4 2.4A3.4 3.4 0 1 1 2.4 2.4" fill="none" stroke="${HAND_INK}" stroke-opacity="0.3" stroke-width="1" stroke-linecap="round"/>`
    + `<path d="M0 0L${x} ${y}" stroke="${HAND_INK}" stroke-width="1.5" stroke-linecap="round"/><circle r="1.1" fill="${HAND_INK}"/></svg>`;
}

/** The dot's hover: what it is at what utilisation, its trains an hour against the most it takes,
 * and each train's wait at the level in force. */
export function capacityTip(kind: number, rho: number, tph: number, waitS: number, level: number): string[] {
  const k = CAPACITY_KINDS[kind] ?? CAPACITY_KINDS[0];
  const tip = [t("capacityMarkers.use", { kind: k.short, load: rho > 1.005 ? ">100%" : `${tNumber(Math.round(rho * 100))}%` })];
  if (rho > 0) tip.push(t("capacityMarkers.tph", { now: tNumber(Math.round(tph)), max: tNumber(Math.round(tph / rho)) }));
  if (Math.round(waitS) >= 2) tip.push(DEMAND[level] ? t("capacityMarkers.slows", { demand: t("inspector.demand", { demand: DEMAND[level] }), wait: duration(waitS) }) : t("capacityMarkers.slowsThis", { wait: duration(waitS) }));
  return tip;
}

export class CapacityMarkers {
  /** Demand discs draw above these DOM dots; wired to the same select handler as the map. */
  demandClick: ((px: number, py: number, add: boolean) => Selection | undefined) | null = null;
  /** The select tool's click at a map point (px), for a click on a dot that opens nothing of its
   * own: the station, train or track under the dot is selected as if the dot were not there (T-256). */
  selectClick: ((px: number, py: number, add: boolean) => void) | null = null;
  private box: HTMLDivElement;
  private plane: HTMLDivElement;
  private items: { el: HTMLDivElement; x: number; y: number; shown?: boolean; transform?: string }[] = [];
  private visible = true;
  private placedMatrix: number[] | null = null;
  private placedSize = "";
  private throttle: PlaceThrottle;
  private key = "";

  private tip: HTMLDivElement;

  constructor(private container: HTMLElement, private overlay: Overlay, private map: MlMap) {
    this.box = document.createElement("div");
    this.box.className = "cap-marks";
    container.appendChild(this.box);
    this.plane = document.createElement("div");
    this.plane.className = "cap-plane";
    this.box.appendChild(this.plane);
    this.throttle = new PlaceThrottle(() => overlay.moving);
    this.tip = document.createElement("div");
    this.tip.className = "tool-tag demand-tag";
    this.tip.style.display = "none";
    container.appendChild(this.tip);
    // T-224: with any tool on, the dots take no pointer events, so every press, move and double
    // click reaches the map (a dot can sit on a station or track the tool wants)
    effect(() => {
      const pass = tool.value !== "select";
      this.box.classList.toggle("pass", pass);
      if (pass) this.tip.style.display = "none";
    });
    overlay.afterMapFrame.push(() => this.place());
    map.on("movestart", () => (this.tip.style.display = "none"));
  }

  set(w: WorldView, level: number) {
    // Keep the worst bottlenecks when several compete for the same screen space.
    const sorted = [...(w.capacity[level] ?? [])].sort((a, b) => b.delayS - a.delayS || b.rho - a.rho || a.kind - b.kind || a.x - b.x || a.y - b.y);
    const junctions = sorted.map((m) => (m.kind === 4 ? nearestJunction(w, m.x, m.y) : null));
    // most edits leave the dots as they were: keep them then, no rebuild and no layout (T-496)
    const key = level + "|" + sorted.map((m, i) => `${m.kind},${m.rho},${m.delayS},${m.tph},${m.held},${m.x},${m.y},${junctions[i]}`).join("|");
    if (key === this.key) return;
    this.key = key;
    this.plane.textContent = "";
    this.tip.style.display = "none";
    this.placedMatrix = null;
    this.items = sorted.map((m, i) => {
      const el = document.createElement("div");
      el.className = "cap-mark";
      el.innerHTML = capacityDot(m.rho, !!m.held);
      const tip = capacityTip(m.kind, m.rho, m.tph, m.delayS, level);
      const junction = m.kind === 4 ? junctions[i] : null;
      if (junction !== null) el.classList.add("link");
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
          // T-256: otherwise what is drawn under the dot (its station, a train, the track)
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
      this.plane.appendChild(el);
      return { el, x: m.x, y: m.y };
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
    if (size === this.placedSize && this.placedMatrix?.every((v, i) => v === m[i])) return;
    // T-1614: between full placements every shown dot follows its resource each frame; which dots
    // show is worked out again only when due
    if (!this.throttle.due()) return this.follow();
    this.throttle.done();
    this.placedMatrix = Array.from(m);
    this.placedSize = size;
    const pad = zoom < 14 ? PAD_FAR : PAD_NEAR, half = DOT / 2;
    const taken = new BoxGrid();
    for (const it of this.items) {
      const [x, y] = this.overlay.toScreen(it.x, it.y);
      const tx = Math.round(x) - half, ty = Math.round(y) - half;
      let on = Number.isFinite(tx + ty) && !(tx + DOT < 0 || tx > W || ty + DOT < 0 || ty > H); // NaN: far side of the globe (T-1777)
      if (on) {
        const r = [tx - pad, ty - pad, tx + DOT + pad, ty + DOT + pad];
        if (taken.hits(r)) on = false;
        else taken.add(r);
      }
      // only what changed is written (T-689)
      if (it.shown !== on) it.el.style.visibility = on ? "" : "hidden";
      it.shown = on;
      if (!on) continue;
      const transform = `translate(${tx}px, ${ty}px)`;
      if (transform !== it.transform) it.el.style.transform = it.transform = transform;
    }
  }

  private follow() {
    const half = DOT / 2;
    for (const it of this.items) {
      if (!it.shown) continue;
      const [x, y] = this.overlay.toScreen(it.x, it.y);
      if (!Number.isFinite(x + y)) {
        it.el.style.visibility = "hidden";
        it.shown = false;
        continue;
      }
      const transform = `translate(${Math.round(x) - half}px, ${Math.round(y) - half}px)`;
      if (transform !== it.transform) it.el.style.transform = it.transform = transform;
    }
  }
}

/** The junction a marker at (x, y) (local units) belongs to: the nearest node where three or
 * more tracks meet. */
export function nearestJunction(w: WorldView, x: number, y: number): number | null {
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
