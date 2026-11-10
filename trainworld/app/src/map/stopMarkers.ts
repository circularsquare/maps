// Stop editing on the map (T-233, docs/ui-building.md): while a line's stops are being edited every
// station carries a round toggle marker over everything else. A stop is a disc in the line's
// colour inside a white ring inside a dark one (no check, T-239); the others are hollow, with no
// line colour at all (a plain grey ring, T-476). Stations a stop can be added at without changing
// the line's path (on it, or past an end on track it can carry on over, T-237) are drawn on top, a
// little smaller than the stops. The tiers are the clock worker's, from the line tool's own search
// (`TrackApi::stop_tiers`). The rest sit behind, smaller and greyer still: either joined to the
// route by track (a click adds the stop and the route changes) or not joined at all (a click says
// why it cannot be added).
//
// The markers are plain DOM over the map and take no pointer events: the map's own click and
// mousemove (main.ts -> map/tools.ts) call `pick`, `hover` and `activate` here, so the wheel and
// dragging still reach the map under a marker. A click adds or removes a stop and nothing else; a right
// click, or a finger held on a marker, closes or opens a loop or turns the line round one (T-3109). Everything else the mode changes (the demand views
// hidden, other lines dimmed) is a class on the page and one renderer option (main.ts, style.css).

import { t } from "../i18n";
import { effect } from "@preact/signals";
import type { Map as MlMap } from "maplibre-gl";
import type { LineInput, WorldView } from "../game/types";
import { lineDraft, newLine, stopEditing } from "../game/ui";
import { world } from "../game/world";
import { turnedLine } from "../game/turnedLine";
import { queryClock } from "../workers/clockClient";
import { GapFlash } from "./gapFlash";
import { tagLines } from "./labelLayout";
import type { Overlay } from "./overlay";
import { hot, onHot } from "./hovered";
import { touchInput } from "./touch";

/** How a station stands to the line being edited. */
export type StopTier =
  /** the line stops here */
  | "on"
  /** adding it keeps the line's path: on track the route passes over, or past an end (T-237) */
  | "route"
  /** joined to the route by track, off it: adding it changes the route */
  | "reroute"
  /** no track joins it to the line */
  | "cut"
  /** adding it makes the line's trains reverse at a station where they did not (T-797) */
  | "reverse"
  /** the first stop, where a right click closes the line into a loop (T-798) */
  | "close"
  /** a loop's first stop, where a right click opens it */
  | "open"
  /** the stop before a loop of track at an end of the line, where a right click turns the line round it (T-3021) */
  | "turn"
  /** that stop on a line so turned, where a right click opens it again */
  | "unturn";

/** Marker diameter by zoom, CSS px. */
const SIZE: [number, number][] = [[9, 9], [12, 14], [14, 20], [16, 24]];
/** Reachable stations on the route are this much of a stop's marker (T-239), the ones behind them
 * this much (style.css repeats both). */
export const NEAR_SCALE = 0.8;
const FAR_SCALE = 0.65;
const REASON_MS = 4000;
/** T-3109: a finger held this long on a marker is a right click; it may move this far, CSS px */
const LONG_PRESS_MS = 500;
const PRESS_SLOP_PX = 8;
/** the tiers whose right click does something */
const LOOP_TIERS: ReadonlySet<StopTier> = new Set(["close", "open", "turn", "unturn"]);

export function sizeAt(zoom: number): number {
  if (zoom <= SIZE[0][0]) return SIZE[0][1];
  for (let i = 1; i < SIZE.length; i++) {
    if (zoom <= SIZE[i][0]) {
      const [z0, s0] = SIZE[i - 1], [z1, s1] = SIZE[i];
      return s0 + ((s1 - s0) * (zoom - z0)) / (z1 - z0);
    }
  }
  return SIZE[SIZE.length - 1][1];
}

/**
 * The tiers as the clock worker works them out (`TrackApi::stop_tiers`, T-237): per station node
 * its tier, 0 on, 1 route, 2 reroute, 3 cut, 4 reverse, 5 close, 6 open, 7 turn, 8 unturn, and for 4
 * the station trains would reverse at (T-797), from the same search a click makes, so a station past
 * an end that the line can carry on to is on route.
 */
function tiersFrom(d: Float64Array, w: WorldView, revAt: Map<string, string>): Map<string, StopTier> {
  const names: StopTier[] = ["on", "route", "reroute", "cut", "reverse", "close", "open", "turn", "unturn"];
  const byNum = new Map(w.save.stations.map((s) => [s.num, s.id]));
  const out = new Map<string, StopTier>();
  revAt.clear();
  for (let i = 0; i + 2 < d.length; i += 3) {
    const id = byNum.get(d[i]);
    if (id === undefined) continue;
    out.set(id, names[d[i + 1]] ?? "cut");
    // a node that is no station is a bare track end (T-797 B3): ""
    if (d[i + 1] === 4) revAt.set(id, byNum.get(d[i + 2]) ?? "");
  }
  return out;
}

/**
 * A first estimate of each station's tier for a line, shown until the clock worker's answer
 * (`tiersFrom`) lands: the line's own stops, then the stations on nodes its route passes (the
 * edges that list the line), then the stations on the same track network as its stops, and the
 * rest. A line with no route yet (fewer than two stops) has nothing to stay on, so what joins its
 * stops is as good as on route; a line with no stops takes any station.
 */
function stopTiers(w: WorldView, line: LineInput): Map<string, StopTier> {
  const out = new Map<string, StopTier>();
  const stops = new Set(line.stops);
  const adj = new Map<number, number[]>();
  const routeNodes = new Set<number>();
  for (const e of w.edges) {
    (adj.get(e.a) ?? adj.set(e.a, []).get(e.a)!).push(e.b);
    (adj.get(e.b) ?? adj.set(e.b, []).get(e.b)!).push(e.a);
    if (e.lines.includes(line.id)) {
      routeNodes.add(e.a);
      routeNodes.add(e.b);
    }
  }
  // the track network the stops lie on
  const joined = new Set<number>();
  const queue = line.stops.map(Number);
  for (const n of queue) joined.add(n);
  for (let k = 0; k < queue.length; k++)
    for (const m of adj.get(queue[k]) ?? []) if (!joined.has(m)) (joined.add(m), queue.push(m));
  const hasRoute = line.stops.length >= 2 && routeNodes.size > 0;
  const n = line.stops.length;
  const loop = n >= 3 && line.stops[0] === line.stops[n - 1];
  for (const s of w.save.stations) {
    if (loop && s.id === line.stops[0]) out.set(s.id, "open");
    else if (stops.has(s.id)) out.set(s.id, "on");
    else if (!line.stops.length) out.set(s.id, "route");
    else if (hasRoute) out.set(s.id, routeNodes.has(s.num) ? "route" : joined.has(s.num) ? "reroute" : "cut");
    else out.set(s.id, joined.has(s.num) ? "route" : "cut");
  }
  return out;
}

interface Mark {
  el: HTMLDivElement;
  cls: string;
  /** on screen at the last `place`: centre and diameter, CSS px */
  x: number;
  y: number;
  d: number;
  shown: boolean;
}

/** In the order a click prefers when markers overlap: the stops, then the ones that can be added. */
const RANK: Record<StopTier, number> = { on: 3, close: 3, open: 3, turn: 3, unturn: 3, route: 2, reroute: 1, reverse: 1, cut: 0 };

export class StopMarkers {
  private box: HTMLDivElement;
  private tag: HTMLDivElement;
  private marks = new Map<string, Mark>();
  private tiers = new Map<string, StopTier>();
  private names = new Map<string, string>();
  private line: LineInput | null = null;
  private colour = "#888";
  /** T-3041: the stops of a turned line served one way, with the stops either side of them on the walk */
  private oneWay = new Map<string, [string, string]>();
  private index = new Map<string, number>();
  private netSeen: unknown = null;
  private size = 0;
  private reasonTimer = 0;
  private reasonFor: string | null = null;
  /** the last tiers asked of the clock worker; an older answer is dropped */
  private asked = 0;
  /** the clock worker's last tiers for the line being edited */
  private fromWorker: Map<string, StopTier> | null = null;
  /** for a "reverse" station, the station trains would reverse at, "" for a bare track end */
  private revAt = new Map<string, string>();
  private tram = false;
  private seenWorld: WorldView | null = null;
  private seenLine: number | null = null;
  private gap: GapFlash;
  /** the last gap asked for; an older answer is dropped */
  private gapAsked = 0;

  /** T-3109: a finger down on a marker, until it is held long enough, moves or lifts */
  private press: { id: string; x: number; y: number; timer: number } | null = null;
  /** the tap a long press leaves behind is not a click on the marker */
  private swallow = false;

  constructor(private map: MlMap, private overlay: Overlay, private toggle: (station: number) => void, private loop: (station: number, tier: StopTier) => void) {
    this.gap = new GapFlash(map, overlay);
    this.box = document.createElement("div");
    this.box.className = "stop-marks";
    this.box.style.display = "none";
    this.tag = document.createElement("div");
    this.tag.className = "tool-tag stop-tag";
    this.tag.style.display = "none";
    map.getContainer().append(this.box, this.tag);
    overlay.afterMapFrame.push(() => this.place());
    onHot(() => this.place());
    map.on("contextmenu", (e) => {
      if (!this.line) return;
      e.originalEvent.preventDefault();
      // a held finger makes one too: its own long press has dealt with it
      if (touchInput.peek()) return;
      const id = this.pick(e.point.x, e.point.y);
      if (id !== null) this.loopAt(id, e.point.x, e.point.y);
    });
    const box = map.getContainer();
    box.addEventListener("pointerdown", (e) => this.pressDown(e), true);
    addEventListener("pointermove", (e) => {
      const p = this.press;
      if (!p || e.pointerType !== "touch") return;
      const r = box.getBoundingClientRect();
      if (Math.hypot(e.clientX - r.left - p.x, e.clientY - r.top - p.y) > PRESS_SLOP_PX) this.pressEnd();
    });
    addEventListener("pointerup", () => {
      this.pressEnd();
      if (this.swallow) window.setTimeout(() => (this.swallow = false), 400);
    });
    addEventListener("pointercancel", () => this.pressEnd());
    effect(() => {
      const on = stopEditing.value;
      const id = lineDraft.value.line;
      const w = world.value;
      document.documentElement.classList.toggle("stop-editing", on);
      this.box.style.display = on ? "" : "none";
      if (!on || id === null || !w) {
        this.line = null;
        this.seenLine = null;
        this.fromWorker = null;
        this.tiers = new Map();
        this.hideTag();
        this.gapAsked++;
        this.gap.clear();
        this.overlay.invalidate();
        return;
      }
      // lineDraft is set again after every stop edit with the same line: nothing to do then
      if (w === this.seenWorld && id === this.seenLine && this.line) return;
      const sameLine = id === this.seenLine && this.line !== null;
      this.seenWorld = w;
      this.seenLine = id;
      this.line = w.save.lines.find((l) => l.num === id) ?? null;
      if (this.line && w.lineStats[this.line.id]?.bus) this.line = null; // a bus line's are map/busStops.ts's (T-2891)
      if (!this.line) return;
      this.colour = this.line.colour;
      this.tram = !!w.lineStats[this.line.id]?.tram;
      this.oneWay = new Map();
      const turned = turnedLine(this.line.stops, w.lineStats[this.line.id] ?? { loop: null, oneWay: false });
      const ls = this.line.stops, lm = ls.length - 1;
      for (const r of turned?.rows ?? []) if (r.once && r.occ[0] >= 0) this.oneWay.set(ls[r.i], [ls[(r.i + lm - 1) % lm], ls[(r.i + 1) % lm]]);
      this.box.style.setProperty("--c", this.colour);
      this.names = new Map(w.save.stations.map((s) => [s.id, s.name]));
      // After an edit to the same line the worker's last tiers stand until its new answer, with only
      // the stops changed; the estimate disagrees with them past the line's ends, and showing it in
      // between made markers change and change back (T-648).
      const estimate = stopTiers(w, this.line);
      if (sameLine && this.fromWorker) {
        const stops = new Set(this.line.stops);
        for (const s of estimate.keys()) {
          const was = this.fromWorker.get(s);
          if (!stops.has(s) && was !== undefined && was !== "on") estimate.set(s, was);
        }
      } else this.fromWorker = null;
      this.tiers = estimate;
      // the clock worker's tiers, from the line tool's own search (T-237), replace this estimate
      const asked = ++this.asked;
      void queryClock({ what: "stopTiers", line: id }).then((d) => {
        if (!d || asked !== this.asked || world.peek() !== w || !this.line) return;
        this.tiers = this.fromWorker = tiersFrom(d, w, this.revAt);
        this.place();
        this.overlay.invalidate();
      });
      this.place();
      // the network the renderer holds follows the world a moment later
      requestAnimationFrame(() => this.place());
      this.overlay.invalidate();
    });
  }

  private tierOf(id: string): StopTier {
    return this.tiers.get(id) ?? "cut";
  }

  /** Move the markers to the camera: after each MapLibre frame and each change. */
  place() {
    if (!this.line) return;
    const n = this.overlay.renderer.net;
    if (!n || !this.overlay.camMatrix) return;
    if (n !== this.netSeen) {
      this.netSeen = n;
      this.index = new Map(n.stationIds.map((id, i) => [id, i]));
    }
    const size = Math.round(sizeAt(this.map.getZoom()));
    if (size !== this.size) {
      this.size = size;
      this.box.style.setProperty("--ms", size + "px");
    }
    const pad = size;
    const lit = new Set(hot().stations);
    for (const [id, tier] of this.tiers) {
      const i = this.index.get(id);
      let m = this.marks.get(id);
      let on = false, x = 0, y = 0;
      if (i !== undefined) {
        // the platform's middle, as the station is drawn
        const mx = n.stationShape?.[i * 4] ?? 0, my = n.stationShape?.[i * 4 + 1] ?? 0;
        [x, y] = this.overlay.toScreen(n.stations[i * 3] + mx, n.stations[i * 3 + 1] + my);
        on = x > -pad && y > -pad && x < this.overlay.cssW + pad && y < this.overlay.cssH + pad;
      }
      if (!on) {
        if (m?.shown) {
          m.el.style.display = "none";
          m.shown = false;
        }
        continue;
      }
      if (!m) {
        const el = document.createElement("div");
        el.setAttribute("aria-hidden", "true");
        m = { el, cls: "", x: 0, y: 0, d: 0, shown: false };
        this.marks.set(id, m);
        this.box.appendChild(el);
      }
      // a stop that closes, opens or turns a loop is drawn as the stop it is
      const stop = tier === "on" || tier === "close" || tier === "open" || tier === "turn" || tier === "unturn";
      const cls = "stop-mark " + (stop ? "on" : tier) + (lit.has(id) ? " hot" : "");
      if (m.cls !== cls) {
        m.cls = cls;
        m.el.className = cls;
      }
      if (!m.shown) {
        m.el.style.display = "";
        m.shown = true;
      }
      m.x = x;
      m.y = y;
      m.d = stop ? size : tier === "route" ? size * NEAR_SCALE : size * FAR_SCALE;
      m.el.style.transform = `translate(${Math.round(x)}px, ${Math.round(y)}px)`;
      m.el.style.background = stop ? this.halfFill(id, n, x, y) : "";
    }
    // stations that left the list (an undo): their markers go
    for (const [id, m] of this.marks)
      if (!this.tiers.has(id)) {
        m.el.remove();
        this.marks.delete(id);
      }
  }

  /** T-3041: a stop served one way is filled on one half only, split along the track, the half on the right
   * of the way the train goes (from the stop before to the stop after); "" for any other marker. */
  private halfFill(id: string, n: NonNullable<Overlay["renderer"]["net"]>, x: number, y: number): string {
    const nb = this.oneWay.get(id);
    if (!nb) return "";
    const at = (s: string): [number, number] | null => {
      const i = this.index.get(s);
      return i === undefined ? null : this.overlay.toScreen(n.stations[i * 3], n.stations[i * 3 + 1]);
    };
    const a = at(nb[0]), b = at(nb[1]);
    if (!a || !b) return "";
    const dx = b[0] - a[0], dy = b[1] - a[1];
    if (!dx && !dy) return "";
    // the right-hand normal on screen (y down) is (-dy, dx); a CSS gradient angle points (sin, -cos)
    const deg = (Math.atan2(-dy, -dx) * 180) / Math.PI;
    return `linear-gradient(${deg.toFixed(1)}deg, #fff 50%, var(--c) 50%)`;
  }

  /** The station whose marker is under a map point (CSS px), preferring stops, then the ones that
   * can be added, then, for a new line, stations no trains stop at yet (T-1899), then the nearest. */
  pick(px: number, py: number): string | null {
    if (!this.line) return null;
    const served = newLine.peek()?.line === this.line.num ? this.served() : null;
    // rank, then 1 for a station no trains stop at on a new line, then nearness
    let best: string | null = null, bestRank = -1, bestD = Infinity;
    for (const [id, m] of this.marks) {
      if (!m.shown) continue;
      const d = Math.hypot(m.x - px, m.y - py);
      if (d > m.d / 2 + 2) continue;
      const rank = RANK[this.tierOf(id)] * 2 + (served && !served.has(id) ? 1 : 0);
      if (rank > bestRank || (rank === bestRank && d < bestD)) [best, bestRank, bestD] = [id, rank, d];
    }
    return best;
  }

  private servedFor: { w: WorldView; ids: Set<string> } | null = null;
  /** The stations some other line's running trains stop at. */
  private served(): Set<string> {
    const w = world.peek();
    if (!w) return new Set();
    if (this.servedFor?.w !== w) {
      const ids = new Set<string>();
      for (const l of w.save.lines) {
        const st = w.lineStats[l.id];
        if (l.num !== this.line?.num && st?.status === "running" && st.trainsNeeded.some((t) => t > 0)) for (const s of l.stops) ids.add(s);
      }
      this.servedFor = { w, ids };
    }
    return this.servedFor.ids;
  }

  /** What hovering a marker says: a stop's click, then its right click where that does something (T-3109). */
  private words(id: string): string[] {
    const name = this.names.get(id) ?? t("stopMarkers.station");
    switch (this.tierOf(id)) {
      case "on": return [name, t("stopMarkers.remove")];
      case "close": return [name, t("stopMarkers.remove"), t("stopMarkers.rightClose")];
      case "open": return [name, t("stopMarkers.remove"), t("stopMarkers.rightOpen")];
      case "turn": return [name, t("stopMarkers.remove"), t("stopMarkers.rightLoopBack")];
      case "unturn": return [name, t("stopMarkers.remove"), t("stopMarkers.rightStopLooping")];
      case "route": return [name, t("stopMarkers.add")];
      case "reroute": return [name, t("stopMarkers.addReroute")];
      case "reverse": {
        const at = this.revAt.get(id);
        return [name, at === "" ? t("stopMarkers.reverseEnd", { tram: this.tram }) : this.names.get(at ?? "") ? t("stopMarkers.reverseAt", { name: this.names.get(at ?? ""), tram: this.tram }) : t("stopMarkers.reverseSome", { tram: this.tram })];
      }
      default: return [name, t("stopMarkers.noTrack")];
    }
  }

  private showTag(lines: string[], px: number, py: number, bad = false) {
    this.tag.className = "tool-tag stop-tag" + (bad ? " bad" : "");
    tagLines(this.tag, lines);
    this.tag.style.display = "";
    this.tag.style.transform = `translate(${Math.round(px + 14)}px, ${Math.round(py + 12)}px)`;
  }

  private hideTag() {
    this.tag.style.display = "none";
    this.reasonFor = null;
    window.clearTimeout(this.reasonTimer);
  }

  /** The pointer moved over the map: the hover tag, and the pointer cursor over a marker. */
  hover(px: number, py: number) {
    const id = this.pick(px, py);
    this.map.getCanvas().style.cursor = id ? "pointer" : "crosshair";
    if (id === null) {
      if (this.reasonFor === null) this.hideTag();
      return;
    }
    if (this.reasonFor !== null && this.reasonFor !== id) this.hideTag();
    if (this.reasonFor === null) this.showTag(this.words(id), px, py);
  }

  /** A click on the marker of station `id`: the stop added or taken out. */
  activate(id: string, px: number, py: number) {
    if (this.swallow) return void (this.swallow = false);
    if (this.tierOf(id) === "cut") {
      const name = this.names.get(id) ?? t("stopMarkers.thatStation");
      this.hideTag();
      this.reasonFor = id;
      this.showTag([t("stopMarkers.cutOff", { name }), t("stopMarkers.drawFirst")], px, py, true);
      this.reasonTimer = window.setTimeout(() => this.hideTag(), REASON_MS);
      this.showGap(id);
      return;
    }
    this.hideTag();
    this.toggle(Number(id));
  }

  /** T-3109: a right click (or a long press) on the marker of station `id`: the loop there closed, opened or
   * turned round, or a word that there is none. */
  private loopAt(id: string, px: number, py: number) {
    const tier = this.tierOf(id);
    this.hideTag();
    if (LOOP_TIERS.has(tier)) return this.loop(Number(id), tier);
    this.reasonFor = id;
    this.showTag([this.names.get(id) ?? t("stopMarkers.station"), t("stopMarkers.noLoop")], px, py, true);
    this.reasonTimer = window.setTimeout(() => this.hideTag(), REASON_MS);
  }

  private pressDown(e: PointerEvent) {
    if (e.pointerType !== "touch" || !this.line || !e.isPrimary) return this.pressEnd();
    const r = this.map.getContainer().getBoundingClientRect();
    const x = e.clientX - r.left, y = e.clientY - r.top;
    const id = this.pick(x, y);
    this.pressEnd();
    if (id === null) return;
    const timer = window.setTimeout(() => {
      this.press = null;
      this.swallow = true;
      navigator.vibrate?.(15);
      this.loopAt(id, x, y);
    }, LONG_PRESS_MS);
    this.press = { id, x, y, timer };
  }

  private pressEnd() {
    if (this.press) window.clearTimeout(this.press.timer);
    this.press = null;
  }

  /** T-1896: where the line's track and station `id`'s are nearest, drawn for a few seconds. */
  private showGap(id: string) {
    if (!this.line?.stops.length) return;
    const asked = ++this.gapAsked, line = this.line.num;
    void queryClock({ what: "stopGap", line, station: Number(id) }).then((d) => {
      if (asked !== this.gapAsked || this.line?.num !== line) return;
      if (d && d.length === 4) this.gap.show([d[0], d[1]], [d[2], d[3]]);
      if (DEBUG) (window as unknown as { twGap?: number[] | null }).twGap = d && d.length === 4 ? Array.from(d) : null;
    });
  }
}

const DEBUG = new URLSearchParams(location.search).get("debug") === "1";
