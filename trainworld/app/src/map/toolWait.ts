// T-4214: a map click made while an edit sent before it has not reached the map the tools pick on (map/clickMap.ts and
// `world`: patched with the edit's word, or set with its network when it came with no patch) waits, the tool's tag showing
// the spinner after WAIT_MS, and is carried out at the same map spot once that map holds every edit sent before it. A tool switch the player makes
// meanwhile waits its turn behind them (game/ui.ts `inClickOrder`), so each click goes with the tool it was made with.
// Dropped if the spot has left the screen, on Escape or a load; after CLICK_WAIT_MAX_MS it is carried out on the map as it
// is. The next
// segment of a track run never waits (stage D: it starts from the end of the one before, which the worker reads as that
// edit's end node), but a click (a tap, the touch bar's confirm) on the run's own track the map lacks yet does
// (`Tools.ownTrackOut`, T-4311), and the line tool's stop clicks never count as edits out (T-4114). A click counts as out
// from the moment it is made, so one whose tool awaits before it sends (a delete's question) holds the clicks after it
// too; a station sends at once and its name after it (stage A).
// notes/T-4214.md "Waiting clicks".

import type { MapMouseEvent } from "maplibre-gl";
import { setSwitchGate, tool } from "../game/ui";
import { editsOut, outDrop, setOutListener } from "../game/pending";
import { clicksMoved, setClicksBusy } from "../game/world";
import { type LngLatish, WAIT_MS } from "./toolParts";
import { evAt, touchConfirm } from "./toolTouch";
import { showTag } from "./toolMarks";
import { namesPending } from "./stationTool";
import type { Tools } from "./tools";

/** A click waits for the map at most this long, ms, then is carried out on the map as it is (a backstop, so the worst case
 * is a click as it went before T-4214) */
const CLICK_WAIT_MAX_MS = 5000;

/** A click waiting, or a tool switch made behind one (`run`): how it came, the tool it was made with (null behind a
 * waiting switch: the tool that switch leaves), its map spot (null: the touch bar's confirm with no touch point) and keys,
 * when it was made and since when it has been the first waiting */
export interface WaitingClick {
  how: "click" | "tap" | "confirm" | "switch";
  run?: () => void;
  tool: string | null;
  at: LngLatish | null;
  shift: boolean;
  ctrl: boolean;
  meta: boolean;
  detail: number;
  made: number;
  first: number;
  /** a double click's second click came while it waited: it ends the track run after it (T-445) */
  double: boolean;
  /** the worker was asked for a network it holds back (T-2404) */
  asked: boolean;
}

/** a waiting tool switch is being carried out: the switches it makes itself go at once */
let switching = false;

export function startClickWait(tk: Tools) {
  setOutListener((dropped) => (dropped ? dropClicks(tk) : queueMicrotask(() => tryClicks(tk))));
  setSwitchGate((run) => {
    if (switching || !tk.waitingClicks.length) return false;
    const now = performance.now();
    tk.waitingClicks.push({ how: "switch", run, tool: null, at: null, shift: false, ctrl: false, meta: false, detail: 1, made: now, first: now, double: false, asked: false });
    return true;
  });
  // (T-4214 stage A: and a station's name still being looked up, which an undo of its station must not overtake)
  setClicksBusy(() => tk.waitingClicks.length > 0 || tk.clicksOut > 0 || tk.placing || namesPending() > 0);
}

/** Whether a click now at map spot `at` would pick on a map that lacks an edit sent before it. */
function mapBehind(tk: Tools, at: LngLatish | null): boolean {
  if (tk.clicksOut > 0) return true;
  if (tool.peek() === "track") return tk.placing || editsOut(tk.start ? tk.run : 0) > 0 || !!tk.ownTrackOut(at);
  return editsOut() > 0;
}

/** A click, a tap or the touch bar's confirm with the tool in hand: true when it waits. */
export function waitClick(tk: Tools, how: WaitingClick["how"], e: MapMouseEvent | null): boolean {
  const q = tk.waitingClicks, o = e?.originalEvent, t = tool.peek();
  // T-445: a double click's second click ends the run with its first click's segment, waiting or not
  if (how === "click" && t === "track" && o && o.detail >= 2) {
    if (!q.length) return false;
    if (q[q.length - 1].how !== "switch") {
      q[q.length - 1].double = true;
      return true;
    }
  }
  const at = e ? { lng: e.lngLat.lng, lat: e.lngLat.lat } : tk.touchAt;
  if (!q.length && !mapBehind(tk, at)) return false;
  const now = performance.now();
  q.push({ how, tool: q.some((x) => x.how === "switch") ? null : t, at, shift: !!o?.shiftKey, ctrl: !!o?.ctrlKey, meta: !!o?.metaKey, detail: o?.detail ?? 1, made: now, first: now, double: false, asked: false });
  if (q.length === 1) {
    askNetwork(tk);
    armWait(tk);
  }
  return true;
}

/** T-2404: a network the worker holds for the pointer's rest is wanted now by the first waiting click (not one for a
 * track run's own segments: the run goes on without it, unless the click is on one whose state is held) */
function askNetwork(tk: Tools) {
  const h = tk.waitingClicks[0], track = tool.peek() === "track";
  if (!h || h.asked || (editsOut(track && tk.start ? tk.run : 0) === 0 && !(track && tk.ownTrackOut(h.at) === "held"))) return;
  h.asked = true;
  tk.client.placedNow();
}

function armWait(tk: Tools) {
  clearTimeout(tk.waitTimer);
  const h = tk.waitingClicks[0];
  if (!h) return;
  const now = performance.now();
  tk.waitTimer = window.setTimeout(() => tryClicks(tk), Math.max(0, h.first + CLICK_WAIT_MAX_MS - now) + 1);
  // the spinner, which a paused map draws no frame for
  if (now - h.made < WAIT_MS) setTimeout(() => tk.place(), h.made + WAIT_MS + 20 - now);
}

let trying = false;
/** Carry out the waiting clicks the map has caught up with, in order. */
export function tryClicks(tk: Tools) {
  if (!tk.waitingClicks.length) return clicksMoved();
  if (trying) return;
  trying = true;
  try {
    const q = tk.waitingClicks;
    while (q.length) {
      const h = q[0];
      if (h.how === "switch") {
        q.shift();
        switching = true;
        try { h.run!(); } finally { switching = false; }
        if (q.length) q[0].first = performance.now();
        continue;
      }
      // (the tool changed some other way: a load)
      if (h.tool !== null && h.tool !== tool.peek()) {
        q.length = 0;
        break;
      }
      const late = performance.now() - h.first >= CLICK_WAIT_MAX_MS;
      if (!late && mapBehind(tk, h.at)) {
        askNetwork(tk);
        break;
      }
      if (late) console.warn("T-4214: a click waited its longest for the map; carried out on the map as it is", { edits: outDrop(), clicksOut: tk.clicksOut, placing: tk.placing });
      q.shift();
      if (q.length) q[0].first = performance.now();
      carryOut(tk, h);
    }
  } finally {
    trying = false;
  }
  armWait(tk);
  tk.place();
  clicksMoved();
}

function carryOut(tk: Tools, h: WaitingClick) {
  const box = tk.map.getContainer(), p = h.at ? tk.map.project([h.at.lng, h.at.lat]) : null;
  if (p && !(p.x >= 0 && p.y >= 0 && p.x <= box.clientWidth && p.y <= box.clientHeight)) return;
  if (h.how === "confirm") {
    if (tk.touchAt) tk.move(evAt(tk, tk.touchAt));
    tk.carrying(touchConfirm(tk));
    return;
  }
  if (!p) return;
  const ev = (detail: number) => ({ point: p, lngLat: tk.map.unproject(p), originalEvent: { shiftKey: h.shift, ctrlKey: h.ctrl, metaKey: h.meta, altKey: false, detail, button: 0, buttons: 0, preventDefault() {}, stopPropagation() {} } }) as unknown as MapMouseEvent;
  tk.move(ev(h.detail));
  if (h.how === "tap") tk.tapNow(ev(h.detail));
  else tk.clickNow(ev(h.detail));
  if (h.double) tk.clickNow(ev(2));
}

/** Every waiting click dropped (Escape, a tool change, a new game). */
export function dropClicks(tk: Tools) {
  if (!tk.waitingClicks.length) return;
  tk.waitingClicks.length = 0;
  clearTimeout(tk.waitTimer);
  tk.place();
  clicksMoved();
}

/** Whether the first waiting click has waited more than WAIT_MS (the track tag's spinner). */
export const clickWaiting = (tk: Tools) => !!tk.waitingClicks.length && performance.now() - tk.waitingClicks[0].made > WAIT_MS;

/** While a click has waited more than WAIT_MS, the tool's tag shows the spinner: in its first row, or alone by the click.
 * (The bus, ferry and building tools draw their own tags.) */
export function waitSpin(tk: Tools) {
  const t = tool.peek(), h = tk.waitingClicks[0];
  if (!clickWaiting(tk) || t === "bus" || t === "ferry" || t === "building" || tk.tag.querySelector(".spin")) return;
  const spin = Object.assign(document.createElement("i"), { className: "spin" });
  if (tk.tag.style.display !== "none") {
    spin.classList.add("row-dot");
    (tk.tag.firstElementChild ?? tk.tag).prepend(spin);
  } else if (h.at) {
    const p = tk.map.project([h.at.lng, h.at.lat]);
    showTag(tk, [spin], false, p.x, p.y);
  }
}
