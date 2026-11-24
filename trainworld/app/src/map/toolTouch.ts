// The tools' touch screen part (T-1260): the tap's preview, the control bar's confirm, back and stop, and the handle (T-2374, moved out of tools.ts).

import { t as tr } from "../i18n";
import { type MapMouseEvent } from "maplibre-gl";
import { tool } from "../game/ui";
import { hiddenOnGlobe } from "./live";
import { SAME_TAP_PX, touchCtl, touchInput, type TouchCtl } from "./touch";
import { TOUCH_GAP_PX, type LngLatish } from "./toolParts";
import { stationClick } from "./stationTool";
import { freeCancel, freeFirst, freeStarted } from "./stationFree";
import { deleteClick } from "./toolDelete";
import type { Tools } from "./tools";

export function nearTouch(tk: Tools, x: number, y: number) {
  const p = tk.touchAt && tk.map.project([tk.touchAt.lng, tk.touchAt.lat]);
  return !!p && Math.hypot(p.x - x, p.y - y) < SAME_TAP_PX;
}

/** a click's event at a map point, for the touch bar's confirm */
export function evAt(tk: Tools, ll: LngLatish): MapMouseEvent {
  const p = tk.map.project([ll.lng, ll.lat]);
  return { point: p, lngLat: tk.map.unproject(p), originalEvent: { detail: 1, shiftKey: false } } as unknown as MapMouseEvent;
}

/** The bar's confirm, or a second tap on the point: what a click there does. */
export async function touchConfirm(tk: Tools) {
  const t = tool.peek(), at = tk.touchAt;
  if (t === "bus") return tk.bus.confirm();
  if (t === "ferry") return tk.ferry.confirm();
  if (t === "building") return tk.building.confirm();
  if (t === "track") {
    const from = tk.start;
    if (!from || !at || tk.placing) return;
    tk.touchAt = null;
    const ev = evAt(tk, at);
    await tk.segmentClick(ev, tk.hover ?? tk.snap(ev.point.x, ev.point.y, ev.lngLat));
    // refused: the end stays, to be moved or confirmed again
    if (tk.start === from && !tk.touchAt) {
      tk.touchAt = at;
      tk.place();
    }
  } else if (t === "station") {
    if (at) {
      tk.touchAt = null;
      tk.place();
      tk.ghostSpent = true;
      const placed = stationClick(tk, evAt(tk, at));
      tk.place();
      await placed;
      tk.place();
    }
  } else if (t === "delete" && at) {
    tk.touchAt = null;
    await deleteClick(tk, evAt(tk, at));
  }
}

/** The bar's back button: a segment being drawn (start and waiting end) is dropped whole, the
 * tool staying on (T-1703); else the last segment is taken back (Backspace), else the start goes. */
export function touchBack(tk: Tools) {
  if (tool.peek() === "bus") return tk.bus.touchBack();
  if (tool.peek() === "ferry") return tk.ferry.touchBack();
  if (tool.peek() === "building") return tk.building.touchBack();
  if (tool.peek() !== "track") return touchStop(tk);
  if (tk.touchAt && tk.start) {
    tk.hover = null;
    tk.reset();
  } else if (tk.chain.length) void tk.takeBackSegment();
  else tk.reset();
}

/** The bar's stop button: the run ends (its placed track stays) or the placement is dropped; the tool stays on. */
export function touchStop(tk: Tools) {
  const t = tool.peek();
  if (t === "bus" || t === "ferry" || t === "building") return tk.exit();
  if (t === "track") return tk.reset();
  tk.touchAt = null;
  tk.hover = null;
  if (t === "station") freeCancel(tk);
  if (t === "delete") {
    tk.delHover = null;
    tk.clearPreview();
  }
  tk.place();
}

/** Dragging the handle moves the point being previewed; a tap on it confirms. */
export function dragHandle(tk: Tools, e: PointerEvent) {
  e.preventDefault();
  const el = tk.handle;
  el.setPointerCapture(e.pointerId);
  let moved = false;
  const move = (m: PointerEvent) => {
    const r = tk.map.getContainer().getBoundingClientRect();
    const x = m.clientX - r.left, y = m.clientY - r.top;
    if (!moved && Math.hypot(m.clientX - e.clientX, m.clientY - e.clientY) < 6) return;
    moved = true;
    const ll = tk.map.unproject([x, y]);
    tk.touchAt = { lng: ll.lng, lat: ll.lat };
    tk.move({ point: { x, y }, lngLat: ll, originalEvent: m } as unknown as MapMouseEvent);
  };
  const up = (u: PointerEvent) => {
    el.removeEventListener("pointermove", move);
    el.removeEventListener("pointerup", up);
    el.removeEventListener("pointercancel", up);
    // a tap on the handle is a second tap on the point
    if (!moved && u.type === "pointerup") void touchConfirm(tk);
  };
  el.addEventListener("pointermove", move);
  el.addEventListener("pointerup", up);
  el.addEventListener("pointercancel", up);
}

/** What the touch bar offers now; null hides it. */
export function touchState(tk: Tools): TouchCtl | null {
  if (!touchInput.peek()) return null;
  const t = tool.peek();
  if (t === "bus") return tk.bus.touchState();
  if (t === "ferry") return tk.ferry.touchState();
  if (t === "building") return tk.building.touchState();
  if (t === "track" && tk.start) {
    const pv = tk.lastPreview;
    const bad = !!pv && !pv.ok && tk.previewKey === tk.wantKey && !tk.snapped;
    return { confirm: tk.touchAt ? !bad && !tk.placing : null, confirmTip: tr("tools.placeTrack"), back: !!tk.touchAt || tk.chain.length > 0, stopTip: tr("tools.stopDrawing") };
  }
  // T-3999: the first end of a station on open ground placed, the next tap its direction
  if (t === "station" && freeStarted()) return { confirm: null, confirmTip: tr("tools.placeStation"), back: false, stopTip: tr("tools.cancelStation") };
  if (t === "station" && tk.touchAt && tk.hover && tk.hover.kind !== "free") return { confirm: tk.shown?.ok !== false, confirmTip: tr("tools.placeStation"), back: false, stopTip: tr("tools.cancelStation") };
  if (t === "delete" && tk.touchAt && tk.delHover) return { confirm: true, confirmTip: tr("tools.remove"), back: false, stopTip: tr("tools.cancelRemove") };
  return null;
}

/** The touch bar beside the point it serves, clear of the finger and the map's edges, and the
 * tool's tag beyond it; the drag handle on the point being previewed. */
export function placeTouch(tk: Tools) {
  if (!tk.touchHost) return; // during construction
  const ctl = touchState(tk);
  const key = JSON.stringify(ctl);
  if (key !== tk.touchKey) {
    tk.touchKey = key;
    touchCtl.value = ctl;
  }
  const host = tk.touchHost, t = tool.peek();
  const at = t === "bus" ? tk.bus.touchPoint() : t === "ferry" ? tk.ferry.touchPoint() : t === "building" ? tk.building.touchPoint() : tk.touchAt ?? (t === "track" && tk.start ? tk.start : t === "station" ? freeFirst() : null);
  if (!ctl || !at) {
    host.style.display = "none";
    tk.handle.style.display = "none";
    return;
  }
  if (hiddenOnGlobe(at.lng, at.lat)) {
    host.style.display = "none";
    tk.handle.style.display = "none";
    tk.tag.style.display = "none";
    return;
  }
  const W = tk.overlay.cssW, H = tk.overlay.cssH;
  const p = tk.map.project([at.lng, at.lat]);
  host.style.display = "";
  const bw = host.offsetWidth, bh = host.offsetHeight;
  const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v));
  // off the side the track comes from: past a waiting end, away from the start (the ghost lies
  // between); at a run's new start, behind it, where the last segment came from (the next
  // segment usually goes on ahead, where the next tap will be)
  const from = tk.touchAt ? (t === "track" ? tk.start : null) : t === "track" ? tk.chain[tk.chain.length - 1] ?? null : null;
  const q = from && tk.map.project([from.lng, from.lat]);
  const wantAbove = !q ? true : tk.touchAt ? q.y > p.y : q.y < p.y;
  const fitsAbove = p.y - TOUCH_GAP_PX - bh >= 8, fitsBelow = p.y + TOUCH_GAP_PX + bh <= H - 8;
  const above = wantAbove ? fitsAbove || !fitsBelow : !fitsBelow && fitsAbove;
  const y = clamp(above ? p.y - TOUCH_GAP_PX - bh : p.y + TOUCH_GAP_PX, 8, H - bh - 8);
  const x = clamp(p.x - bw / 2, 8, W - bw - 8);
  const box = tk.map.getContainer().getBoundingClientRect();
  host.style.transform = `translate(${Math.round(box.left + x)}px, ${Math.round(box.top + y)}px)`;
  if (tk.tag.style.display !== "none") {
    const tw = tk.tag.offsetWidth, th = tk.tag.offsetHeight;
    let ty = above ? y - 6 - th : y + bh + 6;
    if (ty < 8 || ty + th > H - 8) ty = above ? p.y + 24 : p.y - 24 - th;
    tk.tag.style.transform = `translate(${Math.round(clamp(p.x - tw / 2, 8, W - tw - 8))}px, ${Math.round(ty)}px)`;
  }
  const drag = tk.touchAt && t === "track";
  tk.handle.style.display = drag ? "" : "none";
  if (drag) tk.handle.style.transform = `translate(${Math.round(p.x)}px, ${Math.round(p.y)}px)`;
}
