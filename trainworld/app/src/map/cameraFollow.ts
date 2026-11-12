// T-3361: the camera follows the selected train. While following, each overlay draw moves the map's
// centre onto the train; nothing runs when not following (the frame hook is added on start and taken
// out on stop). It ends when the player pans, the train is no longer drawn (zoomed far out, trains
// off, service ended), or the selection stops being this train (panel closed, another pick). A hand-over
// to the train's next trip (map/trainFollow.ts) keeps the selection's identity through `rebaseFollow`.

import { signal } from "@preact/signals";
import { clock } from "../game/clock";
import { display, selection } from "../game/ui";
import type { Selection } from "../game/types";
import { toLngLat } from "./geo";
import { live } from "./live";
import { tripAt } from "./network";

export const following = signal(false);

type Train = Extract<Selection, { kind: "train" }>;
let followed: Train | null = null;
// a pan by hand ends it (mouse or touch drag; the arrow keys)
const onDrag = () => stopFollow();
const onMoveStart = (e: { originalEvent?: Event }) => {
  const o = e.originalEvent;
  if (o instanceof KeyboardEvent && o.key.startsWith("Arrow")) stopFollow();
};

/** The followed train's place as longitude and latitude, null when it is not drawn. */
function whereIs(s: Train): [number, number] | null {
  const ov = live.overlay, r = ov?.renderer, n = r?.net;
  if (!r || !n || r.trainsHidden || !display.peek().trains) return null;
  const i = r.tripIndex(s.profile, s.dep - r.tripsEpoch);
  if (i < 0) return null;
  // drawn where the guard holds it (T-1965), as the panel and the picking place it
  const at = tripAt(n, s.profile, s.dep + r.tripLag(i), clock.now(), r.tripEnds ? r.tripEnds[i] + r.tripsEpoch : -Infinity);
  return at ? toLngLat(at.x, at.y) : null;
}

/** Runs after each overlay draw while following: the map's centre onto the train. */
function frame() {
  const map = live.map, s = followed;
  if (!map || !s) return;
  const p = whereIs(s);
  if (!p) return stopFollow();
  // a gesture's own easing (a scroll zoom) is left to finish; the centre is taken back after it
  if (map.isMoving()) return;
  const c = map.project(p), size = map.getCanvas().getBoundingClientRect();
  if (Math.hypot(c.x - size.width / 2, c.y - size.height / 2) < 0.5) return;
  map.jumpTo({ center: p });
}

export function startFollow() {
  const s = selection.peek(), map = live.map, ov = live.overlay;
  if (s?.kind !== "train" || !map || !ov) return;
  const wasOn = followed !== null;
  followed = s;
  if (wasOn) return;
  const p = whereIs(s);
  if (!p) { followed = null; return; }
  following.value = true;
  ov.afterDraw = frame;
  map.on("dragstart", onDrag);
  map.on("movestart", onMoveStart);
  // zooming keeps the train in the middle
  map.scrollZoom.disable();
  map.scrollZoom.enable({ around: "center" });
  map.touchZoomRotate.disable();
  map.touchZoomRotate.enable({ around: "center" });
  map.touchZoomRotate.disableRotation();
  // a train far from the middle is eased to, as a short glide; the frame hook keeps it there after
  const c = map.project(p), size = map.getCanvas().getBoundingClientRect();
  if (Math.hypot(c.x - size.width / 2, c.y - size.height / 2) > 8) map.easeTo({ center: p, duration: 350 });
  else map.triggerRepaint();
}

export function stopFollow() {
  const map = live.map, ov = live.overlay;
  if (followed === null) return;
  followed = null;
  following.value = false;
  if (ov?.afterDraw === frame) ov.afterDraw = null;
  if (!map) return;
  map.off("dragstart", onDrag);
  map.off("movestart", onMoveStart);
  map.scrollZoom.disable();
  map.scrollZoom.enable();
  map.touchZoomRotate.disable();
  map.touchZoomRotate.enable();
  map.touchZoomRotate.disableRotation();
}

export function toggleFollow() {
  if (followed) stopFollow();
  else startFollow();
}

/** The selected train's hand-over to its next trip is still the train followed. */
export function rebaseFollow(next: Selection) {
  if (followed && next?.kind === "train") followed = next;
}

// closing the panel, or picking anything else, ends it
selection.subscribe((s) => {
  if (followed && s !== followed) stopFollow();
});
