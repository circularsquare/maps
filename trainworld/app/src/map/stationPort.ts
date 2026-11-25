// T-3999: the station tool's port kind: a port placed on the shore with no ferry line yet, found as the ferry tool
// finds one (`FerryWater.port`), drawn in its ghost colour. A ferry line later stops there as at any port.

import type { MapMouseEvent } from "maplibre-gl";
import { t as tr } from "../i18n";
import { lngLatToPack } from "../game/coords";
import { describeIssues } from "../game/issues";
import { tool } from "../game/ui";
import { edit, say, world } from "../game/world";
import { metresBetween } from "../game/walks";
import { portReachM } from "../ferry/route";
import type { PortSnap } from "../ferry/ferryWater";
import { newStationName } from "./stationNames";
import { nameLater, placedSay, takenNames } from "./stationTool";
import { playBuild } from "./buildSound";
import { withName } from "./toolParts";
import type { Tools } from "./tools";

type Pt = { lng: number; lat: number };
/** a new port's landing this near a port the network has is that port (the ferry tool's `PORT_NEAR_M`) */
const PORT_NEAR_M = 30;
const PORT_PX = 10;

/** the last answer for where a click would put a port, and whether one is being asked */
let probed: { at: Pt; reach: number; snap: PortSnap } | null = null;
let probing = false;
let svg: SVGSVGElement | null = null;

const reachAt = (tk: Tools, at: Pt) => portReachM((78271.517 * Math.cos((at.lat * Math.PI) / 180)) / 2 ** tk.map.getZoom());
const probeIs = (tk: Tools, at: Pt) => !!probed && probed.at.lng === at.lng && probed.at.lat === at.lat && probed.reach === reachAt(tk, at);

/** The network's port within `PORT_PX` of the pointer or whose landing is within `PORT_NEAR_M` of `p`. */
function portNear(tk: Tools, p: Pt, px?: number, py?: number): { name: string; lng: number; lat: number; berth: [number, number] } | null {
  let best = Infinity, hit: ReturnType<typeof portNear> = null;
  for (const s of world.peek()?.save.stations ?? []) {
    if (!s.port) continue;
    const dm = metresBetween(s.lng, s.lat, p.lng, p.lat);
    let d = dm <= PORT_NEAR_M ? dm : Infinity;
    if (px !== undefined && py !== undefined) {
      const q = tk.map.project([s.lng, s.lat]);
      if (Math.hypot(q.x - px, q.y - py) <= PORT_PX) d = Math.min(d, 0);
    }
    if (d < best) (best = d), (hit = { name: s.name, lng: s.lng, lat: s.lat, berth: s.port });
  }
  return hit;
}

/** Ask where a click at the pointer would put a port; the newest position wins. */
function probe(tk: Tools) {
  const p = tk.pointer;
  if (probing || !p) return;
  const at = { lng: p.lng, lat: p.lat }, reach = reachAt(tk, at);
  probing = true;
  void tk.ferry.water.port([at.lng, at.lat], reach).then((snap) => {
    probing = false;
    if (snap) probed = { at, reach, snap };
    if (tool.peek() !== "station") return;
    tk.place();
    const q = tk.pointer;
    if (q && !probeIs(tk, { lng: q.lng, lat: q.lat })) probe(tk);
  });
}

export function hidePortGhost() {
  if (svg) svg.style.display = "none";
}

/** The port kind's ghost and tag at the pointer (the ferry tool's look: a circle on the landing, a line to the berth). */
export function portMarks(tk: Tools, showTag: (content: string | (string | Node)[], bad: boolean, x: number, y: number) => void) {
  const p = tk.pointer;
  hidePortGhost();
  if (!p) return void (tk.tag.style.display = "none");
  const at = { lng: p.lng, lat: p.lat };
  if (!probeIs(tk, at)) probe(tk);
  const near = portNear(tk, at, p.px, p.py);
  // the last answer stands while a newer one is asked, so the tag stays up and only its text changes
  const s = probed?.snap;
  if (near) return void showTag(...withName(near.name, tr("issues.stationHere")), p.px, p.py);
  if (!s || (!s.ok && s.reason === "tiles")) return void (tk.tag.style.display = "none");
  if (!s.ok) return void showTag(tr(s.reason === "on-water" ? "ferryDraw.tagOnWater" : "ferryDraw.tagFar"), true, p.px, p.py);
  const landing = { lng: s.landing[0], lat: s.landing[1] };
  const there = portNear(tk, landing);
  if (there) return void showTag(...withName(there.name, tr("issues.stationHere")), p.px, p.py);
  if (!svg) {
    svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("class", "ferry-draw");
    svg.style.cssText = "position:absolute;left:0;top:0;overflow:visible;pointer-events:none";
    tk.box.appendChild(svg);
  }
  const a = tk.map.project([s.landing[0], s.landing[1]]), b = tk.map.project([s.berth[0], s.berth[1]]);
  svg.style.display = "";
  svg.innerHTML = `<path d="M${a.x.toFixed(1)} ${a.y.toFixed(1)}L${b.x.toFixed(1)} ${b.y.toFixed(1)}" style="stroke:var(--ferry-ghost);stroke-width:2.5px"/>`
    + `<circle cx="${a.x.toFixed(1)}" cy="${a.y.toFixed(1)}" r="5.5" style="fill:var(--map-land);stroke:var(--ferry-ghost);stroke-width:2.5px"/>`;
  showTag(tr("ferryDraw.placePort"), false, p.px, p.py);
}

/** A click with the port kind: a blueprint port on the nearest shore, on no line. */
export async function portClick(tk: Tools, e: MapMouseEvent) {
  const at = { lng: e.lngLat.lng, lat: e.lngLat.lat };
  if (portNear(tk, at, e.point.x, e.point.y)) return void say(tr("issues.stationHere"), true);
  const snap = probeIs(tk, at) ? probed!.snap : await tk.ferry.water.port([at.lng, at.lat], reachAt(tk, at));
  if (!snap || (!snap.ok && snap.reason === "tiles")) return void say(tr("ferryDraw.notLoaded"), true);
  if (!snap.ok) return void say(tr(snap.reason === "on-water" ? "ferryDraw.sayOnWater" : "ferryDraw.sayFar"), true);
  const landing = { lng: snap.landing[0], lat: snap.landing[1] };
  if (portNear(tk, landing)) return void say(tr("issues.stationHere"), true);
  playBuild("station");
  // T-4214: sent at once, its name after it
  const naming = nameLater(tk, () => newStationName(tk.map, landing.lng, landing.lat, null, takenNames, false).catch(() => tr("ferryDraw.port")));
  const a = lngLatToPack(landing.lng, landing.lat), b = lngLatToPack(snap.berth[0], snap.berth[1]);
  const r = await edit({ op: "addPort", x: a.x, y: a.y, bx: b.x, by: b.y, name: "", ticket: naming.ticket }, true, true, "click");
  if (r.ok) placedSay(tk, [landing.lng, landing.lat], naming.name);
  else if (r.issues.length) say(describeIssues(r.issues, undefined, undefined, true), true);
}
