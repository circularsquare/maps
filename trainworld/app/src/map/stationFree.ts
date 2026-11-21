// T-3999: a train or tram station on open ground (brought back from T-794/T-907, which T-1820 took out): the first
// click is one end of the platform, the second sets only its direction, and the station lays its own straight piece
// of track, the set station length plus FREE_MARGIN_M at each end, at the level and of the track type picked.

import { t as tr } from "../i18n";
import type { MapMouseEvent } from "maplibre-gl";
import { EARTH_R } from "../game/frame";
import { lngLatToPack } from "../game/coords";
import { describeIssues } from "../game/issues";
import { length, moneyShort } from "../game/format";
import { buildLevel, buildPlatform, buildTramPlatform, quadTrack, stationOnTram, trackType } from "../game/ui";
import { edit, say, world } from "../game/world";
import type { StationAt } from "../workers/protocol";
import { strokesFromParts } from "./network";
import { toLngLat, toLocal } from "./geo";
import { playBuild } from "./buildSound";
import { touchInput } from "./touch";
import type { Snap } from "./toolParts";
import type { Tools } from "./tools";

/** the piece runs this far past each platform end: the track model wants a metre clear, and room for a later route
 * to join without a short piece at the end */
const FREE_MARGIN_M = 10;
/** the express pair's spacing on quad track (sim `QUAD_SPACING`) */
const QUAD_SPACING_M = 8;
/** the pointer this near the first end keeps the last heading */
const SAME_PX = 8;

type Free = Extract<Snap, { kind: "free" }>;

/** the first end once clicked, in degrees (local units go stale when the draw anchor moves); the heading of the last
 * station placed so (radians from east, ground) */
let first: { lng: number; lat: number } | null = null;
const firstAt = (): Free | null => {
  if (!first) return null;
  const [lx, ly] = toLocal(first.lng, first.lat);
  return { kind: "free", lx, ly };
};
let lastHeading = 0;
/** the worker's answer for the last piece asked about, and the piece drawn as the track preview (its key) */
let ans: { key: string; ok: boolean; why: string; cost: number } | null = null;
let asked = "";
let drawn = "";

export const freeStarted = () => first !== null;
export const freeFirst = () => first;
/** whether the piece under the pointer would be placed (null: not known yet) */
export const freeOk = () => (ans && ans.key === asked ? ans.ok : null);

/** The first click on open ground: the platform's first end. */
export function freeStart(tk: Tools, h: Free) {
  const [lng, lat] = toLngLat(h.lx, h.ly);
  first = { lng, lat };
  tk.place();
}

/** Drop the first end and the piece's preview. */
export function freeCancel(tk: Tools) {
  first = null;
  clearPreview(tk);
}

function clearPreview(tk: Tools) {
  if (!drawn) return;
  drawn = "";
  tk.overlay.renderer.setPreview(null);
  tk.overlay.invalidate();
}

// T-4010: the kind picked decides, not the track tool's type (a tram stop is its own kind)
const tram = () => stationOnTram.peek();
const platform = () => (tram() ? buildTramPlatform.peek() : buildPlatform.peek());

/** The heading from the first end towards the pointer. */
function heading(tk: Tools, a: Free): number {
  const p = tk.pointer;
  if (!p) return lastHeading;
  const [sx, sy] = tk.screen(a.lx, a.ly);
  if (Math.hypot(p.px - sx, p.py - sy) < SAME_PX) return lastHeading;
  const [lng0, lat0] = toLngLat(a.lx, a.ly);
  return Math.atan2(p.lat - lat0, (p.lng - lng0) * Math.cos((lat0 * Math.PI) / 180));
}

function piece(h: Free, phi: number) {
  const plat = platform();
  const [lng0, lat0] = toLngLat(h.lx, h.ly);
  const cos = Math.cos((lat0 * Math.PI) / 180), deg = 180 / Math.PI;
  const level = buildLevel.peek(), quad = quadTrack.peek() && !tram(), z = level * 8, tt = tram() ? "tram" : trackType.peek() === "hsr" ? "hsr" : "regular";
  // ground metres east and north of the first end to lng, lat
  const at = (e: number, n: number): [number, number] => [lng0 + (e / (EARTH_R * cos)) * deg, lat0 + (n / EARTH_R) * deg];
  const ce = Math.cos(phi), sn = Math.sin(phi);
  const s0 = -FREE_MARGIN_M, s1 = plat + FREE_MARGIN_M, sm = plat / 2;
  const a = at(s0 * ce, s0 * sn), b = at(s1 * ce, s1 * sn);
  const loc = (p: [number, number]) => toLocal(p[0], p[1]);
  const [ax, ay] = loc(a), [bx, by] = loc(b), [mx, my] = loc(at(sm * ce, sm * sn));
  const track = [new Float32Array([ax, ay, z, bx, by, z])];
  const ways = [new Float32Array([mx, my, z, bx, by, z]), new Float32Array([mx, my, z, ax, ay, z])];
  if (quad) {
    // the express pair to the left, as the track model lays it; the station is on the drawn pair only (T-1687)
    const ox = -sn * QUAD_SPACING_M, on = ce * QUAD_SPACING_M;
    const [qax, qay] = loc(at(s0 * ce + ox, s0 * sn + on)), [qbx, qby] = loc(at(s1 * ce + ox, s1 * sn + on));
    track.push(new Float32Array([qax, qay, z, qbx, qby, z]));
  }
  const pa = lngLatToPack(...a), pb = lngLatToPack(...b);
  const stationAt: StationAt = { kind: "free", x0: pa.x, y0: pa.y, x1: pb.x, y1: pb.y, level, quad, hsr: tt === "hsr", tram: tt === "tram" };
  const mid: Free = { kind: "free", lx: mx, ly: my };
  return { stationAt, plat, mid, track, ways, perM: 1 / cos, lengthM: s1 - s0, key: `${JSON.stringify(stationAt)}:${plat}:${world.peek()?.version}` };
}

/** After the first click: the station and its piece as they would be placed toward the pointer, the worker asked
 * whether it would be taken. `box` draws the platform ghost. */
export function freeGhost(tk: Tools, box: (lx: number, ly: number, ways: ArrayLike<number>[], perM: number, bad: boolean, plat: number, tram: boolean) => void): { ok: boolean | null; why: string; text: string; mid: Free } | null {
  const freeAt = firstAt();
  if (!freeAt) return null;
  const p = piece(freeAt, heading(tk, freeAt));
  if (asked !== p.key) {
    asked = p.key;
    tk.client.checkStation(p.stationAt, p.plat, (ok, issues, cost) => {
      ans = { key: p.key, ok, why: ok ? "" : describeIssues(issues, undefined, undefined, true) || tr("tools.cannotPlace"), cost };
      if (first) tk.place();
    });
  }
  const a = ans?.key === p.key ? ans : null, bad = a?.ok === false;
  const d = `${p.key}:${bad}`;
  if (drawn !== d) {
    drawn = d;
    tk.overlay.renderer.setPreview(strokesFromParts(p.track, bad, true));
    tk.overlay.invalidate();
  }
  box(p.mid.lx, p.mid.ly, p.ways, p.perM, bad, p.plat, tram());
  return { ok: a ? a.ok : null, why: a?.why ?? "", text: a?.ok ? tr("tools.freePiece", { track: length(p.lengthM), cost: moneyShort(a.cost * 1e6) }) : "", mid: p.mid };
}

/** The instruction for the next click on open ground. */
export const freeHint = () => tr(first ? "tools.freeTurn" : "tools.freeStart", { touch: touchInput.peek() });

/** The second click: the station and its own piece of track, one edit. */
export async function freeClick(tk: Tools, e: MapMouseEvent, nameFor: (px: number, py: number) => Promise<string>) {
  const h = firstAt();
  if (!h) return;
  tk.pointer = { px: e.point.x, py: e.point.y, lng: e.lngLat.lng, lat: e.lngLat.lat };
  const phi = heading(tk, h), p = piece(h, phi);
  if (!(ans?.key === p.key && ans.ok === false)) playBuild("station");
  first = null;
  lastHeading = phi;
  clearPreview(tk);
  tk.place();
  const name = await nameFor(...tk.screen(p.mid.lx, p.mid.ly));
  const r = await edit({ op: "addStation", at: p.stationAt, platform: p.plat, name }, true, true);
  if (r.ok) {
    tk.lastPlaced = toLngLat(p.mid.lx, p.mid.ly);
    say(tr("tools.blueprintStation", { name }));
  } else if (r.issues.length) say(describeIssues(r.issues, undefined, undefined, true), true);
}
