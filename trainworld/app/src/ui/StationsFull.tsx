// "N stations full" (T-231, Anita: "a global alert in the top left under the demand buttons"): the
// station complexes where, now, a platform side holds as many riders as fit on it
// (`platformRoom`, T-219) or the last train to leave a platform was full and left riders behind
// (T-284: not this hour's total, which jumped as the hour turned), from the boarding account
// (game/boarding.ts, T-164). One count per complex (game/complexView.ts), so Times Sq's platforms
// are one entry. A click selects the next one in turn and moves the map to it; the hover lists them.
// Hidden while there are none.

import { clock } from "../game/clock";
import { crowdAt, platformRoom, QUEUE_STRIDE } from "../game/boarding";
import { complexOf } from "../game/complexView";
import { count } from "../game/format";
import { selection } from "../game/ui";
import { world } from "../game/world";
import { live } from "../map/live";

/** One complex that is full or leaving riders behind now. */
export interface FullStation {
  /** the member to select: the one with the most riders waiting */
  station: string;
  name: string;
  /** every member of the complex */
  members: string[];
  /** riders waiting on its platforms now, and of them those the last full trains left behind */
  waiting: number;
  leftBehind: number;
}

/** The complexes full or leaving riders behind at game time `now` (s), most left behind first. */
export function fullStations(now: number): FullStation[] {
  const w = world.peek(), r = live.overlay?.renderer;
  const h = r?.board;
  if (!w || !r || !h) return [];
  const t = now - r.tripsEpoch;
  // per station node: riders waiting on each side (run 0, run 1), the sides served, left behind
  const at = new Map<number, { wait: [number, number]; sides: number; left: number }>();
  for (let q = 0; q < h.queueCount; q++) {
    const b = q * QUEUE_STRIDE, node = h.queues[b], run = h.queues[b + 2] ? 1 : 0;
    let e = at.get(node);
    if (!e) at.set(node, (e = { wait: [0, 0], sides: 0, left: 0 }));
    const c = crowdAt(h, q, t);
    e.wait[run] += c.waiting;
    e.sides |= 1 << run;
    // T-284: left by the last train to leave, still waiting, not this hour's total, so the alert
    // follows the trains instead of jumping as the hour turns
    e.left += c.left;
  }
  const byNum = new Map(w.save.stations.map((s) => [s.num, s]));
  const groups = new Map<string, FullStation & { lead: number }>();
  for (const [node, e] of at) {
    const s = byNum.get(node);
    if (!s) continue;
    const room = platformRoom(s.length);
    // as the platforms are drawn (T-219): each side against its own room, or the whole platform
    // when trains use one side only
    const full = e.sides === 3 ? Math.max(e.wait[0], e.wait[1]) >= room : e.wait[0] + e.wait[1] >= 2 * room;
    const waiting = e.wait[0] + e.wait[1];
    if (!full && e.left < 1) continue;
    const members = complexOf(s.id);
    const key = [...members].sort()[0] ?? s.id;
    const g = groups.get(key);
    if (!g) groups.set(key, { station: s.id, name: s.name, members, waiting, leftBehind: e.left, lead: waiting });
    else {
      g.waiting += waiting;
      g.leftBehind += e.left;
      if (waiting > g.lead) { g.lead = waiting; g.station = s.id; g.name = s.name; }
    }
  }
  return [...groups.values()].sort((a, b) => b.leftBehind - a.leftBehind || b.waiting - a.waiting || a.name.localeCompare(b.name));
}

/** Closest zoom a station is shown at from the alert: where platform crowds are drawn. */
const ZOOM = 15;

export function StationsFull() {
  void clock.minute.value; // the crowds change each game minute
  void world.value;
  const sel = selection.value;
  const list = fullStations(clock.now());
  if (!list.length) return null;
  const n = list.length;
  const tip = ["Full platforms, or riders left behind by a full train",
    ...list.slice(0, 12).map((f) => `${f.name}: ${count(f.waiting)} waiting${f.leftBehind >= 1 ? `, ${count(f.leftBehind)} left behind` : ""}`),
    ...(n > 12 ? [`and ${n - 12} more`] : []),
    "Click to go to the next one"].join("\n");
  const next = () => {
    const now = fullStations(clock.now());
    if (!now.length) return;
    const cur = sel?.kind === "station" ? now.findIndex((f) => f.members.includes(sel.station)) : -1;
    const f = now[(cur + 1) % now.length];
    selection.value = { kind: "station", station: f.station };
    const s = world.peek()?.save.stations.find((s) => s.id === f.station);
    const map = live.map;
    if (s && map) map.jumpTo({ center: [s.lng, s.lat], zoom: Math.max(map.getZoom(), ZOOM) });
  };
  return (
    <button type="button" class="btn stations-full" data-tip={tip} onClick={next}>
      {n === 1 ? "1 station full" : `${n} stations full`}
    </button>
  );
}
