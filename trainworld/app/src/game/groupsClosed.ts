// T-4329, T-4337: the Lines tab's closed groups, kept in this browser per save slot, never in the save. Group numbers
// restart in every game, so each closed group is kept with a signature of its oldest line and counts as closed only
// while a group of that number still has it: a new game, or another save loaded into the slot, opens every group.

import { effect } from "@preact/signals";
import { gameStamp, saveSlotsState } from "../workers/clockClient";
import { closedGroups, lineGroups } from "./lineGroups";
import { VIDEO } from "./ui";
import { listedLines, world } from "./world";
import type { StationInput, WorldView } from "./types";

const KEY = "trainworld-closed-groups";
/** a slot's closed groups: group number to signature; a number list is from before T-4337 (no signatures) */
type Closed = Record<string, string> | number[];
type Store = Record<string, Closed>;

const read = (): Store => {
  try {
    const v = JSON.parse(localStorage.getItem(KEY) || "{}");
    return v && typeof v === "object" ? v : {};
  } catch { return {}; }
};
const save = (all: Store) => { try { localStorage.setItem(KEY, JSON.stringify(all)); } catch { /* this page only */ } };

const slotKey = () => { const s = saveSlotsState.peek(); return s.viewing ? "" : s.active; };
/** The oldest line stands for its group: its number and where its first stop is. Line numbers and station ids start over in
 * a new game, so the place is what tells games apart; a rename or recolour changes nothing. Records from the first T-4337
 * version also hold a name and colour between the two: only the first and last parts are read. */
const sigOf = (l: { num: number; stops: string[] }, at: Map<string, StationInput>) => {
  const s = at.get(l.stops[0]);
  return `${l.num}|${s ? s.lng.toFixed(5) + "," + s.lat.toFixed(5) : ""}`;
};
const readSig = (s: string) => { const p = s.split("|"); return `${p[0]}|${p[p.length - 1]}`; };
const sigs = (w: WorldView) => {
  const at = new Map(w.save.stations.map((s) => [s.id, s]));
  return new Map([...lineGroups(listedLines(w))].map(([g, ls]) => [g, sigOf(ls[0], at)]));
};

let fromStore = false;

effect(() => {
  void gameStamp.value; void saveSlotsState.value;
  const w = world.value, key = slotKey();
  const rec = key ? read()[key] : undefined;
  const now = w ? sigs(w) : new Map<number, string>();
  const legacy = Array.isArray(rec);
  const next = new Set<number>();
  if (rec) for (const [g, s] of legacy ? rec.map((n) => [n, null] as const) : Object.entries(rec).map(([k, v]) => [Number(k), v] as const)) {
    if (Number.isInteger(g) && now.has(g) && (s === null || readSig(s) === now.get(g))) next.add(g);
  }
  const cur = closedGroups.peek();
  if (next.size !== cur.size || [...next].some((g) => !cur.has(g))) {
    fromStore = true;
    try { closedGroups.value = next; } finally { fromStore = false; }
  }
  if (legacy && next.size && !VIDEO && key) store(next, now, key);
});

function store(s: ReadonlySet<number>, now: Map<number, string>, key: string) {
  const all = read();
  const rec: Record<string, string> = {};
  for (const g of s) { const sig = now.get(g); if (sig !== undefined) rec[g] = sig; }
  if (Object.keys(rec).length) all[key] = rec;
  else delete all[key];
  save(all);
}

// only the player's own toggles are written: a load that shows nothing yet must not erase what a slot keeps
let started = false;
effect(() => {
  const s = closedGroups.value, key = slotKey();
  if (!started) { started = true; return; }
  if (fromStore || VIDEO || !key) return;
  const w = world.peek();
  store(s, w ? sigs(w) : new Map(), key);
});
