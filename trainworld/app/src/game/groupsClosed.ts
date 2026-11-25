// T-4329: the Lines tab's closed groups, kept in this browser per save slot (group numbers are one game's), never in the save.

import { computed, effect } from "@preact/signals";
import { gameStamp, saveSlotsState } from "../workers/clockClient";
import { closedGroups } from "./lineGroups";
import { VIDEO } from "./ui";

const KEY = "trainworld-closed-groups";
type Store = Record<string, number[]>;

const read = (): Store => {
  try {
    const v = JSON.parse(localStorage.getItem(KEY) || "{}");
    return v && typeof v === "object" ? v : {};
  } catch { return {}; }
};
/** the slot's key; "" for a viewed network, which keeps nothing */
const slot = computed(() => { const s = saveSlotsState.value; return s.viewing ? "" : s.active; });
const slotKey = () => slot.peek();

effect(() => {
  void gameStamp.value; void slot.value;
  const ids = read()[slotKey()];
  closedGroups.value = new Set(Array.isArray(ids) ? ids.filter((n) => Number.isInteger(n)) : []);
});

effect(() => {
  const s = closedGroups.value, key = slotKey();
  if (VIDEO || !key) return;
  const all = read();
  if (s.size) all[key] = [...s];
  else delete all[key];
  try { localStorage.setItem(KEY, JSON.stringify(all)); } catch { /* this page only */ }
});
