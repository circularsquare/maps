// T-2037: one card per game. Every save carries its game's origin (the id of the game it descends
// from); the saves of one origin are versions of one game, and the version with the latest player
// change is the game. Plain functions, used by both threads.

import type { SaveSlot, SlotBackup } from "./saveFile";
import type { AccountBackup, AccountGame } from "./cloud";

/** The wall time of the latest player change in a save; a save from before T-2037 counts its save
 * time, the latest its last change can be. */
export const editOf = (s: { edit?: number; at: number }): number => s.edit ?? s.at;
/** The same for an account game, from its facts. */
export const accountEditOf = (g: AccountGame): number => {
  const f = g.facts ?? {};
  return typeof f.edit === "number" ? f.edit : typeof f.at === "number" ? f.at : g.savedAt * 1000;
};
/** The player change a slot had when it last synced (a record from before T-2037: its save time then). */
export const syncedEditOf = (rec: { edit?: number; syncedAt: number }): number => rec.edit ?? rec.syncedAt;

/** A trailing " at 3:28 AM" (or "at 03:28", "at 15:28") that a fork or restore added before T-2037,
 * or " copy" from Make a copy. */
const SUFFIX = /(?:\s+at\s+\d{1,2}[:.]\d{2}(?:[\s  ]*[ap]\.?\s?m\.?)?|\s+copy)$/i;
/** A name without the time a fork or restore added before T-2037 (those now keep the game's name);
 * " copy" stays, a name Make a copy still gives. */
export function withoutForkTime(name: string): string {
  const at = /\s+at\s+\d{1,2}[:.]\d{2}(?:[\s  ]*[ap]\.?\s?m\.?)?$/i;
  let base = name.trim();
  while (at.test(base)) base = base.replace(at, "");
  return base || name;
}

/** A name without the fork, restore and copy endings. */
export function baseName(name: string): string {
  let base = name.trim();
  while (SUFFIX.test(base)) base = base.replace(SUFFIX, "");
  return base || name;
}

/** One save to give an origin: `id` its account game id, or its slot key when it has none. */
export interface OriginEntry { id: string; name: string; origin?: string }

/** Origins for saves from before T-2037 (T-2037 decision): a save that has one keeps it; a name with no
 * fork, restore or copy ending is a game of its own (its id); a name with one belongs to the save named
 * as its base name ("Tokyo at 6:17 PM" and "Tokyo copy" to "Tokyo"), the one with an origin already or
 * else the lowest id; with no save of that name, the saves sharing the base name go together under the
 * lowest id among them. The same list gives the same answer on every device. */
export function assignOrigins(entries: OriginEntry[]): Map<string, string> {
  const out = new Map<string, string>();
  const byId = (a: OriginEntry, b: OriginEntry) => (a.origin ? 0 : 1) - (b.origin ? 0 : 1) || (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);
  const rootOf = (e: OriginEntry) => e.origin ?? e.id;
  for (const e of entries) {
    if (e.origin) { out.set(e.id, e.origin); continue; }
    const base = baseName(e.name);
    if (base === e.name.trim()) { out.set(e.id, e.id); continue; }
    const named = entries.filter((x) => x.name.trim() === base).sort(byId);
    if (named.length) { out.set(e.id, rootOf(named[0])); continue; }
    const kin = entries.filter((x) => baseName(x.name) === base).sort(byId);
    out.set(e.id, rootOf(kin[0]));
  }
  return out;
}

/** One version of a game: a slot in this browser, a game on the account not in this browser, or a
 * backup of either. */
export type Version =
  | { kind: "slot"; slot: SaveSlot; at: number; edit: number }
  | { kind: "account"; game: AccountGame; at: number; edit: number }
  | { kind: "backup"; backup: SlotBackup; slot: SaveSlot; at: number; edit: number }
  | { kind: "accountBackup"; backup: AccountBackup; game: AccountGame; slot?: SaveSlot; at: number; edit: number };

/** A game's card: `top` is the version shown (the playing slot if any, else the latest player change),
 * `others` the rest, newest first. */
export interface GameGroup { origin: string; top: Extract<Version, { kind: "slot" | "account" }>; others: Version[]; at: number }

/** Slots and the account's games not in this browser grouped by origin, the group with the latest save
 * first. `backups`: each slot's backups, where known; `playing`: the slot being played, which heads
 * its card. Saves with no origin get one by `assignOrigins`. */
export function groupVersions(slots: SaveSlot[], account: AccountGame[], backups: Record<string, SlotBackup[]> = {}, playing = "", accountAll: AccountGame[] = account): GameGroup[] {
  const entries: OriginEntry[] = [
    ...slots.map((s) => ({ id: s.cloud?.id || s.key, name: s.name, origin: s.origin })),
    ...account.map((g) => ({ id: g.id, name: g.name, origin: typeof g.facts?.origin === "string" ? g.facts.origin : undefined })),
  ];
  const origins = assignOrigins(entries);
  const groups = new Map<string, Version[]>();
  const add = (o: string, v: Version) => { const l = groups.get(o); if (l) l.push(v); else groups.set(o, [v]); };
  for (const s of slots) {
    const o = origins.get(s.cloud?.id || s.key)!;
    add(o, { kind: "slot", slot: s, at: s.at, edit: editOf(s) });
    // a backup of the slot's own save, or with no player change between it and the slot (the clock
    // alone moved on), is the same version
    for (const b of (backups[s.key] ?? []).filter((b) => b.at !== s.at && editOf(b) !== editOf(s))) add(o, { kind: "backup", backup: b, slot: s, at: b.at, edit: editOf(b) });
    const g = s.cloud?.id ? accountAll.find((x) => x.id === s.cloud!.id) : undefined;
    for (const b of g?.backups ?? []) add(o, { kind: "accountBackup", backup: b, game: g!, slot: s, at: b.savedAt * 1000, edit: b.savedAt * 1000 });
  }
  for (const g of account) {
    const o = origins.get(g.id)!;
    add(o, { kind: "account", game: g, at: g.savedAt * 1000, edit: accountEditOf(g) });
    for (const b of g.backups ?? []) add(o, { kind: "accountBackup", backup: b, game: g, at: b.savedAt * 1000, edit: b.savedAt * 1000 });
  }
  const out: GameGroup[] = [];
  for (const [origin, list] of groups) {
    const heads = list.filter((v): v is GameGroup["top"] => v.kind === "slot" || v.kind === "account");
    if (!heads.length) continue;
    const top = heads.find((v) => v.kind === "slot" && v.slot.key === playing)
      ?? heads.slice().sort((a, b) => b.edit - a.edit || b.at - a.at)[0];
    const others = list.filter((v) => v !== top).sort((a, b) => b.at - a.at);
    out.push({ origin, top, others, at: Math.max(...list.filter((v) => v.kind === "slot" || v.kind === "account").map((v) => v.at)) });
  }
  return out.sort((a, b) => b.at - a.at);
}

/** The slots and account games a card's delete removes (backups go with them). */
export function groupHeads(g: GameGroup): GameGroup["top"][] {
  return [g.top, ...g.others.filter((v): v is GameGroup["top"] => v.kind === "slot" || v.kind === "account")];
}
