// T-1665: what holds back while demand is worked out. The clock runs no faster than DEMAND_CAP while the
// player's edits are not yet in the figures, and a line the player just changed is pending (Lines tab,
// line panel) until the result that counts it lands (T-1733: its cluster's local result, or the final;
// T-2658: of a network made after the change, never an older one finishing first).

import { computed, effect, signal } from "@preact/signals";
import { clock } from "./clock";
import { editStamp, lineStopped, quietEdits, resetEdits, wideCovered, wideEdits } from "./editCap";
import { demandBehindFrom, demandUpdating, finalWorld, networkWorld } from "./demand";
import { demandStopped, loadProgress } from "./loading";
import { openedZones } from "./openedZones";
import { world } from "./world";
import { t } from "../i18n";

/** T-2392: "build" for a line that has just started running (its blueprint constructed, or new);
 * "change" for one already running whose stops or service changed */
export type PendingKind = "build" | "change";

/** Lines whose riders are being worked out again after an edit, and why. */
export const linesPending = signal<ReadonlyMap<string, PendingKind>>(new Map());

/** The label over a pending line's riders: one word for a new line and a changed one (Anita, 2026-11-06). */
export const pendingLabel = (_k: PendingKind) => t("lineFigures.pendingNew");

/** T-2658: the world version each pending line's last change was seen at */
const since = new Map<string, number>();

/** T-1733: these lines' riders have landed (a cluster's result, local or whole), solved for the world
 * as it was at version `at` (T-2658): a line changed after it stays pending */
export function linesSettled(ids: Iterable<string>, at = Infinity) {
  const set = new Map(linesPending.peek());
  let changed = false;
  for (const id of ids) {
    if ((since.get(id) ?? -1) > at) continue;
    since.delete(id);
    changed = set.delete(id) || changed;
  }
  if (changed) linesPending.value = set;
}
const settleAll = (at = Infinity) => linesSettled([...linesPending.peek().keys()], at);

/** Wall ms to wait for a solve to start before deciding the edit needs none (a line with no riders, or
 * demand idle), so the label never sticks. */
const NO_SOLVE_MS = 8000;
/** Wall ms after an edit with no demand update under way, when the edits noted are forgotten */
const FORGET_EDITS_MS = 30000;

/** T-2622: whether the clock is held at DEMAND_CAP. Nothing the camera alone starts counts (a zone in
 * view refined, a pack opened on a pan, the zone in view's place in a solve's queue), so the cap is the
 * same wherever the view is. */
export const demandCapped = computed(() => {
  // T-1790: demand that stopped will finish nothing, so it holds nothing back
  if (demandStopped.value) return false;
  const pending = linesPending.value, updating = demandUpdating.value;
  // T-2409: a new line with no riders yet costs and earns nothing
  if (updating && [...pending.values()].includes("build")) return true;
  // a loud edit (T-1803, game/editCap.ts) until its lines' riders land, or one no line follows until the
  // whole solve's final
  if (updating && ((!quietEdits.value && pending.size > 0) || (wideEdits.value && demandBehindFrom.value !== null))) return true;
  // a load the network waits on (the game's own, a zone a station reaches), not a zone opened for the view
  const load = loadProgress.value, opened = openedZones.value;
  return !!load && (load.startup || !opened || load.cities.some((id) => opened.has(id)));
});

export function startLineFigures() {
  effect(() => clock.setCapped(demandCapped.value));
  // edits that came to nothing (refused, or no solve followed) are forgotten after a while
  effect(() => {
    const stamp = editStamp.value;
    const t = setTimeout(() => {
      if (editStamp.peek() !== stamp || demandUpdating.peek()) return;
      resetEdits();
      if (demandBehindFrom.peek() === null) wideCovered();
    }, FORGET_EDITS_MS);
    return () => clearTimeout(t);
  });
  let behind = demandBehindFrom.peek();
  demandBehindFrom.subscribe((m) => {
    if (behind !== null && m === null) wideCovered();
    behind = m;
  });

  /** each line's last signature and status, null before the first world */
  let prev: Map<string, { sig: string; running: boolean }> | null = null;
  world.subscribe((w) => {
    if (!w) { prev = null; return; }
    const next = new Map<string, { sig: string; running: boolean }>();
    const changed: [string, PendingKind][] = [];
    for (const l of w.save.lines) {
      const st = w.lineStats[l.id];
      const running = st?.status === "running";
      // (T-2622: its times too, so track moved under it counts as a change)
      const sig = JSON.stringify([l.stops, l.tph, st?.status, st?.cars, st?.fast, st?.roundTripByLevel.map(Math.round)]);
      next.set(l.id, { sig, running });
      const was = prev?.get(l.id);
      if (prev && was?.sig !== sig && running) changed.push([l.id, was?.running ? "change" : "build"]);
    }
    const loading = clock.held.peek();
    if (prev && !loading) for (const [id, was] of prev) if (was.running && !next.get(id)?.running) { lineStopped(); break; }
    prev = next;
    // a loaded game starts with nothing pending (its world versions are its own)
    if (loading) { since.clear(); if (linesPending.peek().size) linesPending.value = new Map(); }
    if (loading || !changed.length) return;
    const set = new Map(linesPending.peek());
    // a line still constructing stays so until its riders land
    for (const [id, k] of changed) {
      if (set.get(id) !== "build") set.set(id, k);
      since.set(id, w.version);
    }
    linesPending.value = set;
    // T-2658: an edit no network came for (nothing demand reads changed) leaves the figures as they are
    setTimeout(() => {
      if (!demandUpdating.peek() && networkWorld.peek() < w.version) settleAll(w.version);
    }, NO_SOLVE_MS);
  });

  // T-2658: the whole final for the world as it is settles every line; a final of an older network (another
  // edit's, the moment before the held one is sent) only those changed before it
  finalWorld.subscribe((at) => settleAll(at));
  // demand that stopped will finish nothing
  demandStopped.subscribe((s) => { if (s) settleAll(); });

  let wasUpdating = demandUpdating.peek();
  demandUpdating.subscribe((on) => {
    if (wasUpdating && !on) resetEdits();
    wasUpdating = on;
  });
}
