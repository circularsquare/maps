// Rust panics (T-1007): the model's recovery, and the failures that name one (T-2374).

import type { MessageKey } from "../i18n"; // type only: this worker sends keys, the page words them (docs/localization.md "Workers")
import { last_panic, TrackApi } from "../wasm/track/sim"; // the track model only (T-051)
import { NEW_GAME_FRAME } from "../game/cities";
import { frameKindNumber } from "../game/frame";
import { timeZoneTable } from "../game/localTime";
import { START_CASH } from "./saveFile";
import { isMemoryFailure, loadFailKey } from "../game/loadText";
import { saves } from "./slots";
import { PROFILE_VARS, SLOTS } from "../map/profiles";
import { putXferSet } from "./clockDemand";
import { dropFollowUp } from "./clockFollowUp";
import { blocksIn, masksIn, openNetwork, putTerrainIndexIn, putWorldWaterIn, roadFeed, streetsIn } from "./clockGround";
import { restart } from "./clockMain";
import { gameNow } from "./clockMoney";
import { api, messages, mode, putApi, putHolidaysSent, putMessages, putRowsSent, rowsSent, send, setMode, syncFrame } from "./clockState";

// ---------------------------------------------------------------- Rust panics (T-1007)
//
// A panic in the track model aborts the call and leaves wasm-bindgen's borrow on the `TrackApi`
// held, so every later call on it throws. The worker then makes a new `TrackApi` and opens the
// network in it as a load does, from the network as it was after the last load or edit (`kept`;
// an edit's panic always goes back to it, any other panic to the live network when it can still be
// read). The main thread logs the panic and tells the player ("recovered").

export const EDIT_UNDONE: MessageKey = "issues.internal";
export const MODEL_RESTORED: MessageKey = "clockWorker.modelRestored";
const MODEL_DOWN: MessageKey = "clockWorker.modelDown";
export const SAVE_PANIC: MessageKey = "clockWorker.savePanic";
/** more panics than this in a minute, or in all, and the model is left down: it is failing for a reason a rebuild does not cure */
const MAX_PANICS_PER_MINUTE = 3;
const MAX_PANICS = 8;

/** the network, cash and fleet after the last load or edit */
type Kept = { net: Uint8Array; cash: number; fleet: number; fleetFast: number; fleetBus: number; fleetFerry: number; fleetTram: number };
export let kept: Kept | null = null;
function modelNow(a: TrackApi): Kept {
  return { net: a.save(), cash: a.cash(), fleet: a.fleet(), fleetFast: a.fleet_fast(), fleetBus: a.fleet_bus(), fleetFerry: a.fleet_ferry(), fleetTram: a.fleet_tram() };
}
export function keep() {
  try {
    kept = modelNow(api!);
  } catch { /* the model is down; the next panic recovery says so */ }
}

const panicTimes: number[] = [];
let panicCount = 0;
export let modelDown = false;
/** when the last recovery ended; an error raised before it came from the model it replaced */
let recoveredAt = 0;
/** `?debug=1` probes (dev server only): the next edit panics after it is applied */
export let panicAfterEdit = false;
/** T-3773 probes (dev server only): the next game opened panics after its decode (`?debug=1&loadPanic=1`) */
export let panicOnLoad = false;

export function isPanic(err: unknown): boolean {
  return err instanceof WebAssembly.RuntimeError || (err instanceof Error && /recursive use of an object|null pointer passed to rust/.test(err.message));
}

/** T-2820: a panic's real message. "recursive use of an object" is only the error of the next call on
 * a model a panic has already stopped, so the Rust panic's own text comes first when there is one. */
export function panicWhy(err: unknown): string {
  let rust = "";
  try { rust = last_panic(); } catch { /* the module did not start */ }
  return (rust || (err instanceof Error ? err.message : String(err))).replace(/\s+/g, " ").trim();
}

/** Panic from a call that panics: an id no model has (a `&mut self` call, so the borrow stays held). */
export function panicNow() {
  api!.set_node(0xffffffff, 0, 0, 0, false);
}

/** A new track model over the same WASM instance, set up as the first one is (a panic leaves the
 * old one's borrow held for ever; its memory is left where it is, since freeing a model that
 * stopped half way through a change is not safe). */
export function newModel(): TrackApi {
  const a = new TrackApi(NEW_GAME_FRAME.lon, NEW_GAME_FRAME.lat, true);
  putApi(a);
  // T-3542: trips name profiles by map/profiles.ts's stride, the sim's `params::SLOTS`
  if (a.profile_slots() !== SLOTS) console.error("clock: map/profiles.ts SLOTS disagrees with sim/src/track/params.rs SLOTS", a.profile_slots()); // i18n-ignore
  // T-3346 stage 2: and each slot's train-by-train variants, the sim's `timing::PROFILE_VARS`
  if (a.profile_vars() !== PROFILE_VARS) console.error("clock: map/profiles.ts PROFILE_VARS disagrees with sim/src/track/timing.rs PROFILE_VARS", a.profile_vars()); // i18n-ignore
  putXferSet(null);
  putWorldWaterIn(false);
  putTerrainIndexIn(false);
  blocksIn.clear();
  roadFeed.reset();
  // T-309: each region's local time, kept across loads
  a.set_time_zones(timeZoneTable());
  a.set_empty_frame(frameKindNumber(NEW_GAME_FRAME.kind));
  syncFrame();
  a.clear_water();
  a.set_cash(START_CASH);
  return a;
}

/** A new, empty model in place of one a panic left unusable. */
function replaceModel() {
  (api as unknown as { __destroy_into_raw(): number }).__destroy_into_raw();
  newModel();
  masksIn.clear();
  streetsIn.clear();
  putRowsSent(""); putHolidaysSent(rowsSent);
}

/** T-2820: a panic while a game opens: nothing more is asked of the stopped model, and what comes
 * next (the slot it was opened over, a new game, the failure screen) starts on a new one. */
export function afterLoadPanic(err: unknown): string {
  const why = panicWhy(err);
  console.error("Rust panic while opening a game:", why); // i18n-ignore
  replaceModel();
  return why;
}

/** T-3773: what a game that failed to open says (`panicked`: its panic's message): a failed allocation, as a panic
 * or a JS error, is the device's out-of-memory line, so its card tells "won't open here" from damaged */
export function openFailReason(err: unknown, panicked: string | null, otherwise: string): { error: string; why?: string } {
  const text = panicked ?? String(err);
  if (isMemoryFailure(text)) return { error: loadFailKey(text) };
  return panicked !== null ? { error: SAVE_PANIC, why: panicked } : { error: otherwise };
}

async function rebuildModel(fromKept: boolean) {
  const old = api!;
  let k = kept;
  if (!fromKept) {
    try { k = modelNow(old); } catch { /* held: use what was kept */ }
  }
  if (!k) throw new Error("no network was kept");
  replaceModel();
  await openNetwork(k.net);
  api!.set_fleet(k.fleet);
  api!.set_fleet_fast(k.fleetFast);
  api!.set_fleet_bus(k.fleetBus);
  api!.set_fleet_ferry(k.fleetFerry);
  api!.set_fleet_tram(k.fleetTram);
  api!.set_cash(k.cash);
  setMode(mode);
  restart(gameNow(), "autosave", "rebuilt");
  saves.dirty = true;
}

/** Rebuild the track model after a panic and tell the main thread. `at`: when the error was raised. */
export async function recover(err: unknown, what: string, notice: string, at: number, fromKept = false) {
  if (modelDown || at <= recoveredAt) return;
  dropFollowUp();
  const now = Date.now();
  panicTimes.push(now);
  while (panicTimes[0] < now - 60_000) panicTimes.shift();
  panicCount++;
  const detail = `Rust panic in ${what}: ${panicWhy(err)}${err instanceof Error && err.stack ? `\n${err.stack}` : ""}`; // i18n-ignore
  try {
    if (panicTimes.length > MAX_PANICS_PER_MINUTE || panicCount > MAX_PANICS) throw new Error("too many panics");
    await rebuildModel(fromKept);
    send({ kind: "recovered", detail, notice });
  } catch (e) {
    modelDown = true;
    send({ kind: "recovered", detail: `${detail}\nnot rebuilt: ${e}`, notice: MODEL_DOWN, down: true }); // i18n-ignore
  }
  recoveredAt = performance.now();
}

/** A panic found outside a message (a timer, a save), rebuilt in turn with the messages. */
export function panicLater(err: unknown, what: string) {
  const at = performance.now();
  putMessages(messages.then(() => recover(err, what, MODEL_RESTORED, at)));
}
export function putModelDown(v: typeof modelDown) { modelDown = v; }
export function putPanicAfterEdit(v: typeof panicAfterEdit) { panicAfterEdit = v; }
export function putPanicOnLoad(v: typeof panicOnLoad) { panicOnLoad = v; }
