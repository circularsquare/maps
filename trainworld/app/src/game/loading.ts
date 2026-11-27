// Loading and readiness (T-149, SPEC 7). The demand pool loads only the city packs a game needs
// (the viewed city and every city with a built station in reach) and the regional market; this
// is what the UI reads about that. Written by workers/demandClient.ts; read by the loading screen
// (ui/LoadingScreen.tsx), the top bar and the demand selector.
//
// A loaded game (the page opening, a file, a browser slot, a new game) holds the clock until the
// zone in view has its final figures from its fresh solve (T-347, T-421: its cluster's final may
// land well before the rest's, whose zones then show their own loading bars). Input is never blocked:
// the player can look around and build while demand loads. Only a failed load puts up a screen
// (ui/LoadingScreen.tsx).

import { t, tList } from "../i18n";
import { tPlace } from "../i18n/places";
import { computed, effect, signal } from "@preact/signals";
import { isMemoryFailure } from "./loadText";
import { leftText } from "./loadBar";
import { clock } from "./clock";
import { cityById, type CityId } from "./cities";
import { webgl2State } from "../map/webgl2";

export interface LoadProgress {
  /** the cities being loaded, the viewed one first */
  cities: CityId[];
  /** 0-1, how far the load is: each stage's share of the bar by its expected time (T-1195,
   * game/loadBar.ts; workers/demandClient.ts) */
  fraction: number;
  /** the stage in hand, in plain words (game/loadBar.ts `STAGE_TEXT`) */
  step: string;
  /** the startup load (the first game's) */
  startup: boolean;
  /** T-1479: the time still expected, ms, counting down; null while there is no estimate */
  leftMs?: number | null;
}

/** What the demand pool is loading now; null when nothing is. */
export const loadProgress = signal<LoadProgress | null>(null);
/** T-4346: the player closed the first load's card into the top status tag (resets on each load) */
export const cardDismissed = signal(false);
export function dismissLoadingCard() {
  cardDismissed.value = true;
}
/** T-2574: a card whose load ended at a cached result, shown full for END_BEAT_MS after the clock runs, so its bar
 * reaches the end before it goes (ui/LoadingScreen.tsx) */
export const cardEnding = signal<LoadProgress | null>(null);
const END_BEAT_MS = 400;
let endingTimer: ReturnType<typeof setTimeout> | undefined;
export function endCardFull(p: LoadProgress | null) {
  clearTimeout(endingTimer);
  cardEnding.value = p && !cardDismissed.peek() ? { ...p, fraction: 1, leftMs: null } : null;
  if (cardEnding.value) endingTimer = setTimeout(() => (cardEnding.value = null), END_BEAT_MS);
}
/** the card's last progress, so any load that ends with the card up shows its bar full first (a guess leaves the bar
 * at half at most, `barPlace`) */
let lastCardLoad: LoadProgress | null = null;
loadProgress.subscribe((p) => { if (p) lastCardLoad = p; });
/** T-343: each zone being loaded whose figures have not arrived yet, with its share done (0-1).
 * A zone of the load above has that load's share; a zone opened for its bubbles only, its own. */
export const zoneLoads = signal<ReadonlyMap<CityId, number>>(new Map());
/** T-677: the start menu is up; the clock and demand workers have not started. */
export const choosing = signal(false);
/** T-895: counts the games the player opened whose first state has arrived (workers/clockClient.ts);
 * ui/locate.ts fits the camera to the network on each. */
export const fitView = signal(0);
/** "loading" until the first game's demand has its first estimate, "failed" if it could not load. */
export const startup = signal<"loading" | "ready" | "failed">("loading");
/** T-318: nothing needs demand now: the view is over dead space and no built station reaches a
 * zone. Nothing loads, no card or status says "Loading data", and the clock is not held for it. */
export const demandIdle = signal(false);
/** Why demand could not load (the pool is stopped); null while it works. */
export const loadFailure = signal<string | null>(null);
/** T-484, T-492: zones whose pack could not be downloaded or opened, with why and whether it is tried
 * again. Demand carries on without them (workers/demandClient.ts); a zone leaves once its pack opens. */
export const zoneFailures = signal<ReadonlyMap<CityId, string>>(new Map());

/** T-1051: the game in hand has been loaded and its first demand is in (the moment the clock is let go,
 * or demand failed or was given up on). False from the start menu's pick and from every whole-game load. */
export const gameReady = signal(false);
/** T-2794: resolves once the game in hand's load is over */
export function whenGameReady(): Promise<void> {
  return new Promise((done) => {
    const stop = effect(() => {
      if (!gameReady.value) return;
      queueMicrotask(() => stop());
      done();
    });
  });
}
/** T-1051: the map shows only the basemap: behind the start menu, and while a game loads (the loading
 * card's span, from the pick to the clock's release). Nothing is drawn over it then, so what comes at
 * the end (hidden realms shaded, circles, cities for the shown realms) arrives in one step. Not while
 * demand has nothing to load (demandIdle). */
export const mapBlank = computed(() => choosing.value || (!gameReady.value && !demandIdle.value && !viewFigures.value));
/** T-1533: the zone in view's first figures of the game being loaded are in (a cached result, a rough
 * solve's, an estimate): the map shows them behind the card while the clock waits for the final.
 * `gameLoadSeq` counts the loads, so the demand client tells figures of this load from the last game's. */
const viewFigures = signal(false);
export const gameLoadSeq = signal(0);
export function viewFiguresIn() {
  viewFigures.value = true;
}

/** The clock waits for the next game's demand: set at startup and by every whole-game load. */
let holdArmed = true;

/** T-884: one wait for a game, from the pick or the load until the card goes. The start menu's pick
 * and the slot's arriving game are the same wait; a game loaded after the last one's network was taken
 * starts a new one. `key` names the game for the remembered length of its wait (the slot). */
let wait = { seq: 0, since: 0, key: "last" };
export const gameWait = () => wait;
/** The key a picked game's wait starts under: its slot; a new game's "new"; T-2552: a file or backup being opened has
 * no slot until it loads (the "loaded" message names it), so "pending", under which nothing is remembered, never
 * the last game's "last" */
export const waitKey = (choice: { slot?: string; newGame?: boolean; pending?: boolean }) => choice.slot ?? (choice.newGame ? "new" : choice.pending ? "pending" : "last");
/** The start menu's pick (main.ts): the startup wait begins. */
export function startGameWait(key: string, choice?: { slot?: string; newGame?: boolean; pending?: boolean; game?: { real?: unknown; start?: string } }) {
  // T-3927: the kind for a died-load report (the slot's name stays out of it)
  const kind = !choice ? "unknown" : choice.game?.real ? "real network" : choice.newGame ? "new game" : choice.pending ? "file, backup or shared game" : "saved slot";
  setLoadFlag(true, kind);
  if (choice?.game?.start) noteLoad("picked", { city: choice.game.start });
  wait ={ seq: wait.seq, since: performance.now(), key };
  cardDismissed.value = false;
  gameReady.value = false;
  viewFigures.value = false;
  gameLoadSeq.value++;
}

/** A whole game was loaded or started (workers/clockClient.ts): hold the clock until its demand is done. */
export function holdForGameLoad(key?: string) {
  if (loadFailure.peek()) return; // no demand to wait for
  endCardFull(null);
  cardDismissed.value = false;
  gameReady.value = false;
  if (!holdArmed) { viewFigures.value = false; gameLoadSeq.value++; }
  wait = holdArmed ? { ...wait, key: key ?? wait.key } : { seq: wait.seq + 1, since: performance.now(), key: key ?? "last" };
  holdArmed = true;
  clock?.setHeld(true);
}
/** A game load whose network has not reached the demand client yet. */
export const gameLoadArmed = () => holdArmed;

/** The demand client takes the hold for the network it is about to solve (true once per hold). */
export function takeGameLoadHold(): boolean {
  const armed = holdArmed;
  holdArmed = false;
  return armed;
}

const LONG_LOAD_MS = 30000;
let onLoadChime: (() => void) | null = null;
export function setLoadChimeHook(f: () => void) {
  onLoadChime = f;
}

/** The held game's demand has its final result: the clock runs, unless a later game load waits for
 * its own. */
export function loadComplete() {
  // T-742: the last game's hold ending after the next game loaded, before the demand client took
  // the next one's hold, ran the clock on the last game's demand (no fares) until the next final
  if (holdArmed) return;
  if (clock?.held.peek() && lastCardLoad && !cardEnding.peek()) endCardFull(lastCardLoad);
  lastCardLoad = null;
  cardDismissed.value = false;
  clock?.setHeld(false);
  // T-1655: a chime for a wait over 30 s, once per game
  if (!gameReady.peek() && performance.now() - wait.since > LONG_LOAD_MS) onLoadChime?.();
  gameReady.value = true;
  if (startup.peek() === "loading") startup.value = "ready";
}

/** T-3851: the page opened its game without calculating trips (`?trips=off`, a Community view's light load): no
 * demand pool starts, and the load is over once the network is in. Calculating trips reloads the page into the
 * game with the pool (workers/clockClient.ts `calculateTrips`). Set once, at startup. */
export const tripsOff = signal(false);
export function startWithoutTrips() {
  tripsOff.value = true;
}
/** T-3851: a full state of the game arrived with trips off: the clock runs and the game is ready */
export function networkInWithoutTrips() {
  if (!tripsOff.peek() || gameReady.peek()) return;
  holdArmed = false;
  loadComplete();
}

/** T-757: demand stopped after startup (a worker failed mid-game: out of memory, an error), as the
 * player is told it in the notice at the top centre (ui/DemandStopped.tsx); the figures shown stay. */
export const demandStopped = signal<"memory" | "error" | "graphics" | null>(null);

/** T-2794: how long a lost WebGL context may stay lost before the player is told (phones lose one when
 * memory is short and often get it back within a second or two) */
const GRAPHICS_BACK_MS = 6000;
const lostCanvases = new Set<HTMLCanvasElement>();
let graphicsTimer = 0;
/** T-2794: tell the player, and pause, when `canvas`'s context is lost and does not come back. */
export function watchGraphics(canvas: HTMLCanvasElement) {
  canvas.addEventListener("webglcontextlost", () => {
    lostCanvases.add(canvas);
    armGraphicsTimer();
  });
  canvas.addEventListener("webglcontextrestored", () => {
    lostCanvases.delete(canvas);
    if (lostCanvases.size) return;
    if (graphicsTimer) { clearTimeout(graphicsTimer); graphicsTimer = 0; }
    // the notice was for a context that never came back: this one did, so undo what the notice did
    if (demandStopped.peek() === "graphics" || (graphicsGone.peek() && !gameFailed.peek())) {
      if (demandStopped.peek() === "graphics") demandStopped.value = null;
      graphicsGone.value = false;
      if (graphicsPaused) clock?.setRunning(true);
    }
    graphicsPaused = false;
  });
}
/** T-3582: the browser has WebGL2 but refuses this page a context, so a reload cannot help: the screen says to
 * restart the browser. At startup nothing else starts; mid-game it covers
 * the game, which is paused and saved (`onGraphicsGone`). */
export const graphicsGone = signal(false);
let graphicsGoneSave: () => void = () => {};
export function onGraphicsGone(save: () => void) { graphicsGoneSave = save; }
export function loadGraphicsGone(reason: string) {
  graphicsGone.value = true;
  gameFailed.value = true;
  loadFailed(`WebGL2 context refused${reason ? ` (${reason})` : ""}`); // i18n-ignore: console and the failure's reason only
}
/** whether the graphics notice itself stopped the clock (a player who had paused stays paused) */
let graphicsPaused = false;
/** The 6 s count only runs while the page is visible: a hidden page loses its context as a matter of course. */
function armGraphicsTimer() {
  if (graphicsTimer || !lostCanvases.size || typeof document === "undefined" || document.visibilityState === "hidden") return;
  graphicsTimer = window.setTimeout(() => {
    graphicsTimer = 0;
    if (!lostCanvases.size || graphicsGone.peek()) return;
    // T-3582: a fresh context refused too means the browser blocked the page, and a reload would land on that
    const probe = webgl2State();
    if (probe.state !== "ok") {
      console.warn("graphics: the lost context did not come back and a new one was refused", probe.state === "refused" ? probe.reason : ""); // i18n-ignore
      if (demandStopped.peek() === "graphics") demandStopped.value = null;
      else graphicsPaused = !!clock?.running.peek();
      graphicsGone.value = true;
      clock?.setRunning(false);
      graphicsGoneSave();
      return;
    }
    if (demandStopped.peek()) return;
    demandStopped.value = "graphics";
    graphicsPaused = !!clock?.running.peek();
    clock?.setRunning(false);
  }, GRAPHICS_BACK_MS);
}
if (typeof document !== "undefined") {
  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "hidden") {
      if (graphicsTimer) { clearTimeout(graphicsTimer); graphicsTimer = 0; }
    } else armGraphicsTimer();
  });
}

/** T-2545: iOS may reload a page that ran out of memory without a word. A flag in sessionStorage is set
 * when a game starts loading and cleared when the load ends (or fails, or the page is left on purpose);
 * found still set at the next start, the last load died, and main.ts says so. */
const LOAD_FLAG = "tw-load-started";
/** T-3927: what the flag also holds, so a report after a died load says which game it was: the kind, the zones, the
 * save's station and line counts, the last step reached and when (ms since the pick). No slot name or account.
 * T-3933: also what a retry needs, kept out of the report: the most solving workers the load ran (`workers`; a worker
 * that ran out of memory may have cut them before the tab died), the slot the game is in once the page knows it
 * (`slot`), whether this load is itself the retry of one that died (`retry`), and whether the page saw its failure and
 * reloaded for the retry (`caught`). */
interface LoadMark { kind: string; zones?: string; city?: string; stations?: number; lines?: number; step: string; ms: number; workers?: number; slot?: string; retry?: boolean; caught?: boolean }
let mark: (LoadMark & { t0: number }) | null = null;
/** T-3933: the next load to start is the retry of one that died (main.ts) */
let nextIsRetry = false;
function writeMark() {
  try {
    if (!mark) { sessionStorage.removeItem(LOAD_FLAG); return; }
    const { t0: _t0, ...out } = mark; void _t0;
    sessionStorage.setItem(LOAD_FLAG, JSON.stringify(out));
  } catch { /* storage unavailable */ }
}
function setLoadFlag(on: boolean, kind = "unknown") {
  if (on) { mark = { kind, step: "picked", ms: 0, t0: Date.now(), ...(nextIsRetry ? { retry: true } : {}) }; nextIsRetry = false; }
  else mark = null;
  writeMark();
}
/** T-3940: the slot whose load last died in this browser, in localStorage so a closed tab keeps it too: the start
 * menu asks before opening it again. Kept until that slot loads with its trips. */
const DIED_SLOT = "tw-died-slot";
export function diedSlot(): string | null {
  try { return localStorage.getItem(DIED_SLOT); } catch { return null; }
}
function setDiedSlot(slot: string | null) {
  try { if (slot) localStorage.setItem(DIED_SLOT, slot); else localStorage.removeItem(DIED_SLOT); } catch { /* storage unavailable */ }
}
/** T-3927: a step of the load in hand (a name), and what is known of the game; no-op when no load is in hand or nothing
 * changed. Zones, a city, counts and the solving workers are kept once known. */
export function noteLoad(step: string, known: { zones?: string[]; city?: string; stations?: number; lines?: number; workers?: number } = {}) {
  if (!mark) return;
  const m = mark;
  const zones = known.zones?.length ? known.zones.slice(0, 12).join(",") : undefined;
  if (step === m.step && (!zones || zones === m.zones) && known.city === undefined && known.stations === m.stations && known.lines === m.lines && (known.workers === undefined || known.workers <= (m.workers ?? 0))) return;
  m.step = step;
  if (zones) m.zones = zones;
  if (known.city !== undefined) m.city = known.city;
  if (known.stations !== undefined) m.stations = known.stations;
  if (known.lines !== undefined) m.lines = known.lines;
  if (known.workers !== undefined) m.workers = Math.max(known.workers, m.workers ?? 0);
  m.ms = Date.now() - m.t0;
  writeMark();
}
/** T-3933: the pool was cut by plan (by memory estimate, not a failure): from here the load runs `n` solving workers */
export function noteLoadWorkers(n: number) {
  if (!mark || mark.workers === n) return;
  mark.workers = n;
  writeMark();
}
/** T-3933: the slot the loading game is in (the clock worker's saves list names it); "" for none (a test page, a
 * network only viewed). */
export function noteLoadSlot(slot: string) {
  if (!mark || (mark.slot ?? "") === slot) return;
  mark.slot = slot || undefined;
  writeMark();
}
/** T-3927: the unfinished load found at this start, as a line for the report; empty when none or an old flag. */
export let diedLoadReport = "";
/** T-4072: the load in hand's last step and when, for a report or the reason of a load that never ends; "" when none */
export function loadStepNow(): string {
  if (!mark) return "";
  return `${mark.step}, ${Math.round(mark.ms / 1000)} s after the pick, ${Math.round((Date.now() - mark.t0 - mark.ms) / 1000)} s ago`; // i18n-ignore
}
/** T-4072: the clock worker's own last step toward its game (the demand pool's steps follow in `noteLoad` and replace it
 * there), and when (performance.now()) */
let clockStep: { step: string; at: number } | null = null;
export function noteClockStep(step: string) {
  clockStep = { step, at: performance.now() };
  noteLoad("clock: " + step); // i18n-ignore
}
export const clockStepNow = () => (clockStep ? `${clockStep.step}, ${Math.round((performance.now() - clockStep.at) / 1000)} s ago` : ""); // i18n-ignore
/** T-3933: what a retry of the died load needs (null when none died, or an old flag) */
let died: Pick<LoadMark, "workers" | "slot" | "retry"> | null = null;
export const diedLoad = () => died;
/** Whether the last load in this tab never ended; the flag is cleared. */
export function lastLoadDied(): boolean {
  try {
    const raw = sessionStorage.getItem(LOAD_FLAG);
    sessionStorage.removeItem(LOAD_FLAG);
    if (!raw) return false;
    try {
      const m = JSON.parse(raw) as Partial<LoadMark> | null;
      if (m && typeof m === "object") {
        const secs = (n?: number) => Math.round((n ?? 0) / 1000);
        diedLoadReport = [
          String(m.kind ?? "unknown").slice(0, 40),
          m.zones ? "zones " + String(m.zones).slice(0, 200) : "",
          m.city ? "start city " + String(m.city).slice(0, 40) : "",
          m.stations !== undefined ? `${m.stations} stations, ${m.lines ?? "?"} lines` : "counts not known yet",
          m.workers !== undefined ? `up to ${m.workers} solving workers` : "",
          m.retry ? "it was the retry with fewer workers" : "",
          m.caught ? "a worker ran out of memory and the page reloaded to load it again" : "",
          `last step: ${String(m.step ?? "?").slice(0, 80)}, ${secs(m.ms)} s after the pick`,
        ].filter(Boolean).join("; ");
        died = {
          workers: Number.isInteger(m.workers) && m.workers! >= 1 ? m.workers : undefined,
          slot: typeof m.slot === "string" && m.slot ? m.slot : undefined,
          retry: m.retry === true,
        };
        if (died.slot) setDiedSlot(died.slot);
      }
    } catch { /* an older flag ("1") has no fields */ }
    return true;
  } catch { return false; }
}
/** T-3933: the load main.ts starts next retries one that died; the notice is up until a while after it is ready */
export const loadRetried = signal(false);
export function startRetry() {
  nextIsRetry = true;
  loadRetried.value = true;
}
/** T-3933: the load in hand, for a retry of it (null when none) */
export const loadInHand = (): Pick<LoadMark, "workers" | "slot" | "retry"> | null => mark && { workers: mark.workers, slot: mark.slot, retry: mark.retry };
/** T-3933: the page is about to reload to open the load in hand again (it saves first), so its flag stays set for the
 * next page; `keepLoadForRetry(false)` lets it go after all */
let leavingForRetry = false;
export function keepLoadForRetry(on: boolean) {
  leavingForRetry = on;
  if (!on && (gameReady.peek() || demandIdle.peek())) setLoadFlag(false);
}
export function reloadForRetry() {
  if (!mark) return;
  leavingForRetry = true;
  mark.caught = true;
  writeMark();
  location.reload();
}
/** T-3933: a load whose demand failed in the page's sight (not the tab dying): game/oomRetry.ts may take the failure over
 * (true) and open the game again in a fresh page when it ran out of memory (`memory`); once it has, every later failure
 * on the way out too */
let onLoadFail: ((why: string, memory: boolean) => boolean) | null = null;
export const handleLoadFail = (f: (why: string, memory: boolean) => boolean) => { onLoadFail = f; };
if (typeof addEventListener === "function") addEventListener("pagehide", () => { if (!leavingForRetry) setLoadFlag(false); });
effect(() => {
  if (!(gameReady.value || demandIdle.value) || leavingForRetry) return;
  // T-3940: the slot that died loaded whole this time
  if (mark?.slot && mark.slot === diedSlot() && !loadFailure.peek()) setDiedSlot(null);
  setLoadFlag(false);
});

/** Demand could not load. At startup the loading screen says so and offers a way on; later the
 * clock is released at once, since there is no result to wait for. `retried`: T-3933's retry already passed on it. */
export function loadFailed(why: string, retried = false) {
  if (!retried && mark && onLoadFail?.(why, isMemoryFailure(why))) { console.warn("load failed, opening it again:", why); return; } // i18n-ignore
  console.error("load failed:", why); // i18n-ignore
  loadFailure.value = why;
  loadProgress.value = null;
  cardDismissed.value = false;
  holdArmed = false;
  gameReady.value = true;
  if (startup.peek() === "loading") startup.value = "failed";
  // after startup; at a failed startup the player's choice releases it (every worker may report)
  else if (startup.peek() === "ready") {
    clock?.setHeld(false);
    // (T-2849: a real failure replaces the graphics notice, which goes away when the context returns)
    if (!demandStopped.peek() || demandStopped.peek() === "graphics") {
      demandStopped.value = isMemoryFailure(why) ? "memory" : "error";
      clock?.setRunning(false); // T-2657: paused once; the player may run it again
    }
  }
}

/** T-950: when the loading card's work last moved (a demand worker's message, a pack's bytes, the
 * clock worker's game or network); the loading gate's watchdog counts the quiet from here. */
let aliveAt = performance.now();
export const markAlive = () => { aliveAt = performance.now(); };
export const lastAlive = () => aliveAt;

/** T-2524: when the demand pool's loading last took a step (a pack's bytes or opening, a worker's result
 * or other report, never a query's answer): a bar or note says it has stalled after STALL_MS without one */
let stepAt = performance.now();
export const markStep = () => { stepAt = performance.now(); };
export const lastStep = () => stepAt;
/** (`?stallMs=N` sets it, for checks) */
export const STALL_MS = Number(typeof location === "undefined" ? 0 : new URLSearchParams(location.search).get("stallMs")) || 90_000;
/** T-2524: what each zone of `zoneLoads` is waiting on (the stage, game/loadBar.ts `STAGE_TEXT`) and the time
 * still expected, ms, as of `at` (performance.now()); null while there is no estimate */
export interface ZoneStep { step: string; leftMs: number | null; at: number }
export const zoneSteps = signal<ReadonlyMap<CityId, ZoneStep>>(new Map());
/** T-2524: the running whole solve's time still expected (a load's, or the same network solved again after
 * a cached result), as of `at`; null when no such solve runs or there is no estimate. T-4264: `bounded` when its
 * workers' units bound it (game/editLeft.ts `editLeftOf`), not the model's guess. */
export const solveLeft = signal<{ version: number; leftMs: number; at: number; bounded?: boolean } | null>(null);
/** The quiet since the last step, ms, once it is STALL_MS or more; else 0. */
export const stalledMs = (now = performance.now()) => (now - stepAt >= STALL_MS ? now - stepAt : 0);
/** T-2524: a load's hover lines after its name: the stage, then the time left counted down from `at`, or once
 * nothing has arrived for STALL_MS, how long it has been */
export function loadLines(z: { step?: string; leftMs: number | null; at: number } | undefined, now = performance.now()): string[] {
  const out = z?.step ? [z.step] : [];
  const quiet = stalledMs(now);
  if (quiet) out.push(quiet < 60_000 ? t("loading.stalledSeconds", { n: Math.floor(quiet / 10_000) * 10 }) : t("loading.stalledMinutes", { n: Math.floor(quiet / 60_000) }));
  else if (z && z.leftMs !== null) out.push(leftText(Math.max(0, z.leftMs - (now - z.at))));
  return out;
}

/** T-2756: the demand pool restarted or cut workers to free memory, and is solving the game again (until that final) */
export const memoryResolve = signal(false);
/** T-2756: the oldest city-data download in flight's start (performance.now()), or null; the demand pool sets it */
let downloadProbe: () => number | null = () => null;
export function setDownloadProbe(f: () => number | null) { downloadProbe = f; }
export const oldestDownload = () => downloadProbe();
/** T-4343: the quiet the loading card's watchdog allows while a solving worker is inside one of a solve's periods,
 * ms (0 when none is); the demand pool sets it */
let unitQuietProbe: () => number = () => 0;
export function setUnitQuietProbe(f: () => number) { unitQuietProbe = f; }
export const unitQuietMs = () => unitQuietProbe();

/** T-950: the demand pool's own stop (workers/demandClient.ts), so a failure found outside it ends
 * its work too */
let stopDemand: ((why: string) => void) | null = null;
export function onLoadStop(f: (why: string) => void) { stopDemand = f; }
/** T-3781: the demand pool's restart of a load that stopped moving (once; false when it cannot) */
let restartStuck: (() => boolean) | null = null;
export function onLoadStuck(f: () => boolean) { restartStuck = f; }
/** T-3781: the watchdog found the load quiet: restart demand's workers, once, before the failure screen */
export function loadStuck(): boolean {
  if (!restartStuck?.()) return false;
  markAlive();
  return true;
}
/** T-950: the game itself never started (the clock worker or the page's setup failed), so there is
 * nothing to play without demand either */
export const gameFailed = signal(false);
/** T-950: a load can never finish (the clock worker did not start, the page's setup threw, or the
 * watchdog found nothing moving): the failure screen goes up in place of the card. */
export function loadCannotFinish(why: string, game = false) {
  if (game) gameFailed.value = true;
  if (loadFailure.peek()) return;
  if (stopDemand) stopDemand(why);
  else loadFailed(why);
}

/** T-3307: this browser has no WebGL2, so the map cannot start. The failure screen says what to do and has no
 * button: nothing it could retry or skip would help. */
export const noWebgl2 = signal(false);
export function loadNoWebgl2() {
  noWebgl2.value = true;
  gameFailed.value = true;
  loadFailed("WebGL2 is not available"); // i18n-ignore: console and the failure's reason only, the screen has its own text
}

/** T-3315: this browser has no WASM SIMD (Safari before 16.4), so the simulation cannot start. Same screen as
 * the WebGL2 one, its own sentence. */
export const noSimd = signal(false);
export function loadNoSimd() {
  noSimd.value = true;
  gameFailed.value = true;
  loadFailed("WebAssembly SIMD is not available"); // i18n-ignore: console and the failure's reason only, the screen has its own text
}

/** Carry on without demand after a failed startup (the loading screen's second button). */
export function continueWithoutDemand() {
  cardDismissed.value = false;
  clock?.setHeld(false);
  gameReady.value = true;
  startup.value = "ready";
}

/** "New York", "New York and Philadelphia", "New York, Philadelphia and Boston". */
export function cityNames(ids: CityId[]): string {
  const names = ids.map((id) => tPlace(id, cityById(id).name));
  return tList(names);
}
