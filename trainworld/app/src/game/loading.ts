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
/** T-2574: a card whose load ended at a cached result, shown full for END_BEAT_MS after the clock runs, so its bar
 * reaches the end before it goes (ui/LoadingScreen.tsx) */
export const cardEnding = signal<LoadProgress | null>(null);
const END_BEAT_MS = 400;
let endingTimer: ReturnType<typeof setTimeout> | undefined;
export function endCardFull(p: LoadProgress | null) {
  clearTimeout(endingTimer);
  cardEnding.value = p && { ...p, fraction: 1, leftMs: null };
  if (p) endingTimer = setTimeout(() => (cardEnding.value = null), END_BEAT_MS);
}
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
export function startGameWait(key: string) {
  setLoadFlag(true);
  wait ={ seq: wait.seq, since: performance.now(), key };
  gameReady.value = false;
  viewFigures.value = false;
  gameLoadSeq.value++;
}

/** A whole game was loaded or started (workers/clockClient.ts): hold the clock until its demand is done. */
export function holdForGameLoad(key?: string) {
  if (loadFailure.peek()) return; // no demand to wait for
  endCardFull(null);
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
  clock?.setHeld(false);
  // T-1655: a chime for a wait over 30 s, once per game
  if (!gameReady.peek() && performance.now() - wait.since > LONG_LOAD_MS) onLoadChime?.();
  gameReady.value = true;
  if (startup.peek() === "loading") startup.value = "ready";
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
    if (demandStopped.peek() === "graphics") {
      demandStopped.value = null;
      if (graphicsPaused) clock?.setRunning(true);
    }
    graphicsPaused = false;
  });
}
/** whether the graphics notice itself stopped the clock (a player who had paused stays paused) */
let graphicsPaused = false;
/** The 6 s count only runs while the page is visible: a hidden page loses its context as a matter of course. */
function armGraphicsTimer() {
  if (graphicsTimer || !lostCanvases.size || typeof document === "undefined" || document.visibilityState === "hidden") return;
  graphicsTimer = window.setTimeout(() => {
    graphicsTimer = 0;
    if (!lostCanvases.size || demandStopped.peek()) return;
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
function setLoadFlag(on: boolean) {
  try { if (on) sessionStorage.setItem(LOAD_FLAG, "1"); else sessionStorage.removeItem(LOAD_FLAG); } catch { /* storage unavailable */ }
}
/** Whether the last load in this tab never ended; the flag is cleared. */
export function lastLoadDied(): boolean {
  try { const died = !!sessionStorage.getItem(LOAD_FLAG); sessionStorage.removeItem(LOAD_FLAG); return died; } catch { return false; }
}
if (typeof addEventListener === "function") addEventListener("pagehide", () => setLoadFlag(false));
effect(() => { if (gameReady.value || demandIdle.value) setLoadFlag(false); });

/** Demand could not load. At startup the loading screen says so and offers a way on; later the
 * clock is released at once, since there is no result to wait for. */
export function loadFailed(why: string) {
  console.error("load failed:", why); // i18n-ignore
  loadFailure.value = why;
  loadProgress.value = null;
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
 * a cached result), as of `at`; null when no such solve runs or there is no estimate */
export const solveLeft = signal<{ version: number; leftMs: number; at: number } | null>(null);
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

/** T-950: the demand pool's own stop (workers/demandClient.ts), so a failure found outside it ends
 * its work too */
let stopDemand: ((why: string) => void) | null = null;
export function onLoadStop(f: (why: string) => void) { stopDemand = f; }
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
  clock?.setHeld(false);
  gameReady.value = true;
  startup.value = "ready";
}

/** "New York", "New York and Philadelphia", "New York, Philadelphia and Boston". */
export function cityNames(ids: CityId[]): string {
  const names = ids.map((id) => tPlace(id, cityById(id).name));
  return tList(names);
}
