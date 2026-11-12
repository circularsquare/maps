// Startup (T-149, T-172, T-180, SPEC 8). Before it, the start menu (ui/StartMenu.tsx, T-677).
// The game is playable from the moment the map and network
// are drawn: the player can look around, inspect and build while demand loads. Only the clock
// waits (game/loading.ts). Meanwhile a card in the middle of the map says what is loading and how
// far it has got (the stage and a filling bar, T-243, T-1194); it takes no key and no pointer but its bar's, and goes when the clock
// starts. If demand cannot load, a screen says so and offers a way on; it takes every key until then.
// T-950: so does a load that can never finish (the game itself not starting, or nothing moving).

import { h, render } from "preact";
import { clock } from "../game/clock";
import { world } from "../game/world";
import { cardEnding, choosing, continueWithoutDemand, demandIdle, gameFailed, lastAlive, loadCannotFinish, loadFailure, loadProgress, noSimd, noWebgl2, startup } from "../game/loading";
import { leftText, STAGE_TEXT } from "../game/loadBar";
import { isMemoryFailure, loadFailText } from "../game/loadText";
import { marketOff } from "../game/ui";
import { playWithoutIntercity } from "./DemandStopped";
import { blockKeysOutside } from "./keys";
import { BunnyTrain } from "./Spinner";
import { t, tNumber } from "../i18n";

function LoadingScreen() {
  if (startup.value !== "failed") return null;
  // T-3307: no WebGL2: what to do about it, and no button (a retry or a way on cannot help)
  // T-3315: no WASM SIMD (Safari before 16.4): the same, with its own sentence
  if (noWebgl2.value || noSimd.value) {
    const text = t(noWebgl2.value ? "loadText.noWebgl2" : "loadText.noSimd");
    return <div class="loading-screen" role="alertdialog" aria-label={text}>
      <div class="loading-box failed">
        <p class="big">{text}</p>
      </div>
    </div>;
  }
  const game = gameFailed.value;
  const why = loadFailure.value ?? "";
  const text = loadFailText(why);
  // a reload of an address that names a game loads that game again; without the choice it is the start menu
  const named = location.search !== "";
  return <div class="loading-screen" role="alertdialog" aria-label={text}>
    <div class="loading-box failed">
      <p class="big">{text}</p>
      {/* the reason as the failure gave it (also in the console), so a rare one can be told apart */}
      {why && <p class="small muted load-why">{why}</p>}
      <div class="loading-actions">
        <button class="btn" onClick={() => location.reload()}>{t("loadingScreen.tryAgain")}</button>
        {named && <button class="btn" onClick={() => location.assign(location.pathname)}>{t("loadingScreen.backToStart")}</button>}
        {/* T-2867: out of memory, the way on keeps demand and drops intercity trips, while there are any to drop */}
        {!game && (isMemoryFailure(why) && !marketOff()
          ? <button class="btn" onClick={() => playWithoutIntercity()}>{t("loadingScreen.playWithoutIntercity")}</button>
          : <button class="btn" onClick={continueWithoutDemand}>{t("loadingScreen.playWithoutDemand")}</button>)}
      </div>
    </div>
  </div>;
}

/** True while the loading card is up: a loaded game's clock waits for its demand (T-180). */
export function loadingCardShown(): boolean {
  return clock.held.value && !loadFailure.value && !demandIdle.value && !choosing.value; // T-318: nothing to load over dead space
}

/** T-243: a bar filling as the load goes, in place of a percentage; the percentage is its hover.
 * The fill jumps to each new value (no transition). */
export function LoadBar({ fraction, step, thin, tip, quiet }: { fraction: number; step: string; thin?: boolean; tip?: string; quiet?: boolean }) {
  const pct = Math.floor(fraction * 100);
  return <span class={"load-bar" + (thin ? " thin" : "") + (quiet ? " quiet" : "")} role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={pct}
    data-tip={quiet ? undefined : tip ?? t("loadingScreen.loaded", { pct: tNumber(pct), step })}>
    <i class="load-fill" style={{ width: `${fraction * 100}%` }} />
  </span>;
}

/** T-180: while a loaded game's clock waits for its first demand estimate, a card in the middle of
 * the map. Pointer events pass through it to the map, except on the bar (its hover).
 * T-677: the bunny's train on top (only here; the small statuses have their bar alone, T-2728). T-1194: it names the stage in hand (game/loadBar.ts), "Loading"
 * until the demand pool has a load to describe. */
function LoadingCard() {
  // (T-2574: a load ended at a cached result shows its bar full a moment first)
  const up = loadingCardShown(), ending = cardEnding.value;
  if (!up && (!ending || loadFailure.value)) return null;
  const load = up ? loadProgress.value : ending;
  const step = load?.step ?? STAGE_TEXT.game;
  const fraction = load?.fraction ?? 0;
  const left = load?.leftMs;
  // T-1479: the share done and, once there is an estimate, the time left
  return <div class="loading-card-wrap">
    <div class="loading-card" role="status">
      <BunnyTrain />
      <p class="big">{step}</p>
      <LoadBar fraction={fraction} step={step} />
      <p class="load-left small muted num"><span>{Math.floor(fraction * 100)}%</span>{left != null && <span>{leftText(left)}</span>}</p>
    </div>
  </div>;
}

/**
 * T-950: how long the card may stay up with nothing moving (game/loading.ts `markAlive`) before the
 * failure screen takes its place: with no bar, when nothing measurable is under way, and with one,
 * since a big batch of packs opening on a slow phone sends nothing for minutes. A hidden or frozen
 * page counts no quiet. `?quietMs=N` sets both (checks).
 */
const QUIET_MS = 180_000, QUIET_BAR_MS = 600_000, WATCH_MS = 5_000;
/** T-950: main.ts's setup ran to its end; an uncaught error before then means no game will start */
let setupFinished = false;
export function setupDone() { setupFinished = true; }

function startWatchdog() {
  const q = Number(new URLSearchParams(location.search).get("quietMs"));
  const quietMs = (bar: boolean) => (q > 0 ? q : bar ? QUIET_BAR_MS : QUIET_MS);
  let from = performance.now(), lastCheck = from;
  const timer = setInterval(() => {
    const now = performance.now();
    if (loadFailure.peek()) return clearInterval(timer);
    if (!loadingCardShown() || document.visibilityState !== "visible" || now - lastCheck > 3 * WATCH_MS) from = now;
    lastCheck = now;
    const limit = quietMs(!!loadProgress.peek());
    if (now - Math.max(from, lastAlive()) < limit) return;
    const m = Math.round(limit / 60_000), mins = m >= 1 ? t("loadingScreen.minutes", { count: m }) : t("loadingScreen.seconds", { count: Math.round(limit / 1000) });
    const started = !!world.peek();
    loadCannotFinish(t(started ? "loadingScreen.demandTimeout" : "loadingScreen.simTimeout", { time: mins }), !started);
  }, WATCH_MS);
  window.addEventListener("error", (e) => {
    if (!setupFinished) loadCannotFinish(t("loadingScreen.pageStopped", { reason: e.error instanceof Error ? e.error.message : e.message }), true);
  });
}

/** Hold the clock until the first game's demand has its first estimate; say so if it fails. */
export function startLoadingGate() {
  clock.setHeld(true);
  startWatchdog();
  const root = document.createElement("div");
  document.body.appendChild(root);
  render(h(LoadingScreen, null), root);
  // the card is centred over the map (in the map column, under the top bar; not in the map's own
  // container, which MapLibre wants empty when it starts)
  const card = document.createElement("div");
  document.getElementById("main")?.appendChild(card);
  render(h(LoadingCard, null), card);
  // While the failure screen is up, keys go nowhere but its buttons (the map's, the tools', space
  // for the clock)
  blockKeysOutside(root, () => startup.peek() === "failed");
}
