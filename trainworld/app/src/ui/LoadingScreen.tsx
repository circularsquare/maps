// Startup (T-149, T-172, T-180, SPEC 8). Before it, the start menu (ui/StartMenu.tsx, T-677).
// The game is playable from the moment the map and network
// are drawn: the player can look around, inspect and build while demand loads. Only the clock
// waits (game/loading.ts). Meanwhile a card in the middle of the map says what is loading and how
// far it has got (the stage and a filling bar, T-243, T-1194); it takes no key and no pointer but its bar's, and goes when the clock
// starts. If demand cannot load, a screen says so and offers a way on; it takes every key until then.
// T-950: so does a load that can never finish (the game itself not starting, or nothing moving).

import { h, render } from "preact";
import { useEffect, useRef, useState } from "preact/hooks";
import { clock } from "../game/clock";
import { world } from "../game/world";
import { cardDismissed, cardEnding, choosing, clockStepNow, continueWithoutDemand, demandIdle, dismissLoadingCard, gameFailed, gameLoadSeq, graphicsGone, lastAlive, loadCannotFinish, loadFailure, loadGraphicsGone, loadStepNow, loadStuck, loadProgress, noSimd, noWebgl2, startup, unitQuietMs } from "../game/loading";
import { contextsLost, freshContextWorks } from "../map/graphicsInfo";
import { GL_LOST } from "../map/gl";
import { leftText, STAGE_TEXT } from "../game/loadBar";
import { isMemoryFailure, loadFailText } from "../game/loadText";
import { leaveToMenu, reloadIntoGame } from "../workers/clockClient";
import { marketOff } from "../game/ui";
import { playWithoutIntercity } from "./DemandStopped";
import { FeedbackOpen, feedbackOpen, startFeedbackUi } from "./FeedbackPanel";
import { blockKeysOutside } from "./keys";
import { BunnyTrain } from "./Spinner";
import { Icon } from "./widgets";
import { t, tNumber } from "../i18n";

function LoadingScreen() {
  // T-3582: the browser refuses the page graphics, at startup or mid-game: the same screen, a restart sentence
  if (startup.value !== "failed" && !graphicsGone.value) return null;
  // T-3307: no WebGL2: what to do about it, and no retry or way on (neither can help); T-3675: only a way to report it
  // T-3315: no WASM SIMD (Safari before 16.4): the same, with its own sentence
  if (graphicsGone.value || noWebgl2.value || noSimd.value) {
    const text = t(graphicsGone.value ? "loadText.graphicsGone" : noWebgl2.value ? "loadText.noWebgl2" : "loadText.noSimd");
    return <div class="loading-screen" role="alertdialog" aria-label={text}>
      <div class="loading-box failed">
        <p class="big">{text}</p>
        <div class="loading-actions"><FeedbackOpen class="btn" /></div>
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
      {why && why !== text && <p class="small muted load-why">{why}</p>}
      <div class="loading-actions">
        <button class="btn" onClick={() => reloadIntoGame(undefined, true)}>{t("loadingScreen.tryAgain")}</button>
        {/* T-3630: where a failed load leaves the player; the report carries the failure's text */}
        <FeedbackOpen class="btn" />
        {named && <button class="btn" onClick={() => leaveToMenu(() => {}, true)}>{t("loadingScreen.backToStart")}</button>}
        {/* T-2867: out of memory, the way on keeps demand and drops intercity trips, while there are any to drop */}
        {!game && (isMemoryFailure(why) && !marketOff()
          ? <button class="btn" onClick={() => playWithoutIntercity()}>{t("loadingScreen.playWithoutIntercity")}</button>
          : <button class="btn" onClick={continueWithoutDemand}>{t("loadingScreen.playWithoutDemand")}</button>)}
      </div>
    </div>
  </div>;
}

/** True while a loaded game's clock waits for its demand (T-180), whether the card is up or closed (T-4346). */
export function loadingHeld(): boolean {
  return clock.held.value && !loadFailure.value && !demandIdle.value && !choosing.value; // T-318: nothing to load over dead space
}

/** True while the loading card is up in the middle of the map (T-180, T-4346). */
export function loadingCardShown(): boolean {
  return loadingHeld() && !cardDismissed.value;
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
 * the map. Pointer events pass through it to the map, except on the bar (its hover), and once the
 * map and network are in and only trips remain (T-4346: STAGE_TEXT.demand), a drag moves the card
 * anywhere in the window and its x closes it into the top status bar.
 * T-677: the bunny's train on top (only here; the small statuses have their bar alone, T-2728). T-1194: it names the stage in hand (game/loadBar.ts), "Loading"
 * until the demand pool has a load to describe. */
function LoadingCard() {
  // (T-2574: a load ended at a cached result shows its bar full a moment first)
  const up = loadingCardShown(), ending = cardEnding.value;
  const shown = up || (!!ending && !loadFailure.value);
  const load = up ? loadProgress.value : ending;
  const seq = gameLoadSeq.value;
  const canDismiss = up && !!world.value && load?.step === STAGE_TEXT.demand;
  const [pos, setPos] = useState<{ x: number; y: number } | null>(null);
  const [dragging, setDragging] = useState(false);
  const cardRef = useRef<HTMLDivElement>(null);
  const dragRef = useRef<{ id: number; startX: number; startY: number; origX: number; origY: number; w: number; h: number } | null>(null);

  useEffect(() => {
    setPos(null);
    setDragging(false);
    dragRef.current = null;
  }, [seq, shown]);

  useEffect(() => {
    if (!pos) return;
    const onResize = () => {
      const el = cardRef.current;
      if (!el) return;
      const r = el.getBoundingClientRect();
      setPos((p) => p && {
        x: Math.max(0, Math.min(window.innerWidth - r.width, p.x)),
        y: Math.max(0, Math.min(window.innerHeight - r.height, p.y)),
      });
    };
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, [!!pos]);

  if (!shown) return null;
  const step = load?.step ?? STAGE_TEXT.game;
  const fraction = load?.fraction ?? 0;
  const left = load?.leftMs;

  const onDown = (e: PointerEvent) => {
    if (!canDismiss || e.button !== 0) return;
    const el = cardRef.current;
    if (!el) return;
    const r = el.getBoundingClientRect();
    dragRef.current = { id: e.pointerId, startX: e.clientX, startY: e.clientY, origX: r.left, origY: r.top, w: r.width, h: r.height };
    (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
    setDragging(true);
  };
  const onMove = (e: PointerEvent) => {
    const d = dragRef.current;
    if (!d || d.id !== e.pointerId) return;
    const x = Math.max(0, Math.min(window.innerWidth - d.w, d.origX + e.clientX - d.startX));
    const y = Math.max(0, Math.min(window.innerHeight - d.h, d.origY + e.clientY - d.startY));
    setPos({ x, y });
  };
  const onUp = (e: PointerEvent) => {
    const d = dragRef.current;
    if (!d || d.id !== e.pointerId) return;
    dragRef.current = null;
    setDragging(false);
  };

  // T-1479: the share done and, once there is an estimate, the time left
  const posStyle = pos ? { position: "fixed" as const, left: `${pos.x}px`, top: `${pos.y}px` } : undefined;
  return <div class={"loading-card-wrap" + (pos ? " dragged" : "")}>
    <div class="loading-card-pos" style={posStyle}>
      <div class={"loading-card" + (canDismiss ? " movable" : "") + (dragging ? " dragging" : "")} role="status" ref={cardRef}
        onPointerDown={canDismiss ? onDown : undefined} onPointerMove={onMove} onPointerUp={onUp} onPointerCancel={onUp}>
        {canDismiss && <button type="button" class="btn icon loading-card-close" aria-label={t("modal.close")}
          onPointerDown={(e) => e.stopPropagation()} onClick={() => dismissLoadingCard()}>
          <Icon.cross />
        </button>}
        <BunnyTrain />
        <p class="big">{step}</p>
        <LoadBar fraction={fraction} step={step} />
        <p class="load-left small muted num"><span>{Math.floor(fraction * 100)}%</span>{left != null && <span>{leftText(left)}</span>}</p>
      </div>
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
export function setupDone() {
  setupFinished = true;
  try { sessionStorage.removeItem(GL_RELOAD_KEY); } catch { /* storage unavailable */ } // T-4073: a later loss may reload once again
}

function startWatchdog() {
  const q = Number(new URLSearchParams(location.search).get("quietMs"));
  // (T-4343: and while a demand worker is inside a period, as long as the periods here take; `?quietMs` stays exact)
  const quietMs = (bar: boolean) => (q > 0 ? q : Math.max(bar ? QUIET_BAR_MS : QUIET_MS, unitQuietMs()));
  let from = performance.now(), lastCheck = from;
  const timer = setInterval(() => {
    const now = performance.now();
    if (loadFailure.peek()) return clearInterval(timer);
    if (!loadingHeld() || document.visibilityState !== "visible" || now - lastCheck > 3 * WATCH_MS) from = now;
    lastCheck = now;
    const limit = quietMs(!!loadProgress.peek());
    if (now - Math.max(from, lastAlive()) < limit) return;
    const started = !!world.peek();
    // T-3781: a stuck demand load is restarted once before the failure screen
    if (started && loadStuck()) return;
    const m = Math.round(limit / 60_000), mins = m >= 1 ? t("loadingScreen.minutes", { count: m }) : t("loadingScreen.seconds", { count: Math.round(limit / 1000) });
    // T-4072: where it stuck, for the report (the clock worker's step before its game is in, else the load's last)
    const step = started ? loadStepNow() : clockStepNow();
    loadCannotFinish(t(started ? "loadingScreen.demandTimeout" : "loadingScreen.simTimeout", { time: mins }) + (step ? ` (${step})` : ""), !started); // i18n-ignore: the step is for the report
  }, WATCH_MS);
  window.addEventListener("error", (e) => {
    if (setupFinished || startupFailed) return;
    startupFailed = true;
    const reason = e.error instanceof Error ? e.error.message : e.message;
    // T-4073: the graphics were lost while the page started (Safari after its GPU process went down: every WebGL
    // answer is null, so a shader "fails"): the graphics' failure, not the page's
    if (reason?.includes(GL_LOST) || contextsLost() || !freshContextWorks()) graphicsLostAtStart(reason);
    else loadCannotFinish(t("loadingScreen.pageStopped", { reason }), true);
  });
}
/** the page's setup already failed (only its first error is the cause) */
let startupFailed = false;

/** T-4073: how long a page whose graphics were lost as it started waits for them to come back, and how often it asks */
const GRAPHICS_BACK_MS = 8_000, GRAPHICS_ASK_MS = 500;
const GL_RELOAD_KEY = "tw-gl-reload";
/** T-4073: the setup stopped on lost graphics. Once they work again the page starts afresh, once per tab session (a
 * page set up halfway cannot rebuild what it skipped); if they do not, or it happened again after that reload, the
 * graphics screen says to restart the browser (T-3582). Nothing was loaded yet, so nothing is lost by the reload. */
function graphicsLostAtStart(reason: string) {
  console.warn("graphics lost while the page started:", reason); // i18n-ignore
  let reloaded = false;
  try { reloaded = sessionStorage.getItem(GL_RELOAD_KEY) === "1"; } catch { /* storage unavailable */ }
  const giveUp = () => loadGraphicsGone(`lost while the page started: ${reason}`); // i18n-ignore
  if (reloaded) return giveUp();
  const until = performance.now() + GRAPHICS_BACK_MS;
  const timer = setInterval(() => {
    if (freshContextWorks()) {
      clearInterval(timer);
      try { sessionStorage.setItem(GL_RELOAD_KEY, "1"); } catch { /* storage unavailable: no second try either way */ }
      location.reload();
    } else if (performance.now() > until) {
      clearInterval(timer);
      giveUp();
    }
  }, GRAPHICS_ASK_MS);
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
  // (T-3630: but the feedback form, a dialog of its own, takes its typing)
  blockKeysOutside(root, () => (startup.peek() === "failed" || graphicsGone.peek()) && !feedbackOpen.peek());
  startFeedbackUi();
}
