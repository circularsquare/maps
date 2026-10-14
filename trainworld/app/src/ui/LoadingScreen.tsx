// Startup (T-149, T-172, T-180, SPEC 8). The game is playable from the moment the map and network
// are drawn: the player can look around, inspect and build while demand loads. Only the clock
// waits (game/loading.ts). Meanwhile a card in the middle of the map says what is loading and how
// far it has got ("Loading data", a filling bar and the step, T-212, T-243); it takes no pointer or key, and goes when the clock
// starts. If demand cannot load, a screen says so and offers a way on; it takes every key until then.

import { h, render } from "preact";
import { clock } from "../game/clock";
import { continueWithoutDemand, demandIdle, loadFailure, loadProgress, startup } from "../game/loading";

function LoadingScreen() {
  if (startup.value !== "failed") return null;
  return <div class="loading-screen" role="alertdialog" aria-label="Demand could not load">
    <div class="loading-box">
      <p class="big">Demand could not load</p>
      <p class="muted small">{loadFailure.value}</p>
      <div class="loading-actions">
        <button class="btn" onClick={() => location.reload()}>Try again</button>
        <button class="btn" onClick={continueWithoutDemand}>Play without demand</button>
      </div>
    </div>
  </div>;
}

/** True while the loading card is up: a loaded game's clock waits for its demand (T-180). */
export function loadingCardShown(): boolean {
  return clock.held.value && !loadFailure.value && !demandIdle.value; // T-318: nothing to load over dead space
}

/** T-243: a bar filling as the load goes, in place of a percentage; the percentage is its hover.
 * The fill jumps to each new value (no transition). */
export function LoadBar({ fraction, step, thin }: { fraction: number; step: string; thin?: boolean }) {
  const pct = Math.floor(fraction * 100);
  return <span class={"load-bar" + (thin ? " thin" : "")} role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={pct}
    data-tip={`${pct}% loaded\n${step}`}>
    <i class="load-fill" style={{ width: `${fraction * 100}%` }} />
  </span>;
}

/** T-180: while a loaded game's clock waits for its first demand estimate, a card in the middle of
 * the map. Pointer events pass through it to the map, except on the bar (its hover). No animation:
 * the bar's fill is the motion. */
function LoadingCard() {
  if (!loadingCardShown()) return null;
  const load = loadProgress.value;
  return <div class="loading-card-wrap">
    <div class="loading-card" role="status">
      {/* T-212: "Loading data", not a list of the cities being loaded */}
      <p class="big">{load ? "Loading data" : "Calculating demand"}</p>
      {load && <LoadBar fraction={load.fraction} step={load.step} />}
      {/* T-253: no "The clock starts when demand has loaded" line under "Calculating demand" */}
      {load && <p class="muted small">{load.step}</p>}
    </div>
  </div>;
}

/** Hold the clock until the first game's demand has its first estimate; say so if it fails. */
export function startLoadingGate() {
  clock.setHeld(true);
  const root = document.createElement("div");
  document.body.appendChild(root);
  render(h(LoadingScreen, null), root);
  // the card is centred over the map (in the map column, under the top bar; not in the map's own
  // container, which MapLibre wants empty when it starts)
  const card = document.createElement("div");
  document.getElementById("main")?.appendChild(card);
  render(h(LoadingCard, null), card);
  // While the failure screen is up, keys go nowhere but its buttons (the map's, the tools', space
  // for the clock); captured at the window before any other listener.
  const block = (e: KeyboardEvent) => {
    if (startup.peek() !== "failed") return;
    if (e.target instanceof Node && root.contains(e.target)) return;
    e.stopImmediatePropagation();
    e.preventDefault();
  };
  window.addEventListener("keydown", block, true);
  window.addEventListener("keyup", block, true);
}
