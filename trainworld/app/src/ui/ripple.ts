// The tutorial's ripple (T-1822 stage 2, T-2123; notes/T-1822.md "The ripple"): rectangles grown
// round the control the step in hand uses, one every RING_GAP_MS until the player clicks that control.
// One fixed canvas over everything, no pointer events, its own requestAnimationFrame loop (no CSS
// transitions), the target's box read from getBoundingClientRect every frame so a scrolling dock
// keeps it on target. Targets carry `data-guide="<step id>"`; the tab icons carry `data-guide-tab`.
// A target out of view is replaced by what brings it into view, and the ripple moves to the target
// once it shows (the tab opened).

import { effect } from "@preact/signals";
import { GUIDE_SPLIT, guideShown, guideSteps, guideTicks, type GuideId } from "../game/guide";
import { moneyState } from "../game/money";
import { stopEditing, tab } from "../game/ui";
import { world } from "../game/world";

/** What a pulse points at: a step's control, or "end", the card's Continue or Done! once every step
 * is ticked (T-2263). */
type Target = GuideId | "end";

/** a ring grows GROW_PX over RING_MS while it fades; rings start RING_GAP_MS apart (T-2123 halved
 * T-1822's speed, T-2263 halved it again), RINGS of them in an idle reminder, without end while a
 * step waits for its click */
const RINGS = 3, RING_GAP_MS = 1000, RING_MS = 2800, GROW_PX = 16, LINE_PX = 2;
/** reduced motion: one still outline, STILL_PX out, held STILL_MS (or until the click) */
const STILL_PX = 8, STILL_MS = 1500;
/** once its control was clicked, a step in hand pulses again after this long without input, at most
 * IDLE_MAX times per step */
const IDLE_MS = 20000, IDLE_MAX = 3;
/** a step that becomes the step in hand waits this long, so the card's tick lands first */
const SETTLE_MS = 350;

const reduced = matchMedia("(prefers-reduced-motion: reduce)");

/** `loop`: rings keep coming until the step's own control is clicked */
interface Pulse { step: Target; at: number; colour: string; el: Element | null; sel: string; loop: boolean }

let canvas: HTMLCanvasElement | null = null;
let pulse: Pulse | null = null;
let looping = false;

/** the pointer is held (a drag in progress, wherever it started) */
let held = false;
let lastInput = performance.now();
/** a step waiting to pulse once nothing blocks it, and when it may */
let pending: { step: Target; from: number } | null = null;
let idleStep: Target | null = null, idleCount = 0;

/** The selectors of the target of a step, tried in order: the first one on the page that can be seen. */
function selectors(step: Target): string[] {
  const own = `[data-guide="${step}"]`;
  switch (step) {
    case "zoom": return [own, ".map-search input"]; // i18n-ignore (selector)
    case "split": return [GUIDE_SPLIT];
    case "track": case "stations": case "construct": return [own, '[data-guide-tab="build"]'];
    case "line": return [own, '[data-guide-tab="lines"]'];
    case "destinations": return [own, '[data-fold="bubbles"]'];
    case "traffic": return [own, '[data-fold="track"]'];
    case "play": case "end": return [own];
    // on the map itself, nothing to point at
    case "bubble": case "trip": case "star": return [];
    // the line panel's controls need a line picked: the Lines tab's first row, or the tab itself
    default: return tab.peek() === "lines" ? [own, "#dock .pane .rows li"] : [own, '[data-guide-tab="lines"]']; // i18n-ignore (selector)
  }
}

/** Whether the box of `el` shows: not empty, and not scrolled out of a clipping ancestor. */
function inView(el: Element): "yes" | "scrolled" | "no" {
  const r = el.getBoundingClientRect();
  if (r.width <= 0 || r.height <= 0) return "no";
  for (let p = el.parentElement; p; p = p.parentElement) {
    const o = getComputedStyle(p);
    if (o.overflowY === "visible" && o.overflowX === "visible") continue;
    const c = p.getBoundingClientRect();
    if (r.top < c.top - 1 || r.bottom > c.bottom + 1 || r.left < c.left - 1 || r.right > c.right + 1) return "scrolled";
  }
  return "yes";
}

/** The element to ripple for a step, scrolling a target that sits scrolled out of its pane into view. */
function findTarget(step: Target): { el: Element; sel: string } | null {
  for (const sel of selectors(step)) {
    const el = document.querySelector(sel);
    if (!el) continue;
    const v = inView(el);
    if (v === "no") continue;
    if (v === "scrolled") {
      el.scrollIntoView({ block: "nearest", inline: "nearest" });
      if (inView(el) !== "yes") continue;
    }
    return { el, sel };
  }
  return null;
}

/** Whether the card is out of sight (hidden behind a dialog, F2, or a hidden page). */
function cardAway(): boolean {
  const card = document.querySelector(".guide");
  return !card || getComputedStyle(card).visibility === "hidden" || document.hidden;
}

/** Whether a pulse may start now: the card is up, and the player is not mid-drag or in the stop editor on the stops step. */
function blocked(step: Target): boolean {
  return cardAway() || held || (step === "stops" && stopEditing.peek());
}

function ensureCanvas(): HTMLCanvasElement {
  if (canvas) return canvas;
  canvas = document.createElement("canvas");
  canvas.className = "guide-ripple";
  canvas.setAttribute("aria-hidden", "true");
  canvas.style.cssText = "position:fixed;left:0;top:0;width:100vw;height:100vh;pointer-events:none;z-index:99999";
  document.body.appendChild(canvas);
  return canvas;
}

function frame(now: number) {
  const cv = ensureCanvas();
  const dpr = devicePixelRatio || 1, w = Math.round(innerWidth * dpr), h = Math.round(innerHeight * dpr);
  if (cv.width !== w || cv.height !== h) { cv.width = w; cv.height = h; }
  // T-2795: 100vw/100vh is the large viewport on a phone (its address bar hidden), taller than innerHeight, which
  // stretched the drawing and put rings an inch off; the box is the drawing's own size so one unit is one css px
  const cw = innerWidth + "px", ch = innerHeight + "px";
  if (cv.style.width !== cw) cv.style.width = cw;
  if (cv.style.height !== ch) cv.style.height = ch;
  const g = cv.getContext("2d")!;
  g.setTransform(1, 0, 0, 1, 0, 0);
  g.clearRect(0, 0, w, h);
  const p = pulse;
  const total = reduced.matches ? STILL_MS : (RINGS - 1) * RING_GAP_MS + RING_MS;
  if (!p || (!p.loop && now - p.at > total)) { pulse = null; looping = false; return; }
  if (p.el && !p.el.isConnected) p.el = p.sel ? document.querySelector(p.sel) : null;
  if (p.el && !blocked(p.step)) {
    const r = p.el.getBoundingClientRect();
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.strokeStyle = p.colour;
    g.lineWidth = LINE_PX;
    const ring = (grow: number, alpha: number) => {
      g.globalAlpha = alpha;
      g.strokeRect(r.left - grow, r.top - grow, r.width + 2 * grow, r.height + 2 * grow);
    };
    if (reduced.matches) ring(STILL_PX, 1);
    else {
      const since = now - p.at;
      const last = p.loop ? Math.floor(since / RING_GAP_MS) : RINGS - 1;
      for (let k = Math.max(0, Math.floor((since - RING_MS) / RING_GAP_MS)); k <= last; k++) {
        const t = (since - k * RING_GAP_MS) / RING_MS;
        if (t < 0 || t >= 1) continue;
        ring(GROW_PX * (1 - (1 - t) * (1 - t)), 1 - t);
      }
    }
  }
  requestAnimationFrame(frame);
}

function start(step: Target, found: { el: Element; sel: string } | null, loop: boolean) {
  const colour = getComputedStyle(document.documentElement).getPropertyValue("--accent").trim() || "#d9788e";
  pulse = { step, at: performance.now(), colour, el: found?.el ?? null, sel: found?.sel ?? "", loop };
  if (!looping) { looping = true; requestAnimationFrame(frame); }
}

/** Ripple the control a step uses, from now until it is clicked, unless the player is busy with a
 * drag or the stop editor. A step with nothing to point at does nothing. */
export function ripple(step: Target) {
  if (blocked(step) || selectors(step).length === 0) return;
  start(step, findTarget(step), true);
}

/** A looping pulse on a stand-in (a tab icon) or on nothing moves to the step's own control once it
 * shows; one whose control went away falls back to the stand-in. */
function retarget(p: Pulse) {
  const own = `[data-guide="${p.step}"]`;
  const ok = p.el && p.el.isConnected && inView(p.el) !== "no";
  if (ok && p.sel === own) return;
  const ownEl = document.querySelector(own);
  if (ok && !(ownEl && inView(ownEl) !== "no")) return;
  const found = findTarget(p.step);
  if (found && found.el !== p.el) { p.el = found.el; p.sel = found.sel; }
  else if (!found) { p.el = null; p.sel = ""; }
}

/** Watch the step in hand and the player's input, and pulse by the timing rules. */
export function startRipple() {
  const touch = () => { lastInput = performance.now(); };
  addEventListener("pointerdown", (e) => {
    held = true;
    touch();
    // the step's own control clicked: its rings stop
    const p = pulse;
    if (p?.loop && (e.target as Element | null)?.closest?.(`[data-guide="${p.step}"]`)) pulse = null;
  }, true);
  const up = () => { held = false; touch(); };
  addEventListener("pointerup", up, true);
  addEventListener("pointercancel", up, true);
  addEventListener("blur", () => { held = false; });
  addEventListener("keydown", touch, true);
  addEventListener("wheel", touch, { capture: true, passive: true });

  // the step in hand changes: the last one's rings stop, and the new one's start once the card's tick has landed
  let seen: Target | undefined;
  effect(() => {
    const m = moneyState.value, w = world.value, g = m?.guide;
    if (!m || !w || !g || !guideShown(g, m.mode === "sandbox")) { seen = undefined; pending = null; pulse = null; return; }
    const steps = guideSteps(w, g);
    const i = guideTicks(steps, g).now;
    const now: Target = i >= 0 ? steps[i].id : "end";
    if (now === seen) return;
    seen = now;
    if (pulse && pulse.step !== now) pulse = null;
    pending = now ? { step: now, from: performance.now() + SETTLE_MS } : null;
    if (now !== idleStep) { idleStep = now; idleCount = 0; }
  });

  // pending pulses wait for nothing to block them; a looping pulse follows its target; once its
  // control was clicked, an open step pulses again when the player has been idle
  setInterval(() => {
    const t = performance.now();
    if (pending && t >= pending.from && !blocked(pending.step)) {
      const s = pending.step;
      pending = null;
      ripple(s);
      return;
    }
    if (pulse?.loop) { retarget(pulse); return; }
    if (idleStep && idleCount < IDLE_MAX && !pending && t - lastInput >= IDLE_MS && (!pulse || t - pulse.at >= IDLE_MS) && !blocked(idleStep)) {
      const found = selectors(idleStep).length ? findTarget(idleStep) : null;
      idleCount++;
      lastInput = t;
      if (found) start(idleStep, found, false);
    }
  }, 250);
}
