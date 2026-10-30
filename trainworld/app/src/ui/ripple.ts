// The Getting started ripple (T-1822 stage 2, notes/T-1822.md "The ripple"): a rectangle grown round
// the control the step in hand uses. One fixed canvas over everything, no pointer events, its own
// requestAnimationFrame loop (no CSS transitions), the target's box read from getBoundingClientRect
// every frame so a scrolling dock keeps it on target. Targets carry `data-guide="<step 1-6>"`; the
// tab icons carry `data-guide-tab`. A target out of view is replaced by what brings it into view.

import { effect } from "@preact/signals";
import { guideShown, guideSteps, guideTicks } from "../game/guide";
import { moneyState } from "../game/money";
import { stopEditing, tab } from "../game/ui";
import { world } from "../game/world";

/** a pulse is three rings, RING_GAP_MS apart, each growing GROW_PX over RING_MS while it fades */
const RINGS = 3, RING_GAP_MS = 250, RING_MS = 700, GROW_PX = 16, LINE_PX = 2;
/** reduced motion: one still outline, STILL_PX out, held STILL_MS */
const STILL_PX = 8, STILL_MS = 1500;
/** a step in hand pulses again after this long without input, at most IDLE_MAX times per step */
const IDLE_MS = 20000, IDLE_MAX = 3;
/** a step that becomes the step in hand waits this long, so the card's tick lands first */
const SETTLE_MS = 350;

const reduced = matchMedia("(prefers-reduced-motion: reduce)");

interface Pulse { step: number; at: number; colour: string; el: Element | null; sel: string }

let canvas: HTMLCanvasElement | null = null;
let pulse: Pulse | null = null;
let looping = false;

/** the pointer is held (a drag in progress, wherever it started) */
let held = false;
let lastInput = performance.now();
/** a step waiting to pulse once nothing blocks it, and when it may */
let pending: { step: number; from: number } | null = null;
let idleStep = -1, idleCount = 0;

/** The selector of the target of a step, tried in order: the first one on the page that can be seen. */
function selectors(step: number): string[] {
  const own = `[data-guide="${step + 1}"]`;
  const t = tab.peek();
  switch (step) {
    case 0: case 1: case 2: return [own, '[data-guide-tab="build"]'];
    case 3: return [own, '[data-guide-tab="lines"]'];
    // the line panel's controls need a line picked: the Lines tab's first row, or the tab itself
    default: return t === "lines" ? [own, "#dock .pane .rows li"] : [own, '[data-guide-tab="lines"]'];
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
function findTarget(step: number): { el: Element; sel: string } | null {
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

/** Whether a pulse may start now: the card is up, and the player is not mid-drag or in the stop editor on step 5. */
function blocked(step: number): boolean {
  const card = document.querySelector(".guide");
  if (!card || getComputedStyle(card).visibility === "hidden" || document.hidden) return true;
  return held || (step === 4 && stopEditing.peek());
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
  const g = cv.getContext("2d")!;
  g.setTransform(1, 0, 0, 1, 0, 0);
  g.clearRect(0, 0, w, h);
  const p = pulse;
  const total = reduced.matches ? STILL_MS : (RINGS - 1) * RING_GAP_MS + RING_MS;
  if (!p || now - p.at > total) { pulse = null; looping = false; return; }
  if (!p.el || !p.el.isConnected) p.el = document.querySelector(p.sel);
  if (p.el) {
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
      for (let k = 0; k < RINGS; k++) {
        const t = (now - p.at - k * RING_GAP_MS) / RING_MS;
        if (t < 0 || t >= 1) continue;
        ring(GROW_PX * (1 - (1 - t) * (1 - t)), 1 - t);
      }
    }
  }
  requestAnimationFrame(frame);
}

function start(step: number, found: { el: Element; sel: string }) {
  const colour = getComputedStyle(document.documentElement).getPropertyValue("--accent").trim() || "#d9788e";
  pulse = { step, at: performance.now(), colour, el: found.el, sel: found.sel };
  if (!looping) { looping = true; requestAnimationFrame(frame); }
}

/** Ripple the control a step uses, now, unless the player is busy with a drag or the stop editor. */
export function ripple(step: number) {
  if (blocked(step)) return;
  const found = findTarget(step);
  if (found) start(step, found);
}

/** Watch the step in hand and the player's input, and pulse by the timing rules. */
export function startRipple() {
  const touch = () => { lastInput = performance.now(); };
  addEventListener("pointerdown", () => { held = true; touch(); }, true);
  const up = () => { held = false; touch(); };
  addEventListener("pointerup", up, true);
  addEventListener("pointercancel", up, true);
  addEventListener("blur", () => { held = false; });
  addEventListener("keydown", touch, true);
  addEventListener("wheel", touch, { capture: true, passive: true });

  // the step in hand changes: pulse once the card's tick has landed
  let seen = -2;
  effect(() => {
    const m = moneyState.value, w = world.value, g = m?.guide;
    if (!m || !w || !g || !guideShown(g, m.mode === "sandbox")) { seen = -2; pending = null; return; }
    const now = guideTicks(guideSteps(w), g).now;
    if (now === seen) return;
    seen = now;
    pending = now >= 0 ? { step: now, from: performance.now() + SETTLE_MS } : null;
    if (now !== idleStep) { idleStep = now; idleCount = 0; }
  });

  // pending pulses wait for nothing to block them; an open step pulses again when the player has been idle
  setInterval(() => {
    const t = performance.now();
    if (pending && t >= pending.from && !blocked(pending.step)) {
      const s = pending.step;
      pending = null;
      ripple(s);
      return;
    }
    if (idleStep >= 0 && idleCount < IDLE_MAX && !pending && t - lastInput >= IDLE_MS && (!pulse || t - pulse.at >= IDLE_MS) && !blocked(idleStep)) {
      idleCount++;
      lastInput = t;
      ripple(idleStep);
    }
  }, 250);
}
