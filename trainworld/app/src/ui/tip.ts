// The game's own hover tag for the page's DOM (T-210; CLAUDE.md: every hover is the game's own
// tag, never a browser `title` tooltip). An element with `data-tip` gets a tag by the pointer the
// moment the pointer is over it, in the map tags' style (map/demandHover.ts), following the pointer
// and kept inside the window; a "\n" in the text starts a new line. The map's canvas has its own
// tags (demand discs, track and network, capacity), which this leaves alone.
//
// Panels, the top bar, the search box and the map's DOM markers write `data-tip`; SVG drawings
// put it on the shape instead of a <title> child. As a net for markup
// the game does not write (MapLibre's attribution button), a `title` met under the pointer is moved
// to `data-tip` at once, before the browser's own tooltip would show.

import { hoverHeld, isModifierKey } from "./blurHold";

export function startTips() {
  const tag = document.createElement("div");
  tag.className = "tool-tag ui-tag";
  tag.setAttribute("role", "tooltip");
  tag.style.display = "none";
  document.body.appendChild(tag);
  let el: Element | null = null;
  let text = "";
  let px = 0, py = 0;

  const hide = () => {
    if (el) watch.disconnect();
    el = null;
    text = "";
    tag.style.display = "none";
  };
  /** shown by a finger's hold: above the finger, which covers what is below it */
  let lifted = false;
  const place = () => {
    const w = tag.offsetWidth, h = tag.offsetHeight;
    if (lifted) {
      const x = Math.max(4, Math.min(innerWidth - 4 - w, px - w / 2)), y = Math.max(4, py - 36 - h);
      tag.style.transform = `translate(${Math.round(x)}px, ${Math.round(y)}px)`;
      return;
    }
    // below right of the pointer; flipped where it would leave the window
    let x = px + 14, y = py + 16;
    if (x + w > innerWidth - 4) x = Math.max(4, px - 10 - w);
    if (y + h > innerHeight - 4) y = Math.max(4, py - 10 - h);
    tag.style.transform = `translate(${Math.round(x)}px, ${Math.round(y)}px)`;
  };
  const fill = () => {
    const t = el?.getAttribute("data-tip") ?? "";
    if (!t || !el?.isConnected) return hide();
    if (t !== text) {
      text = t;
      tag.replaceChildren(...t.split("\n").map((line) => {
        const d = document.createElement("div");
        d.textContent = line;
        return d;
      }));
    }
    tag.style.display = "";
    place();
  };
  // the text can change under a still pointer (the clock's day, a count updating)
  const watch = new MutationObserver(fill);

  const move = (e: PointerEvent) => {
    if (e.pointerType === "touch") return;
    lifted = false;
    if (e.buttons) return hide();
    px = e.clientX;
    py = e.clientY;
    const target = e.target instanceof Element ? e.target : null;
    const found = target?.closest("[data-tip], [title]") ?? null;
    if (found?.hasAttribute("title")) {
      const t = found.getAttribute("title");
      found.removeAttribute("title");
      if (t && !found.hasAttribute("data-tip")) found.setAttribute("data-tip", t);
    }
    if (found !== el) {
      if (el) watch.disconnect();
      el = found;
      text = "";
      if (!el) return hide();
      watch.observe(el, { attributes: true, attributeFilter: ["data-tip"] });
    }
    fill();
  };
  document.addEventListener("pointermove", move, { capture: true, passive: true });
  document.addEventListener("pointerover", move, { capture: true, passive: true });
  // a press, a scroll, a key or the pointer leaving the page puts it away; the next move brings it back
  document.addEventListener("pointerdown", hide, true);
  document.addEventListener("wheel", hide, { capture: true, passive: true });
  document.addEventListener("scroll", hide, true);
  document.addEventListener("keydown", (e) => isModifierKey(e) || hide(), true);
  document.documentElement.addEventListener("pointerleave", () => hoverHeld() || hide());

  // T-1260: a finger has no hover. A press held still on a control for HOLD_MS shows its tag above
  // the finger, and the tap it ends in does nothing; the tag stays a moment after the finger lifts.
  const HOLD_MS = 500, SLOP_PX = 10, LINGER_MS = 1500;
  let hold: { timer: number; x: number; y: number } | null = null;
  let swallow = false, linger = 0;
  const drop = () => { if (hold) clearTimeout(hold.timer); hold = null; };
  document.addEventListener("pointerdown", (e) => {
    drop();
    clearTimeout(linger);
    swallow = false; // a hold that ended without a click (a long press can end in none)
    if (e.pointerType !== "touch") return;
    const found = e.target instanceof Element ? e.target.closest("[data-tip]") : null;
    if (!found) return;
    const x = e.clientX, y = e.clientY;
    hold = { x, y, timer: setTimeout(() => {
      hold = null;
      if (!found.isConnected) return;
      el = found;
      text = "";
      px = x;
      py = y;
      lifted = true;
      swallow = true;
      fill();
    }, HOLD_MS) as unknown as number };
  }, true);
  document.addEventListener("pointermove", (e) => {
    if (hold && e.pointerType === "touch" && Math.hypot(e.clientX - hold.x, e.clientY - hold.y) > SLOP_PX) drop();
  }, { capture: true, passive: true });
  const lift = () => {
    drop();
    if (lifted && el) linger = setTimeout(hide, LINGER_MS) as unknown as number;
  };
  document.addEventListener("pointerup", lift, true);
  document.addEventListener("pointercancel", lift, true);
  document.addEventListener("click", (e) => {
    if (!swallow) return;
    swallow = false;
    e.preventDefault();
    e.stopPropagation();
  }, true);
  document.addEventListener("contextmenu", (e) => { if (swallow) e.preventDefault(); }, true);
}
