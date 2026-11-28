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

import { inkOn } from "../game/palette";
import { isLineShape, type LineShape } from "../game/lineShape";
import { bulletNode } from "./bullet";
import { hoverHeld, isModifierKey } from "./blurHold";
import { mobileDevice } from "../workers/deviceMemory";

/** T-1817: a line's bullet inside a hover's text, drawn as a small chip (`tipBullet`), in its shape (T-3319). */
const BULLET = /\u0002([^|\u0003]*)\|([^|\u0003]*)(?:\|([^\u0003]*))?\u0003/;
export const tipBullet = (letter: string, colour: string, shape?: LineShape) => `\u0002${letter.replace(/[\u0002\u0003|]/g, "")}|${colour}${shape ? "|" + shape : ""}\u0003`;

/** T-1926: a line of its own in a hover's text drawn as a thin strip, one cell per colour
 * (`tipStrip`), as wide as the hover's text. T-2334: with `stops`, the cells are a line's stretches,
 * a station dot between each two and at both ends; where the dots would leave a cell less than
 * `STOP_CELL_PX`, only the end dots stay and the cells are thinner pills. */
const STRIP = /^\u0004(\|?)([^\u0005]*)\u0005$/;
/** T-2365: with `weights`, each cell as wide as its weight (a mode split); cells weighing 0 are left out */
export const tipStrip = (colours: string[], stops = false, weights?: number[]) =>
  `\u0004${stops ? "|" : ""}${colours.map((c, k) => (weights ? `${c}@${weights[k] > 0 ? weights[k] : 0}` : c)).join(";")}\u0005`;
/** T-2772: a line of its own under a strip with a name at each end, the first left, the second right, each cut short with an ellipsis */
const ENDS = /^\u000e([^\u001f]*)\u001f([^\u001f]*)$/;
export const tipEnds = (a: string, b: string) => `\u000e${a.replace(/[\u0000-\u001f]/g, " ")}\u001f${b.replace(/[\u0000-\u001f]/g, " ")}`;
const STOP_CELL_PX = 6;
/** a station dot and the gaps either side of it, and the widest a hover's text runs (`.tool-tag`) */
const STOP_DOT_PX = 9, TAG_TEXT_PX = 328;
function stripCells(d: HTMLElement, cells: string[], stops: boolean) {
  const dot = () => { const s = document.createElement("b"); d.append(s); };
  // a strip with a dot at every station widens its hover to fit them while it can
  const want = cells.length * (STOP_CELL_PX + STOP_DOT_PX) + STOP_DOT_PX;
  if (stops && want <= TAG_TEXT_PX) d.style.width = `${want}px`;
  if (stops) dot();
  cells.forEach((c, k) => {
    const [colour, weight] = c.split("@");
    if (weight !== undefined && !(Number(weight) > 0)) return;
    if (stops && k > 0) dot();
    const cell = document.createElement("i");
    cell.style.background = colour;
    if (weight !== undefined) cell.style.flex = `${weight} 1 0`;
    d.append(cell);
  });
  if (stops) dot();
}

/** T-2365: a line of its own in a hover's text drawn as a small table, `rows` of cells, each cell
 * drawn as a line is (text, bullets, a strip); `tipSpan` makes a cell take several columns. The first
 * column is left-aligned, the rest right; a row of one empty cell is a small gap. */
const GRID = /^\u0006([^\u0006]*)\u0006$/;
export const tipGrid = (rows: string[][]) => `\u0006${rows.map((r) => r.join("\u001f")).join("\u001e")}\u0006`;
const SPAN = /^\u0007(\d+)/;
export const tipSpan = (n: number, cell: string) => `\u0007${n}${cell}`;
function gridCells(d: HTMLElement, text: string) {
  const rows = text.split("\u001e").map((r) => r.split("\u001f").map((c) => {
    const m = SPAN.exec(c);
    return m ? { span: Number(m[1]), text: c.slice(m[0].length) } : { span: 1, text: c };
  }));
  const cols = Math.max(1, ...rows.map((r) => r.reduce((s, c) => s + c.span, 0)));
  d.style.gridTemplateColumns = `repeat(${cols}, auto)`; // i18n-ignore (CSS)
  for (const r of rows) {
    if (r.length === 1 && !r[0].text) {
      const gap = document.createElement("div");
      gap.className = "tip-gap";
      gap.style.gridColumn = `span ${cols}`;
      d.append(gap);
      continue;
    }
    r.forEach((c, k) => {
      const cell = lineEl(c.text);
      if (k > 0) cell.classList.add("num");
      if (c.span > 1) cell.style.gridColumn = `span ${c.span}`;
      d.append(cell);
    });
  }
}

/** T-4068: a line of its own in a hover's text drawn in grey (`tipMuted`), for a side note such as a
 * shortcut. A phone has no Shift key, so there the line is left out. */
const MUTED = "\u0008";
export const tipMutedLine = (text: string) => MUTED + text;
export const tipMuted = (text: string) => "\n" + tipMutedLine(text);

/** One line of a hover's text (or a table's cell): a strip, a table, or text with bullets. A bullet
 * with no letter is a plain square of its colour. */
function lineEl(line: string): HTMLElement {
  const d = document.createElement("div");
  if (line.startsWith(MUTED)) {
    d.className = "tip-muted";
    appendText(d, line.slice(MUTED.length));
    return d;
  }
  const strip = STRIP.exec(line);
  if (strip) {
    d.className = strip[1] ? "tip-strip stops" : "tip-strip"; // i18n-ignore (CSS)
    stripCells(d, strip[2].split(";"), !!strip[1]);
    return d;
  }
  const ends = ENDS.exec(line);
  if (ends) {
    d.className = "tip-ends";
    for (const name of [ends[1], ends[2]]) {
      const s = document.createElement("span");
      s.textContent = name;
      d.append(s);
    }
    return d;
  }
  const grid = GRID.exec(line);
  if (grid) {
    d.className = "tip-grid";
    gridCells(d, grid[1]);
    return d;
  }
  const parts = line.split(BULLET);
  for (let i = 0; i < parts.length; i += 4) {
    if (parts[i]) appendWithKeys(d, parts[i]);
    if (i + 3 >= parts.length) break;
    const s = document.createElement(parts[i + 1] ? "span" : "i");
    if (parts[i + 1]) {
      s.className = "chip sm tip-chip"; // i18n-ignore (CSS)
      s.style.setProperty("--c", parts[i + 2]);
      s.style.setProperty("--ct", inkOn(parts[i + 2]));
      const shape = parts[i + 3];
      if (isLineShape(shape)) s.append(bulletNode(parts[i + 1], "sm", shape));
      else s.textContent = parts[i + 1];
    } else {
      s.className = "tip-swatch";
      s.style.background = parts[i + 2];
    }
    d.append(s);
  }
  return d;
}
/** T-2334: a stops strip too narrow for a dot at every station keeps the end ones (`STOP_CELL_PX`) */
function fitStrips(tag: HTMLElement) {
  for (const s of tag.querySelectorAll<HTMLElement>(".tip-strip.stops")) {
    const cells = s.querySelectorAll("i").length, dots = s.querySelectorAll("b");
    if (dots.length < 3) continue;
    const dotW = dots[0].getBoundingClientRect().width + 2;
    if ((s.clientWidth - dots.length * dotW) / cells >= STOP_CELL_PX) continue;
    s.classList.add("dense");
    for (let k = 1; k < dots.length - 1; k++) dots[k].remove();
  }
}

/** T-1835 (T-4327: full-width brackets too, as ja and zh write them): a key named in brackets at the end of a hover's name, "Track (T)", "Build (Shift+1)",
 * "Keep placing (Caps Lock)", is lettered in the theme's accent, the brackets and all faded as a hint (T-2157). */
const KEY_HINT = /([(（])((?:Ctrl\+|Shift\+)?(?:[A-Z0-9/,.]|Esc|Enter|Delete|Space|Backspace|Caps Lock|Tab)|\/ or Ctrl\+F)([)）])/;
/** T-1851: a negative amount of money in a hover ("−$1.2M", the minus `money()` writes) is red, as in the panels. */
const NEG_MONEY = /(−\$[\d.,]+[KMBT]?)/;
function appendText(d: HTMLElement, text: string) {
  const parts = text.split(NEG_MONEY);
  for (let i = 0; i < parts.length; i++) {
    if (!parts[i]) continue;
    if (i % 2 === 0) { d.append(parts[i]); continue; }
    const s = document.createElement("span");
    s.className = "money-neg";
    s.textContent = parts[i];
    d.append(s);
  }
}
/** T-2863: a phone or tablet has no keys, so the hints go (and the space before each) */
const KEY_HINT_GONE = new RegExp(" ?" + KEY_HINT.source, "g");
function appendWithKeys(d: HTMLElement, text: string) {
  if (mobileDevice()) text = text.replace(KEY_HINT_GONE, "");
  const parts = text.split(KEY_HINT);
  // split keeps the three groups of each match: opening bracket, key, closing bracket
  for (let i = 0; i < parts.length; i += 4) {
    appendText(d, parts[i]);
    if (i + 3 >= parts.length) break;
    const hint = document.createElement("span"), k = document.createElement("span");
    hint.className = "tip-hint";
    k.className = "tip-key";
    k.textContent = parts[i + 2];
    hint.append(parts[i + 1], k, parts[i + 3]);
    d.append(hint);
  }
}

export function startTips() {
  const tag = document.createElement("div");
  tag.className = "tool-tag ui-tag";
  tag.setAttribute("role", "tooltip");
  tag.setAttribute("popover", "manual");
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
      tag.style.transform = `translate(${Math.round(x)}px, ${Math.round(y)}px)`; // i18n-ignore (CSS)
      return;
    }
    // below right of the pointer; flipped where it would leave the window
    let x = px + 14, y = py + 16;
    if (x + w > innerWidth - 4) x = Math.max(4, px - 10 - w);
    if (y + h > innerHeight - 4) y = Math.max(4, py - 10 - h);
    tag.style.transform = `translate(${Math.round(x)}px, ${Math.round(y)}px)`; // i18n-ignore (CSS)
  };
  const fill = () => {
    const t = el?.getAttribute("data-tip") ?? "";
    if (!t || !el?.isConnected) return hide();
    const fresh = t !== text;
    if (fresh) {
      text = t;
      tag.replaceChildren(...t.split("\n").filter((l) => !(l.startsWith(MUTED) && mobileDevice())).map(lineEl));
    }
    if (tag.style.display === "none") raise();
    tag.style.display = "";
    if (fresh) fitStrips(tag);
    place();
  };
  // T-1656: a modal dialog (the saves panel) is in the top layer, over any z-index; a popover shown
  // after it is above it, so the tag is put on top each time it appears
  const raise = () => {
    if (typeof tag.showPopover !== "function") return;
    if (tag.matches(":popover-open")) tag.hidePopover();
    tag.showPopover();
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
  document.documentElement.addEventListener("pointerleave", (e) => e.pointerType === "touch" || hoverHeld() || hide()); // a lifted finger leaves too (T-2862)

  // T-1260: a finger has no hover. A press held still on a control for HOLD_MS shows its tag above
  // the finger, and the tap it ends in does nothing; the tag stays a moment after the finger lifts.
  // T-2862: a short tap on a control with a tag shows it as well (the tap still does what it does), and it stays
  // TAP_MS or until the next press anywhere.
  const HOLD_MS = 500, SLOP_PX = 10, LINGER_MS = 1500, TAP_MS = 3000;
  let hold: { timer: number; x: number; y: number; found: Element } | null = null;
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
    hold = { x, y, found, timer: setTimeout(() => {
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
  const lift = (tap?: Element) => {
    drop();
    if (tap?.isConnected) {
      el = tap;
      text = "";
      lifted = true;
      fill();
      linger = setTimeout(hide, TAP_MS) as unknown as number;
      return;
    }
    if (lifted && el) linger = setTimeout(hide, LINGER_MS) as unknown as number;
  };
  document.addEventListener("pointerup", (e) => {
    // a touch press that was not held and did not move is a tap
    const h = e.pointerType === "touch" ? hold : null;
    if (h) { px = h.x; py = h.y; }
    lift(h?.found);
  }, true);
  document.addEventListener("pointercancel", () => lift(), true);
  document.addEventListener("click", (e) => {
    if (!swallow) return;
    swallow = false;
    e.preventDefault();
    e.stopPropagation();
  }, true);
  document.addEventListener("contextmenu", (e) => { if (swallow) e.preventDefault(); }, true);
}
