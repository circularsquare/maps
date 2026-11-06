// T-1491: F2 hides the whole interface for screenshots (the html.ui-hidden rules in style.css);
// F2 or Esc brings it back. The map, the network and the bubbles stay.

import { live } from "../map/live";
import { helpOpen } from "./HelpPanel";
import { HIDE_UI_KEY } from "./toolKeys";
import { t } from "../i18n";

const CLASS = "ui-hidden";

function set(hidden: boolean) {
  const root = document.documentElement;
  if (root.classList.contains(CLASS) === hidden) return;
  root.classList.toggle(CLASS, hidden);
  if (hidden) { helpOpen.value = false; showHint(); } // T-1893: the reminder shows at once, not only on a mouse move
  else { clearTimeout(hintTimer); hint?.classList.remove("on"); }
  live.map?.resize(); // the map takes the dock's and the bar's room
}

// T-1548, T-1893: with the interface hidden, hiding it and then moving the mouse show a small reminder
// at the top centre until the mouse has been still for HINT_MS; it is hidden outright, with no fade.
const HINT_MS = 3000;
let hint: HTMLElement | null = null;
let hintTimer = 0;

function showHint() {
  if (!document.documentElement.classList.contains(CLASS)) return;
  if (!hint) {
    hint = document.createElement("div");
    hint.className = "ui-hint";
    document.body.appendChild(hint);
  }
  hint.textContent = t("hideUi.hint", { key: HIDE_UI_KEY });
  hint.classList.add("on");
  clearTimeout(hintTimer);
  hintTimer = window.setTimeout(() => hint?.classList.remove("on"), HINT_MS);
}

/** Capture-phase, so Esc comes back before any panel's own Esc; the keys are ignored with a modifier held. */
export function hideUiKey() {
  window.addEventListener("pointermove", showHint, true);
  window.addEventListener("keydown", (e) => {
    const hidden = document.documentElement.classList.contains(CLASS);
    if (e.ctrlKey || e.metaKey || e.altKey || e.shiftKey) return;
    if (e.key === HIDE_UI_KEY) {
      e.preventDefault();
      if (!e.repeat) set(!hidden);
    } else if (e.key === "Escape" && hidden) {
      e.preventDefault();
      e.stopImmediatePropagation();
      set(false);
    }
  }, true);
}
