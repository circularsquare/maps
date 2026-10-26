// T-1491: F2 hides the whole interface for screenshots (the html.ui-hidden rules in style.css);
// F2 or Esc brings it back. The map, the network and the bubbles stay.

import { live } from "../map/live";
import { helpOpen } from "./HelpPanel";
import { HIDE_UI_KEY } from "./toolKeys";

const CLASS = "ui-hidden";

function set(hidden: boolean) {
  const root = document.documentElement;
  if (root.classList.contains(CLASS) === hidden) return;
  root.classList.toggle(CLASS, hidden);
  if (hidden) helpOpen.value = false;
  live.map?.resize(); // the map takes the dock's and the bar's room
}

/** Capture-phase, so Esc comes back before any panel's own Esc; the keys are ignored with a modifier held. */
export function hideUiKey() {
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
