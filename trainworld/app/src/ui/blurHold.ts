// T-483: a hover stays up when the window loses focus (Win+Shift+S for a screenshot sends a blur
// and often a leave before the snipping overlay is up) until the pointer next moves over the page.

let held = false;

if (typeof window !== "undefined") {
  window.addEventListener("blur", () => (held = true));
  document.addEventListener("pointermove", () => (held = false), { capture: true, passive: true });
}

/** Whether hovers should ignore a leave just now. */
export const hoverHeld = () => held;

/** Keys that never put a hover away: modifiers, pressed on their own or to start a system shortcut. */
export const isModifierKey = (e: KeyboardEvent) =>
  e.key === "Shift" || e.key === "Control" || e.key === "Alt" || e.key === "Meta" || e.key === "OS" || e.key === "AltGraph";
