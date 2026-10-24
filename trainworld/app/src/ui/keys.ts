// Which keys the focused control keeps for itself (T-183). One rule for the game's keys (main.ts:
// 1-3, Esc, Space, Ctrl+Z/Y, the line tool's Enter and Backspace) and the search box's "/",
// Ctrl+K and Ctrl+F (ui/Search.tsx, T-211), so the two never disagree:
// - a text field (a text-like <input>, a <textarea>, anything contenteditable) keeps every key;
// - another form control (a <select>, a range, checkbox, radio or colour input) keeps only the
//   keys it works with: the arrows, Home, End, Page Up and Down, Enter and Space, and in a select a
//   plain letter (it jumps to the option starting with it). Everything else reaches the game;
// - a button keeps nothing: Space is the pause key even there (main.ts stops its click on keyup).
// A select picked with the mouse also gives its focus up (releaseSelectsAfterMousePick), so the
// next Space pauses rather than opening the list again.

import { keepPlacing } from "../game/ui";

/** Input types that are buttons: they keep no keys, like a <button>. */
const BUTTONS = new Set(["button", "submit", "reset", "image"]);
/** Input types that take no typing: they keep only their own keys. */
const CONTROLS = new Set(["checkbox", "radio", "range", "color", "file"]);
/** The keys a non-text control works with. */
const CONTROL_KEYS = new Set(["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight", "Home", "End", "PageUp", "PageDown", "Enter", " "]);

/** Is `t` a field the player types in? */
function isTextField(t: EventTarget | null): boolean {
  if (t instanceof HTMLTextAreaElement) return true;
  if (t instanceof HTMLInputElement) return !BUTTONS.has(t.type) && !CONTROLS.has(t.type) && t.type !== "hidden";
  return t instanceof HTMLElement && t.isContentEditable;
}

/** Does the element with focus use this key itself, so the game should leave it alone? */
export function keyIsTheirs(e: KeyboardEvent): boolean {
  const t = e.target;
  if (isTextField(t)) return true;
  const select = t instanceof HTMLSelectElement;
  if (!select && !(t instanceof HTMLInputElement && CONTROLS.has(t.type))) return false;
  if (CONTROL_KEYS.has(e.key)) return true;
  return select && !e.ctrlKey && !e.metaKey && !e.altKey && /^\p{L}$/u.test(e.key);
}

/** While `blocking()`, keys go nowhere but into `root`: the map's keys and the hotkeys wait, and Tab
 * and Enter work inside it. Captured at the window before any other listener. Returns the way to
 * stop. */
export function blockKeysOutside(root: Element, blocking: () => boolean): () => void {
  const block = (e: KeyboardEvent) => {
    if (!blocking() || (e.target instanceof Node && root.contains(e.target))) return;
    e.stopImmediatePropagation();
    e.preventDefault();
  };
  window.addEventListener("keydown", block, true);
  window.addEventListener("keyup", block, true);
  return () => {
    window.removeEventListener("keydown", block, true);
    window.removeEventListener("keyup", block, true);
  };
}

/** Caps Lock flips the Build tab's keep-placing switch (T-445), whatever the lock's own state: keep
 * placing is on by default (T-942), so mirroring the lock would make an unlit Caps Lock mean on. */
export function followCapsLock() {
  window.addEventListener("keydown", (e) => {
    if (e.key === "CapsLock" && !e.repeat) keepPlacing.value = !keepPlacing.peek();
  }, true);
}

/** After a pick made with the mouse, a <select> gives up its focus, so the game's keys work at
 * once (Space pauses, not reopens the list). Picked from the keyboard it keeps focus, and its
 * arrow keys, so the player can carry on through the options. */
export function releaseSelectsAfterMousePick() {
  let pointer = false;
  window.addEventListener("pointerdown", () => { pointer = true; }, true);
  window.addEventListener("keydown", () => { pointer = false; }, true);
  window.addEventListener("change", (e) => {
    if (pointer && e.target instanceof HTMLSelectElement) e.target.blur();
  });
}
