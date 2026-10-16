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
export function isTextField(t: EventTarget | null): boolean {
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

/** Caps Lock is the same switch as the Build tab's keep-placing button (T-445): each change of its
 * state, seen on any key or press, sets `keepPlacing`; the button sets it too and the two follow
 * whichever moved last. The state is unknown until the first event, which only records it (unless
 * it is the Caps Lock key), so the stored setting stands over a Caps Lock left on (T-487). */
export function followCapsLock() {
  let last: boolean | null = null;
  const sync = (e: Event) => {
    if (!(e instanceof KeyboardEvent) && !(e instanceof MouseEvent)) return;
    const on = e.getModifierState("CapsLock");
    const first = last === null;
    if (on === last) return;
    last = on;
    if (first && !(e instanceof KeyboardEvent && e.key === "CapsLock")) return;
    keepPlacing.value = on;
  };
  for (const type of ["keydown", "keyup", "pointerdown"]) window.addEventListener(type, sync, true);
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
