// A passing note at the map's top centre, under the Select tool's bar (T-1709: the feedback panel's
// "sent"): one line of words in a small box for a few seconds. It took T-1478's daily rides note's place
// and look; that note is now ui/EditNote.tsx at the upper left (T-2125).

import { signal } from "@preact/signals";
import { h, render } from "preact";

/** Wall ms the note stays. */
const NOTE_MS = 5000;

const passing = signal<{ text: string; at: number } | null>(null);
let passingTimer = 0;

export function showPassingNote(text: string) {
  const at = performance.now();
  passing.value = { text, at };
  clearTimeout(passingTimer);
  passingTimer = setTimeout(() => { passing.value = null; }, NOTE_MS) as unknown as number;
}

function PassingNote() {
  const say = passing.value;
  if (!say) return null;
  return <div class="rides-notes"><div class="rides-note" key={say.at}>{say.text}</div></div>;
}

/** Mount the note in the map's container. */
export function startPassingNote(container: HTMLElement) {
  const root = document.createElement("div");
  container.appendChild(root);
  render(h(PassingNote, null), root);
}
