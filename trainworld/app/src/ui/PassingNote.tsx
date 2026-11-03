// A passing note at the map's upper left, under the edit note and in its look (T-2166; T-1709: the
// feedback panel's "sent"): one line of plain text on the map for a few seconds.

import { signal } from "@preact/signals";

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

/** In the map's upper left (DemandSelector.tsx's `MapTopLeft`). */
export function PassingNote() {
  const say = passing.value;
  if (!say) return null;
  return <div class="edit-note" key={say.at}><div class="edit-row">{say.text}</div></div>;
}
