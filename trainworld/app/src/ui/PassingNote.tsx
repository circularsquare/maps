// A passing note at the map's upper left, under the edit note and in its look (T-2166; T-1709: the
// feedback panel's "sent"): one line of plain text on the map for a few seconds.

import { signal } from "@preact/signals";
import { choosing, startup } from "../game/loading";

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

/** T-3630: while the start menu or the failure screen covers the map, the note is at the page's top centre instead */
const overPage = () => choosing.value || startup.value === "failed";

/** In the map's upper left (DemandSelector.tsx's `MapTopLeft`), or with `page` over the page (FeedbackPanel.tsx). */
export function PassingNote({ page }: { page?: boolean }) {
  const say = passing.value;
  if (!say || overPage() !== !!page) return null;
  const note = <div class="edit-note" key={say.at}><div class="edit-row">{say.text}</div></div>;
  return page ? <div class="page-note">{note}</div> : note;
}
