// The "+N daily rides" indicator (T-1478, moved here by T-1499): at the top centre of the map for a few
// seconds after a line's stops changed and demand answered, one small label per line with its chip, the
// number easing up from 0 a frame at a time. A fall is the same words, quietly. game/ridesNote.ts says when.

import { h, render } from "preact";
import { useEffect, useState } from "preact/hooks";
import { count } from "../game/format";
import { NOTE_COUNT_MS, ridesNotes, shownDelta, type RidesNote } from "../game/ridesNote";
import { world } from "../game/world";
import { Chip } from "./widgets";

function Note({ id, n }: { id: string; n: RidesNote }) {
  const [, frame] = useState(0);
  useEffect(() => {
    let raf = 0;
    const step = () => {
      frame((k) => k + 1);
      if (performance.now() - n.at < NOTE_COUNT_MS) raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [n.at]);
  const line = world.value?.save.lines.find((l) => l.id === id);
  if (!line) return null;
  const shown = shownDelta(n), up = n.delta > 0;
  const word = Math.abs(n.delta) === 1 ? "daily ride" : "daily rides";
  return <div class={"rides-note" + (up ? "" : " down")} data-tip={`Riders a day went from ${count(n.from)} to ${count(n.to)} after the stops changed`}>
    <Chip line={line} size="sm" />
    <span>{up ? "+" : "−"}{Math.abs(shown).toLocaleString("en-US")} {word}</span>
  </div>;
}

function RidesNotes() {
  const all = ridesNotes.value;
  if (!all.size) return null;
  return <div class="rides-notes">{[...all].map(([id, n]) => <Note key={id + n.at} id={id} n={n} />)}</div>;
}

/** Mount the indicator in the map's container. */
export function startRidesNotes(container: HTMLElement) {
  const root = document.createElement("div");
  container.appendChild(root);
  render(h(RidesNotes, null), root);
}
