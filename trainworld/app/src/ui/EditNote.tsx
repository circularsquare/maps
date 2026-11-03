// "What this edit did" (T-2125): under the view rows at the map's upper left, for a few seconds after
// demand settles from the player's edits, the change in riders a day and profit a day, each after a
// green up or red down arrow, as plain text on the map; the numbers ease up from 0. game/editNote.ts
// says when and what the figures are.

import { useEffect, useState } from "preact/hooks";
import { count, exact, money, moneyNegative } from "../game/format";
import { editNote, holdNote, NOTE_COUNT_MS, shownChange, type FromTo } from "../game/editNote";
import { world } from "../game/world";

const Arrow = ({ up }: { up: boolean }) => (
  <svg class={"edit-arrow " + (up ? "up" : "down")} viewBox="0 0 12 12" aria-hidden="true">
    <path d={up ? "M6 10.5V2M2.5 5.5L6 2l3.5 3.5" : "M6 1.5V10M2.5 6.5L6 10l3.5-3.5"} />
  </svg>
);

const signed = (n: number, text: string) => (n > 0 ? "+" : n < 0 ? "−" : "") + text.replace(/^−/, "");

export function EditNote() {
  const n = editNote.value;
  const [, frame] = useState(0);
  useEffect(() => {
    if (!n) return;
    let raf = 0;
    const step = () => {
      frame((k) => k + 1);
      if (performance.now() - n.at < NOTE_COUNT_MS) raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [n?.at]);
  if (!n) return null;
  const name = n.line && world.value?.save.lines.find((l) => l.id === n.line!.id)?.name;
  const dr = n.riders.to - n.riders.from, dm = n.money.to - n.money.from;
  const r = Math.round(shownChange(dr, n.at)), m = shownChange(dm, n.at);
  const ridersText = signed(dr, Math.abs(dr) >= 10_000 ? count(Math.abs(r)) : exact(Math.abs(r)));
  const tip = (what: string, t: FromTo, line: FromTo | undefined, f: (v: number) => string) =>
    `${what} ${n.day}: ${f(t.from)} to ${f(t.to)}` + (name && line ? `\n${name}: ${f(line.from)} to ${f(line.to)}` : "");
  return (
    <div class="edit-note" onPointerEnter={() => holdNote(true)} onPointerLeave={() => holdNote(false)}>
      {dr !== 0 && (
        <div class="edit-row" data-tip={tip("Riders", n.riders, n.line?.riders, exact)}>
          <Arrow up={dr > 0} />
          <span>{ridersText} riders/day</span>
        </div>
      )}
      {Math.abs(dm) >= 500 && (
        <div class={"edit-row" + (moneyNegative(dm) ? " money-neg" : "")} data-tip={tip("Profit", n.money, n.line?.money, money)}>
          <Arrow up={dm > 0} />
          <span>{signed(dm, money(m))} profit/day</span>
        </div>
      )}
    </div>
  );
}
