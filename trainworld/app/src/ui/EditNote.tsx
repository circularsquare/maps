// "What this edit did" (T-2125): under the view rows at the map's upper left, for a few seconds after
// demand settles from the player's edits, the change in train trips a day and profit a day, each after a
// green up or red down arrow, as plain text on the map; the numbers ease up from 0; then the bullets of
// the lines whose riders moved most (T-2390). game/editNote.ts says when and what the figures are.

import { useEffect, useState } from "preact/hooks";
import { count, exact, money, moneyNegative, moneyShows } from "../game/format";
import { editNote, holdNote, NOTE_COUNT_MS, shownChange, type FromTo } from "../game/editNote";
import type { ProfitEstimate } from "../game/profitEstimate";
import { world } from "../game/world";
import { t, type MessageKey } from "../i18n";
import { Chip } from "./widgets";

const Arrow = ({ up }: { up: boolean }) => (
  <svg class={"edit-arrow " + (up ? "up" : "down")} viewBox="0 0 12 12" aria-hidden="true">
    <path d={up ? "M6 10.5V2M2.5 5.5L6 2l3.5 3.5" : "M6 1.5V10M2.5 6.5L6 10l3.5-3.5"} />
  </svg>
);

const signed = (n: number, text: string) => (n > 0 ? "+" : n < 0 ? "−" : "") + text.replace(/^−/, "");

/** T-2626: the estimate's hover, its two parts; a part under $500 is left out. */
const estimateTip = (e: ProfitEstimate, day: string) =>
  ([[e.running, "editNote.runningTip"], [e.fares, "editNote.faresTip"]] as const).filter(([v]) => moneyShows(v))
    .map(([v, key]) => t(key, { change: signed(v, money(Math.abs(v))), day })).join("\n");

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
  const lines = world.value?.save.lines ?? [];
  const name = n.line && lines.find((l) => l.id === n.line!.id)?.name;
  const moved = n.moved.flatMap((x) => { const l = lines.find((y) => y.id === x.id); return l ? [{ l, riders: x.riders }] : []; });
  const dt = n.trips.to - n.trips.from, dm = n.money.to - n.money.from;
  const tr = Math.round(shownChange(dt, n.at)), m = n.estimating ? dm : shownChange(dm, n.at, performance.now(), n.start);
  const tripsText = signed(dt, Math.abs(dt) >= 10_000 ? count(Math.abs(tr)) : exact(Math.abs(tr)));
  // the line's part: its riders (its boardings) under the train trips, its profit under the profit
  const tip = (key: MessageKey, ft: FromTo, lineKey: MessageKey, line: FromTo | undefined, f: (v: number) => string) =>
    t(key, { day: n.day, from: f(ft.from), to: f(ft.to) }) + (name && line ? "\n" + t(lineKey, { line: name, from: f(line.from), to: f(line.to) }) : "");
  return (
    <div class="edit-note" onPointerEnter={() => holdNote(true)} onPointerLeave={() => holdNote(false)}>
      {dt !== 0 && (
        <div class="edit-row" data-tip={tip("editNote.tripsTip", n.trips, "editNote.lineRidersTip", n.line?.riders, exact)}>
          <Arrow up={dt > 0} />
          <span>{t("editNote.trips", { change: tripsText })}</span>
        </div>
      )}
      {moneyShows(dm) && (
        <div class={"edit-row" + (n.estimating ? " estimating" : moneyNegative(dm) ? " money-neg" : "")} data-tip={n.est ? estimateTip(n.est, n.day) : tip("editNote.profitTip", n.money, "editNote.lineProfitTip", n.line?.money, money)}>
          <Arrow up={dm > 0} />
          <span>{t("editNote.profit", { estimating: !!n.estimating, change: signed(dm, money(m)) })}</span>
        </div>
      )}
      {moved.length > 0 && (
        <div class="edit-row edit-lines">
          {moved.map(({ l, riders }) => <Chip line={l} size="sm" tip={l.name + "\n" + t("editNote.lineChipTip", { from: exact(riders.from), to: exact(riders.to), day: n.day })} />)}
        </div>
      )}
    </div>
  );
}
