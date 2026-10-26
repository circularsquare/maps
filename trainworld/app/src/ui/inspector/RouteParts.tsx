// The parts the trip explorer's two panels share (TripInspector.tsx, MarketTrip.tsx): a route's
// name row, its leg rows, and the train share by period.

import type { JSX } from "preact";
import { MODE_COLOURS } from "../../game/palette";
import { PERIODS } from "../../game/periods";
import { tripPeriod, type TripLeg, type TripRoute } from "../../game/trip";
import type { LineInput } from "../../game/types";
import { world } from "../../game/world";
import { shownPeriod } from "../../map/tripView";
import { Chip } from "../widgets";
import { minText, pctShare } from "./tripText";

/** A route by its rides: the lines' names, the first ride's start and the last leg's end, and the
 * row contents that show them (the line chips, then the names and the end). */
export function routeRow(r: TripRoute, st: ReadonlyMap<string, string>, ln: ReadonlyMap<string, LineInput>) {
  const rides = r.legs.filter((l) => l.kind === "ride");
  const from = rides[0] ? st.get(rides[0].from) : "", to = r.legs.length ? st.get(r.legs[r.legs.length - 1].to) : "";
  const names = rides.map((l) => ln.get(l.line)?.name ?? "a line").join(" then ");
  const label = (
    <>
      {rides.map((l) => { const line = ln.get(l.line); return line && <Chip line={line} size="sm" />; })}
      <span class="grow">{names} to {to}</span>
    </>
  );
  return { names, from, to, label };
}

/** The rows for a route's legs: a walk between stations, and each ride with its line chip. `rideTip`
 * words a ride's hover. */
export function legRows(r: TripRoute, name: (station: string) => string, ln: ReadonlyMap<string, LineInput>, rideTip: (l: TripLeg) => string): JSX.Element[] {
  return r.legs.map((l, i) => {
    if (l.kind === "walk") {
      return (
        <li data-tip={`${l.min.toFixed(1)} min on foot`}>
          <span class="grow">Change: walk to {name(l.to)}</span>
          <span class="val">{minText(l.min)}</span>
        </li>
      );
    }
    const line = ln.get(l.line);
    const bus = !!world.peek()?.lineStats[l.line]?.bus;
    return (
      <li data-tip={rideTip(l)}>
        {line && <Chip line={line} size="sm" />}
        <span class="grow">{i > 0 && r.legs[i - 1].kind === "ride" ? "Change to " : ""}{line?.name ?? "A line"}{bus ? " bus" : ""} to {name(l.to)}</span>
        <span class="val">{minText(l.waitMin + l.min)}</span>
      </li>
    );
  });
}

/** The train's share in each period, a row to click that shows the period. `train(i)` is null where
 * the period has no answer. */
export function PeriodRows({ train, tip }: { train: (period: number) => number | null; tip: (period: number) => string }) {
  const q = shownPeriod.value;
  return (
    <>
      <div class="h">Train by period<span class="right muted small">click to show</span></div>
      <ul class="rows pick">
        {PERIODS.map((p, i) => {
          const v = train(i);
          return (
            <li class={i === q ? "on" : ""} onClick={() => (tripPeriod.value = i)} data-tip={tip(i)}>
              <span class="grow">{p.name}</span>
              <span class="trip-hbar"><i class="hbar" style={{ width: `${Math.round(60 * (v ?? 0))}px`, background: MODE_COLOURS.train }} /></span>
              <span class="val w-num">{v === null ? "" : pctShare(v)}</span>
            </li>
          );
        })}
      </ul>
    </>
  );
}
