// The parts the trip explorer's two panels share (TripInspector.tsx, MarketTrip.tsx): a route's
// name row, its leg rows, and the train share by period.

import type { JSX } from "preact";
import { MODE_COLOURS } from "../../game/palette";
import { t, tNumber, tRich } from "../../i18n";
import { PERIODS } from "../../game/periods";
import { tripPeriod, type TripLeg, type TripRoute } from "../../game/trip";
import type { LineInput } from "../../game/types";
import { nameBeforeNoun, vehicleOf } from "../../game/vehicle";
import { world } from "../../game/world";
import { live } from "../../map/live";
import { shownPeriod } from "../../map/tripView";
import { Chip } from "../widgets";
import { minText, pctShare } from "./tripText";

/** A route by its rides: the lines' names, the first ride's start and the last leg's end, and the
 * row contents that show them (the line chips, then the names and the end). */
export function routeRow(r: TripRoute, st: ReadonlyMap<string, string>, ln: ReadonlyMap<string, LineInput>) {
  const rides = r.legs.filter((l) => l.kind === "ride");
  const from = rides[0] ? st.get(rides[0].from) : "", to = r.legs.length ? st.get(r.legs[r.legs.length - 1].to) : "";
  const names = rides.map((l) => ln.get(l.line)?.name ?? t("routeParts.aLine")).join(t("routeParts.then"));
  const label = (
    <>
      <span class="bullets">{rides.map((l) => { const line = ln.get(l.line); return line && <Chip line={line} size="sm" />; })}</span>
      <span class="grow">{t("routeParts.namesTo", { names, to: to ?? "" })}</span>
    </>
  );
  return { names, from, to, label };
}

/** T-1902: the map moved to a station, the trip kept selected (its focused zone too). */
export function showStation(id: string) {
  const map = live.map, w = world.peek();
  const s = w?.save.stations.find((x) => x.id === id);
  if (!map || !s) return;
  map.jumpTo({ center: [s.lng, s.lat], zoom: Math.max(map.getZoom(), STATION_ZOOM) });
}
/** How far in a click on a station's name goes at least: the station and the streets around it. */
const STATION_ZOOM = 12;

/** A station's name in a route's rows, a click on it moving the map there. */
export function StationLink({ id, name }: { id: string; name: string }) {
  return <span class="place-link" onClick={(e) => { e.stopPropagation(); showStation(id); }}>{name}</span>;
}

/** The rows for a route's legs: a walk between stations, and each ride with its line chip. `rideTip`
 * words a ride's hover. */
export function legRows(r: TripRoute, name: (station: string) => string, ln: ReadonlyMap<string, LineInput>, rideTip: (l: TripLeg) => string): JSX.Element[] {
  return r.legs.map((l, i) => {
    if (l.kind === "walk") {
      return (
        <li data-tip={t("routeParts.onFoot", { min: tNumber(l.min, { minimumFractionDigits: 1, maximumFractionDigits: 1 }) })}>
          <span class="grow">{tRich("routeParts.walkTo", { station: <StationLink id={l.to} name={name(l.to)} /> })}</span>
          <span class="val">{minText(l.min)}</span>
        </li>
      );
    }
    const line = ln.get(l.line);
    const st = world.peek()?.lineStats[l.line];
    const v = st ? vehicleOf(st) : "train";
    const bus = v === "bus" ? "true" : v === "tram" ? "tram" : "false";
    return (
      <li data-tip={rideTip(l)}>
        {line && <Chip line={line} size="sm" />}
        <span class="grow">{tRich("routeParts.rideTo", { change: i > 0 && r.legs[i - 1].kind === "ride", line: line ? (bus === "false" ? line.name : nameBeforeNoun(line.name, v)) : t("routeParts.aLineCap"), bus, station: <StationLink id={l.to} name={name(l.to)} /> })}</span>
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
      <div class="h">{t("routeParts.byPeriod")}<span class="right muted small">{t("routeParts.clickToShow")}</span></div>
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
