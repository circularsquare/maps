// The city view's own rows (T-1438), around what a bubble click shows (CityInspector.tsx): the train
// share, points, profit a day and goals reached, then for a city with stations its boardings by
// purpose and its busiest lines and stations, each a click away from its own view. Lines and
// stations count boardings at the city's stations in the past 24 hours, as the Stations tab does.

import { type City, type CityId } from "../../game/cities";
import { t as tr, tSpaced } from "../../i18n";
import { complexes } from "../../game/complexView";
import { membersOf } from "../../game/complexes";
import { cityDemandViews, demandView, ridesOf } from "../../game/demand";
import { count, exact, money, moneyNegative } from "../../game/format";
import { stationZone } from "../../game/listScope";
import { moneyState } from "../../game/money";
import { GOAL_SHARES, goalsReached, pointsPast24h, pointsShort, STAR_SHARE } from "../../game/points";
import { splitTable } from "../../game/splitTip";
import { lengthsOf } from "../../game/zoneLengths";
import { stationsToday, today, useToday } from "../../game/today";
import { selection, shownDay, shownDayOn, shownDayPhrase } from "../../game/ui";
import { builtCities } from "../../game/worldTotals";
import { AVERAGE_DAY } from "../../game/days";
import { stationsById, world } from "../../game/world";
import type { LineInput, StationInput } from "../../game/types";
import { figures, moneyTip, net, pointsTip } from "../Cities";
import { goalTip } from "../GoalTicks";
import { Chip, Icon } from "../widgets";
import { PurposeSplit, purposeSplit, type SplitScope } from "./PurposeSplit";
import { pct1, pctWhole } from "./tripText";

/** How many lines and stations the city view lists. */
const TOP = 5;

/** Each line stopping in the city, with the indices of its stops there. */
function cityScope(id: CityId, stations: Map<string, StationInput>, lines: readonly LineInput[]): { line: LineInput; at: number[] }[] {
  const out: { line: LineInput; at: number[] }[] = [];
  for (const l of lines) {
    const at = l.stops.flatMap((s, k) => { const st = stations.get(s); return st && stationZone(st) === id ? [k] : []; });
    if (at.length) out.push({ line: l, at });
  }
  return out;
}

export function CityFigures({ city, built }: { city: City; built: boolean }) {
  const v = cityDemandViews.value.get(city.id);
  const day = shownDay.value === AVERAGE_DAY ? "" : tSpaced(shownDayOn.value);
  const rides = v ? ridesOf(v) : null;
  const m = moneyState.value;
  const f = built && m ? figures(city.id, m) : null;
  const at = goalsReached(city.id);
  const reached = GOAL_SHARES.map((g, k) => ({ g, t: at[k] })).filter((x) => x.t !== undefined);
  // T-1892: the same counts as the Cities tab's hover and Trips column, and the lines that stop here
  const w = world.value;
  const stations = builtCities().get(city.id) ?? 0;
  const lines = w ? cityScope(city.id, stationsById(w), w.save.lines).length : 0;
  return (
    <>
    <ul class="rows">
      <li>
        <span class="grow">{tr("cityDetails.trainShare")}</span>
        <span class="val" data-tip={splitTable(tr("cityDetails.where", { city: city.name, day }), v?.split ?? null, rides, lengthsOf(city.id), "")}>{v ? pct1(v.split.train / 100) : "–"}</span>
      </li>
      {built && (
        <>
          <li>
            <span class="grow">{tr("cityDetails.stations")}</span>
            <span class="val">{exact(stations)}</span>
          </li>
          <li>
            <span class="grow">{tr("cityDetails.lines")}</span>
            <span class="val">{exact(lines)}</span>
          </li>
          <li>
            <span class="grow">{tr("cityDetails.trainTrips")}</span>
            <span class="val" data-tip={v ? tr("cityDetails.tripsTip", { count: exact(v.railTripsPerDay), phrase: shownDayPhrase.value, city: city.name }) : undefined}>{v ? tr("inspector.perDay", { value: count(v.railTripsPerDay) }) : "–"}</span>
          </li>
          <li>
            <span class="grow">{tr("cityDetails.points")}</span>
            <span class="val" data-tip={pointsTip(city.id, city.name)}>{pointsShort(pointsPast24h.value.get(city.id) ?? 0)}</span>
          </li>
          <li>
            <span class="grow">{tr("cityDetails.profit")}</span>
            <span class={"val" + (f && moneyNegative(net(f.d) * 1e6) ? " money-neg" : "")} data-tip={f ? moneyTip(city.name, f) : undefined}>{f ? tr("inspector.perDay", { value: money(net(f.d) * 1e6) }) : "–"}</span>
          </li>
          <li>
            <span class="grow">{tr("cityDetails.goalsReached")}</span>
            <span class="val goals-reached">
              {reached.length ? reached.map(({ g, t }) => (
                <span class="goal-mark" data-tip={goalTip(city, g, t)}>{g === STAR_SHARE && <Icon.star />}{pctWhole(g / 100)}</span>
              )) : tr("cityDetails.none")}
            </span>
          </li>
        </>
      )}
    </ul>
    </>
  );
}

export function CityNetwork({ city }: { city: City }) {
  useToday();
  const w = world.value;
  if (!w) return null;
  const byId = stationsById(w);
  const scope = cityScope(city.id, byId, w.save.lines);
  if (!scope.length) return null;
  const t = today.value, known = t.size > 0;
  const lineRows = scope
    .map(({ line, at }) => ({ line, n: at.reduce((s, k) => s + (t.get(line.id)?.stops[k] ?? 0), 0) }))
    .sort((a, b) => b.n - a.n)
    .slice(0, TOP);
  // one row per complex (as the Stations tab), named after its member with most lines
  const cx = complexes.value, by = stationsToday.value;
  const linesAt = new Map<string, number>();
  for (const l of w.save.lines) for (const s of new Set(l.stops)) linesAt.set(s, (linesAt.get(s) ?? 0) + 1);
  const seen = new Set<string>();
  const stationRows: { lead: StationInput; n: number }[] = [];
  for (const s of w.save.stations) {
    if (!s.built || seen.has(s.id) || stationZone(s) !== city.id) continue;
    const members = membersOf(cx, s.id);
    let lead = s, n = 0;
    for (const id of members) {
      seen.add(id);
      const m = byId.get(id);
      if (m && (linesAt.get(id) ?? 0) > (linesAt.get(lead.id) ?? 0)) lead = m;
      n += by.get(id) ?? 0;
    }
    stationRows.push({ lead, n });
  }
  stationRows.sort((a, b) => b.n - a.n);
  const purposeScope: SplitScope = scope.map(({ line, at }) => ({ line: line.id, at }));
  const purposes = purposeSplit(demandView.value, purposeScope);
  return (
    <>
      {purposes && purposes.parts.some((p) => p.n > 0) && (
        <>
          <div class="h">{tr("cityDetails.byPurpose")}</div>
          <ul class="rows">
            <PurposeSplit scope={purposeScope} unit={tr("cityDetails.boardingsIn", { city: city.name })} />
          </ul>
        </>
      )}
      <div class="h">{tr("cityDetails.topLines")}<span class="right muted small">{tr("cityDetails.boardingsToday")}</span></div>
      <ul class="rows pick">
        {lineRows.map(({ line, n }) => (
          <li onClick={() => (selection.value = { kind: "line", line: line.id })}>
            <Chip line={line} />
            <span class="grow">{line.name}</span>
            <span class="val" data-tip={known ? tr("cityDetails.lineBoardings", { count: count(n), line: line.name, city: city.name }) : undefined}>{known ? count(n) : ""}</span>
          </li>
        ))}
      </ul>
      <div class="h">{tr("cityDetails.topStations")}<span class="right muted small">{tr("cityDetails.boardingsToday")}</span></div>
      <ul class="rows pick">
        {stationRows.slice(0, TOP).map(({ lead, n }) => (
          <li onClick={() => (selection.value = { kind: "station", station: lead.id })}>
            <span class="grow">{lead.name}</span>
            <span class="val" data-tip={known ? tr("cityDetails.stationBoardings", { count: count(n), station: lead.name }) : undefined}>{known ? count(n) : ""}</span>
          </li>
        ))}
      </ul>
    </>
  );
}
