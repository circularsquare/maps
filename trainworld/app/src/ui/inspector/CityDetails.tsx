// The city view's own rows (T-1438), around what a bubble click shows (CityInspector.tsx): the train
// share, points, profit a day and goals reached, then for a city with stations its boardings by
// purpose and its busiest lines and stations, each a click away from its own view. Lines and
// stations count boardings at the city's stations in the past 24 hours, as the Stations tab does.

import type { City, CityId } from "../../game/cities";
import { complexes } from "../../game/complexView";
import { membersOf } from "../../game/complexes";
import { cityDemandViews, demandView } from "../../game/demand";
import { count, money } from "../../game/format";
import { stationZone } from "../../game/listScope";
import { moneyState } from "../../game/money";
import { GOAL_SHARES, goalsReached, pointsPast24h, pointsShort, STAR_SHARE } from "../../game/points";
import { splitTip } from "../../game/splitTip";
import { stationsToday, today, useToday } from "../../game/today";
import { selection, shownDay, shownDayOn } from "../../game/ui";
import { AVERAGE_DAY } from "../../game/days";
import { stationsById, world } from "../../game/world";
import type { LineInput, StationInput } from "../../game/types";
import { figures, moneyTip, net, pointsTip } from "../Cities";
import { goalTip } from "../GoalTicks";
import { Chip, Icon } from "../widgets";
import { PurposeSplit, purposeSplit, type SplitScope } from "./PurposeSplit";

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
  const day = shownDay.value === AVERAGE_DAY ? "" : " " + shownDayOn.value;
  const all = v?.tripsPerDay ?? 0;
  const rides = v ? { all, train: v.railTripsPerDay, walk: all * v.split.walk / 100, drive: all * v.split.drive / 100 } : null;
  const m = moneyState.value;
  const f = built && m ? figures(city.id, m) : null;
  const at = goalsReached(city.id);
  const reached = GOAL_SHARES.map((g, k) => ({ g, t: at[k] })).filter((x) => x.t !== undefined);
  return (
    <ul class="rows">
      <li>
        <span class="grow">Train share</span>
        <span class="val" data-tip={splitTip(` in ${city.name}${day}`, v?.split ?? null, rides, "")}>{v ? v.split.train.toFixed(1) + "%" : "–"}</span>
      </li>
      {built && (
        <>
          <li>
            <span class="grow">Points</span>
            <span class="val" data-tip={pointsTip(city.id, city.name)}>{pointsShort(pointsPast24h.value.get(city.id) ?? 0)}</span>
          </li>
          <li>
            <span class="grow">Profit</span>
            <span class={"val" + (f && net(f.d) * 1e6 <= -500 ? " money-neg" : "")} data-tip={f ? moneyTip(city.name, f) : undefined}>{f ? `${money(net(f.d) * 1e6)} a day` : "–"}</span>
          </li>
          <li>
            <span class="grow">Goals reached</span>
            <span class="val goals-reached">
              {reached.length ? reached.map(({ g, t }) => (
                <span class="goal-mark" data-tip={goalTip(city, g, t)}>{g === STAR_SHARE && <Icon.star />}{g}%</span>
              )) : "None"}
            </span>
          </li>
        </>
      )}
    </ul>
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
          <div class="h">By purpose</div>
          <ul class="rows">
            <PurposeSplit scope={purposeScope} unit={`boardings in ${city.name}`} />
          </ul>
        </>
      )}
      <div class="h">Top lines<span class="right muted small">boardings in past 24 h</span></div>
      <ul class="rows pick">
        {lineRows.map(({ line, n }) => (
          <li onClick={() => (selection.value = { kind: "line", line: line.id })}>
            <Chip line={line} />
            <span class="grow">{line.name}</span>
            <span class="val" data-tip={known ? `${count(n)} boardings on ${line.name} in ${city.name} in the past 24 hours` : undefined}>{known ? count(n) : ""}</span>
          </li>
        ))}
      </ul>
      <div class="h">Top stations<span class="right muted small">boardings in past 24 h</span></div>
      <ul class="rows pick">
        {stationRows.slice(0, TOP).map(({ lead, n }) => (
          <li onClick={() => (selection.value = { kind: "station", station: lead.id })}>
            <span class="grow">{lead.name}</span>
            <span class="val" data-tip={known ? `${count(n)} boardings at ${lead.name} in the past 24 hours` : undefined}>{known ? count(n) : ""}</span>
          </li>
        ))}
      </ul>
    </>
  );
}
