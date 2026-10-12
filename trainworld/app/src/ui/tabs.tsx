// Tab contents: build, lines, stations, money, city.

import { clock } from "../game/clock";
import { activeCity, CITIES, cityById } from "../game/cities";
import { useState } from "preact/hooks";
import { createLine } from "../game/lineActions";
import { lineDemand, stationDemand, demandView, intercityDemand, INTERCITY_PLACES } from "../game/demand";
import type { JSX } from "preact";
import { count, levelLabel, money } from "../game/format";
import { DEMAND_INDEX, LEVELS, type Level } from "../game/types";
import { buildLevel, buildPlatform, lineSort, selection, singleTrack, tool, type Tool } from "../game/ui";
import { linesAt, setFares, world } from "../game/world";
import { income, moneyState } from "../game/money";
import type { DayLedger } from "../workers/protocol";
import { live } from "../map/live";
import { Chip, Icon, Row, Segmented, Stepper } from "./widgets";
import { BlueprintBreakdown, BlueprintControls } from "./Blueprint";

const STATUS: Record<string, string> = { running: "", planned: "Planned", broken: "Broken" };

// SPEC 6.4. A display copy: the track model (sim/src/track/params.rs) owns the real price list.
const BASE_M_PER_KM = 90;
const MULT: Record<Level, number> = { 3: 1.8, 2: 1.3, 1: 0.8, 0: 0.3, [-1]: 1.0, [-2]: 1.1, [-3]: 1.2 };
const LEVEL_NAME: Record<Level, string> = {
  3: "High viaduct",
  2: "Viaduct",
  1: "Viaduct",
  0: "Ground level",
  [-1]: "Cut and cover",
  [-2]: "Tunnel",
  [-3]: "Deep tunnel",
};

/** The build tools and their keys (T-096: Esc goes back to Select from anywhere, main.ts). */
const TOOLS: [Tool, string, () => JSX.Element, string, string][] = [
  ["track", "Track", Icon.track, "1", "Draw track"],
  ["station", "Station", Icon.station, "2", "Place stations"],
  ["delete", "Delete", Icon.bin, "3", "Delete track, stations and flyovers"],
];

/** The tool each number key picks (T-096). */
export const TOOL_KEYS: Record<string, Tool> = { "1": "track", "2": "station", "3": "delete" };

/** Pick a build tool; the one in use again, or Select, goes back to selecting. */
export function pickTool(v: Tool) {
  const t = tool.peek();
  live.tools?.exit();
  if (t !== v && v !== "select") tool.value = v;
}

/** All building controls: tools, blueprint quote and construction, level, track and platforms. */
export function BuildTab() {
  const t = tool.value;
  const lv = buildLevel.value;
  return (
    <>
      <div class="build-options">
        <div class="group tool-pick">
          {TOOLS.map(([id, label, I, key, title]) => (
            <button class={"btn text" + (t === id ? " on" : "")} title={`${title} (${key})`} onClick={() => pickTool(id)}>
              <I />
              {label}
              {key !== "Esc" && <span class="key muted">{key}</span>}
            </button>
          ))}
        </div>
        <div class="h">Level</div>
        <div class="group levels">
          {LEVELS.map((l) => (
            <button class={"btn lv" + (l === lv ? " on" : "")} onClick={() => (buildLevel.value = l)}>
              {levelLabel(l)}
            </button>
          ))}
        </div>
        <p class="muted small price">
          {LEVEL_NAME[lv]} at ${Math.round(BASE_M_PER_KM * MULT[lv])}M per km ({MULT[lv].toFixed(1)}x)
        </p>
        <div class="h">Track</div>
        <Segmented<boolean> options={[[false, "Double"], [true, "Single"]]} value={singleTrack.value} onChange={(v) => (singleTrack.value = v)} />
        <div class="h">New stations</div>
        <ul class="rows">
          <li>
            <span class="grow">Platform length</span>
            <Stepper value={buildPlatform.value + " m"} wide label="platform length" onStep={(s) => (buildPlatform.value = Math.max(60, Math.min(400, buildPlatform.value + s * 20)))} />
          </li>
        </ul>
        <BlueprintBreakdown />
      </div>
      <BlueprintControls />
    </>
  );
}

export function LinesTab() {
  const [creating, setCreating] = useState(false);
  const w = world.value;
  if (!w) return null;
  const sel = selection.value;
  const lev = DEMAND_INDEX[clock.demand.value];
  const total = w.save.lines.reduce((s, l) => s + (lineDemand(l.id)?.ridersPerDay ?? 0), 0);
  const newLine = async () => {
    if (creating) return;
    setCreating(true);
    try { await createLine(); } finally { setCreating(false); }
  };
  const natural = (a: string, b: string) => a.localeCompare(b, undefined, { numeric: true, sensitivity: "base" });
  const lines = [...w.save.lines].sort((a, b) =>
    (lineSort.value === "passengers" ? (lineDemand(b.id)?.ridersPerDay ?? 0) - (lineDemand(a.id)?.ridersPerDay ?? 0) : 0)
    || natural(a.name, b.name) || a.num - b.num);
  return (
    <>
      <div class="line-list-actions">
        <button class="btn text" onClick={newLine} disabled={creating}><Icon.plus />New line</button>
        <label>Sort <select aria-label="Sort lines" value={lineSort.value} onChange={e => lineSort.value = e.currentTarget.value as "name" | "passengers"}>
          <option value="name">Name</option><option value="passengers">Forecast riders</option>
        </select></label>
      </div>
      {w.save.lines.length > 0 && <p class="muted small">Riders are steady-state forecasts for the current network and schedules.</p>}
      {w.save.lines.length === 0 && <p class="muted">No lines yet. Create a line, then choose its stations.</p>}
      <ul class="rows pick">
        {lines.map((l) => {
          const st = w.lineStats[l.id];
          const on = (sel?.kind === "line" || sel?.kind === "train") && sel.line === l.id;
          const dm = lineDemand(l.id);
          return (
            <li class={on ? "on" : ""} onClick={() => (selection.value = { kind: "line", line: l.id })}>
              <Chip line={l} />
              <span class="grow">{l.name}</span>
              {st.status !== "running" ? (
                <span class="val muted">{STATUS[st.status]}</span>
              ) : (
                <>
                  <span class="val" title={dm ? `${count(dm.ridersPerDay)} forecast riders a day` : undefined}>{dm ? count(dm.ridersPerDay) : ""}</span>
                  <span class="val muted w-trains">{st.trainsNeeded[lev]} trains</span>
                </>
              )}
            </li>
          );
        })}
      </ul>
      <div class="pane-foot">
        <span class="muted">{demandView.value ? count(total) + " forecast riders a day" : ""}</span>
      </div>
    </>
  );
}

export function StationsTab() {
  const w = world.value;
  if (!w) return null;
  const sel = selection.value;
  const st = [...w.save.stations].sort((a, b) => (stationDemand(b.id)?.boardings ?? 0) - (stationDemand(a.id)?.boardings ?? 0) || a.name.localeCompare(b.name));
  return (
    <>
      {st.length === 0 && <p class="muted">No stations yet. Pick the station tool and click on track.</p>}
      <ul class="rows pick">
        {st.map((s) => (
          <li class={sel?.kind === "station" && sel.station === s.id ? "on" : ""} onClick={() => (selection.value = { kind: "station", station: s.id })}>
            <span class={"grow" + (s.built ? "" : " muted")}>{s.name}</span>
            {linesAt(s.id).map((l) => (
              <Chip line={l} size="sm" />
            ))}
            <span class="val w-num">{s.built ? count(stationDemand(s.id)?.boardings ?? 0) : "Plan"}</span>
          </li>
        ))}
      </ul>
      {st.length > 0 && (
        <div class="pane-foot">
          <span class="muted">{st.length} stations, boardings a day</span>
        </div>
      )}
    </>
  );
}

export function MoneyTab() {
  const w = world.value;
  const m = moneyState.value;
  if (!w || !m) return null;
  const inc = income.value;
  const today = m.ledger[m.ledger.length - 1];
  const prev = m.ledger.length > 1 ? m.ledger[m.ledger.length - 2] : null;
  const net = (d: DayLedger | null) => (d ? d.fares - d.running - d.trains - d.build : 0);
  const cell = (d: DayLedger | null, v: (d: DayLedger) => number) => <td class="num">{d ? money(v(d) * 1e6) : ""}</td>;
  const f = m.fares;
  return (
    <>
      <ul class="rows">
        <Row label="Cash" bold>
          {money(m.cash * 1e6)}
        </Row>
      </ul>
      <table class="ledger">
        <tr>
          <th />
          <th>Yesterday</th>
          <th>Today</th>
        </tr>
        <tr>
          <td>Fares</td>
          {cell(prev, (d) => d.fares)}
          {cell(today, (d) => d.fares)}
        </tr>
        <tr>
          <td>Running trains</td>
          {cell(prev, (d) => -d.running)}
          {cell(today, (d) => -d.running)}
        </tr>
        <tr>
          <td>New trains</td>
          {cell(prev, (d) => -d.trains)}
          {cell(today, (d) => -d.trains)}
        </tr>
        <tr>
          <td>Construction</td>
          {cell(prev, (d) => -d.build)}
          {cell(today, (d) => -d.build)}
        </tr>
        <tr class="b">
          <td>Net</td>
          {cell(prev, net)}
          {cell(today, net)}
        </tr>
      </table>
      <div class="h">Fare</div>
      <ul class="rows">
        <li>
          <span class="grow">Each ride</span>
          <Stepper value={"$" + f.base.toFixed(2)} wide label="fare a ride" onStep={(s) => setFares(f.base + s * 0.25, f.perKm)} />
        </li>
        <li>
          <span class="grow">Each km</span>
          <Stepper value={"$" + f.perKm.toFixed(2)} wide label="fare a km" onStep={(s) => setFares(f.base, f.perKm + s * 0.01)} />
        </li>
        <Row label="A 10 km ride">{"$" + (f.base + 10 * f.perKm).toFixed(2)}</Row>
      </ul>
      <div class="h">A day at this schedule</div>
      <ul class="rows">
        <Row label="Fares">{demandView.value ? money(inc.faresPerDay) : ""}</Row>
        <Row label="Running trains">{money(-inc.runningPerDay)}</Row>
      </ul>
      <div class="h">Trains</div>
      <ul class="rows">
        <Row label="Cars owned">{m.fleet}</Row>
        <Row label="Cars the lines need">{m.carsNeeded}</Row>
      </ul>
      <ul class="rows">
        <Row label="Track and stations built">{money(w.money.builtValue)}</Row>
      </ul>
      <p class="note muted">
        A car costs {money(m.carPrice * 1e6)} and {"$" + m.runCostCarKm.toFixed(2)} a km to run. Trains are bought when a line needs more than the spare cars. Spare cars are kept for other lines.
      </p>
      <p class="note muted">
        Fares and running costs count {m.economy} times over in a game day. Track, stations and trains cost what they would in real life.
      </p>
    </>
  );
}

export function CityTab() {
  const city = cityById(activeCity.value);
  const market = intercityDemand.value;
  // The market publishes undirected corridors: each row includes both travel directions.
  const corridors = market?.corridors
    .filter(c => c.from !== c.to && (c.from === city.market || c.to === city.market) && c.trips > 0)
    .sort((a, b) => b.trips - a.trips);
  const total = corridors?.reduce((sum, c) => sum + c.trips, 0) ?? 0;
  const rail = corridors?.reduce((sum, c) => sum + c.rail, 0) ?? 0;
  return <>
    <div class="h">{city.name}</div>
    <div class="city-navigation">
      <button class="btn text" onClick={() => live.map?.flyTo({center: city.center, zoom: city.zoom})}>City centre</button>
      <button class="btn text" onClick={() => {
        const lon = CITIES.map(c => c.center[0]), lat = CITIES.map(c => c.center[1]);
        live.map?.fitBounds([[Math.min(...lon) - 0.7, Math.min(...lat) - 0.5], [Math.max(...lon) + 0.7, Math.max(...lat) + 0.5]], {padding:40});
      }}>Whole region</button>
    </div>
    {!market ? <p class="note muted" role="status">Calculating demand…</p> : <>
      <p class="city-trip-total small"><b>{count(total)}</b> intercity trips · <b>{count(rail)}</b> by train</p>
      <p class="muted small">Both directions per day · Top 5 connections</p>
      <ul class="corridors">
        {corridors?.slice(0, 5).map(c => {
          const other = c.from === city.market ? c.to : c.from;
          const share = Math.max(0, Math.min(100, 100 * c.rail / c.trips));
          const description = `${INTERCITY_PLACES[other]}: ${count(c.trips)} trips a day; train ${count(c.rail)} (${Math.round(share)}%), other ${count(c.trips - c.rail)} (${Math.round(100-share)}%)`;
          return <li key={other} title={description}>
            <div class="corridor-heading"><span>{INTERCITY_PLACES[other]}</span><span class="num">{count(c.rail)}/{count(c.trips)} trips</span></div>
            <div class="bar corridor-bar" role="img" aria-label={description}>
              <i class="seg seg-train" style={{width: `${share}%`}} />
              <i class="seg seg-drive" style={{width: `${100-share}%`}} />
            </div>
          </li>;
        })}
      </ul>
      {!corridors?.length && <p class="muted small">No intercity demand is modelled here yet.</p>}
    </>}
  </>;
}
