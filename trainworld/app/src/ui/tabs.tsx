// Tab contents: build, lines, stations, money, city.

import { clock } from "../game/clock";
import { lineDemand, stationDemand, demandView } from "../game/demand";
import type { JSX } from "preact";
import { count, levelLabel, money } from "../game/format";
import { DEMAND_INDEX, LEVELS, type Level } from "../game/types";
import { buildLevel, buildPlatform, lineDraft, selection, singleTrack, tool, type Tool } from "../game/ui";
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
  const w = world.value;
  if (!w) return null;
  const sel = selection.value;
  const lev = DEMAND_INDEX[clock.demand.value];
  const total = w.save.lines.reduce((s, l) => s + (lineDemand(l.id)?.ridersPerDay ?? 0), 0);
  const newLine = () => {
    live.tools?.exit();
    lineDraft.value = { line: null, stops: [] };
    tool.value = "line";
  };
  return (
    <>
      {w.save.lines.length === 0 && <p class="muted">No lines yet. Put stations on track, then make a line through them.</p>}
      <ul class="rows pick">
        {w.save.lines.map((l) => {
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
                  <span class="val">{dm ? count(dm.ridersPerDay) : ""}</span>
                  <span class="val muted w-trains">{st.trainsNeeded[lev]} trains</span>
                </>
              )}
            </li>
          );
        })}
      </ul>
      <div class="pane-foot">
        <span class="muted">{demandView.value ? count(total) + " riders a day" : ""}</span>
        <button class={"btn text" + (tool.value === "line" ? " on" : "")} onClick={newLine}>
          <Icon.plus />
          New line
        </button>
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
  const w = world.value;
  if (!w) return null;
  const by = demandView.value?.trainShareByPeriod ?? w.city.trainShareByPeriod;
  const max = Math.max(1e-9, ...by.map((p) => p[1]));
  return (
    <>
      <ul class="rows">
        <Row label="People">{count(w.city.population)}</Row>
        <Row label="Jobs">{count(w.city.jobs)}</Row>
      </ul>
      {by.length > 0 && <div class="h">Trips by train</div>}
      <ul class="rows">
        {by.map(([n, v]) => (
          <li>
            <span class="w-period">{n}</span>
            <span class="grow">
              <i class="hbar" style={{ width: (v / max) * 100 + "%" }} />
            </span>
            <span class="val w-num">{v.toFixed(1)}%</span>
          </li>
        ))}
      </ul>
    </>
  );
}
