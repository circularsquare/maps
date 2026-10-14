// Tab contents: build, lines, stations, money (settings: SettingsTab.tsx). The City tab is gone
// (T-220): the city overview's bubbles and the city inspector show its trips, search goes to a
// city, and Explore a trip is at the bottom of Settings.

import { clock } from "../game/clock";
import { useState } from "preact/hooks";
import { createLine } from "../game/lineActions";
import { lineDemand, stationDemand, demandView } from "../game/demand";
import type { JSX } from "preact";
import { count, levelLabel, money, moneyExact } from "../game/format";
import { DEMAND_INDEX, LEVELS, type Level } from "../game/types";
import { buildLevel, buildPlatform, display, lineSort, selection, setDisplay, trackCount, tool, type Tool } from "../game/ui";
import { linesAt, setFares, world } from "../game/world";
import { income, liveMoney, moneyState } from "../game/money";
import { accountVersion } from "../game/boarding";
import { stationsToday, today, todayLoaded, useToday } from "../game/today";
import type { DayLedger } from "../workers/protocol";
import { live } from "../map/live";
import { Check, Chip, Icon, Row, Segmented, Stepper } from "./widgets";
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
/** The trip explorer's key (T-195); unlike the build tools it leaves the tab as it is. */
export const TRIP_KEY = "4";

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
            <button class={"btn text" + (t === id ? " on" : "")} data-tip={`${title} (${key})`} onClick={() => pickTool(id)}>
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
        <Segmented<1 | 2 | 4> options={[[1, "Single"], [2, "Double"], [4, "Quad"]]} value={trackCount.value} onChange={(v) => (trackCount.value = v)} />
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

/** T-293: one line's riders today, its own component so only it re-renders as the count moves. */
function RidersToday({ line }: { line: string }) {
  const d = today.value.get(line);
  return <span class="val" data-tip={d ? `${count(d.riders)} riders today` : undefined}>{d ? count(d.riders) : ""}</span>;
}

function TotalToday() {
  const t = today.value;
  if (!t.size) return null;
  let n = 0;
  for (const d of t.values()) n += d.riders;
  return <span class="muted">{count(n)} riders today</span>;
}

/** T-293: one station's boardings today. */
function StationToday({ id }: { id: string }) {
  const n = stationsToday.value.get(id) ?? null;
  return <span class="val w-num" data-tip={n !== null ? `${count(n)} boardings today` : undefined}>{n !== null ? count(n) : ""}</span>;
}

export function LinesTab() {
  const [creating, setCreating] = useState(false);
  useToday(); // T-293: riders today, kept fresh while the tab is open
  void accountVersion.value; // re-sorted each game hour; the counts themselves move on their own
  void todayLoaded.value;
  const w = world.value;
  if (!w) return null;
  const sel = selection.value;
  const lev = DEMAND_INDEX[clock.demand.value];
  const t = today.peek();
  const newLine = async () => {
    if (creating) return;
    setCreating(true);
    try { await createLine(); } finally { setCreating(false); }
  };
  const natural = (a: string, b: string) => a.localeCompare(b, undefined, { numeric: true, sensitivity: "base" });
  const lines = [...w.save.lines].sort((a, b) =>
    (lineSort.value === "passengers" ? (t.get(b.id)?.riders ?? 0) - (t.get(a.id)?.riders ?? 0) : 0)
    || natural(a.name, b.name) || a.num - b.num);
  return (
    <>
      <div class="line-list-actions">
        <button class="btn text" onClick={newLine} disabled={creating}><Icon.plus />New line</button>
        <label>Sort <select aria-label="Sort lines" value={lineSort.value} onChange={e => lineSort.value = e.currentTarget.value as "name" | "passengers"}>
          <option value="name">Name</option><option value="passengers">Riders today</option>
        </select></label>
      </div>
      {w.save.lines.length === 0 && <p class="muted">No lines yet.</p>}
      <ul class="rows pick">
        {lines.map((l) => {
          const st = w.lineStats[l.id];
          const on = (sel?.kind === "line" || sel?.kind === "train") && sel.line === l.id;
          return (
            <li class={on ? "on" : ""} onClick={() => (selection.value = { kind: "line", line: l.id })}>
              <Chip line={l} />
              <span class="grow">{l.name}</span>
              {st.status !== "running" ? (
                <span class="val muted">{STATUS[st.status]}</span>
              ) : (
                <>
                  <RidersToday line={l.id} />
                  <span class="val muted w-trains">{st.trainsNeeded[lev]} {st.trainsNeeded[lev] === 1 ? "train" : "trains"}</span>
                </>
              )}
            </li>
          );
        })}
      </ul>
      <div class="pane-foot">
        <TotalToday />
      </div>
    </>
  );
}

export function StationsTab() {
  useToday(); // T-293
  void accountVersion.value; // re-sorted each game hour, as the lines
  void todayLoaded.value;
  const w = world.value;
  if (!w) return null;
  const sel = selection.value;
  const by = stationsToday.peek();
  const st = [...w.save.stations].sort((a, b) => (by.get(b.id) ?? 0) - (by.get(a.id) ?? 0) || a.name.localeCompare(b.name));
  const d = display.value;
  return (
    <>
      {/* T-213: the stations' own map setting sits with them, above the list */}
      <div class="checks">
        <Check checked={d.stationRiders} onChange={(v) => setDisplay("stationRiders", v)}>
          <span data-tip="Circle area shows daily entries and exits, up to 150,000">Station demand circles</span>
        </Check>
        {d.stationRiders && <Check checked={d.complexView} onChange={(v) => setDisplay("complexView", v)}>
          One circle per station complex
        </Check>}
      </div>
      {st.length === 0 && <p class="muted">No stations yet.</p>}
      <ul class="rows pick">
        {st.map((s) => (
          <li class={sel?.kind === "station" && sel.station === s.id ? "on" : ""} onClick={() => (selection.value = { kind: "station", station: s.id })}>
            <span class={"grow" + (s.built ? "" : " muted")}>{s.name}</span>
            {linesAt(s.id).map((l) => (
              <Chip line={l} size="sm" />
            ))}
            {s.built ? <StationToday id={s.id} /> : <span class="val w-num">Plan</span>}
          </li>
        ))}
      </ul>
      {st.length > 0 && (
        <div class="pane-foot">
          <span class="muted">{st.length} stations, boardings today</span>
        </div>
      )}
    </>
  );
}

const ledgerNet = (d: DayLedger) => d.fares - d.running - d.trains - d.build;

/** T-284: the cash as fares land, to the dollar, as in the top bar. */
function LiveCash() {
  return <>{moneyExact(liveMoney.value.cash * 1e6)}</>;
}

/** T-284: one figure of today's column as fares land. */
function TodayCell({ v }: { v: (d: DayLedger) => number }) {
  const d = liveMoney.value.today;
  return <td class="num">{d ? money(v(d) * 1e6) : ""}</td>;
}

export function MoneyTab() {
  const w = world.value;
  const m = moneyState.value;
  if (!w || !m) return null;
  const inc = income.value;
  // T-284: the cash and today's column follow fares as they land (`liveMoney`, own components)
  const todayDay = m.ledger[m.ledger.length - 1]?.day ?? 0;
  const prev = m.ledger.find((d) => d.day === todayDay - 1) ?? null;
  const cell = (d: DayLedger | null, v: (d: DayLedger) => number) => <td class="num">{d ? money(v(d) * 1e6) : ""}</td>;
  const f = m.fares;
  return (
    <>
      <ul class="rows">
        <Row label="Cash" bold>
          <LiveCash />
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
          <TodayCell v={(d) => d.fares} />
        </tr>
        <tr>
          <td>Running trains</td>
          {cell(prev, (d) => -d.running)}
          <TodayCell v={(d) => -d.running} />
        </tr>
        <tr>
          <td>New trains</td>
          {cell(prev, (d) => -d.trains)}
          <TodayCell v={(d) => -d.trains} />
        </tr>
        <tr>
          <td>Construction</td>
          {cell(prev, (d) => -d.build)}
          <TodayCell v={(d) => -d.build} />
        </tr>
        <tr class="b">
          <td>Net</td>
          {cell(prev, ledgerNet)}
          <TodayCell v={ledgerNet} />
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
      <div class="h"><span data-tip={`Fares and running costs count ${m.economy} times over in a game day. Track, stations and trains cost what they would in real life.`}>A day at this schedule</span></div>
      <ul class="rows">
        <Row label="Fares">{demandView.value ? money(inc.faresPerDay) : ""}</Row>
        <Row label="Running trains">{money(-inc.runningPerDay)}</Row>
      </ul>
      <div class="h">Trains</div>
      <ul class="rows">
        <Row label={<span data-tip={`A car costs ${money(m.carPrice * 1e6)} and $${m.runCostCarKm.toFixed(2)} a km to run. Cars are bought when a line needs more than the spare ones.`}>Cars owned</span>}>{m.fleet}</Row>
        <Row label="Cars the lines need">{m.carsNeeded}</Row>
      </ul>
      <ul class="rows">
        <Row label="Track and stations built">{money(w.money.builtValue)}</Row>
      </ul>
    </>
  );
}

