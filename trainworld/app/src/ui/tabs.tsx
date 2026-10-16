// Tab contents: build, lines, stations, money (settings: SettingsTab.tsx). The City tab is gone
// (T-220): the city overview's bubbles and the city inspector show its trips, search goes to a
// city, and Explore a trip is at the bottom of Settings.

import { clock, demandAt } from "../game/clock";
import { PERIODS, periodAt } from "../game/periods";
import { useLayoutEffect, useRef, useState } from "preact/hooks";
import { createLine } from "../game/lineActions";
import type { JSX } from "preact";
import { count, distUnit, length, levelLabel, money, moneyExact, perDistance, perKmFrom, shortDistance } from "../game/format";
import { inkOn, LEVEL_COLOURS } from "../game/palette";
import { DEMAND_INDEX, LEVELS, type Level, type LineInput, type StationInput } from "../game/types";
import { complexes } from "../game/complexView";
import { membersOf } from "../game/complexes";
import { buildLevel, buildPlatform, display, keepPlacing, lineSort, selection, setDisplay, stationSort, trackCount, tool, type ListSort, type StationUseMode, type Tool } from "../game/ui";
import type { Signal } from "@preact/signals";
import { setFares, world } from "../game/world";
import { liveMoney, moneyState } from "../game/money";
import { cityById, viewedZone } from "../game/cities";
import { listScope, scopedLists, scopeZone } from "../game/listScope";
import { regionLedger } from "../game/regionMoney";
import { accountVersion } from "../game/boarding";
import { crowdClass, crowding, stationsToday, today, todayLoaded, useToday, waitText } from "../game/today";
import type { DayLedger } from "../workers/protocol";
import { live } from "../map/live";
import { Check, Chip, Icon, Row, Segmented, Stepper } from "./widgets";
import { BlueprintBreakdown, BlueprintControls, BlueprintPick } from "./Blueprint";
import { MoneyHistory } from "./MoneyHistory";

const STATUS: Record<string, string> = { running: "", planned: "Planned", broken: "Broken" };

/** Line names in natural order ("2" before "10"). One collator for every comparison: localeCompare
 * with options builds a new one each call, which made sorting 109 lines take 15-400 ms (T-348). */
const natural = new Intl.Collator(undefined, { numeric: true, sensitivity: "base" }).compare;

// SPEC 6.4. A display copy: the track model (sim/src/track/params.rs) owns the real price list.
const BASE_M_PER_KM = 45;
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
  ["crossover", "Crossover", Icon.crossover, "6", "Click on track beside other track to join the two with a crossover"],
  ["delete", "Delete", Icon.bin, "3", "Delete track, stations and flyovers"],
  ["blueprint", "Select", Icon.marquee, "5", "Drag a box round blueprint track and stations to disable or enable them"],
];

/** The tool each number key picks (T-096). */
export const TOOL_KEYS: Record<string, Tool> = { "1": "track", "2": "station", "3": "delete", "5": "blueprint", "6": "crossover" };
/** The trip explorer's key (T-195, a letter since T-444: the numbers are for building); unlike the
 * build tools it leaves the tab as it is. Compared lowercase, so Caps Lock does not matter. */
export const TRIP_KEY = "t";

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
        <div class="h">Elevation</div>
        <div class="group levels">
          {LEVELS.map((l) => (
            <button class={"btn lv" + (l === lv ? " on" : "")} style={{ "--lc": LEVEL_COLOURS[l], "--lk": inkOn(LEVEL_COLOURS[l]) }} data-tip={LEVEL_NAME[l]} aria-pressed={l === lv} onClick={() => (buildLevel.value = l)}>
              {levelLabel(l)}
            </button>
          ))}
        </div>
        <p class="muted small price">
          {LEVEL_NAME[lv]} at ${Math.round(perDistance(BASE_M_PER_KM * MULT[lv]) * 10) / 10}M per {distUnit()} ({MULT[lv].toFixed(1)}x)
        </p>
        <div class="h">Track</div>
        <div class="track-row">
          <Segmented<1 | 2 | 4> options={[[1, "Single"], [2, "Double"], [4, "Quad"]]} value={trackCount.value} onChange={(v) => (trackCount.value = v)} />
          <button class={"btn icon" + (keepPlacing.value ? " on" : "")} aria-label="Keep placing track" aria-pressed={keepPlacing.value}
            data-tip="Keep placing track (Caps Lock). Double click or press Esc to stop."
            onClick={() => (keepPlacing.value = !keepPlacing.peek())}><Icon.keepPlacing /></button>
        </div>
        <div class="h">New stations</div>
        <ul class="rows">
          <li>
            <span class="grow">Platform length</span>
            <Stepper value={length(buildPlatform.value)} wide label="platform length" onStep={(s) => (buildPlatform.value = Math.max(60, Math.min(400, buildPlatform.value + s * 20)))} />
          </li>
        </ul>
        <BlueprintPick />
        <BlueprintBreakdown />
      </div>
      <BlueprintControls />
    </>
  );
}

/** T-310: a list's column headers. A click sorts by the column (numbers biggest first), a second
 * click reverses it, and the sorted column carries a small arrow. A column with a `cls` is a
 * right-aligned number column of that width; the arrow sits before its label so the label stays
 * over the figures. */
function SortHead<K extends string>({ sort, cols }: { sort: Signal<ListSort<K>>; cols: { key: K; label: string; tip: string; first: 1 | -1; cls?: string }[] }) {
  const s = sort.value;
  return (
    <div class="sort-head">
      {cols.map((c) => {
        const on = s.key === c.key;
        const arrow = <i class={"sort-arrow" + (on ? (s.dir > 0 ? " up" : " down") : "")} />;
        return (
          <button class={"sort-btn" + (c.cls ? " val " + c.cls : " grow") + (on ? " on" : "")} data-tip={c.tip}
            onClick={() => (sort.value = on ? { key: c.key, dir: s.dir === 1 ? -1 : 1 } : { key: c.key, dir: c.first })}>
            {c.cls && arrow}{c.label}{!c.cls && arrow}
          </button>
        );
      })}
    </div>
  );
}

/** T-332: the viewed place or the whole world. Over dead space only the world is on offer. `tip`
 * adds a line to the place's hover (T-531). */
function ScopeToggle({ tip }: { tip?: string }) {
  const zone = viewedZone.value;
  const name = zone ? cityById(zone).name : null;
  const here = listScope.value === "here" && !!zone;
  return (
    <div class="group scope">
      <button class={"btn" + (here ? " on" : "")} disabled={!zone} data-tip={name ? `Only ${name}` + (tip ? "\n" + tip : "") : "Nothing modelled here"}
        onClick={() => (listScope.value = "here")}>
        <Icon.pin />{name}
      </button>
      <button class={"btn" + (here ? "" : " on")} data-tip="The whole world" aria-label="The whole world" onClick={() => (listScope.value = "world")}>
        <Icon.globe />
      </button>
    </div>
  );
}

/** T-332: a row's distance from the centre of the screen. */
function Near({ m }: { m: number | undefined }) {
  return <span class="val muted w-near" data-tip={m !== undefined ? `${shortDistance(m)} from the centre of the screen` : undefined}>{m !== undefined ? shortDistance(m) : ""}</span>;
}

/** Rows with no distance (a line with no stops) sort last whichever way the column runs. */
const byNear = (a: number | undefined, b: number | undefined, dir: number) =>
  a === undefined || b === undefined ? (a === undefined ? 1 : 0) - (b === undefined ? 1 : 0) : (a - b) * dir;

/** T-446: one line's riders in the past 24 hours, its own component so only it re-renders as the count moves. */
function RidersToday({ line }: { line: string }) {
  const d = today.value.get(line);
  return <span class="val w-riders" data-tip={d ? `${count(d.riders)} riders in the past 24 hours` : undefined}>{d ? count(d.riders) : ""}</span>;
}

/** T-472, T-491: a line's trains now, coloured by its riders' wait for a train with room in the period now. */
function TrainsNow({ line, n, period }: { line: string; n: number; period: string }) {
  const d = today.value.get(line);
  const c = d ? crowding(d.crowdWait, d.crowdRiders, d.crowdRide) : null;
  const cls = c ? crowdClass(c.score) : "";
  const tip = d && c && d.crowdRiders > 0
    ? `${period}: riders wait ${waitText(c.wait)} on average for a train with room` + (d.trainsNow > 0 ? `\n${count(d.fullNow)}/${count(d.trainsNow)} trains running now are packed full on their trip` : "")
    : undefined;
  return <span class={"val w-trains " + (cls || "muted")} data-tip={tip}>{n}</span>;
}

function TotalToday() {
  const t = today.value;
  if (!t.size) return null;
  let n = 0;
  for (const d of t.values()) n += d.riders;
  return <span class="muted">{count(n)} riders in the past 24 hours</span>;
}

/** T-446: a station's or complex's boardings in the past 24 hours. */
function StationToday({ ids }: { ids: string[] }) {
  const by = stationsToday.value;
  let n: number | null = null;
  for (const id of ids) { const v = by.get(id); if (v !== undefined) n = (n ?? 0) + v; }
  return <span class="val w-num" data-tip={n !== null ? `${count(n)} boardings in the past 24 hours` : undefined}>{n !== null ? count(n) : ""}</span>;
}

export function LinesTab() {
  const [creating, setCreating] = useState(false);
  useToday(); // T-446: riders in the past 24 hours, kept fresh while the tab is open
  void accountVersion.value; // re-sorted each game hour; the counts themselves move on their own
  void todayLoaded.value;
  const w = world.value;
  if (!w) return null;
  const sel = selection.value;
  // T-309: each line's demand level now in its own local time
  const hour = clock.hour.value;
  const levOf = (id: string) => DEMAND_INDEX[demandAt(hour * 3600 + (w.lineStats[id]?.shift ?? 0))];
  const t = today.peek();
  const newLine = async () => {
    if (creating) return;
    setCreating(true);
    try { await createLine(); } finally { setCreating(false); }
  };
  // a line that is not running has no train count and no riders: it sorts as zero
  const trains = (id: string) => (w.lineStats[id]?.status === "running" ? w.lineStats[id].trainsNeeded[levOf(id)] : 0);
  const { key, dir } = lineSort.value;
  // T-332: only the viewed place unless the world is chosen
  const sc = scopedLists(w.save.stations, w.save.lines, (k, id) => (k === "line" && (sel?.kind === "line" || sel?.kind === "train") && sel.line === id));
  const lines = [...sc.lines].sort((a, b) =>
    (key === "riders" ? ((t.get(a.id)?.riders ?? 0) - (t.get(b.id)?.riders ?? 0)) * dir
      : key === "trains" ? (trains(a.id) - trains(b.id)) * dir
      : key === "near" ? byNear(sc.lineDist.get(a.id), sc.lineDist.get(b.id), dir) : 0)
    || natural(a.name, b.name) * (key === "name" ? dir : 1) || a.num - b.num);
  return (
    <>
      <div class="line-list-actions">
        <button class="btn text" onClick={newLine} disabled={creating}><Icon.plus />New line</button>
        <ScopeToggle />
      </div>
      {lines.length === 0 && <p class="muted">{w.save.lines.length === 0 ? "No lines yet." : "No lines here."}</p>}
      {w.save.lines.length > 0 && <SortHead sort={lineSort} cols={[
        { key: "name", label: "Name", tip: "Sort by name", first: 1 },
        { key: "riders", label: "Riders", tip: "Sort by riders in the past 24 hours", first: -1, cls: "w-riders" },
        { key: "trains", label: "Trains", tip: "Sort by trains needed", first: -1, cls: "w-trains" },
        { key: "near", label: "Near", tip: "Sort by distance from the centre of the screen, by the nearest stop", first: 1, cls: key === "near" ? "w-near" : "near-off" },
      ]} />}
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
                  <TrainsNow line={l.id} n={st.trainsNeeded[levOf(l.id)]} period={PERIODS[periodAt(hour * 3600 + (st.shift ?? 0))]?.name ?? ""} />
                </>
              )}
              {key === "near" && <Near m={sc.lineDist.get(l.id)} />}
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

/** Width kept for the "+" after a row's last whole bullet, px. */
const PLUS_W = 10;
const fitters = new WeakMap<Element, () => void>();
let chipsObserver: ResizeObserver | null = null;
function watchSize(el: Element, f: () => void) {
  chipsObserver ??= new ResizeObserver((es) => { for (const e of es) fitters.get(e.target)?.(); });
  fitters.set(el, f);
  chipsObserver.observe(el);
  return () => { chipsObserver?.unobserve(el); fitters.delete(el); };
}

/** T-521, T-530: a station row's bullets, laid out with the row's name (`.grow > span`, its sibling)
 * in the room the fixed columns leave (Anita: the name comes first). Every bullet shows when the
 * whole name fits beside them; otherwise as many as fit beside the whole name, never fewer than 3,
 * and only then is the name cut. Bullets left out sit behind a "+" whose hover names their lines;
 * they stay in the box, clipped, so their widths can be measured. */
function Chips({ lines, name }: { lines: LineInput[]; name: string }) {
  const ref = useRef<HTMLSpanElement>(null);
  const [fit, setFit] = useState<{ shown: number; width: number }>({ shown: lines.length, width: -1 });
  const key = lines.map((l) => l.id).join(",");
  useLayoutEffect(() => {
    const el = ref.current, row = el?.parentElement;
    if (!el || !row) return;
    const measure = () => {
      const chips = el.querySelectorAll<HTMLElement>(":scope > .chip");
      const n = chips.length;
      const right = (i: number) => chips[i].offsetLeft + chips[i].offsetWidth;
      const cs = getComputedStyle(row);
      const gap = parseFloat(cs.columnGap) || 0;
      let room = row.clientWidth - parseFloat(cs.paddingLeft) - parseFloat(cs.paddingRight) - gap * (row.children.length - 1);
      let nameW = 0;
      for (const c of row.children) {
        if (c === el) continue;
        if (c.classList.contains("grow")) nameW = (c.firstElementChild as HTMLElement | null)?.offsetWidth ?? 0;
        else room -= (c as HTMLElement).offsetWidth;
      }
      let shown = n;
      if (n && nameW + right(n - 1) > room) {
        shown = Math.min(3, n);
        for (let k = n - 1; k > 3; k--) if (nameW + right(k - 1) + 2 + PLUS_W <= room) { shown = k; break; }
      }
      const width = n ? right(Math.max(shown, 1) - 1) + (shown < n ? 2 + PLUS_W : 0) : 0;
      setFit((f) => (f.shown === shown && f.width === width ? f : { shown, width }));
    };
    measure();
    return watchSize(row, measure);
  }, [key, name]);
  const shown = Math.min(fit.shown, lines.length);
  return (
    <span class="chips" ref={ref} style={{ width: fit.width >= 0 ? `${fit.width}px` : undefined }}>
      {lines.map((l, i) => <Chip line={l} size="sm" gone={i >= shown} />)}
      {shown < lines.length && <span class="chips-more" style={{ left: `${fit.width - PLUS_W}px` }}
        data-tip={[...new Set(lines.slice(shown).map((l) => l.name))].join("\n")}>+</span>}
    </span>
  );
}

/** A Stations tab row: a station complex (T-530), or a station on its own. */
interface StationRow {
  /** the member a click selects, whose name the row takes: most lines, then constructed, then save order */
  lead: StationInput;
  members: string[];
  lines: LineInput[];
  built: boolean;
  dist: number | undefined;
}

/** The Stations tab's rows: one per complex with a member in `stations` (T-530, Anita: Times Sq's
 * platforms were rows of one bullet each), its lines and numbers over the whole complex. */
function stationRows(stations: readonly StationInput[], all: readonly StationInput[], lines: readonly LineInput[], dist: Map<string, number>): StationRow[] {
  const cx = complexes.value;
  const byId = new Map(all.map((s) => [s.id, s]));
  const lineIdx = new Map<string, number[]>();
  lines.forEach((l, i) => { for (const id of l.stops) { const a = lineIdx.get(id); if (!a) lineIdx.set(id, [i]); else if (a[a.length - 1] !== i) a.push(i); } });
  const seen = new Set<number | string>();
  const rows: StationRow[] = [];
  for (const s of stations) {
    const k = cx.of.get(s.id) ?? s.id;
    if (seen.has(k)) continue;
    seen.add(k);
    const members = membersOf(cx, s.id);
    const at = new Set<number>();
    let lead = s, leadLines = -1, built = false, d: number | undefined;
    for (const id of members) {
      const m = byId.get(id);
      if (!m) continue;
      const li = lineIdx.get(id) ?? [];
      for (const i of li) at.add(i);
      if (li.length > leadLines || (li.length === leadLines && m.built && !lead.built)) { lead = m; leadLines = li.length; }
      built ||= m.built;
      const dm = dist.get(id);
      if (dm !== undefined && (d === undefined || dm < d)) d = dm;
    }
    rows.push({ lead, members, lines: [...at].sort((a, b) => a - b).map((i) => lines[i]), built, dist: d });
  }
  return rows;
}

export function StationsTab() {
  useToday(); // T-293
  void accountVersion.value; // re-sorted each game hour, as the lines
  void todayLoaded.value;
  const w = world.value;
  if (!w) return null;
  const sel = selection.value;
  const by = stationsToday.peek();
  const { key, dir } = stationSort.value;
  const sc = scopedLists(w.save.stations, w.save.lines, (k, id) => k === "station" && sel?.kind === "station" && sel.station === id);
  const riders = (r: StationRow) => r.members.reduce((t, id) => t + (by.get(id) ?? 0), 0);
  const st = stationRows(sc.stations, w.save.stations, w.save.lines, sc.stationDist).sort((a, b) =>
    (key === "riders" ? (riders(a) - riders(b)) * dir
      : key === "near" ? byNear(a.dist, b.dist, dir) : 0)
    || a.lead.name.localeCompare(b.lead.name) * (key === "name" ? dir : 1));
  const d = display.value;
  return (
    <>
      <div class="line-list-actions"><ScopeToggle /></div>
      {/* T-213: the stations' own map setting sits with them, above the list */}
      {/* T-482: station use circles, in place of the demand circles */}
      <div class="h">Show station ridership</div>
      <div class="station-use"><Segmented<StationUseMode>
        options={[
          ["none", "None"],
          ["origins", <span data-tip="Riders starting a rail trip at the station, a day">Origins</span>],
          ["transfers", <span data-tip="Riders changing lines at the station, a day">Transfers</span>],
          ["total", <span data-tip="Every boarding at the station, a day: origins and transfers">Total</span>],
        ]}
        value={d.stationUse}
        onChange={(v) => setDisplay("stationUse", v)}
      /></div>
      {d.stationUse !== "none" && <div class="checks">
        <Check checked={d.complexView} onChange={(v) => setDisplay("complexView", v)}>
          One circle per station complex
        </Check>
      </div>}
      {st.length === 0 && <p class="muted">{w.save.stations.length === 0 ? "No stations yet." : "No stations here."}</p>}
      {st.length > 0 && <SortHead sort={stationSort} cols={[
        { key: "name", label: "Name", tip: "Sort by name", first: 1 },
        { key: "riders", label: "24 h", tip: "Sort by boardings in the past 24 hours", first: -1, cls: "w-num" },
        { key: "near", label: "Near", tip: "Sort by distance from the centre of the screen", first: 1, cls: key === "near" ? "w-near" : "near-off" },
      ]} />}
      <ul class="rows pick stations">
        {st.map((r) => (
          <li class={sel?.kind === "station" && r.members.includes(sel.station) ? "on" : ""} onClick={() => (selection.value = { kind: "station", station: r.lead.id })}>
            <span class={"grow" + (r.built ? "" : " muted")}><span>{r.lead.name}</span></span>
            <Chips lines={r.lines} name={r.lead.name} />
            {r.built ? <StationToday ids={r.members} /> : <span class="val w-num">Plan</span>}
            {key === "near" && <Near m={r.dist} />}
          </li>
        ))}
      </ul>
      {st.length > 0 && (
        <div class="pane-foot">
          <span class="muted">{sc.zone ? `${st.length} of ${complexes.value.members.length}` : st.length} stations, boardings in the past 24 hours</span>
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

/** T-284: one figure of today's column as fares land; the network's or a region's (T-531). */
function TodayCell({ v, zone }: { v: (d: DayLedger) => number; zone: string | null }) {
  const d = regionLedger(zone).today;
  return <td class="num">{d ? money(v(d) * 1e6) : ""}</td>;
}

/** T-531: how a region's money is counted, in the scope toggle's hover. */
const REGION_SPLIT = "A line's fares count where its riders board and travel, its running costs where its trains run. Construction counts where it was built.";

export function MoneyTab() {
  const w = world.value;
  const m = moneyState.value;
  if (!w || !m) return null;
  // T-284: the cash and today's column follow fares as they land (`liveMoney`, own components)
  // T-531: the ledger is the viewed region's unless the whole world is chosen, as in the lists
  const zone = scopeZone();
  const todayDay = m.ledger[m.ledger.length - 1]?.day ?? 0;
  const prevAll = m.ledger.find((d) => d.day === todayDay - 1) ?? null;
  const prev = zone === null || !prevAll ? prevAll
    : regionLedger(zone).ledger.find((d) => d.day === todayDay - 1) ?? { day: todayDay - 1, fares: 0, running: 0, trains: 0, build: 0 };
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
          <th style={{ textAlign: "left" }}><ScopeToggle tip={REGION_SPLIT} /></th>
          <th>Yesterday</th>
          <th>Today</th>
        </tr>
        <tr>
          <td>Fares</td>
          {cell(prev, (d) => d.fares)}
          <TodayCell zone={zone} v={(d) => d.fares} />
        </tr>
        <tr>
          <td>Running trains</td>
          {cell(prev, (d) => -d.running)}
          <TodayCell zone={zone} v={(d) => -d.running} />
        </tr>
        <tr>
          <td>New trains</td>
          {cell(prev, (d) => -d.trains)}
          <TodayCell zone={zone} v={(d) => -d.trains} />
        </tr>
        <tr>
          <td>Construction</td>
          {cell(prev, (d) => -d.build)}
          <TodayCell zone={zone} v={(d) => -d.build} />
        </tr>
        <tr class="b">
          <td>Net</td>
          {cell(prev, ledgerNet)}
          <TodayCell zone={zone} v={ledgerNet} />
        </tr>
      </table>
      <MoneyHistory zone={zone} />
      <div class="h">Fare</div>
      <ul class="rows">
        <li>
          <span class="grow">Each ride</span>
          <Stepper value={"$" + f.base.toFixed(2)} wide label="fare a ride" onStep={(s) => setFares(f.base + s * 0.25, f.perKm)} />
        </li>
        <li>
          <span class="grow">Each {distUnit()}</span>
          <Stepper value={"$" + perDistance(f.perKm).toFixed(2)} wide label={`fare a ${distUnit()}`} onStep={(s) => setFares(f.base, perKmFrom(Math.max(0, Math.round((perDistance(f.perKm) + s * 0.01) * 100) / 100)))} />
        </li>
        <Row label={`A 10 ${distUnit()} ride`}>{"$" + (f.base + 10 * perDistance(f.perKm)).toFixed(2)}</Row>
      </ul>
      <div class="h">Trains</div>
      <ul class="rows">
        <Row label={<span data-tip={`A car costs ${money(m.carPrice * 1e6)} and $${perDistance(m.runCostCarKm).toFixed(2)} a ${distUnit()} to run. Cars are bought when a line needs more than the spare ones.`}>Cars owned</span>}>{m.fleet}</Row>
        <Row label="Cars the lines need">{m.carsNeeded}</Row>
      </ul>
      <ul class="rows">
        <Row label="Track and stations built">{money(w.money.builtValue)}</Row>
      </ul>
    </>
  );
}

