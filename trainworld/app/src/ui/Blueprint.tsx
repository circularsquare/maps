import { useEffect, useMemo, useState } from "preact/hooks";
import { describeIssues } from "../game/issues";
import { useLatestAnswer } from "./useClockQuery";
import { queryClock } from "../workers/clockClient";
import { inkOn, LEVEL_COLOURS } from "../game/palette";
import type { Level } from "../game/types";
import { distUnit, distance, levelLabel, levelName, money, moneyCash, perDistance, plural, unitShort, upkeepText } from "../game/format";
import { edgeUpkeep, stationUpkeep } from "../game/regionMoney";
import { bpPicked, clearBpPicked, doubleUpPicked, doubleUpPieces, enableAll, levelPicked, offTotals, pickedState, removePicked, switchPicked } from "../game/blueprintPick";
import { liveMoney, moneyState } from "../game/money";
import { tool } from "../game/ui";
import type { BlueprintItem } from "../game/types";
import { constructBlueprints } from "../game/lineActions";
import { world } from "../game/world";
import { useBlueprintQuote } from "../game/quotes";
import { redo, undo } from "../game/fareUndo";
import { Icon, LevelButtons } from "./widgets";
import { TERRAIN_NAMES, TERRAIN_SURCHARGES, WATER_SURCHARGE } from "../game/terrain";

// 6 T-1270's terrain, 7 left for another row, 8 and 9 T-1281's, 10 and 11 T-1464's bus routes and buses
const KINDS = ["Track", "Stations", "Junctions", "Flyovers", "Flat crossings", "Trains", "Terrain", "", "Station upgrades", "Passages", "Bus routes", "Buses"];
/** a quote row's track count for high-speed track (sim `HSR_ROW`, T-817) */
const HSR_ROW = 3;

const NOUN = ["track", "station", "junction", "flyover", "flat crossing", "car", "ground", "", "station upgrade", "passage", "route", "bus"];
const levelWords = (r: BlueprintItem) => `${levelName(r.level).toLowerCase()} (${levelLabel(r.level)})`;

/** T-1839: a quote row as it reads in the list, "3.7 km viaduct (+2)"; the price's factors are in `rowTip`. */
function rowName(r: BlueprintItem) {
  if (r.kind === 5) return plural(r.quantity, "new car");
  if (r.kind === 10) return distance(r.quantity);
  if (r.kind === 11) return plural(r.quantity, "new bus", "new buses");
  if (r.kind === 9) return plural(r.quantity, "passage");
  if (r.kind === 0) return `${distance(r.quantity)} ${levelWords(r)}`;
  return `${plural(r.quantity, NOUN[r.kind] || "item")}, ${levelWords(r)}`;
}

// A display copy of the price list (sim/src/track/params.rs and cost.rs own the real one), so a row's
// hover can show its price as the sum it is.
const BASE_M_PER_KM = 90;
const MULT: Record<Level, number> = { 3: 1.8, 2: 1.3, 1: 1.0, 0: 0.5, [-1]: 1.0, [-2]: 1.1, [-3]: 1.2 };
const STATION_M: Record<Level, number> = { 3: 70, 2: 55, 1: 40, 0: 25, [-1]: 75, [-2]: 100, [-3]: 125 };
const STATION_FIXED_M = 50;
const SINGLE_STATION = 0.7, SINGLE_TRACK = 0.6, HSR_TRACK = 1.5, WATER = 2, BUS_ROUTE_M_PER_KM = 1;
/** the level a flying junction's flyover sits at (sim `junction_cost`) */
const awayLevel = (l: Level): Level => (l === 3 ? 2 : l === -3 ? -2 : l >= 0 ? l + 1 : l - 1) as Level;
const num = (v: number) => String(Number(v.toFixed(2)));
/** a price in US$M as the Build tab quotes it: "$90M", "$1.5M" */
const mil = (v: number) => `$${Number(v.toFixed(1))}M`;

/** T-1839: a row's hover is its price worked out: the base price, one line per factor that is not 1,
 * then the total, as in the trip inspector's "label × weight" lines. */
function rowTip(r: BlueprintItem) {
  const level = r.level as Level, n = r.quantity;
  const factors: [string, string, number][] = []; // [name, shown amount, multiplier]
  const lenKm = (km: number) => [`${num(km / perDistance(1))} ${unitShort()}`, km / perDistance(1)] as const;
  let base: string;
  const water = r.wet ? WATER : 1;
  const wet = () => { if (r.wet) factors.push(["Over water", "", WATER]); };
  const lvl = (l: Level) => { if (MULT[l] !== 1) factors.push([levelName(l), "", MULT[l]]); };
  if (r.kind === 0) {
    const [shown] = lenKm(n / 1000);
    base = `Base cost ${mil(perDistance(BASE_M_PER_KM))} per ${distUnit()}`;
    factors.push(["Length", shown, 1]);
    // T-1890: terrain and water add to the level's multiplier: "Ground level 0.5", "Mountains + 2.5",
    // "Cost factor × 3"
    const adds: [string, number][] = [];
    if (r.ground > 0) adds.push([TERRAIN_NAMES[r.ground] ?? "Rough ground", TERRAIN_SURCHARGES[r.ground] ?? 0]);
    if (r.wet) adds.push(["Over water", WATER_SURCHARGE]);
    if (adds.length) {
      factors.push([levelName(level), "=", MULT[level]], ...adds.map(([a, m]): [string, string, number] => [a, "+", m]));
      factors.push(["Cost factor", "", adds.reduce((s, [, m]) => s + m, MULT[level])]);
    } else lvl(level);
    if (r.tracks === 1) factors.push(["Single track", "", SINGLE_TRACK]);
    if (r.tracks === HSR_ROW) factors.push(["High-speed track", "", HSR_TRACK]);
  } else if (r.kind === 1) {
    const b = STATION_FIXED_M + STATION_M[level];
    base = `Base cost ${mil(b)} per station at ${levelName(level).toLowerCase()}`;
    if (n !== 1) factors.push(["Stations", String(n), 1]);
    const fixed = n * b * (r.tracks === 1 ? SINGLE_STATION : 1) * water;
    const p = r.cost / 1e6 / fixed;
    if (Math.abs(p - 1) > 0.005) factors.push(["Platform length", "", p]);
    if (r.tracks === 1) factors.push(["Single track", "", SINGLE_STATION]);
    wet();
  } else if (r.kind === 2 || r.kind === 3 || r.kind === 4) {
    const km = r.kind === 2 ? 0.25 : r.kind === 3 ? 0.6 : 0.1;
    const l = r.kind === 3 ? awayLevel(level) : level;
    base = `Base cost ${mil(perDistance(BASE_M_PER_KM))} per ${distUnit()}`;
    factors.push([r.kind === 2 ? "Junction track" : r.kind === 3 ? "Flyover track" : "Crossing track", `${num(n * km / perDistance(1))} ${unitShort()}`, 1]);
    lvl(l);
    wet();
    const product = BASE_M_PER_KM * n * km * MULT[l] * water;
    if (Math.abs(r.cost / 1e6 / product - 1) > 0.005) factors.push(["Crossover track", "", r.cost / 1e6 / product]);
  } else if (r.kind === 10) {
    base = `Base cost ${mil(perDistance(BUS_ROUTE_M_PER_KM))} per ${distUnit()}`;
    factors.push(["Length", `${num(n / 1000 / perDistance(1))} ${unitShort()}`, 1]);
  } else {
    // cars, buses, upgrades and passages: the average price of one, times how many
    base = `Base cost ${money(r.cost / Math.max(1, n))} per ${NOUN[r.kind] || "item"}`;
    if (n !== 1) factors.push([plural(n, NOUN[r.kind] || "item").replace(/^\S+ /, "").replace(/^./, (c) => c.toUpperCase()), String(n), 1]);
  }
  const lines = [base, ...factors.map(([name, shown, m]) => (shown === "+" ? `${name} + ${num(m)}` : shown === "=" ? `${name} ${num(m)}` : shown && m === 1 ? `${name} × ${shown}` : `${name} × ${num(m)}`))];
  lines.push(`= ${money(r.cost)}`);
  return lines.join("\n");
}

/** Build's quote and undo controls. Every item is priced by the clock worker's track model. */
export function BlueprintControls() {
  const w = world.value;
  const items = useBlueprintQuote();
  const total = items.reduce((sum, r) => sum + r.cost, 0);
  const m = moneyState.value;
  // T-867: what the blueprint will cost a day once constructed, the pieces switched off left out
  let upkeep = 0;
  if (w && m && total > 0) {
    for (const e of w.edges) if (!e.built && !e.off) upkeep += edgeUpkeep({ ...e, built: true }, m);
    for (const s of w.save.stations) if (!s.built && !s.off) upkeep += stationUpkeep(m);
    for (const st of Object.values(w.lineStats)) if (st.bus && !st.bus.built) upkeep += (st.bus.roadM / 1000) * m.upkeepBusKm * m.economy;
  }
  return (
    <div class="blueprint-controls">
      <div class="blueprint-head">
        <span class="b">Blueprint</span>
        <span class="bp-total">
          <b class={"num" + (m?.mode !== "sandbox" && total > liveMoney.value.cash * 1e6 ? " bad" : "")} data-tip="Total construction cost">{money(total)}</b>
          <OffCost items={items} />
        </span>
        <div class="group">
          <button class="btn icon" data-tip="Undo (Ctrl+Z)" aria-label="Undo" disabled={!w?.canUndo} onClick={() => void undo()}><Icon.undo /></button>
          <button class="btn icon" data-tip="Redo (Ctrl+Y)" aria-label="Redo" disabled={!w?.canRedo} onClick={() => void redo()}><Icon.redo /></button>
        </div>
      </div>
      {m && upkeep > 0 && (
        <p class="small muted blueprint-upkeep" data-tip={upkeepText(m, "Once constructed, upkeep")}>
          {money(upkeep)} / day upkeep
        </p>
      )}
      <button class="btn text blueprint-construct" data-guide="3" data-tip="Construct every blueprint piece (C)" disabled={total <= 0} onClick={() => void constructBlueprints()}>Construct blueprints<span class="key muted">C</span></button>
    </div>
  );
}

const pieces = (e: number, st: number) => [e ? `${e} track` : "", st ? plural(st, "station") : ""].filter(Boolean).join(", ");

/** T-1296: what the disabled blueprint would cost, beside the total: everything unbuilt
 * (`blueprintCost`, disabled pieces in it) less the quote's rows other than new cars. */
function OffCost({ items }: { items: BlueprintItem[] }) {
  const w = world.value, off = offTotals.value;
  if (!w || !(off.edges + off.stations)) return null;
  const on = items.reduce((sum, r) => sum + (r.kind === 5 || r.kind === 11 ? 0 : r.cost), 0);
  const cost = w.money.blueprintCost - on;
  if (!(cost >= 0.5e6)) return null;
  return <span class="small muted num off-cost" onDblClick={() => void enableAll()} data-tip={`Disabled blueprint: ${pieces(off.edges, off.stations)}, ${moneyCash(cost)}, left out of the total\nDouble click to enable them all`}>{money(cost)} disabled</span>;
}

/** How many blueprint pieces are switched off (T-333), whatever the tool, since Construct and the
 * total leave them out. Selecting, disabling and enabling are in the map's BlueprintBar (T-353). */
export function BlueprintPick() {
  const off = offTotals.value;
  const offN = off.edges + off.stations;
  if (!offN) return null;
  return (
    <div class="blueprint-pick">
      <p class="small muted" data-tip={`${pieces(off.edges, off.stations)} will not be constructed`}>{offN} disabled</p>
    </div>
  );
}

/** T-1680: Double up for every run of double track in the pick; the hover gives the summed price,
 * or why not. */
function PickDoubleUp() {
  const p = bpPicked.value, w = world.value;
  const pieces = useMemo(() => doubleUpPieces(p), [p, w]);
  const key = pieces.join(",");
  const answer = useLatestAnswer(pieces.length ? async () => {
    let price = 0, why = "";
    for (const edge of pieces) {
      const d = await queryClock({ what: "doubleUp", edge });
      if (!d) return null;
      if (d[0] >= 0) price += d[0];
      else why ||= describeIssues(String.fromCharCode(...d.subarray(1)).split("\n").filter(Boolean));
    }
    return { price, why };
  } : null, [key, w?.version]);
  if (!pieces.length || !answer) return null;
  const runs = pieces.length === 1 ? "" : ` on ${pieces.length} runs`;
  const tip = answer.price > 0 ? `Double up to quad track${runs} for ${money(answer.price * 1e6)} (blueprint)\nEach run goes to the next junction or track end${answer.why ? `\nSome cannot: ${answer.why}` : ""}` : answer.why;
  return (
    <button class="btn icon" disabled={!(answer.price > 0)} data-tip={tip} aria-label="Double up" onClick={() => void doubleUpPicked(pieces)}>
      <Icon.doubleUp />
    </button>
  );
}

/** T-353, T-815, T-829: while the Select tool has a selection, a bar at the map's top centre, the
 * selection's inspector: how many pieces and how much track, then for blueprint the elevation
 * buttons, Disable and Enable; Remove and Clear for either kind. Appears and goes in one step. */
export function BlueprintBar() {
  const s = pickedState.value;
  const n = s.edges + s.stations;
  const [levels, setLevels] = useState(false);
  useEffect(() => { if (n === 0) setLevels(false); }, [n]);
  if (tool.value !== "blueprint" || n === 0) return null;
  const piece = (s.built ? "Built: " : "Blueprint: ") + pieces(s.edges, s.stations);
  const levelTip = (l: Level) => `Put the selected track at ${levelName(l).toLowerCase()} (${levelLabel(l)})`;
  const canLevel = !s.built && s.edges > 0;
  return (
    <>
      <div class="blueprint-bar" role="group" aria-label="Selection">
        <span class="count" data-tip={`Selected: ${piece}`}><span class="bp-ico"><Icon.marquee /></span>{n}<span class="bp-label"> selected</span></span>
        {s.edges > 0 && <span class="num" data-tip="Length of the selected track">{distance(s.lengthM)}</span>}
        {canLevel && <div class="bp-levels"><LevelButtons value={s.level} tip={levelTip} onPick={(l) => void levelPicked(l)} /></div>}
        {canLevel && (
          <button class={"btn bp-level" + (s.level === null ? "" : " lv on")} style={s.level === null ? undefined : { "--lc": LEVEL_COLOURS[s.level], "--lk": inkOn(LEVEL_COLOURS[s.level]) }}
            data-tip={s.level === null ? "Elevation of the selected track: mixed" : `Elevation ${levelName(s.level)}`} aria-label="Elevation" aria-expanded={levels}
            onClick={() => setLevels(!levels)}>{s.level === null ? <Icon.autoLevel /> : levelLabel(s.level)}</button>
        )}
      {!s.built && (
        <div class="group">
          <button class="btn text" data-tip="Leave these out when constructing (D)" disabled={!s.on} onClick={() => void switchPicked(true)}><Icon.switchOff /><span class="bp-label">Disable</span></button>
          <button class="btn text" data-tip="Enable the selection (E)" disabled={!s.off} onClick={() => void switchPicked(false)}><Icon.switchOn /><span class="bp-label">Enable</span></button>
        </div>
      )}
      <PickDoubleUp />
      <button class="btn text" data-tip="Remove the selection (Delete)" onClick={() => void removePicked()}><Icon.bin /><span class="bp-label">Remove</span></button>
      <button class="btn text" data-tip="Clear the selection (Esc)" onClick={() => clearBpPicked()}><Icon.cross /><span class="bp-label">Clear</span></button>
      </div>
      {canLevel && levels && (
        <div class="blueprint-levels touch-levels">
          <LevelButtons value={s.level} tip={levelTip} onPick={(l) => { void levelPicked(l); setLevels(false); }} />
        </div>
      )}
    </>
  );
}

export function BlueprintBreakdown() {
  const items = useBlueprintQuote();
  return (
    <div class="blueprint-panel">
      {items.length > 0 ? (
        <details class="blueprint-breakdown" open>
          <summary>Blueprint cost breakdown</summary>
          {KINDS.map((kind, k) => {
            const rows = items.filter((r) => r.kind === k);
            if (!rows.length) return null;
            return (
              <div class="blueprint-category">
                <div class="small b">{kind}</div>
                <ul class="rows blueprint-items">
                  {rows.map((r) => (
                    <li>
                      <span class="what" data-tip={rowTip(r)}>{rowName(r)}</span>
                      <span class="val" data-tip={rowTip(r)}>{money(r.cost)}</span>
                    </li>
                  ))}
                </ul>
              </div>
            );
          })}
        </details>
      ) : <p class="small muted">No blueprint to construct.</p>}
    </div>
  );
}
