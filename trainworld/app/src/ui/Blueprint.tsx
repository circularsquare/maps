import { useEffect, useMemo, useState } from "preact/hooks";
import { describeIssues } from "../game/issues";
import { useLatestAnswer } from "./useClockQuery";
import { queryClock } from "../workers/clockClient";
import { inkOn, LEVEL_COLOURS } from "../game/palette";
import type { Level } from "../game/types";
import { distUnit, distance, levelLabel, levelName, money, moneyCash, perDistance, plural, unitShort, upkeepText } from "../game/format";
import { edgeUpkeep, stationUpkeep } from "../game/regionMoney";
import { bpPicked, clearBpPicked, doubleUpPicked, doubleUpPieces, enableAll, hsrPicked, levelPicked, offTotals, pickedState, removePicked, switchPicked } from "../game/blueprintPick";
import { liveMoney, moneyState } from "../game/money";
import { tool } from "../game/ui";
import type { BlueprintItem } from "../game/types";
import { constructBlueprints } from "../game/lineActions";
import { world } from "../game/world";
import { useBlueprintPieces, useBlueprintQuote } from "../game/quotes";
import { blueprintParts, bpHover, switchRow, type BpPart } from "../game/blueprintStretches";
import { locateNodes } from "./locate";
import { redo, undo } from "../game/fareUndo";
import { Icon, LevelButtons } from "./widgets";
import { TERRAIN_NAMES, TERRAIN_SURCHARGES, WATER_SURCHARGE } from "../game/terrain";

// 6 T-1270's terrain, 7 T-1929's roads crossed at ground level, 8 and 9 T-1281's, 10 and 11 T-1464's
// bus routes and buses
const KINDS = ["Track", "Stations", "Junctions", "Flyovers", "Flat crossings", "Trains", "Terrain", "Road crossings", "Station upgrades", "Passages", "Bus routes", "Buses"];
/** a quote row's track count for high-speed track (sim `HSR_ROW`, T-817) */
const HSR_ROW = 3;

const NOUN = ["track", "station", "junction", "flyover", "flat crossing", "car", "ground", "road crossing", "station upgrade", "passage", "route", "bus", "port", "boat", "ferry route"];
const levelWords = (r: BlueprintItem) => `${levelName(r.level).toLowerCase()} (${levelLabel(r.level)})`;

/** T-1839: a quote row as it reads in the list, "3.7 km viaduct (+2)"; the price's factors are in `rowTip`. */
function rowName(r: BlueprintItem) {
  if (r.kind === 5) return plural(r.quantity, "new car");
  if (r.kind === 10) return distance(r.quantity);
  if (r.kind === 11) return plural(r.quantity, "new bus", "new buses");
  // T-2193: ports, boats and a ferry route's metres (free)
  if (r.kind === 12) return plural(r.quantity, "port");
  if (r.kind === 13) return plural(r.quantity, "new boat");
  if (r.kind === 14) return `${distance(r.quantity)} ferry route`;
  if (r.kind === 9) return plural(r.quantity, "passage");
  if (r.kind === 7) return plural(r.quantity, "road crossing");
  if (r.kind === 0) return `${distance(r.quantity)} ${levelWords(r)}`;
  return `${plural(r.quantity, NOUN[r.kind] || "item")}, ${levelWords(r)}`;
}

// A display copy of the price list (sim/src/track/params.rs and cost.rs own the real one), so a row's
// hover can show its price as the sum it is.
const BASE_M_PER_KM = 75;
const MULT: Record<Level, number> = { 3: 2.0, 2: 1.5, 1: 1.0, 0: 0.5, [-1]: 1.0, [-2]: 1.6, [-3]: 2.0 };
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
  } else if (r.kind === 14) {
    base = "Ferry routes are free";
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
    // T-2194: a port's upkeep is its own, a fifth of a station's
    for (const s of w.save.stations) if (!s.built && !s.off) upkeep += s.port ? m.upkeepPort * m.economy : stationUpkeep(m);
    for (const st of Object.values(w.lineStats)) if (st.bus && !st.bus.built && !st.bus.ferry) upkeep += (st.bus.roadM / 1000) * m.upkeepBusKm * m.economy;
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
        <p class="small money-neg blueprint-upkeep" data-tip={upkeepText(m, "Once constructed, upkeep")}>
          {money(upkeep)} / day upkeep
        </p>
      )}
      <BlueprintBreakdown items={items} />
      <button class="btn text blueprint-construct" data-guide="construct" data-tip="Construct every blueprint piece (C)" disabled={total <= 0} onClick={() => void constructBlueprints()}>Construct blueprints<span class="key muted">C</span></button>
    </div>
  );
}

const pieces = (e: number, st: number) => [e ? `${e} track` : "", st ? plural(st, "station") : ""].filter(Boolean).join(", ");

/** T-1296: what the disabled blueprint would cost, beside the total: everything unbuilt
 * (`blueprintCost`, disabled pieces in it) less the quote's rows other than new cars. */
function OffCost({ items }: { items: BlueprintItem[] }) {
  const w = world.value, off = offTotals.value;
  if (!w || !(off.edges + off.stations)) return null;
  const on = items.reduce((sum, r) => sum + (r.kind === 5 || r.kind === 11 || r.kind === 13 ? 0 : r.cost), 0);
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

/** T-353, T-815, T-829: while the Select tool has a selection (or a blueprint piece is clicked, T-2283), a bar at the map's top centre, the
 * selection's inspector: how many pieces and how much track, then for blueprint the elevation
 * buttons, regular or high-speed track (T-1898, for its double track), Disable and Enable; Remove
 * and Clear for either kind. Appears and goes in one step. */
export function BlueprintBar() {
  const s = pickedState.value;
  const n = s.edges + s.stations;
  const [levels, setLevels] = useState(false);
  useEffect(() => { if (n === 0) setLevels(false); }, [n]);
  // T-2283: the plain select tool's pick is the blueprint track clicked (game/blueprintPick.ts)
  if ((tool.value !== "blueprint" && tool.value !== "select") || n === 0) return null;
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
            data-tip={s.level === null ? "Level of the selected track: mixed" : levelName(s.level)} aria-label="Level" aria-expanded={levels}
            onClick={() => setLevels(!levels)}>{s.level === null ? <Icon.autoLevel /> : levelLabel(s.level)}</button>
        )}
      {!s.built && s.hsr + s.regular > 0 && (
        <div class="group" role="group" aria-label="Track type">
          <button class={"btn" + (s.hsr === 0 ? " on" : "")} onClick={() => void hsrPicked(false)}>Regular</button>
          <button class={"btn" + (s.regular === 0 ? " on" : "")} data-tip={`Costs ${HSR_TRACK}x as much as regular track`} onClick={() => void hsrPicked(true)}>High speed</button>
        </div>
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

/** The quote's rows of the kinds in `only`, under their kinds' headings, each with its price worked out. */
function KindRows({ items, only }: { items: BlueprintItem[]; only?: Set<number> }) {
  return (
    <>
      {KINDS.map((kind, k) => {
        const rows = items.filter((r) => r.kind === k && (!only || only.has(k)));
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
    </>
  );
}

/** the quote's kinds that are no piece of track or station: trains, bus routes and buses, ports, boats, ferry routes */
const NOT_TRACK = new Set([5, 10, 11, 12, 13, 14]);

const levelsText = (r: { levelMin: Level; levelMax: Level }) =>
  r.levelMin === r.levelMax ? `${levelName(r.levelMin)} (${levelLabel(r.levelMin)})` : `Levels ${levelLabel(r.levelMin)} to ${levelLabel(r.levelMax)}`;

/** A row's or a part's off switch, a checkbox ticked while any of its pieces is on. */
function OffSwitch({ on, what, flip }: { on: boolean; what: string; flip: () => void }) {
  const tip = `${on ? "Disable" : "Enable"} ${what}`;
  return (
    <span class="check bp-switch" onClick={(e) => e.stopPropagation()}>
      <input type="checkbox" checked={on} aria-label={tip} data-tip={tip} onChange={() => flip()} />
    </span>
  );
}

const hover = (edges: number[]) => ({ onPointerEnter: () => (bpHover.value = edges), onPointerLeave: () => (bpHover.value = []) });

/** T-2269: the blueprint by part joined by track, each part's stations and stretches in track order;
 * a click puts the camera on it, the pointer over it highlights it on the map. Then trains, buses
 * and ferries, and the whole quote by kind. Collapsed until opened. */
function BlueprintBreakdown({ items }: { items: BlueprintItem[] }) {
  const [open, setOpen] = useState(false);
  const pieces = useBlueprintPieces(open);
  const w = world.value;
  const parts = useMemo(() => (open && w && pieces ? blueprintParts(w, pieces) : []), [open, w, pieces]);
  // T-2339: each part one line until its chevron opens it, kept by its first piece across edits
  const [unfolded, setUnfolded] = useState<ReadonlySet<string>>(new Set());
  const partKey = (p: BpPart) => (p.edges.length ? `e${Math.min(...p.edges)}` : `s${p.stations[0]}`);
  const fold = (k: string) => setUnfolded((was) => { const s = new Set(was); s.has(k) ? s.delete(k) : s.add(k); return s; });
  useEffect(() => () => { bpHover.value = []; }, []);
  // the switched-off blueprint counts, so a blueprint all switched off can be switched back on here
  if (!items.length && !(w && w.money.blueprintCost > 0)) return null;
  return (
    <details class="blueprint-breakdown" open={open} onToggle={(e) => setOpen((e.currentTarget as HTMLDetailsElement).open)}>
      <summary>Cost breakdown</summary>
      <div class="bp-breakdown-list">
        {parts.map((p) => {
          const k = partKey(p), many = p.rows.length > 1 || p.junctions > 0, isOpen = many && unfolded.has(k);
          return (
          <div class="bp-part">
            <ul class="rows pick blueprint-items">
              <li class={"bp-part-head" + (p.on ? "" : " bp-off")} {...hover(p.edges)} onClick={() => locateNodes(p.nodes)}
                data-tip={[p.stationCount ? plural(p.stationCount, "station") : "", p.lengthM ? distance(p.lengthM) : ""].filter(Boolean).join(", ")}>
                {many ? (
                  <button class={"btn icon mini chevron" + (isOpen ? " open" : "")} aria-label={isOpen ? "Fold" : "Unfold"} aria-expanded={isOpen}
                    onClick={(e) => { e.stopPropagation(); fold(k); }}><Icon.chevron /></button>
                ) : <span class="bp-fold-gap" />}
                <OffSwitch on={p.on} what="all joined track" flip={() => void switchRow(p)} />
                <span class="grow">{p.label}</span>
                <span class="val">{money(p.on ? p.onCost : p.cost)}</span>
              </li>
              {isOpen && p.rows.map((r) => (
                <li class={"bp-sub" + (r.on ? "" : " bp-off")} {...hover(r.edges)} onClick={() => locateNodes(r.nodes)}
                  data-tip={r.kind === "stretch" ? [levelsText(r), r.crossings ? `${plural(r.crossings, "flat crossing")}, ${money(r.crossingCost)}` : ""].filter(Boolean).join("\n") : undefined}>
                  <OffSwitch on={r.on} what={r.kind === "stretch" ? "this stretch" : "this station"} flip={() => void switchRow(r)} />
                  <span class="grow">{r.label}</span>
                  {r.kind === "stretch" && <span class="val muted">{distance(r.lengthM)}</span>}
                  <span class="val">{money(r.on ? r.onCost : r.cost)}</span>
                </li>
              ))}
              {isOpen && p.junctions > 0 && (
                <li class={"bp-sub bp-fixed" + (p.junctionOnCost > 0 || !p.junctionCost ? "" : " bp-off")} {...hover(p.edges)}>
                  <span class="grow">{plural(p.junctions, "junction")}</span>
                  <span class="val">{money(p.junctionOnCost > 0 ? p.junctionOnCost : p.junctionCost)}</span>
                </li>
              )}
            </ul>
          </div>
          );
        })}
        <KindRows items={items} only={NOT_TRACK} />
        {items.length > 0 && (
          <details class="bp-by-kind">
            <summary>By kind</summary>
            <KindRows items={items} />
          </details>
        )}
      </div>
    </details>
  );
}
