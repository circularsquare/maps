import { useEffect, useMemo, useState } from "preact/hooks";
import { lowerWord, t, tNumber, tSeries, type MessageKey } from "../i18n";
import { describeIssues } from "../game/issues";
import { useLatestAnswer } from "./useClockQuery";
import { queryClock } from "../workers/clockClient";
import { inkOn, LEVEL_COLOURS } from "../game/palette";
import type { Level } from "../game/types";
import { distUnit, distance, levelLabel, levelName, millions, money, moneyCash, perDistance, unitShort, upkeepText } from "../game/format";
import { edgeUpkeep, stationUpkeepOf } from "../game/regionMoney";
import { bpPicked, clearBpPicked, doubleUpPicked, doubleUpPieces, enableAll, hsrPicked, levelPicked, offTotals, pickedState, removePicked, switchPicked } from "../game/blueprintPick";
import { rebuildKind, rebuildLevel, rebuildPicked, rebuildQuote, rebuildTip } from "../game/rebuild";
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
import { SWITCH_PICK_KEY } from "./toolKeys";
import { TERRAIN_NAMES, TERRAIN_SURCHARGES, WATER_SURCHARGE } from "../game/terrain";
import { kindLevelMult, TRACK_SPECS } from "../game/trackKinds";

// 6 T-1270's terrain, 7 T-1929's roads crossed at ground level, 8 and 9 T-1281's, 10 and 11 T-1464's
// bus routes and buses
const KINDS: MessageKey[] = ["blueprint.kindTrack", "blueprint.kindStations", "blueprint.kindJunctions", "blueprint.kindFlyovers", "blueprint.kindFlatCrossings", "blueprint.kindTrains", "blueprint.kindTerrain", "blueprint.kindRoadCrossings", "blueprint.kindStationUpgrades", "blueprint.kindPassages", "blueprint.kindBusRoutes", "blueprint.kindBuses"];
/** a quote row's track count for high-speed track (sim `HSR_ROW`, T-817) */
const HSR_ROW = 3;
/** and for tram track (sim `TRAM_ROW`, T-2926), priced by the tram table */
const TRAM_ROW = 4;
/** and for street tram track (sim `STREET_TRAM_ROW`, T-2927) */
const STREET_TRAM_ROW = 5;
/** the quote row of tram cars (sim `blueprint_cost_items`; 14 is a ferry route) */
const TRAM_CARS = 15;

const NOUN = ["track", "station", "junction", "flyover", "flatCrossing", "car", "ground", "roadCrossing", "stationUpgrade", "passage", "route", "bus", "port", "boat", "ferryRoute", "tram car"];
/** "3 stations", "2 new cars": a counted noun by its id, the plural chosen in the message */
const counted = (noun: string, count: number) => t("blueprint.count", { noun, count });
const levelWords = (r: BlueprintItem) => `${lowerWord(levelName(r.level))} (${levelLabel(r.level)})`;

/** T-1839: a quote row as it reads in the list, "3.7 km viaduct (+2)"; the price's factors are in `rowTip`. */
function rowName(r: BlueprintItem) {
  if (r.kind === 5) return counted("newCar", r.quantity);
  if (r.kind === 10) return distance(r.quantity);
  if (r.kind === 11) return counted("newBus", r.quantity);
  // T-2193: ports, boats and a ferry route's metres (free)
  if (r.kind === 12) return counted("port", r.quantity);
  if (r.kind === 13) return counted("newBoat", r.quantity);
  if (r.kind === 14) return t("blueprint.ferryRoute", { distance: distance(r.quantity) });
  if (r.kind === TRAM_CARS) return t("blueprint.newTramCars", { count: r.quantity });
  if (r.kind === 9) return counted("passage", r.quantity);
  if (r.kind === 7) return counted("roadCrossing", r.quantity);
  if (r.kind === 0 && r.tracks === TRAM_ROW) return t("blueprint.lengthLevelTram", { distance: distance(r.quantity), level: levelWords(r) });
  if (r.kind === 0 && r.tracks === STREET_TRAM_ROW) return t("blueprint.lengthLevelStreetTram", { distance: distance(r.quantity), level: levelWords(r) });
  if (r.kind === 0) return t("blueprint.lengthLevel", { distance: distance(r.quantity), level: levelWords(r) });
  return t("blueprint.withLevel", { what: counted(NOUN[r.kind] || "item", r.quantity), level: levelWords(r) });
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
const num = (v: number) => tNumber(v, { maximumFractionDigits: 2 });
const mil = millions;
/** "3.7 km" */
const withUnit = (value: number) => t("blueprint.amountUnit", { value: num(value), unit: unitShort() });

/** T-1839: a row's hover is its price worked out: the base price, one line per factor that is not 1,
 * then the total, as in the trip inspector's "label × weight" lines. */
function rowTip(r: BlueprintItem) {
  const level = r.level as Level, n = r.quantity;
  const factors: [string, string, number][] = []; // [name, shown amount, multiplier]
  const lenKm = (km: number) => [withUnit(km / perDistance(1)), km / perDistance(1)] as const;
  let base: string;
  const water = r.wet ? WATER : 1;
  const wet = () => { if (r.wet) factors.push([t("blueprint.overWater"), "", WATER]); };
  const mult = (l: Level) => (r.kind === 0 && r.tracks === TRAM_ROW ? kindLevelMult("tram", l) : r.kind === 0 && r.tracks === STREET_TRAM_ROW ? kindLevelMult("streetTram", l) : MULT[l]);
  const lvl = (l: Level) => { if (mult(l) !== 1) factors.push([levelName(l), "", mult(l)]); };
  if (r.kind === 0) {
    const [shown] = lenKm(n / 1000);
    base = t("blueprint.baseCostDist", { price: mil(perDistance(BASE_M_PER_KM)), unit: distUnit() });
    factors.push([t("blueprint.length"), shown, 1]);
    // T-1890: terrain and water add to the level's multiplier: "Ground level 0.5", "Mountains + 2.5",
    // "Cost factor × 3"
    // (tram track pays a share of each, T-2928, T-3004)
    const adds: [string, number][] = [];
    const spec = r.tracks === STREET_TRAM_ROW ? TRACK_SPECS.streetTram : r.tracks === TRAM_ROW ? TRACK_SPECS.tram : TRACK_SPECS.regular;
    if (r.ground > 0) adds.push([TERRAIN_NAMES[r.ground] ?? t("blueprint.roughGround"), (TERRAIN_SURCHARGES[r.ground] ?? 0) * spec.terrain]);
    if (r.wet) adds.push([t("blueprint.overWater"), WATER_SURCHARGE * spec.water]);
    if (adds.length) {
      factors.push([levelName(level), "=", mult(level)], ...adds.map(([a, m]): [string, string, number] => [a, "+", m]));
      factors.push([t("blueprint.costFactor"), "", adds.reduce((s, [, m]) => s + m, mult(level))]);
    } else lvl(level);
    if (r.tracks === 1) factors.push([t("blueprint.singleTrack"), "", SINGLE_TRACK]);
    if (r.tracks === HSR_ROW) factors.push([t("blueprint.hsrTrack"), "", HSR_TRACK]);
  } else if (r.kind === 1) {
    const b = STATION_FIXED_M + STATION_M[level];
    base = t("blueprint.baseCostStation", { price: mil(b), level: lowerWord(levelName(level)) });
    if (n !== 1) factors.push([t("blueprint.stationsFactor"), String(n), 1]);
    const fixed = n * b * (r.tracks === 1 ? SINGLE_STATION : 1) * water;
    const p = r.cost / 1e6 / fixed;
    if (Math.abs(p - 1) > 0.005) factors.push([t("blueprint.platformLength"), "", p]);
    if (r.tracks === 1) factors.push([t("blueprint.singleTrack"), "", SINGLE_STATION]);
    wet();
  } else if (r.kind === 2 || r.kind === 3 || r.kind === 4) {
    const km = r.kind === 2 ? 0.25 : r.kind === 3 ? 0.6 : 0.1;
    const l = r.kind === 3 ? awayLevel(level) : level;
    base = t("blueprint.baseCostDist", { price: mil(perDistance(BASE_M_PER_KM)), unit: distUnit() });
    factors.push([r.kind === 2 ? t("blueprint.junctionTrack") : r.kind === 3 ? t("blueprint.flyoverTrack") : t("blueprint.crossingTrack"), withUnit(n * km / perDistance(1)), 1]);
    lvl(l);
    wet();
    const product = BASE_M_PER_KM * n * km * MULT[l] * water;
    if (Math.abs(r.cost / 1e6 / product - 1) > 0.005) factors.push([t("blueprint.crossoverTrack"), "", r.cost / 1e6 / product]);
  } else if (r.kind === 10) {
    base = t("blueprint.baseCostDist", { price: mil(perDistance(BUS_ROUTE_M_PER_KM)), unit: distUnit() });
    factors.push([t("blueprint.length"), withUnit(n / 1000 / perDistance(1)), 1]);
  } else if (r.kind === 14) {
    base = t("blueprint.ferryFree");
  } else {
    // cars, buses, upgrades and passages: the average price of one, times how many
    base = t("blueprint.baseCostPer", { price: money(r.cost / Math.max(1, n)), noun: NOUN[r.kind] || "item" });
    if (n !== 1) factors.push([t("blueprint.factorName", { noun: NOUN[r.kind] || "item" }), String(n), 1]);
  }
  const lines = [base, ...factors.map(([name, shown, m]) => (shown === "+" ? t("blueprint.factorAdd", { name, amount: num(m) }) : shown === "=" ? t("blueprint.factorSet", { name, amount: num(m) }) : shown && m === 1 ? t("blueprint.factorTimes", { name, amount: shown }) : t("blueprint.factorTimes", { name, amount: num(m) })))];
  lines.push(t("blueprint.sum", { money: money(r.cost) }));
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
    for (const s of w.save.stations) if (!s.built && !s.off) upkeep += s.port ? m.upkeepPort * m.economy : stationUpkeepOf(m, s);
    for (const st of Object.values(w.lineStats)) if (st.bus && !st.bus.built && !st.bus.ferry) upkeep += (st.bus.roadM / 1000) * m.upkeepBusKm * m.economy;
  }
  return (
    <div class="blueprint-controls">
      <div class="blueprint-head">
        <span class="b">{t("blueprint.title")}</span>
        <span class="bp-total">
          <b class={"num" + (m?.mode !== "sandbox" && total > liveMoney.value.cash * 1e6 ? " bad" : "")} data-tip={t("blueprint.totalTip")}>{money(total)}</b>
          <OffCost items={items} />
        </span>
        <div class="group">
          <button class="btn icon" data-tip={t("blueprint.undoTip")} aria-label={t("blueprint.undo")} disabled={!w?.canUndo} onClick={() => void undo()}><Icon.undo /></button>
          <button class="btn icon" data-tip={t("blueprint.redoTip")} aria-label={t("blueprint.redo")} disabled={!w?.canRedo} onClick={() => void redo()}><Icon.redo /></button>
        </div>
      </div>
      {m && upkeep > 0 && (
        <p class="small money-neg blueprint-upkeep" data-tip={upkeepText(m, t("blueprint.upkeepLead"))}>
          {t("blueprint.upkeepDay", { money: money(upkeep) })}
        </p>
      )}
      <BlueprintBreakdown items={items} />
      <button class="btn text blueprint-construct" data-guide="construct" data-tip={t("blueprint.constructTip", { key: "C" })} disabled={total <= 0} onClick={() => void constructBlueprints()}>{t("blueprint.construct")}<span class="key muted">C</span></button>
    </div>
  );
}

const pieces = (e: number, st: number) => tSeries([e ? t("blueprint.trackPieces", { count: e }) : "", st ? counted("station", st) : ""].filter(Boolean));

/** T-1296: what the disabled blueprint would cost, beside the total: everything unbuilt
 * (`blueprintCost`, disabled pieces in it) less the quote's rows other than new cars. */
function OffCost({ items }: { items: BlueprintItem[] }) {
  const w = world.value, off = offTotals.value;
  if (!w || !(off.edges + off.stations)) return null;
  const on = items.reduce((sum, r) => sum + (r.kind === 5 || r.kind === 11 || r.kind === 13 || r.kind === TRAM_CARS ? 0 : r.cost), 0);
  const cost = w.money.blueprintCost - on;
  if (!(cost >= 0.5e6)) return null;
  return <span class="small muted num off-cost" onDblClick={() => void enableAll()} data-tip={t("blueprint.offTip", { pieces: pieces(off.edges, off.stations), cost: moneyCash(cost) })}>{t("blueprint.offCost", { money: money(cost) })}</span>;
}

/** How many blueprint pieces are switched off (T-333), whatever the tool, since Construct and the
 * total leave them out. Selecting, disabling and enabling are in the map's BlueprintBar (T-353). */
export function BlueprintPick() {
  const off = offTotals.value;
  const offN = off.edges + off.stations;
  if (!offN) return null;
  return (
    <div class="blueprint-pick">
      <p class="small muted" data-tip={t("blueprint.offPiecesTip", { pieces: pieces(off.edges, off.stations) })}>{t("blueprint.offCount", { count: tNumber(offN) })}</p>
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
  const tip = answer.price > 0 ? [t("blueprint.doubleUp", { runs: pieces.length, price: money(answer.price * 1e6) }), t("blueprint.doubleUpRuns"), ...(answer.why ? [t("blueprint.doubleUpSome", { why: answer.why })] : [])].join("\n") : answer.why;
  return (
    <button class="btn icon" disabled={!(answer.price > 0)} data-tip={tip} aria-label={t("blueprint.doubleUpLabel")} onClick={() => void doubleUpPicked(pieces)}>
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
  const piece = t("blueprint.piece", { built: s.built, pieces: pieces(s.edges, s.stations) });
  const levelTip = (l: Level) => t("blueprint.levelTip", { level: lowerWord(levelName(l)), label: levelLabel(l) });
  const canLevel = s.edges > 0;
  // T-3059: a built pick only chooses; the Rebuild button spends (game/rebuild.ts)
  const chosenLevel = s.built ? rebuildLevel.value ?? s.level : s.level;
  const pickLevel = (l: Level) => { if (s.built) rebuildLevel.value = l; else void levelPicked(l); };
  const q = rebuildQuote.value;
  const hsrOn = s.built && rebuildKind.value ? rebuildKind.value === "hsr" : s.regular === 0;
  const regularOn = s.built && rebuildKind.value ? rebuildKind.value === "regular" : s.hsr === 0;
  const pickType = (hsr: boolean) => { if (s.built) rebuildKind.value = hsr ? "hsr" : "regular"; else void hsrPicked(hsr); };
  // T-2397: E and Delete act on the pick with the Select tool only (map/blueprintBox.ts); Esc in both
  const keys = tool.value === "blueprint";
  const switchKey = SWITCH_PICK_KEY.toUpperCase();
  return (
    <>
      <div class="blueprint-bar" role="group" aria-label={t("blueprint.selection")}>
        <span class="count" data-tip={t("blueprint.selectedTip", { piece })}><span class="bp-ico"><Icon.marquee /></span><span class="bp-short">{tNumber(n)}</span><span class="bp-label">{t("blueprint.selected", { count: tNumber(n) })}</span></span>
        {s.edges > 0 && <span class="num" data-tip={t("blueprint.lengthTip")}>{distance(s.lengthM)}</span>}
        {canLevel && <div class="bp-levels"><LevelButtons value={chosenLevel} tip={levelTip} onPick={pickLevel} /></div>}
        {canLevel && (
          <button class={"btn bp-level" + (s.level === null ? "" : " lv on")} style={s.level === null ? undefined : { "--lc": LEVEL_COLOURS[s.level], "--lk": inkOn(LEVEL_COLOURS[s.level]) }}
            data-tip={s.level === null ? t("blueprint.levelMixed") : levelName(s.level)} aria-label={t("blueprint.level")} aria-expanded={levels}
            onClick={() => setLevels(!levels)}>{s.level === null ? <Icon.autoLevel /> : levelLabel(s.level)}</button>
        )}
      {s.hsr + s.regular > 0 && (
        <div class="group" role="group" aria-label={t("blueprint.trackType")}>
          <button class={"btn" + (regularOn ? " on" : "")} onClick={() => pickType(false)}>{t("blueprint.regular")}</button>
          <button class={"btn" + (hsrOn ? " on" : "")} data-tip={t("blueprint.hsrTip", { factor: HSR_TRACK })} onClick={() => pickType(true)}>{t("blueprint.highSpeed")}</button>
        </div>
      )}
      {s.built && (
        <button class="btn text" data-tip={rebuildTip(q)} disabled={!q || (!q.changed.length && !q.stations)} onClick={() => rebuildPicked()}><Icon.wrench /><span class="bp-label">{t("blueprint.rebuild")}</span></button>
      )}
      {!s.built && (
        <div class="group">
          <button class="btn text" data-tip={t("blueprint.disableTip", { keyed: keys && !s.off, key: switchKey })} disabled={!s.on} onClick={() => void switchPicked(true)}><Icon.switchOff /><span class="bp-label">{t("blueprint.disable")}</span></button>
          <button class="btn text" data-tip={t("blueprint.enableTip", { keyed: keys, key: switchKey })} disabled={!s.off} onClick={() => void switchPicked(false)}><Icon.switchOn /><span class="bp-label">{t("blueprint.enable")}</span></button>
        </div>
      )}
      <PickDoubleUp />
      <button class="btn text" data-tip={t("blueprint.removeTip", { keyed: keys })} onClick={() => void removePicked()}><Icon.bin /><span class="bp-label">{t("blueprint.remove")}</span></button>
      <button class="btn text" data-tip={t("blueprint.clearTip")} onClick={() => clearBpPicked()}><Icon.cross /><span class="bp-label">{t("blueprint.clear")}</span></button>
      </div>
      {canLevel && levels && (
        <div class="blueprint-levels touch-levels">
          <LevelButtons value={chosenLevel} tip={levelTip} onPick={(l) => { pickLevel(l); setLevels(false); }} />
        </div>
      )}
    </>
  );
}

/** The quote's rows of the kinds in `only`, under their kinds' headings, each with its price worked out. */
function KindRows({ items, only }: { items: BlueprintItem[]; only?: Set<number> }) {
  return (
    <>
      {[...KINDS.map((kind, k): [MessageKey, number] => [kind, k]), ["blueprint.kindTramCars", TRAM_CARS] as [MessageKey, number]].map(([kind, k]) => {
        const rows = items.filter((r) => r.kind === k && (!only || only.has(k)));
        if (!rows.length) return null;
        return (
          <div class="blueprint-category">
            <div class="small b">{t(kind)}</div>
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
const NOT_TRACK = new Set([5, 10, 11, 12, 13, 14, TRAM_CARS]);

const levelsText = (r: { levelMin: Level; levelMax: Level }) =>
  r.levelMin === r.levelMax ? `${levelName(r.levelMin)} (${levelLabel(r.levelMin)})` : t("blueprint.levels", { from: levelLabel(r.levelMin), to: levelLabel(r.levelMax) });

/** A row's or a part's off switch, a checkbox ticked while any of its pieces is on. */
function OffSwitch({ on, what, flip }: { on: boolean; what: "joined" | "stretch" | "station"; flip: () => void }) {
  const tip = t("blueprint.switchTip", { on, what });
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
      <summary>{t("blueprint.breakdown")}</summary>
      <div class="bp-breakdown-list">
        {parts.map((p) => {
          const k = partKey(p), many = p.rows.length > 1 || p.junctions > 0, isOpen = many && unfolded.has(k);
          return (
          <div class="bp-part">
            <ul class="rows pick blueprint-items">
              <li class={"bp-part-head" + (p.on ? "" : " bp-off")} {...hover(p.edges)} onClick={() => locateNodes(p.nodes)}
                data-tip={tSeries([p.stationCount ? counted("station", p.stationCount) : "", p.lengthM ? distance(p.lengthM) : ""].filter(Boolean))}>
                {many ? (
                  <button class={"btn icon mini chevron" + (isOpen ? " open" : "")} aria-label={isOpen ? t("blueprint.fold") : t("blueprint.unfold")} aria-expanded={isOpen}
                    onClick={(e) => { e.stopPropagation(); fold(k); }}><Icon.chevron /></button>
                ) : <span class="bp-fold-gap" />}
                <OffSwitch on={p.on} what="joined" flip={() => void switchRow(p)} />
                <span class="grow">{p.label}</span>
                <span class="val">{money(p.on ? p.onCost : p.cost)}</span>
              </li>
              {isOpen && p.rows.map((r) => (
                <li class={"bp-sub" + (r.on ? "" : " bp-off")} {...hover(r.edges)} onClick={() => locateNodes(r.nodes)}
                  data-tip={r.kind === "stretch" ? [levelsText(r), r.crossings ? t("blueprint.crossingsTip", { count: r.crossings, money: money(r.crossingCost) }) : ""].filter(Boolean).join("\n") : undefined}>
                  <OffSwitch on={r.on} what={r.kind === "stretch" ? "stretch" : "station"} flip={() => void switchRow(r)} />
                  <span class="grow">{r.label}</span>
                  {r.kind === "stretch" && <span class="val muted">{distance(r.lengthM)}</span>}
                  <span class="val">{money(r.on ? r.onCost : r.cost)}</span>
                </li>
              ))}
              {isOpen && p.junctions > 0 && (
                <li class={"bp-sub bp-fixed" + (p.junctionOnCost > 0 || !p.junctionCost ? "" : " bp-off")} {...hover(p.edges)}>
                  <span class="grow">{counted("junction", p.junctions)}</span>
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
            <summary>{t("blueprint.byKind")}</summary>
            <KindRows items={items} />
          </details>
        )}
      </div>
    </details>
  );
}
