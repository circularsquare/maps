// The Lines tab: the list of lines with their riders, money and trains now.

import { demandAt, lineSeconds } from "../game/clock";
import { PERIODS, periodAt } from "../game/periods";
import { t } from "../i18n";
import { tPlace } from "../i18n/places";
import { useEffect, useRef, useState } from "preact/hooks";
import type { JSX } from "preact";
import { COUNT_MS, countingRiders, openings } from "../game/openingDay";
import { createLine } from "../game/lineActions";
import { boatsText, busesText, count, money } from "../game/format";
import { DEMAND_INDEX, type Demand, type LineInput, type LineStats } from "../game/types";
import { lineSort, selection, uiScale } from "../game/ui";
import { busStopsById, listedLines, edit, renameLine, stationsById } from "../game/world";
import { closedGroups, groupName, groupTree, lineGroups, toggleGroup } from "../game/lineGroups";
import { byHand, handOrder, setHandOrder } from "../game/handOrder";
import { GroupPill } from "./GroupPill";
import { trainsNowAt } from "../game/lineRunning";
import type { WorldView } from "../game/types";
import { cityById } from "../game/cities";
import { listWorld, scopedLists, stationZone } from "../game/listScope";
import { accountVersion } from "../game/boarding";
import { crowdClass, crowdLevel, dayNote, endNames, PERIOD_STRIDE, pendingOf, pendingRiders, periodLoad, profitKnown, runningToday, today, todayLoaded, upkeepToday, useToday, type LineToday } from "../game/today";
import { lineReversed } from "../game/northUp";
import { kindIn, lineCountry } from "../game/holidays";
import { DAY_KINDS } from "../game/days";
import { demandView, lineFiguresIn } from "../game/demand";
import { useClockQuery } from "./useClockQuery";
import { Chip, Icon, SortHead } from "./widgets";
import { lineMoneyTip, Money } from "./MoneyText";
import { locateLine } from "./locate";
import { byNear, natural, Near, ScopeToggle } from "./listParts";
import { pendingLabel, type PendingKind } from "../game/lineFigures";
import { hiddenLines, setLinesHidden, showAllLines, visToggles } from "../game/lineVisibility";
import { vehicleOf } from "../game/vehicle";
import { stationCity } from "../game/money";
import { linesView, type LineCol } from "../game/linesView";
import { LinesGear, TYPE_ICONS, typeName } from "./LinesGear";

/** the word for a line that is not running; nothing for a running one */
const statusText = (s: string) => (s === "planned" ? t("linesTab.planned") : s === "broken" ? t("linesTab.broken") : "");
/** T-3798: the Type sort's order: train, high speed, tram, bus, ferry */
const lineTypeRank = (st: LineStats | undefined) => {
  if (!st) return 0;
  const v = vehicleOf(st);
  return v === "train" ? (st.fast ? 1 : 0) : v === "tram" ? 2 : v === "bus" ? 3 : 4;
};
/** T-3808: a row's type, its icon with its name in the hover */
function TypeCell({ rank }: { rank: number }) {
  const I = TYPE_ICONS[rank];
  return <span class="val w-kind muted" data-tip={typeName(rank)}><I /></span>;
}
/** how far a row is dragged before it counts as a drag, not a click (px) */
const DRAG_PX = 6;

/** T-2392: a line's riders while they are worked out again: the old figure, the label small above it
 * (alone where there is no figure yet). */
function PendingRiders({ k, text }: { k: PendingKind; text: string }) {
  return (
    <span class={"val w-riders pending" + (text ? " stacked" : " muted")}>
      {text ? <><span class="pending-label muted">{pendingLabel(k)}</span><span>{text}</span></> : pendingLabel(k)}
    </span>
  );
}

/** T-446: one line's riders in the past 24 hours, its own component so only it re-renders as the count moves. */
function RidersToday({ l }: { l: LineInput }) {
  const line = l.id;
  const d = today.value.get(line);
  const o = openings.value.get(line);
  const [, frame] = useState(0);
  // T-1434: on its opening day the figure counts up from 0 once, a frame at a time
  useEffect(() => {
    if (!o?.landed) return;
    let raf = 0;
    const step = () => {
      frame((n) => n + 1);
      if (performance.now() - o.landed < COUNT_MS) raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [o?.landed]);
  const k = pendingOf(l);
  if (k) { const r = pendingRiders(l); return <PendingRiders k={k} text={r !== null ? count(r) : ""} />; }
  const shown = countingRiders(o, d?.riders ?? 0);
  const text = shown !== null ? count(shown) : d ? count(d.riders) : "";
  return <span class="val w-riders" data-tip={d ? t("linesTab.ridersTip", { count: count(d.riders) }) : undefined}>{text}</span>;
}

/** T-735, T-867, T-2475: a line's expected profit today: a whole day's fares of its kind of day now
 * minus a day's running cost and its share of upkeep. */
const lineNet = (d: LineToday | undefined, l: LineInput, st: LineStats) => (d && profitKnown(l) ? d.dayFares - runningToday(l, st) - upkeepToday(l, st) : 0);
function MoneyToday({ l, st }: { l: LineInput; st: LineStats }) {
  const d = today.value.get(l.id);
  // T-2536: no figure before the first profit with trains (the upkeep alone is not a profit)
  if (!d || !profitKnown(l)) return <span class="val w-linemoney" />;
  const run = runningToday(l, st), up = upkeepToday(l, st), net = d.dayFares - run - up;
  return (
    <span class="val w-linemoney" data-tip={lineMoneyTip(d.dayFares, run, up, st.bus?.ferry ? "boats" : st.tram ? "trams" : !!st.bus, dayNote(l, st))}>
      <Money dollars={net} />
    </span>
  );
}

/** T-472, T-1677, T-1684, T-2236: a line's trains now, coloured by its load past the rated load in
 * the period now, graded over its stretches (T-2279) from `v`, its own kind of day's numbers, and
 * hovered with that day's figures for the level now as the line panel's dot is (`crowdLevel`). */
function TrainsNow({ line, n, period, at, bus, ferry, tram, v, lev, stops, flip, ends }: { line: string; n: number; period: string; at: number; bus?: boolean; ferry?: boolean; tram?: boolean; v?: Float64Array; lev: Demand; stops: number; flip: boolean; ends?: [string, string] }) {
  const c = v && n > 0 ? crowdLevel(v, lev, stops, flip, ends, tram ? "tram" : "train") : null;
  const d = today.value.get(line);
  const load = d && (v ? periodLoad(v, at, stops) ?? d.crowdLoad : d.crowdLoad);
  const cls = d && n > 0 ? crowdClass(load!) : "";
  // T-2919: a bus or boat line's cell is a train line's; until its crowding is in, its vehicles
  const tip = c?.tip ?? (bus ? t("linesTab.vehiclesTip", { period, vehicles: ferry ? boatsText(n) : busesText(n) }) : undefined);
  return <span class={"val w-trains " + (cls || "muted")} data-tip={tip}>{n}</span>;
}

/** T-1791: a group's riders, money and trains, each line's added up; hovers list them by line. */
function GroupFigures({ ls, w, trains, period, col }: { ls: LineInput[]; w: WorldView; trains: (id: string) => number; period: string; col: (c: LineCol) => boolean }) {
  const tod = today.value;
  const run = ls.filter((l) => w.lineStats[l.id]?.status === "running");
  if (!run.length) {
    const st = ls.some((l) => w.lineStats[l.id]?.status === "planned") ? "planned" : "broken";
    return <span class="val muted">{statusText(st)}</span>;
  }
  const by = (f: (l: LineInput) => string) => run.map((l) => `${l.name}: ${f(l)}`).join("\n");
  // a line just started names the group's state over one only loading
  const pending = run.map((l) => pendingOf(l)).filter((k): k is PendingKind => !!k).sort()[0];
  // (T-2678: while pending, each line's figure as its row shows it)
  const ridersOf = (l: LineInput) => (pending && pendingOf(l) ? pendingRiders(l) : tod.get(l.id)?.riders) ?? null;
  const riders = run.reduce((n, l) => n + (ridersOf(l) ?? 0), 0);
  const any = run.some((l) => ridersOf(l) !== null);
  const anyProfit = run.some((l) => tod.has(l.id) && profitKnown(l));
  const net = run.reduce((n, l) => n + lineNet(tod.get(l.id), l, w.lineStats[l.id]), 0);
  const n = run.reduce((k, l) => k + trains(l.id), 0);
  return (
    <>
      {!col("riders") ? null : pending ? <PendingRiders k={pending} text={any ? count(riders) : ""} />
        : <span class="val w-riders" data-tip={any ? t("linesTab.groupRidersTip", { count: count(riders), lines: by((l) => count(tod.get(l.id)?.riders ?? 0)) }) : undefined}>{any ? count(riders) : ""}</span>}
      {!col("money") ? null : anyProfit ? <span class="val w-linemoney" data-tip={t("linesTab.groupProfitTip", { money: money(net), lines: run.filter((l) => profitKnown(l)).map((l) => `${l.name}: ${money(lineNet(tod.get(l.id), l, w.lineStats[l.id]))}`).join("\n") })}><Money dollars={net} /></span> : <span class="val w-linemoney" />}
      {col("trains") && <span class="val w-trains" data-tip={t("linesTab.groupTrainsTip", { period, count: n, lines: by((l) => String(trains(l.id))) })}>{n}</span>}
    </>
  );
}

/** T-3420: a row's box, ticked while its lines are drawn on the map; a group's is ticked while all are, and a click
 * on it shows them all once any is hidden */
function VisToggle({ ids, group }: { ids: string[]; group?: boolean }) {
  const hid = hiddenLines.value, n = ids.filter((id) => hid.has(id)).length;
  const label = group ? t("linesTab.showGroup") : t("linesTab.showLine");
  return (
    <input type="checkbox" class={"vis-check" + (n > 0 && n < ids.length ? " some" : "")} checked={n === 0} aria-label={label}
      onClick={(e) => { e.stopPropagation(); setLinesHidden(ids, n === 0); }} onDblClick={(e) => e.stopPropagation()} />
  );
}

/** A drag of a row (T-1791, T-1813): the line (or "g:" and the group), where it started, the row
 * it is over and which part of it. */
interface Drag { id: string; x0: number; y0: number; x: number; y: number; on: boolean; over: string | null; part: "before" | "on" | "after" }

function TotalToday() {
  const tod = today.value;
  if (!tod.size) return null;
  let n = 0;
  for (const d of tod.values()) n += d.riders;
  return <span class="muted">{t("linesTab.ridersToday", { count: count(n) })}</span>;
}

export function LinesTab() {
  const [creating, setCreating] = useState(false);
  const [drag, setDrag] = useState<Drag | null>(null);
  const [renaming, setRenaming] = useState<number | null>(null);
  const [renamingLine, setRenamingLine] = useState<string | null>(null);
  /** a drag just ended: the click that follows it selects nothing */
  const dragged = useRef(false);
  useToday(); // T-446: riders in the past 24 hours, kept fresh while the tab is open
  void accountVersion.value; // re-sorted each game hour; the counts themselves move on their own
  void todayLoaded.value;
  const w = listWorld.value;
  // T-1677, T-2279: every running train line's crowding by period on its own kind of day, for the
  // trains cells, asked again when the network or the boarding account changes: one ask per kind of
  // day, a fixed six so the hooks hold
  const crowdLines = (w ? listedLines(w) : []).filter((l) => w!.lineStats[l.id]?.status === "running");
  const acct = accountVersion.value;
  const kindOf = (l: LineInput) => kindIn(lineCountry(l.stops), Math.floor(lineSeconds(w!.lineStats[l.id]) / 86400));
  const byKind = DAY_KINDS.map((k) => crowdLines.filter((l) => kindOf(l) === k).map((l) => l.num));
  const kindCrowd = byKind.map((nums, k) => useClockQuery(nums.length ? { what: "linesPeriods", lines: nums, day: k } : null, [w?.version, acct, nums.join(",")], (d) => d));
  // T-3799: a sort on a column the gear turned off goes to Name (the rows sort by name meanwhile, below)
  const sortOff = (linesView.value.cols as readonly string[]).includes(lineSort.value.key);
  useEffect(() => { if (sortOff) lineSort.value = { key: "name", dir: 1 }; }, [sortOff]);
  if (!w) return null;
  // T-1926: each line's numbers come after their count, in `nums`' order
  const split = (d: Float64Array | null, nums: number[]) => {
    const at: Float64Array[] = [];
    for (let i = 0; d && i < d.length; i += 1 + d[i]) at.push(d.subarray(i + 1, i + 1 + d[i]));
    return at.length === nums.length ? new Map(nums.map((n, i) => [n, at[i]])) : new Map<number, Float64Array>();
  };
  const kindAt = byKind.map((nums, k) => split(kindCrowd[k], nums));
  const usable = (l: LineInput, v: Float64Array | undefined) => (v && v.length >= PERIODS.length * PERIOD_STRIDE && lineFiguresIn(demandView.value, l.id, l.stops.length) ? v : undefined);
  const crowdOf = (l: LineInput) => usable(l, kindAt[kindOf(l)].get(l.num));
  const sel = selection.value;
  // T-309: each line's demand level now in its own local time
  const levOf = (id: string) => DEMAND_INDEX[demandAt(lineSeconds(w.lineStats[id]))];
  const tod = today.peek();
  const newLine = async () => {
    if (creating) return;
    setCreating(true);
    try { await createLine(); } finally { setCreating(false); }
  };
  // a line that is not running has no train count and no riders: it sorts as zero
  // T-3521: today's kind of day's trains
  const trains = (id: string) => (w.lineStats[id]?.status === "running" ? trainsNowAt(id, levOf(id)) : 0);
  // T-1813: while the player's hand order holds, no sort
  const hand = handOrder.value.lines;
  // T-3799: the columns the gear turned off
  const view = linesView.value;
  const offCols = new Set<string>(view.cols);
  const col = (c: LineCol) => !offCols.has(c);
  const key = hand.length ? null : sortOff ? "name" : lineSort.value.key, dir = sortOff ? 1 : lineSort.value.dir;
  const isSel = (id: string) => (sel?.kind === "line" || sel?.kind === "train") && sel.line === id;
  // T-332: only the viewed place unless the world is chosen
  const sc = scopedLists(w.save.stations, listedLines(w), (k, id) => k === "line" && isSel(id), key === "near");
  // T-3816: the gear's filters, on top of the scope. The choices are the whole game's, so they hold still as the
  // view moves; a turned-off choice the game does not offer now (another game's city) counts for nothing.
  const listed = listedLines(w);
  const stAt = stationsById(w), busAt = busStopsById(w);
  const cityIds = new Map<string, Set<string>>();
  for (const l of listed) {
    const cs = new Set<string>();
    for (const id of l.stops) { const s = stAt.get(id) ?? busAt.get(id), c = s && stationCity(s); if (c) cs.add(c); }
    cityIds.set(l.id, cs);
  }
  const rankOf = (l: LineInput) => lineTypeRank(w.lineStats[l.id]);
  const modeChoices: number[] = [...new Set(listed.filter((l) => l.stops.length).map(rankOf))].sort((a, b) => a - b);
  const cityChoices = [...new Set([...cityIds.values()].flatMap((cs) => [...cs]))].map((id): [string, string] => [id, tPlace(id, cityById(id).name)]).sort((a, b) => natural(a[1], b[1]));
  const offModes = new Set(modeChoices.length >= 2 ? view.modes.filter((m) => modeChoices.includes(m)) : []);
  const offCities = new Set(cityChoices.length >= 2 ? view.cities.filter((c) => cityChoices.some(([id]) => id === c)) : []);
  /** a line the filters let through: one with no stops yet always, and one in no city whatever cities are chosen */
  const passes = (l: LineInput) => {
    if (!l.stops.length) return true;
    if (offModes.has(rankOf(l))) return false;
    const cs = cityIds.get(l.id)!;
    return !offCities.size || !cs.size || [...cs].some((c) => !offCities.has(c));
  };
  const leftOut = offModes.size || offCities.size ? listed.filter((l) => !passes(l)).length : 0;
  /** the selected line stays listed, as the scope keeps it */
  const shows = (l: LineInput) => passes(l) || isSel(l.id);
  // T-670: on the whole world, a city column once lines start in two or more cities (a line's city is where its first stop is)
  const cityOf = new Map<string, string>();
  if (sc.zone === null && col("city")) {
    for (const l of sc.lines) {
      const s = stAt.get(l.stops[0]), z = s ? stationZone(s) : null;
      cityOf.set(l.id, z ? tPlace(z, cityById(z).name) : "");
    }
  }
  const showCity = new Set(sc.lines.filter(shows).map((l) => cityOf.get(l.id)).filter(Boolean)).size >= 2;
  // T-1791: a group is one row, sorted by its lines' figures added up; it shows, with all its
  // lines, when any of them is in the viewed place. T-3423: a group's members are lines and groups,
  // to any depth; `ls` is every line in it. T-3816: a group shows if any line in it passes the gear's filters, and
  // inside it only those that pass; its figures are still every line's.
  const groups = lineGroups(listed), tree = groupTree(listed);
  const parentOf = (g: number) => { const p = tree.get(g)?.parent ?? 0; return tree.has(p) ? p : 0; };
  /** the group a line sits straight in, 0 for none */
  const homeOf = (l: LineInput) => (l.group && tree.has(l.group) ? l.group : 0);
  /** `g` and the groups around it, nearest first */
  const upFrom = (g: number) => { const out: number[] = []; for (let c = g; c && !out.includes(c); c = parentOf(c)) out.push(c); return out; };
  type Item = { ls: LineInput[]; g: number; name: string; kids: Item[] };
  const lineItem = (l: LineInput): Item => ({ ls: [l], g: 0, name: l.name, kids: [] });
  const groupItem = (g: number): Item => {
    const n = tree.get(g)!, ls = groups.get(g) ?? n.lines;
    return { ls, g, name: groupName(ls, g), kids: [...n.lines.map(lineItem), ...n.groups.filter((k) => tree.has(k)).map(groupItem)] };
  };
  const shownIds = new Set(sc.lines.map((l) => l.id));
  // every line's item, the viewed place's or not, so a hand order made here keeps the others' places
  const allItems = [...listed.filter((l) => !homeOf(l)).map(lineItem), ...[...tree.keys()].filter((g) => !parentOf(g)).map(groupItem)];
  const sum = (ls: LineInput[], f: (l: LineInput) => number) => ls.reduce((n, l) => n + f(l), 0);
  const riders = (ls: LineInput[]) => sum(ls, (l) => tod.get(l.id)?.riders ?? 0);
  const net = (ls: LineInput[]) => sum(ls, (l) => lineNet(tod.get(l.id), l, w.lineStats[l.id]));
  const near = (ls: LineInput[]) => ls.reduce<number | undefined>((m, l) => { const d = sc.lineDist.get(l.id); return d === undefined ? m : m === undefined ? d : Math.min(m, d); }, undefined);
  // T-3798: a group goes with the type most of its lines are, the earlier type on a tie
  const typeOf = (ls: LineInput[]) => {
    const n = [0, 0, 0, 0, 0];
    for (const l of ls) n[lineTypeRank(w.lineStats[l.id])]++;
    return n.indexOf(Math.max(...n));
  };
  const order = (a: Item, b: Item) =>
    (key === "kind" ? (typeOf(a.ls) - typeOf(b.ls)) * dir
      : key === "city" ? natural(cityOf.get(a.ls[0].id) ?? "", cityOf.get(b.ls[0].id) ?? "") * dir
      : key === "riders" ? (riders(a.ls) - riders(b.ls)) * dir
      : key === "money" ? (net(a.ls) - net(b.ls)) * dir
      : key === "trains" ? (sum(a.ls, (l) => trains(l.id)) - sum(b.ls, (l) => trains(l.id))) * dir
      : key === "near" ? byNear(near(a.ls), near(b.ls), dir) : 0)
    || natural(a.name, b.name) * (key === "name" ? dir : 1) || a.ls[0].num - b.ls[0].num;
  /** items in the order shown, a group's members too: the hand order, else the tab's sort */
  const sortItems = (its: readonly Item[]): Item[] => {
    const out = hand.length ? byHand([...its].sort(order), (it) => it.ls.map((l) => l.id), hand) : [...its].sort(order);
    for (const it of out) if (it.g) it.kids = sortItems(it.kids);
    return out;
  };
  const sorted = sortItems(allItems);
  const items = sorted.filter((it) => it.ls.some((l) => shownIds.has(l.id) && shows(l)));
  /** a group's members as listed: the lines and groups with a line the filters let through */
  const kidsShown = (it: Item) => it.kids.filter((k) => k.ls.some(shows));
  /** an item's lines in the order shown */
  const linesIn = (it: Item): string[] => (it.g ? it.kids.flatMap(linesIn) : [it.ls[0].id]);
  /** every line in the order shown, the hidden ones included */
  const fullOrder = () => sorted.flatMap(linesIn);
  const findGroup = (g: number, its: readonly Item[] = sorted): Item | undefined => {
    for (const it of its) {
      if (it.g === g) return it;
      const k = it.g ? findGroup(g, it.kids) : undefined;
      if (k) return k;
    }
    return undefined;
  };
  const closed = closedGroups.value;
  const periodOf = (l: LineInput) => PERIODS[periodAt(lineSeconds(w.lineStats[l.id]))]?.name ?? "";

  // T-1791, T-1813, T-3423: drag a row onto the middle of a group's row to put it in that group, or
  // onto a line's row to join the line's group (or, beside it where it is, to make a new group of the
  // two); into a gap between rows to put it there, out of its group unless the gap is inside it (or
  // another open group, which it then joins). A line and a group's row move alike; never into itself.
  type Row = { key: string; it: Item; line?: LineInput; home: number; depth: number };
  const rowsOf = (it: Item, home: number, depth: number): Row[] => it.g === 0
    ? [{ key: "l:" + it.ls[0].id, it, line: it.ls[0], home, depth }]
    : [{ key: "g:" + it.g, it, home, depth }, ...(!closed.has(it.g) ? kidsShown(it).flatMap((k) => rowsOf(k, it.g, depth + 1)) : [])];
  const rowList: Row[] = items.flatMap((it) => rowsOf(it, 0, 0));
  /** What a drop does: `with` the line or group dropped onto, or `at` the gap (a row index) */
  type Drop = { with: Row } | { at: number };
  const dropAt = (d: Drag): Drop | null => {
    const i = rowList.findIndex((r) => r.key === d.over);
    if (i < 0) return null;
    if (d.part === "on") return { with: rowList[i] };
    return { at: d.part === "before" ? i : i + 1 };
  };
  /** the gap's group (0 for none) for a row from group `from`, and the line it goes before or after:
   * the group of the row under the gap, unless `from` closes between the rows (the gap under its last
   * row counts as inside it for its own members) */
  const gapOf = (at: number, from: number): { g: number; before?: string; after?: string } => {
    const a = rowList[at - 1], b = rowList[at];
    const under = b ? b.home : 0;
    if (a && from && from !== under && upFrom(a.home).includes(from) && !upFrom(under).includes(from)) { const ls = linesIn(a.it); return { g: from, after: ls[ls.length - 1] }; }
    if (!b) return { g: 0 };
    return { g: under, before: linesIn(b.it)[0] };
  };
  /** what the drag would change, or null for nothing */
  type Plan = { join?: number | "new"; with?: string; parent?: number; order?: string[]; gapG?: number };
  const plan = (d: Drag): Plan | null => {
    const to = dropAt(d);
    if (!to) return null;
    const grp = d.id.startsWith("g:") ? Number(d.id.slice(2)) : 0;
    const dragged = grp ? undefined : listed.find((l) => l.id === d.id);
    const from = grp ? parentOf(grp) : dragged ? homeOf(dragged) : 0;
    /** `g` is the dragged group or inside it */
    const within = (g: number) => !!grp && upFrom(g).includes(grp);
    if ("with" in to) {
      const r = to.with;
      if (r.line?.id === d.id || r.it.g === grp && grp) return null;
      if (r.it.g) return within(r.it.g) || r.it.g === from ? null : { join: r.it.g };
      const c = r.home;
      if (within(c)) return null;
      return c && c !== from ? { join: c } : { join: "new", with: r.line!.id, parent: c };
    }
    const gap = gapOf(to.at, from);
    if (within(gap.g)) return null;
    const moved = grp ? linesIn(findGroup(grp)!) : [d.id];
    let order = fullOrder().filter((id) => !moved.includes(id));
    const anchor = gap.before ?? gap.after;
    if (anchor && moved.includes(anchor)) return null; // onto its own place
    const i = anchor ? order.indexOf(anchor) + (gap.after ? 1 : 0) : order.length;
    order = [...order.slice(0, i), ...moved, ...order.slice(i)];
    const out: Plan = { gapG: gap.g, ...(order.join() === fullOrder().join() ? {} : { order }), ...(gap.g !== from ? { join: gap.g } : {}) };
    return out.order || out.join !== undefined ? out : null;
  };
  const dropLine = (d: Drag) => {
    const p = plan(d);
    if (!p) return;
    if (p.order) setHandOrder("lines", p.order);
    // T-1855: a group's row is never sent as a line
    const grp = d.id.startsWith("g:") ? Number(d.id.slice(2)) : 0;
    const lines = grp ? [] : [Number(d.id)], groups = grp ? [grp] : [];
    if (p.join === "new") void edit({ op: "groupMembers", lines: [Number(p.with), ...lines], groups, group: "new", parent: p.parent ?? 0 });
    else if (p.join !== undefined) void edit({ op: "groupMembers", lines, groups, group: p.join, parent: 0 });
  };
  const pointerDown = (e: PointerEvent, id: string) => {
    if (e.button !== 0 || e.pointerType === "touch") return;
    if ((e.target as Element | null)?.closest?.("button, input")) return;
    const d: Drag = { id, x0: e.clientX, y0: e.clientY, x: e.clientX, y: e.clientY, on: false, over: null, part: "on" };
    const move = (m: PointerEvent) => {
      if (!d.on && Math.hypot(m.clientX - d.x0, m.clientY - d.y0) < DRAG_PX) return;
      d.on = true;
      d.x = m.clientX; d.y = m.clientY;
      const row = document.elementFromPoint(m.clientX, m.clientY)?.closest("[data-drop]") as HTMLElement | null;
      d.over = row?.dataset.drop ?? null;
      if (row) {
        // the middle third groups, the rest is the gap above or below
        const b = row.getBoundingClientRect(), f = (m.clientY - b.top) / Math.max(1, b.height);
        d.part = f < 1 / 3 ? "before" : f > 2 / 3 ? "after" : "on";
      }
      setDrag({ ...d });
    };
    const end = (u: PointerEvent | KeyboardEvent) => {
      removeEventListener("pointermove", move);
      removeEventListener("pointerup", end);
      removeEventListener("keydown", key);
      if (d.on) {
        dragged.current = true;
        setTimeout(() => { dragged.current = false; }, 0);
        if (u.type === "pointerup") dropLine(d);
      }
      setDrag(null);
    };
    const key = (k: KeyboardEvent) => { if (k.key === "Escape") end(k); };
    addEventListener("pointermove", move);
    addEventListener("pointerup", end);
    addEventListener("keydown", key);
  };
  /** the rows marked while dragging: the one dropped onto, or the gap's line above or below a row */
  const marks = new Map<string, string>();
  let markDepth = 0; // the indent of the group the gap marker puts the row in
  if (drag?.on) {
    const p = plan(drag), to = p && dropAt(drag);
    if (to && "with" in to) {
      marks.set(to.with.key, " drop-on");
      if (typeof p.join === "number" && to.with.line) marks.set("g:" + p.join, " drop-on");
    } else if (to) {
      marks.set(drag.over!, drag.part === "before" ? " drop-before" : " drop-after");
      const gr = p?.gapG ? rowList.find((r) => r.key === "g:" + p.gapG) : undefined;
      markDepth = gr ? gr.depth + 1 : 0;
    }
  }
  const pick = (id: string) => { if (!dragged.current) selection.value = { kind: "line", line: id }; };
  // T-3420: the lines hidden on the map, and whether the rows show their eyes
  const hid = hiddenLines.value, eyes = visToggles.value;
  /** T-3423: a row in a group, indented a step for each group it is in */
  const depthOf = (depth: number, drop?: string) => {
    const style: Record<string, number> = {};
    if (depth) style["--depth"] = depth;
    if (drop && marks.has(drop)) style["--mark-depth"] = markDepth;
    return { cls: depth ? " in-group" : "", style: depth || drop && marks.has(drop) ? style : undefined };
  };
  const lineRow = (l: LineInput, depth: number) => {
    const st = w.lineStats[l.id];
    const on = (sel?.kind === "line" || sel?.kind === "train") && sel.line === l.id;
    const drop = "l:" + l.id, dp = depthOf(depth, drop);
    return (
      <li class={(on ? "on" : "") + dp.cls + (hid.has(l.id) ? " hidden-line" : "") + (marks.get(drop) ?? "") + (drag?.on && drag.id === l.id ? " dragged" : "")} style={dp.style}
        data-drop={drop} onPointerDown={(e) => pointerDown(e, l.id)} onClick={() => pick(l.id)} onDblClick={() => locateLine(l.stops)}>
        {eyes && <VisToggle ids={[l.id]} />}
        <Chip line={l} />
        {renamingLine === l.id ? (
          <input class="name-edit grow" value={l.name} aria-label={t("linesTab.lineName")} ref={(el) => { if (el && document.activeElement !== el) { el.focus(); el.select(); } }}
            onClick={(e) => e.stopPropagation()} onDblClick={(e) => e.stopPropagation()} onPointerDown={(e) => e.stopPropagation()}
            onBlur={(e) => { setRenamingLine(null); renameLine(l.id, e.currentTarget.value); }} onKeyDown={(e) => { if (e.key === "Enter") e.currentTarget.blur(); if (e.key === "Escape") setRenamingLine(null); }} />
        ) : (
          // T-2786: a triple click renames (a double click snaps the camera)
          <span class="grow line-name" onClick={(e) => { if (e.detail === 3) { e.stopPropagation(); setRenamingLine(l.id); } }}>{l.name}</span>
        )}
        {showCity && <span class="val muted w-city" data-tip={cityOf.get(l.id) ? t("linesTab.cityTip", { city: cityOf.get(l.id) }) : undefined}>{cityOf.get(l.id)}</span>}
        {st.status !== "running" ? (
          <span class="val muted">{statusText(st.status)}</span>
        ) : (
          <>
            {col("riders") && <RidersToday l={l} />}
            {col("money") && <MoneyToday l={l} st={st} />}
            {col("trains") && <TrainsNow line={l.id} n={trainsNowAt(l.id, levOf(l.id))} period={periodOf(l)} at={periodAt(lineSeconds(st))} bus={!!st.bus} ferry={!!st.bus?.ferry} tram={!!st.tram} v={crowdOf(l)} lev={demandAt(lineSeconds(st))} stops={l.stops.length} flip={lineReversed(l, w, !!st.loop)} ends={endNames(l)} />}
          </>
        )}
        {col("kind") && <TypeCell rank={lineTypeRank(st)} />}
        {key === "near" && <Near m={sc.lineDist.get(l.id)} />}
      </li>
    );
  };
  const groupRows = (it: Item, depth: number): JSX.Element => {
    const isOpen = !closed.has(it.g), drop = "g:" + it.g, dp = depthOf(depth, drop);
    const on = !isOpen && sel?.kind === "group" && sel.group === it.g;
    return (
      <>
        <li class={"group-row" + (on ? " on" : "") + dp.cls + (it.ls.every((l) => hid.has(l.id)) ? " hidden-line" : "") + (marks.get(drop) ?? "") + (drag?.on && drag.id === drop ? " dragged" : "")} style={dp.style} data-drop={drop} onPointerDown={(e) => pointerDown(e, drop)}
          onClick={() => { if (!dragged.current) selection.value = { kind: "group", group: it.g }; }} onDblClick={() => locateLine(it.ls.flatMap((l) => l.stops))}>
          {eyes && <VisToggle ids={it.ls.map((l) => l.id)} group />}
          <button class={"btn icon mini chevron" + (isOpen ? " open" : "")} data-tip={isOpen ? t("linesTab.closeGroup") : t("linesTab.openGroup")} aria-label={isOpen ? t("linesTab.closeGroup") : t("linesTab.openGroup")}
            aria-expanded={isOpen} onClick={(e) => { e.stopPropagation(); toggleGroup(it.g); }} onDblClick={(e) => e.stopPropagation()}><Icon.chevron /></button>
          <GroupPill ls={it.ls} g={it.g} tip={it.ls.map((l) => l.name).join("\n")} />
          {renaming === it.g ? (
            <input class="name-edit grow" value={it.name} aria-label={t("linesTab.groupName")} ref={(el) => { if (el && document.activeElement !== el) { el.focus(); el.select(); } }}
              onClick={(e) => e.stopPropagation()} onDblClick={(e) => e.stopPropagation()} onPointerDown={(e) => e.stopPropagation()}
              onBlur={(e) => renameGroup(it, e.currentTarget.value)} onKeyDown={(e) => { if (e.key === "Enter") e.currentTarget.blur(); if (e.key === "Escape") setRenaming(null); }} />
          ) : (
            <span class="grow line-name" onDblClick={(e) => { e.stopPropagation(); setRenaming(it.g); }}>{it.name}</span>
          )}
          {showCity && <span class="val muted w-city">{cityOf.get(it.ls[0].id)}</span>}
          <GroupFigures ls={it.ls} w={w} trains={trains} period={periodOf(it.ls[0])} col={col} />
          {col("kind") && <TypeCell rank={typeOf(it.ls)} />}
          {key === "near" && <Near m={near(it.ls)} />}
        </li>
        {isOpen && kidsShown(it).map((k) => (k.g ? groupRows(k, depth + 1) : lineRow(k.ls[0], depth + 1)))}
      </>
    );
  };
  const renameGroup = (it: Item, typed: string) => {
    setRenaming(null);
    const n = typed.trim();
    if (n && n !== it.name) void edit({ op: "groupName", group: it.g, name: n });
  };
  const ghost = drag?.on ? w.save.lines.find((l) => l.id === drag.id) : undefined;
  const ghostGroup = drag?.on && drag.id.startsWith("g:") ? findGroup(Number(drag.id.slice(2))) : undefined;
  return (
    <>
      <div class="line-list-actions">
        <span style={{ display: "flex", gap: "6px" }}>
          <button class="btn text" data-guide="line" data-tip={t("linesTab.newLineTip", { key: "N" })} onClick={newLine} disabled={creating}><Icon.plus />{t("linesTab.newLine")}<span class="key muted">N</span></button>
        </span>
        <span class="list-vis">
          {/* T-3420: hide mode; leaving it shows every line again */}
          <button class={"btn icon" + (eyes ? " on" : "")} data-tip={eyes ? t("linesTab.showAllLines") : t("linesTab.hideLines")} aria-label={eyes ? t("linesTab.showAllLines") : t("linesTab.hideLines")} aria-pressed={eyes}
            onClick={() => { if (visToggles.peek()) showAllLines(); visToggles.value = !visToggles.peek(); }}><Icon.eye /></button>
          <ScopeToggle />
          <LinesGear modes={modeChoices} cities={cityChoices} leftOut={leftOut} />
        </span>
      </div>
      {items.length === 0 && <p class="muted">{listedLines(w).length === 0 ? t("linesTab.none") : t("linesTab.noneHere")}</p>}
      <div class={"lines-box" + (showCity ? " with-city" : "") + (key === "near" ? " near-on" : "") + (eyes ? " vis-on" : "")}>
      {listedLines(w).length > 0 && <SortHead sort={lineSort} hand={hand.length ? () => setHandOrder("lines", []) : undefined} cols={[
        { key: "name", label: t("linesTab.colName"), tip: t("linesTab.sortName"), first: 1 },
        ...(showCity ? [{ key: "city" as const, label: t("linesTab.colCity"), tip: t("linesTab.sortCity"), first: 1 as const, cls: "w-city", left: true }] : []),
        ...([
          { key: "riders", label: t("linesTab.colRiders"), tip: t("linesTab.sortRiders"), first: -1, cls: "w-riders" },
          { key: "money", label: t("linesTab.colMoney"), tip: t("linesTab.sortMoney"), first: -1, cls: "w-linemoney" },
          { key: "trains", label: t("linesTab.colTrains"), tip: t("linesTab.sortTrains"), first: -1, cls: "w-trains" },
          { key: "kind", label: t("linesTab.colType"), tip: t("linesTab.sortType"), first: 1, cls: "w-kind", left: true },
        ] as const).filter((c) => col(c.key)),
        { key: "near", label: t("linesTab.colNear"), tip: t("linesTab.sortNear"), first: 1, ...(key === "near" ? { cls: "w-near" } : { cls: "near-off", beside: true }) },
      ]} />}
      <ul class={"rows pick" + (drag?.on ? " dragging" : "")}>
        {items.map((it) => (it.g ? groupRows(it, 0) : lineRow(it.ls[0], 0)))}
      </ul>
      </div>
      {ghost && drag && (
        // the dock is zoomed by the UI scale, the pointer is not
        <div class="row-ghost" style={{ left: (drag.x + 12) / uiScale.value + "px", top: (drag.y + 8) / uiScale.value + "px" }}>
          <Chip line={ghost} plain /><span>{ghost.name}</span>
        </div>
      )}
      {ghostGroup && drag && (
        <div class="row-ghost" style={{ left: (drag.x + 12) / uiScale.value + "px", top: (drag.y + 8) / uiScale.value + "px" }}>
          <GroupPill ls={ghostGroup.ls} g={ghostGroup.g} /><span>{ghostGroup.name}</span>
        </div>
      )}
      <div class="pane-foot">
        <TotalToday />
      </div>
    </>
  );
}
