// The Lines tab: the list of lines with their riders, money and trains now.

import { demandAt, lineSeconds } from "../game/clock";
import { PERIODS, periodAt } from "../game/periods";
import { t } from "../i18n";
import { useEffect, useRef, useState } from "preact/hooks";
import { COUNT_MS, countingRiders, openings } from "../game/openingDay";
import { createLine, newBusLine, newFerryLine } from "../game/lineActions";
import { boatsText, busesText, count, money } from "../game/format";
import { DEMAND_INDEX, type Demand, type LineInput, type LineStats } from "../game/types";
import { lineSort, selection, uiScale } from "../game/ui";
import { edit, stationsById } from "../game/world";
import { closedGroups, groupName, lineGroups, toggleGroup } from "../game/lineGroups";
import { byHand, handOrder, setHandOrder } from "../game/handOrder";
import { GroupPill } from "./GroupPill";
import type { WorldView } from "../game/types";
import { cityById } from "../game/cities";
import { listWorld, scopedLists, stationZone } from "../game/listScope";
import { accountVersion } from "../game/boarding";
import { crowdClass, crowdLevel, dayNote, PERIOD_STRIDE, pendingOf, pendingRiders, periodLoad, profitKnown, runningToday, today, todayLoaded, useToday, type LineToday } from "../game/today";
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

/** the word for a line that is not running; nothing for a running one */
const statusText = (s: string) => (s === "planned" ? t("linesTab.planned") : s === "broken" ? t("linesTab.broken") : "");
/** how far a row is dragged before it counts as a drag, not a click (px) */
const DRAG_PX = 6;

/** T-2392: a line's riders while they are worked out again: the old figure, the label small above it
 * (alone where there is no figure yet). */
function PendingRiders({ k, text }: { k: PendingKind; text: string }) {
  return (
    <span class={"val w-riders pending" + (text ? " stacked" : " muted")} data-tip={t("linesTab.pendingTip")}>
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
const lineNet = (d: LineToday | undefined, l: LineInput, st: LineStats) => (d && profitKnown(l) ? d.dayFares - runningToday(l, st) - st.upkeep : 0);
function MoneyToday({ l, st }: { l: LineInput; st: LineStats }) {
  const d = today.value.get(l.id);
  // T-2536: no figure before the first profit with trains (the upkeep alone is not a profit)
  if (!d || !profitKnown(l)) return <span class="val w-linemoney" />;
  const run = runningToday(l, st), net = d.dayFares - run - st.upkeep;
  return (
    <span class="val w-linemoney" data-tip={lineMoneyTip(d.dayFares, run, st.upkeep, st.bus?.ferry ? "boats" : !!st.bus, dayNote(l, st))}>
      <Money dollars={net} />
    </span>
  );
}

/** T-472, T-1677, T-1684, T-2236: a line's trains now, coloured by its load past the rated load in
 * the period now, graded over its stretches (T-2279) from `v`, its own kind of day's numbers, and
 * hovered with that day's figures for the level now as the line panel's dot is (`crowdLevel`). */
function TrainsNow({ line, n, period, at, bus, ferry, v, lev, stops, flip }: { line: string; n: number; period: string; at: number; bus?: boolean; ferry?: boolean; v?: Float64Array; lev: Demand; stops: number; flip: boolean }) {
  const c = !bus && v && n > 0 ? crowdLevel(v, lev, stops, flip) : null;
  const d = today.value.get(line);
  const load = d && (v ? periodLoad(v, at, stops) ?? d.crowdLoad : d.crowdLoad);
  const cls = !bus && d && n > 0 ? crowdClass(load!) : "";
  const tip = bus ? t("linesTab.vehiclesTip", { period, vehicles: ferry ? boatsText(n) : busesText(n) }) : c?.tip;
  return <span class={"val w-trains " + (cls || "muted")} data-tip={tip}>{n}</span>;
}

/** T-1791: a group's riders, money and trains, each line's added up; hovers list them by line. */
function GroupFigures({ ls, w, trains, period }: { ls: LineInput[]; w: WorldView; trains: (id: string) => number; period: string }) {
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
      {pending ? <PendingRiders k={pending} text={any ? count(riders) : ""} />
        : <span class="val w-riders" data-tip={any ? t("linesTab.groupRidersTip", { count: count(riders), lines: by((l) => count(tod.get(l.id)?.riders ?? 0)) }) : undefined}>{any ? count(riders) : ""}</span>}
      {anyProfit ? <span class="val w-linemoney" data-tip={t("linesTab.groupProfitTip", { money: money(net), lines: run.filter((l) => profitKnown(l)).map((l) => `${l.name}: ${money(lineNet(tod.get(l.id), l, w.lineStats[l.id]))}`).join("\n") })}><Money dollars={net} /></span> : <span class="val w-linemoney" />}
      <span class="val w-trains" data-tip={t("linesTab.groupTrainsTip", { period, count: n, lines: by((l) => String(trains(l.id))) })}>{n}</span>
    </>
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
  /** a drag just ended: the click that follows it selects nothing */
  const dragged = useRef(false);
  useToday(); // T-446: riders in the past 24 hours, kept fresh while the tab is open
  void accountVersion.value; // re-sorted each game hour; the counts themselves move on their own
  void todayLoaded.value;
  const w = listWorld.value;
  // T-1677, T-2279: every running train line's crowding by period on its own kind of day, for the
  // trains cells, asked again when the network or the boarding account changes: one ask per kind of
  // day, a fixed six so the hooks hold
  const crowdLines = (w?.save.lines ?? []).filter((l) => w!.lineStats[l.id]?.status === "running" && !w!.lineStats[l.id].bus);
  const acct = accountVersion.value;
  const kindOf = (l: LineInput) => kindIn(lineCountry(l.stops), Math.floor(lineSeconds(w!.lineStats[l.id]) / 86400));
  const byKind = DAY_KINDS.map((k) => crowdLines.filter((l) => kindOf(l) === k).map((l) => l.num));
  const kindCrowd = byKind.map((nums, k) => useClockQuery(nums.length ? { what: "linesPeriods", lines: nums, day: k } : null, [w?.version, acct, nums.join(",")], (d) => d));
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
  const trains = (id: string) => (w.lineStats[id]?.status === "running" ? w.lineStats[id].trainsNeeded[levOf(id)] : 0);
  // T-1813: while the player's hand order holds, no sort
  const hand = handOrder.value.lines;
  const key = hand.length ? null : lineSort.value.key, dir = lineSort.value.dir;
  // T-332: only the viewed place unless the world is chosen
  const sc = scopedLists(w.save.stations, w.save.lines, (k, id) => (k === "line" && (sel?.kind === "line" || sel?.kind === "train") && sel.line === id), key === "near");
  // T-670: on the whole world, a city column once lines start in two or more cities (a line's city is where its first stop is)
  const cityOf = new Map<string, string>();
  if (sc.zone === null) {
    const at = stationsById(w);
    for (const l of sc.lines) {
      const s = at.get(l.stops[0]), z = s ? stationZone(s) : null;
      cityOf.set(l.id, z ? cityById(z).name : "");
    }
  }
  const showCity = new Set([...cityOf.values()].filter(Boolean)).size >= 2;
  // T-1791: a group is one row, sorted by its lines' figures added up; it shows, with all its
  // lines, when any of them is in the viewed place
  const groups = lineGroups(w.save.lines);
  const groupOf = new Map<string, number>();
  for (const [g, ls] of groups) for (const l of ls) groupOf.set(l.id, g);
  type Item = { ls: LineInput[]; g: number; name: string };
  const itemsOf = (lines: readonly LineInput[]) => {
    const out: Item[] = [];
    const placed = new Set<number>();
    for (const l of lines) {
      const g = groupOf.get(l.id);
      if (g === undefined) out.push({ ls: [l], g: 0, name: l.name });
      else if (!placed.has(g)) {
        placed.add(g);
        out.push({ ls: groups.get(g)!, g, name: groupName(groups.get(g)!) });
      }
    }
    return out;
  };
  const shownIds = new Set(sc.lines.map((l) => l.id));
  // every line's item, the viewed place's or not, so a hand order made here keeps the others' places
  const allItems = itemsOf(w.save.lines);
  const sum = (ls: LineInput[], f: (l: LineInput) => number) => ls.reduce((n, l) => n + f(l), 0);
  const riders = (ls: LineInput[]) => sum(ls, (l) => tod.get(l.id)?.riders ?? 0);
  const net = (ls: LineInput[]) => sum(ls, (l) => lineNet(tod.get(l.id), l, w.lineStats[l.id]));
  const near = (ls: LineInput[]) => ls.reduce<number | undefined>((m, l) => { const d = sc.lineDist.get(l.id); return d === undefined ? m : m === undefined ? d : Math.min(m, d); }, undefined);
  const order = (a: Item, b: Item) =>
    (key === "city" ? natural(cityOf.get(a.ls[0].id) ?? "", cityOf.get(b.ls[0].id) ?? "") * dir
      : key === "riders" ? (riders(a.ls) - riders(b.ls)) * dir
      : key === "money" ? (net(a.ls) - net(b.ls)) * dir
      : key === "trains" ? (sum(a.ls, (l) => trains(l.id)) - sum(b.ls, (l) => trains(l.id))) * dir
      : key === "near" ? byNear(near(a.ls), near(b.ls), dir) : 0)
    || natural(a.name, b.name) * (key === "name" ? dir : 1) || a.ls[0].num - b.ls[0].num;
  const one = (l: LineInput): Item => ({ ls: [l], g: 0, name: l.name });
  /** a group's lines in the order shown: the hand order, else the tab's sort */
  const membersOf = (it: Item) => (hand.length ? byHand(it.ls, (l) => [l.id], hand) : [...it.ls].sort((a, b) => order(one(a), one(b))));
  const sorted = hand.length ? byHand([...allItems].sort(order), (it) => it.ls.map((l) => l.id), hand) : [...allItems].sort(order);
  const items = sorted.filter((it) => it.ls.some((l) => shownIds.has(l.id)));
  /** every line in the order shown, the hidden ones included */
  const fullOrder = () => sorted.flatMap((it) => membersOf(it).map((l) => l.id));
  const closed = closedGroups.value;
  const periodOf = (l: LineInput) => PERIODS[periodAt(lineSeconds(w.lineStats[l.id]))]?.name ?? "";

  // T-1791, T-1813: drag a line's row onto the middle of another to group the two, or into a gap
  // between rows to put it there, out of its group unless the gap is inside it (or another open
  // group, which it then joins); a group's row moves the whole group, between rows only
  type Row = { key: string; it: Item; line?: LineInput; member: number };
  const rowList: Row[] = items.flatMap((it): Row[] => it.g === 0 ? [{ key: "l:" + it.ls[0].id, it, line: it.ls[0], member: 0 }]
    : [{ key: "g:" + it.g, it, member: 0 }, ...(!closed.has(it.g) ? membersOf(it).map((l) => ({ key: "l:" + l.id, it, line: l, member: it.g })) : [])]);
  /** What a drop does: `with` the line or group dropped onto, or `at` the gap (a row index) */
  type Drop = { with: Row } | { at: number };
  const dropAt = (d: Drag): Drop | null => {
    const i = rowList.findIndex((r) => r.key === d.over);
    if (i < 0) return null;
    if (d.part === "on") return { with: rowList[i] };
    return { at: d.part === "before" ? i : i + 1 };
  };
  /** the gap's group (0 for none) for the dragged line, and the line it goes before or after */
  const gapOf = (at: number, from: number): { g: number; before?: string; after?: string } => {
    const a = rowList[at - 1], b = rowList[at];
    if (b?.member) return { g: b.member, before: b.line!.id };
    if (a?.member && from === a.member) return { g: a.member, after: a.line!.id };
    if (!b) return { g: 0 };
    return { g: 0, before: membersOf(b.it)[0].id };
  };
  /** what the drag would change, or null for nothing */
  const plan = (d: Drag): { join?: number | "new"; with?: string; leave?: boolean; order?: string[] } | null => {
    const to = dropAt(d);
    if (!to) return null;
    const grp = d.id.startsWith("g:") ? Number(d.id.slice(2)) : 0;
    const from = grp || (groupOf.get(d.id) ?? 0);
    if ("with" in to) {
      if (grp) return null;
      const r = to.with, g = r.it.g;
      if (r.line?.id === d.id || (g !== 0 && g === from)) return null;
      return g ? { join: g } : { join: "new", with: r.line!.id };
    }
    const moved = grp ? membersOf(sorted.find((it) => it.g === grp)!).map((l) => l.id) : [d.id];
    const gap = gapOf(to.at, grp ? -1 : from);
    let order = fullOrder().filter((id) => !moved.includes(id));
    const anchor = gap.before ?? gap.after;
    if (anchor && moved.includes(anchor)) return null; // onto its own place
    const i = anchor ? order.indexOf(anchor) + (gap.after ? 1 : 0) : order.length;
    order = [...order.slice(0, i), ...moved, ...order.slice(i)];
    const same = order.join() === fullOrder().join();
    // a group's row only moves the group (T-1855: it once sent a leave for "g:N", which became line 0)
    const g = grp ? 0 : gap.g;
    const out: { order?: string[]; join?: number; leave?: boolean } = { order: same ? undefined : order, ...(!grp && g !== from ? (g ? { join: g } : { leave: true }) : {}) };
    return out.order || out.join || out.leave ? out : null;
  };
  const dropLine = (d: Drag) => {
    const p = plan(d);
    if (!p) return;
    const num = Number(d.id);
    if (p.order) setHandOrder("lines", p.order);
    if (!Number.isInteger(num)) return; // a group's row: no line to group or ungroup
    if (p.join === "new") void edit({ op: "lineGroup", lines: [Number(p.with), num], group: "new" });
    else if (p.join) void edit({ op: "lineGroup", lines: [num], group: p.join });
    else if (p.leave) void edit({ op: "lineGroup", lines: [num], group: 0 });
  };
  const pointerDown = (e: PointerEvent, id: string) => {
    if (e.button !== 0 || e.pointerType === "touch") return;
    if ((e.target as Element | null)?.closest?.("button, input")) return;
    const d: Drag = { id, x0: e.clientX, y0: e.clientY, x: e.clientX, y: e.clientY, on: false, over: null, part: "on" };
    const group = id.startsWith("g:");
    const move = (m: PointerEvent) => {
      if (!d.on && Math.hypot(m.clientX - d.x0, m.clientY - d.y0) < DRAG_PX) return;
      d.on = true;
      d.x = m.clientX; d.y = m.clientY;
      const row = document.elementFromPoint(m.clientX, m.clientY)?.closest("[data-drop]") as HTMLElement | null;
      d.over = row?.dataset.drop ?? null;
      if (row) {
        // the middle third groups, the rest is the gap above or below (a group's row has no middle)
        const b = row.getBoundingClientRect(), f = (m.clientY - b.top) / Math.max(1, b.height);
        d.part = group ? (f < 0.5 ? "before" : "after") : f < 1 / 3 ? "before" : f > 2 / 3 ? "after" : "on";
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
  if (drag?.on) {
    const p = plan(drag), to = p && dropAt(drag);
    if (to && "with" in to) {
      marks.set(to.with.key, " drop-on");
      if (to.with.member) marks.set("g:" + to.with.member, " drop-on");
    } else if (to) marks.set(drag.over!, drag.part === "before" ? " drop-before" : " drop-after");
  }
  const pick = (id: string) => { if (!dragged.current) selection.value = { kind: "line", line: id }; };
  const lineRow = (l: LineInput, inGroup: boolean) => {
    const st = w.lineStats[l.id];
    const on = (sel?.kind === "line" || sel?.kind === "train") && sel.line === l.id;
    const drop = "l:" + l.id;
    return (
      <li class={(on ? "on" : "") + (inGroup ? " in-group" : "") + (marks.get(drop) ?? "") + (drag?.on && drag.id === l.id ? " dragged" : "")}
        data-drop={drop} onPointerDown={(e) => pointerDown(e, l.id)} onClick={() => pick(l.id)} onDblClick={() => locateLine(l.stops)}>
        <Chip line={l} />
        <span class="grow line-name">{l.name}</span>
        {showCity && <span class="val muted w-city" data-tip={cityOf.get(l.id) ? t("linesTab.cityTip", { city: cityOf.get(l.id) }) : undefined}>{cityOf.get(l.id)}</span>}
        {st.status !== "running" ? (
          <span class="val muted">{statusText(st.status)}</span>
        ) : (
          <>
            <RidersToday l={l} />
            <MoneyToday l={l} st={st} />
            <TrainsNow line={l.id} n={st.trainsNeeded[levOf(l.id)]} period={periodOf(l)} at={periodAt(lineSeconds(st))} bus={!!st.bus} ferry={!!st.bus?.ferry} v={crowdOf(l)} lev={demandAt(lineSeconds(st))} stops={l.stops.length} flip={lineReversed(l, w, !!st.loop)} />
          </>
        )}
        {key === "near" && <Near m={sc.lineDist.get(l.id)} />}
      </li>
    );
  };
  const groupRows = (it: Item) => {
    const isOpen = !closed.has(it.g), drop = "g:" + it.g;
    const on = !isOpen && sel?.kind === "group" && sel.group === it.g;
    const ls = isOpen ? membersOf(it) : [];
    return (
      <>
        <li class={"group-row" + (on ? " on" : "") + (marks.get(drop) ?? "") + (drag?.on && drag.id === drop ? " dragged" : "")} data-drop={drop} onPointerDown={(e) => pointerDown(e, drop)}
          onClick={() => { if (!dragged.current) selection.value = { kind: "group", group: it.g }; }} onDblClick={() => locateLine(it.ls.flatMap((l) => l.stops))}>
          <button class={"btn icon mini chevron" + (isOpen ? " open" : "")} data-tip={isOpen ? t("linesTab.closeGroup") : t("linesTab.openGroup")} aria-label={isOpen ? t("linesTab.closeGroup") : t("linesTab.openGroup")}
            aria-expanded={isOpen} onClick={(e) => { e.stopPropagation(); toggleGroup(it.g); }} onDblClick={(e) => e.stopPropagation()}><Icon.chevron /></button>
          <GroupPill ls={it.ls} tip={it.ls.map((l) => l.name).join("\n")} />
          {renaming === it.g ? (
            <input class="name-edit grow" value={it.name} aria-label={t("linesTab.groupName")} ref={(el) => { if (el && document.activeElement !== el) { el.focus(); el.select(); } }}
              onClick={(e) => e.stopPropagation()} onDblClick={(e) => e.stopPropagation()} onPointerDown={(e) => e.stopPropagation()}
              onBlur={(e) => renameGroup(it, e.currentTarget.value)} onKeyDown={(e) => { if (e.key === "Enter") e.currentTarget.blur(); if (e.key === "Escape") setRenaming(null); }} />
          ) : (
            <span class="grow line-name" onDblClick={(e) => { e.stopPropagation(); setRenaming(it.g); }}>{it.name}</span>
          )}
          {showCity && <span class="val muted w-city">{cityOf.get(it.ls[0].id)}</span>}
          <GroupFigures ls={it.ls} w={w} trains={trains} period={periodOf(it.ls[0])} />
          {key === "near" && <Near m={near(it.ls)} />}
        </li>
        {ls.map((l) => lineRow(l, true))}
      </>
    );
  };
  const renameGroup = (it: Item, typed: string) => {
    setRenaming(null);
    const n = typed.trim(), name = n === it.ls[0].name ? "" : n;
    if (n && name !== (it.ls[0].groupName ?? "")) void edit({ op: "groupName", group: it.g, name });
  };
  const ghost = drag?.on ? w.save.lines.find((l) => l.id === drag.id) : undefined;
  const ghostGroup = drag?.on && drag.id.startsWith("g:") ? items.find((it) => it.g === Number(drag.id.slice(2))) : undefined;
  return (
    <>
      <div class="line-list-actions">
        <span style={{ display: "flex", gap: "6px" }}>
          <button class="btn text" data-guide="line" data-tip={t("linesTab.newLineTip", { key: "N" })} onClick={newLine} disabled={creating}><Icon.plus />{t("linesTab.newLine")}<span class="key muted">N</span></button>
          <button class="btn icon new-bus" data-tip={t("linesTab.newBusTip", { key: "U" })} aria-label={t("linesTab.newBus")} onClick={() => newBusLine()}><Icon.bus /></button>
          <button class="btn icon new-ferry" data-tip={t("linesTab.newFerryTip", { key: "O" })} aria-label={t("linesTab.newFerry")} onClick={() => newFerryLine()}><Icon.ferry /></button>
        </span>
        <ScopeToggle />
      </div>
      {items.length === 0 && <p class="muted">{w.save.lines.length === 0 ? t("linesTab.none") : t("linesTab.noneHere")}</p>}
      <div class={"lines-box" + (showCity ? " with-city" : "") + (key === "near" ? " near-on" : "")}>
      {w.save.lines.length > 0 && <SortHead sort={lineSort} hand={hand.length ? () => setHandOrder("lines", []) : undefined} cols={[
        { key: "name", label: t("linesTab.colName"), tip: t("linesTab.sortName"), first: 1 },
        ...(showCity ? [{ key: "city" as const, label: t("linesTab.colCity"), tip: t("linesTab.sortCity"), first: 1 as const, cls: "w-city", left: true }] : []),
        { key: "riders", label: t("linesTab.colRiders"), tip: t("linesTab.sortRiders"), first: -1, cls: "w-riders" },
        { key: "money", label: t("linesTab.colMoney"), tip: t("linesTab.sortMoney"), first: -1, cls: "w-linemoney" },
        { key: "trains", label: t("linesTab.colTrains"), tip: t("linesTab.sortTrains"), first: -1, cls: "w-trains" },
        { key: "near", label: t("linesTab.colNear"), tip: t("linesTab.sortNear"), first: 1, cls: key === "near" ? "w-near" : "near-off" },
      ]} />}
      <ul class={"rows pick" + (drag?.on ? " dragging" : "")}>
        {items.map((it) => (it.g ? groupRows(it) : lineRow(it.ls[0], false)))}
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
          <GroupPill ls={ghostGroup.ls} /><span>{ghostGroup.name}</span>
        </div>
      )}
      <div class="pane-foot">
        <TotalToday />
      </div>
    </>
  );
}
