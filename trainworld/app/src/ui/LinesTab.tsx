// The Lines tab: the list of lines with their riders, money and trains now.

import { demandAt, lineSeconds } from "../game/clock";
import { PERIODS, periodAt } from "../game/periods";
import { useEffect, useRef, useState } from "preact/hooks";
import { COUNT_MS, countingRiders, openings } from "../game/openingDay";
import { createLine, newBusLine } from "../game/lineActions";
import { busesText, count, money } from "../game/format";
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
import { crowdClass, crowdLevel, crowding, PERIOD_STRIDE, runningToday, today, todayLoaded, useToday, type LineToday } from "../game/today";
import { demandView, lineFiguresIn } from "../game/demand";
import { useClockQuery } from "./useClockQuery";
import { Chip, Icon, SortHead } from "./widgets";
import { Money } from "./MoneyText";
import { locateLine } from "./locate";
import { byNear, natural, Near, ScopeToggle } from "./listParts";
import { linesPending } from "../game/lineFigures";

const STATUS: Record<string, string> = { running: "", planned: "Planned", broken: "Broken" };
/** how far a row is dragged before it counts as a drag, not a click (px) */
const DRAG_PX = 6;

/** T-446: one line's riders in the past 24 hours, its own component so only it re-renders as the count moves. */
function RidersToday({ line }: { line: string }) {
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
  if (linesPending.value.has(line)) return <span class="val w-riders muted" data-tip="Riders are being worked out for the change">constructing...</span>;
  const shown = countingRiders(o, d?.riders ?? 0);
  const text = shown !== null ? count(shown) : d ? count(d.riders) : "";
  return <span class="val w-riders" data-tip={d ? `${count(d.riders)} riders in the past 24 hours` : undefined}>{text}</span>;
}

/** T-735, T-867: a line's fares less running cost and its share of upkeep over the past 24 hours. */
const lineNet = (d: LineToday | undefined, l: LineInput, st: LineStats) => (d ? d.fares - runningToday(l, st) - st.upkeep : 0);
function MoneyToday({ l, st }: { l: LineInput; st: LineStats }) {
  const d = today.value.get(l.id);
  if (!d) return <span class="val w-linemoney" />;
  const run = runningToday(l, st), net = d.fares - run - st.upkeep;
  return (
    <span class="val w-linemoney" data-tip={[`${money(net)} net in the past 24 hours`, `Fares ${money(d.fares)}`, `Running ${st.bus ? "buses" : "trains"} ${money(-run)}`, `Upkeep ${money(-st.upkeep)}`].join("\n")}>
      <Money dollars={net} />
    </span>
  );
}

/** T-472, T-491, T-1677, T-1684: a line's trains now, coloured by its riders' wait for a train with
 * room in the period now, hovered with the weekday's figures for the level now as the line panel's
 * ring is (`crowdLevel`). */
function TrainsNow({ line, n, period, bus, v, lev }: { line: string; n: number; period: string; bus?: boolean; v?: Float64Array; lev: Demand }) {
  const c = !bus && v && n > 0 ? crowdLevel(v, lev) : null;
  const d = today.value.get(line);
  const now = d && n > 0 ? crowding(d.crowdWait, d.crowdRiders, d.crowdRide, d.crowdWaitKm, d.crowdRidersKm) : null;
  const cls = now ? crowdClass(now.score) : "";
  const tip = bus ? `${period}: ${busesText(n)} on the line` : c?.tip;
  return <span class={"val w-trains " + (cls || "muted")} data-tip={tip}>{n}</span>;
}

/** T-1791: a group's riders, money and trains, each line's added up; hovers list them by line. */
function GroupFigures({ ls, w, trains, period }: { ls: LineInput[]; w: WorldView; trains: (id: string) => number; period: string }) {
  const t = today.value;
  const run = ls.filter((l) => w.lineStats[l.id]?.status === "running");
  if (!run.length) {
    const st = ls.some((l) => w.lineStats[l.id]?.status === "planned") ? "planned" : "broken";
    return <span class="val muted">{STATUS[st]}</span>;
  }
  const by = (f: (l: LineInput) => string) => run.map((l) => `${l.name}: ${f(l)}`).join("\n");
  const pending = run.some((l) => linesPending.value.has(l.id));
  const riders = run.reduce((n, l) => n + (t.get(l.id)?.riders ?? 0), 0);
  const any = run.some((l) => t.has(l.id));
  const net = run.reduce((n, l) => n + lineNet(t.get(l.id), l, w.lineStats[l.id]), 0);
  const n = run.reduce((k, l) => k + trains(l.id), 0);
  return (
    <>
      {pending ? <span class="val w-riders muted" data-tip="Riders are being worked out for the change">constructing...</span>
        : <span class="val w-riders" data-tip={any ? `${count(riders)} riders in the past 24 hours, each line's added up\n${by((l) => count(t.get(l.id)?.riders ?? 0))}` : undefined}>{any ? count(riders) : ""}</span>}
      {any ? <span class="val w-linemoney" data-tip={`${money(net)} net in the past 24 hours\n${by((l) => money(lineNet(t.get(l.id), l, w.lineStats[l.id])))}`}><Money dollars={net} /></span> : <span class="val w-linemoney" />}
      <span class="val w-trains" data-tip={`${period}: ${n} trains\n${by((l) => String(trains(l.id)))}`}>{n}</span>
    </>
  );
}

/** A drag of a row (T-1791, T-1813): the line (or "g:" and the group), where it started, the row
 * it is over and which part of it. */
interface Drag { id: string; x0: number; y0: number; x: number; y: number; on: boolean; over: string | null; part: "before" | "on" | "after" }

function TotalToday() {
  const t = today.value;
  if (!t.size) return null;
  let n = 0;
  for (const d of t.values()) n += d.riders;
  return <span class="muted">{count(n)} riders in the past 24 hours</span>;
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
  // T-1677: every running train line's weekday crowding by period, for the trains cells' hovers,
  // asked again when the network or the boarding account changes
  const crowdNums = (w?.save.lines ?? []).filter((l) => w!.lineStats[l.id]?.status === "running" && !w!.lineStats[l.id].bus).map((l) => l.num);
  const acct = accountVersion.value;
  const crowd = useClockQuery(crowdNums.length ? { what: "linesPeriods", lines: crowdNums, day: 0 } : null, [w?.version, acct, crowdNums.join(",")], (d) => d);
  if (!w) return null;
  const crowdOf = (l: LineInput) => {
    const k = crowdNums.indexOf(l.num), block = PERIODS.length * PERIOD_STRIDE;
    return crowd && k >= 0 && crowd.length === crowdNums.length * block && lineFiguresIn(demandView.value, l.id, l.stops.length) ? crowd.subarray(k * block, (k + 1) * block) : undefined;
  };
  const sel = selection.value;
  // T-309: each line's demand level now in its own local time
  const levOf = (id: string) => DEMAND_INDEX[demandAt(lineSeconds(w.lineStats[id]))];
  const t = today.peek();
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
  const riders = (ls: LineInput[]) => sum(ls, (l) => t.get(l.id)?.riders ?? 0);
  const net = (ls: LineInput[]) => sum(ls, (l) => lineNet(t.get(l.id), l, w.lineStats[l.id]));
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
    const g = grp ? 0 : gap.g;
    const out: { order?: string[]; join?: number; leave?: boolean } = { order: same ? undefined : order, ...(g !== from ? (g ? { join: g } : { leave: true }) : {}) };
    return out.order || out.join || out.leave ? out : null;
  };
  const dropLine = (d: Drag) => {
    const p = plan(d);
    if (!p) return;
    const num = Number(d.id);
    if (p.order) setHandOrder("lines", p.order);
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
        {showCity && <span class="val muted w-city" data-tip={cityOf.get(l.id) ? `The first stop is in ${cityOf.get(l.id)}` : undefined}>{cityOf.get(l.id)}</span>}
        {st.status !== "running" ? (
          <span class="val muted">{STATUS[st.status]}</span>
        ) : (
          <>
            <RidersToday line={l.id} />
            <MoneyToday l={l} st={st} />
            <TrainsNow line={l.id} n={st.trainsNeeded[levOf(l.id)]} period={periodOf(l)} bus={!!st.bus} v={crowdOf(l)} lev={demandAt(lineSeconds(st))} />
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
          <button class={"btn icon mini chevron" + (isOpen ? " open" : "")} data-tip={isOpen ? "Close the group" : "Show the group's lines"} aria-label={isOpen ? "Close the group" : "Show the group's lines"}
            aria-expanded={isOpen} onClick={(e) => { e.stopPropagation(); toggleGroup(it.g); }} onDblClick={(e) => e.stopPropagation()}><Icon.chevron /></button>
          <GroupPill ls={it.ls} tip={it.ls.map((l) => l.name).join("\n")} />
          {renaming === it.g ? (
            <input class="name-edit grow" value={it.name} aria-label="Group name" ref={(el) => { if (el && document.activeElement !== el) { el.focus(); el.select(); } }}
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
          <button class="btn text" data-guide="4" data-tip="Make a new train line (N)" onClick={newLine} disabled={creating}><Icon.plus />New line<span class="key muted">N</span></button>
          <button class="btn icon new-bus" data-tip="Draw a new bus line (U)" aria-label="New bus line" onClick={() => newBusLine()}><Icon.bus /></button>
        </span>
        <ScopeToggle />
      </div>
      {items.length === 0 && <p class="muted">{w.save.lines.length === 0 ? "No lines yet." : "No lines here."}</p>}
      <div class={"lines-box" + (showCity ? " with-city" : "") + (key === "near" ? " near-on" : "")}>
      {w.save.lines.length > 0 && <SortHead sort={lineSort} hand={hand.length ? () => setHandOrder("lines", []) : undefined} cols={[
        { key: "name", label: "Name", tip: "Sort by name", first: 1 },
        ...(showCity ? [{ key: "city" as const, label: "City", tip: "Sort by the city of the first stop", first: 1 as const, cls: "w-city", left: true }] : []),
        { key: "riders", label: "Riders", tip: "Sort by riders in the past 24 hours", first: -1, cls: "w-riders" },
        { key: "money", label: "Money", tip: "Sort by money in the past 24 hours, fares minus running costs and upkeep", first: -1, cls: "w-linemoney" },
        { key: "trains", label: "Trains", tip: "Sort by trains needed", first: -1, cls: "w-trains" },
        { key: "near", label: "Near", tip: "Sort by distance from the center of the screen, by the nearest stop", first: 1, cls: key === "near" ? "w-near" : "near-off" },
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
