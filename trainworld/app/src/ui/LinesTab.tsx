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
import { groupName, lineGroups, openGroups, toggleGroup } from "../game/lineGroups";
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

/** A drag of a line row (T-1791): the line, where it started, and the row it is over. */
interface Drag { id: string; x0: number; y0: number; x: number; y: number; on: boolean; over: string | null }

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
  const { key, dir } = lineSort.value;
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
  const items: Item[] = [];
  const placed = new Set<number>();
  for (const l of sc.lines) {
    const g = groupOf.get(l.id);
    if (g === undefined) items.push({ ls: [l], g: 0, name: l.name });
    else if (!placed.has(g)) {
      placed.add(g);
      items.push({ ls: groups.get(g)!, g, name: groupName(groups.get(g)!) });
    }
  }
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
  items.sort(order);
  const open = openGroups.value;
  const periodOf = (l: LineInput) => PERIODS[periodAt(lineSeconds(w.lineStats[l.id]))]?.name ?? "";

  // T-1791: drag a row onto another to group the two; out of its group's rows to leave it
  const dropOf = (over: string | null): { g: number; with?: string } | null => {
    if (!over) return null;
    if (over.startsWith("g:")) return { g: Number(over.slice(2)) };
    const id = over.slice(2), g = groupOf.get(id);
    return g !== undefined ? { g } : { g: 0, with: id };
  };
  const dropLine = (d: Drag) => {
    const from = groupOf.get(d.id) ?? 0, to = dropOf(d.over);
    const num = Number(d.id);
    if (to && to.g === 0 && to.with !== undefined && to.with !== d.id) void edit({ op: "lineGroup", lines: [Number(to.with), num], group: "new" });
    else if (to && to.g !== 0 && to.g !== from) void edit({ op: "lineGroup", lines: [num], group: to.g });
    else if (from !== 0 && (!to || (to.g !== from && to.with !== d.id))) void edit({ op: "lineGroup", lines: [num], group: 0 });
  };
  const pointerDown = (e: PointerEvent, l: LineInput) => {
    if (e.button !== 0 || e.pointerType === "touch") return;
    const d: Drag = { id: l.id, x0: e.clientX, y0: e.clientY, x: e.clientX, y: e.clientY, on: false, over: null };
    const move = (m: PointerEvent) => {
      if (!d.on && Math.hypot(m.clientX - d.x0, m.clientY - d.y0) < DRAG_PX) return;
      d.on = true;
      d.x = m.clientX; d.y = m.clientY;
      d.over = (document.elementFromPoint(m.clientX, m.clientY)?.closest("[data-drop]") as HTMLElement | null)?.dataset.drop ?? null;
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
  const dropTarget = (() => {
    if (!drag?.on) return null;
    const to = dropOf(drag.over), from = groupOf.get(drag.id) ?? 0;
    if (!to || to.with === drag.id || (to.g !== 0 && to.g === from)) return null;
    return drag.over;
  })();
  const pick = (id: string) => { if (!dragged.current) selection.value = { kind: "line", line: id }; };
  const lineRow = (l: LineInput, inGroup: boolean) => {
    const st = w.lineStats[l.id];
    const on = (sel?.kind === "line" || sel?.kind === "train") && sel.line === l.id;
    const drop = "l:" + l.id;
    return (
      <li class={(on ? "on" : "") + (inGroup ? " in-group" : "") + (dropTarget === drop ? " drop-on" : "") + (drag?.on && drag.id === l.id ? " dragged" : "")}
        data-drop={drop} onPointerDown={(e) => pointerDown(e, l)} onClick={() => pick(l.id)} onDblClick={() => locateLine(l.stops)}>
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
    const isOpen = open.has(it.g), drop = "g:" + it.g;
    const on = !isOpen && sel?.kind === "group" && sel.group === it.g;
    const ls = isOpen ? [...it.ls].sort((a, b) => order({ ls: [a], g: 0, name: a.name }, { ls: [b], g: 0, name: b.name })) : [];
    return (
      <>
        <li class={"group-row" + (on ? " on" : "") + (dropTarget === drop || (dropTarget && dropOf(dropTarget)?.g === it.g) ? " drop-on" : "")} data-drop={drop}
          onClick={() => { if (!dragged.current) selection.value = { kind: "group", group: it.g }; }} onDblClick={() => locateLine(it.ls.flatMap((l) => l.stops))}>
          <button class={"btn icon mini chevron" + (isOpen ? " open" : "")} data-tip={isOpen ? "Close the group" : "Show the group's lines"} aria-label={isOpen ? "Close the group" : "Show the group's lines"}
            aria-expanded={isOpen} onClick={(e) => { e.stopPropagation(); toggleGroup(it.g); }} onDblClick={(e) => e.stopPropagation()}><Icon.chevron /></button>
          <span class="group-chips">{it.ls.map((l) => <Chip line={l} />)}</span>
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
  return (
    <>
      <div class="line-list-actions">
        <span style={{ display: "flex", gap: "6px" }}>
          <button class="btn text" data-tip="Make a new train line (N)" onClick={newLine} disabled={creating}><Icon.plus />New line<span class="key muted">N</span></button>
          <button class="btn icon new-bus" data-tip="Draw a new bus line (B)" aria-label="New bus line" onClick={() => newBusLine()}><Icon.bus /></button>
        </span>
        <ScopeToggle />
      </div>
      {items.length === 0 && <p class="muted">{w.save.lines.length === 0 ? "No lines yet." : "No lines here."}</p>}
      <div class={"lines-box" + (showCity ? " with-city" : "") + (key === "near" ? " near-on" : "")}>
      {w.save.lines.length > 0 && <SortHead sort={lineSort} cols={[
        { key: "name", label: "Name", tip: "Sort by name", first: 1 },
        ...(showCity ? [{ key: "city" as const, label: "City", tip: "Sort by the city of the first stop", first: 1 as const, cls: "w-city", left: true }] : []),
        { key: "riders", label: "Riders", tip: "Sort by riders in the past 24 hours", first: -1, cls: "w-riders" },
        { key: "money", label: "Money", tip: "Sort by money in the past 24 hours, fares minus running costs and upkeep", first: -1, cls: "w-linemoney" },
        { key: "trains", label: "Trains", tip: "Sort by trains needed", first: -1, cls: "w-trains" },
        { key: "near", label: "Near", tip: "Sort by distance from the centre of the screen, by the nearest stop", first: 1, cls: key === "near" ? "w-near" : "near-off" },
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
      <div class="pane-foot">
        <TotalToday />
      </div>
    </>
  );
}
