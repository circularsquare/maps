// The Stations tab: station ridership setting and the list of stations and complexes.

import { signal, untracked } from "@preact/signals";
import { demandView } from "../game/demand";
import { stationUse, useFigure } from "../game/stationUse";
import { useLayoutEffect, useRef, useState } from "preact/hooks";
import { count } from "../game/format";
import { type LineInput, type StationInput } from "../game/types";
import { complexes } from "../game/complexView";
import { membersOf } from "../game/complexes";
import { leadOf } from "../game/complexLead";
import { edit } from "../game/world";
import { display, selection, shownDayPhrase, stationSort, uiScale, type StationUseMode } from "../game/ui";
import { listWorld, scopedLists } from "../game/listScope";
import { accountVersion } from "../game/boarding";
import { stationsToday, todayLoaded, useToday } from "../game/today";
import { Chip, SortHead } from "./widgets";
import { locateStation } from "./locate";
import { byNear, natural, Near, ScopeToggle } from "./listParts";
import { byHand, handOrder, setHandOrder } from "../game/handOrder";
import { t } from "../i18n";

/** how far a row is dragged before it counts as a drag, not a click (px) */
const DRAG_PX = 6;
/** T-1813: a drag of a station row: its first member, where it started, the row it is over and
 * which half of it */
interface Drag { id: string; x0: number; y0: number; x: number; y: number; on: boolean; over: string | null; after: boolean }

/** T-905: a station's or complex's riders: under the map's Origins or Transfers, the circle's own
 * figure (the demand solve's, game/stationUse.ts); otherwise its boardings in the past 24 hours (T-446). */
function stationRiders(ids: string[], mode: StationUseMode): number | null {
  const split = mode === "origins" || mode === "transfers";
  const by = stationsToday.value, dv = demandView.value;
  if (split) void stationUse.value;
  let n: number | null = null;
  for (const id of ids) {
    const v = split ? useFigure(dv, mode, id) : by.get(id);
    if (v !== undefined) n = (n ?? 0) + v;
  }
  return n;
}
function stationRidersTip(n: number, mode: StationUseMode): string {
  return t("stationsTab.ridersTip", { mode, count: count(n), when: shownDayPhrase.value });
}
function StationToday({ ids }: { ids: string[] }) {
  const mode = display.value.stationUse, n = stationRiders(ids, mode);
  return <span class="val w-num" data-tip={n !== null ? stationRidersTip(n, mode) : undefined}>{n !== null ? count(n) : ""}</span>;
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

/** T-1200: rows drawn at first, and more each time the list's end scrolls into view. New York's
 * 1,000-odd rows at once froze the page 0.45 s on opening the tab (1.8 s on a phone-speed CPU). */
const ROWS_FIRST = 80, ROWS_MORE = 160;

/** An empty row at the list's end that calls `more` when it scrolls into view. */
function MoreRows({ more }: { more: () => void }) {
  const ref = useRef<HTMLLIElement>(null);
  const call = useRef(more);
  call.current = more;
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el || typeof IntersectionObserver === "undefined") return;
    const io = new IntersectionObserver((es) => { if (es.some((e) => e.isIntersecting)) call.current(); }, { rootMargin: "400px 0px" }); // i18n-ignore (CSS)
    io.observe(el);
    return () => io.disconnect();
  }, []);
  return <li ref={ref} class="rows-more" aria-hidden="true" />;
}

/** A Stations tab row: a station complex (T-530), or a station on its own. */
interface StationRow {
  /** the member a click selects, whose name the row takes (`leadOf`: a typed name, then most lines) */
  lead: StationInput;
  members: string[];
  lines: LineInput[];
  built: boolean;
  dist: number | undefined;
}

/** The Stations tab's rows: one per complex with a member in `stations` (T-530, Anita: Times Sq's
 * platforms were rows of one bullet each), its lines and numbers over the whole complex. */
function stationRows(stations: readonly StationInput[], all: readonly StationInput[], lines: readonly LineInput[], dist: Map<string, number>, apart = false): StationRow[] {
  const cx = complexes.value;
  const byId = new Map(all.map((s) => [s.id, s]));
  const lineIdx = new Map<string, number[]>();
  lines.forEach((l, i) => { for (const id of l.stops) { const a = lineIdx.get(id); if (!a) lineIdx.set(id, [i]); else if (a[a.length - 1] !== i) a.push(i); } });
  const seen = new Set<number | string>();
  const rows: StationRow[] = [];
  for (const s of stations) {
    const k = apart ? s.id : cx.of.get(s.id) ?? s.id;
    if (seen.has(k)) continue;
    seen.add(k);
    const members = apart ? [s.id] : membersOf(cx, s.id);
    const at = new Set<number>();
    let built = false, d: number | undefined;
    const present: StationInput[] = [];
    for (const id of members) {
      const m = byId.get(id);
      if (!m) continue;
      present.push(m);
      for (const i of lineIdx.get(id) ?? []) at.add(i);
      built ||= m.built;
      const dm = dist.get(id);
      if (dm !== undefined && (d === undefined || dm < d)) d = dm;
    }
    const lead = present.length ? leadOf(present, (id) => lineIdx.get(id)?.length ?? 0) : s;
    rows.push({ lead, members, lines: [...at].sort((a, b) => a - b).map((i) => lines[i]), built, dist: d });
  }
  return rows;
}

/** T-2788: list each station of a complex on its own row (off: one row per complex) */
const apart = signal(false);

export function StationsTab() {
  useToday(); // T-293
  const [limit, setLimit] = useState(ROWS_FIRST);
  const [renamingStation, setRenamingStation] = useState<string | null>(null);
  const [drag, setDrag] = useState<Drag | null>(null);
  /** a drag just ended: the click that follows it selects nothing */
  const dragged = useRef(false);
  void accountVersion.value; // re-sorted each game hour, as the lines
  void todayLoaded.value;
  const w = listWorld.value;
  if (!w) return null;
  const sel = selection.value;
  // T-1813: while the player's hand order holds, no sort
  const hand = handOrder.value.stations;
  const key = hand.length ? null : stationSort.value.key, dir = stationSort.value.dir;
  const d = display.value;
  const sc = scopedLists(w.save.stations, w.save.lines, (k, id) => k === "station" && sel?.kind === "station" && sel.station === id);
  // re-sorted each game hour (the past 24 hours), or as the split for the map's mode arrives
  const riders = (r: StationRow) => untracked(() => stationRiders(r.members, d.stationUse)) ?? 0;
  const split = d.stationUse === "origins" || d.stationUse === "transfers";
  if (split) void stationUse.value;
  const order = (a: StationRow, b: StationRow) =>
    (key === "riders" ? (riders(a) - riders(b)) * dir
      : key === "near" ? byNear(a.dist, b.dist, dir) : 0)
    || natural(a.lead.name, b.lead.name) * (key === "name" ? dir : 1);
  const inOrder = (rows: StationRow[]) => (hand.length ? byHand(rows.sort(order), (r) => r.members, hand) : rows.sort(order));
  const st = inOrder(stationRows(sc.stations, w.save.stations, w.save.lines, sc.stationDist, apart.value));
  // T-1813: a row dragged into the gap above or below another goes there; the order then covers
  // every station, the viewed place's or not
  const dropRow = (d: Drag) => {
    const from = st.find((r) => r.members[0] === d.id), to = st.find((r) => r.members[0] === d.over);
    if (!from || !to || from === to) return;
    const all = inOrder(stationRows(w.save.stations, w.save.stations, w.save.lines, sc.stationDist)).flatMap((r) => r.members);
    const rest = all.filter((id) => !from.members.includes(id));
    const anchor = d.after ? to.members[to.members.length - 1] : to.members[0];
    const i = rest.indexOf(anchor) + (d.after ? 1 : 0);
    const next = [...rest.slice(0, i), ...from.members, ...rest.slice(i)];
    if (next.join() !== all.join()) setHandOrder("stations", next);
  };
  const pointerDown = (e: PointerEvent, id: string) => {
    if (e.button !== 0 || e.pointerType === "touch") return;
    const d: Drag = { id, x0: e.clientX, y0: e.clientY, x: e.clientX, y: e.clientY, on: false, over: null, after: false };
    const move = (m: PointerEvent) => {
      if (!d.on && Math.hypot(m.clientX - d.x0, m.clientY - d.y0) < DRAG_PX) return;
      d.on = true;
      d.x = m.clientX; d.y = m.clientY;
      const row = document.elementFromPoint(m.clientX, m.clientY)?.closest("[data-drop]") as HTMLElement | null;
      d.over = row?.dataset.drop ?? null;
      if (row) { const b = row.getBoundingClientRect(); d.after = m.clientY > b.top + b.height / 2; }
      setDrag({ ...d });
    };
    const end = (u: PointerEvent | KeyboardEvent) => {
      removeEventListener("pointermove", move);
      removeEventListener("pointerup", end);
      removeEventListener("keydown", key);
      if (d.on) {
        dragged.current = true;
        setTimeout(() => { dragged.current = false; }, 0);
        if (u.type === "pointerup") dropRow(d);
      }
      setDrag(null);
    };
    const key = (k: KeyboardEvent) => { if (k.key === "Escape") end(k); };
    addEventListener("pointermove", move);
    addEventListener("pointerup", end);
    addEventListener("keydown", key);
  };
  const mark = (r: StationRow) => (drag?.on && drag.over === r.members[0] && drag.id !== drag.over ? (drag.after ? " drop-after" : " drop-before") : "")
    + (drag?.on && drag.id === r.members[0] ? " dragged" : "");
  const ghost = drag?.on ? st.find((r) => r.members[0] === drag.id) : undefined;
  const ridersWhat = t("stationsTab.ridersWhat", { mode: d.stationUse, when: shownDayPhrase.value });
  return (
    <>
      <div class="line-list-actions"><ScopeToggle />
        <button class={"btn" + (apart.value ? " on" : "")} aria-pressed={apart.value} onClick={() => (apart.value = !apart.value)}>{t("stationsTab.apart")}</button>
      </div>
      {st.length === 0 && <p class="muted">{w.save.stations.length === 0 ? t("stationsTab.none") : t("stationsTab.noneHere")}</p>}
      {st.length > 0 && <SortHead sort={stationSort} hand={hand.length ? () => setHandOrder("stations", []) : undefined} cols={[
        { key: "name", label: t("stationsTab.name"), tip: t("stationsTab.sortName"), first: 1 },
        { key: "riders", label: t("stationsTab.riders"), tip: t("stationsTab.sortRiders", { what: ridersWhat }), first: -1, cls: "w-num" },
        { key: "near", label: t("stationsTab.near"), tip: t("stationsTab.sortNear"), first: 1, cls: key === "near" ? "w-near" : "near-off" },
      ]} />}
      <ul class={"rows pick stations" + (drag?.on ? " dragging" : "")}>
        {st.slice(0, limit).map((r) => (
          <li class={(sel?.kind === "station" && r.members.includes(sel.station) ? "on" : "") + mark(r)} data-drop={r.members[0]} onPointerDown={(e) => pointerDown(e, r.members[0])}
            onClick={() => { if (!dragged.current) selection.value = { kind: "station", station: r.lead.id }; }} onDblClick={() => locateStation(r.lead.lng, r.lead.lat)}>
            {renamingStation === r.lead.id ? (
              <input class="name-edit grow" value={r.lead.name} aria-label={t("stationsTab.stationName")} ref={(el) => { if (el && document.activeElement !== el) { el.focus(); el.select(); } }}
                onClick={(e) => e.stopPropagation()} onDblClick={(e) => e.stopPropagation()} onPointerDown={(e) => e.stopPropagation()}
                onBlur={(e) => { const n = e.currentTarget.value.trim(); setRenamingStation(null); if (n && n !== r.lead.name) void edit({ op: "stationProps", node: r.lead.num, platform: r.lead.length, name: n }); }}
                onKeyDown={(e) => { if (e.key === "Enter") e.currentTarget.blur(); if (e.key === "Escape") setRenamingStation(null); }} />
            ) : (
              // T-2786: a triple click renames (a double click snaps the camera)
              <span class={"grow" + (r.built ? "" : " muted")}><span onClick={(e) => { if (e.detail === 3) { e.stopPropagation(); setRenamingStation(r.lead.id); } }}>{r.lead.name}</span></span>
            )}
            <Chips lines={r.lines} name={r.lead.name} />
            {r.built ? <StationToday ids={r.members} /> : <span class="val w-num">{t("stationsTab.plan")}</span>}
            {key === "near" && <Near m={r.dist} />}
          </li>
        ))}
        {st.length > limit && <MoreRows key={limit} more={() => setLimit((n) => n + ROWS_MORE)} />}
      </ul>
      {ghost && drag && (
        // the dock is zoomed by the UI scale, the pointer is not
        <div class="row-ghost" style={{ left: (drag.x + 12) / uiScale.value + "px", top: (drag.y + 8) / uiScale.value + "px" }}><span>{ghost.lead.name}</span></div>
      )}
      {st.length > 0 && (
        <div class="pane-foot">
          <span class="muted">{t("stationsTab.footer", { scoped: !!sc.zone, shown: st.length, count: st.length, total: complexes.value.members.length, what: ridersWhat })}</span>
        </div>
      )}
    </>
  );
}
