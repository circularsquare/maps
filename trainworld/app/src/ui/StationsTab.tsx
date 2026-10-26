// The Stations tab: station ridership setting and the list of stations and complexes.

import { untracked } from "@preact/signals";
import { demandView } from "../game/demand";
import { stationUse, useFigure } from "../game/stationUse";
import { useLayoutEffect, useRef, useState } from "preact/hooks";
import { count } from "../game/format";
import { type LineInput, type StationInput } from "../game/types";
import { complexes } from "../game/complexView";
import { membersOf } from "../game/complexes";
import { display, selection, shownDayPhrase, stationSort, type StationUseMode } from "../game/ui";
import { world } from "../game/world";
import { scopedLists } from "../game/listScope";
import { accountVersion } from "../game/boarding";
import { stationsToday, todayLoaded, useToday } from "../game/today";
import { Chip, SortHead } from "./widgets";
import { locateStation } from "./locate";
import { byNear, natural, Near, ScopeToggle } from "./listParts";

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
  return mode === "origins" ? `${count(n)} riders start a rail trip here ${shownDayPhrase.value}`
    : mode === "transfers" ? `${count(n)} riders change lines here ${shownDayPhrase.value}`
    : `${count(n)} boardings in the past 24 hours`;
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
    const io = new IntersectionObserver((es) => { if (es.some((e) => e.isIntersecting)) call.current(); }, { rootMargin: "400px 0px" });
    io.observe(el);
    return () => io.disconnect();
  }, []);
  return <li ref={ref} class="rows-more" aria-hidden="true" />;
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
  const [limit, setLimit] = useState(ROWS_FIRST);
  void accountVersion.value; // re-sorted each game hour, as the lines
  void todayLoaded.value;
  const w = world.value;
  if (!w) return null;
  const sel = selection.value;
  const { key, dir } = stationSort.value;
  const d = display.value;
  const sc = scopedLists(w.save.stations, w.save.lines, (k, id) => k === "station" && sel?.kind === "station" && sel.station === id);
  // re-sorted each game hour (the past 24 hours), or as the split for the map's mode arrives
  const riders = (r: StationRow) => untracked(() => stationRiders(r.members, d.stationUse)) ?? 0;
  const split = d.stationUse === "origins" || d.stationUse === "transfers";
  if (split) void stationUse.value;
  const st = stationRows(sc.stations, w.save.stations, w.save.lines, sc.stationDist).sort((a, b) =>
    (key === "riders" ? (riders(a) - riders(b)) * dir
      : key === "near" ? byNear(a.dist, b.dist, dir) : 0)
    || natural(a.lead.name, b.lead.name) * (key === "name" ? dir : 1));
  const ridersWhat = d.stationUse === "origins" ? `riders starting a rail trip ${shownDayPhrase.value}`
    : d.stationUse === "transfers" ? `riders changing lines ${shownDayPhrase.value}` : "boardings in the past 24 hours";
  return (
    <>
      <div class="line-list-actions"><ScopeToggle /></div>
      {st.length === 0 && <p class="muted">{w.save.stations.length === 0 ? "No stations yet." : "No stations here."}</p>}
      {st.length > 0 && <SortHead sort={stationSort} cols={[
        { key: "name", label: "Name", tip: "Sort by name", first: 1 },
        { key: "riders", label: "Riders", tip: `Sort by ${ridersWhat}`, first: -1, cls: "w-num" },
        { key: "near", label: "Near", tip: "Sort by distance from the centre of the screen", first: 1, cls: key === "near" ? "w-near" : "near-off" },
      ]} />}
      <ul class="rows pick stations">
        {st.slice(0, limit).map((r) => (
          <li class={sel?.kind === "station" && r.members.includes(sel.station) ? "on" : ""} onClick={() => (selection.value = { kind: "station", station: r.lead.id })} onDblClick={() => locateStation(r.lead.lng, r.lead.lat)}>
            <span class={"grow" + (r.built ? "" : " muted")}><span>{r.lead.name}</span></span>
            <Chips lines={r.lines} name={r.lead.name} />
            {r.built ? <StationToday ids={r.members} /> : <span class="val w-num">Plan</span>}
            {key === "near" && <Near m={r.dist} />}
          </li>
        ))}
        {st.length > limit && <MoreRows key={limit} more={() => setLimit((n) => n + ROWS_MORE)} />}
      </ul>
      {st.length > 0 && (
        <div class="pane-foot">
          <span class="muted">{sc.zone ? `${st.length} of ${complexes.value.members.length}` : st.length} stations, {ridersWhat}</span>
        </div>
      )}
    </>
  );
}
