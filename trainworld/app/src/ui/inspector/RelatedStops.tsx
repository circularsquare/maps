// The line panel's stops drawn with the lines related to the line (T-1168, notes/T-1168.md; T-1391):
// the selected line down the middle, lines beside it in thin columns that turn out where they
// leave, stop pills over every column stopping at a row, crossing lines as bars. Layout in
// game/relatedLayout.ts; the selected line's own rows come from the line panel (`lRow`).

import type { ComponentChildren, RefObject } from "preact";
import { useEffect, useLayoutEffect, useRef, useState } from "preact/hooks";
import { decodeRelated, type RelatedGroup } from "../../game/relatedLines";
import { BAR, LINE, ROUND_MAX, TALL_MAX, barOffset, pillHeight, relatedLayout } from "../../game/relatedLayout";
import type { LineInput } from "../../game/types";
import { selection } from "../../game/ui";
import { t } from "../../i18n";
import { linesByNum, stationNames, world } from "../../game/world";
import { useClockQuery } from "../useClockQuery";

/** The other lines at a row whose bullets show after its name, in the order they show. */
export type RowTransfers = (stop: number, drawn: ReadonlySet<number>) => LineInput[];

/** T-1391: a stop row whose name, time and bullets do not all fit drops its time (`data-tight`),
 * measured after every render and when the list's width changes. */
export function useTightTimes(ref: RefObject<HTMLElement | null>) {
  const fit = () => {
    const el = ref.current;
    if (!el) return;
    const rows = Array.from(el.querySelectorAll<HTMLElement>(":scope > li"));
    for (const li of rows) delete li.dataset.tight;
    const tight = rows.filter((li) => {
      const g = li.querySelector<HTMLElement>(".grow");
      return !!g && !!g.querySelector(".t") && g.scrollWidth > g.clientWidth + 0.5;
    });
    for (const li of tight) li.dataset.tight = "";
  };
  useLayoutEffect(fit);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver(() => fit());
    ro.observe(el);
    return () => ro.disconnect();
  }, [ref.current]);
}

export function RelatedStops({ l, stopS, colour, loop, flip, lRow, after, adder, onAdd, transfers, rows: turnRows }: {
  l: LineInput;
  /** T-3040: a turned line is drawn as one row per station: each row's first stop index on the line; the rows' stops
   * stand for every pass of the station, and what the other lines meet on the way back is left out */
  rows?: { i: number; occ: number[]; lead?: boolean }[];
  stopS: number[];
  colour: string;
  /** the line is a loop drawn once round */
  loop: boolean;
  /** the line drawn last stop first (T-1894, north up) */
  flip: boolean;
  /** while stops are edited: the station the line passes at a side line's stop (its node), which a
   * click on the row makes a stop (`onAdd`), else null; without it side lines' stops the line passes
   * get no row (T-1894) */
  adder?: (station: string) => number | null;
  onAdd?: (node: number) => void;
  /** the selected line's stop `i` as the plain list draws it, without the bullets of `drawn`
   * lines, with those of `marks` after the crossing lines' */
  lRow: (i: number, drawn: ReadonlySet<number>, marks: LineInput[]) => ComponentChildren;
  /** rows after the selected line's stop `i` that are not stops (the stations it passes while
   * its stops are edited) */
  after?: (i: number) => ComponentChildren;
  transfers: RowTransfers;
}) {
  const w = world.value!;
  const answer = useClockQuery({ what: "lineRelated", line: l.num }, [l.num, w.version], (d) => ({ line: l.num, groups: decodeRelated(d) }));
  const rowOf = new Map<number, number>();
  turnRows?.forEach((r, k) => r.occ.forEach((j) => rowOf.set(j, k)));
  const lastS = turnRows ? stopS[turnRows[turnRows.length - 1].i] : Infinity;
  // T-3050: rows listed before the walk's first stop (a loop above the stem) lie a lap earlier along the line
  const lap = stopS[stopS.length - 1] ?? 0;
  const leadS = turnRows?.find((r) => r.lead) ? stopS[turnRows.find((r) => r.lead)!.i] : Infinity;
  const shift = (s: number) => (s <= lastS + 1 ? s : s >= leadS - 1 ? s - lap : null);
  const groups: RelatedGroup[] = (answer?.line === l.num ? answer.groups : []).flatMap((g) => {
    if (!turnRows) return [g];
    const s0 = shift(g.sL0);
    if (s0 === null) return [];
    const back = s0 !== g.sL0 ? -lap : 0;
    const meetings = g.meetings.flatMap((m) => { const sL = shift(m.sL); return sL === null ? [] : [{ ...m, sL, stopL: rowOf.get(m.stopL) ?? -1 }]; });
    return [{ ...g, sL0: s0, sL1: g.sL1 + back, meetings, own: g.own.flatMap((o) => { const sL = shift(o.sL); return sL === null ? [] : [{ ...o, sL }]; }), gaps: g.gaps.map((x) => ({ ...x, sL0: x.sL0 + back, sL1: x.sL1 + back })) }];
  });
  const nStops = turnRows ? turnRows.length : l.stops.length;
  const ul = useRef<HTMLUListElement>(null);
  useTightTimes(ul);
  // the rows' centres in the list, and the room kept above and below it for ends past the rows
  const [geo, setGeo] = useState<{ ys: number[]; width: number; pad: [number, number] }>({ ys: [], width: 340, pad: [0, 0] });
  const byNum = linesByNum(w), names = stationNames(w);
  const stopName = (line: number, stop: number) => names.get(byNum.get(line)?.stops[line === l.num && turnRows ? turnRows[stop]?.i ?? -1 : stop] ?? "") ?? "";
  const addAt = (line: number, stop: number) => { const s = byNum.get(line)?.stops[stop]; return s && adder ? adder(s) : null; };
  const lay = relatedLayout(groups, {
    line: l.num, stopS: turnRows ? turnRows.map((r) => stopS[r.i] - (r.lead ? lap : 0)) : stopS, width: geo.width, loop, flip, sideStop: (line, stop) => addAt(line, stop) !== null,
    rowY: (i) => geo.ys[i] ?? i * 25 + 12.5,
    lineName: (n) => byNum.get(n)?.name ?? "",
    stopName,
  });
  const listH = geo.ys.length ? geo.ys[geo.ys.length - 1] + 13 : lay.rows.length ? lay.rows[lay.rows.length - 1].y + 13 : 0;
  const pad: [number, number] = [Math.ceil(Math.max(0, -lay.extent[0])), Math.ceil(Math.max(0, lay.extent[1] - listH))];
  // the rows' centres as laid out, for the drawing (a row is 25 px, 24 on a phone); rows that are
  // not the drawing's (stations passed, while stops are edited) are skipped
  useLayoutEffect(() => {
    const el = ul.current;
    if (!el) return;
    const ys = Array.from(el.querySelectorAll<HTMLElement>(":scope > li[data-r]"), (c) => c.offsetTop - el.offsetTop + c.offsetHeight / 2);
    const width = el.parentElement?.clientWidth ?? 340;
    const same = width === geo.width && ys.length === geo.ys.length && ys.every((y, i) => Math.abs(y - geo.ys[i]) <= 0.5) && pad[0] === geo.pad[0] && pad[1] === geo.pad[1];
    if (!same) setGeo({ ys, width, pad });
  });
  const pick = (n: number) => { const o = byNum.get(n); if (o) selection.value = { kind: "line", line: o.id }; };
  const oneWay = w.lineStats[l.id]?.oneWayAt ?? {}, halfId = `rs-half-${l.num}`;

  // crossing bars: the bullets after a row's name (all of them, up to TALL_MAX, where more than two
  // stretch the row's mark, T-1786), or the first two lines crossing a side line's row
  const bars = lay.rows.map((r) => {
    const drawn = new Set(r.drawn);
    const lines = r.lStop >= 0 ? transfers(r.lStop, drawn).map((o) => o.num) : r.cross.slice(0, ROUND_MAX);
    return lines.slice(0, lines.length > ROUND_MAX ? TALL_MAX : ROUND_MAX);
  });
  const W = lay.width;
  const rows: ComponentChildren[] = [];
  const n = nStops;
  lay.rows.forEach((r) => {
    // the stations passed between two stops go between their rows, whichever way up
    if (after && (flip ? r.lStop >= 0 && r.lStop < n - 1 : r.lStop > 0)) rows.push(after(flip ? r.lStop : r.lStop - 1));
    if (r.lStop >= 0) { rows.push(lRow(r.lStop, new Set(r.drawn), r.marks.map((n) => byNum.get(n)).filter((o) => o !== undefined))); return; }
    const add = addAt(r.ref.line, r.ref.stop), here = stopName(r.ref.line, r.ref.stop);
    rows.push(
      <li class="side" data-r>
        {add !== null && onAdd && <span class="stop-hit add" data-tip={t("relatedStops.add", { name: here })} onClick={() => onAdd(add)} />}
        <span class="grow">
          <span class="stop-name muted" onClick={() => { const s = byNum.get(r.ref.line)?.stops[r.ref.stop]; if (s) selection.value = { kind: "station", station: s }; }}>
            {here}
          </span>
        </span>
      </li>
    );
  });
  return (
    <div class="related" style={{ "--c": colour, "--gw": W + "px", "--spine": lay.spineX + "px", paddingTop: geo.pad[0] + "px", paddingBottom: geo.pad[1] + "px" }}>
      <svg class="related-draw" width={W} height={listH + geo.pad[0] + geo.pad[1]}>
        <defs><linearGradient id={halfId}><stop offset="50%" stop-color={colour} /><stop offset="50%" stop-color="#fff" /></linearGradient></defs>
        <g transform={`translate(0 ${geo.pad[0]})`}>
        {lay.rows.map((r, i) => bars[i].map((n, k) => (
          <line x1={1} x2={W - 1} y1={r.y + barOffset(k, bars[i].length)} y2={r.y + barOffset(k, bars[i].length)}
            stroke={byNum.get(n)?.colour} stroke-width={LINE} onClick={() => pick(n)} />
        )))}
        {lay.columns.map((c) => {
          const stroke = byNum.get(c.line)?.colour;
          return (
            <g class="link" onClick={() => pick(c.line)}>
              {[c.d, ...c.ends.map((e) => e.d)].map((d) => <path d={d} fill="none" stroke="var(--bg)" stroke-width={LINE + 3} />)}
              <path d={c.d} fill="none" stroke={stroke} stroke-width={LINE} />
              {c.ends.map((e) => <path d={e.d} fill="none" stroke={stroke} stroke-width={LINE} />)}
            </g>
          );
        })}
        {lay.rows.length > 0 && <line x1={lay.spineX} x2={lay.spineX} y1={lay.rows[lay.spine[0]].y} y2={lay.rows[lay.spine[1]].y} stroke={colour} stroke-width={BAR} />}
        {lay.loopEnds.map((d) => <path d={d} fill="none" stroke={colour} stroke-width={BAR} />)}
        {lay.pills.map((p) => {
          const r = lay.rows[p.row], n = bars[p.row].length, h = pillHeight(n, p.big), w = Math.min(h, 10);
          // a stop the line serves one way (T-3041, T-3122, T-3332): half filled, as the plain list's dot
          const one = r.lStop >= 0 && !p.passed && oneWay[l.stops[turnRows ? turnRows[r.lStop]?.i ?? -1 : r.lStop] ?? ""] !== undefined;
          return <rect class={p.passed ? "pill passed" : "pill"} style={one ? { fill: `url(#${halfId})` } : undefined} x={p.x0 - w / 2 - 1} y={r.y - h / 2 - 1} width={p.x1 - p.x0 + w + 2} height={h + 2} rx={w / 2 + 1} data-tip={p.tip} />;
        })}
        </g>
      </svg>
      <ul class="stops related-rows" ref={ul}>
        {rows}
      </ul>
    </div>
  );
}
