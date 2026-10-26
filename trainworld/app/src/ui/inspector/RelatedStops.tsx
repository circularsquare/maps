// The line panel's stops drawn with the lines related to the line (T-1168, notes/T-1168.md; T-1391):
// the selected line down the middle, lines beside it in thin columns that turn out where they
// leave, stop pills over every column stopping at a row, crossing lines as bars. Layout in
// game/relatedLayout.ts; the selected line's own rows come from the line panel (`lRow`).

import type { ComponentChildren, RefObject } from "preact";
import { useEffect, useLayoutEffect, useRef, useState } from "preact/hooks";
import { decodeRelated, type RelatedGroup } from "../../game/relatedLines";
import { BAR, LINE, relatedLayout } from "../../game/relatedLayout";
import { distance } from "../../game/format";
import type { LineInput } from "../../game/types";
import { selection } from "../../game/ui";
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

export function RelatedStops({ l, stopS, colour, loop, lRow, after, transfers }: {
  l: LineInput;
  stopS: number[];
  colour: string;
  /** the line is a loop drawn once round */
  loop: boolean;
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
  const groups: RelatedGroup[] = answer?.line === l.num ? answer.groups : [];
  const ul = useRef<HTMLUListElement>(null);
  useTightTimes(ul);
  // the rows' centres in the list, and the room kept above and below it for ends past the rows
  const [geo, setGeo] = useState<{ ys: number[]; width: number; pad: [number, number] }>({ ys: [], width: 340, pad: [0, 0] });
  const byNum = linesByNum(w), names = stationNames(w);
  const stopName = (line: number, stop: number) => names.get(byNum.get(line)?.stops[stop] ?? "") ?? "";
  const lay = relatedLayout(groups, {
    line: l.num, stopS, width: geo.width, loop,
    rowY: (i) => geo.ys[i] ?? i * 25 + 12.5,
    lineName: (n) => byNum.get(n)?.name ?? "",
    stopName,
    stopCount: (n) => byNum.get(n)?.stops.length ?? 0,
    dist: distance,
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

  // crossing bars: the first two bullets after a row's name, or the lines crossing a side line's row
  const bars = lay.rows.map((r, i) => {
    const drawn = new Set(r.drawn);
    const lines = r.lStop >= 0 ? transfers(r.lStop, drawn).map((o) => o.num) : r.cross;
    return lines.slice(0, 2).map((n) => ({ n, tip: lay.crossTip(i, n) }));
  });
  const W = lay.width;
  const rows: ComponentChildren[] = [];
  lay.rows.forEach((r) => {
    if (r.lStop > 0 && after) rows.push(after(r.lStop - 1));
    rows.push(r.lStop >= 0
      ? lRow(r.lStop, new Set(r.drawn), r.marks.map((n) => byNum.get(n)).filter((o) => o !== undefined))
      : (
        <li class="side" data-r>
          <span class="grow">
            <span class="stop-name muted" onClick={() => { const s = byNum.get(r.ref.line)?.stops[r.ref.stop]; if (s) selection.value = { kind: "station", station: s }; }}>
              {stopName(r.ref.line, r.ref.stop)}
            </span>
          </span>
        </li>
      ));
  });
  return (
    <div class="related" style={{ "--gw": W + "px", "--spine": lay.spineX + "px", paddingTop: geo.pad[0] + "px", paddingBottom: geo.pad[1] + "px" }}>
      <svg class="related-draw" width={W} height={listH + geo.pad[0] + geo.pad[1]}>
        <g transform={`translate(0 ${geo.pad[0]})`}>
        {lay.rows.map((r, i) => bars[i].map((b, k) => (
          <line x1={1} x2={W - 1} y1={r.y + (bars[i].length > 1 ? (k ? 3 : -3) : 0)} y2={r.y + (bars[i].length > 1 ? (k ? 3 : -3) : 0)}
            stroke={byNum.get(b.n)?.colour} stroke-width={LINE} data-tip={b.tip} onClick={() => pick(b.n)} />
        )))}
        {lay.columns.map((c) => {
          const stroke = byNum.get(c.line)?.colour;
          return (
            <g class="link" data-tip={c.tip} onClick={() => pick(c.line)}>
              {[c.d, ...c.ends.map((e) => e.d)].map((d) => <path d={d} fill="none" stroke="var(--bg)" stroke-width={LINE + 3} />)}
              <path d={c.d} fill="none" stroke={stroke} stroke-width={LINE} />
              {c.ends.map((e) => <path d={e.d} fill="none" stroke={stroke} stroke-width={LINE} data-tip={e.tip} />)}
            </g>
          );
        })}
        {lay.rows.length > 0 && <line x1={lay.spineX} x2={lay.spineX} y1={lay.rows[lay.spine[0]].y} y2={lay.rows[lay.spine[1]].y} stroke={colour} stroke-width={BAR} />}
        {lay.loopEnds.map((d) => <path d={d} fill="none" stroke={colour} stroke-width={BAR} />)}
        {lay.pills.map((p) => {
          const r = lay.rows[p.row], h = p.big || bars[p.row].length ? 10 : 8;
          return <rect class="pill" x={p.x0 - h / 2 - 1} y={r.y - h / 2 - 1} width={p.x1 - p.x0 + h + 2} height={h + 2} rx={h / 2 + 1} data-tip={p.tip} />;
        })}
        </g>
      </svg>
      <ul class="stops related-rows" ref={ul}>
        {rows}
      </ul>
    </div>
  );
}
