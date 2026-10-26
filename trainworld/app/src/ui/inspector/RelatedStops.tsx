// The line panel's stops drawn with the lines related to the line (T-1168, notes/T-1168.md): the
// selected line down the middle, lines beside it in columns that turn out where they leave, stop
// pills over every column stopping at a row, crossing lines as bars. Layout in
// game/relatedLayout.ts; the selected line's own rows come from the line panel (`lRow`).

import type { ComponentChildren } from "preact";
import { useLayoutEffect, useRef, useState } from "preact/hooks";
import { decodeRelated, type RelatedGroup } from "../../game/relatedLines";
import { BAR, relatedLayout } from "../../game/relatedLayout";
import { distance } from "../../game/format";
import type { LineInput } from "../../game/types";
import { selection } from "../../game/ui";
import { linesByNum, stationNames, world } from "../../game/world";
import { Chip } from "../widgets";
import { useClockQuery } from "../useClockQuery";

/** The other lines at a row whose bullets show after its name, in the order they show. */
export type RowTransfers = (stop: number, drawn: ReadonlySet<number>) => LineInput[];

export function RelatedStops({ l, stopS, colour, lRow, transfers }: {
  l: LineInput;
  stopS: number[];
  colour: string;
  /** the selected line's stop `i` as the plain list draws it, without the bullets of `drawn` lines */
  lRow: (i: number, drawn: ReadonlySet<number>) => ComponentChildren;
  transfers: RowTransfers;
}) {
  const w = world.value!;
  const answer = useClockQuery({ what: "lineRelated", line: l.num }, [l.num, w.version], (d) => ({ line: l.num, groups: decodeRelated(d) }));
  const groups: RelatedGroup[] = answer?.line === l.num ? answer.groups : [];
  const ul = useRef<HTMLUListElement>(null);
  // the rows' centres in the list, and the room kept above and below it for turns past the ends
  const [geo, setGeo] = useState<{ ys: number[]; width: number; pad: [number, number] }>({ ys: [], width: 340, pad: [0, 0] });
  const byNum = linesByNum(w), names = stationNames(w);
  const stopName = (line: number, stop: number) => names.get(byNum.get(line)?.stops[stop] ?? "") ?? "";
  const lay = relatedLayout(groups, {
    line: l.num, stopS, width: geo.width,
    rowY: (i) => geo.ys[i] ?? i * 25 + 12.5,
    lineName: (n) => byNum.get(n)?.name ?? "",
    stopName,
    stopCount: (n) => byNum.get(n)?.stops.length ?? 0,
    dist: distance,
  });
  const listH = lay.rows.length ? lay.rows[lay.rows.length - 1].y + 13 : 0;
  const tops = [0, ...lay.bullets.map((b) => b.y - 10)], bottoms = [listH, ...lay.bullets.map((b) => b.y + 10)];
  const pad: [number, number] = [Math.ceil(Math.max(0, -Math.min(...tops))), Math.ceil(Math.max(0, Math.max(...bottoms) - listH))];
  // the rows' centres as laid out, for the drawing (a row is 25 px, 24 on a phone)
  useLayoutEffect(() => {
    const el = ul.current;
    if (!el) return;
    const ys = Array.from(el.children, (c) => (c as HTMLElement).offsetTop - el.offsetTop + (c as HTMLElement).offsetHeight / 2);
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
  return (
    <div class="related" style={{ "--gw": W + "px", "--spine": lay.spineX + "px", paddingTop: geo.pad[0] + "px", paddingBottom: geo.pad[1] + "px" }}>
      <svg class="related-draw" width={W} height={listH + geo.pad[0] + geo.pad[1]}>
        <g transform={`translate(0 ${geo.pad[0]})`}>
        {lay.rows.map((r, i) => bars[i].map((b, k) => (
          <line x1={1} x2={W - 1} y1={r.y + (bars[i].length > 1 ? (k ? 3 : -3) : 0)} y2={r.y + (bars[i].length > 1 ? (k ? 3 : -3) : 0)}
            stroke={byNum.get(b.n)?.colour} stroke-width={3} data-tip={b.tip} onClick={() => pick(b.n)} />
        )))}
        {lay.columns.map((c) => (
          <g class="link" data-tip={c.tip} onClick={() => pick(c.line)}>
            <path d={c.d} fill="none" stroke="var(--bg)" stroke-width={BAR + 3} />
            <path d={c.d} fill="none" stroke={byNum.get(c.line)?.colour} stroke-width={BAR} />
          </g>
        ))}
        {lay.rows.length > 0 && <line x1={lay.spineX} x2={lay.spineX} y1={lay.rows[lay.spine[0]].y} y2={lay.rows[lay.spine[1]].y} stroke={colour} stroke-width={BAR} />}
        {lay.pills.map((p) => {
          const r = lay.rows[p.row], h = p.big || bars[p.row].length ? 10 : 8;
          return <rect class="pill" x={p.x0 - h / 2 - 1} y={r.y - h / 2 - 1} width={p.x1 - p.x0 + h + 2} height={h + 2} rx={h / 2 + 1} data-tip={p.tip} />;
        })}
        </g>
      </svg>
      {lay.bullets.map((b) => {
        const o = byNum.get(b.line);
        return o && <span class="related-bullet" style={{ left: b.x + "px", top: b.y + geo.pad[0] + "px" }}><Chip line={o} size="sm" tip={b.tip} /></span>;
      })}
      <ul class="stops related-rows" ref={ul}>
        {lay.rows.map((r) => (r.lStop >= 0
          ? lRow(r.lStop, new Set(r.drawn))
          : (
            <li class="side">
              <span class="grow">
                <span class="stop-name muted" onClick={() => { const s = byNum.get(r.ref.line)?.stops[r.ref.stop]; if (s) selection.value = { kind: "station", station: s }; }}>
                  {stopName(r.ref.line, r.ref.stop)}
                </span>
              </span>
            </li>
          )))}
      </ul>
    </div>
  );
}
