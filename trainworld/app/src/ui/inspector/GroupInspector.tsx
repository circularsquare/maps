// T-1791 stages 2 and 3: a line group's view, with stop editing in the diagram. The title (the group's bullets and name), a row
// per line (its trains now, Copy line), then one combined diagram: a column per line in its colour,
// a row per place (`game/groupLayout.ts`), a white pill where the line stops, a thin notch where it
// passes a station, its bar between. No station names: a hover names them. A click on a pill takes
// the stop out (Shift-click ends the line there); a click where it passes adds one, a ghost pill
// showing on hover. All keep the line's path (sim `drop_stop`, `end_line_at`, `stop_passed`).

import { useEffect } from "preact/hooks";
import { demandAt, lineSeconds } from "../../game/clock";
import { stopKey } from "../../game/complexView";
import { groupLayout, pathPlaces } from "../../game/groupLayout";
import { groupHover, groupName, lineGroups } from "../../game/lineGroups";
import { DEMAND_INDEX, type LineInput, type WorldView } from "../../game/types";
import { selection } from "../../game/ui";
import { edit, stationNames, world } from "../../game/world";
import type { EditOp } from "../../workers/protocol";
import { locateLine } from "../locate";
import { Chip, Icon } from "../widgets";
import { NameEdit } from "./NameEdit";
import { DEMAND_LABEL } from "./LineInspector";

/** a column's width and a row's height, px */
const COL = 22, ROW = 18;
/** a line's bar, its stop pill's size, and the notch where it passes a station, px */
const BAR = 6, PILL = 9, NOTCH = BAR + 4;
/** room above the first row and below the last, px */
const PAD = 8;

/** The line's trains at its level now, and that level; 0 for a line not running. */
function trainsNow(w: WorldView, l: LineInput) {
  const st = w.lineStats[l.id];
  const lev = demandAt(lineSeconds(st));
  return { n: st?.status === "running" ? st.trainsNeeded[DEMAND_INDEX[lev]] : 0, lev };
}

/** One stop edit at a time: a second click before the world has the first would count from stale indices. */
let editing = false;
async function change(op: EditOp) {
  if (editing) return;
  editing = true;
  try { await edit(op); } finally { editing = false; }
}

export function GroupInspector({ group }: { group: number }) {
  const w = world.value;
  const ls = w ? lineGroups(w.save.lines).get(group) : undefined;
  useEffect(() => () => { groupHover.value = null; }, []);
  if (!w || !ls) return <p class="muted">No such group.</p>;
  const names = stationNames(w);
  const name = groupName(ls);
  const rename = (n: string) => void edit({ op: "groupName", group, name: n === ls[0].name ? "" : n });
  const copy = async (l: LineInput) => { await edit({ op: "copyLine", line: l.num }); };
  const lay = groupLayout(ls.map((l) => ({ num: l.num, places: pathPlaces(l.stops, w.lineStats[l.id]?.passed ?? [], stopKey) })));
  const x = (k: number) => k * COL + COL / 2, y = (r: number) => PAD + r * ROW + ROW / 2;
  const width = ls.length * COL, height = PAD * 2 + lay.rows.length * ROW;
  const now = ls.map((l) => trainsNow(w, l));
  const nm = (s: string) => names.get(s) ?? "";
  // a row's hover: its station, or each line's where they differ (unrelated lines side by side)
  const rowTip = (r: number) => {
    const cells = lay.rows[r].map((c, k) => (c ? { k, c } : null)).filter((v) => v !== null);
    const places = new Set(cells.map((v) => stopKey(v.c.station)));
    if (places.size <= 1) return [...new Set(cells.map((v) => nm(v.c.station)))].join(" / ");
    return cells.map((v) => `${ls[v.k].name}: ${nm(v.c.station)}`).join("\n");
  };
  // where lines stop at one place side by side in a row, one elongated pill over their columns, as the
  // related lines view draws an interchange
  const runs = lay.rows.map((row) => {
    const out: [number, number][] = [];
    for (let k = 0; k < row.length;) {
      const c = row[k];
      if (!c?.stop) { k++; continue; }
      let e = k;
      while (e + 1 < row.length && row[e + 1]?.stop && stopKey(row[e + 1]!.station) === stopKey(c.station)) e++;
      out.push([k, e]);
      k = e + 1;
    }
    return out;
  });
  return (
    <>
      <div class="title">
        <span class="group-chips">{ls.map((l) => <Chip line={l} />)}</span>
        <NameEdit key={group} name={name} onRename={rename}>
          <button class="btn icon" data-tip="Show on the map" aria-label="Show on the map" onClick={() => locateLine(ls.flatMap((l) => l.stops))}>
            <Icon.locate />
          </button>
        </NameEdit>
      </div>
      <ul class="rows pick group-lines">
        {ls.map((l, k) => (
          <li onClick={() => (selection.value = { kind: "line", line: l.id })}>
            <Chip line={l} />
            <span class="grow">{l.name}</span>
            <span class={"val" + (now[k].n ? "" : " muted")} data-tip={`${DEMAND_LABEL[now[k].lev]} demand: ${now[k].n} trains`}>{now[k].n}</span>
            <button class="btn icon mini" data-tip="Copy line" aria-label={`Copy ${l.name}`} onClick={(e) => { e.stopPropagation(); void copy(l); }}><Icon.copy /></button>
          </li>
        ))}
      </ul>
      <div class="group-diagram">
        <div class="group-heads" style={{ width: width + "px" }}>
          {ls.map((l, k) => <span class={"group-head" + (now[k].n ? "" : " idle")} style={{ width: COL + "px" }}><Chip line={l} size="sm" tip={now[k].n ? l.name : `${l.name}\nNo trains at ${DEMAND_LABEL[now[k].lev].toLowerCase()} demand`} /></span>)}
        </div>
        <div class="group-body">
          <svg width={width} height={height} style={{ width: width + "px", height: height + "px" }} onPointerLeave={() => (groupHover.value = null)}>
            {lay.rows.map((_, r) => <rect class="group-row" x={0} y={y(r) - ROW / 2} width={width} height={ROW} data-tip={rowTip(r)} onPointerEnter={() => (groupHover.value = null)} />)}
            {ls.map((l, k) => {
              const own = lay.rows.map((c, r) => (c[k] ? r : -1)).filter((r) => r >= 0);
              const segs = own.slice(1).map((r, i) => [own[i], r]).filter(([, b]) => !lay.breaks[k].includes(b));
              return (
                <g class={"group-col" + (now[k].n ? "" : " idle")} onPointerEnter={() => (groupHover.value = l.num)}>
                  {segs.map(([a, b]) => <line x1={x(k)} x2={x(k)} y1={y(a)} y2={y(b)} stroke={l.colour} stroke-width={BAR} data-tip={l.name} />)}
                </g>
              );
            })}
            {lay.rows.flatMap((_, r) => runs[r].map(([a, b]) => (
              <rect class={"pill" + (now.slice(a, b + 1).every((s) => !s.n) ? " idle" : "")} x={x(a) - PILL / 2} y={y(r) - PILL / 2} width={x(b) - x(a) + PILL} height={PILL} rx={PILL / 2} />
            )))}
            {ls.map((l, k) => (
              <g class={"group-col" + (now[k].n ? "" : " idle")} onPointerEnter={() => (groupHover.value = l.num)}>
                {lay.rows.map((row, r) => {
                  const c = row[k];
                  if (!c) return null;
                  const st = w.lineStats[l.id];
                  const n = l.stops.length;
                  const rail = !!st && !st.bus;
                  const hit = <rect class="hit" x={x(k) - COL / 2} y={y(r) - ROW / 2} width={COL} height={ROW} />;
                  if (!c.stop) {
                    const tip = rail ? `Click to add the ${l.name}'s stop at ${nm(c.station)}` : `${l.name} passes ${nm(c.station)}`;
                    return (
                      <g class={"group-cell" + (rail ? " edit" : "")} data-tip={tip} onClick={rail ? () => void change({ op: "stopPassed", line: l.num, station: Number(c.station) }) : undefined}>
                        {hit}
                        <rect class="pill pass" x={x(k) - PILL / 2} y={y(r) - PILL / 2} width={PILL} height={PILL} rx={PILL / 2} />
                      </g>
                    );
                  }
                  // a stop comes out unless the line needs it: a bus line, a line of two stops, a loop's
                  // ends (and a loop of three), or where its trains turn back
                  const turns = !!st?.reversals.includes(c.index);
                  const loop = !!st?.loop;
                  const keep = !rail || n <= 2 || turns || (loop && (c.index === 0 || c.index === n - 1 || n <= 3));
                  const here = nm(c.station);
                  const end = c.index === 0 || c.index === n - 1;
                  const tip = keep
                    ? (turns ? `${l.name} turns back at ${here}` : `${l.name} stops at ${here}`)
                    : `Click to remove the ${l.name}'s stop at ${here}` + (end || loop || n < 3 ? "" : `\nShift-click to end the ${l.name} here`);
                  const click = (e: MouseEvent) => {
                    if (e.shiftKey && !end && !loop) void change({ op: "endLineAt", line: l.num, index: c.index });
                    else void change({ op: "dropStop", line: l.num, index: c.index });
                  };
                  return (
                    <g class={"group-cell" + (keep ? "" : " edit")} data-tip={tip} onClick={keep ? undefined : click}>
                      {hit}
                      <circle class="dot" cx={x(k)} cy={y(r)} r={PILL / 2 - 1.5} fill={l.colour} />
                    </g>
                  );
                })}
              </g>
            ))}
          </svg>
          <div class="group-names" style={{ paddingTop: PAD + "px" }}>
            {lay.rows.map((row, r) => {
              const cells = row.filter((c) => c !== null);
              const first = cells[0].station;
              return (
                <div class={"group-name" + (cells.some((c) => c.stop) ? "" : " muted")} style={{ height: ROW + "px" }} data-tip={rowTip(r)} onClick={() => (selection.value = { kind: "station", station: first })}>
                  {[...new Set(cells.map((c) => nm(c.station)))].join(" / ")}
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </>
  );
}

