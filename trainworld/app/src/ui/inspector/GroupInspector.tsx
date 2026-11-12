// T-1791, T-1809: a line group's view, with stop editing in the diagram. The title (the group's
// bullets and name, Show on the map, the all-stops switch, Copy line), then one combined diagram laid
// out as a schematic (`game/groupLayout.ts`): bars that bend, 2 px apart where lines run together;
// key stops named in full rows, the rest as ticks; crossing lines as bars and bullets as the related
// lines view draws them; each line's bullet at its ends, floating at the top while its top is
// scrolled away. A click on a stop takes it out (Shift-click ends the line there); a click where it
// passes adds one. All keep the line's path (sim `drop_stop`, `end_line_at`, `stop_passed`).

import { useEffect, useLayoutEffect, useRef, useState } from "preact/hooks";
import { demandAt, lineSeconds } from "../../game/clock";
import { stopKey } from "../../game/complexView";
import { BAR, CHIP, CHIP_GAP, CHIP_W, groupLayout, pathPlaces, PITCH, type GroupLayout } from "../../game/groupLayout";
import { bulletDraw, bulletText } from "../bullet";
import { shapeOf } from "../../game/lineShapeState";
import { groupAllStops, groupBullet, groupHover, groupLabel, groupName, lineGroups } from "../../game/lineGroups";
import { byHand, handOrder } from "../../game/handOrder";
import { LINE_PALETTE } from "../../game/palette";
import { ColourPicker } from "../ColourPicker";
import { GroupPill } from "../GroupPill";
import { LINE, ROUND_MAX, TALL_MAX, barOffset, pillHeight } from "../../game/relatedLayout";
import { today } from "../../game/today";
import { DEMAND_INDEX, type LineInput, type WorldView } from "../../game/types";
import { selection } from "../../game/ui";
import { edit, stationNames, stationsById, world } from "../../game/world";
import { drawnReversed } from "../../game/northUp";
import { scrollerOf } from "../scroller";
import type { EditOp } from "../../workers/protocol";
import { locateLine } from "../locate";
import { tipBullet } from "../tip";
import { Chip, Icon } from "../widgets";
import { NameEdit } from "./NameEdit";
import { TwoRows } from "./LineInspector";
import { GroupChartButton } from "./SchedulePanel";
import { t, tSeries } from "../../i18n";
import { mobileDevice } from "../../workers/deviceMemory";

/** A stop pill's height in a named row with no crossing bars, a tick's height, px. */
const PILL = 9, TICK = 3;
/** How far a crossing bar runs past the outermost line it crosses, px. */
const CROSS_STUB = 10;
/** Bullets shown after a named row's name, as the line panel's. */
const STOP_CHIPS = 2;

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

/** The last layout, kept while its inputs are the same (ordering tries every arrangement per row). */
let kept: { sig: string; lay: GroupLayout } | null = null;


export function GroupInspector({ group }: { group: number }) {
  const w = world.value;
  const byNum = w ? lineGroups(w.save.lines).get(group) : undefined;
  // T-1813: the columns in the Lines tab's hand order, where it has one
  const ls = byNum && byHand(byNum, (l) => [l.id], handOrder.value.lines);
  const all = groupAllStops.value;
  const [copying, setCopying] = useState(false);
  const [editing, setEditing] = useState(false);
  const [preview, setPreview] = useState<string | null>(null);
  const [symbol, setSymbol] = useState<string | null>(null);
  // a click away, or Esc outside the name box (it handles its own), closes the editor
  useEffect(() => {
    if (!editing) return;
    const away = (e: PointerEvent) => { if (!(e.target as Element | null)?.closest?.("[data-group-edit]")) setEditing(false); };
    const esc = (e: KeyboardEvent) => {
      if (e.key !== "Escape" || (e.target as Element | null)?.closest?.(".name-edit")) return;
      e.preventDefault();
      e.stopPropagation();
      setEditing(false);
    };
    window.addEventListener("pointerdown", away, true);
    window.addEventListener("keydown", esc, true);
    return () => { window.removeEventListener("pointerdown", away, true); window.removeEventListener("keydown", esc, true); };
  }, [editing]);
  useEffect(() => { setEditing(false); setSymbol(null); setPreview(null); }, [group]);
  // T-1894: the editor sits under the pinned title, so opening it scrolls the panel to its top
  const titleEl = useRef<HTMLDivElement>(null);
  useEffect(() => { if (editing) scrollerOf(titleEl.current)?.scrollTo({ top: 0 }); }, [editing]);
  // how far the panel is scrolled past the diagram's top, px (0 while its top shows)
  const [past, setPast] = useState(0);
  const body = useRef<HTMLDivElement>(null);
  useEffect(() => () => { groupHover.value = null; }, []);
  useLayoutEffect(() => {
    const on = () => {
      const el = body.current, sc = scrollerOf(el);
      if (!el || !sc) return;
      // the panels are drawn at the UI scale's CSS zoom; the bullets are placed in unzoomed px
      const b = el.getBoundingClientRect(), zoom = el.offsetHeight ? b.height / el.offsetHeight : 1;
      // below the panel's title, which sticks at its top
      const title = sc.querySelector<HTMLElement>(":scope > .title, #inspector > .title");
      const top = Math.max(sc.getBoundingClientRect().top, title ? title.getBoundingClientRect().bottom : -Infinity);
      setPast(Math.max(0, Math.round((top - b.top) / zoom)));
    };
    on();
    document.addEventListener("scroll", on, { capture: true, passive: true });
    return () => document.removeEventListener("scroll", on, { capture: true });
  }, []);
  if (!w || !ls || !byNum) return <p class="muted">{t("groupInspector.none")}</p>;
  const names = stationNames(w);
  const nm = (s: string) => names.get(s) ?? "";
  const name = groupName(byNum);
  const rename = (n: string) => void edit({ op: "groupName", group, name: n });
  // T-1813: the bullet's text and colour set by hand, sent together
  const look = { symbol: byNum[0].groupSymbol ?? "", colour: byNum[0].groupColour || null };
  const setLook = (v: { symbol?: string; colour?: string | null }) => void edit({ op: "groupLook", group, symbol: v.symbol ?? look.symbol, colour: v.colour === undefined ? look.colour : v.colour }).then(() => setPreview(null));
  const commitSymbol = () => {
    if (symbol === null) return;
    const v = symbol.trim();
    setSymbol(null);
    if (v !== look.symbol) setLook({ symbol: v });
  };
  const copy = async (l: LineInput) => { setCopying(false); await edit({ op: "copyLine", line: l.num }); };

  // lines outside the group stopping at each place, busiest there first (as the line panel's bullets)
  const inGroup = new Set(ls.map((l) => l.id));
  const d = today.value;
  const outside = new Map<string, { l: LineInput; n: number }[]>();
  for (const o of w.save.lines) {
    if (inGroup.has(o.id)) continue;
    const at = new Map<string, number>();
    o.stops.forEach((s, k) => { const key = stopKey(s); at.set(key, (at.get(key) ?? 0) + (d.get(o.id)?.stops[k] ?? 0)); });
    for (const [key, n] of at) (outside.get(key) ?? outside.set(key, []).get(key)!).push({ l: o, n });
  }
  const across = (key: string) => (outside.get(key) ?? []).sort((a, b) => b.n - a.n).map((x) => x.l);

  const ins = ls.map((l) => ({ num: l.num, places: pathPlaces(l.stops, w.lineStats[l.id]?.passed ?? [], stopKey) }));
  const keys = [...new Set(ins.flatMap((x) => x.places.map((p) => p.key)))];
  const crossN = new Map(keys.map((k) => [k, (outside.get(k) ?? []).length]));
  const at = stationsById(w);
  const ups = ls.map((l) => drawnReversed(at.get(l.stops[0]), at.get(l.stops[l.stops.length - 1]), l.stops.length > 2 && l.stops[0] === l.stops[l.stops.length - 1]));
  // T-2016: each line's bullet width, so a two-character pill sits as snugly beside its neighbours
  const chipW = ls.map((l) => { const b = bulletText(l.letter, "sm"); return b ? bulletDraw(b, shapeOf(l.id)).w : CHIP_W; });
  const sig = JSON.stringify([all, ins, [...crossN], ups, chipW]);
  if (kept?.sig !== sig) kept = { sig, lay: groupLayout(ins, { all, transfer: (k) => (crossN.get(k) ?? 0) > 0, crossings: (k) => crossN.get(k) ?? 0, reversed: (k) => ups[k], chipW: (k) => chipW[k] }) };
  const lay = kept.lay;
  const now = ls.map((l) => trainsNow(w, l));
  const W = lay.width;

  // a row's hover: its station, and the lines stopping and passing there
  const rowTip = (r: number) => {
    const cells = lay.rows[r].cells.map((c, k) => (c ? { k, c } : null)).filter((v) => v !== null);
    const st = [...new Set(cells.map((v) => nm(v.c.station)))].join(" / ");
    const stop = cells.filter((v) => v.c.stop).map((v) => ls[v.k].name), pass = cells.filter((v) => !v.c.stop).map((v) => ls[v.k].name);
    return [st, stop.length ? t("groupInspector.stops", { list: tSeries(stop) }) : "", pass.length ? t("groupInspector.passes", { list: tSeries(pass) }) : ""].filter(Boolean).join("\n");
  };
  // stops side by side at one place in a row: one mark over their bars
  const runs = (r: number) => {
    const row = lay.rows[r], out: [number, number][] = [];
    const ord = row.order.filter((k) => row.cells[k]?.stop);
    for (let i = 0; i < ord.length;) {
      let j = i;
      while (j + 1 < ord.length && Math.abs(lay.xs[ord[j + 1]][r] - lay.xs[ord[j]][r]) <= PITCH + 0.5 && row.cells[ord[j + 1]]!.key === row.cells[ord[j]]!.key) j++;
      out.push([lay.xs[ord[i]][r], lay.xs[ord[j]][r]]);
      i = j + 1;
    }
    return out;
  };
  // a line's click target at a row: half way to its neighbours, at most 11 px a side
  const hitOf = (k: number, r: number) => {
    const ord = lay.rows[r].order, i = ord.indexOf(k), x = lay.xs[k][r];
    const l = i > 0 ? Math.max(x - 11, (lay.xs[ord[i - 1]][r] + x) / 2) : x - 11;
    const rr = i + 1 < ord.length ? Math.min(x + 11, (lay.xs[ord[i + 1]][r] + x) / 2) : x + 11;
    return [l, rr];
  };

  // bullets that float at the panel's top while the line's top bullet is scrolled away
  const floating: { k: number; x: number }[] = [];
  if (past > 0) {
    const y = past + CHIP / 2 + 2;
    lay.bullets.forEach((b) => {
      if (b.end !== 0 || b.y >= y) return;
      const x = lay.xAt(b.col, y);
      const last = lay.rows[lay.spans[b.col][1]];
      if (Number.isFinite(x) && last.y > y + CHIP) floating.push({ k: b.col, x });
    });
    floating.sort((a, b) => a.x - b.x);
    for (let i = 1; i < floating.length; i++) floating[i].x = Math.max(floating[i].x, floating[i - 1].x + (chipW[floating[i - 1].k] + chipW[floating[i].k]) / 2 + CHIP_GAP);
  }

  const shown = lay.rows.map((row, r) => ({ row, r }));
  return (
    <>
      <div class="title" data-group-edit ref={titleEl}>
        <button class={"badge-edit" + (editing ? " on" : "")} data-tip={t("lineTitle.edit")} aria-label={t("lineTitle.edit")}
          onClick={() => setEditing(true)} onDblClick={() => setEditing(true)}><GroupPill ls={byNum} size="lg" look={preview ? { colour: preview } : undefined} /></button>
        <NameEdit key={group} name={name} onRename={rename} control={{ open: editing, set: setEditing }}>
          <button class="btn icon" data-tip={t("lineTitle.show")} aria-label={t("lineTitle.show")} onClick={() => locateLine(ls.flatMap((l) => l.stops))}>
            <Icon.locate />
          </button>
        </NameEdit>
      </div>
      {editing && (
        <div class="line-identity" data-group-edit>
          <label>{t("groupInspector.bullet")} <input aria-label={t("groupInspector.bulletLabel")} value={symbol ?? look.symbol} placeholder={groupLabel(name)}
            onInput={(e) => setSymbol(e.currentTarget.value)} onBlur={commitSymbol} onKeyDown={(e) => { if (e.key === "Enter") commitSymbol(); }} />
            {look.colour && <button class="btn icon mini" data-tip={t("groupInspector.backToColour")} aria-label={t("groupInspector.backToColour")} onClick={(e) => { e.preventDefault(); setLook({ colour: null }); }}><Icon.cycle /></button>}
          </label>
          <ColourPicker value={preview ?? groupBullet(byNum, (id) => today.value.get(id)?.riders ?? 0).colour} presets={LINE_PALETTE} onPreview={setPreview} onCommit={(hex) => { setPreview(hex); setLook({ colour: hex }); }} />
        </div>
      )}
      <div class="h divided">
        {t("groupInspector.stopsHeading")}
        <button class={"btn icon mini related-toggle" + (all ? " on" : "")} data-tip={t(all ? "groupInspector.fewer" : "groupInspector.all")} aria-label={t(all ? "groupInspector.fewer" : "groupInspector.all")} aria-pressed={all} onClick={() => (groupAllStops.value = !all)}>
          <Icon.allStops />
        </button>
        <button class={"btn icon mini related-toggle" + (copying ? " on" : "")} data-tip={t("groupInspector.copy")} aria-label={t("groupInspector.copy")} aria-pressed={copying} onClick={() => setCopying(!copying)}>
          <Icon.copy />
        </button>
        <GroupChartButton ls={byNum} />
      </div>
      {copying && (
        <div class="group-copy">
          {ls.map((l) => <Chip line={l} pick={(x) => void copy(x)} tip={t("groupInspector.copyLine", { name: l.name })} />)}
        </div>
      )}
      <div class="group-diagram">
        <div class="group-body" ref={body} style={{ height: lay.height + "px", "--gw": W + "px" }}>
          <svg width={W} height={lay.height} style={{ width: W + "px", height: lay.height + "px" }} onPointerLeave={() => (groupHover.value = null)}>
            {shown.map(({ row, r }) => <rect class="group-row" x={0} y={row.y - row.h / 2} width={W} height={row.h} data-tip={rowTip(r)} onPointerEnter={() => (groupHover.value = null)} />)}
            {shown.filter(({ row }) => row.named).flatMap(({ row, r }) => {
              const key = row.cells.find((c) => c)!.key;
              const xs = across(key).slice(0, TALL_MAX);
              const bars = xs.length > ROUND_MAX ? xs : xs.slice(0, ROUND_MAX);
              // T-1817: across the lines with a place here only, a short way past either side
              const at = row.cells.map((c, k) => (c ? lay.xs[k][r] : NaN)).filter(Number.isFinite);
              const x0 = Math.max(1, Math.min(...at) - CROSS_STUB), x1 = Math.min(W - 1, Math.max(...at) + CROSS_STUB);
              return bars.map((o, i) => (
                <line class="group-cross" x1={x0} x2={x1} y1={row.y + barOffset(i, bars.length)} y2={row.y + barOffset(i, bars.length)} stroke={o.colour} stroke-width={LINE - 1}
                  data-tip={t("groupInspector.stopsAt", { line: o.name, station: rowTip(r).split("\n")[0] })} onClick={() => (selection.value = { kind: "line", line: o.id })} />
              ));
            })}
            {ls.map((l, k) => (
              <g class={"group-col" + (now[k].n ? "" : " idle")} onPointerEnter={() => (groupHover.value = l.num)}>
                <path d={lay.paths[k]} fill="none" stroke="var(--bg)" stroke-width={BAR + 3} />
                <path d={lay.paths[k]} fill="none" stroke={l.colour} stroke-width={BAR} data-tip={l.name} />
              </g>
            ))}
            {shown.flatMap(({ row, r }) => {
              if (!row.named) return runs(r).map(([a, b]) => <rect class="tick" x={a - BAR / 2 - 1} y={row.y - TICK / 2} width={b - a + BAR + 2} height={TICK} />);
              const n = Math.min(TALL_MAX, across(row.cells.find((c) => c)!.key).length);
              const h = n > ROUND_MAX ? pillHeight(n, true) - 1 : PILL;
              const idle = (a: number, b: number) => row.order.filter((k) => row.cells[k]?.stop && lay.xs[k][r] >= a - 0.5 && lay.xs[k][r] <= b + 0.5).every((k) => !now[k].n);
              return runs(r).map(([a, b]) => <rect class={"pill" + (idle(a, b) ? " idle" : "")} x={a - PILL / 2} y={row.y - h / 2} width={b - a + PILL} height={h} rx={PILL / 2} />);
            })}
            {ls.map((l, k) => (
              <g class={"group-col" + (now[k].n ? "" : " idle")} onPointerEnter={() => (groupHover.value = l.num)}>
                {shown.map(({ row, r }) => {
                  const c = row.cells[k];
                  if (!c) return null;
                  const st = w.lineStats[l.id];
                  const n = l.stops.length;
                  const rail = !!st && !st.bus;
                  const B = tipBullet(l.letter, l.colour, shapeOf(l.id));
                  const [x0, x1] = hitOf(k, r), x = lay.xs[k][r];
                  const hit = <rect class="hit" x={x0} y={row.y - row.h / 2} width={x1 - x0} height={row.h} />;
                  if (!c.stop) {
                    const tip = rail ? t("groupInspector.addStop", { line: B, station: nm(c.station) }) : t("groupInspector.passesStation", { line: B, station: nm(c.station) });
                    return (
                      <g class={"group-cell" + (rail ? " edit" : "")} data-tip={tip} onClick={rail ? () => void change({ op: "stopPassed", line: l.num, station: Number(c.station) }) : undefined}>
                        {hit}
                        {row.named
                          ? <rect class="pill pass" x={x - PILL / 2} y={row.y - PILL / 2} width={PILL} height={PILL} rx={PILL / 2} />
                          : <rect class="ghost" x={x - BAR / 2 - 1} y={row.y - TICK / 2} width={BAR + 2} height={TICK} />}
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
                    ? (turns ? t("groupInspector.turnsBack", { line: B, station: here }) : t("groupInspector.stopsAt", { line: B, station: here }))
                    : t("groupInspector.removeStop", { line: B, station: here }) + (end || loop || n < 3 || mobileDevice() ? "" : "\n" + t("groupInspector.shiftEnd", { line: B }));
                  const click = (e: MouseEvent) => {
                    if (e.shiftKey && !end && !loop) void change({ op: "endLineAt", line: l.num, index: c.index });
                    else void change({ op: "dropStop", line: l.num, index: c.index });
                  };
                  return (
                    <g class={"group-cell" + (keep ? "" : " edit")} data-tip={tip} onClick={keep ? undefined : click}>
                      {hit}
                      {row.named
                        ? <circle class="dot" cx={x} cy={row.y} r={PILL / 2 - 1.5} fill={l.colour} />
                        : <rect class="dot" x={x - BAR / 2 - 1} y={row.y - TICK / 2 - 1} width={BAR + 2} height={TICK + 2} fill={l.colour} />}
                    </g>
                  );
                })}
              </g>
            ))}
          </svg>
          {lay.bullets.map((b) => {
            const l = ls[b.col];
            return (
              <span class={"group-bullet" + (now[b.col].n ? "" : " idle")} style={{ left: b.x - chipW[b.col] / 2 + "px", top: b.y - 8 + "px" }} onPointerEnter={() => (groupHover.value = l.num)}>
                <Chip line={l} size="sm" tip={now[b.col].n ? l.name : t("groupInspector.noTrains", { line: l.name, demand: t("inspector.demand", { demand: now[b.col].lev }) })} />
              </span>
            );
          })}
          {floating.map((f) => (
            <span class={"group-bullet float" + (now[f.k].n ? "" : " idle")} style={{ left: f.x - chipW[f.k] / 2 + "px", top: past + 2 + "px" }} onPointerEnter={() => (groupHover.value = ls[f.k].num)}>
              <Chip line={ls[f.k]} size="sm" />
            </span>
          ))}
          <div class="group-names" style={{ left: W + 6 + "px" }}>
            {shown.filter(({ row }) => row.named).map(({ row, r }) => {
              const cells = row.cells.filter((c) => c !== null);
              const xs = across(cells[0].key);
              const tall = Math.min(TALL_MAX, xs.length) > ROUND_MAX;
              const chips = xs.slice(0, tall ? TALL_MAX : STOP_CHIPS), rest = xs.slice(chips.length);
              return (
                <div class={"group-name" + (cells.some((c) => c.stop) ? "" : " muted")} style={{ top: row.y - row.h / 2 + "px", height: row.h + "px" }} data-tip={rowTip(r)}>
                  <span class="grow" onClick={() => (selection.value = { kind: "station", station: cells[0].station })}>{[...new Set(cells.map((c) => nm(c.station)))].join(" / ")}</span>
                  {tall
                    ? <TwoRows lines={chips} />
                    : chips.map((o) => <Chip line={o} size="sm" />)}
                  {rest.length > 0 && <span class="stop-more" data-tip={[...new Set(rest.map((o) => o.name))].join("\n")}>+</span>}
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </>
  );
}
