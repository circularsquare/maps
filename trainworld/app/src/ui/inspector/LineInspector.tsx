// The line inspector (SPEC 6.3, 8).

import { ConstructButton } from "./ConstructButton";
import type { ComponentChildren } from "preact";
import { useEffect, useRef, useState } from "preact/hooks";
import { nearestStation } from "../../game/places";
import { demandAt, lineSeconds } from "../../game/clock";
import { Money } from "../MoneyText";
import { andList, count, distance, duration, forecast, headway, length, money, plural, speed, trainCostLine, trainKind } from "../../game/format";
import { peakLoadText, today, useToday } from "../../game/today";
import { CAR_M, CRUSH_PER_CAR, PLATFORM_MAX_M, RATED_PER_CAR, type CrowdedStretch } from "../../game/boarding";
import { describeIssues } from "../../game/issues";
import { complexOf } from "../../game/complexView";
import { DEMAND_INDEX, DEMANDS, type Demand, type LineInput, type LineStats, type StationInput } from "../../game/types";
import { ask, display, lineDraft, purposeFilter, relatedLines, selection, shownDayPhrase, tool } from "../../game/ui";
import { purposeName } from "../../game/purposes";
import { dailyRiders } from "../../map/trackHover";
import { CROWD_LOAD_FROM, CROWD_LOAD_STRONG, FULL_FLOOR } from "../../map/rendererShaders";
import { liveMoney, moneyState } from "../../game/money";
import { fullParts, fullTip, fullTitle, heldText, heldTip } from "../../game/capacityKinds";
import { toggleEditStops } from "../../game/lineActions";
import { noteHeadwayLimits, edit, linesAt, say, stationNames, stationsById, stationsByNum, stepLineCars, pendingCars, world, linesById, linesByNum } from "../../game/world";
import { live } from "../../map/live";
import { Alert, Chip, Icon, Row, Segmented, shakeEl, Stepper, SwapLabel } from "../widgets";
import { PurposeSplit } from "./PurposeSplit";
import { RelatedStops, useTightTimes } from "./RelatedStops";
import { ROUND_MAX, TALL_MAX } from "../../game/relatedLayout";
import { BusInspector } from "./BusInspector";
import { locateStation } from "../locate";
import { demandView } from "../../game/demand";
import { useClockQuery } from "../useClockQuery";
import { carsCap, copySchedule, pasteSchedule, scheduleCopy, scheduleOf, scheduleText } from "../../game/scheduleCopy";
import { drawnReversed } from "../../game/northUp";
import { MareyButton } from "./MareyChart";
import type { EditOp } from "../../workers/protocol";
import { UPGRADE } from "../../game/upgrades";
import { AverageSpeed, carsBought, carsText, DEMAND_LABEL, type Limit, NO_LIMITS, PastDay, Schedule, useLinePeriods } from "./schedule";
import { LineTitle, useLineTitle } from "./LineTitle";

const stationsNow = (): ReadonlyMap<number, StationInput> => (world.value ? stationsByNum(world.value) : new Map());

/** Where a line waits, in words (T-031): from `TrackApi.line_delays`. `behind`: the part of its
 * delay spent following slower trains on shared track (kind 11, T-1924); `waits`: standing at
 * passing tracks for faster trains (kind 13, T-2231), whose places come from `lineWaits`; the rest
 * is waiting at busy track. */
interface Spot { text: string; s: number }
interface Spots { spots: Spot[]; behindSpots: Spot[]; behind: number; waits: number }
function spotsFrom(d: Float64Array): Spots {
  const busy: Spot[] = [], following: Spot[] = [];
  const st = stationsNow();
  const lines = world.value ? linesByNum(world.value) : new Map<number, LineInput>();
  let behind = 0, waits = 0;
  for (let i = 0; i + 4 < d.length; i += 5) {
    const [kind, node, x, y, s] = [d[i], d[i + 1], d[i + 2], d[i + 3], d[i + 4]];
    if (kind === 13) { waits += s; continue; }
    const near = nearestStation(x, y)?.name;
    const at = node >= 0 ? st.get(node)?.name : undefined;
    if (kind === 11) behind += s;
    const ahead = kind === 11 && node >= 0 ? lines.get(node)?.name : undefined;
    const text =
      kind === 11
        ? ahead ? `Behind ${ahead} trains` : "Behind slower trains"
        : (kind === 2 || kind === 3 || kind === 6 || kind === 8) && at
          ? `At ${at}`
          : kind === 4
            ? `At the junction${near ? " near " + near : ""}`
            : kind === 5
              ? `At the crossing${near ? " near " + near : ""}`
              : kind === 12
                ? `At the merge${near ? " near " + near : ""}`
                : `On the track${near ? " near " + near : ""}`;
    const out = kind === 11 ? following : busy;
    const same = out.find((o) => o.text === text); // both directions, or two stretches by one station
    if (same) same.s += s;
    else out.push({ text, s });
  }
  return { spots: busy, behindSpots: following, behind, waits };
}
const NO_SPOTS: Spots = { spots: [], behindSpots: [], behind: 0, waits: 0 };

/** T-2232: the stations a line stands at for faster trains, from `TrackApi.line_waits`' triples,
 * both directions summed: "At E, for Express trains". */
function waitSpotsFrom(d: Float64Array): Spot[] {
  const st = stationsNow();
  const lines = world.value ? linesByNum(world.value) : new Map<number, LineInput>();
  const by = new Map<number, { s: number; faster: string[] }>();
  for (let i = 0; i + 2 < d.length; i += 3) {
    const x = by.get(d[i]) ?? { s: 0, faster: [] };
    const name = lines.get(d[i + 1])?.name;
    if (name && !x.faster.includes(name)) x.faster.push(name);
    x.s += d[i + 2];
    by.set(d[i], x);
  }
  return [...by].map(([node, x]) => ({ text: `At ${st.get(node)?.name ?? ""}${x.faster.length ? `, for ${andList(x.faster)} trains` : ""}`, s: x.s }));
}

/** Where full track holds a line's fleet back, per level (T-517): from `TrackApi.line_held`. */
interface Held { d: Demand; wait: number; place: string; head: string; name: string; tail: string; at: { id: string; lng: number; lat: number } | null; tip: string }
function heldFrom(v: Float64Array): Held[] {
  const st = stationsNow();
  const out: Held[] = [];
  DEMANDS.forEach((d, lev) => {
    const [wait, kind, node, x, y, , most] = v.subarray(7 * lev, 7 * lev + 7);
    if (!(wait > 0)) return;
    const s = (node >= 0 ? st.get(node) : undefined) ?? nearestStation(x, y);
    const name = s?.name ?? "";
    out.push({ d, wait, place: fullTitle(kind, name), ...fullParts(kind, name), name, at: s ? { id: s.id, lng: s.lng, lat: s.lat } : null, tip: fullTip(kind, most) });
  });
  return out;
}

/** What limits a line per level (T-679, `Limit`); null where nothing does. From the tail of
 * `TrackApi.line_held`. */
function limitsFrom(v: Float64Array): (Limit | null)[] {
  const st = stationsNow();
  return DEMANDS.map((_, lev) => {
    const [s, kind, node, x, y] = v.subarray(21 + 5 * lev, 26 + 5 * lev);
    if (!(s > 0)) return null;
    const name = ((node >= 0 ? st.get(node) : undefined) ?? nearestStation(x, y))?.name ?? "";
    return { s, what: fullParts(kind, name).head + name };
  });
}

/** A flat junction or flat crossing on the route (T-629): `TrackApi.line_marks`, after the stop
 * `gap` and drawn on the stops diagram between it and the next. */
interface RouteMark { crossing: boolean; gap: number; near: string; others: string[] }
function marksFrom(d: Float64Array): RouteMark[] {
  const lines = world.value ? linesByNum(world.value) : new Map<number, LineInput>();
  const out: RouteMark[] = [];
  for (let i = 0; i + 5 < d.length; ) {
    const n = d[i + 5];
    const others = Array.from(d.subarray(i + 6, i + 6 + n), (o) => lines.get(o)?.name ?? `Line ${o}`);
    out.push({ crossing: d[i] === 5, gap: d[i + 1], near: nearestStation(d[i + 3], d[i + 4])?.name ?? "", others });
    i += 6 + n;
  }
  return out;
}

/** The marks of one gap between stops, spread evenly down it: the diagram is not to scale. */
function RouteMarks({ marks }: { marks: RouteMark[] }) {
  return (
    <>
      {marks.map((m, k) => (
        <span class={"route-mark " + (m.crossing ? "crossing" : "junction")} style={{ "--at": (k + 1) / (marks.length + 1) }} data-tip={markTip(m)} />
      ))}
    </>
  );
}

function markTip(m: RouteMark): string {
  const where = m.near ? ` near ${m.near}` : "";
  if (m.crossing) return `Flat crossing${where}\n${m.others.length ? "Crosses " + m.others.join(", ") : "Crosses track no line uses"}`;
  return `Flat junction${where}\n${m.others.length ? "Also used by " + m.others.join(", ") : "No other line uses it"}`;
}

/** The round trip at the level in force, split into running, dwell, turnaround and delay (T-031),
 * and where the delay is. */
function RoundTrip({ st, lev, stops, spots: { spots: busy, behindSpots, behind: behindAll, waits: waitsAll }, waitSpots }: { st: LineStats; lev: number; stops: number; spots: Spots; waitSpots: Spot[] }) {
  // T-798: a loop shows its lap, one way round (the longer way's, half its round trip); a circle
  // stands the dwell at its first stop and turns nowhere, a balloon turns there
  const runs = st.loop ? 1 : 2;
  const total = (st.roundTripByLevel[lev] ?? st.roundTripS) / (st.loop ? 2 : 1);
  // T-797: a stop where trains reverse stands the turnaround both ways, not the dwell
  const reversals = st.reversals?.length ?? 0;
  const dwell = runs * Math.max(0, stops - 2 - reversals) * st.dwellS + (st.loop === "circle" ? st.dwellS : 0);
  const turn = st.loop === "circle" ? 0 : st.loop === "balloon" ? st.turnaroundS : 2 * st.turnaroundS;
  // and so does a bare track end it reverses at (B3), which is no stop
  const reversing = runs * (reversals + (st.endTurns?.length ?? 0)) * st.turnaroundS;
  const delay = (st.delayS[lev] ?? 0) / (st.loop ? 2 : 1);
  // T-1924: following slower trains, apart from waiting at busy track
  const behind = Math.min(delay, behindAll / (st.loop ? 2 : 1));
  // T-2232: and standing at passing tracks for faster ones
  const forFaster = Math.min(delay - behind, waitsAll / (st.loop ? 2 : 1));
  const waiting = delay - behind - forFaster;
  const running = Math.max(0, total - dwell - turn - reversing - delay);
  // T-2232: each delay row with its own places right under it, the largest three
  const delayRow = (label: string, v: number, places: Spot[]) => v >= 0.5 && (
    <>
      <ul class="rows"><Row label={label}>{duration(v)}</Row></ul>
      <ul class="rows sub-rows">
        {places.filter((p) => p.s >= 0.5).sort((a, b) => b.s - a.s).slice(0, 3).map((p) => (
          <Row label={p.text}>
            <span class="muted">{duration(p.s)}</span>
          </Row>
        ))}
      </ul>
    </>
  );
  return (
    <>
      <ul class="rows">
        <Row label={st.loop ? "Lap" : "Round trip"} bold>
          {duration(total)}
        </Row>
        <Row label="Running">{duration(running)}</Row>
        <Row label="Dwell at stops">{duration(dwell)}</Row>
        {turn > 0 && <Row label={st.loop ? "Turning at the end" : "Turning at the ends"}>{duration(turn)}</Row>}
        {reversing > 0 && <Row label="Reversing partway">{duration(reversing)}</Row>}
      </ul>
      {delayRow("Waiting at busy track", waiting, busy)}
      {delayRow("Behind slower trains", behind, behindSpots)}
      {delayRow("Waiting for faster trains", forFaster, waitSpots)}
    </>
  );
}

/** Bullets shown at a transfer stop; the rest sit behind a "+". */
const STOP_CHIPS = 2;

/** T-1391: the line stops at stop `i`'s complex again further along or before (not the next or
 * previous stop, nor a loop's first stop at its end), so it is a transfer to itself there. */
function selfTransfer(l: LineInput, i: number): boolean {
  const members = new Set(complexOf(l.stops[i]));
  const n = l.stops.length, loop = n > 2 && l.stops[0] === l.stops[n - 1];
  const seam = (a: number, b: number) => loop && a === 0 && b === n - 1;
  return l.stops.some((s, j) => Math.abs(j - i) > 1 && members.has(s) && !seam(i, j) && !seam(j, i));
}

/** The lines stopping at stop `i`'s complex (T-549), busiest there first by their boardings in the
 * past 24 hours, the line itself where it comes back there (T-1391), leaving out `drawn` (line
 * numbers drawn beside the line there, T-1168). */
function transferLines(l: LineInput, i: number, drawn?: ReadonlySet<number>): LineInput[] {
  const w = world.value;
  const members = new Set(complexOf(l.stops[i]));
  const d = today.value;
  const riders = (o: LineInput) => o.stops.reduce((n, s, k) => n + (members.has(s) ? (d.get(o.id)?.stops[k] ?? 0) : 0), 0);
  return (w?.save.lines ?? []).filter((o) => !drawn?.has(o.num) && (o.id === l.id ? selfTransfer(l, i) : o.stops.some((s) => members.has(s))))
    .map((o) => ({ o, n: riders(o) })).sort((a, b) => b.n - a.n).map((x) => x.o);
}

/** T-1786: bullets in two rows, the first row the longer; T-2016: each row on its own, so a pill above a
 * circle leaves no gap beside the circle. */
export function TwoRows({ lines }: { lines: LineInput[] }) {
  const k = Math.ceil(lines.length / 2);
  return (
    <span class="stop-chips2">
      <span>{lines.slice(0, k).map((o) => <Chip line={o} size="sm" />)}</span>
      <span>{lines.slice(k).map((o) => <Chip line={o} size="sm" />)}</span>
    </span>
  );
}

/** The busiest two as bullets, then `extra` (lines drawn beside it whose bullet shows at this row,
 * T-1391), the rest behind a "+" whose hover names them. */
export function StopTransfers({ l, i, drawn, extra = [], tall }: { l: LineInput; i: number; drawn?: ReadonlySet<number>; extra?: LineInput[]; tall?: boolean }) {
  const others = [...transferLines(l, i, drawn), ...extra];
  if (!others.length) return null;
  // T-1786: a tall row of the related-lines view has its bullets in two rows, in the bars' order
  const shown = others.slice(0, tall ? TALL_MAX : STOP_CHIPS), rest = others.slice(shown.length);
  return (
    <>
      {tall
        ? <TwoRows lines={shown} />
        : <span class="bullets">{shown.map((o) => <Chip line={o} size="sm" />)}</span>}
      {rest.length > 0 && <span class="stop-more" data-tip={[...new Set(rest.map((o) => o.name))].join("\n")}>+</span>}
    </>
  );
}

/** T-1272: in the Traffic view, the stops diagram's stretch widths, px: riders a day against the
 * line's own busiest stretch, so a quiet line varies as much as a busy one. */
const STRETCH_MIN_PX = 2, STRETCH_MAX_PX = 14;

/** One stretch of the stops diagram in the Traffic view: its width, px, the strength of its
 * crowding mark (0 for none, as the map's, `FULL_FLOOR`), and its hover. */
interface StretchLook { w: number; full: number; tip: string }

/** The map's crowding marks on line `num` (T-2236, `Renderer.fullStretch`), null for none. They
 * change with the boarding account's hour, so they are read again every half second and the panel
 * re-renders only when one changes. */
function useCrowded(num: number, stretches: number, on: boolean): (CrowdedStretch | null)[] {
  const read = () => Array.from({ length: on ? Math.max(0, stretches) : 0 }, (_, k) => live.overlay?.renderer.fullStretch(num, k) ?? null);
  const [crowded, setCrowded] = useState<(CrowdedStretch | null)[]>(read);
  useEffect(() => {
    let key = "";
    const tick = () => {
      const w = read(), k = w.map((x) => (x ? `${x.load.toFixed(3)}@${x.hour}` : "")).join(",");
      if (k !== key) { key = k; setCrowded(w); }
    };
    tick();
    if (!on) return;
    const h = setInterval(tick, 500);
    return () => clearInterval(h);
  }, [num, stretches, on]);
  return crowded;
}

/** The stretches' looks in the Traffic view, or null outside it or before demand has riders for the
 * line (the plain bar then). */
export function stretchLooks(l: LineInput, names: string[], crowded: (CrowdedStretch | null)[]): StretchLook[] | null {
  const n = l.stops.length, purpose = purposeFilter.value, dv = demandView.value;
  if (n < 2) return null;
  const day: number[] = [];
  for (let k = 0; k < n - 1; k++) {
    const v = dailyRiders(dv, l.id, l.stops, k, purpose);
    if (v === null) return null;
    day.push(v);
  }
  const most = Math.max(...day);
  const per = shownDayPhrase.value;
  return day.map((v, k) => {
    const c = crowded[k] ?? null;
    const full = c ? FULL_FLOOR + (1 - FULL_FLOOR) * Math.min(1, Math.max(0, (c.load - CROWD_LOAD_FROM) / (CROWD_LOAD_STRONG - CROWD_LOAD_FROM))) : 0;
    const tip = `${names[k]} to ${names[k + 1]}\n${v < 0.5 ? "0" : forecast(v)} riders ${per}` + (purpose === "all" ? "" : `, ${purposeName(purpose)} trips only`)
      + (c ? `\n${peakLoadText(c)}` : "");
    return { w: STRETCH_MIN_PX + (STRETCH_MAX_PX - STRETCH_MIN_PX) * (most > 0 ? v / most : 0), full, tip };
  });
}

/** One half of a stretch of the diagram, below a stop's dot (`down`) or above the next one's. */
export function StretchHalf({ s, down }: { s: StretchLook; down: boolean }) {
  // the red edges start at PANEL_MARK_FLOOR, not the map's FULL_FLOOR: with no casing to mix from,
  // a 15% edge on the panel's background cannot be seen
  const edge = s.full > 0 ? PANEL_MARK_FLOOR + ((1 - PANEL_MARK_FLOOR) * (s.full - FULL_FLOOR)) / (1 - FULL_FLOOR) : 0;
  return <span class={"stretch " + (down ? "down" : "up") + (s.full > 0 ? " full" : "")} style={{ "--w": `${s.w.toFixed(1)}px`, "--f": s.full.toFixed(3), "--e": edge.toFixed(3) }} data-tip={s.tip} />;
}
const PANEL_MARK_FLOOR = 0.5;

/** One stop's boardings in the past 24 hours (T-446), in the stops list. */
function StopToday({ line, i }: { line: string; i: number }) {
  const d = today.value.get(line);
  return <span class="n num" data-tip={d ? `${count(d.stops[i] ?? 0)} boardings in past 24h` : undefined}>{d ? count(d.stops[i] ?? 0) : ""}</span>;
}

/** The plain stops strip; a loop's bar goes on past its ends and curves round (T-1391). */
function PlainStops({ looks, colour, loop, onDblClick, children }: { looks: boolean; colour: string; loop: boolean; onDblClick: (e: MouseEvent) => void; children: ComponentChildren }) {
  const ul = useRef<HTMLUListElement>(null);
  useTightTimes(ul);
  return (
    <ul ref={ul} class={"stops" + (looks ? " traffic" : "") + (loop ? " loop" : "")} style={{ "--c": colour }} onDblClick={onDblClick}>
      {loop && <li class="loop-end top" aria-hidden="true" />}
      {children}
      {loop && <li class="loop-end bottom" aria-hidden="true" />}
    </ul>
  );
}

/** Train length (T-733): 1 car up to what the line's shortest platform fits. At the cap the line
 * stores 0, the most that fits, so it follows the platforms as a new line does. */
function TrainLength({ l, st }: { l: LineInput; st: LineStats }) {
  const [want, setWant] = useState<{ line: string; cars: number } | null>(null);
  const stations = world.value ? stationsById(world.value) : new Map<string, StationInput>();
  const plats = l.stops.map((s) => stations.get(s)).filter((s) => s !== undefined);
  if (!plats.length || !st.cars) return null;
  const short = plats.reduce((a, b) => (b.length < a.length ? b : a));
  const cap = Math.max(1, Math.floor(short.length / CAR_M));
  const cars = pendingCars.value.get(l.id)?.to ?? Math.min(st.cars, cap);
  const capText = `The ${length(short.length)} platform at ${short.name} fits ${carsText(cap)}`;
  // T-1893: the number between the buttons carries the hover; the buttons have none
  const m = moneyState.value;
  const tip = `${carsText(cars)} can carry ${(cars * RATED_PER_CAR).toLocaleString("en-US")} riders, ${(cars * CRUSH_PER_CAR).toLocaleString("en-US")} at most` + (m ? "\n" + trainCostLine(trainKind(m, st.fast)) : "");
  // T-751: a + at the cap offers to lengthen the platforms that stop it
  const step = (d: number) => {
    const r = stepLineCars(l.id, Math.min(st.cars, cap), cap, d);
    if (r === true) setWant(null);
    else if (d > 0 && r === false && cap < PLATFORM_MAX_M / CAR_M) setWant({ line: l.id, cars: cap + 1 });
    return r;
  };
  // T-1894: a typed length past the platforms' cap goes to the cap and offers to expand them for it
  const typed = (v: number) => {
    const to = Math.max(1, Math.round(v));
    if (to !== cars) void stepLineCars(l.id, Math.min(st.cars, cap), cap, to - cars);
    setWant(to > cap && cap < PLATFORM_MAX_M / CAR_M ? { line: l.id, cars: Math.min(to, Math.floor(PLATFORM_MAX_M / CAR_M)) } : null);
  };
  const offer = want && want.line === l.id && want.cars > cap ? want.cars : 0;
  return (
    <ul class="rows">
      <li>
        <span class="grow" data-tip={capText}>Train length</span>
        <Stepper value={carsText(cars)} wide label="train length" onStep={step} tipValue={tip} type={{ text: String(cars), set: typed, tip }} />
      </li>
      {offer > 0 && <LongerPlatforms l={l} st={st} stations={plats} cars={cars} to={offer} onDone={() => setWant(null)} />}
    </ul>
  );
}

/** T-751: under a train length refused for short platforms, the stops that cannot take `to` cars and
 * one button that lengthens their platforms (each a station edit, paid at once for a built station)
 * and then the trains. Refused for want of cash before anything is changed. */
function LongerPlatforms({ l, st, stations, cars, to, onDone }: { l: LineInput; st: LineStats; stations: StationInput[]; cars: number; to: number; onDone: () => void }) {
  const metres = to * CAR_M;
  const short = [...new Map(stations.filter((s) => s.length < metres).map((s) => [s.id, s])).values()];
  const key = short.map((s) => s.num).join(",");
  const cost = useClockQuery(short.length ? { what: "platformCost", nodes: short.map((s) => s.num), platform: metres } : null, [key, metres], (d) => d[0]);
  const btn = useRef<HTMLButtonElement>(null);
  const busy = useRef(false);
  const m = moneyState.value;
  const k = m && trainKind(m, st.fast);
  const trains = Math.max(...st.trainsNeeded);
  const carsPrice = k && st.status === "running" ? Math.max(0, (to - cars) * trains - k.spare) * k.carPrice : 0;
  if (!short.length) return null;
  const n = short.length;
  const price = cost === null ? null : cost;
  const names = short.length > 4 ? short.slice(0, 3).map((s) => s.name).join(", ") + ` and ${short.length - 3} more` : andList(short.map((s) => s.name));
  const go = async () => {
    if (busy.current) return;
    const have = liveMoney.value.cash, need = (price ?? 0) + carsPrice;
    if (m && m.mode !== "sandbox" && need > 1e-9 && need > have + 1e-6) {
      say(describeIssues([`Money:${need.toFixed(1)}:${have.toFixed(1)}`]), true);
      shakeEl(btn.current);
      return;
    }
    busy.current = true;
    try {
      for (let i = 0; i < short.length; i++) {
        const s = short[i];
        const a = await edit({ op: "stationProps", node: s.num, platform: Math.max(s.length, metres), name: s.name }, false, i > 0);
        if (!a.ok) return;
      }
      // the cash was checked above, and the platform edits may have bought the cars already
      stepLineCars(l.id, cars, to, 1, false);
      onDone();
    } finally {
      busy.current = false;
    }
  };
  const label = price === null ? "Expand all" : price > 1e-9 ? `${n === 1 ? "Expand it" : "Expand all"} for ${money(price * 1e6)}` : n === 1 ? "Expand it" : "Expand all";
  const tip = `Platform${n === 1 ? "" : "s"} at ${names} to ${length(metres)}, then trains to ${carsText(to)}`
    + (price !== null && price <= 1e-9 ? "\nFree while still in the blueprint" : "")
    + (carsPrice > 0 ? `\nExtra cars cost ${money(carsPrice * 1e6)}` : "");
  return (
    <li class="offer">
      <span class="grow">{n === 1 ? "1 station is" : `${n} stations are`} too short for {to}-car trains.</span>
      <button ref={btn} class="btn text" data-tip={tip} onClick={() => void go()} onAnimationEnd={() => btn.current?.classList.remove("shake")}>{label}</button>
    </li>
  );
}

/** T-2232: under the round trip, one button that adds passing tracks at every station where the
 * line's trains get passed or pass a slower line's (`TrackApi.passing_spots`), in one edit; built
 * stations pay at once, blueprint ones join the blueprint. Gone once none is missing. */
function PassingTracks({ l, version }: { l: LineInput; version: number }) {
  const spots = useClockQuery({ what: "passingSpots", line: l.num }, [l.num, version], (d) => {
    const out: { num: number; price: number }[] = [];
    for (let i = 0; i + 1 < d.length; i += 2) out.push({ num: d[i], price: d[i + 1] });
    return out;
  });
  const btn = useRef<HTMLButtonElement>(null);
  const busy = useRef(false);
  const byNum = stationsNow();
  const at = (spots ?? []).map((x) => ({ ...x, s: byNum.get(x.num) })).filter((x): x is { num: number; price: number; s: StationInput } => !!x.s);
  if (!at.length) return null;
  const built = at.filter((x) => x.s.built);
  const price = built.reduce((a, x) => a + x.price, 0);
  const n = plural(at.length, "station");
  const names = at.length > 8 ? at.slice(0, 7).map((x) => x.s.name).join(", ") + ` and ${at.length - 7} more` : andList(at.map((x) => x.s.name));
  const go = async () => {
    if (busy.current) return;
    busy.current = true;
    try {
      const a = await edit({ op: "stationUpgrade", nodes: at.map((x) => x.num), bit: UPGRADE.passing, on: true });
      if (!a.ok) { shakeEl(btn.current); return; }
      // the edit also buys the trains a slower line now needs for its headway
      const trains = a.charge - price;
      say(price > 1e-9
        ? `Passing tracks built at ${plural(built.length, "station")} for ${money(price * 1e6)}.` + (trains > 0.05 ? ` New trains cost ${money(trains * 1e6)}.` : "")
        : `Passing tracks added to the blueprint at ${n}.`);
    } finally {
      busy.current = false;
    }
  };
  return (
    <div class="line-stop-actions">
      <button ref={btn} class="btn text" data-tip={`At ${names}`} onClick={() => void go()} onAnimationEnd={() => btn.current?.classList.remove("shake")}>
        Add passing tracks at {n}{price > 1e-9 ? ` for ${money(price * 1e6)}` : ""}
      </button>
    </div>
  );
}

/** T-817: the line's kind of train, normal or high speed, and its average speed (T-1159). A
 * switch buys the line's cars of the other kind beyond that kind's spares; the cars it leaves
 * stay in the depot. */
function TrainType({ l, st, lev }: { l: LineInput; st: LineStats; lev: number }) {
  const m = moneyState.value;
  const tip = (fast: boolean) => {
    const k = m && trainKind(m, fast);
    let t = fast ? "High speed trains run up to 320 km/h on high speed track or 160 km/h on normal track" : "Normal trains run up to 160 km/h";
    const need = Math.max(...st.trainsNeeded) * st.cars;
    if (k && fast !== st.fast && st.status === "running" && need > 0) t += "\n" + carsBought(k, need);
    return t;
  };
  return (
    <ul class="rows">
      <li>
        <span class="grow">Trains</span>
        <Segmented<boolean>
          options={[
            [false, <span data-tip={tip(false)}>Normal</span>],
            [true, <span data-tip={tip(true)}>High speed</span>],
          ]}
          value={st.fast}
          onChange={(fast) => { if (fast !== st.fast) void edit({ op: "lineFast", line: l.num, fast }); }}
        />
      </li>
      {l.stops.length >= 2 && <AverageSpeed st={st} lev={lev} />}
    </ul>
  );
}

/** The hover on a stop's time: how fast the trains run from the stop before, and what holds them. */
function stretchTip(s: { reached: number; limit: number; why: number } | undefined): string | undefined {
  if (!s) return undefined;
  const lim = speed(s.limit);
  const why = s.why === 0 ? `The trains' top speed is ${lim}` : s.why === 1 ? `Normal track holds trains to ${lim}` : s.why === 2 ? `Curves hold trains to ${lim} here` : `A crossover holds trains to ${lim} here`;
  return `Trains reach ${speed(s.reached)}\n${why}` + (s.reached < s.limit - 1 ? "\nThe stops are too close together to reach it" : "");
}

/** T-1293: copy the line's schedule settings, and paste the copy on it (game/scheduleCopy.ts). */
function ScheduleCopyButtons({ l, st }: { l: LineInput; st: LineStats }) {
  const c = scheduleCopy.value;
  const fit = carsCap(l);
  const cut = c && fit && c.cars > fit.cap ? `\nThe platform at ${fit.name} is too short for ${carsText(c.cars)}. Trains here stay at ${carsText(fit.cap)}.` : "";
  return (
    <span class="sched-copy">
      <button class="btn icon mini" aria-label="Copy the schedule" data-tip={`Copy the schedule: ${scheduleText(scheduleOf(l, st))}`}
        onClick={() => copySchedule(l, st)}><Icon.copy /></button>
      <button class="btn icon mini" aria-label="Paste the schedule" disabled={!c} data-tip={c ? `Paste the schedule copied from ${c.from}: ${scheduleText(c)}${cut}` : "Paste a schedule copied from a line"}
        onClick={() => void pasteSchedule(l, st)}><Icon.paste /></button>
      <MareyButton st={st} />
    </span>
  );
}

/** One diagram stop edit at a time: a second click before the world has the first would count from stale indices. */
let stopEditing = false;

function RailLineInspector({ id }: { id: string }) {
  const w = world.value;
  const [copying, setCopying] = useState(false);
  const l = w ? linesById(w).get(id) : undefined;
  const title = useLineTitle(id, l);
  // T-309: the demand level now in the line's own local time
  const now = demandAt(lineSeconds(w?.lineStats[id]));
  const lev = DEMAND_INDEX[now];
  // where it loses time to busy track, asked again after every edit and when the level turns
  const spots = useClockQuery(l ? { what: "lineDelays", line: l.num, level: lev } : null, [l?.num, w?.version, lev], spotsFrom) ?? NO_SPOTS;
  const waitSpots = useClockQuery(l ? { what: "lineWaits", line: l.num, level: lev } : null, [l?.num, w?.version, lev], waitSpotsFrom) ?? [];
  const heldAnswer = useClockQuery(l ? { what: "lineHeld", line: l.num } : null, [l?.num, w?.version], (d) => ({ held: heldFrom(d), limits: limitsFrom(d) }));
  const held = heldAnswer?.held ?? [], limits = heldAnswer?.limits ?? NO_LIMITS;
  if (l && heldAnswer) noteHeadwayLimits(l.id, limits.map((x) => x?.s ?? 0));
  const marks = useClockQuery(l ? { what: "lineMarks", line: l.num } : null, [l?.num, w?.version], marksFrom) ?? [];
  const { kind, periods } = useLinePeriods(id, l);
  // T-646: while stops are edited, the stations the line passes between stops, and the gaps whose
  // passed stations are shown as stops to add, waiting for Add or Cancel
  // (the passed stations come with the world, so new stops never show with the old gaps, T-648;
  // the gaps picked hold for the world version they were picked in)
  const editingStops = tool.value === "line" && !!l && lineDraft.value.line === l.num;
  const [pendingAt, setPendingAt] = useState<{ v: number; gaps: number[] } | null>(null);
  useEffect(() => setPendingAt(null), [l?.num, editingStops]);
  // its day so far (T-293), kept fresh while the panel is open
  useToday();
  // T-1272: the Traffic view draws the stops diagram by load, with the map's crowding marks
  const traffic = display.value.trackColour === "traffic" && !editingStops;
  const crowded = useCrowded(l?.num ?? -1, (l?.stops.length ?? 1) - 1, traffic && display.value.fullMarks);
  if (!w || !l) return null;
  const st = w.lineStats[l.id];
  const name = stationNames(w);
  const looks = traffic && st.status === "running" ? stretchLooks(l, l.stops.map((s) => name.get(s) ?? ""), crowded) : null;
  const times = st.fromStartByLevel[lev] ?? st.fromStartS;
  const canAuto = l.stops.length >= 2 && st.status !== "broken";

  const build = async () => {
    const a = await edit({ op: "constructLine", line: l.num });
    if (a.ok) say(`${l.name} constructed for ${money(a.charge * 1e6)}. Its trains start with the next departure.`);
  };
  const passed = editingStops ? st.passed : [];
  const pending = pendingAt && pendingAt.v === w.version ? pendingAt.gaps : null;
  const setPending = (gaps: number[] | null) => setPendingAt(gaps && { v: w.version, gaps });
  const setStops = async (stops: number[]) => {
    const a = await edit({ op: "setStops", line: l.num, stops });
    if (a.ok && lineDraft.peek().line === l.num) lineDraft.value = { line: l.num, stops };
  };
  // the passed stations related lines draw as rows of their own (set as it draws them)
  let sideShown: ReadonlySet<number> = new Set();
  const addGaps = (gs: number[]) => {
    const add = passed.filter((p) => gs.includes(p.gap) && !sideShown.has(p.node));
    const stops = l.stops.flatMap((s, i) => [Number(s), ...add.filter((p) => p.gap === i).map((p) => p.node)]);
    setPending(null);
    return setStops(stops);
  };
  const addPassed = () => addGaps(pending ?? []);
  const stationsText = (n: number) => plural(n, "station");
  const removeStop = (i: number) => setStops(l.stops.filter((_, k) => k !== i).map(Number));
  const stopEdit = async (op: EditOp) => {
    if (stopEditing) return;
    stopEditing = true;
    try {
      const a = await edit(op);
      const now = world.peek()?.save.lines.find((x) => x.num === l.num);
      if (a.ok && now && lineDraft.peek().line === l.num) lineDraft.value = { line: l.num, stops: now.stops.map(Number) };
    } finally { stopEditing = false; }
  };
  const removeNow = () => {
    ask.value = null;
    void edit({ op: "removeLine", line: l.num }).then(a => {
      if (!a.ok) return;
      if (lineDraft.peek().line === l.num) live.tools?.exit();
      const selected = selection.peek();
      if (selected?.kind === "line" && selected.line === l.id) selection.value = null;
    });
  };
  const remove = () => {
    // T-2535: a line with no stops has nothing to lose, so it goes at once
    if (l.stops.length === 0) return removeNow();
    ask.value = { text: `Delete ${l.name}? Track and stations stay.`, yes: "Delete line", no: "Keep line", run: removeNow };
  };
  const copy = async () => {
    if (copying) return;
    setCopying(true);
    try {
      const a = await edit({ op: "copyLine", line: l.num });
      if (!a.ok || a.line === undefined) return;
      // T-1894: the copy's own panel opens with its stops being edited
      const made = world.peek()?.save.lines.find((x) => x.num === a.line);
      if (made) toggleEditStops(made);
    } finally { setCopying(false); }
  };

  const shown = title.shown;
  const nStops = l.stops.length;
  // T-1894: north (or for an east-west line, west) end on top; a loop as built
  const byId = stationsById(w);
  const flip = drawnReversed(byId.get(l.stops[0]), byId.get(l.stops[nStops - 1]), !!st.loop);
  // T-1894: while stops are edited a click on a stop's dot takes it out keeping the path, Shift-click
  // ends the line there, as in the group view; a stop the line needs has no click
  const dotClick = (i: number): { tip: string; click: (e: MouseEvent) => void } | null => {
    const end = i === 0 || i === nStops - 1, loop = !!st.loop;
    if (!editingStops || nStops <= 2 || st.reversals?.includes(i) || (loop && (end || nStops <= 3))) return null;
    const here = name.get(l.stops[i]) ?? "";
    return {
      tip: `Click to remove the stop at ${here}` + (end || loop || nStops < 3 ? "" : "\nShift-click to end the line here"),
      click: (e) => void stopEdit(e.shiftKey && !end && !loop ? { op: "endLineAt", line: l.num, index: i } : { op: "dropStop", line: l.num, index: i }),
    };
  };
  // the station the line passes at a side line's stop (related lines while stops are edited)
  const adder = (station: string): number | null => {
    const members = new Set(complexOf(station));
    return passed.find((x) => members.has(String(x.node)))?.node ?? null;
  };
  // T-1168: stop `i`'s row; `drawn` (in the related-lines view) are the lines drawn beside the line
  // there, `bullets` those of them whose bullet shows here
  const stopRow = (i: number, drawn?: ReadonlySet<number>, bullets?: LineInput[]) => {
    const s = l.stops[i];
    const xfer = complexOf(s).some((m) => linesAt(m).some((o) => o.id !== l.id)) || selfTransfer(l, i);
    const tall = !!drawn && transferLines(l, i, drawn).length > ROUND_MAX;
    // the stretches above and below the row, and the time from the stop on top
    const above = flip ? (i < nStops - 1 ? i : -1) : i - 1, below = flip ? i - 1 : (i < nStops - 1 ? i : -1);
    const total = times[nStops - 1] ?? 0;
    const t = flip ? (i < nStops - 1 && times[i] !== undefined ? total - times[i] : undefined) : i > 0 ? times[i] : undefined;
    const dot = dotClick(i);
    return (
      <li class={(xfer ? "xfer" : "") + (tall ? " tall" : "")} data-r={drawn ? true : undefined}>
        {looks && !drawn && above >= 0 && looks[above] && <StretchHalf s={looks[above]} down={false} />}
        {looks && !drawn && below >= 0 && looks[below] && <StretchHalf s={looks[below]} down />}
        {dot && <span class="stop-hit" data-tip={dot.tip} onClick={dot.click} />}
        <span class="grow">
          <span class="stop-name" onClick={() => (selection.value = { kind: "station", station: s })}>
            {name.get(s)}
          </span>
          {t !== undefined && <span class="t muted num" data-tip={stretchTip(st.stretches[above])}>{duration(t)}</span>}
          {st.reversals?.includes(i) && (
            <span class="rev-stop" data-tip={`Trains reverse here and stand ${duration(st.turnaroundS)}\nAt most 12 trains an hour reverse on each side`}>
              <Icon.uturn />
            </span>
          )}
        </span>
        <StopTransfers l={l} i={i} drawn={drawn} extra={bullets} tall={tall} />
        <StopToday line={l.id} i={i} />
        <RouteMarks marks={flip ? marks.filter((m) => m.gap === i - 1).reverse() : marks.filter((m) => m.gap === i)} />
        {editingStops && (
          <button class="btn icon mini danger" data-tip={st.loop && i > 0 && i === l.stops.length - 1 ? "Open the loop" : "Take this stop out of the line"} aria-label={`Remove stop ${name.get(s)}`} onClick={() => removeStop(i)}>
            <Icon.cross />
          </button>
        )}
      </li>
    );
  };
  const related = relatedLines.value && l.stops.length >= 2;
  // T-1391: a loop shows each station once, its bar going on past both ends; while its stops are
  // edited the last row stays, holding the button that opens the loop
  const loopOnce = !!st.loop && !editingStops && l.stops.length > 2;
  // T-646: after stop `i`, while stops are edited, the stations the line passes before the next
  const afterStop = (i: number, shown?: ReadonlySet<number>) => {
    // T-1894: a station with its own row in related lines is added there, not here
    if (shown) sideShown = shown;
    const here = editingStops ? passed.filter((p) => p.gap === i && !sideShown.has(p.node)) : [];
    if (!here.length || i + 1 >= l.stops.length) return null;
    if (flip) here.reverse();
    // Add and Cancel take the clicked row's place, Add stretched over where the pointer is, so a double
    // click lands on Add; the stations are listed below (T-1489)
    return pending?.includes(i)
      ? <>
        <li class="passed confirm">
          <button class="btn text mini grow" onClick={() => void addPassed()}>Add {stationsText(here.length)}</button>
          <button class="btn text mini" onClick={() => setPending(null)}>Cancel</button>
        </li>
        {here.map((p) => (
          <li class="adding"><span class="grow"><span class="stop-name">+ {name.get(String(p.node)) ?? ""}</span></span></li>
        ))}
      </>
      : !pending && (
        <li class="passed">
          <button class="btn text mini" data-tip={`Add ${andList(here.map((p) => name.get(String(p.node)) ?? ""))}\nDouble-click to add them at once`} onClick={() => setPending([i])} onDblClick={() => void addGaps([i])}>
            <Icon.plus />{stationsText(here.length)} passed
          </button>
        </li>
      );
  };
  const onListDblClick = (e: MouseEvent) => { if (pending && (e.target as Element).closest("li.adding")) void addPassed(); };
  // T-670: the stops editor sits at the top while stops are being edited, else after the day's figures
  const stopsBlock = (
    <>
      <div class="h divided">
        Stops
        {l.stops.length >= 2 && (
          <button class={"btn icon mini related-toggle" + (related ? " on" : "")} data-tip={related ? "Hide related lines" : "Show related lines"}
            aria-label={related ? "Hide related lines" : "Show related lines"} aria-pressed={related} onClick={() => (relatedLines.value = !relatedLines.value)}>
            <Icon.related />
          </button>
        )}
        <span class="right muted small">boardings today</span>
      </div>
      {related ? (
        <div onDblClick={onListDblClick}>
          <RelatedStops l={l} stopS={st.stopS[0]} colour={shown} loop={loopOnce} flip={flip} lRow={stopRow} after={editingStops ? afterStop : undefined}
            adder={editingStops ? adder : undefined} onAdd={(node) => void stopEdit({ op: "stopPassed", line: l.num, station: node })} transfers={(i, drawn) => transferLines(l, i, drawn)} />
        </div>
      ) : <PlainStops looks={!!looks} colour={shown} loop={loopOnce} onDblClick={onListDblClick}>
        {flip
          ? l.stops.map((_, k) => { const i = nStops - 1 - k; return <>{stopRow(i)}{i > 0 && afterStop(i - 1)}</>; })
          : l.stops.map((_, i) => (loopOnce && i === nStops - 1 ? null : <>
            {stopRow(i)}
            {afterStop(i)}
          </>))}
      </PlainStops>}
    </>
  );

  return (
    <>
      <LineTitle l={l} t={title} />
      <div class="sub muted flush">
        {st.fast && "High-speed, "}{plural(l.stops.length - (st.loop ? 1 : 0), "stop")}, {distance(st.km * 1000)}
      </div>
      {st.status === "planned" && l.stops.length >= 2 && (
        <div class="callout">
          <p>Planned</p>
          <ConstructButton num={l.num} build={build} what="Track and stations" vehicles="trains" runs="Trains run" />
        </div>
      )}
      {st.status === "broken" && (
        <ul class="alerts"><Alert>Track this line ran on is gone</Alert></ul>
      )}

      <div class="line-stop-actions">
        <button class={"btn text" + (editingStops ? " on" : "")} data-guide={editingStops ? undefined : "stops"} data-tip={editingStops ? "Finish editing stops (S)" : "Click stations on the map to add stops to the line (S)"} onClick={() => toggleEditStops(l)}><Icon.pencil /><SwapLabel on={editingStops} off="Edit stops" onText="Stop editing" /></button>
        <button class="btn text" onClick={() => void copy()} disabled={copying}>Copy line</button>
        <button class="btn text danger" onClick={remove}>Delete line</button>
      </div>
      {editingStops && stopsBlock}

      <Schedule l={l} st={st} now={now} limits={limits} crowd={periods} canAuto={canAuto} copy={<ScheduleCopyButtons l={l} st={st} />} />
      {st.status !== "broken" && <TrainType l={l} st={st} lev={lev} />}
      {st.status !== "broken" && <TrainLength l={l} st={st} />}
      {st.status === "running" && held.length > 0 && (
        <ul class="alerts">
          {[...new Set(held.map((h) => h.place))].map((place) => {
            const hs = held.filter((h) => h.place === place), h0 = hs[0];
            return (
              <Alert tip={h0.tip}>
                {h0.head}{h0.at ? <span class="place-link" onClick={() => { locateStation(h0.at!.lng, h0.at!.lat); selection.value = { kind: "station", station: h0.at!.id }; }}>{h0.name}</span> : h0.name}{h0.tail}
                {hs.map((h) => <div class="detail" data-tip={h.tip + "\n" + heldTip}>{DEMAND_LABEL[h.d]} demand: {heldText(h.wait)}</div>)}
              </Alert>
            );
          })}
        </ul>
      )}

      {st.status === "running" && <PastDay l={l} st={st} periods={periods} kind={kind} extra={<PurposeSplit scope={[{ line: l.id }]} unit="riders" />} />}

      {!editingStops && stopsBlock}
      {st.status !== "broken" && l.stops.length >= 2 && <RoundTrip st={st} lev={lev} stops={l.stops.length} spots={spots} waitSpots={waitSpots} />}
      {st.status !== "broken" && l.stops.length >= 2 && <PassingTracks l={l} version={w.version} />}
    </>
  );
}

/** A bus line (T-1464) has its own, shorter panel; a rail line's is above. */
export function LineInspector({ id }: { id: string }) {
  return world.value?.lineStats[id]?.bus ? <BusInspector id={id} /> : <RailLineInspector id={id} />;
}
