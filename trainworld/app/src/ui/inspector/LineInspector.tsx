// The line inspector (SPEC 6.3, 8).

import type { ComponentChildren } from "preact";
import { useEffect, useRef, useState } from "preact/hooks";
import { nearestStation } from "../../game/places";
import { demandAt, lineSeconds } from "../../game/clock";
import { andList, clockTime, count, distance, duration, headway, length, money, plural, runCostRule, speed, trainKind, trainRunCost } from "../../game/format";
import { crowdColour, LINE_PALETTE } from "../../game/palette";
import { levelHours, PERIODS } from "../../game/periods";
import { crowdGrade, crowding, runningToday, today, useToday, waitText } from "../../game/today";
import { accountVersion, CAR_M, CRUSH_PER_CAR } from "../../game/boarding";
import { complexOf } from "../../game/complexView";
import { DAY_KIND_NAME } from "../../game/days";
import { kindIn, lineCountry } from "../../game/holidays";
import { DEMAND_INDEX, DEMANDS, type Demand, type LineInput, type LineStats, type StationInput } from "../../game/types";
import { ask, lineDraft, renameLinePrompt, selection, tool } from "../../game/ui";
import { liveMoney, moneyState } from "../../game/money";
import { fullParts, fullTip, fullTitle, heldText, heldTip } from "../../game/capacityKinds";
import { toggleEditStops } from "../../game/lineActions";
import { ColourPicker } from "../ColourPicker";
import { AUTO_SLACK, autoTrainsFor, edit, headwayFromMinutes, headwayShown, linesAt, nextHeadway, refusedForCash, renameLine, say, setAutoTrains, setHeadway, stationNames, stationsById, stationsByNum, stepHeadway, stepTrainCount, trainsShown, world, linesById, linesByNum } from "../../game/world";
import { live } from "../../map/live";
import { Alert, Chip, Icon, Row, Segmented, Stepper } from "../widgets";
import { PurposeSplit } from "./PurposeSplit";
import { NameEdit } from "./NameEdit";
import { locateLine, locateStation } from "../locate";
import { demandPending, Spinner } from "../Spinner";
import { useClockQuery } from "../useClockQuery";

const DEMAND_LABEL: Record<Demand, string> = { high: "High", medium: "Medium", low: "Low" };

/** When a demand level runs, in the line's local time (the hour spans of game/periods.ts): "High
 * demand: 06:00 to 10:00, 16:00 to 20:00". */
function levelTimes(d: Demand): string {
  const t = (h: number) => clockTime((h * 60) % 1440);
  return `${DEMAND_LABEL[d]} demand: ${levelHours(d).map(([a, b]) => `${t(a)} to ${t(b)}`).join(", ")}`;
}

const stationsNow = (): ReadonlyMap<number, StationInput> => (world.value ? stationsByNum(world.value) : new Map());

/** Where a line waits, in words (T-031): from `TrackApi.line_delays`. */
function spotsFrom(d: Float64Array): { text: string; s: number }[] {
  const out: { text: string; s: number }[] = [];
  const st = stationsNow();
  for (let i = 0; i + 4 < d.length; i += 5) {
    const [kind, node, x, y, s] = [d[i], d[i + 1], d[i + 2], d[i + 3], d[i + 4]];
    const near = nearestStation(x, y)?.name;
    const at = node >= 0 ? st.get(node)?.name : undefined;
    const text =
      (kind === 2 || kind === 3 || kind === 6 || kind === 8) && at
        ? `At ${at}`
        : kind === 4
          ? `At the junction${near ? " near " + near : ""}`
          : kind === 5
            ? `At the crossing${near ? " near " + near : ""}`
            : `On the track${near ? " near " + near : ""}`;
    const same = out.find((o) => o.text === text); // both directions, or two stretches by one station
    if (same) same.s += s;
    else out.push({ text, s });
  }
  return out.sort((a, b) => b.s - a.s).slice(0, 3);
}

/** Where full track holds a line's fleet back, per level (T-517): from `TrackApi.line_held`. */
interface Held { d: Demand; wait: number; place: string; head: string; name: string; tail: string; at: { lng: number; lat: number } | null; tip: string }
function heldFrom(v: Float64Array): Held[] {
  const st = stationsNow();
  const out: Held[] = [];
  DEMANDS.forEach((d, lev) => {
    const [wait, kind, node, x, y, , most] = v.subarray(7 * lev, 7 * lev + 7);
    if (!(wait > 0)) return;
    const s = (node >= 0 ? st.get(node) : undefined) ?? nearestStation(x, y);
    const name = s?.name ?? "";
    out.push({ d, wait, place: fullTitle(kind, name), ...fullParts(kind, name), name, at: s ? { lng: s.lng, lat: s.lat } : null, tip: fullTip(kind, most) });
  });
  return out;
}

/** What limits a line per level (T-679): its shortest headway in seconds with the other lines as
 * they run, and the spot named as the capacity alerts name it ("Terminus at West"); null where
 * nothing does. From the tail of `TrackApi.line_held`. */
interface Limit { s: number; what: string }
function limitsFrom(v: Float64Array): (Limit | null)[] {
  const st = stationsNow();
  return DEMANDS.map((_, lev) => {
    const [s, kind, node, x, y] = v.subarray(21 + 5 * lev, 26 + 5 * lev);
    if (!(s > 0)) return null;
    const name = ((node >= 0 ? st.get(node) : undefined) ?? nearestStation(x, y))?.name ?? "";
    return { s, what: fullParts(kind, name).head + name };
  });
}

/** "Terminus at West takes one train every 4m00s at most", or with no room left "... is full" */
const limitText = (lim: Limit) => (Number.isFinite(lim.s) ? `${lim.what} takes one train every ${duration(lim.s)} at most` : `${lim.what} is full`);

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
function RoundTrip({ st, lev, stops, spots }: { st: LineStats; lev: number; stops: number; spots: { text: string; s: number }[] }) {
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
  const running = Math.max(0, total - dwell - turn - reversing - delay);
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
        {delay > 0 && <Row label="Waiting at busy track">{duration(delay)}</Row>}
      </ul>
      {delay > 0 && spots.length > 0 && (
        <ul class="rows sub-rows">
          {spots.map((p) => (
            <Row label={p.text}>
              <span class="muted">{duration(p.s)}</span>
            </Row>
          ))}
        </ul>
      )}
    </>
  );
}

/** The line's average speed end to end: the distance both ways over the round trip less the turning
 * at the ends, so it counts the stops and the waiting at busy track but not the layover. */
function AverageSpeed({ st, lev }: { st: LineStats; lev: number }) {
  const rt = st.roundTripByLevel[lev] ?? st.roundTripS;
  // T-798: a loop once round in its lap, less a balloon's turning
  const s = st.loop ? rt / 2 - (st.loop === "balloon" ? st.turnaroundS : 0) : rt - 2 * st.turnaroundS;
  if (!(s > 0) || !(st.km > 0)) return null;
  const km = st.loop ? st.km : 2 * st.km;
  const kmh = km / (s / 3600);
  const tip = st.loop
    ? `${distance(km * 1000)} once round in ${duration(s)}${st.loop === "balloon" ? ", the lap less the turning at the end" : ""}. Dwell at stops and waiting at busy track are in it.`
    : `${distance(km * 1000)} there and back in ${duration(s)}, the round trip less the turning at the ends. Dwell at stops and waiting at busy track are in it, so it is the average speed of both directions together.`;
  return (
    <li data-tip={tip}>
      <span class="grow">Average speed</span>
      <span class="val">{speed(kmh)}</span>
    </li>
  );
}

/** The line's past 24 hours (T-446, game/today.ts): riders, fares less running cost, and its trains
 * from the boarding account (T-230): how many left full, and who was left on the platform. Its own
 * component: it re-renders as the counts move. */
function PastDay({ l, st, periods, kind }: { l: LineInput; st: LineStats; periods: Float64Array | null; kind: number }) {
  const d = today.value.get(l.id);
  if (!d) return null;
  const load = periods ? averageLoad(periods, st) : null;
  const run = runningToday(l, st);
  const m = moneyState.value;
  const k = m && trainKind(m, st.fast);
  const rule = k ? `\nRunning costs ${runCostRule(k)}: ${trainRunCost(k, st.cars)} for this line's ${carsText(st.cars)}` : "";
  return (
    <>
      <div class="h divided">
        <span data-tip="The past 24 hours, counted train by train. Whoever does not fit on a full train waits for the next one.">Past 24 hours</span>
      </div>
      <ul class="rows">
        <Row label="Riders">{count(d.riders)}</Row>
        <PurposeSplit scope={[{ line: l.id }]} unit="riders" />
        <li data-tip={`Fares ${money(d.fares)} minus running cost ${money(run)} for the trains and upkeep of its track and stations ${money(st.upkeep)}: ${money(d.fares - run - st.upkeep)} net${rule}\nUpkeep of track and stations several lines use is shared by their trains a day.`}>
          <span class="grow">Profit</span>
          <span class="val">{money(d.fares)} −<span class="crowd-bad">{money(run + st.upkeep)}</span></span>
        </li>
        {load !== null && (
          <li data-tip={`Riders aboard on average, as a share of the ${(st.cars * CRUSH_PER_CAR).toLocaleString("en-US")} a train holds at crush, over every trip on a ${DAY_KIND_NAME[kind].toLowerCase()}. Each trip counts for as long as it runs.`}>
            <span class="grow">Average load</span>
            <span class="val">{Math.round(load * 100)}%</span>
          </li>
        )}
        {d.fullTrains > 0 && <Row label="Trains leaving full"><span data-tip={`${count(d.fullTrains)}/${count(d.trains)} trains left full`}>{count(d.fullTrains)}/{count(d.trains)}</span></Row>}
        <li data-tip={`${count(d.leftBehind)} of the ${count(d.riders + d.leftBehind)} riders who wanted a train were left at the platform by a full one`}>
          <span class="grow">Riders left at platform</span>
          <span class="val">{count(d.leftBehind)}/{count(d.riders + d.leftBehind)}</span>
        </li>
      </ul>
    </>
  );
}

/** Numbers per period in `TrackApi.line_periods` (`PERIOD_STRIDE` in boarding.rs). */
const PERIOD_STRIDE = 11;

/** T-820: the day's rider-seconds aboard over what the trips could hold at crush, each trip taken
 * as its level's time from the first stop to the last; null with no trips. */
function averageLoad(v: Float64Array, st: LineStats): number | null {
  let aboard = 0, room = 0;
  for (let i = 0; i < PERIODS.length; i++) {
    const o = i * PERIOD_STRIDE;
    if (!(v[o] > 0)) continue;
    const times = st.fromStartByLevel[DEMAND_INDEX[PERIODS[i].demand]] ?? st.fromStartS;
    aboard += v[o + 5];
    room += v[o] * (times[times.length - 1] ?? 0) * CRUSH_PER_CAR * st.cars;
  }
  return room > 0 ? Math.min(1, aboard / room) : null;
}

/** T-477, T-515, T-1255: one dot per demand level, graded by `crowding` of the riders at the
 * platforms where trains come full (`BUSY_PLATFORM_S` in boarding.rs: their wait for a train with
 * room, a rider turned away counted at 30 minutes) from the boarding account: the level's periods
 * averaged by those riders. The hover names the stop with the most rider-minutes waited
 * (`line_periods`' tail, per period and stop). */
function CrowdDots({ v, d, stops }: { v: Float64Array; d: Demand; stops: string[] }) {
  let trains = 0, full = 0, weight = 0, wait = 0, score = 0, gone = 0;
  let worst: { name: string; wait: number; score: number } | null = null;
  const n = stops.length, tail = PERIODS.length * PERIOD_STRIDE, at = new Array<number>(n).fill(0);
  for (let i = 0; i < PERIODS.length; i++) {
    if (PERIODS[i].demand !== d) continue;
    const o = i * PERIOD_STRIDE;
    if (!(v[o] > 0)) continue;
    const c = crowding(v[o + 9] ?? 0, v[o + 10] ?? 0, v[o + 5]);
    trains += v[o]; full += v[o + 1]; weight += c.weight; gone += v[o + 8] ?? 0;
    wait += c.wait * c.weight; score += c.score * c.weight;
    if (!worst || c.score > worst.score) worst = { name: PERIODS[i].name, wait: c.wait, score: c.score };
    if (v.length >= tail + PERIODS.length * n) for (let k = 0; k < n; k++) at[k] += v[tail + i * n + k];
  }
  if (!(trains > 0)) return null;
  const avg = weight > 0 ? { wait: wait / weight, score: score / weight } : { wait: 0, score: 0 };
  const grade = crowdGrade(avg.score);
  const most = at.reduce((b, x, k) => (x > at[b] ? k : b), 0);
  const tip = (avg.wait >= 0.05 ? `Where trains come full, riders face a wait of ${waitText(avg.wait)} on average for a train with room` : "Riders get on the first train with room")
    + `\n${count(full)}/${count(trains)} trains are full`
    + (gone >= 1 ? `\n${count(gone)} riders found the platform too full and went another way` : "")
    + (worst && crowdGrade(worst.score) > grade ? `\n${worst.name}: ${waitText(worst.wait)}` : "")
    + (avg.wait >= 0.05 && at[most] > 0 && stops[most] ? `\nMost of the waiting is at ${stops[most]}` : "");
  return <i class="crowd-dot" style={{ background: crowdColour(avg.wait) }} data-tip={tip} />;
}

/** Bullets shown at a transfer stop; the rest sit behind a "+". */
const STOP_CHIPS = 2;

/** The other lines at a stop's station complex (T-549), the busiest two as bullets by their
 * boardings there in the past 24 hours, the rest behind a "+" whose hover names them. */
function StopTransfers({ line, stop }: { line: string; stop: string }) {
  const w = world.value;
  const members = new Set(complexOf(stop));
  const d = today.value;
  const riders = (o: LineInput) => o.stops.reduce((n, s, i) => n + (members.has(s) ? (d.get(o.id)?.stops[i] ?? 0) : 0), 0);
  const others = (w?.save.lines ?? []).filter((o) => o.id !== line && o.stops.some((s) => members.has(s)))
    .map((o) => ({ o, n: riders(o) })).sort((a, b) => b.n - a.n).map((x) => x.o);
  if (!others.length) return null;
  const rest = others.slice(STOP_CHIPS);
  return (
    <>
      {others.slice(0, STOP_CHIPS).map((o) => <Chip line={o} size="sm" />)}
      {rest.length > 0 && <span class="stop-more" data-tip={[...new Set(rest.map((o) => o.name))].join("\n")}>+</span>}
    </>
  );
}

/** One stop's boardings in the past 24 hours (T-446), in the stops list. */
function StopToday({ line, i }: { line: string; i: number }) {
  const d = today.value.get(line);
  return <span class="n num" data-tip={d ? `${count(d.stops[i] ?? 0)} boardings in the past 24 hours` : undefined}>{d ? count(d.stops[i] ?? 0) : ""}</span>;
}

const carsText = (n: number) => plural(n, "car");

/** What buying the cars a change needs costs, or that the depot's spare ones cover it. */
const carsBought = (k: { carPrice: number; spare: number }, cars: number): string => {
  const bought = Math.max(0, cars - k.spare);
  return bought > 0 ? `Buys ${carsText(bought)} for ${money(bought * k.carPrice * 1e6)}` : `Uses ${carsText(cars)} from the depot`;
};

/** What a stepper's button leads to: the train count, and for a running line the cars it needs
 * beyond the line's busiest level, bought at the car price unless the depot has spare ones, or
 * the cars it frees, which stay owned for any line to use (T-631). */
function changeTip(counts: [number, number, number], lev: number, to: number, st: LineStats): string {
  const head = `${to} ${to === 1 ? "train" : "trains"}`;
  const m = moneyState.value;
  if (!m || st.status !== "running" || !st.cars) return head;
  const next = [...counts];
  next[lev] = to;
  const cars = (Math.max(...next) - Math.max(...counts)) * st.cars;
  if (cars > 0) {
    return head + "\n" + carsBought(trainKind(m, st.fast), cars);
  }
  return cars < 0 ? head + `\nFrees ${carsText(-cars)}, which stay in the depot` : head;
}

/** Train length (T-733): 1 car up to what the line's shortest platform fits. At the cap the line
 * stores 0, the most that fits, so it follows the platforms as a new line does. */
function TrainLength({ l, st }: { l: LineInput; st: LineStats }) {
  const stations = world.value ? stationsById(world.value) : new Map<string, StationInput>();
  const plats = l.stops.map((s) => stations.get(s)).filter((s) => s !== undefined);
  if (!plats.length || !st.cars) return null;
  const short = plats.reduce((a, b) => (b.length < a.length ? b : a));
  const cap = Math.max(1, Math.floor(short.length / CAR_M));
  const cars = Math.min(st.cars, cap);
  const capText = `The ${length(short.length)} platform at ${short.name} fits ${carsText(cap)}`;
  const tip = (to: number) => {
    let t = `${carsText(to)}, ${(to * CRUSH_PER_CAR).toLocaleString("en-US")} riders a train at most`;
    const m = moneyState.value, trains = Math.max(...st.trainsNeeded);
    const k = m && trainKind(m, st.fast);
    if (k) t += `\nRunning costs ${trainRunCost(k, to)}${to !== cars ? `, ${trainRunCost(k, cars)} now` : ""}`;
    if (k && st.status === "running" && to > cars) {
      t += "\n" + carsBought(k, (to - cars) * trains);
    }
    return t + "\n" + capText;
  };
  const step = (d: number) => {
    const to = Math.max(1, Math.min(cap, cars + d));
    if (to !== cars) void edit({ op: "lineCars", line: l.num, cars: to >= cap ? 0 : to });
  };
  return (
    <ul class="rows">
      <li>
        <span class="grow" data-tip={capText}>Train length</span>
        <Stepper value={carsText(cars)} wide label="train length" onStep={step} tipDown={cars > 1 ? tip(cars - 1) : capText} tipUp={cars < cap ? tip(cars + 1) : capText} />
      </li>
    </ul>
  );
}

/** T-817: the line's kind of train, ordinary or high speed, and its average speed (T-1159). A
 * switch buys the line's cars of the other kind beyond that kind's spares; the cars it leaves
 * stay in the depot. */
function TrainType({ l, st, lev }: { l: LineInput; st: LineStats; lev: number }) {
  const m = moneyState.value;
  const tip = (fast: boolean) => {
    const k = m && trainKind(m, fast);
    let t = fast ? "High-speed trains run up to 320 km/h on high-speed track and 160 km/h on other track" : "Ordinary trains run up to 160 km/h";
    if (!k) return t;
    t += `\nA car costs ${money(k.carPrice * 1e6)}. Running costs ${runCostRule(k)}`;
    const need = Math.max(...st.trainsNeeded) * st.cars;
    if (fast !== st.fast && st.status === "running" && need > 0) {
      t += "\n" + carsBought(k, need);
      t += `\nThe ${carsText(need)} running now stay in the depot`;
    }
    return t;
  };
  return (
    <ul class="rows">
      <li>
        <span class="grow">Trains</span>
        <Segmented<boolean>
          options={[
            [false, <span data-tip={tip(false)}>Ordinary</span>],
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
  const why = s.why === 0 ? `The trains' top speed is ${lim}` : s.why === 1 ? `Ordinary track holds trains to ${lim}` : s.why === 2 ? `Curves hold trains to ${lim} here` : `A crossover holds trains to ${lim} here`;
  return `Trains reach ${speed(s.reached)}\n${why}` + (s.reached < s.limit - 1 ? "\nThe stops are too close together to reach it" : "");
}

const headwayText = (s: number) => (s > 0 ? `${+(s / 60).toFixed(2)} min` : "none");

/** How long a refused edit's control shakes, ms (T-625; style.css `.shake`) */
const SHAKE_MS = 300;

/** One demand level's trains and headway cells: a count stepper and the headway it gives, or (T-625)
 * the headway stepper and the count it gives. A count an edit was refused for cash is red, and
 * its control shakes once; a headway shorter than `limit` allows is red (T-625, T-679). */
function LevelCells({ l, st, d, auto, limit }: { l: LineInput; st: LineStats; d: Demand; auto: [number, number, number] | null; limit: Limit | null }) {
  const lev = DEMAND_INDEX[d];
  const counts = auto ? st.trainsNeeded : (trainsShown(l.id) ?? st.trainsNeeded);
  const trip = st.roundTripByLevel[lev];
  const r = refusedForCash.value;
  const refused = r && r.id === l.id && r.levs[lev] ? r : null;
  const broke = refused !== null && liveMoney.value.cash < refused.need;
  const shake = refused !== null && performance.now() - refused.at < SHAKE_MS;
  const wrap = (child: ComponentChildren) => <span key={refused?.at ?? 0} class={shake ? "shake" : undefined}>{child}</span>;
  const n = <span class={broke ? "bad" : undefined}>{counts[lev]}</span>;
  // T-798: a loop's trains are half each way, so its count steps by two
  const k = st.loop ? 2 : 1;
  const loop = !!st.loop;
  if (!auto) {
    const over = limit !== null && counts[lev] > 0 && counts[lev] - trip / limit.s > 0.01;
    return (
      <>
        <td>
          {wrap(<Stepper value={n} label="trains on the line" jump onStep={(s) => stepTrainCount(l.id, d, s)}
            tipDown={changeTip(counts, lev, Math.max(0, counts[lev] - k), st)} tipUp={changeTip(counts, lev, counts[lev] + k, st)} />)}
        </td>
        <td>{over ? <span class="bad" data-tip={`${limitText(limit)}, so the trains past that wait at the ends`}>{headway(l.tph[d])}</span> : headway(l.tph[d])}</td>
      </>
    );
  }
  const lim = limit?.s ?? 0;
  const short = limit !== null && auto[lev] > 0 && auto[lev] * AUTO_SLACK < limit.s;
  const real = l.tph[d] > 0 ? 3600 / l.tph[d] : 0;
  const clamped = !short && limit !== null && auto[lev] > 0 && real > auto[lev] * AUTO_SLACK + 0.5;
  const typeTip = "Click to type a headway in minutes. 0 is no service.";
  return (
    <>
      <td>
        <span class={"num" + (broke ? " bad" : "")} data-tip={loop
          ? `${counts[lev]} trains on the line, ${counts[lev] / 2} each way, keep the headway over its lap of ${duration(trip / 2)}`
          : `${counts[lev]} ${counts[lev] === 1 ? "train" : "trains"} on the line keep the headway over its round trip of ${duration(trip)}`}>{counts[lev]}</span>
      </td>
      <td>
        {clamped && <span class="hw-real" data-tip={`${limitText(limit)}, so ${counts[lev]} ${counts[lev] === 1 ? "train runs" : "trains run"} every ${duration(real)}`}>{headwayText(real)}</span>}
        {wrap(<Stepper wide label="headway" value={short ? <span class="bad">{headwayText(auto[lev])}</span> : headwayText(auto[lev])} onStep={(s) => stepHeadway(l.id, d, s)}
          tipDown={changeTip(counts, lev, autoTrainsFor(trip, nextHeadway(auto[lev], -1), lim, loop), st)}
          tipUp={changeTip(counts, lev, autoTrainsFor(trip, nextHeadway(auto[lev], 1), lim, loop), st)}
          type={{ text: auto[lev] ? String(+(auto[lev] / 60).toFixed(2)) : "0", set: (v) => setHeadway(l.id, d, headwayFromMinutes(v)), tip: short ? `${limitText(limit)}, so the trains past that wait at the ends.\n${typeTip}` : typeTip }} />)}
      </td>
    </>
  );
}

const NO_LIMITS: (Limit | null)[] = [null, null, null];

/** T-672: dragging a colour shows it at once (`setPreview`), and the line on the map follows with
 * one edit every 0.2 s at most; letting go sends the last colour. */
function useLiveColour(l: LineInput | undefined, setPreview: (hex: string | null) => void) {
  const live = useRef({ hex: null as string | null, timer: 0, last: 0, busy: false, gap: 200 });
  useEffect(() => () => clearTimeout(live.current.timer), []);
  const sendLive = () => {
    const s = live.current;
    s.timer = 0;
    if (s.hex === null || !l) return;
    if (s.busy) { s.timer = window.setTimeout(sendLive, 50); return; }
    const hex = s.hex;
    s.hex = null;
    s.busy = true;
    s.last = performance.now();
    void edit({ op: "lineLook", line: l.num, name: l.name, colour: hex }, true).then(() => {
      s.busy = false;
      s.gap = Math.max(200, 6 * (performance.now() - s.last)); // a slow machine follows less often, so edits take a fraction of its time
    });
  };
  const dragColour = (hex: string) => {
    setPreview(hex);
    const s = live.current;
    s.hex = hex;
    if (!s.timer) s.timer = window.setTimeout(sendLive, Math.max(0, s.gap - (performance.now() - s.last)));
  };
  const commitColour = (hex: string) => {
    if (!l) return;
    clearTimeout(live.current.timer);
    live.current.timer = 0;
    live.current.hex = null;
    setPreview(hex);
    void edit({ op: "lineLook", line: l.num, name: l.name, colour: hex }).then((a) => { if (!a.ok) setPreview(null); });
  };
  return { dragColour, commitColour };
}

export function LineInspector({ id }: { id: string }) {
  const w = world.value;
  // T-670: one editor for the name, symbol and colour; T-672: the colour a drag shows before the edit is sent
  const [editFor, setEditFor] = useState<string | null>(null);
  const [previewOf, setPreviewOf] = useState<{ id: string; hex: string } | null>(null);
  const editing = editFor === id;
  const setEditing = (open: boolean) => setEditFor(open ? id : null);
  const preview = previewOf?.id === id ? previewOf.hex : null;
  const setPreview = (hex: string | null) => setPreviewOf(hex === null ? null : { id, hex });
  const [copying, setCopying] = useState(false);
  const l = w ? linesById(w).get(id) : undefined;
  const { dragColour, commitColour } = useLiveColour(l, setPreview);
  const [symbol, setSymbol] = useState("");
  useEffect(() => { setSymbol(l?.letter ?? ""); }, [l?.letter]);
  useEffect(() => { if (preview !== null && l && l.colour.toLowerCase() === preview.toLowerCase()) setPreview(null); }, [l?.colour, preview]);
  // a click away, or Esc outside the name box (it handles its own), closes the editor
  useEffect(() => {
    if (!editing) return;
    const away = (e: PointerEvent) => { if (!(e.target as Element | null)?.closest?.("[data-line-edit]")) setEditing(false); };
    const esc = (e: KeyboardEvent) => {
      if (e.key !== "Escape" || (e.target as Element | null)?.closest?.(".name-edit")) return;
      e.preventDefault();
      e.stopPropagation();
      setEditing(false);
    };
    window.addEventListener("pointerdown", away, true);
    window.addEventListener("keydown", esc, true);
    return () => {
      window.removeEventListener("pointerdown", away, true);
      window.removeEventListener("keydown", esc, true);
    };
  }, [editing]);
  // T-309: the demand level now in the line's own local time
  const now = demandAt(lineSeconds(w?.lineStats[id]));
  const lev = DEMAND_INDEX[now];
  // where it loses time to busy track, asked again after every edit and when the level turns
  const spots = useClockQuery(l ? { what: "lineDelays", line: l.num, level: lev } : null, [l?.num, w?.version, lev], spotsFrom) ?? [];
  const heldAnswer = useClockQuery(l ? { what: "lineHeld", line: l.num } : null, [l?.num, w?.version], (d) => ({ held: heldFrom(d), limits: limitsFrom(d) }));
  const held = heldAnswer?.held ?? [], limits = heldAnswer?.limits ?? NO_LIMITS;
  const marks = useClockQuery(l ? { what: "lineMarks", line: l.num } : null, [l?.num, w?.version], marksFrom) ?? [];
  // T-477, T-491: crowding by period on today's kind of day, asked again as the account changes
  // T-872, T-262: the line's country decides the kind of day
  const kind = kindIn(lineCountry(l?.stops ?? []), Math.floor(lineSeconds(w?.lineStats[id]) / 86400));
  const acct = accountVersion.value;
  const periods = useClockQuery(l ? { what: "linePeriods", line: l.num, day: kind } : null, [l?.num, w?.version, acct, kind], (d) => (d.length ? d : null));
  // T-646: while stops are edited, the stations the line passes between stops, and the gaps whose
  // passed stations are shown as stops to add, waiting for Add or Cancel
  // (the passed stations come with the world, so new stops never show with the old gaps, T-648;
  // the gaps picked hold for the world version they were picked in)
  const editingStops = tool.value === "line" && !!l && lineDraft.value.line === l.num;
  const [pendingAt, setPendingAt] = useState<{ v: number; gaps: number[] } | null>(null);
  useEffect(() => setPendingAt(null), [l?.num, editingStops]);
  // its day so far (T-293), kept fresh while the panel is open
  useToday();
  if (!w || !l) return null;
  const st = w.lineStats[l.id];
  const name = stationNames(w);
  const times = st.fromStartByLevel[lev] ?? st.fromStartS;
  const auto = headwayShown(l.id);
  const canAuto = l.stops.length >= 2 && st.status !== "broken";

  const build = async () => {
    const a = await edit({ op: "constructLine", line: l.num });
    if (a.ok) say(`${l.name} constructed for ${money(a.charge * 1e6)}. Its trains start with the next departure.`);
  };
  const passed = editingStops ? st.passed : [];
  const pending = pendingAt && pendingAt.v === w.version ? pendingAt.gaps : null;
  const setPending = (gaps: number[] | null) => setPendingAt(gaps && { v: w.version, gaps });
  const gaps = [...new Set(passed.map((p) => p.gap))];
  const adding = pending ? passed.filter((p) => pending.includes(p.gap)) : [];
  const setStops = async (stops: number[]) => {
    const a = await edit({ op: "setStops", line: l.num, stops });
    if (a.ok && lineDraft.peek().line === l.num) lineDraft.value = { line: l.num, stops };
  };
  const addPassed = () => {
    const stops = l.stops.flatMap((s, i) => [Number(s), ...adding.filter((p) => p.gap === i).map((p) => p.node)]);
    setPending(null);
    return setStops(stops);
  };
  const stationsText = (n: number) => plural(n, "station");
  const removeStop = (i: number) => setStops(l.stops.filter((_, k) => k !== i).map(Number));
  const remove = () => {
    ask.value = { text: `Delete ${l.name}? Track and stations stay.`, yes: "Delete line", no: "Keep line", run: () => {
      ask.value = null;
      void edit({ op: "removeLine", line: l.num }).then(a => {
        if (!a.ok) return;
        if (lineDraft.peek().line === l.num) live.tools?.exit();
        const selected = selection.peek();
        if (selected?.kind === "line" && selected.line === l.id) selection.value = null;
      });
    }};
  };
  const copy = async () => {
    if (copying) return;
    setCopying(true);
    try {
      const a = await edit({ op: "copyLine", line: l.num });
      if (!a.ok || a.line === undefined) return;
      live.tools?.exit();
      lineDraft.value = { line: a.line, stops: [] };
      renameLinePrompt.value = String(a.line);
      selection.value = { kind: "line", line: String(a.line) };
      tool.value = "line";
    } finally { setCopying(false); }
  };

  const shown = preview ?? l.colour;
  // T-670: the stops editor sits at the top while stops are being edited, else after the day's figures
  const stopsBlock = (
    <>
      <div class="h divided">
        Stops<span class="right muted small">boardings in past 24 h</span>
      </div>
      {editingStops && gaps.length > 1 && !pending && (
        <button class="btn text mini passed-all" onClick={() => setPending(gaps)}>
          <Icon.plus />Add all {stationsText(passed.length)} the line passes
        </button>
      )}
      {editingStops && pending && adding.length > 0 && (
        <div class="passed-confirm">
          <button class="btn text" onClick={() => void addPassed()}>Add {stationsText(adding.length)}</button>
          <button class="btn text" onClick={() => setPending(null)}>Cancel</button>
        </div>
      )}
      <ul class="stops" style={{ "--c": shown }}>
        {l.stops.map((s, i) => {
          const xfer = complexOf(s).some((m) => linesAt(m).some((o) => o.id !== l.id));
          const here = editingStops ? passed.filter((p) => p.gap === i) : [];
          const shownHere = pending?.includes(i);
          const after = here.length > 0 && i + 1 < l.stops.length && (shownHere
            ? here.map((p) => (
              <li class="adding"><span class="grow"><span class="stop-name">+ {name.get(String(p.node)) ?? ""}</span></span></li>
            ))
            : !pending && (
              <li class="passed">
                <button class="btn text mini" data-tip={`Add ${andList(here.map((p) => name.get(String(p.node)) ?? ""))}`} onClick={() => setPending([i])}>
                  <Icon.plus />{stationsText(here.length)} passed
                </button>
              </li>
            ));
          return (<>
            <li class={xfer ? "xfer" : ""}>
              <span class="grow">
                <span class="stop-name" onClick={() => (selection.value = { kind: "station", station: s })}>
                  {name.get(s)}
                </span>
                {i > 0 && times[i] !== undefined && <span class="t muted num" data-tip={stretchTip(st.stretches[i - 1])}>{duration(times[i])}</span>}
                {st.reversals?.includes(i) && (
                  <span class="rev-stop" data-tip={`Trains reverse here and stand ${duration(st.turnaroundS)}\nAt most 12 trains an hour reverse on each side`}>
                    <Icon.uturn />
                  </span>
                )}
              </span>
              <StopTransfers line={l.id} stop={s} />
              <StopToday line={l.id} i={i} />
              <RouteMarks marks={marks.filter((m) => m.gap === i)} />
              {editingStops && (
                <button class="btn icon mini danger" data-tip={st.loop && i > 0 && i === l.stops.length - 1 ? "Open the loop" : "Take this stop out of the line"} aria-label={`Remove stop ${name.get(s)}`} onClick={() => removeStop(i)}>
                  <Icon.cross />
                </button>
              )}
            </li>
            {after}
          </>);
        })}
      </ul>
    </>
  );

  return (
    <>
      <div class="title" data-line-edit>
        <button class={"badge-edit" + (editing ? " on" : "")} data-tip="Edit the name, symbol and colour (double click also opens it)" aria-label="Edit the name, symbol and colour"
          onClick={() => setEditing(true)} onDblClick={() => setEditing(true)}><Chip line={{ ...l, colour: shown }} size="lg" plain /></button>
        <NameEdit key={l.id} name={l.name} onRename={(n) => renameLine(l.id, n)} prompt={renameLinePrompt.value === id} onPrompted={() => { renameLinePrompt.value = null; }}
          control={{ open: editing, set: setEditing }}>
          <button class="btn icon" data-tip="Show on the map" aria-label="Show on the map" onClick={() => locateLine(l.stops)}>
            <Icon.locate />
          </button>
        </NameEdit>
      </div>
      {editing && (
        <div class="line-identity" data-line-edit>
          <label>Symbol <input aria-label="Line symbol" value={symbol} onInput={e => setSymbol(e.currentTarget.value)} onChange={e => void edit({op:"lineLook",line:l.num,name:l.name,colour:l.colour,letter:e.currentTarget.value.trim()})} /></label>
          <ColourPicker value={l.colour} presets={LINE_PALETTE} onPreview={dragColour} onCommit={commitColour} />
        </div>
      )}
      <div class="sub muted">
        {plural(l.stops.length - (st.loop ? 1 : 0), "stop")}, {distance(st.km * 1000)}
      </div>
      {st.status === "planned" && l.stops.length >= 2 && (
        <div class="callout">
          <p>Planned</p>
          <button class="btn text" data-tip={`Track and stations ${money(st.buildCost)}${st.trainCost > 0 ? `, trains ${money(st.trainCost)}` : ""}. Trains run once it is constructed.`} onClick={build}>
            Construct for {money(st.buildCost + st.trainCost)}
          </button>
        </div>
      )}
      {st.status === "broken" && (
        <ul class="alerts"><Alert>Track this line ran on is gone</Alert></ul>
      )}

      <div class="line-stop-actions">
        <button class={"btn text" + (editingStops ? " on" : "")} data-tip={editingStops ? "Finish editing stops (L)" : "Click stations on the map to add stops to the line (L)"} onClick={() => toggleEditStops(l)}><Icon.pencil />{editingStops ? "Done editing stops" : "Edit stops"}</button>
        <button class="btn text" onClick={() => void copy()} disabled={copying}>Copy line</button>
        <button class="btn text danger" onClick={remove}>Delete line</button>
      </div>
      {editingStops && stopsBlock}

      <div class="h">
        Schedule
        {canAuto && (
          <span class="right">
            <Segmented<boolean>
              options={[
                [false, <span data-tip="Set the number of trains on the line for each demand level">Trains</span>],
                [true, <span data-tip="Set the headway for each demand level. The line runs the fewest trains that keep it over the round trip, and works them out again when the line or its track changes.">Headway</span>],
              ]}
              value={auto !== null}
              onChange={(v) => setAutoTrains(l.id, v)}
            />
          </span>
        )}
      </div>
      <table class="sched">
        <tr>
          <th>Demand</th>
          <th>Trains</th>
          <th>Headway</th>
        </tr>
        {DEMANDS.map((d) => (
          <tr class={d === now ? "now" : ""}>
            <td>{st.status === "running" && (periods ? <CrowdDots v={periods} d={d} stops={l.stops.map((s) => name.get(s) ?? "")} /> : demandPending.value && <Spinner class="row-dot" tip="Crowding appears when demand has loaded" />)}<span data-tip={levelTimes(d) + "\nHow many trains run in this time interval. (Which time interval a scheduled train falls into is determined by when the train reaches the midpoint of its route.)"}>{DEMAND_LABEL[d]}</span></td>
            <LevelCells l={l} st={st} d={d} auto={auto} limit={limits[DEMAND_INDEX[d]]} />
          </tr>
        ))}
      </table>
      {st.status !== "broken" && <TrainType l={l} st={st} lev={lev} />}
      {st.status !== "broken" && <TrainLength l={l} st={st} />}
      {st.status === "running" && held.length > 0 && (
        <ul class="alerts">
          {[...new Set(held.map((h) => h.place))].map((place) => {
            const hs = held.filter((h) => h.place === place), h0 = hs[0];
            return (
              <Alert tip={h0.tip}>
                {h0.head}{h0.at ? <span class="place-link" onClick={() => locateStation(h0.at!.lng, h0.at!.lat)}>{h0.name}</span> : h0.name}{h0.tail}
                {hs.map((h) => <div class="detail" data-tip={h.tip + "\n" + heldTip}>{DEMAND_LABEL[h.d]} demand: {heldText(h.wait)}</div>)}
              </Alert>
            );
          })}
        </ul>
      )}

      {st.status === "running" && <PastDay l={l} st={st} periods={periods} kind={kind} />}

      {!editingStops && stopsBlock}
      {st.status !== "broken" && l.stops.length >= 2 && <RoundTrip st={st} lev={lev} stops={l.stops.length} spots={spots} />}
    </>
  );
}
