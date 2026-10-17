// The line inspector: name and colour, whether it runs (and what constructing the rest costs), the
// schedule by demand level with headway and trains needed (SPEC 6.3), its past 24 hours (riders,
// fares, running cost and its trains, T-293, T-446), the stops with each stop's time from the start
// of the line and boardings in the past 24 hours, and the round
// trip split into running, dwell, turning and waiting, with where it waits (T-031).

import { useEffect, useRef, useState } from "preact/hooks";
import { nearestStation } from "../../game/places";
import { queryClock } from "../../workers/clockClient";
import { clock, demandAt } from "../../game/clock";
import { clockTime, count, distance, duration, headway, money, speed } from "../../game/format";
import { LINE_PALETTE } from "../../game/palette";
import { levelHours, PERIODS, periodAt } from "../../game/periods";
import { signal } from "@preact/signals";
import { crowdGrade, crowding, today, useToday, useYesterday, waitText, yesterday } from "../../game/today";
import { accountVersion } from "../../game/boarding";
import { complexOf } from "../../game/complexView";
import { dayKindAt } from "../../game/days";
import { DEMAND_INDEX, DEMANDS, type Demand, type LineInput, type LineStats } from "../../game/types";
import { ask, lineDraft, renameLinePrompt, selection, tool } from "../../game/ui";
import { moneyState } from "../../game/money";
import { fullTip, fullTitle, heldText, heldTip } from "../../game/capacityKinds";
import { autoTrainsFor, edit, headwayFromMinutes, headwayShown, linesAt, nextHeadway, renameLine, say, setAutoTrains, setHeadway, setLineColour, stepHeadway, stepTrainCount, trainsShown, world } from "../../game/world";
import { live } from "../../map/live";
import { Chip, Icon, Row, Segmented, Stepper } from "../widgets";
import { PurposeSplit } from "./PurposeSplit";
import { NameEdit } from "./NameEdit";
import { locateLine } from "../locate";
import { demandPending, Spinner } from "../Spinner";

const DEMAND_LABEL: Record<Demand, string> = { high: "High", medium: "Medium", low: "Low" };

/** Which day the Today block shows (T-408), kept while the game is open. */
const dayShown = signal<"today" | "yesterday">("today");

/** When a demand level runs, in the line's local time (the hour spans of game/periods.ts): "High
 * demand: 06:00 to 10:00, 16:00 to 20:00". */
function levelTimes(d: Demand): string {
  const t = (h: number) => clockTime((h * 60) % 1440);
  return `${DEMAND_LABEL[d]} demand: ${levelHours(d).map(([a, b]) => `${t(a)} to ${t(b)}`).join(", ")}`;
}

/** Where a line waits, in words (T-031): from `TrackApi.line_delays`. */
function spotsFrom(d: Float64Array): { text: string; s: number }[] {
  const out: { text: string; s: number }[] = [];
  const st = new Map((world.value?.save.stations ?? []).map((s) => [s.num, s.name]));
  for (let i = 0; i + 4 < d.length; i += 5) {
    const [kind, node, x, y, s] = [d[i], d[i + 1], d[i + 2], d[i + 3], d[i + 4]];
    const near = nearestStation(x, y)?.name;
    const at = node >= 0 ? st.get(node) : undefined;
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
interface Held { d: Demand; wait: number; place: string; tip: string }
function heldFrom(v: Float64Array): Held[] {
  const st = new Map((world.value?.save.stations ?? []).map((s) => [s.num, s.name]));
  const out: Held[] = [];
  DEMANDS.forEach((d, lev) => {
    const [wait, kind, node, x, y, , most] = v.subarray(7 * lev, 7 * lev + 7);
    if (!(wait > 0)) return;
    const name = (node >= 0 ? st.get(node) : undefined) ?? nearestStation(x, y)?.name ?? "";
    out.push({ d, wait, place: fullTitle(kind, name), tip: fullTip(kind, most) });
  });
  return out;
}

/** A flat junction or flat crossing on the route (T-629): `TrackApi.line_marks`, after the stop
 * `gap` and drawn on the stops diagram between it and the next. */
interface RouteMark { crossing: boolean; gap: number; near: string; others: string[] }
function marksFrom(d: Float64Array): RouteMark[] {
  const lines = new Map((world.value?.save.lines ?? []).map((l) => [l.num, l.name]));
  const out: RouteMark[] = [];
  for (let i = 0; i + 5 < d.length; ) {
    const n = d[i + 5];
    const others = Array.from(d.subarray(i + 6, i + 6 + n), (o) => lines.get(o) ?? `Line ${o}`);
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
  const total = st.roundTripByLevel[lev] ?? st.roundTripS;
  const dwell = 2 * Math.max(0, stops - 2) * st.dwellS;
  const turn = 2 * st.turnaroundS;
  const delay = st.delayS[lev] ?? 0;
  const running = Math.max(0, total - dwell - turn - delay);
  return (
    <>
      <ul class="rows">
        <Row label="Round trip" bold>
          {duration(total)}
        </Row>
        <Row label="Running">{duration(running)}</Row>
        <Row label="Dwell at stops">{duration(dwell)}</Row>
        <Row label="Turning at the ends">{duration(turn)}</Row>
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
  const s = (st.roundTripByLevel[lev] ?? st.roundTripS) - 2 * st.turnaroundS;
  if (!(s > 0) || !(st.km > 0)) return null;
  const kmh = (2 * st.km) / (s / 3600);
  return (
    <ul class="rows">
      <li data-tip={`${distance(2 * st.km * 1000)} there and back in ${duration(s)}, the round trip less the turning at the ends. Dwell at stops and waiting at busy track are in it, so it is the average speed of both directions together.`}>
        <span class="grow">Average speed</span>
        <span class="val">{speed(kmh)}</span>
      </li>
    </ul>
  );
}

/** Times of day for the full-train spans: "07:40 to 08:50, 17:35 to 18:45". */
function spansText(spans: [number, number][]): string {
  const t = (s: number) => clockTime(((Math.round(s / 60) % 1440) + 1440) % 1440);
  return spans.map(([a, b]) => (b - a < 60 ? t(a) : `${t(a)} to ${t(b)}`)).join(", ");
}

/** US$ of running cost over the past 24 hours (T-446): each hour's trains at its demand level, the
 * hours at either end in part, as the worker drains the cash (T-284). */
function runningToday(l: LineInput, st: LineStats): number {
  const m = moneyState.value;
  if (!m) return 0;
  // T-309: in the line's own local time
  const t = (clock.now() + st.shift) / 3600, t0 = t - 24;
  const tph = { high: l.tph.high, medium: l.tph.medium, low: l.tph.low };
  let cost = 0;
  for (let h = Math.floor(t0); h < t; h++) {
    const lev = PERIODS[periodAt(h * 3600)].demand;
    cost += tph[lev] * 2 * st.km * st.cars * m.runCostCarKm * m.economy * (Math.min(h + 1, t) - Math.max(h, t0));
  }
  return cost;
}

/** The line's past 24 hours (T-446, game/today.ts) or yesterday: riders, fares and running cost,
 * and its trains from the boarding account (T-230): how many left packed full and when, the
 * fullest, and who was left on the platform. Its own component: it re-renders as the counts move. */
function LineToday({ l, st }: { l: LineInput; st: LineStats }) {
  const yest = dayShown.value === "yesterday";
  useYesterday(yest);
  const d = (yest ? yesterday : today).value.get(l.id);
  if (!d && !yest) return null;
  const fullest = d ? Math.round(100 * d.fullest) : 0;
  return (
    <>
      <div class="h">
        <Segmented<"today" | "yesterday">
          options={[
            ["today", <span data-tip="The past 24 hours, counted train by train. Whoever does not fit on a full train waits for the next one.">Past 24 h</span>],
            ["yesterday", <span data-tip="The last complete day, midnight to midnight in the line's local time">Yesterday</span>],
          ]}
          value={dayShown.value}
          onChange={(v) => { dayShown.value = v; }}
        />
      </div>
      {d && <ul class="rows">
        <Row label="Riders">{count(d.riders)}</Row>
        <PurposeSplit scope={[{ line: l.id }]} unit="riders" />
        <Row label="Fares">{money(d.fares)}</Row>
        <Row label="Running trains">{money(-(yest ? d.running : runningToday(l, st)))}</Row>
        <Row label="Fullest train">{d.fullTrains > 0 ? "100%" : `${fullest}% full`}</Row>
        {d.fullTrains > 0 && <Row label="Trains leaving full"><span data-tip={`${count(d.fullTrains)}/${count(d.trains)} trains left packed full`}>{count(d.fullTrains)}/{count(d.trains)}</span></Row>}
        {d.spans.length > 0 && <Row label="When">{spansText(d.spans)}</Row>}
        <Row label="Riders left on the platform">{count(d.leftBehind)}</Row>
      </ul>}
    </>
  );
}

/** Numbers per period in `TrackApi.line_periods` (`PERIOD_STRIDE` in boarding.rs). */
const PERIOD_STRIDE = 6;

/** T-477, T-515: one dot per demand level, graded by `crowding` (the riders' wait for a train with
 * room against their ride) from the boarding account: the level's periods averaged by riders. */
function CrowdDots({ v, d }: { v: Float64Array; d: Demand }) {
  let trains = 0, full = 0, riders = 0, wait = 0, score = 0;
  let worst: { name: string; wait: number; score: number } | null = null;
  for (let i = 0; i < PERIODS.length; i++) {
    if (PERIODS[i].demand !== d) continue;
    const o = i * PERIOD_STRIDE;
    if (!(v[o] > 0)) continue;
    const c = crowding(v[o + 4], v[o + 2], v[o + 5]);
    trains += v[o]; full += v[o + 1]; riders += v[o + 2];
    wait += c.wait * v[o + 2]; score += c.score * v[o + 2];
    if (!worst || c.score > worst.score) worst = { name: PERIODS[i].name, wait: c.wait, score: c.score };
  }
  if (!(trains > 0)) return null;
  const avg = riders > 0 ? { wait: wait / riders, score: score / riders } : { wait: 0, score: 0 };
  const grade = crowdGrade(avg.score);
  const tip = `Riders wait ${waitText(avg.wait)} on average for a train with room\n${count(full)}/${count(trains)} trains are packed full`
    + (worst && crowdGrade(worst.score) > grade ? `\n${worst.name}: ${waitText(worst.wait)}` : "");
  return <i class={"crowd-dot c" + grade} data-tip={tip} />;
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

const carsText = (n: number) => `${n} ${n === 1 ? "car" : "cars"}`;

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
    const bought = Math.max(0, cars - Math.max(0, m.fleet - m.carsNeeded));
    return head + (bought > 0 ? `\nBuys ${carsText(bought)} for ${money(bought * m.carPrice * 1e6)}` : `\nUses ${carsText(cars)} from the depot`);
  }
  return cars < 0 ? head + `\nFrees ${carsText(-cars)}, which stay in the depot` : head;
}

const headwayText = (s: number) => (s > 0 ? `${+(s / 60).toFixed(2)} min` : "none");

/** One demand level's trains and headway cells: a count stepper and the headway it gives, or (T-625)
 * the headway stepper and the count it gives. */
function LevelCells({ l, st, d, auto }: { l: LineInput; st: LineStats; d: Demand; auto: [number, number, number] | null }) {
  const lev = DEMAND_INDEX[d];
  const counts = auto ? st.trainsNeeded : (trainsShown(l.id) ?? st.trainsNeeded);
  if (!auto) {
    return (
      <>
        <td>
          <Stepper value={counts[lev]} label="trains on the entire line" jump onStep={(s) => stepTrainCount(l.id, d, s)}
            tipDown={changeTip(counts, lev, Math.max(0, counts[lev] - 1), st)} tipUp={changeTip(counts, lev, counts[lev] + 1, st)} />
        </td>
        <td>{headway(l.tph[d])}</td>
      </>
    );
  }
  const trip = st.roundTripByLevel[lev];
  return (
    <>
      <td>
        <span class="num" data-tip={`${counts[lev]} ${counts[lev] === 1 ? "train" : "trains"} on the entire line keep the headway over its round trip of ${duration(trip)}`}>{counts[lev]}</span>
      </td>
      <td>
        <Stepper wide label="headway" value={headwayText(auto[lev])} onStep={(s) => stepHeadway(l.id, d, s)}
          tipDown={changeTip(counts, lev, autoTrainsFor(trip, nextHeadway(auto[lev], -1)), st)}
          tipUp={changeTip(counts, lev, autoTrainsFor(trip, nextHeadway(auto[lev], 1)), st)}
          type={{ text: auto[lev] ? String(+(auto[lev] / 60).toFixed(2)) : "0", set: (v) => setHeadway(l.id, d, headwayFromMinutes(v)), tip: "Click to type a headway in minutes. 0 is no service." }} />
      </td>
    </>
  );
}

export function LineInspector({ id }: { id: string }) {
  const w = world.value;
  const [colours, setColours] = useState(false);
  const [copying, setCopying] = useState(false);
  const [spots, setSpots] = useState<{ text: string; s: number }[]>([]);
  const [held, setHeld] = useState<Held[]>([]);
  const [marks, setMarks] = useState<RouteMark[]>([]);
  const l =w?.save.lines.find((l) => l.id === id);
  const [symbol, setSymbol] = useState("");
  useEffect(() => { setSymbol(l?.letter ?? ""); }, [l?.letter]);
  // T-309: the demand level now in the line's own local time
  const now = demandAt(clock.hour.value * 3600 + (w?.lineStats[id]?.shift ?? 0));
  const lev = DEMAND_INDEX[now];
  // where it loses time to busy track, asked again after every edit and when the level turns
  useEffect(() => {
    if (!l) return;
    let on = true;
    void queryClock({ what: "lineDelays", line: l.num, level: lev }).then((d) => on && d && setSpots(spotsFrom(d)));
    return () => {
      on = false;
    };
  }, [l?.num, w?.version, lev]);
  useEffect(() => {
    if (!l) return;
    let on = true;
    void queryClock({ what: "lineHeld", line: l.num }).then((d) => on && d && setHeld(heldFrom(d)));
    void queryClock({ what: "lineMarks", line: l.num }).then((d) => on && d && setMarks(marksFrom(d)));
    return () => { on = false; };
  }, [l?.num, w?.version]);
  // T-477, T-491: crowding by period on today's kind of day, asked again as the account changes
  const kind = dayKindAt(clock.hour.value * 3600 + (w?.lineStats[id]?.shift ?? 0));
  const [periods, setPeriods] = useState<Float64Array | null>(null);
  const acct = accountVersion.value;
  useEffect(() => {
    if (!l) return;
    let on = true;
    void queryClock({ what: "linePeriods", line: l.num, day: kind }).then((d) => on && setPeriods(d && d.length ? d : null));
    return () => { on = false; };
  }, [l?.num, w?.version, acct, kind]);
  // its day so far (T-293), kept fresh while the panel is open
  useToday();
  if (!w || !l) return null;
  const st = w.lineStats[l.id];
  const name = new Map(w.save.stations.map((s) => [s.id, s.name]));
  const times = st.fromStartByLevel[lev] ?? st.fromStartS;
  const editingStops = tool.value === "line" && lineDraft.value.line === l.num;
  const auto = headwayShown(l.id);
  const canAuto = l.stops.length >= 2 && st.status !== "broken";

  const build = async () => {
    const a = await edit({ op: "constructLine", line: l.num });
    if (a.ok) say(`${l.name} constructed for ${money(a.charge * 1e6)}. Its trains start with the next departure.`);
  };
  const addStops = () => {
    if (editingStops) { live.tools?.exit(); return; }
    live.tools?.exit();
    lineDraft.value = { line: l.num, stops: l.stops.map(Number) };
    tool.value = "line";
    selection.value = { kind: "line", line: l.id };
  };
  const removeStop = async (i: number) => {
    const stops = l.stops.filter((_, k) => k !== i).map(Number);
    const a = await edit({ op: "setStops", line: l.num, stops });
    if (a.ok && lineDraft.peek().line === l.num) lineDraft.value = { line: l.num, stops };
  };
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
  /** A copy to edit; an express copy (T-255) runs on the express pair of quad track where there
   * is one and goes straight to stop editing, so the player clicks off the stops it skips. */
  const copy = async (express = false) => {
    if (copying) return;
    setCopying(true);
    try {
      const a = await edit({ op: "copyLine", line: l.num, express });
      if (!a.ok || a.line === undefined) return;
      live.tools?.exit();
      lineDraft.value = { line: a.line, stops: [] };
      if (!express) renameLinePrompt.value = String(a.line);
      selection.value = { kind: "line", line: String(a.line) };
      tool.value = "line";
    } finally { setCopying(false); }
  };

  return (
    <>
      <div class="title">
        <button class={"badge-edit" + (colours ? " on" : "")} data-tip="Edit line colour and symbol" aria-label="Edit line colour and symbol" onClick={() => setColours(!colours)}><Chip line={l} size="lg" plain /></button>
        <NameEdit key={l.id} name={l.name} onRename={(n) => renameLine(l.id, n)} prompt={renameLinePrompt.value === id} onPrompted={() => { renameLinePrompt.value = null; }}>
          <button class="btn icon" data-tip="Show on the map" aria-label="Show on the map" onClick={() => locateLine(l.stops)}>
            <Icon.locate />
          </button>
        </NameEdit>
      </div>
      {colours && (
        <div class="line-identity">
          <label>Symbol <input aria-label="Line symbol" value={symbol} onInput={e => setSymbol(e.currentTarget.value)} onChange={e => void edit({op:"lineLook",line:l.num,name:l.name,colour:l.colour,letter:e.currentTarget.value.trim()})} /></label>
          <div class="palette">
          {LINE_PALETTE.map((c) => (
            <button class={"swatch" + (c === l.colour ? " on" : "")} style={{ "--c": c }} data-tip={c} aria-label={c} onClick={() => setLineColour(l.id, c)} />
          ))}
          <label class="swatch custom" data-tip="Any colour" style={{ "--c": l.colour }}>
            <input type="color" value={l.colour} onChange={(e) => setLineColour(l.id, (e.target as HTMLInputElement).value)} />
          </label>
          </div>
        </div>
      )}
      <div class="sub muted">
        {l.stops.length} stops, {distance(st.km * 1000)}{st.cars ? `, ${st.cars} car trains` : ""}
      </div>
      {l.stops.length < 2 && <p class="muted small">No route: fewer than two stops.</p>}
      {st.status === "planned" && l.stops.length >= 2 && (
        <div class="callout">
          <p>Planned</p>
          <button class="btn text" data-tip={`Track and stations ${money(st.buildCost)}${st.trainCost > 0 ? `, trains ${money(st.trainCost)}` : ""}. Trains run once it is constructed.`} onClick={build}>
            Construct for {money(st.buildCost + st.trainCost)}
          </button>
        </div>
      )}
      {st.status === "broken" && (
        <div class="callout bad">
          <p>Broken: track it used is gone.</p>
        </div>
      )}

      <div class="h">
        Schedule
        {canAuto && (
          <span class="right">
            <Segmented<boolean>
              options={[
                [false, <span data-tip="Set the number of trains on the entire line for each demand level">Trains</span>],
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
          <th>Trains on line</th>
          <th>Headway</th>
        </tr>
        {DEMANDS.map((d) => (
          <tr class={d === now ? "now" : ""}>
            <td>{st.status === "running" && (periods ? <CrowdDots v={periods} d={d} /> : demandPending.value && <Spinner class="row-dot" tip="Crowding appears when demand has loaded" />)}<span data-tip={levelTimes(d)}>{DEMAND_LABEL[d]}</span></td>
            <LevelCells l={l} st={st} d={d} auto={auto} />
          </tr>
        ))}
      </table>
      {st.status === "running" && held.length > 0 && (
        <div class="callout bad">
          <p data-tip={held[0].tip}>{held[0].place}</p>
          <ul class="rows">
            {held.map((h) => (
              <Row label={`${DEMAND_LABEL[h.d]} demand`}>
                <span data-tip={(h.place === held[0].place ? "" : h.place + "\n") + h.tip + "\n" + heldTip}>{heldText(h.wait)}</span>
              </Row>
            ))}
          </ul>
        </div>
      )}

      {st.status === "running" && <LineToday l={l} st={st} />}

      <div class="h">
        Stops<span class="right muted small">boardings, past 24 h</span>
      </div>
      <div class="line-stop-actions">
        <button class={"btn text" + (editingStops ? " on" : "")} onClick={addStops}><Icon.pencil />{editingStops ? "Done editing stops" : "Edit stops"}</button>
        <button class="btn text" onClick={() => void copy()} disabled={copying}>Copy line</button>
        <button class="btn text" onClick={() => void copy(true)} disabled={copying} data-tip="Copy onto the express tracks where there are any, then click the stops it skips">Express copy</button>
        <button class="btn text danger" onClick={remove}>Delete line</button>
      </div>
      <ul class="stops" style={{ "--c": l.colour }}>
        {l.stops.map((s, i) => {
          const xfer = complexOf(s).some((m) => linesAt(m).some((o) => o.id !== l.id));
          return (
            <li class={xfer ? "xfer" : ""}>
              <span class="grow">
                <span class="stop-name" onClick={() => (selection.value = { kind: "station", station: s })}>
                  {name.get(s)}
                </span>
                {i > 0 && times[i] !== undefined && <span class="t muted num">{duration(times[i])}</span>}
              </span>
              <StopTransfers line={l.id} stop={s} />
              <StopToday line={l.id} i={i} />
              <RouteMarks marks={marks.filter((m) => m.gap === i)} />
              {editingStops && (
                <button class="btn icon mini danger" data-tip="Take this stop out of the line" aria-label={`Remove stop ${name.get(s)}`} onClick={() => removeStop(i)}>
                  <Icon.cross />
                </button>
              )}
            </li>
          );
        })}
      </ul>
      {st.status !== "broken" && l.stops.length >= 2 && <RoundTrip st={st} lev={lev} stops={l.stops.length} spots={spots} />}
      {st.status !== "broken" && l.stops.length >= 2 && <AverageSpeed st={st} lev={lev} />}
    </>
  );
}
