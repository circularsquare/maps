// The parts of a line's panel that rail, bus and ferry lines share (T-2373): the schedule table and
// its hovers, the average speed, and the past 24 hours. The vehicle words come from the line
// (`vehicleOf`); what only a mode has (a train's track limits, crowding dots and copy buttons, a bus's
// Changing to trains row) the panel passes in.

import type { ComponentChildren } from "preact";
import { lineMoneyTip, Money } from "../MoneyText";
import { count, duration, headway, money, speed, trainKind } from "../../game/format";
import { lowerWord, t, tList, tNumber, tSeries } from "../../i18n";
import { pctWhole } from "./tripText";
import { crowdColour } from "../../game/palette";
import { hourSpan, levelHours, PERIODS } from "../../game/periods";
import { lineReversed } from "../../game/northUp";
import { crowdLevel, dayNote, endNames, hasTrains, PERIOD_STRIDE, pendingOf, pendingRiders, profitKnown, runningToday, today, upkeepToday } from "../../game/today";
import { accountVersion, capacity } from "../../game/boarding";
import { onRails, vehicleCars, vehicleOf, type Vehicle } from "../../game/vehicle";
import { lineSeconds } from "../../game/clock";
import { DAY_KIND_NAME } from "../../game/days";
import { kindIn, lineCountry, rowIn } from "../../game/holidays";
import { ask } from "../../game/ui";
import { useClockQuery } from "../useClockQuery";
import { DEMAND_INDEX, DEMANDS, loopHalves, type Demand, type LineInput, type LineStats } from "../../game/types";
import { pendingLabel } from "../../game/lineFigures";
import { headwaysFor, lineKind, makeDefaultHeadway, newHeadway, applyDefaultHeadway } from "../../game/newHeadway";
import { liveMoney, moneyState, type MoneyState } from "../../game/money";
import { demandView, lineFiguresIn } from "../../game/demand";
import { AUTO_SLACK, autoTrainsFor, headwayFromMinutes, headwayShown, nextHeadway, refusedForCash, restHeadwayStep, restTphShown, setAutoTrains, setHeadway, setTrainCount, setWeekendCount, setWeekendShare, setWeekendShareAll, stepHeadway, stepTrainCount, stepWeekendCount, stepWeekendHeadway, trainCap, typeWeekendHeadway, trainsShown, weekendShown, world, MAX_TYPED_VEHICLES, WEEKEND_SHARE_STEP } from "../../game/world";
import { Row, Segmented, Stepper } from "../widgets";
import { demandPending, Spinner } from "../Spinner";

/** A demand level's name, capitalised: "High". */
export const demandLabel = (d: Demand): string => t("schedule.demandName", { demand: d });

/** When a demand level runs, in the line's local time (the hour spans of game/periods.ts), every
 * kind of day alike (T-2335): "High demand period: 06:00-20:00". */
export function levelTimes(d: Demand): string {
  return t("schedule.levelTimes", { demand: d, hours: tSeries(levelHours(d).map(([a, b]) => hourSpan(a, b))) });
}

// ---------------------------------------------------------------- the vehicle

export { vehicleCars, vehicleOf, type Vehicle };
/** The vehicle's plural noun: "trains". */
export const manyOf = (v: Vehicle): string => t("schedule.many", { vehicle: v });
/** As `manyOf`, capitalised: "Trains". */
const manyCap = (v: Vehicle): string => t("schedule.manyCap", { vehicle: v });

/** "1 bus", "3 buses" */
export const vehiclesText = (v: Vehicle, n: number) => t("schedule.vehicles", { vehicle: v, count: n });

export const carsText = (n: number) => t("inspector.cars", { count: n });

/** What buying `n` of a stock costs, or that the depot's spare ones cover it. */
const buys = (unit: (n: number) => string, n: number, price: number, spare: number): string => {
  const bought = Math.max(0, n - spare);
  return bought > 0 ? t("schedule.buys", { unit: unit(bought), price: money(bought * price * 1e6) }) : t("schedule.uses", { unit: unit(n) });
};
export const carsBought = (k: { carPrice: number; spare: number }, cars: number): string => buys(carsText, cars, k.carPrice, k.spare);

/** What a schedule change draws on: a train's cars of its kind (T-631), or buses or boats. */
function stockOf(m: MoneyState, st: LineStats, v: Vehicle): { unit: (n: number) => string; per: number; price: number; spare: number } | null {
  if (onRails(v)) {
    if (!st.cars) return null;
    const k = trainKind(m, st.fast, st.tram);
    return { unit: carsText, per: st.cars, price: k.carPrice, spare: k.spare };
  }
  const [owned, needed, price] = v === "boat" ? [m.fleetFerry, m.boatsNeeded, m.boatPrice] : [m.fleetBus, m.busesNeeded, m.busPrice];
  return { unit: (n) => vehiclesText(v, n), per: 1, price, spare: Math.max(0, owned - needed) };
}

/** What a stepper's button leads to: the vehicle count, and for a running line what it needs beyond
 * its busiest level, bought unless the depot has spare ones, or what it frees, which stays owned for
 * any line to use. `others`: the other kind of day's counts (T-3521), as the fleet is the most of
 * either. */
function changeTip(counts: readonly number[], lev: number, to: number, st: LineStats, others: readonly number[] = []): string {
  const v = vehicleOf(st);
  const head = vehiclesText(v, to);
  const m = moneyState.value;
  const k = m && st.status === "running" ? stockOf(m, st, v) : null;
  if (!k) return head;
  const next = [...counts];
  next[lev] = to;
  const more = (Math.max(...next, ...others) - Math.max(...counts, ...others)) * k.per;
  if (more > 0) return head + "\n" + buys(k.unit, more, k.price, k.spare);
  return more < 0 ? head + "\n" + t("schedule.frees", { unit: k.unit(-more) }) : head;
}

// ---------------------------------------------------------------- the schedule table

/** What limits a rail line per level (T-679): its shortest headway in seconds with the other lines
 * as they run, and the spot named as the capacity alerts name it ("Terminus at West"). */
/** `capS`: the shortest headway the train count is capped by, which on single track counts the
 * meeting margin for a lone train too (T-3526); `s` otherwise */
export interface Limit { s: number; what: string; capS: number }
export const NO_LIMITS: (Limit | null)[] = [null, null, null];

/** "Terminus at West takes one train every 4m00s at most", or with no room left "... is full" */
export const limitText = (lim: Limit, vehicle: Vehicle) => t("schedule.limit", { full: !Number.isFinite(lim.s), what: lim.what, time: Number.isFinite(lim.s) ? duration(lim.s) : "", vehicle });

const headwayText = (s: number) => (s > 0 ? t("inspector.min", { n: tNumber(s / 60, { maximumFractionDigits: 2 }) }) : t("schedule.noHeadway"));

/** How long a refused edit's control shakes, ms (T-625; style.css `.shake`) */
const SHAKE_MS = 300;

/** One demand level's vehicles and headway cells: a count stepper and the headway it gives, or (T-625)
 * the headway stepper and the count it gives. A count an edit was refused for cash is red, and
 * its control shakes once; a headway shorter than `limit` allows is red (T-625, T-679). */
function LevelCells({ l, st, d, auto, limits }: { l: LineInput; st: LineStats; d: Demand; auto: [number, number, number] | null; limits: (Limit | null)[] }) {
  const v = vehicleOf(st);
  const lev = DEMAND_INDEX[d];
  const trip = st.roundTripByLevel[lev];
  const limit = limits[lev];
  const lim = limit?.s ?? 0;
  const loop = loopHalves(st);
  // T-1276: a headway or count still on its way to the worker (an edit takes seconds on a big
  // network) shows the other cell as it will be, not the world's from an edit or two before
  const autoPending = auto ? auto.map((s, i) => s !== st.autoS?.[i]) : null;
  const counts = (auto
    ? st.trainsNeeded.map((n, i) => (autoPending![i] ? autoTrainsFor(st.roundTripByLevel[i], auto[i], limits[i]?.s ?? 0, loop) : n))
    : (trainsShown(l.id) ?? st.trainsNeeded)) as [number, number, number];
  const countPending = !auto && counts[lev] !== st.trainsNeeded[lev];
  // T-3521: a weekday change buys or frees only past the weekends' trains
  const others = weekendShown(l.id)?.counts ?? [];
  const r = refusedForCash.value;
  const refused = r && r.id === l.id && r.levs[lev] ? r : null;
  const broke = refused !== null && liveMoney.value.cash < refused.need;
  const shake = refused !== null && performance.now() - refused.at < SHAKE_MS;
  // the tutorial's marks are on a train line's cells
  const guide = !onRails(v) ? undefined : d === DEMANDS[0] ? "trains" : counts[lev] < 1 ? "levels" : undefined;
  const wrap = (child: ComponentChildren) => <span key={refused?.at ?? 0} class={shake ? "shake" : undefined} data-guide={guide}>{child}</span>;
  const n = <span class={broke ? "bad" : undefined}>{counts[lev]}</span>;
  // T-798: a loop's trains are half each way, so its count steps by two
  const k = loop ? 2 : 1;
  if (!auto) {
    const over = limit !== null && counts[lev] > 0 && counts[lev] - trip / limit.s > 0.01;
    // T-3333: a line on both pairs: its hidden twin's trains an hour too
    const twin = st.twin ? world.value?.save.lines.find((x) => x.id === st.twin) : undefined;
    const tph = countPending ? (trip > 0 ? (counts[lev] * 3600) / trip : 0) : l.tph[d] + (twin?.tph[d] ?? 0);
    // T-3165: one train more than the track lets run, so one is held and the held messages show
    const cap = trainCap(st, trip, limit?.capS ?? null);
    const full = counts[lev] >= cap;
    const grow = (to: number) => setTrainCount(l.id, d, Math.min(to, cap));
    return (
      <>
        <td>
          {wrap(<Stepper value={n} label={t("schedule.stepperLabel", { vehicle: v })} jump
            onStep={(s) => (s > 0 ? !full && grow(counts[lev] + s * k) : stepTrainCount(l.id, d, s))}
            type={{ text: String(counts[lev]), set: (x) => {
              const to = Math.max(0, Math.round(x)), now = counts[lev];
              if (to === now) return;
              if (to < now) { setTrainCount(l.id, d, to); return; }
              return !full && setTrainCount(l.id, d, Math.min(to, cap)) && to <= cap ? undefined : false;
            } }}
            tipDown={changeTip(counts, lev, Math.max(0, counts[lev] - k), st, others)}
            tipUp={full && limit !== null && cap < MAX_TYPED_VEHICLES ? limitText(limit, v) : changeTip(counts, lev, counts[lev] + k, st, others)} />)}
        </td>
        <td>{over ? <span class="bad" data-tip={t("schedule.limitWait", { text: limitText(limit, v), vehicle: v })}>{headway(tph)}</span> : headway(tph)}</td>
      </>
    );
  }
  const short = limit !== null && auto[lev] > 0 && auto[lev] * AUTO_SLACK < limit.s;
  return (
    <>
      <td>
        <span class={"num" + (broke ? " bad" : "")} data-tip={loop
          ? t("schedule.lapTip", { vehicles: vehiclesText(v, counts[lev]), half: counts[lev] / 2, lap: duration(trip / 2) })
          : st.loop ? t("schedule.lapOneWayTip", { vehicles: vehiclesText(v, counts[lev]), lap: duration(trip) })
          : t("schedule.roundTip", { vehicles: vehiclesText(v, counts[lev]), count: counts[lev], trip: duration(trip) })}>{counts[lev]}</span>
      </td>
      <td>
        {wrap(<Stepper wide jump jumpUnit={t("schedule.minUnit")} label={t("schedule.headwayLabel")} value={short ? <span class="bad">{headwayText(auto[lev])}</span> : headwayText(auto[lev])} onStep={(s) => stepHeadway(l.id, d, s)}
          tipDown={changeTip(counts, lev, autoTrainsFor(trip, nextHeadway(auto[lev], -1), lim, loop), st, others)}
          tipUp={changeTip(counts, lev, autoTrainsFor(trip, nextHeadway(auto[lev], 1), lim, loop), st, others)}
          type={{ text: auto[lev] ? String(+(auto[lev] / 60).toFixed(2)) : "0", set: (v) => setHeadway(l.id, d, headwayFromMinutes(v)), tip: short ? t("schedule.limitWait", { text: limitText(limit, v), vehicle: v }) : undefined }} />)}
      </td>
    </>
  );
}

// ---------------------------------------------------------------- weekends (T-3521)

/** One demand level's weekend cells in the schedule panel: as the weekday row's (`LevelCells`), a
 * count stepper and the headway it runs, or under headways the count and a headway stepper. A step
 * sets the line's weekend counts by hand (`setWeekendCount`), with the headway stepped to under headways. The headway is the rate settled for
 * weekends (`restTph`), or for a count still on its way the round trip over it. */
function WeekendCells({ l, st, d, auto, limits }: { l: LineInput; st: LineStats; d: Demand; auto: boolean; limits: (Limit | null)[] }) {
  const v = vehicleOf(st), lev = DEMAND_INDEX[d];
  const week = trainsShown(l.id) ?? st.trainsNeeded;
  const counts = weekendShown(l.id)?.counts ?? week;
  const n = counts[lev];
  const trip = st.roundTripByLevel[lev], limit = limits[lev];
  // the track holds as many on a weekend as on a weekday: the weekdays' cap (T-3165)
  const most = trainCap(st, trip, limit?.s ?? null);
  const tph = restTphShown(l.id, st, lev, n);
  const loop = loopHalves(st), k = loop ? 2 : 1;
  if (auto) {
    const lim = limit?.s ?? 0, h = tph > 0 ? 3600 / tph : 0;
    const short = limit !== null && h > 0 && h * AUTO_SLACK < limit.s;
    const stepTip = (dir: number) => {
      const s = restHeadwayStep(n, dir, trip, lim, loop, h);
      return s === null ? undefined : changeTip(counts, lev, Math.min(s.to, most), st, week);
    };
    return (
      <>
        <td>
          <span class="num" data-tip={loop
            ? t("schedule.lapTip", { vehicles: vehiclesText(v, n), half: n / 2, lap: duration(trip / 2) })
            : st.loop ? t("schedule.lapOneWayTip", { vehicles: vehiclesText(v, n), lap: duration(trip) })
            : t("schedule.roundTip", { vehicles: vehiclesText(v, n), count: n, trip: duration(trip) })}>{n}</span>
        </td>
        <td>
          <Stepper wide jump jumpUnit={t("schedule.minUnit")} label={t("schedule.weekendHeadwayLabel")}
            value={short ? <span class="bad">{headwayText(h)}</span> : headwayText(h)}
            onStep={(s) => stepWeekendHeadway(l.id, d, s, trip, lim, loop, most, h)}
            tipDown={stepTip(-1)} tipUp={stepTip(1)}
            type={{ text: h ? String(+(h / 60).toFixed(2)) : "0", set: (x) => { typeWeekendHeadway(l.id, d, x, trip, lim, loop, most); } }} />
        </td>
      </>
    );
  }
  const over = limit !== null && n > 0 && n - trip / limit.s > 0.01;
  const full = n >= most;
  return (
    <>
      <td>
        <Stepper value={n} label={t("schedule.weekendLabel", { vehicle: v })} jump
          onStep={(s) => (s > 0 ? !full && setWeekendCount(l.id, d, n + s * k, most) : stepWeekendCount(l.id, d, s, most))}
          type={{ text: String(n), set: (x) => {
            const to = Math.max(0, Math.round(x));
            if (to === n) return;
            if (to < n) { setWeekendCount(l.id, d, to, most); return; }
            return !full && setWeekendCount(l.id, d, to, most) && to <= most ? undefined : false;
          } }}
          tipDown={changeTip(counts, lev, Math.max(0, n - k), st, week)}
          tipUp={full && limit !== null && most < MAX_TYPED_VEHICLES ? limitText(limit, v) : changeTip(counts, lev, n + k, st, week)} />
      </td>
      <td>{over ? <span class="bad" data-tip={t("schedule.limitWait", { text: limitText(limit, v), vehicle: v })}>{headway(tph)}</span> : headway(tph)}</td>
    </>
  );
}

/** The Weekends section's heading row: its name, the line's percent of its weekdays (greyed while its
 * counts are set by hand, since it no longer says what runs), and the button that sets every line's
 * weekends to that percent after a question. */
function WeekendHead({ l }: { l: LineInput }) {
  const wk = weekendShown(l.id);
  const share = wk?.share ?? 100, custom = !!wk?.custom;
  const pct = pctWhole(share / 100);
  const all = () => {
    ask.value = {
      text: t("schedule.weekendAllAsk", { pct }),
      yes: t("schedule.weekendAll"),
      no: t("toolNotes.cancel"),
      run: () => setWeekendShareAll(share),
    };
  };
  return (
    <tr class="sched-day">
      <th colSpan={3}>
        <span class="sched-day-row">
          <span data-tip={t("schedule.weekendsTip")}>{t("schedule.weekends")}</span>
          <Stepper value={<span class={custom ? "muted" : undefined}>{pct}</span>} label={t("schedule.weekendShareLabel")}
            tipValue={custom ? t("schedule.weekendShareOff") : undefined}
            onStep={(s) => setWeekendShare(l.id, share + Math.sign(s) * WEEKEND_SHARE_STEP)}
            type={{ text: String(share), set: (x) => setWeekendShare(l.id, x) }} />
          <button type="button" class="btn text mini" onClick={all}>{t("schedule.weekendAll")}</button>
        </span>
      </th>
    </tr>
  );
}

/** T-477, T-515, T-2236, T-2335: a demand level's name in the schedule table, after its crowding dot
 * on today's kind of day (`crowdLevel`, from the boarding account) while `v` has one: the fill graded
 * over the line's stretches, the ring (1 px out with a 1 px gap) at its fullest. One hover for both
 * (T-2334): the level's hours, then the dot's. */
function LevelName({ v, d, stops, flip, pending, ends, vehicle }: { v: Float64Array | null; d: Demand; stops: number; flip: boolean; pending: boolean; ends?: [string, string]; vehicle: Vehicle }) {
  const c = v ? crowdLevel(v, d, stops, flip, ends, vehicle) : null;
  return (
    <span data-tip={c ? `${levelTimes(d)}\n${c.tip}` : levelTimes(d)}>
      {c ? <i class="crowd-dot" style={{ background: c.fill, outlineColor: crowdColour(c.peak) }} data-guide={d === "high" ? "crowd" : undefined} />
        : pending && <Spinner class="row-dot" tip={t("schedule.waitingDemand")} />}
      {demandLabel(d)}
    </span>
  );
}

/** The Schedule heading, its Trains (Buses, Boats) or Headway switch, and the table of levels. `crowd`:
 * the line's periods for the crowding dots (a rail line's); `copy`: the copy and paste buttons;
 * `label`: the heading's words in place of "Schedule" (the schedule panel's "Set by", T-3346);
 * `weekends`: a Weekdays and a Weekends section (the schedule panel's, T-3521). */
export function Schedule({ l, st, now, limits = NO_LIMITS, crowd = null, canAuto = true, copy, label, weekends = false }: {
  l: LineInput; st: LineStats; now: Demand; limits?: (Limit | null)[]; crowd?: Float64Array | null; canAuto?: boolean; copy?: ComponentChildren; label?: string; weekends?: boolean;
}) {
  const v = vehicleOf(st);
  const auto = headwayShown(l.id);
  const stops = l.stops.length;
  // T-2661: the crowding bar in a level's hover reads west to east
  const w = world.value, flip = !!w && lineReversed(l, w, !!st.loop);
  return (
    <>
      <div class={label ? "h set-by" : "h"}>
        {label ?? t("schedule.title")}
        {canAuto && copy}
        {canAuto && (
          <span class="right">
            <Segmented<boolean>
              options={[
                [false, manyCap(v)],
                [true, t("schedule.headway")],
              ]}
              value={auto !== null}
              onChange={(on) => setAutoTrains(l.id, on)}
            />
          </span>
        )}
      </div>
      <table class="sched">
        <tr>
          <th>{t("schedule.demand")}</th>
          <th>{manyCap(v)}</th>
          <th>{t("schedule.headway")}</th>
        </tr>
        {!weekends && DEMANDS.map((d) => {
          // T-1783: no dot on a level with no vehicles
          const dot = st.status === "running" && (trainsShown(l.id) ?? st.trainsNeeded)[DEMAND_INDEX[d]] > 0;
          const p = dot && crowd && lineFiguresIn(demandView.value, l.id, stops) ? crowd : null;
          return (
            <tr class={d === now ? "now" : ""}>
              <td><LevelName v={p} d={d} stops={stops} flip={flip} ends={endNames(l)} vehicle={vehicleOf(st)} pending={dot && !p && demandPending.value} /></td>
              <LevelCells l={l} st={st} d={d} auto={auto} limits={limits} />
            </tr>
          );
        })}
        {weekends && <DaySections l={l} st={st} now={now} limits={limits} crowd={crowd} auto={auto} flip={flip} />}
      </table>
      {weekends && <DefaultHeadways id={l.id} auto={auto} />}
    </>
  );
}

/** T-3543, T-3588: the headways every new line of this line's kind starts with, in the schedule panel: what they
 * are now, the button that makes this line's the default for the kind, and the way back. A bus or ferry line's
 * (T-3703) are its tool's until one is set. */
function DefaultHeadways({ id, auto }: { id: string; auto: [number, number, number] | null }) {
  const kind = lineKind(id);
  if (!kind) return null;
  const now = headwaysFor(kind, newHeadway.value);
  const same = !!auto && !!now && auto.every((s, i) => s === now[i]);
  return (
    <div class="sched-default">
      <div class="sched-default-head">{t(KIND_TITLE[kind])}</div>
      <div class="muted">{now ? tList(now.map(headwayText)) : t("schedule.defaultNone")}</div>
      <div class="sched-default-btns">
        <button type="button" class="btn text mini" disabled={same} onClick={() => makeDefaultHeadway(id)}>{t("schedule.defaultSet")}</button>
        <button type="button" class="btn text mini" disabled={!now || same} onClick={() => applyDefaultHeadway(id)}>{t("schedule.defaultUse")}</button>
      </div>
    </div>
  );
}
const KIND_TITLE = { rail: "schedule.defaultTitle", tram: "schedule.defaultTitleTram", fast: "schedule.defaultTitleFast", bus: "schedule.defaultTitleBus", ferry: "schedule.defaultTitleFerry" } as const;

/** T-3521: the schedule panel's table body: a Weekdays section and a Weekends section, each a row a
 * level. Today's kind of day bolds its level now, as the line panel does, and has `crowd`'s dots;
 * the other section's dots are a weekday's or a Saturday's (the boarding account's kinds 0 and 1). */
function DaySections({ l, st, now, limits, crowd, auto, flip }: { l: LineInput; st: LineStats; now: Demand; limits: (Limit | null)[]; crowd: Float64Array | null; auto: [number, number, number] | null; flip: boolean }) {
  const w = world.value, stops = l.stops.length;
  const today = rowIn(lineCountry(l.stops), Math.floor(lineSeconds(st) / 86400));
  const acct = accountVersion.value;
  const other = 1 - today;
  const asked = useClockQuery(st.status === "running" ? { what: "linePeriods", line: l.num, day: other } : null, [l.num, w?.version, acct, other], (x) => (x.length ? x : null)) ?? null;
  const crowdOf = (row: number) => (row === today ? crowd : asked);
  return (
    <>
      {[0, 1].map((row) => (
        <>
          {row === 0 ? <tr class="sched-day"><th colSpan={3}>{t("schedule.weekdays")}</th></tr> : <WeekendHead l={l} />}
          {DEMANDS.map((d) => {
            const lev = DEMAND_INDEX[d];
            const shown = row === 0 ? trainsShown(l.id) ?? st.trainsNeeded : weekendShown(l.id)?.counts ?? st.trainsNeeded;
            // T-1783: no dot on a level with no vehicles that day
            const dot = st.status === "running" && shown[lev] > 0;
            const c = crowdOf(row);
            const p = dot && c && lineFiguresIn(demandView.value, l.id, stops) ? c : null;
            return (
              <tr class={d === now && today === row ? "now" : ""}>
                <td><LevelName v={p} d={d} stops={stops} flip={flip} ends={endNames(l)} vehicle={vehicleOf(st)} pending={dot && !p && demandPending.value} /></td>
                {row === 0 ? <LevelCells l={l} st={st} d={d} auto={auto} limits={limits} /> : <WeekendCells l={l} st={st} d={d} auto={auto !== null} limits={limits} />}
              </tr>
            );
          })}
        </>
      ))}
    </>
  );
}

/** The line's average speed end to end: the distance both ways over the round trip minus the turning
 * at the ends, so it counts the stops and the waiting at busy track but not the layover. */
export function AverageSpeed({ st, lev }: { st: LineStats; lev: number }) {
  const rt = st.roundTripByLevel[lev] ?? st.roundTripS;
  // T-798: a loop once round in its lap, minus a balloon's turning
  const s = st.loop ? rt / (loopHalves(st) ? 2 : 1) - (st.loop === "balloon" ? st.turnaroundS : 0) : rt - 2 * st.turnaroundS;
  if (!(s > 0) || !(st.km > 0)) return null;
  const km = st.loop ? st.km : 2 * st.km;
  const kmh = km / (s / 3600);
  return (
    <li>
      <span class="grow">{t("schedule.averageSpeed")}</span>
      <span class="val">{speed(kmh)}</span>
    </li>
  );
}

// ---------------------------------------------------------------- the past 24 hours

/** The line's kind of day now (T-872, T-262: its country decides) and the boarding account's numbers
 * by period on it (`TrackApi.line_periods`, T-477, T-491), asked again as the account changes. */
export function useLinePeriods(id: string, l: LineInput | undefined): { kind: number; periods: Float64Array | null } {
  const w = world.value;
  const kind = kindIn(lineCountry(l?.stops ?? []), Math.floor(lineSeconds(w?.lineStats[id]) / 86400));
  const acct = accountVersion.value;
  const periods = useClockQuery(l ? { what: "linePeriods", line: l.num, day: kind } : null, [l?.num, w?.version, acct, kind], (d) => (d.length ? d : null)) ?? null;
  return { kind, periods };
}

/** T-820, T-2373: the day's rider-seconds aboard over what the trips hold at the vehicle's rated load
 * (T-2186: a train's 160 a car, so past 1 up to crush's 1.5; a bus's 80, which is its crush; a boat's
 * 320, up to 480), each trip taken as its level's time from the first stop to the last; null with no
 * trips. */
function averageLoad(v: Float64Array, st: LineStats): number | null {
  const { rated, crush } = capacity(vehicleCars(st));
  let aboard = 0, room = 0;
  for (let i = 0; i < PERIODS.length; i++) {
    const o = i * PERIOD_STRIDE;
    if (!(v[o] > 0)) continue;
    const times = st.fromStartByLevel[DEMAND_INDEX[PERIODS[i].demand]] ?? st.fromStartS;
    aboard += v[o + 5];
    room += v[o] * (times[times.length - 1] ?? 0) * rated;
  }
  return room > 0 ? Math.min(crush / rated, aboard / room) : null;
}

/** The line's past 24 hours (T-446, game/today.ts): riders, `extra` rows (hidden while the riders are
 * worked out again), the expected profit for today (a whole day, T-2475), the average load, and its vehicles from the boarding account (T-230): how
 * many left full, and for a train line who was left on the platform. Its own component: it re-renders
 * as the counts move. */
export function PastDay({ l, st, periods, kind, extra }: { l: LineInput; st: LineStats; periods: Float64Array | null; kind: number; extra?: ComponentChildren }) {
  const d = today.value.get(l.id);
  const pending = pendingOf(l);
  // T-2678: while pending, the last figure there is (its count, its last one, or demand's), the label beside it
  const shown = pending ? pendingRiders(l) : null;
  if (!d) {
    if (!pending) return null;
    return (
      <>
        <div class="h divided">{shown !== null && <span class="pending-note muted">{pendingLabel(pending)}</span>}</div>
        <ul class="rows"><Row label={t("schedule.riders")}>{shown !== null ? count(shown) : <span class="muted">{pendingLabel(pending)}</span>}</Row></ul>
      </>
    );
  }
  const v = vehicleOf(st);
  const load = periods ? averageLoad(periods, st) : null;
  const run = runningToday(l, st);
  // T-2392: while the riders are worked out again the old figures stay, the label above them
  return (
    <>
      <div class="h divided">{pending && <span class="pending-note muted">{pendingLabel(pending)}</span>}</div>
      <ul class="rows">
        <Row label={t("schedule.riders")}>{count(shown ?? d.riders)}</Row>
        {!pending && extra}
        {profitKnown(l) ? (
          <li data-tip={lineMoneyTip(d.dayFares, run, upkeepToday(l, st), v === "boat" ? "boats" : v === "tram" ? "trams" : v === "bus", dayNote(l, st))}>
            <span class="grow">{t("schedule.profit")}</span>
            <span class="val"><Money dollars={d.dayFares - run - upkeepToday(l, st)} /></span>
          </li>
        ) : hasTrains(l) && <Row label={t("schedule.profit")}><span class="muted">{t("schedule.calculating")}</span></Row>}
        {load !== null && (
          <li data-tip={t("schedule.loadTip", { day: lowerWord(DAY_KIND_NAME[kind]).replace(/ day$/, "") })}>
            <span class="grow">{t("schedule.averageLoad")}</span>
            <span class="val">{pctWhole(load)}</span>
          </li>
        )}
        {d.fullTrains > 0 && <Row label={t("schedule.leavingFull", { vehicle: v })}><span data-tip={t("schedule.departedFull", { full: count(d.fullTrains), all: count(d.trains), vehicle: v })}>{count(d.fullTrains)}/{count(d.trains)}</span></Row>}
        {v !== "bus" && (
          <li data-tip={t("schedule.leftBehindTip", { left: count(d.leftBehind), all: count(d.riders + d.leftBehind) })}>
            <span class="grow">{t("schedule.leftBehind", { vehicle: v })}</span>
            <span class="val">{count(d.leftBehind)}/{count(d.riders + d.leftBehind)}</span>
          </li>
        )}
      </ul>
    </>
  );
}
