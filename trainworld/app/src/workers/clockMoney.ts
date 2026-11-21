// Money and the ledger: the day and hour entries, regions, settling, and what the Money panel reads (T-2374).

import { TrackApi } from "../wasm/track/sim"; // the track model only (T-051)
import { PERIODS } from "../game/periods";
import { loanDrain } from "../game/loan";
import { foldLedger } from "./ledgerFold";
import { kindOfIn, rowOfDay, rowOfIn, type DayRow } from "../game/holidayCalendar";
import { type KindOf } from "../game/moneyTiming";
import { newDay, newHour, addEntry, type DayLedger, type HourLedger, type SaveHeader } from "./saveFile";
import type { MoneyState, RegionNow } from "./protocol";
import { saves } from "./slots";
import { serviceEditMs } from "./clockFollowUp";
import { game } from "./clockMain";
import { kept } from "./clockPanic";
import { moneyPost } from "./clockPost";
import { achievements, api, clk, copyOf, costHour, dayShift, fareHour, fares, goals, guide, heldCharges, holidayData, hourLedger, hourShape, incomePerHour, kindOfLine, ledger, LEDGER_DAYS, LEDGER_HOURS, ledgerDay, lineCountries, lineIncome, lineNotes, linePoints, lineRides, lineShares, loan, modded, mode, moneyFrozen, NP, partCost, partFare, partHour, prices, putHourLedger, putLedger, putLoan, putPartCost, putPartFare, putPartHour, putSettledT, rideHour, ridesPerHour, rowOfLine, runTail, send, settledT, stars, syncHolidays, TEN, upkeepRegions, wallNow } from "./clockState";

/** T-531: each region's ledgers, kept as the network's (a zone id; "" outside every zone or not yet
 * placed, so the regions add up to the network) */
export let regions = new Map<string, { days: DayLedger[]; hours: HourLedger[] }>();

/** the cash and today's fares last sent in a "cash" message */
let sentCash = NaN, sentFares = NaN;

// ---------------------------------------------------------------- money (T-028)

export function gameNow(): number {
  return clk.t + Math.max(0, (wallNow() - clk.at) / 1000) * clk.rate;
}

/** T-1071: wall seconds after the clock a train count or headway edit takes effect: room for this
 * worker's own time on it, twice over, and the main thread's frame, so its new trips are on screen
 * before anything they change is drawn. */
export function serviceLeadS(): number {
  return Math.min(2, 0.25 + (2 * serviceEditMs) / 1000);
}

/** Money's now: the clock, or the time settled up to if the clock has been set back behind it, so
 * a charge or the day shown never lands on a day already closed (T-488). */
export function moneyNow(): number {
  return Math.max(gameNow(), settledT);
}

/** The entry for `key` in a ledger kept in order of `at`, made if missing (in its place, so a day or
 * hour is never listed twice, T-488); the ledger keeps the newest `cap`. */
function entryIn<T extends { fares: number; running: number; trains: number; build: number; rides?: number }>(list: T[], at: (e: T) => number, key: number, make: () => T, cap: number): T {
  let i = list.length;
  while (i > 0 && at(list[i - 1]) > key) i--;
  if (i > 0 && at(list[i - 1]) === key) return list[i - 1];
  const e = make();
  list.splice(i, 0, e);
  if (list.length > cap) list.splice(0, list.length - cap);
  return e;
}

/** Game day `day`'s ledger entry; the ledger keeps LEDGER_DAYS days. */
export function dayEntry(day: number): DayLedger {
  return dayIn(ledger, day);
}

/** Day `day`'s entry in a day ledger, made if missing; a new day folds the ledger's old days (T-3757) */
function dayIn(list: DayLedger[], day: number): DayLedger {
  const n = list.length, e = entryIn(list, (d) => d.day, day, () => newDay(day), LEDGER_DAYS);
  if (list.length > n) foldLedger(list, day);
  return e;
}

/** The hour's entry in the hourly ledger (T-447); it keeps LEDGER_HOURS hours. */
export function hourEntry(hour: number): HourLedger {
  return entryIn(hourLedger, (d) => d.hour, hour, () => newHour(hour), LEDGER_HOURS);
}

/** T-531: a region's entries for game day `day` and hour `hour`, made if missing. */
export function regionEntries(zone: string, day: number, hour: number): [DayLedger, HourLedger] {
  let r = regions.get(zone);
  if (!r) regions.set(zone, (r = { days: [], hours: [] }));
  return [
    dayIn(r.days, day),
    entryIn(r.hours, (d) => d.hour, hour, () => newHour(hour), LEDGER_HOURS),
  ];
}

/** T-2787: each line's recent days of money, US$M: [day, fares, running cost] per day, oldest first,
 * the day now last. A forecast is not kept, these are what settling paid. Saved in `SaveHeader.lineDays`. */
export let lineDays = new Map<number, [number, number, number][]>();
/** Days kept per line: 30 closed days and the day now (T-3566: the line panel's chart is the Money tab's 30 days). */
const LINE_DAYS = 31;

function lineDayEntry(id: number, day: number): [number, number, number] {
  let l = lineDays.get(id);
  if (!l) lineDays.set(id, (l = []));
  let i = l.length;
  while (i > 0 && l[i - 1][0] > day) i--;
  if (i > 0 && l[i - 1][0] === day) return l[i - 1];
  const e: [number, number, number] = [day, 0, 0];
  l.splice(i, 0, e);
  if (l.length > LINE_DAYS) l.splice(0, l.length - LINE_DAYS);
  return e;
}

/** The lines' days as the save and the panels read them: per line id, [day, fares, running] flattened;
 * `round` keeps $100 (the charts cannot tell). */
export function savedLineDays(round = false): Record<string, number[]> {
  const r = (v: number) => (round ? Math.round(v * 1e4) / 1e4 : v);
  return Object.fromEntries([...lineDays].filter(([, l]) => l.length).map(([id, l]) => [id, l.flatMap(([d, f, c]) => [d, r(f), r(c)])]));
}

/** Days after `last` are the ones settling pays again on a load. */
export function loadLineDays(saved: SaveHeader["lineDays"], last: number) {
  lineDays = new Map();
  for (const [id, v] of Object.entries(saved ?? {})) {
    const l: [number, number, number][] = [];
    for (let i = 0; i + 2 < v.length; i += 3) if (v[i] <= last) l.push([v[i], v[i + 1], v[i + 2]]);
    if (l.length) lineDays.set(Number(id), l.slice(-LINE_DAYS));
  }
  dropOldLineDays(last);
}

/** A line's days older than the month kept go, also those of a line that has made none since (T-3771). */
function dropOldLineDays(today: number) {
  for (const [id, l] of lineDays) {
    const n = l.findIndex((e) => e[0] > today - LINE_DAYS);
    if (n === 0) continue;
    if (n < 0) lineDays.delete(id);
    else l.splice(0, n);
  }
}

/** T-531: every region's today and this hour that has an entry today. */
function regionNow(): RegionNow {
  const day = ledgerDay(moneyNow()), hour = Math.floor(moneyNow() / 3600), out: RegionNow = {};
  for (const [k, r] of regions) {
    const d = r.days[r.days.length - 1], h = r.hours[r.hours.length - 1];
    if (d?.day === day) out[k] = { today: { ...d }, hour: h?.hour === hour ? { ...h } : newHour(hour) };
  }
  return out;
}

/** T-531: the running lines' money by region for settling: each line's fare income tables and its
 * running cost per game hour at each demand level (as `running_cost`), with its local time. */
function regionLines() {
  const a = api!, li = a.lines_info(), unit = prices.economy / 1e6;
  const costs: { id: number; shift: number; run: number[]; tail: number[][]; rowOf: (day: number) => DayRow; shares: [string, number][] }[] = [];
  for (let i = 0; i < li.length; i += 22) {
    if (li[i + 3] !== 1) continue;
    const id = li[i], km = li[i + 9] / 1000, cars = li[i + 21] || li[i + 8];
    const [perTrainKm, perCarKm] = a.line_fast(id) ? [prices.hsRunCostTrainKm, prices.hsRunCostCarKm] : [prices.runCostTrainKm, prices.runCostCarKm];
    const f = a.line_run_factor(id); // T-2669: its countries' prices
    const ways = a.line_one_way(id) ? 1 : 2; // T-2981
    // the rates from the track model, so the cost drained is that of the trains it runs
    const tph = TEN.map((k) => a.line_tph_period_on(id, Math.floor(k / NP), k % NP));
    const run = tph.map((t) => t * ways * km * (perTrainKm + perCarKm * cars) * unit * f);
    costs.push({ id, shift: a.line_shift(id), run, tail: runTail(li, i, run, tph), rowOf: rowOfLine(id), shares: lineShares.get(id)?.run ?? [] });
  }
  const fares = [...lineIncome].map(([id, per]) => ({ id, shift: a.line_shift(id), kindOf: kindOfLine(id), per, shares: lineShares.get(id)?.fare ?? [] }));
  const rides = [...lineRides].map(([id, per]) => ({ shift: a.line_shift(id), kindOf: kindOfLine(id), per, shares: lineShares.get(id)?.ride ?? [] }));
  return { costs, fares, rides };
}

/** T-531: hour `t`'s fares (US$M), running cost (US$M) and train rides (T-565) in each region. A
 * line not yet placed counts under "". */
function regionHour(t: number, rl: ReturnType<typeof regionLines>): Map<string, [number, number, number]> {
  const out = new Map<string, [number, number, number]>();
  const spread = (shares: [string, number][], v: number, k: 0 | 1 | 2) => {
    if (!v) return;
    for (const [z, s] of shares.length ? shares : [["", 1] as [string, number]]) {
      let e = out.get(z);
      if (!e) out.set(z, (e = [0, 0, 0]));
      e[k] += v * s;
    }
  };
  for (const f of rl.fares) spread(f.shares, fareHour(f.per, t + f.shift, f.kindOf), 0);
  for (const f of rl.rides) spread(f.shares, rideHour(f.per, t + f.shift, f.kindOf), 2);
  for (const c of rl.costs) spread(c.shares, costHour(c.run, t + c.shift, c.tail, c.rowOf), 1);
  return out;
}

/** T-531: a region's saved ledgers, as `mergeLedger` makes the network's. */
export function loadRegions(saved: SaveHeader["regions"]) {
  regions = new Map(Object.entries(saved ?? {}).map(([k, r]) => [k, { days: mergeLedger(r.ledger, (d) => d.day), hours: mergeLedger(r.hours, (d) => d.hour) }]));
  heldCharges.clear();
}

export function savedRegions(copy: <T extends object>(e: T) => T = (e) => ({ ...e })): NonNullable<SaveHeader["regions"]> {
  return Object.fromEntries([...regions].map(([k, r]) => [k, { ledger: r.days.map(copy), hours: r.hours.map(copy) }]));
}

/** An entry as the save keeps it (T-567): money to the dollar and rides to a tenth, which the
 * charts cannot tell from the full figures and which gzips to about 60% of them. */
export function rounded<T extends object>(e: T): T {
  const out: Record<string, number> = { ...e } as Record<string, number>;
  for (const k of ["fares", "running", "upkeep", "loan", "trains", "build"]) if (typeof out[k] === "number") out[k] = Math.round(out[k] * 1e6) / 1e6;
  if (typeof out.rides === "number") out.rides = Math.round(out.rides * 10) / 10;
  if (typeof out.points === "number") out.points = Math.round(out.points);
  return out as T;
}

/** A saved ledger in order with one entry per day or hour: saves written before T-488 can list a
 * day twice, the second time empty, which merging adds back where it belongs. */
export function mergeLedger<T extends { fares: number; running: number; trains: number; build: number; upkeep?: number; loan?: number; interest?: number; rides?: number; points?: number }>(list: T[] | undefined, at: (e: T) => number): T[] {
  const by = new Map<number, T>();
  for (const old of list ?? []) {
    // T-2022: a save's interest is its loan payments
    const { interest, ...e } = old as T;
    if (interest) (e as T).loan = (e.loan ?? 0) + interest;
    const k = at(e as T), d = by.get(k);
    if (!d) by.set(k, { ...e, rides: e.rides ?? 0, points: e.points ?? 0 } as T);
    else { addEntry(d, e); if (e.loan) d.loan = (d.loan ?? 0) + e.loan; d.trains += e.trains; d.build += e.build; }
  }
  return [...by.values()].sort((a, b) => at(a) - at(b));
}

/** A load settles again from the save's clock, so what the loaded ledgers hold after it leaves them
 * and its net comes back into the cash; edits' charges stay, on the clock's day and hour. Such
 * entries come from a clock pulled back (T-488, minutes) or, before T-715, a clock of the game open
 * before (up to a year of running costs with no fares, charged again by each such load). The running
 * cost is worked out again on the loaded network up to where the hour ledger ends, times the whole
 * days kept over the same worked out again. Returns US$M. */
export function unsettleAfter(clockT: number): number {
  const D = ledgerDay(clockT), H = Math.floor(clockT / 3600);
  const last = hourLedger[hourLedger.length - 1];
  if (!last || last.hour <= H) return 0;
  const groups = localGroups();
  const costAt = (h: number) => groups.reduce((s, g) => s + g.costs.reduce((x, c) => x + costHour(c.run, h * 3600 + g.shift, c.tail, c.rowOf), 0), 0);
  const replay = (from: number, to: number) => {
    let c = 0;
    for (let t = from; t < to;) {
      const h = Math.floor(t / 3600), e = Math.min(to, (h + 1) * 3600);
      c += costAt(h) * (e - t) / 3600;
      t = e;
    }
    return c;
  };
  const whole = ledger.filter((d) => d.day > D && d.day < ledgerDay(last.hour * 3600));
  const kept = whole.reduce((s, d) => s + d.running, 0), again = whole.reduce((s, d) => s + replay(d.day * 86400 - dayShift, (d.day + 1) * 86400 - dayShift), 0);
  const times = kept > 0 && again > 0 ? kept / again : 1;
  const full = costAt(last.hour) * times;
  const end = last.hour * 3600 + 3600 * (full > 0 ? Math.min(1, last.running / full) : 1);
  const back = times * replay(Math.max(clockT, end - 86400 * 366), end); // settle's year at most
  // T-867: upkeep drains evenly, so its hours since the clock are worked out again the same way
  const upHour = upkeepPerHour(), upEnd = last.hour * 3600 + 3600 * (upHour > 0 ? Math.min(1, (last.upkeep ?? 0) / upHour) : 0);
  const upBack = upHour * Math.min(366 * 24, Math.max(0, upEnd - clockT) / 3600);
  const trim = (days: DayLedger[], hours: HourLedger[]) => {
    const later = days.filter((d) => d.day > D), laterH = hours.filter((e) => e.hour > H);
    const today = laterH.filter((e) => ledgerDay(e.hour * 3600) === D);
    const fares = [...later, ...today].reduce((s, e) => s + e.fares, 0);
    const keep = { days: days.filter((d) => d.day <= D), hours: hours.filter((e) => e.hour <= H) };
    const d = keep.days.find((x) => x.day === D);
    if (d) for (const e of today) addEntry(d, e, -1);
    const carry = <T extends DayLedger | HourLedger>(list: T[], from: T[], at: (e: T) => number, key: number, make: () => T, cap: number) => {
      const trains = from.reduce((s, e) => s + e.trains, 0), build = from.reduce((s, e) => s + e.build, 0);
      if (trains || build) { const e = entryIn(list, at, key, make, cap); e.trains += trains; e.build += build; }
    };
    carry(keep.days, later, (x) => x.day, D, () => newDay(D), LEDGER_DAYS);
    carry(keep.hours, laterH, (x) => x.hour, H, () => newHour(H), LEDGER_HOURS);
    return { ...keep, fares };
  };
  const net = trim(ledger, hourLedger);
  putLedger(net.days);
  putHourLedger(net.hours);
  for (const [k, r] of regions) { const t = trim(r.days, r.hours); regions.set(k, { days: t.days, hours: t.hours }); }
  return back + upBack - net.fares;
}

/** T-867: the network's upkeep, US$M a game hour: it drains evenly through the day, trains or none. */
function upkeepPerHour(): number {
  return (api?.upkeep()[0] ?? 0) / 24;
}
function upkeepNow(): MoneyState["upkeep"] {
  const u = api!.upkeep();
  return { track: u[1], stations: u[2], km: u[3], count: u[4], buses: u[5] ?? 0, busKm: u[6] ?? 0, ports: u[7] ?? 0, portCount: u[8] ?? 0, upgrades: u[9] ?? 0 };
}

export function moneyState(): MoneyState {
  const a = api!;
  dayEntry(ledgerDay(moneyNow()));
  hourEntry(Math.floor(moneyNow() / 3600));
  return {
    cash: a.cash(),
    dayShift,
    mode,
    ...(copyOf ? { copy: true } : {}),
    ...(modded ? { modded: true } : {}),
    game,
    fleet: a.fleet(),
    carsNeeded: a.cars_needed(),
    fleetFast: a.fleet_fast(),
    carsNeededFast: a.cars_needed_fast(),
    fleetBus: a.fleet_bus(),
    busesNeeded: a.buses_needed(),
    fleetFerry: a.fleet_ferry(),
    boatsNeeded: a.boats_needed(),
    fleetFerrySmall: a.fleet_ferry_small(),
    boatsNeededSmall: a.boats_needed_small(),
    fleetFerryLarge: a.fleet_ferry_large(),
    boatsNeededLarge: a.boats_needed_large(),
    fleetTram: a.fleet_tram(),
    tramsNeeded: a.trams_needed(),
    tramPrice: TrackApi.kind_defaults(3)[4],
    ...prices,
    upkeep: upkeepNow(),
    loan: loan && { ...loan },
    fares: { ...fares },
    runningPerHour: PERIODS.map((_, p) => a.running_cost_period(p)),
    ledger: ledger.map((d) => ({ ...d })),
    hours: hourLedger.map((d) => ({ ...d })),
    regions: savedRegions(),
    lineDays: savedLineDays(),
    stars: Object.fromEntries(stars),
    goals: Object.fromEntries(goals),
    notes: Object.fromEntries(lineNotes),
    ...(guide ? { guide } : {}),
    ...(achievements ? { achievements: Object.fromEntries(achievements) } : {}),
  };
}

export function sendMoney() {
  const m = moneyState();
  sentCash = m.cash;
  sentFares = m.ledger[m.ledger.length - 1]?.fares ?? NaN;
  send({ kind: "money", ...moneyPost(m) });
}

/** T-284: the cash and today's ledger, if either moved since last sent. */
export function sendCash() {
  const cash = api!.cash(), today = dayEntry(ledgerDay(moneyNow())), hour = hourEntry(Math.floor(moneyNow() / 3600));
  if (cash === sentCash && today.fares === sentFares) return;
  sentCash = cash;
  sentFares = today.fares;
  send({ kind: "cash", cash, today: { ...today }, hour: { ...hour }, regions: regionNow() });
}

/** One local time's fares and rides in one country (T-872): per kind of day and period. */
interface CountryPart { kindOf: KindOf; income: number[][]; rides: number[][] }
/** One local time's running cost: per hour by level and kind of day (`costHour`), its tails
 * (`runTail`), and the kind of each local day (its country's, when it has one). */
interface CostPart { run: number[]; tail: number[][] | undefined; rowOf: (day: number) => DayRow }

/** T-309: the running lines grouped by the local time they run on, each group's running cost per
 * demand level and its fare income per kind of day and period, split by country (T-872, T-262: each
 * country's holidays and weekend decide its kinds of day). With one local time and one country
 * (every game in the Northeast) the network's totals. */
function localGroups(): { shift: number; costs: CostPart[]; parts: CountryPart[] }[] {
  const a = api!;
  const shifts = [...a.running_shifts()];
  const costs = regionLines().costs;
  // lines in several countries have no one holiday calendar between them, so such a group takes the
  // week's
  const countries = [...new Set([...lineIncome.keys(), ...lineRides.keys()].map((id) => lineCountries.get(id) ?? ""))];
  const costParts = (s: number | null, running: number[]): CostPart[] => {
    let tail: number[][] | undefined;
    for (const c of costs) {
      if (s !== null && c.shift !== s) continue;
      tail ??= [0, 1].map(() => PERIODS.map(() => 0));
      c.tail.forEach((row, r) => row.forEach((v, p) => { tail![r][p] += v; }));
    }
    return [{ run: running, tail, rowOf: countries.length === 1 ? rowOfIn(holidayData, countries[0]) : rowOfDay }];
  };
  const one = shifts.length <= 1;
  if (one && countries.length <= 1) return [{ shift: shifts[0] ?? 0, costs: costParts(null, TEN.map((k) => a.running_cost_period(k))), parts: [{ kindOf: kindOfIn(holidayData, countries[0]), income: incomePerHour, rides: ridesPerHour }] }];
  return (one ? [shifts[0] ?? 0] : shifts).map((s) => {
    const sum = (zero: number[][], by: Map<number, number[][]>, c: string) => {
      const out = zero.map((p) => p.map(() => 0));
      for (const [id, per] of by) {
        if ((!one && a.line_shift(id) !== s) || (lineCountries.get(id) ?? "") !== c) continue;
        per.forEach((row, k) => row.forEach((v, p) => { if (out[k]) out[k][p] += v; }));
      }
      return out;
    };
    const parts = countries.map((c) => ({ kindOf: kindOfIn(holidayData, c), income: sum(incomePerHour, lineIncome, c), rides: sum(ridesPerHour, lineRides, c) }));
    return { shift: s, costs: costParts(one ? null : s, TEN.map((k) => (one ? a.running_cost_period(k) : a.running_cost_period_in(k, s)))), parts };
  });
}

/** Fares in and running costs out up to now (T-284: to the game second, not the whole hour). An
 * hour's fare (`fareHour`: the period's income shaped as the riders arrive, landing as rides end)
 * is paid as its trains pick up riders (`hourShape`), its running cost (`costHour`) evenly; by the
 * hour's end both are paid in full. Returns whether an hour ended. */
export function settle(): boolean {
  if (!api) return false;
  const now = gameNow();
  if (now - settledT > 86400 * 366) { // a year at most in one go
    putSettledT(Math.floor((now - 86400 * 366) / 3600) * 3600);
    putPartHour(settledT / 3600);
    putPartCost(0); putPartFare(partCost);
  }
  if (now <= settledT) return false;
  // T-3851: with trips off the trains run empty, so nothing is settled: the clock moves past, cash and ledgers stay
  if (moneyFrozen) {
    const ended = Math.floor(now / 3600) > Math.floor(settledT / 3600);
    putSettledT(now);
    putPartHour(Math.floor(now / 3600)); putPartCost(0); putPartFare(0);
    return ended;
  }
  const groups = localGroups(), rl = regionLines(), up = upkeepPerHour();
  const upShares: [string, number][] = upkeepRegions.length ? upkeepRegions : [["", 1]];
  const pl = [...linePoints].map(([id, regs]) => ({ shift: api!.line_shift(id), kindOf: kindOfLine(id), regs }));
  let cash = api.cash(), ended = false;
  while (settledT < now) {
    const h = Math.floor(settledT / 3600);
    if (h !== partHour) { putPartHour(h); putPartCost(0); putPartFare(partCost); }
    const t = h * 3600, end = Math.min(now, t + 3600), full = end >= t + 3600;
    // T-309: each local time's lines at their own period and day; T-260: the hour's own kind of
    // day, the weekday's until a weekend's has come; T-736: shaped and lagged (fareHour)
    let fare = 0, cost = 0, rides = 0;
    for (const g of groups) {
      const tl = t + g.shift;
      for (const c of g.parts) {
        fare += fareHour(c.income, tl, c.kindOf);
        rides += rideHour(c.rides, tl, c.kindOf);
      }
      for (const c of g.costs) cost += costHour(c.run, tl, c.tail, c.rowOf);
    }
    const wf = full ? 1 : shareAt(h, end - t), wc = full ? 1 : (end - t) / 3600;
    const df = Math.max(0, wf - partFare), dc = Math.max(0, wc - partCost);
    putPartFare(Math.max(partFare, wf));
    putPartCost(Math.max(partCost, wc));
    const owed = loan ? loanDrain(loan, dc / 24) : 0;
    if (loan && loan.days <= 0) putLoan(null);
    cash += fare * df - (cost + up) * dc - owed;
    const d = dayEntry(ledgerDay(t));
    d.fares += fare * df;
    d.running += cost * dc;
    d.upkeep = (d.upkeep ?? 0) + up * dc;
    d.rides = (d.rides ?? 0) + rides * df;
    const e = hourEntry(h);
    e.fares += fare * df;
    e.running += cost * dc;
    e.upkeep = (e.upkeep ?? 0) + up * dc;
    e.rides = (e.rides ?? 0) + rides * df;
    if (owed > 0) {
      d.loan = (d.loan ?? 0) + owed;
      e.loan = (e.loan ?? 0) + owed;
    }
    for (const [z, [f, c, n]] of regionHour(t, rl)) {
      for (const r of regionEntries(z, d.day, h)) { r.fares += f * df; r.running += c * dc; r.rides = (r.rides ?? 0) + n * df; }
    }
    if (d.day > ledgerDay(now) - LINE_DAYS) { // T-2787: each line's own, for its panel's history
      for (const f of rl.fares) { const v = fareHour(f.per, t + f.shift, f.kindOf) * df; if (v) lineDayEntry(f.id, d.day)[1] += v; }
      for (const c of rl.costs) { const v = costHour(c.run, t + c.shift, c.tail) * dc; if (v) lineDayEntry(c.id, d.day)[2] += v; }
    }
    if (up * dc > 0) for (const [z, s] of upShares) {
      for (const r of regionEntries(z, d.day, h)) r.upkeep = (r.upkeep ?? 0) + up * dc * s;
    }
    // T-684: points land as fares do, each line's at its own local period and kind of day
    if (df > 0) for (const { shift, kindOf, regs } of pl) {
      for (const [z, per] of regs) {
        const v = fareHour(per, t + shift, kindOf) * df;
        if (!(v > 0)) continue;
        d.points = (d.points ?? 0) + v;
        e.points = (e.points ?? 0) + v;
        for (const r of regionEntries(z, d.day, h)) r.points = (r.points ?? 0) + v;
      }
    }
    putSettledT(full ? t + 3600 : end);
    if (full) { ended = true; hourShape.delete(h); }
  }
  if (ended && lineDays.size) { // T-2787: a deleted line's days go with it
    dropOldLineDays(ledgerDay(now));
    const li = api.lines_info(), have = new Set<number>();
    for (let i = 0; i < li.length; i += 22) have.add(li[i]);
    for (const id of [...lineDays.keys()]) if (!have.has(id)) lineDays.delete(id);
  }
  api.set_cash(cash);
  if (kept) kept.cash = cash;
  saves.dirty = true;
  return ended;
}

/** T-1677, T-2236: the per-period crowding numbers (the first `PERIODS * PERIOD_STRIDE` of
 * `line_periods`) and the busiest hour's load per period and stretch that ends it of each line in
 * `lines`, each line's as its count of numbers and then them (none for a line with no account). */
export function linesPeriods(a: TrackApi, lines: number[], day: number): Float64Array {
  const P = PERIODS.length, block = P * 11;
  const per = lines.map((n) => {
    const v = a.line_periods(n, day);
    // per period: 11 numbers, 2 per stop, 3 per stretch (stops - 1): 5 * stops + 8
    const stops = (v.length / P - 8) / 5, tail = Number.isInteger(stops) && stops > 1 ? P * (stops - 1) : 0;
    if (v.length < block) return v;
    const out = new Float64Array(block + tail);
    out.set(v.subarray(0, block));
    if (tail) out.set(v.subarray(v.length - tail), block);
    return out;
  });
  const out = new Float64Array(per.reduce((s, v) => s + 1 + v.length, 0));
  let i = 0;
  for (const v of per) { out[i++] = v.length; out.set(v, i); i += v.length; }
  return out;
}

/** T-446: every line's past 24 hours (`TrackApi.past_day`, `today`'s layout), then each listed
 * line's fares taken in them, US$M, in the same order: the hour now by the share already paid, as
 * `settle` pays the network's, and the hour a day ago by the rest; then each one's fares for the
 * whole of its local day now (T-2475). */
export function today(): Float64Array {
  if (settle()) sendMoney();
  const now = Math.min(gameNow(), settledT);
  // T-1815: `past_day` takes ~40-50 ms on shared New York, and the panels ask several times over
  // while a solve publishes; the same question with nothing changed since gets the last answer
  const key = `${todayGen}|${now}|${partHour}|${partFare}`;
  if (todayLast?.key === key) return todayLast.data.slice();
  // T-2793: while the account works through new riders a line can have a new account for one kind of day and none yet
  // for yesterday's, and its past 24 hours would read only today so far: the last answer stands until it has finished
  if (todayLast && api!.boarding_step(0)) return todayLast.data.slice();
  const data = packDays(pastDays(now));
  todayLast = { key, data };
  return data.slice();
}
/** T-1815: moved on by every message and task that can change what `today` reads (the account,
 * the network, the fares); previews and queries leave it */
export let todayGen = 0;
export let todayLast: { key: string; data: Float64Array } | null = null;

/** One line's day, in `TrackApi.past_day`'s per-line layout, with its fares (US$M). */
interface LineDay { id: number; block: ArrayLike<number>; fare: number; day: number }

/** Every line's block of `TrackApi.past_day(now)` with its fares taken: each hour at the line's
 * local period and day (T-309), the hour now by the share already paid and the hour a day ago by
 * the rest; and (T-2475) its fares for the whole of its local day now, every hour on its own kind
 * of day. */
function pastDays(now: number): LineDay[] {
  const a = api!;
  syncHolidays();
  const v = a.past_day(now), count = v[0], hNow = Math.floor(now / 3600);
  const h0 = hNow - 24, wNow = hNow === partHour ? partFare : 0;
  const out: LineDay[] = [];
  let i = 1;
  for (let k = 0; k < count; k++) {
    const id = v[i], n = v[i + 1], spans = v[i + 6], len = 15 + 2 * spans + n;
    const sh = a.line_shift(id), per = lineIncome.get(id);
    let f = 0, day = 0;
    if (per) {
      for (let h = h0; h <= hNow; h++) f += fareHour(per, h * 3600 + sh, kindOfLine(id)) * (h === hNow ? wNow : h === h0 ? 1 - wNow : 1);
      for (let h = Math.floor((now + sh) / 86400) * 24 - sh / 3600, e = h + 24; h < e; h++) day += fareHour(per, h * 3600 + sh, kindOfLine(id));
    }
    out.push({ id, block: v.subarray(i, i + len), fare: f, day });
    i += len;
  }
  return out;
}

/** The flat array the panels read (game/today.ts `readToday`): the line count, each line's block,
 * then each line's fares, then each line's fares for its whole day now (T-2475). */
function packDays(items: LineDay[]): Float64Array {
  const len = items.reduce((s, d) => s + d.block.length, 1);
  const out = new Float64Array(len + 2 * items.length);
  out[0] = items.length;
  let i = 1;
  for (const d of items) { out.set(d.block, i); i += d.block.length; }
  items.forEach((d, k) => { out[i + k] = d.fare; out[i + items.length + k] = d.day; });
  return out;
}

/** The share of hour `h`'s fares paid by `s` seconds into it (T-284). */
function shareAt(h: number, s: number): number {
  const c = hourShape.get(h);
  if (!c) return s / 3600;
  return c[Math.max(0, Math.min(3600, Math.floor(s)))];
}

/** T-284: when hour `epoch`'s fares are paid, from its slice of the boarding account
 * (game/boarding.ts `readBoardHour` has the layout): each train leaving a platform in the hour
 * takes the riders waiting less those it leaves behind, and the hour's fares are paid in
 * proportion as they board. Only the timing within the hour comes from here. */
export function shapeHour(epoch: number, b: Float32Array) {
  const h = Math.round(epoch / 3600);
  for (const k of hourShape.keys()) if (k < partHour - 1) hourShape.delete(k);
  const nt = b[0], nl = b[1], nq = b[2], ne = b[3], S = 12;
  const q0 = 5 + 2 * nt + nl, e0 = q0 + nq * S, p0 = e0 + ne * 2;
  const bins = new Float64Array(3600);
  let total = 0;
  for (let q = 0; q < nq; q++) {
    const Q = q0 + q * S, rate = b[Q + 4], first = b[Q + 8], n = b[Q + 9];
    let p = b[Q + 10];
    const pEnd = p + b[Q + 11];
    for (let j = first + 1; j < first + n; j++) {
      const t = b[e0 + j * 2];
      const tPrev = b[e0 + (j - 1) * 2], leftPrev = b[e0 + (j - 1) * 2 + 1];
      // riders changing from other lines' trains since the train before (T-554)
      let changed = 0;
      for (; p < pEnd && b[p0 + p * 2] <= t; p++) if (b[p0 + p * 2] > tPrev) changed += b[p0 + p * 2 + 1];
      if (!(t >= 0 && t < 3600)) continue;
      const w = Math.max(0, leftPrev + rate * Math.max(0, t - tPrev) + changed - b[e0 + j * 2 + 1]);
      bins[Math.floor(t)] += w;
      total += w;
    }
  }
  if (!(total > 0)) { hourShape.delete(h); return; }
  const c = new Float32Array(3601);
  let acc = 0;
  for (let s = 0; s < 3600; s++) { c[s] = acc / total; acc += bins[s]; }
  c[3600] = 1;
  hourShape.set(h, c);
}
export function putTodayGen(v: typeof todayGen) { todayGen = v; }
export function putTodayLast(v: typeof todayLast) { todayLast = v; }
