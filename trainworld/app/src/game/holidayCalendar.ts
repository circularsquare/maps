// The holiday calendar's lookups (T-862, T-872, T-262), plain so the clock worker imports them too:
// game/holidays.ts loads the data on the main thread, the worker its own copy. A holiday is a
// whole local day of its country and runs as a Sunday there, unless it is one most people work
// through; the country's own weekend decides its Saturdays and Sundays, and a weekend day it works
// runs as a weekday (docs/world.md "Calendar").

import { dayKindOf, dayOfYear, newYearWeekday, weekdayOf, type DayKind } from "./days";

export interface HolidayData {
  names: string[];
  /** ISO country -> January 1's weekday -> [day of year, name index] */
  countries: Record<string, Record<string, [number, number][]>>;
  /** ISO country -> each weekday's kind, Monday first, where not Saturday and Sunday */
  weeks?: Record<string, DayKind[]>;
  /** ISO country -> January 1's weekday -> weekend days of the year worked in place of a day off */
  workdays?: Record<string, Record<string, number[]>>;
  /** ISO country -> name indices of its holidays that keep their weekday */
  working?: Record<string, number[]>;
  /** T-264: ISO country -> January 1's weekday -> superholiday runs: [first going-out day of the year
   * (may be before January 1), days going out, days away, days coming back, name index] */
  supers?: Record<string, Record<string, [number, number, number, number, number][]>>;
}

/** A superholiday run in game days (T-264): going out from `out`, away from `away`, coming back
 * from `back`, over at `end` (exclusive). */
export interface SuperRun { name: string; out: number; away: number; back: number; end: number }

const runsBy = new WeakMap<HolidayData, Map<string, Map<number, SuperRun[]>>>();
/** The superholiday runs of `country` that meet game days [d0, d1), in order. A run listed under
 * two game years (it crosses New Year) is taken once. */
export function superRuns(d: HolidayData | null, country: string | null | undefined, d0: number, d1: number): SuperRun[] {
  const years = country ? d?.supers?.[country] : undefined;
  if (!d || !years) return [];
  let m = runsBy.get(d);
  if (!m) runsBy.set(d, (m = new Map()));
  let byYear = m.get(country!);
  if (!byYear) m.set(country!, (byYear = new Map()));
  const out: SuperRun[] = [];
  // a game year starts on its January 1; runs reach at most a few weeks either side of theirs
  for (let y0 = d0 - dayOfYear(d0) - 365; y0 <= d1 + 365; y0 += 365) {
    let list = byYear.get(y0);
    if (!list) {
      list = (years[String(newYearWeekday(y0))] ?? []).map(([s, a, b, c, n]) => ({ name: d.names[n], out: y0 + s, away: y0 + s + a, back: y0 + s + a + b, end: y0 + s + a + b + c }));
      byYear.set(y0, list);
    }
    for (const r of list) if (r.end > d0 && r.out < d1 && !out.some((x) => x.out < r.end && r.out < x.end)) out.push(r);
  }
  return out.sort((a, b) => a.out - b.out);
}

/** Which part of a superholiday local game day `day` is in `country`: 1 going out, 2 away, 3 coming
 * back, 0 none; with its run. */
export function superPart(d: HolidayData | null, country: string | null | undefined, day: number): [0 | 1 | 2 | 3, SuperRun | null] {
  const r = superRuns(d, country, day, day + 1)[0];
  return !r ? [0, null] : [day < r.away ? 1 : day < r.back ? 2 : 3, r];
}

/** The holiday's name in `country` on its local game day `day`, or null. */
export function holidayName(d: HolidayData | null, country: string | null | undefined, day: number): string | null {
  const rows = country ? d?.countries[country]?.[String(newYearWeekday(day))] : undefined;
  if (!d || !rows) return null;
  const doy = dayOfYear(day);
  const r = rows.find((x) => x[0] === doy);
  return r ? d.names[r[1]] : null;
}

const kinds = new WeakMap<HolidayData, Map<string, (day: number) => DayKind>>();
/** The kind of each local game day in `country`: a superholiday's away, going-out or coming-back day
 * on its days (T-264, T-1373), a Sunday on its holidays (working ones aside), a weekday on the
 * weekend days it works, else its own week's kind for the weekday. Without data or a country,
 * `dayKindOf`. */
export function kindOfIn(d: HolidayData | null, country: string | null | undefined): (day: number) => DayKind {
  if (!d || !country || !d.countries[country]) return dayKindOf;
  let m = kinds.get(d);
  if (!m) kinds.set(d, (m = new Map()));
  let f = m.get(country);
  if (!f) {
    const key = (wd: string, doy: number) => Number(wd) * 1000 + doy;
    const working = new Set(d.working?.[country] ?? []);
    const off = new Set<number>(), worked = new Set<number>();
    for (const [wd, rows] of Object.entries(d.countries[country])) for (const [doy, name] of rows) if (!working.has(name)) off.add(key(wd, doy));
    for (const [wd, days] of Object.entries(d.workdays?.[country] ?? {})) for (const doy of days) worked.add(key(wd, doy));
    // a run's days before January 1 or after December 31 belong to the year before or after,
    // whose January 1 is a weekday earlier or later
    const rush = new Map<number, DayKind>();
    for (const [wd, runs] of Object.entries(d.supers?.[country] ?? {})) {
      const w = Number(wd);
      for (const [s, a, b, c] of runs) {
        for (let doy = s; doy < s + a + b + c; doy++) {
          const at = doy < 0 ? key(String((w + 6) % 7), doy + 365) : doy >= 365 ? key(String((w + 1) % 7), doy - 365) : key(wd, doy);
          rush.set(at, doy < s + a ? 4 : doy < s + a + b ? 3 : 5);
        }
      }
    }
    const week = d.weeks?.[country];
    const own = week ? (day: number): DayKind => week[weekdayOf(day)] ?? 0 : dayKindOf;
    f = (day) => {
      const k = newYearWeekday(day) * 1000 + dayOfYear(day);
      return rush.get(k) ?? (off.has(k) ? 2 : worked.has(k) ? 0 : own(day));
    };
    m.set(country, f);
  }
  return f;
}
