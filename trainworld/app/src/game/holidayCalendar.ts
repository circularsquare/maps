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
/** The kind of each local game day in `country`: a Sunday on its holidays (working ones aside),
 * a weekday on the weekend days it works, else its own week's kind for the weekday. Without data
 * or a country, `dayKindOf`. */
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
    const week = d.weeks?.[country];
    const own = week ? (day: number): DayKind => week[weekdayOf(day)] ?? 0 : dayKindOf;
    f = (day) => {
      const k = newYearWeekday(day) * 1000 + dayOfYear(day);
      return off.has(k) ? 2 : worked.has(k) ? 0 : own(day);
    };
    m.set(country, f);
  }
  return f;
}
