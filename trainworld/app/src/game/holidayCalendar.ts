// The holiday calendar's lookups (T-862, T-872), plain so the clock worker imports them too:
// game/holidays.ts loads the data on the main thread, the worker its own copy. A holiday is a
// whole local day of its country and runs as a weekend day there (docs/world.md "Calendar").

import { dayKindOf, dayOfYear, newYearWeekday, type DayKind } from "./days";

export interface HolidayData {
  names: string[];
  /** ISO country -> January 1's weekday -> [day of year, name index] */
  countries: Record<string, Record<string, [number, number][]>>;
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
/** The kind of each local game day in `country`: a weekend day on its holidays, else the day
 * number's. Without data or a country, `dayKindOf`. */
export function kindOfIn(d: HolidayData | null, country: string | null | undefined): (day: number) => DayKind {
  if (!d || !country || !d.countries[country]) return dayKindOf;
  let m = kinds.get(d);
  if (!m) kinds.set(d, (m = new Map()));
  let f = m.get(country);
  if (!f) {
    const keys = new Set<number>();
    for (const [wd, rows] of Object.entries(d.countries[country])) for (const [doy] of rows) keys.add(Number(wd) * 1000 + doy);
    f = (day) => (keys.has(newYearWeekday(day) * 1000 + dayOfYear(day)) ? 1 : dayKindOf(day));
    m.set(country, f);
  }
  return f;
}
