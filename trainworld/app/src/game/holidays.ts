// Public holidays (T-862, docs/world.md "Calendar"): each country's national public holidays from
// holidays.json (pipeline/holidays_json.py), fetched after startup as its own chunk. A game year
// takes the holidays of the real non-leap year whose January 1 falls on the same weekday, so a
// holiday tied to a weekday keeps it. A holiday is a whole day of its country's local time.
//
// The top bar's sky marks them for the countries the player has stations in. A holiday runs as a
// weekend day for the lines in its country (T-872): the clock worker sends the track model each
// line's holiday days and settles money by them; here, the view's Today and the line inspector.

import { computed, signal } from "@preact/signals";
import { clock } from "./clock";
import { cityById, cityWithin } from "./cities";
import { holidayName, kindOfIn, type HolidayData } from "./holidayCalendar";
import { shiftAt, viewShift } from "./localTime";
import { world } from "./world";

export const data = signal<HolidayData | null>(null);
let asked = false;
/** Fetch the calendar once (about 240 kB, its own chunk). */
export function loadHolidays() {
  if (asked) return;
  asked = true;
  void import("./holidays.json").then((m) => { data.value = m.default as unknown as HolidayData; }, () => { asked = false; });
}

/** The holiday's name in `country` on its local game day `day`, or null. */
export function holidayOn(country: string, day: number): string | null {
  return holidayName(data.value, country, day);
}

/** The country at (lng, lat): its zone's, or null outside every zone. */
export function countryAt(lng: number, lat: number): string | null {
  const z = cityWithin(lng, lat);
  return z ? cityById(z).country ?? null : null;
}

/** The kind of local game day `day` in `country` (T-872). Reads the calendar signal. */
export const kindIn = (country: string | null, day: number) => kindOfIn(data.value, country)(day);

/** A line's country (T-872): its first stop's, as its local time is. */
export function lineCountry(stops: readonly string[]): string | null {
  const s = stops.length ? world.value?.save.stations.find((x) => x.id === stops[0]) : undefined;
  return s ? countryAt(s.lng, s.lat) : null;
}

const regionNames = typeof Intl !== "undefined" && "DisplayNames" in Intl ? new Intl.DisplayNames(["en"], { type: "region" }) : null;
export const countryName = (cc: string): string => regionNames?.of(cc) ?? cc;

/** per station node: its zone's country and local time, kept while the node stays put */
const placeOf = new Map<number, { key: string; country: string | null; shift: number }>();

/** The countries the player has stations in, each with the local times (seconds ahead of the game
 * clock) its stations keep: "country shift" keys. */
export const railCountries = computed<string[]>(() => {
  const w = world.value;
  if (!w) return [];
  const out = new Set<string>();
  for (const n of w.nodes.values()) {
    if (n.platform <= 0) continue;
    const key = `${n.lng} ${n.lat}`;
    let p = placeOf.get(n.id);
    if (!p || p.key !== key) {
      const z = cityWithin(n.lng, n.lat);
      p = { key, country: z ? cityById(z).country ?? null : null, shift: shiftAt(n.lng, n.lat) };
      placeOf.set(n.id, p);
    }
    if (p.country) out.add(`${p.country} ${p.shift}`);
  }
  return [...out].sort();
});

export interface HolidaySpan {
  country: string;
  name: string;
  /** the holiday's day in the local time in view, seconds: [t0, t1) */
  t0: number;
  t1: number;
}

let lastKey = "", last: HolidaySpan[] = [];
/** Holidays in rail countries that meet the local day in view or the next one, in the local time in
 * view. Changes at most once a game minute and keeps its identity while the list is the same. */
export const holidaysNear = computed<HolidaySpan[]>(() => {
  const now = clock.local.value * 60, vs = viewShift.value;
  if (!data.value) return last;
  const out: HolidaySpan[] = [];
  const seen = new Set<string>();
  for (const k of railCountries.value) {
    const [country, s] = k.split(" "), shift = Number(s);
    const here = now - vs + shift, d0 = Math.floor(here / 86400);
    for (let d = d0 - 1; d <= d0 + 2; d++) {
      const name = holidayOn(country, d);
      if (!name) continue;
      const t0 = d * 86400 - shift + vs, t1 = t0 + 86400;
      if (t1 <= now - 6 * 3600 || t0 >= now + 30 * 3600) continue;
      const id = `${country} ${t0}`;
      if (seen.has(id)) continue;
      seen.add(id);
      out.push({ country, name, t0, t1 });
    }
  }
  out.sort((a, b) => a.t0 - b.t0 || a.country.localeCompare(b.country));
  const key = out.map((h) => `${h.country}${h.t0}`).join();
  if (key !== lastKey) { lastKey = key; last = out; }
  return last;
});
