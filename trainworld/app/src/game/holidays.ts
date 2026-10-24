// Public holidays (T-862, docs/world.md "Calendar"): each country's national public holidays from
// holidays.json (pipeline/holidays_json.py), fetched after startup as its own chunk. A game year
// takes the holidays of the real non-leap year whose January 1 falls on the same weekday, so a
// holiday tied to a weekday keeps it. A holiday is a whole day of its country's local time.
//
// The top bar's sky marks them for the countries the player has built in. A holiday runs as a
// weekend day for the lines in its country (T-872), and a country's own weekend decides its other
// days (T-262): the clock worker sends the track model each line's days that differ from the day
// number's and settles money by them; here, the view's Today and the line inspector.

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

/** The countries the player has built track or stations in (blueprint left out, T-1098), each with
 * the local times (seconds ahead of the game clock) its built nodes keep: "country shift" keys. */
export const railCountries = computed<string[]>(() => {
  const w = world.value;
  if (!w) return [];
  const out = new Set<string>();
  for (const n of w.nodes.values()) {
    if (n.builtPorts <= 0) continue;
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
  /** the country's own local game day */
  day: number;
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
      const names = holidayOn(country, d);
      if (!names) continue;
      const t0 = d * 86400 - shift + vs, t1 = t0 + 86400;
      if (t1 <= now - 6 * 3600 || t0 >= now + 30 * 3600) continue;
      // two holidays on one day come as "Holy Saturday; Labor Day"
      for (const name of names.split("; ")) {
        const id = `${country} ${t0} ${name}`;
        if (seen.has(id)) continue;
        seen.add(id);
        out.push({ country, name, day: d, t0, t1 });
      }
    }
  }
  out.sort((a, b) => a.t0 - b.t0 || a.country.localeCompare(b.country));
  const key = out.map((h) => `${h.country}${h.t0}${h.name}`).join();
  if (key !== lastKey) { lastKey = key; last = out; }
  return last;
});

/** Names the calendar gives one holiday in different countries (T-1098), and the name it is shown by
 * when several countries keep it on one date. */
const SAME: [RegExp, string][] = [
  [/^((international|national) )?(labou?r|workers'|worker's)( day)?$|^may day$|^labou?r day and international workers' solidarity day$|^international workers' solidarity day$|^holiday of spring and labou?r$|^festa dei lavoratori$/i, "Labor Day"],
  [/^(catholic )?christmas( day)?( \(by new style\))?$|^natale$/i, "Christmas Day"],
  [/^boxing day$|^saint stephen's day$|^santo stefano$|^second day of christmas$|^christmas second day$|^day (after|following) christmas$/i, "Boxing Day"],
  [/^new year's day$|^international new year day$|^capodanno$/i, "New Year's Day"],
];
const sameAs = (name: string) => SAME.find(([re]) => re.test(name))?.[1] ?? name;

export interface HolidayGroup {
  name: string;
  /** country codes, by their names */
  countries: string[];
  /** from the first country's start to the last one's end, in the local time in view */
  t0: number;
  t1: number;
}

/** One entry per holiday kept on the same date in several countries (T-1098): the same name, or
 * names `SAME` counts as one, on the same local date in each country. A holiday known by one name
 * keeps it; one merged under several is shown by `SAME`'s. In order of start. */
export function groupHolidays(list: readonly HolidaySpan[]): HolidayGroup[] {
  const by = new Map<string, { names: Set<string>; g: HolidayGroup }>();
  for (const h of list) {
    const kind = sameAs(h.name), key = `${h.day} ${kind.toLowerCase()}`;
    const e = by.get(key);
    if (!e) { by.set(key, { names: new Set([h.name]), g: { name: kind, countries: [h.country], t0: h.t0, t1: h.t1 } }); continue; }
    e.names.add(h.name);
    if (!e.g.countries.includes(h.country)) e.g.countries.push(h.country);
    e.g.t0 = Math.min(e.g.t0, h.t0);
    e.g.t1 = Math.max(e.g.t1, h.t1);
  }
  return [...by.values()].map(({ names, g }) => {
    g.countries.sort((a, b) => countryName(a).localeCompare(countryName(b)));
    return names.size === 1 ? { ...g, name: [...names][0] } : g;
  }).sort((a, b) => a.t0 - b.t0 || a.name.localeCompare(b.name));
}
