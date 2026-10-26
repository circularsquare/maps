// Top bar: city (or the world, zoomed far out), the mode split (the goal number: the bar and the train
// share), cash, the time and date on one line beside or over the sky and demand band, speed and pause. The clock is
// its own component so a game minute re-renders only it.

import type { ComponentChildren } from "preact";
import { clock, SPEEDS } from "../game/clock";
import { activeCity, cityById, viewedZone } from "../game/cities";
import { demandView, modeRides, modeSplit, viewFinal, viewUpdating } from "../game/demand";
import { RIDES_PER_PERSON, smallPct, WORLD_PEOPLE, WORLD_RIDES, worldTotals, worldView } from "../game/worldTotals";
import { clockTime, count, dayOf, exact, money, moneyCash } from "../game/format";
import { AVERAGE_DAY, awayNear, dateName, dateOf, MONTH_LONG, weekdayOf } from "../game/days";
import { PERIODS, periodAt } from "../game/periods";
import { SkyBanner } from "./SkyBanner";
import { countryName, groupHolidays, holidaysNear, loadHolidays, rushNear, runToday, supersNear, type HolidaySpan, type RushSpan } from "../game/holidays";
import { useEffect } from "preact/hooks";
import { display, setDisplay, shownDay, shownDayOn } from "../game/ui";
import { splitTip } from "../game/splitTip";

const WEEKDAY_LONG = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"];
import { world } from "../game/world";
import { liveMoney, moneyState } from "../game/money";
import { pointsPast24h, pointsPerDay } from "../game/points";
import { Icon } from "./widgets";
import { perfOff } from "../perfFlags";
import { saveSlotsState } from "../workers/clockClient";
import { START_CASH } from "../workers/saveFile";
import { CACHE_VERSION } from "../workers/demandCache";
import { savesOpen } from "./SavesPanel";

const NO_CLOCK = perfOff("clock");

const hourTime = (h: number) => clockTime(Math.round(h * 60) % 1440);

/** The demand band's hover (T-902, Anita; T-1242): "Low demand until 06:00", the end being where
 * the level next changes. */
function periodTip(minute: number): string {
  let p = PERIODS[periodAt(minute * 60)];
  const level = p.demand;
  for (let k = 0; k < PERIODS.length; k++) {
    const next = PERIODS.find((q) => q.from === p.to % 24)!;
    if (next.demand !== level) break;
    p = next;
  }
  return `${level[0].toUpperCase()}${level.slice(1)} demand until ${hourTime(p.to)}`;
}

/** T-862: one line per holiday on the sky, "Today: Thanksgiving Day, United States", "Tomorrow: …",
 * or with the hour it starts where its country keeps another local time. T-1098: one line for a
 * holiday several countries keep that date, "Labor Day, Belgium, China and 4 others", or every
 * country named (`full`, the sky's hover). */
function holidayLines(minute: number, list: HolidaySpan[], full: boolean): string[] {
  const now = minute * 60, today = Math.floor(now / 86400);
  return groupHolidays(list.filter((h) => h.t1 > now && h.t0 < now + 12 * 3600)).map((h) => {
    const names = h.countries.map(countryName);
    const where = full || names.length <= 3 ? names : [...names.slice(0, 2), `${names.length - 2} others`];
    const at = `${h.name}, ${where.length > 1 ? `${where.slice(0, -1).join(", ")} and ${where[where.length - 1]}` : where[0]}`;
    if (h.t0 <= now) return `Today: ${at}`;
    const when = Math.floor(h.t0 / 86400) === today ? "Today" : "Tomorrow";
    return h.t0 % 86400 === 0 ? `${when}: ${at}` : `${when} from ${clockTime(h.t0 / 60)}: ${at}`;
  });
}

/** T-264: one line per superholiday day on the sky that no holiday line names: "Tomorrow: going out
 * for Golden Week, China", "Today: coming back from Golden Week, China", a day away that is no holiday
 * "Today: Golden Week, China". Several countries on one date share a line, as `holidayLines`. */
function rushLines(minute: number, list: RushSpan[], hols: HolidaySpan[], full: boolean): string[] {
  const now = minute * 60, today = Math.floor(now / 86400);
  const by = new Map<string, { r: RushSpan; countries: string[] }>();
  for (const r of list) {
    if (r.t1 <= now || r.t0 >= now + 12 * 3600) continue;
    if (r.part === 2 && hols.some((h) => h.country === r.country && h.day === r.day)) continue;
    const key = `${r.day} ${r.part} ${r.name}`, e = by.get(key);
    if (e) { if (!e.countries.includes(r.country)) e.countries.push(r.country); } else by.set(key, { r, countries: [r.country] });
  }
  return [...by.values()].map(({ r, countries }) => {
    const names = countries.map(countryName).sort();
    const where = full || names.length <= 3 ? names : [...names.slice(0, 2), `${names.length - 2} others`];
    const place = where.length > 1 ? `${where.slice(0, -1).join(", ")} and ${where[where.length - 1]}` : where[0];
    const what = r.part === 1 ? `going out for ${r.name}` : r.part === 3 ? `coming back from ${r.name}` : r.name;
    if (r.t0 <= now) return `Today: ${what}, ${place}`;
    const when = Math.floor(r.t0 / 86400) === today ? "Today" : "Tomorrow";
    return r.t0 % 86400 === 0 ? `${when}: ${what}, ${place}` : `${when} from ${clockTime(r.t0 / 60)}: ${what}, ${place}`;
  });
}

const DAY_SHOWN_ON: Record<string, string> = { average: "the average day", weekday: "a weekday", saturday: "a Saturday", sunday: "a Sunday" };
/** the Day choice before the calendar icon set it to a day away, to go back to */
let dayBefore = "average";

/** "Out Fri Feb 13 and Sat Feb 14", "Away Sun Feb 15 to Sat Feb 21" */
function daysLine(label: string, from: number, to: number): string | null {
  if (to <= from) return null;
  return `${label} ${dateName(from)}${to - from === 2 ? ` and ${dateName(from + 1)}` : to - from > 2 ? ` to ${dateName(to - 1)}` : ""}`;
}

/** T-264: the calendar icon by the clock. A dot while a superholiday goes out within 21 days where the
 * player has built, pink while one is under way; its hover has each run's days, and a click shows
 * demand on a day away (the Day choice's own option) or back to the day shown before. */
function Calendar() {
  const runs = supersNear.value;
  const away = awayNear.value;
  const d = display.value;
  const showing = d.dayShown === "away" && away !== null;
  const under = runs.some((r) => runToday(r) >= r.out);
  const lines: string[] = [];
  for (const r of runs) {
    const n = r.out - runToday(r);
    if (lines.length) lines.push("");
    lines.push(`${r.name} in ${countryName(r.country)}, ${n > 1 ? `in ${n} days` : n === 1 ? "tomorrow" : "now"}`);
    for (const l of [daysLine("Out", r.out, r.away), daysLine("Away", r.away, r.back), daysLine("Back", r.back, r.end)]) if (l) lines.push(l);
  }
  if (!runs.length) lines.push("No long holidays in the next 3 weeks");
  if (away) lines.push("", showing ? `Click to show demand on ${DAY_SHOWN_ON[dayBefore] ?? DAY_SHOWN_ON.average} again` : `Click to show demand on a day of ${away}`);
  const click = () => {
    if (!away) return;
    if (showing) setDisplay("dayShown", (DAY_SHOWN_ON[dayBefore] ? dayBefore : "average") as typeof d.dayShown);
    else { dayBefore = d.dayShown; setDisplay("dayShown", "away"); }
  };
  return (
    <button class={"btn icon cal" + (showing ? " on" : "") + (under ? " under" : "")} data-tip={lines.join("\n")} aria-label="Long holidays" onClick={click}>
      <Icon.calendar />
      {runs.length > 0 && <i class="cal-dot" />}
    </button>
  );
}

/** T-861, T-1136: the time and date on one line, beside or over the sky banner (ui/SkyBanner.tsx) and its demand band. */
function Clock() {
  // ?perfOff=clock (T-045, measurement only): read without subscribing, so it never re-renders.
  // T-309: the local time and day where the player is looking
  const minute = NO_CLOCK ? clock.local.peek() : clock.local.value;
  // T-172: a loaded game's clock waits for its first demand estimate; building does not
  const held = clock.held.value;
  const day = dayOf(minute), { month, date } = dateOf(day);
  useEffect(loadHolidays, []);
  const near = holidaysNear.value, rush = rushNear.value;
  const hols = [...holidayLines(minute, near, true), ...rushLines(minute, rush, near, true)];
  const todayHols = holidayLines(minute, near, false).filter((l) => l.startsWith("Today: "));
  const dateTip = `${WEEKDAY_LONG[weekdayOf(day)]}, ${MONTH_LONG[month]} ${date}, day ${day}`;
  return (
    <div class="clock">
      <Calendar />
      <div class="clock-box">
        <div class="clock-text" data-tip={held ? "The clock starts when demand has loaded" : [dateTip, ...todayHols].join("\n")}>
          <b class={"time lg" + (held ? " held" : "")}>{clockTime(minute)}</b>
          <span class="date">{todayHols.length > 0 && <i class="holiday-dot" />}{dateName(day)}</span>
        </div>
        <SkyBanner skyTip={hols.length ? hols.join("\n") : undefined} bandTip={periodTip(minute)} />
      </div>
    </div>
  );
}

/** T-284: its own component, re-rendered as fares land (several times a wall second while the clock
 * runs) without the rest of the top bar. */
function Cash() {
  const live = liveMoney.value, m = moneyState.value;
  const today = live.today;
  const tip = today ? `Today: fares ${money(today.fares * 1e6)}, running trains ${money(-today.running * 1e6)}, upkeep ${money(-(today.upkeep ?? 0) * 1e6)}` : undefined;
  // T-728: a sandbox shows no cash; the hover has the money since the start
  if (m?.mode === "sandbox") return <div class="cash sandbox" data-tip={["Infinite money", `Since day 1: ${money((live.cash - START_CASH) * 1e6)}`, ...(tip ? [tip] : [])].join("\n")}>Sandbox</div>;
  // T-1136: right-aligned in a box of fixed width, so it ends the usual gap before the clock
  const shown = m ? moneyCash(live.cash * 1e6) : "";
  return <div class={"cash num" + (shown.startsWith("−") ? " money-neg" : "")} data-tip={m ? tip : undefined}>{shown}</div>;
}

/** The city's name, the whole of it and its points (T-684, T-694: the past 24 game hours') on hover. */
function CityName({ id, name, tip }: { id: string | null; name: string; tip?: string }) {
  const past = id ? pointsPast24h.value.get(id) : undefined;
  const rate = id ? pointsPerDay.value.get(id)?.[shownDay.value] : undefined;
  const lines = [name];
  if (id && (past || rate)) lines.push(`${exact(past ?? 0)} points in the past 24 hours`, `${exact(rate ?? 0)} points ${shownDayOn.value} with the riders now`);
  return <div class="city" data-tip={tip ?? (id ? lines.join("\n") : undefined)}>{name}</div>;
}

function Speed() {
  const running = clock.running.value;
  const speed = clock.speed.value;
  const btn = (i: number, icon: ComponentChildren, title: string) => (
    <button class={"btn icon" + (running && speed === i ? " on" : "")} data-tip={title} aria-label={title} onClick={() => clock.setSpeed(i)}>
      {icon}
    </button>
  );
  return (
    <div class="group speed">
      <button class={"btn icon" + (!running ? " on" : "")} data-tip="Pause (Space)" aria-label="Pause" onClick={() => clock.setRunning(!running)}>
        <Icon.pause />
      </button>
      {btn(0, <Icon.play1 />, "Normal speed")}
      {btn(1, <Icon.play2 />, `${SPEEDS[1] / SPEEDS[0]} times as fast`)}
      {btn(2, <Icon.play3 />, `${SPEEDS[2] / SPEEDS[0]} times as fast`)}
    </div>
  );
}

/** T-1301: the world's bar hover: the train share of the world's fixed rides a day, how that is
 * reached, then the cities' own split. */
function worldShareTip(t: ReturnType<typeof worldTotals>, where: string, note: string): string {
  const world = `${WORLD_RIDES / 1e9} billion rides a day in the world (${WORLD_PEOPLE / 1e9} billion people, about ${RIDES_PER_PERSON} rides each)`;
  if (!t.split || !t.rides) return `Train share of the ${world}${note}`;
  const cities = `${t.cities} ${t.cities === 1 ? "city" : "cities"}`;
  return [
    `Train ${smallPct(t.split.train)} of the ${world}${note}`,
    `${count(t.rides.train)} train rides a day in ${cities}`,
    splitTip(where.replace(" in the world", ` in the ${cities} with stations`), t.citySplit, t.rides, ""),
  ].join("\n");
}

function SaveNotice() {
  const state = saveSlotsState.value;
  if (state.writable) return null;
  return <div class="save-notice" role="status">
    <span>{state.reason ?? "Autosave is off for this game."}</span>
    <button class="btn" onClick={() => { savesOpen.value = true; }}>Saves</button>
  </div>;
}

export function TopBar() {
  const w = world.value;
  if (!w) return <header id="topbar" />;
  // T-1110: zoomed far out the bar names the world and sums every city, as the Cities tab's first row does
  const far = worldView.value;
  const total = far ? worldTotals() : null;
  const known = total ? total.split : modeSplit.value; // T-026: from the demand workers
  const s = known ?? { train: 0, walk: 0, drive: 0 };
  // T-318: no name over dead space (the focus stays on the last zone, which is not where the view is)
  const city = !far && viewedZone.value ? cityById(activeCity.value) : null;
  // T-172: no figures until the city has opened (the game is playable while it loads)
  // T-1301: the world's share is of every ride in the world, so small shares get two figures
  const pct = (v: number) => (!known ? "–" : far ? smallPct(v) : v.toFixed(1) + "%");
  // T-342: the shares are of rides (trip legs) a day; the train's count beside its share, all in the hover
  const rides = total ? total.rides : modeRides.value;
  const where = [far ? " in the world" : city ? " in " + city.name : "", (shownDay.value === AVERAGE_DAY ? "" : " " + shownDayOn.value)].join("");
  const note = demandView.value && !viewFinal.value ? " (an early estimate)" : demandView.value?.version === CACHE_VERSION ? " (from the last time this game was open)" : viewUpdating.value ? " (before your last change)" : "";
  const worldTip = total ? `World\n${total.cities} ${total.cities === 1 ? "city" : "cities"}, ${total.stations} ${total.stations === 1 ? "station" : "stations"}` : undefined;
  return (
    <><header id="topbar">
      {/* T-1108: the name takes what it needs up to a cap, on two lines at most, the whole name on hover */}
      {far ? <CityName id={null} name="World" tip={worldTip} /> : <CityName id={city?.id ?? null} name={city?.name ?? ""} />}
      <div class="split">
        <div data-tip={total ? worldShareTip(total, where, note) : splitTip(where, known, rides, note)} class={"split-row" + (viewUpdating.value ? " updating" : "")}>
          <span class="bar-box">
            <span class="bar">
              <i class="seg seg-train" style={{ width: s.train + "%" }} />
              <i class="seg seg-walk" style={{ width: s.walk + "%" }} />
              <i class="seg seg-drive" style={{ width: s.drive + "%" }} />
            </span>
          </span>
          <b class="lg pct">{pct(s.train)}</b>
        </div>
      </div>
      <Cash />
      <Clock />
      <Speed />
    </header><SaveNotice /></>
  );
}
