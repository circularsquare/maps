// Top bar: city (or the world, zoomed far out), the mode split (the goal number: the bar and the
// train share), cash, the sky and demand band with the time and date on one line right of it (T-1853),
// speed and pause. The clock is its own component so a game minute re-renders only it.

import type { ComponentChildren } from "preact";
import { clock, DEMAND_CAP, SLIDER_STOPS, SPEEDS } from "../game/clock";
import { activeCity, cityById, viewedZone } from "../game/cities";
import { demandView, modeRides, modeSplit, viewRecomputing, viewUpdating } from "../game/demand";
import { RIDES_PER_PERSON, smallPct, WORLD_PEOPLE, WORLD_RIDES, worldTotals, worldView } from "../game/worldTotals";
import { clockTime, dayOf, exact, money, moneyCash } from "../game/format";
import { AVERAGE_DAY, awayNear, dateName, dateOf, dateRange, MONTH_LONG, rushRun, weekdayOf, whenAway } from "../game/days";
import { PERIODS, periodAt } from "../game/periods";
import { SkyBanner } from "./SkyBanner";
import { countryName, groupHolidays, holidaysNear, loadHolidays, rushNear, runToday, superDays, supersNear, type HolidaySpan, type RushSpan } from "../game/holidays";
import { useEffect, useRef, useState } from "preact/hooks";
import { display, setDisplay, shownDay, shownDayOn } from "../game/ui";
import { splitTip } from "../game/splitTip";

const WEEKDAY_LONG = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"];
import { world } from "../game/world";
import { liveMoney, moneyState } from "../game/money";
import { pointsPast24h, pointsPerDay, pointsShort } from "../game/points";
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

/** T-264, T-1697: one line per superholiday day on the sky that no holiday line names, with the dates
 * of its part: "Tomorrow: Golden Week, Apr 29-30, China", "Today: Golden Week, May 1-5, China".
 * Several countries on one date share a line, as `holidayLines`. */
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
    const what = `${r.name}, ${r.dates}`;
    if (r.t0 <= now) return `Today: ${what}, ${place}`;
    const when = Math.floor(r.t0 / 86400) === today ? "Today" : "Tomorrow";
    return r.t0 % 86400 === 0 ? `${when}: ${what}, ${place}` : `${when} from ${clockTime(r.t0 / 60)}: ${what}, ${place}`;
  });
}

const DAY_SHOWN_ON: Record<string, string> = { average: "the average day", weekday: "a weekday", saturday: "a Saturday", sunday: "a Sunday" };
/** the Day choice before the calendar icon set it to a day away, to go back to */
let dayBefore = "average";

/** How many days ahead of a superholiday's going-out day the pink dot is a button (T-1476). */
const CAL_AHEAD = 2;

/** T-264, T-1476, T-1373, T-1865: the pink dot by the date. It shows while a holiday is today (`today`),
 * and is a button while a superholiday where the player has built is under way or goes out within
 * `CAL_AHEAD` days (the calendar icon it replaced). Its hover has each run's days within 21 days, and a
 * click shows demand on its next going-out day (the Day choice's own option; once those are past, a
 * coming-back day, then a day away) or back to the day shown before. */
function HolidayDot({ today }: { today: boolean }) {
  const runs = supersNear.value;
  void superDays.value; // a day turning, for `runToday` below
  const away = awayNear.value, rush = rushRun.value;
  const d = display.value;
  if (!runs.some((r) => r.out - runToday(r) <= CAL_AHEAD)) return today ? <i class="holiday-dot" /> : null;
  // the day a click shows: the first run's next rush day still to come
  const first = runs.find((r) => runToday(r) < r.end);
  const t = first ? runToday(first) : 0;
  const target: { key: "out" | "back" | "away"; what: string } | null = !first ? null
    : rush && first.out < first.away && t < first.away ? { key: "out", what: `a day of ${first.name}, ${dateRange(first.out, first.away)}` }
    : rush && first.back < first.end ? { key: "back", what: `a day of ${first.name}, ${dateRange(first.back, first.end)}` }
    : away ? { key: "away", what: `a day of ${whenAway.value ?? away}` } : null;
  const showing = (d.dayShown === "out" || d.dayShown === "back") ? rush !== null : d.dayShown === "away" && away !== null;
  const lines: string[] = [];
  for (const r of runs) {
    const n = r.out - runToday(r);
    if (lines.length) lines.push("");
    const dates = r.back > r.away ? dateRange(r.away, r.back) : dateRange(r.out, r.end);
    lines.push(`${r.name}, ${dates}, ${countryName(r.country)}: ${n > 1 ? `in ${n} days` : n === 1 ? "tomorrow" : "now"}`);
  }
  if (target || showing) lines.push("", showing ? `Click to show demand on ${DAY_SHOWN_ON[dayBefore] ?? DAY_SHOWN_ON.average} again` : `Click to show demand on ${target!.what}`);
  const click = () => {
    if (showing) setDisplay("dayShown", (DAY_SHOWN_ON[dayBefore] ? dayBefore : "average") as typeof d.dayShown);
    else if (target) { dayBefore = d.dayShown; setDisplay("dayShown", target.key); }
  };
  return <button class={"holiday-dot btn-dot" + (showing ? " on" : "")} data-tip={lines.join("\n")} aria-label="Long holidays" onClick={click} />;
}

/** T-861, T-1853: the time and date on one line right of the sky banner (ui/SkyBanner.tsx) and its demand
 * band (over it in a map column under 360 px). */
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
      <div class="clock-box">
        <SkyBanner skyTip={hols.length ? hols.join("\n") : undefined} bandTip={periodTip(minute)} />
        <div class="clock-text" data-tip={held ? "The clock starts when demand has loaded" : [dateTip, ...todayHols].join("\n")}>
          <b class={"time lg" + (held ? " held" : "")}>{clockTime(minute)}</b>
          <span class="date"><HolidayDot today={todayHols.length > 0} />{dateName(day)}</span>
        </div>
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
  // T-1865: after the split, in a box of fixed width (style.css .cash), so nothing beside it moves
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

const times = (x: number) => `${x}x`;

/** T-1664: double-click the fastest button for a slider of speeds, 0.5x to 100x (the speed then shows on
 * the button); T-1665: while demand updates the clock is held at DEMAND_CAP, the chosen speed dimmed. */
function Speed() {
  const running = clock.running.value;
  const speed = clock.speed.value;
  const custom = clock.custom.value;
  const capped = clock.capped.value;
  const [open, setOpen] = useState(false);
  const box = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!open) return;
    const away = (e: Event) => { if (!box.current?.contains(e.target as Node)) setOpen(false); };
    const esc = (e: KeyboardEvent) => { if (e.key === "Escape") setOpen(false); };
    document.addEventListener("pointerdown", away);
    document.addEventListener("keydown", esc);
    return () => { document.removeEventListener("pointerdown", away); document.removeEventListener("keydown", esc); };
  }, [open]);
  const chosen = clock.chosenRate.value / SPEEDS[0];
  const held = capped && chosen > DEMAND_CAP;
  const heldTip = (title: string) => (held ? `${times(chosen)} (capped at ${times(DEMAND_CAP)} while updating demand)` : title);
  const btn = (i: number, icon: ComponentChildren, title: string, extra: Record<string, unknown> = {}) => {
    const on = running && (custom !== null ? 2 : speed) === i;
    return (
      <button class={"btn icon" + (on ? " on" : "") + (on && held ? " held" : "") + (i === 2 && custom !== null ? " rate" : "")} data-tip={on ? heldTip(title) : title} aria-label={title} onClick={() => (i === 2 && custom !== null ? clock.setMultiple(custom) : clock.setSpeed(i))} {...extra}>
        {i === 2 && custom !== null ? times(custom) : icon}
      </button>
    );
  };
  const stop = Math.max(0, SLIDER_STOPS.findIndex((s) => s >= chosen));
  return (
    <div class="group speed" ref={box}>
      <button class={"btn icon" + (!running ? " on" : "")} data-tip="Pause (Space)" aria-label="Pause" onClick={() => clock.setRunning(!running)}>
        <Icon.pause />
      </button>
      {btn(0, <Icon.play1 />, "Normal speed")}
      {btn(1, <Icon.play2 />, `${times(SPEEDS[1] / SPEEDS[0])} speed`)}
      {btn(2, <Icon.play3 />, `${times(custom ?? SPEEDS[2] / SPEEDS[0])} speed\nDouble-click to pick a speed`, { onDblClick: () => setOpen((o) => !o) })}
      {open && (
        <div class="slider speed-pop">
          <input type="range" min={0} max={SLIDER_STOPS.length - 1} step={1} value={stop} aria-label="Clock speed"
            onInput={(e) => clock.setMultiple(SLIDER_STOPS[Number((e.target as HTMLInputElement).value)])} />
          <span class="num">{times(chosen)}</span>
        </div>
      )}
    </div>
  );
}

/** The world's bar hover: the train trips/day in the cities (T-1440), the share of the world's
 * fixed trips a day and how that is reached (T-1301), then the cities' own split. */
function worldShareTip(t: ReturnType<typeof worldTotals>, where: string, note: string): string {
  const world = `${WORLD_RIDES / 1e9} billion trips/day in the world (${WORLD_PEOPLE / 1e9} billion people, about ${RIDES_PER_PERSON} trips each)`;
  if (!t.split || !t.rides) return `Train share of the ${world}${note}`;
  const cities = `${t.cities} ${t.cities === 1 ? "city" : "cities"}`;
  return [
    `${exact(t.rides.train)} train trips/day in ${cities}${note}`,
    `Trains carry ${smallPct(t.split.train)} of ${world}`,
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
  // T-1440: zoomed out, the train trips/day in place of the world's share
  const pct = (v: number) => (!known ? "–" : far ? `${pointsShort(total?.rides?.train ?? 0)} trips/day` : v.toFixed(1) + "%");
  // T-342: the shares are of rides (trip legs) a day; the train's count beside its share, all in the hover
  const rides = total ? total.rides : modeRides.value;
  const where = [far ? " in the world" : city ? " in " + city.name : "", (shownDay.value === AVERAGE_DAY ? "" : " " + shownDayOn.value)].join("");
  // T-1223: the zone in view shown rough while it is worked out in detail
  // T-1851: one note for every case where the trips are being recomputed
  const note = demandView.value && (viewRecomputing.value || demandView.value.version === CACHE_VERSION) ? " (trips being recomputed…)" : "";
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
          <b class={"lg pct" + (far ? " trips" : "")}>{pct(s.train)}</b>
        </div>
      </div>
      <Cash />
      <Clock />
      <Speed />
    </header><SaveNotice /></>
  );
}
