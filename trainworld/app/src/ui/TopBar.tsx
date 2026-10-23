// Top bar: city, the mode split (the goal number, train first, in the fixed mode colours), cash,
// time and date over the sky and demand band, speed and pause. The clock is its own
// component so a game minute re-renders only it.

import type { ComponentChildren } from "preact";
import { clock, SPEEDS } from "../game/clock";
import { activeCity, cityById, viewedZone } from "../game/cities";
import { demandView, modeRides, modeSplit, viewFinal, viewUpdating } from "../game/demand";
import { clockTime, count, dayOf, money, moneyCash } from "../game/format";
import { dateName, dateOf, MONTH_LONG, weekdayOf } from "../game/days";
import { PERIODS, periodAt } from "../game/periods";
import { SkyBanner } from "./SkyBanner";
import { countryName, groupHolidays, holidaysNear, loadHolidays, type HolidaySpan } from "../game/holidays";
import { useEffect } from "preact/hooks";
import { shownDay } from "../game/ui";
import { splitTip } from "../game/splitTip";

const WEEKDAY_LONG = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"];
import { world } from "../game/world";
import { liveMoney, moneyState } from "../game/money";
import { pointsExact, pointsPast24h, pointsPerDay } from "../game/points";
import { Icon } from "./widgets";
import { perfOff } from "../perfFlags";
import { saveSlotsState } from "../workers/clockClient";
import { START_CASH } from "../workers/saveFile";
import { CACHE_VERSION } from "../workers/demandCache";
import { savesOpen } from "./SavesPanel";

const NO_CLOCK = perfOff("clock");

const hourTime = (h: number) => clockTime(Math.round(h * 60) % 1440);

/** The demand band's hover (T-902, Anita): "Currently low demand until 06:00", the end being where
 * the level next changes. */
function periodTip(minute: number): string {
  let p = PERIODS[periodAt(minute * 60)];
  const level = p.demand;
  for (let k = 0; k < PERIODS.length; k++) {
    const next = PERIODS.find((q) => q.from === p.to % 24)!;
    if (next.demand !== level) break;
    p = next;
  }
  return `Currently ${level} demand until ${hourTime(p.to)}`;
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

/** T-861: the time and date over the sky banner (ui/SkyBanner.tsx) and its demand band. */
function Clock() {
  // ?perfOff=clock (T-045, measurement only): read without subscribing, so it never re-renders.
  // T-309: the local time and day where the player is looking
  const minute = NO_CLOCK ? clock.local.peek() : clock.local.value;
  // T-172: a loaded game's clock waits for its first demand estimate; building does not
  const held = clock.held.value;
  const day = dayOf(minute), { month, date } = dateOf(day);
  useEffect(loadHolidays, []);
  const near = holidaysNear.value;
  const hols = holidayLines(minute, near, true);
  const todayHols = holidayLines(minute, near, false).filter((l) => l.startsWith("Today: "));
  const dateTip = `${WEEKDAY_LONG[weekdayOf(day)]}, ${MONTH_LONG[month]} ${date}, day ${day}`;
  return (
    <div class="clock">
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
  if (m?.mode === "sandbox") return <div class="cash sandbox" data-tip={["Sandbox: nothing is refused for money", `Since day 1: ${money((live.cash - START_CASH) * 1e6)}`, ...(tip ? [tip] : [])].join("\n")}>Sandbox</div>;
  // T-305: a box of fixed width, the minus then the $ at fixed places and the digits left-aligned after the $
  const exact = m ? moneyCash(live.cash * 1e6) : "";
  const neg = exact.startsWith("−");
  return <div class={"cash num" + (neg ? " money-neg" : "")} data-tip={m ? tip : undefined}>
    <span class="cash-sign">{neg ? "−" : ""}</span>
    <span class="cash-cur">{m ? "$" : ""}</span>
    <span class="cash-digits">{exact.replace(/^−?\$/, "")}</span>
  </div>;
}

/** The city's name, the whole of it and its points (T-684, T-694: the past 24 game hours') on hover. */
function CityName({ id, name }: { id: string | null; name: string }) {
  const past = id ? pointsPast24h.value.get(id) : undefined;
  const rate = id ? pointsPerDay.value.get(id)?.[shownDay.value] : undefined;
  const lines = [name];
  if (id && (past || rate)) lines.push(`${pointsExact(past ?? 0)} points in the past 24 hours`, `${pointsExact(rate ?? 0)} points ${["on a weekday", "on a weekend day", "a day"][shownDay.value]} with the riders now`);
  return <div class="city" data-tip={id ? lines.join("\n") : undefined}>{name}</div>;
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
  const known = modeSplit.value; // T-026: from the demand workers
  const s = known ?? { train: 0, walk: 0, drive: 0 };
  // T-318: no name over dead space (the focus stays on the last zone, which is not where the view is)
  const city = viewedZone.value ? cityById(activeCity.value) : null;
  // T-172: no figures until the city has opened (the game is playable while it loads)
  const pct = (v: number) => (known ? v.toFixed(1) + "%" : "–");
  // T-342: the shares are of rides (trip legs) a day; the train's count beside its share, all in the hover
  const rides = modeRides.value;
  const where = [city ? " in " + city.name : "", [" on a weekday", " on a weekend day", ""][shownDay.value]].join("");
  const note = demandView.value && !viewFinal.value ? " (an early estimate)" : demandView.value?.version === CACHE_VERSION ? " (from the last time this game was open)" : viewUpdating.value ? " (before your last change)" : "";
  // T-406: walking and driving are icons; each has its own hover with the share and its rides
  const modeTip = (name: string, share: number, n: number | undefined) =>
    `${name}: ${pct(share)} of rides per day${where}${rides && n !== undefined ? ` (${count(n)} of ${count(rides.all)})` : ""}${note}`;
  return (
    <><header id="topbar">
      {/* T-341: every part has a fixed width; a long name is cut short, the whole name on hover */}
      <CityName id={city?.id ?? null} name={city?.name ?? ""} />
      <div class="split">
        <div data-tip={splitTip(where, known, rides, note)} class={"split-row" + (viewUpdating.value ? " updating" : "")}>
          <span class="bar-box">
            <span class="bar">
              <i class="seg seg-train" style={{ width: s.train + "%" }} />
              <i class="seg seg-walk" style={{ width: s.walk + "%" }} />
              <i class="seg seg-drive" style={{ width: s.drive + "%" }} />
            </span>
          </span>
          <span class="split-train">
            <i class="sw sw-train" />
            <b class="lg pct">{pct(s.train)}</b>
          </span>
          <span class="split-other sm" data-tip={modeTip("Walking", s.walk, rides?.walk)}>
            <span class="mode-icon" style={{ color: "var(--mode-walk)" }}><Icon.walk /></span>
            <span class="pct">{pct(s.walk)}</span>
          </span>
          <span class="split-other sm" data-tip={modeTip("Driving", s.drive, rides?.drive)}>
            <span class="mode-icon" style={{ color: "var(--mode-drive)" }}><Icon.ride /></span>
            <span class="pct">{pct(s.drive)}</span>
          </span>
        </div>
      </div>
      <Cash />
      <Clock />
      <Speed />
    </header><SaveNotice /></>
  );
}
