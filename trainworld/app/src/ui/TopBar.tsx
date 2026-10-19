// Top bar: city, the mode split (the goal number, train first, in the fixed mode colours), cash,
// day and time (the demand level is in the clock's hover), speed and pause. The clock is its own
// component so a game minute re-renders only it.

import type { ComponentChildren } from "preact";
import { clock, SPEEDS } from "../game/clock";
import { activeCity, cityById, viewedZone } from "../game/cities";
import { demandView, modeRides, modeSplit, viewFinal, viewUpdating } from "../game/demand";
import { clockTime, count, dayOf, money, moneyCash } from "../game/format";
import { weekdayName, weekdayOf } from "../game/days";
import { shownDay } from "../game/ui";
import { splitTip } from "../game/splitTip";

const WEEKDAY_LONG = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"];
import { world } from "../game/world";
import { liveMoney, moneyState } from "../game/money";
import { pointsExact, pointsPast24h, pointsPerDay } from "../game/points";
import { Icon } from "./widgets";
import { perfOff } from "../perfFlags";
import { saveSlotsState } from "../workers/clockClient";
import { CACHE_VERSION } from "../workers/demandCache";
import { savesOpen } from "./SavesPanel";

const NO_CLOCK = perfOff("clock");

const DEMAND_NAME = { high: "High demand", medium: "Medium demand", low: "Low demand" };

function Clock() {
  // ?perfOff=clock (T-045, measurement only): read without subscribing, so it never re-renders.
  // T-309: the local time and day where the player is looking
  const minute = NO_CLOCK ? clock.local.peek() : clock.local.value;
  // T-172: a loaded game's clock waits for its first demand estimate; building does not
  const held = clock.held.value;
  return (
    <div class="clock" data-tip={held ? "The clock starts when demand has loaded" : `${WEEKDAY_LONG[weekdayOf(dayOf(minute))]}, day ${dayOf(minute)} · ${DEMAND_NAME[NO_CLOCK ? clock.demand.peek() : clock.demand.value]}`}>
      <span class="day sm">{weekdayName(dayOf(minute))} {dayOf(minute)}</span>
      <b class={"time lg" + (held ? " held" : "")}>{clockTime(minute)}</b>
    </div>
  );
}

/** T-284: its own component, re-rendered as fares land (several times a wall second while the clock
 * runs) without the rest of the top bar. */
function Cash() {
  const live = liveMoney.value, m = moneyState.value;
  const today = live.today;
  const tip = today ? `Today: fares ${money(today.fares * 1e6)}, running trains ${money(-today.running * 1e6)}` : undefined;
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
            <span class="sm rides">{rides ? count(rides.train) + " rides" : ""}</span>
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
