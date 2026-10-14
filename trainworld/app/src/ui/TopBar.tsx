// Top bar: city, the mode split (the goal number, train first, in the fixed mode colours), cash,
// day and time, the demand level, speed and pause. The clock is its own component so a game minute
// re-renders only it.

import type { ComponentChildren } from "preact";
import { clock, SPEEDS } from "../game/clock";
import { activeCity, cityById, viewedZone } from "../game/cities";
import { demandStep, demandUpdating, demandView, modeRides, modeSplit } from "../game/demand";
import { clockTime, count, dayOf, money, moneyExact } from "../game/format";
import { weekdayName, weekdayOf } from "../game/days";
import { shownDay } from "../game/ui";

const WEEKDAY_LONG = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"];
import { cityNames, demandIdle, loadFailure, loadProgress } from "../game/loading";
import { world } from "../game/world";
import { liveMoney, moneyState } from "../game/money";
import { Icon } from "./widgets";
import { LoadBar, loadingCardShown } from "./LoadingScreen";
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
      <span class="day muted">{weekdayName(dayOf(minute))} {dayOf(minute)}</span>
      <b class={"time big" + (held ? " muted" : "")}>{clockTime(minute)}</b>
      <span class="demand muted">{DEMAND_NAME[NO_CLOCK ? clock.demand.peek() : clock.demand.value]}</span>
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
  const exact = m ? moneyExact(live.cash * 1e6) : "";
  const neg = exact.startsWith("−");
  return <div class="cash num" data-tip={m ? tip : undefined}>
    <span class="cash-sign">{neg ? "−" : ""}</span>
    <span class="cash-cur">{m ? "$" : ""}</span>
    <span class="cash-digits">{exact.replace(/^−?\$/, "")}</span>
  </div>;
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
  const step = demandStep.value;
  // T-149, T-172: a city loading (at startup, after a pan, or construction reaching it) says how
  // far it has got
  const load = loadProgress.value;
  const busy = loadFailure.value ? "Demand could not load"
    // T-180: while the clock waits, the card in the middle of the map says it instead
    : loadingCardShown() ? null
    // T-212: "Loading data", not the cities' names (those are in the hover)
    // T-243: a thin bar after it, not a percentage
    : load ? "Loading data"
    : !demandView.value ? (demandIdle.value ? null : "Loading data…") // T-318: nothing to load over dead space
    // T-232: figures before a solve's final round are shown only where nothing final is, and say so
    : !demandView.value.final ? `Estimate, refining ${Math.min(step.done + 1, step.of)}/${step.of}…`
    : demandUpdating.value ? `Updating demand ${Math.min(step.done + 1, step.of)}/${step.of}…` : null;
  const busyTip = loadFailure.value ?? (load && !loadingCardShown() ? `Loading ${cityNames(load.cities)}, ${Math.floor(load.fraction * 100)}%. ${load.step}`
    : !demandView.value || !busy ? busy
    : !demandView.value.final ? "These figures are an early estimate. They settle when the forecast is finished."
    // T-328: the figures kept from the last time this game was open, while it is worked out again
    : demandView.value.version === CACHE_VERSION ? "These figures are from the last time this game was open. They are being worked out again."
    : "These figures are for the network before your last change. They change once the new forecast is finished.");
  return (
    <><header id="topbar">
      {/* T-341: every part has a fixed width; a long name is cut short, the whole name on hover */}
      <div class="city" data-tip={city?.name}>{city?.name}</div>
      <div class="split">
        <div class="split-row" data-tip={`Rides a day ${city ? "in " + city.name + " " : ""}${shownDay.value ? "at weekends" : "on weekdays"}: ${rides ? `Train ${pct(s.train)} (${count(rides.train)}), Walking ${pct(s.walk)} (${count(rides.walk)}), Driving ${pct(s.drive)} (${count(rides.drive)}), all ${count(rides.all)} rides` : `Train ${pct(s.train)}, Walking ${pct(s.walk)}, Driving ${pct(s.drive)}`}${demandView.value && !demandView.value.final ? " (an early estimate)" : demandView.value?.version === CACHE_VERSION ? " (from the last time this game was open)" : demandUpdating.value ? " (before your last change)" : ""}`} style={{ opacity: demandUpdating.value ? 0.6 : 1 }}>
          <span class="bar">
            <i class="seg seg-train" style={{ width: s.train + "%" }} />
            <i class="seg seg-walk" style={{ width: s.walk + "%" }} />
            <i class="seg seg-drive" style={{ width: s.drive + "%" }} />
          </span>
          <span class="split-labels">
          <span class="split-train">
            <i class="sw sw-train" />
            Train <b class="big pct">{pct(s.train)}</b>
            <span class="muted rides">{rides ? count(rides.train) + " rides" : ""}</span>
          </span>
          <span class="split-other">
            <i class="sw sw-walk" />
            Walking <span class="pct">{pct(s.walk)}</span>
          </span>
          <span class="split-other">
            <i class="sw sw-drive" />
            Driving <span class="pct">{pct(s.drive)}</span>
          </span>
          </span>
        </div>
        {/* T-341: its own line under the bar, reserved whether or not there is a status, so nothing moves.
            T-180: cut short with an ellipsis in a narrow column; the whole text on hover */}
        <div class="split-status">
          {busy && <span class="demand-busy muted" role="status" data-tip={busyTip ?? undefined}>{busy}</span>}
          {busy && load && !loadFailure.value && <LoadBar fraction={load.fraction} step={load.step} thin />}
        </div>
      </div>
      <Cash />
      <Clock />
      <Speed />
    </header><SaveNotice /></>
  );
}
