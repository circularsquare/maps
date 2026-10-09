// Top bar: city, the mode split (the goal number, train first, in the fixed mode colours), cash,
// day and time, the demand level, speed and pause. The clock is its own component so a game minute
// re-renders only it.

import type { ComponentChildren } from "preact";
import { clock, SPEEDS } from "../game/clock";
import { demandUpdating, modeSplit } from "../game/demand";
import { clockTime, dayOf, money } from "../game/format";
import { world } from "../game/world";
import { moneyState } from "../game/money";
import { Icon } from "./widgets";
import { perfOff } from "../perfFlags";

const NO_CLOCK = perfOff("clock");

const DEMAND_NAME = { high: "High demand", medium: "Medium demand", low: "Low demand" };

function Clock() {
  // ?perfOff=clock (T-045, measurement only): read without subscribing, so it never re-renders.
  const minute = NO_CLOCK ? clock.minute.peek() : clock.minute.value;
  return (
    <div class="clock">
      <span class="day muted">Day {dayOf(minute)}</span>
      <b class="time big">{clockTime(minute)}</b>
      <span class="demand muted">{DEMAND_NAME[NO_CLOCK ? clock.demand.peek() : clock.demand.value]}</span>
    </div>
  );
}

function Speed() {
  const running = clock.running.value;
  const speed = clock.speed.value;
  const btn = (i: number, icon: ComponentChildren, title: string) => (
    <button class={"btn icon" + (running && speed === i ? " on" : "")} title={title} onClick={() => clock.setSpeed(i)}>
      {icon}
    </button>
  );
  return (
    <div class="group speed">
      <button class={"btn icon" + (!running ? " on" : "")} title="Pause" onClick={() => clock.setRunning(!running)}>
        <Icon.pause />
      </button>
      {btn(0, <Icon.play1 />, "Normal speed")}
      {btn(1, <Icon.play2 />, `${SPEEDS[1] / SPEEDS[0]} times as fast`)}
      {btn(2, <Icon.play3 />, `${SPEEDS[2] / SPEEDS[0]} times as fast`)}
    </div>
  );
}

export function TopBar() {
  const w = world.value;
  if (!w) return <header id="topbar" />;
  const s = modeSplit.value ?? { train: 0, walk: 0, drive: 0 }; // T-026: from the demand workers
  const pct = (v: number) => v.toFixed(1) + "%";
  return (
    <header id="topbar">
      <div class="city">{w.city.name}</div>
      <div class="split" title={`How people in ${w.city.name} get to work${demandUpdating.value ? " (updating)" : ""}`} style={{ opacity: demandUpdating.value ? 0.6 : 1 }}>
        <span class="split-train">
          <i class="sw sw-train" />
          Train <b class="big">{pct(s.train)}</b>
        </span>
        <span class="bar">
          <i class="seg seg-train" style={{ width: s.train + "%" }} />
          <i class="seg seg-walk" style={{ width: s.walk + "%" }} />
          <i class="seg seg-drive" style={{ width: s.drive + "%" }} />
        </span>
        <span class="split-other">
          <i class="sw sw-walk" />
          Walking {pct(s.walk)}
        </span>
        <span class="split-other">
          <i class="sw sw-drive" />
          Driving {pct(s.drive)}
        </span>
      </div>
      <div class="cash num">{money((moneyState.value?.cash ?? 0) * 1e6)}</div>
      <Clock />
      <Speed />
    </header>
  );
}
