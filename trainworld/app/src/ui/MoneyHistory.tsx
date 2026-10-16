// The Money tab's history (T-447): fares against costs, then net, per game day over the ledger's 30
// days or per game hour over its 48 (the clock worker keeps both and saves them). Days and hours are
// the game clock's (T-321). The day and hour now are drawn so far, moving as fares land (T-284).
// The network's, or the region the Money tab is set to (T-531).

import { signal } from "@preact/signals";
import { clockTime, money } from "../game/format";
import { weekdayName } from "../game/days";
import { moneyState } from "../game/money";
import { regionLedger } from "../game/regionMoney";
import { Segmented } from "./widgets";

/** Days or hours, kept while the game is open. */
const span = signal<"day" | "hour">("day");

interface Entry { at: number; fares: number; running: number; trains: number; build: number }

const W = 300, H = 96, PAD = 3;
/** A young game's few entries sit at the right of this many slots, not stretched over the width. */
const MIN_SLOTS = { day: 7, hour: 12 };

const oneOff = (e: Entry) => e.trains + e.build;
const costs = (e: Entry) => e.running + oneOff(e);
const run = (e: Entry) => e.running;
const net = (e: Entry) => e.fares - costs(e);
const usd = (v: number) => money(v * 1e6);

function when(e: Entry, hours: boolean): string {
  if (!hours) return `${weekdayName(e.at)} ${e.at}`;
  const day = Math.floor(e.at / 24);
  return `${weekdayName(day)} ${day}, ${clockTime((e.at % 24) * 60)}`;
}

/** The ledger's entries (the network's, or a region's: T-531) with the day or hour now replaced by the live one. */
function entries(hours: boolean, zone: string | null): Entry[] {
  if (!moneyState.value) return [];
  const r = regionLedger(zone);
  const list: Entry[] = hours ? r.hours.map((h) => ({ ...h, at: h.hour })) : r.ledger.map((d) => ({ ...d, at: d.day }));
  const now = hours ? (r.hour && { ...r.hour, at: r.hour.hour }) : (r.today && { ...r.today, at: r.today.day });
  if (now) {
    const last = list[list.length - 1];
    if (last && last.at === now.at) list[list.length - 1] = now;
    else if (!last || now.at > last.at) list.push(now);
  }
  return list;
}

export function MoneyHistory({ zone }: { zone: string | null }) {
  const hours = span.value === "hour";
  const list = entries(hours, zone);
  if (!list.length) return null;
  const slots = Math.max(list.length, MIN_SLOTS[span.value]);
  const col = W / slots;
  const x0 = W - list.length * col;
  const cx = (i: number) => x0 + (i + 0.5) * col;
  const last = list.length - 1;
  const bw = Math.max(1, col * 0.7);
  const name = (e: Entry, i: number) => when(e, hours) + (i === last ? " so far" : "");

  // fares and running costs as lines, scaled to the larger of the two; one-off costs (new trains,
  // construction) as bars on the same scale, cut at the top with a cap so a build day cannot
  // flatten the lines (T-452)
  const top = Math.max(...list.map((e) => Math.max(e.fares, e.running)), 1e-9);
  const y1 = (v: number) => H - PAD - (v / top) * (H - 2 * PAD);
  // the day or hour now is drawn dashed: it is still filling
  const pt = (v: (e: Entry) => number, i: number) => `${cx(i).toFixed(1)},${y1(v(list[i])).toFixed(1)}`;
  const path = (v: (e: Entry) => number) => list.slice(0, last).map((_, i) => `${i ? "L" : "M"}${pt(v, i)}`).join("");
  const tail = (v: (e: Entry) => number) => (last > 0 ? `M${pt(v, last - 1)}L${pt(v, last)}` : "");
  const tip1 = (e: Entry, i: number) => [
    name(e, i),
    `Fares ${usd(e.fares)}`,
    `Running trains ${usd(-e.running)}`,
    ...(oneOff(e) > 0 ? [`One-off costs ${usd(-oneOff(e))}`] : []),
    ...(e.trains > 0 ? [`New trains ${usd(-e.trains)}`] : []),
    ...(e.build > 0 ? [`Construction ${usd(-e.build)}`] : []),
  ].join("\n");

  // net: bars up or down from zero
  const hi = Math.max(0, ...list.map(net)), lo = Math.min(0, ...list.map(net));
  const range = hi - lo || 1e-9;
  const y2 = (v: number) => PAD + ((hi - v) / range) * (H - 2 * PAD);
  const zero = y2(0);

  const first = list[0];
  return (
    <>
      <div class="h">
        History
        <span class="right">
          <Segmented<"day" | "hour">
            options={[
              ["day", <span data-tip="Each game day, up to the past 30">Days</span>],
              ["hour", <span data-tip="Each game hour, up to the past 48">Hours</span>],
            ]}
            value={span.value}
            onChange={(v) => { span.value = v; }}
          />
        </span>
      </div>
      <div class="hist-title small"><span><span class="hist-fares">Fares</span>, <span class="hist-costs">running costs</span> and <span class="hist-oneoff">one-off costs</span></span><span class="right muted" data-tip={`The chart's top is ${usd(top)}`}>{usd(top)}</span></div>
      <svg class="hist" viewBox={`0 0 ${W} ${H}`}>
        <line class="hist-axis" x1={0} x2={W} y1={H - PAD} y2={H - PAD} />
        {list.map((e, i) => {
          const v = oneOff(e);
          if (v <= 0) return null;
          const y = Math.max(PAD, y1(v));
          return <>
            <rect class={"hist-oneoff" + (i === last ? " now" : "")} x={cx(i) - bw / 2} y={y} width={bw} height={Math.max(0.5, H - PAD - y)} />
            {v > top && <path class="hist-oneoff cap" d={`M${cx(i) - bw / 2},${PAD + 3}L${cx(i)},0L${cx(i) + bw / 2},${PAD + 3}Z`} />}
          </>;
        })}
        <path class="hist-costs" d={path(run)} />
        <path class="hist-fares" d={path((e) => e.fares)} />
        <path class="hist-costs now" d={tail(run)} />
        <path class="hist-fares now" d={tail((e) => e.fares)} />
        {list.map((e, i) => <circle class="hist-costs dot" cx={cx(i)} cy={y1(e.running)} r={1.6} />)}
        {list.map((e, i) => <circle class="hist-fares dot" cx={cx(i)} cy={y1(e.fares)} r={1.6} />)}
        {list.map((e, i) => <rect class="hist-col" x={x0 + i * col} y={0} width={col} height={H} data-tip={tip1(e, i)} />)}
      </svg>
      <div class="hist-title small"><span>Net</span><span class="right muted" data-tip={`The chart runs from ${usd(lo)} to ${usd(hi)}`}>{usd(hi)}{lo < 0 ? ` to ${usd(lo)}` : ""}</span></div>
      <svg class="hist" viewBox={`0 0 ${W} ${H}`}>
        <line class="hist-axis" x1={0} x2={W} y1={zero} y2={zero} />
        {list.map((e, i) => {
          const v = net(e), y = y2(v);
          return <rect class={(v < 0 ? "hist-loss" : "hist-gain") + (i === last ? " now" : "")} x={cx(i) - bw / 2} y={Math.min(y, zero)} width={bw} height={Math.max(0.5, Math.abs(y - zero))} />;
        })}
        {list.map((e, i) => <rect class="hist-col" x={x0 + i * col} y={0} width={col} height={H} data-tip={`${name(e, i)}\nNet ${usd(net(e))}`} />)}
      </svg>
      <div class="hist-x small muted"><span>{when(first, hours)}</span><span>{when(list[last], hours)}</span></div>
    </>
  );
}
