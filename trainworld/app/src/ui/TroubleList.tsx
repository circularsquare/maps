// The trouble list (T-1433, docs/ui-panels.md "Trouble list"): one list of what is wrong on the
// network, at the map's lower right. Collapsed it is a quiet icon with a count (some players will
// not see these as real problems); open, a short list, worst first, each row a click from the place.
// It reads what the map and the line panel already have: the world's capacity tags and reversals
// (redrawn with the world) and the past 24 hours' counts, asked once a game hour.

import { h, render, type ComponentChildren } from "preact";
import { useEffect, useMemo, useState } from "preact/hooks";
import { accountVersion } from "../game/boarding";
import { CAPACITY_KINDS } from "../game/capacityKinds";
import { clock } from "../game/clock";
import { lngLatToPack } from "../game/coords";
import { count, duration } from "../game/format";
import { nearestStation, stationNameNear } from "../game/places";
import { readToday, type LineToday } from "../game/today";
import { DEMAND_INDEX, type LineInput, type Selection, type WorldView } from "../game/types";
import { selection, tool } from "../game/ui";
import { world } from "../game/world";
import { nearestJunction } from "../map/capacityMarkers";
import { toLngLat } from "../map/geo";
import { reversalsOf } from "../map/reversalMarks";
import { queryClock } from "../workers/clockClient";
import { locateLine, locateStation } from "./locate";
import { Chip, Icon } from "./widgets";

const STORE = "anitabuilder-trouble-open";
const DEMAND_WORD = ["high", "medium", "low"];
/** capacity kinds tied to one station (`CAPACITY_KINDS`): platform, terminus, turning, reversing, sidings */
const AT_STATION = new Set([2, 3, 6, 8, 9, 10]);
/** a line is listed for crowding past these: riders left at the platform as a share of those who
 * wanted its trains, or trains that left full as a share of those that ran, in the past 24 hours */
const LEFT_SHARE = 0.03, LEFT_MIN = 10, FULL_SHARE = 0.25;
/** capacity tags from the map's "near full" orange up (`capacityMarkers.ts`) */
const NEAR_RHO = 0.9;

type Item = {
  key: string;
  /** worst first: 0 broken line, 1 reversal, 2 spot over capacity, 3 riders left behind, 4 delay, 5 full trains */
  tier: number;
  size: number;
  lead: ComponentChildren;
  label: string;
  value: ComponentChildren;
  tip: string;
  go: () => void;
};

const selecting = () => tool.peek() === "select";
const pick = (s: Selection) => { if (selecting()) selection.value = s; };
const goLine = (l: LineInput) => () => { locateLine(l.stops); pick({ kind: "line", line: l.id }); };

function itemsOf(w: WorldView, level: number, day: Map<string, LineToday>): Item[] {
  const out: Item[] = [];
  for (const l of w.save.lines) {
    const st = w.lineStats[l.id];
    if (!st) continue;
    if (st.status === "broken") {
      out.push({ key: "b" + l.id, tier: 0, size: 0, lead: <Chip line={l} size="sm" plain />, label: l.name,
        value: <span class="crowd-bad"><Icon.alert /></span>, tip: "Track this line ran on is gone", go: goLine(l) });
      continue;
    }
    const d = st.status === "running" ? day.get(l.id) : undefined;
    if (!d) continue;
    const wanted = d.riders + d.leftBehind;
    if (d.leftBehind >= LEFT_MIN && d.leftBehind >= LEFT_SHARE * wanted) {
      out.push({ key: "l" + l.id, tier: 3, size: d.leftBehind, lead: <Chip line={l} size="sm" plain />, label: l.name,
        value: <span class="trouble-fig"><Icon.person />{count(d.leftBehind)}</span>,
        tip: `${count(d.leftBehind)} of the ${count(wanted)} riders who wanted a train were left at the platform by a full one in the past 24 hours\n${count(d.fullTrains)}/${count(d.trains)} trains left full`,
        go: goLine(l) });
    } else if (d.trains > 0 && d.fullTrains >= FULL_SHARE * d.trains) {
      out.push({ key: "f" + l.id, tier: 5, size: d.fullTrains / d.trains, lead: <Chip line={l} size="sm" plain />, label: l.name,
        value: <span class="trouble-fig"><Icon.train />{count(d.fullTrains)}/{count(d.trains)}</span>,
        tip: `${count(d.fullTrains)}/${count(d.trains)} trains left full in the past 24 hours`, go: goLine(l) });
    }
  }
  const lines = new Map(w.save.lines.map((l) => [l.id, l]));
  for (const r of reversalsOf(w)) {
    const ls = r.lines.map((x) => lines.get(x.id)).filter((x): x is LineInput => !!x);
    out.push({ key: "r" + r.station, tier: 1, size: ls.length, lead: <i class="trouble-rev"><Icon.uturn /></i>, label: r.name,
      value: <span class="trouble-chips">{ls.slice(0, 3).map((l) => <Chip line={l} size="sm" plain />)}</span>,
      tip: r.lines.map((x) => `${x.name} reverses here`).join("\n"),
      go: () => { locateStation(r.lng, r.lat); if (r.lines[0]) pick({ kind: "line", line: r.lines[0].id }); } });
  }
  // the orange and red tags only (90% and up); a kind's tags by one station are one row, the worst
  const spots = new Map<string, Item>();
  for (const m of [...(w.capacity[level] ?? [])].sort((a, b) => b.delayS - a.delayS || b.rho - a.rho)) {
    const d = Math.round(m.delayS), over = m.rho > 1 || m.held;
    if (m.rho < NEAR_RHO && !over) continue;
    const k = CAPACITY_KINDS[m.kind] ?? CAPACITY_KINDS[0];
    // the tags are in the draw anchor's local units, the stations in the pack's
    const [lng, lat] = toLngLat(m.x, m.y), p = lngLatToPack(lng, lat);
    const near = nearestStation(p.x, p.y);
    const place = AT_STATION.has(m.kind) ? near?.name ?? "" : stationNameNear(p.x, p.y);
    const key = `c${m.kind},${place || m.x + "," + m.y}`;
    if (spots.has(key)) continue;
    const cls = over ? "over" : "near";
    const tip = [`${k.name} at ${Math.round(m.rho * 100)}% of capacity`];
    if (m.held) tip.push("The line's extra trains are held out of service at its ends");
    if (d >= 2) tip.push(`Each train waits ${duration(d)} here at ${DEMAND_WORD[level] ?? "this"} demand`);
    spots.set(key, { key, tier: over ? 2 : 4, size: d + m.rho, lead: <i class={"trouble-mark " + cls}><Icon.alert /></i>,
      label: place ? `${k.name} ${k.prep} ${place}` : k.name,
      value: <span class={"trouble-tag " + cls}>{d >= 2 ? `+${Math.floor(d / 60)}:${String(d % 60).padStart(2, "0")}` : `${Math.round(m.rho * 100)}%`}</span>,
      tip: tip.join("\n"),
      go: () => {
        locateStation(lng, lat);
        const j = m.kind === 4 ? nearestJunction(w, m.x, m.y) : null;
        if (j !== null) pick({ kind: "junction", node: j });
        else if (AT_STATION.has(m.kind) && near) pick({ kind: "station", station: near.id });
      } });
  }
  out.push(...spots.values());
  return out.sort((a, b) => a.tier - b.tier || b.size - a.size);
}

function readOpen(): boolean {
  try { return localStorage.getItem(STORE) === "1"; } catch { return false; }
}

function TroubleList() {
  const w = world.value;
  if (!w || !clock) return null;
  return <Trouble w={w} />;
}

function Trouble({ w }: { w: WorldView }) {
  const hour = clock.hour.value, level = DEMAND_INDEX[clock.demand.value], acct = accountVersion.value;
  const [day, setDay] = useState<Map<string, LineToday>>(new Map());
  const [open, setOpen] = useState(readOpen);
  // the past 24 hours once a game hour and when the riders are worked out again, not the panels'
  // four times a second
  useEffect(() => {
    let live = true;
    void queryClock({ what: "today" }).then((d) => { if (live && d) setDay(readToday(d)); });
    return () => { live = false; };
  }, [hour, acct]);
  const items = useMemo(() => itemsOf(w, level, day), [w, level, day]);
  if (!items.length) return null;
  const toggle = () => {
    setOpen(!open);
    try { localStorage.setItem(STORE, open ? "0" : "1"); } catch { /* remembered for this visit only */ }
  };
  return (
    <div class={"trouble" + (open ? " open" : "")}>
      {open && (
        <ul class="rows pick trouble-list">
          {items.map((it) => (
            <li key={it.key} data-tip={it.tip} onClick={it.go}>
              {it.lead}
              <span class="grow">{it.label}</span>
              <span class="val">{it.value}</span>
            </li>
          ))}
        </ul>
      )}
      <button class={"btn trouble-btn" + (open ? " on" : "")} aria-expanded={open} aria-label="Network trouble"
        data-tip={open ? "Hide the trouble list" : "Show the trouble list"} onClick={toggle}>
        <Icon.alert />{count(items.length)}
      </button>
    </div>
  );
}

/** Mount the trouble list in the map's container. */
export function startTroubleList(container: HTMLElement) {
  const root = document.createElement("div");
  container.appendChild(root);
  render(h(TroubleList, null), root);
}
