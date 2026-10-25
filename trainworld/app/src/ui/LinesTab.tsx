// The Lines tab: the list of lines with their riders, money and trains now.

import { demandAt, lineSeconds } from "../game/clock";
import { PERIODS, periodAt } from "../game/periods";
import { useState } from "preact/hooks";
import { createLine } from "../game/lineActions";
import { count, money } from "../game/format";
import { DEMAND_INDEX, type LineInput, type LineStats } from "../game/types";
import { lineSort, selection } from "../game/ui";
import { stationsById, world } from "../game/world";
import { cityById } from "../game/cities";
import { scopedLists, stationZone } from "../game/listScope";
import { accountVersion } from "../game/boarding";
import { crowdClass, crowding, runningToday, today, todayLoaded, useToday, waitText, type LineToday } from "../game/today";
import { Chip, Icon, SortHead } from "./widgets";
import { Money } from "./MoneyText";
import { locateLine } from "./locate";
import { byNear, natural, Near, ScopeToggle } from "./listParts";

const STATUS: Record<string, string> = { running: "", planned: "Planned", broken: "Broken" };

/** T-446: one line's riders in the past 24 hours, its own component so only it re-renders as the count moves. */
function RidersToday({ line }: { line: string }) {
  const d = today.value.get(line);
  return <span class="val w-riders" data-tip={d ? `${count(d.riders)} riders in the past 24 hours` : undefined}>{d ? count(d.riders) : ""}</span>;
}

/** T-735, T-867: a line's fares less running cost and its share of upkeep over the past 24 hours. */
const lineNet = (d: LineToday | undefined, l: LineInput, st: LineStats) => (d ? d.fares - runningToday(l, st) - st.upkeep : 0);
function MoneyToday({ l, st }: { l: LineInput; st: LineStats }) {
  const d = today.value.get(l.id);
  if (!d) return <span class="val w-linemoney" />;
  const run = runningToday(l, st), net = d.fares - run - st.upkeep;
  return (
    <span class="val w-linemoney" data-tip={`Fares ${money(d.fares)} less running cost ${money(run)} and upkeep of its track and stations ${money(st.upkeep)} in the past 24 hours: ${money(net)} net`}>
      <Money dollars={net} />
    </span>
  );
}

/** T-472, T-491: a line's trains now, coloured by its riders' wait for a train with room in the period now. */
function TrainsNow({ line, n, period }: { line: string; n: number; period: string }) {
  const d = today.value.get(line);
  const c = d ? crowding(d.crowdWait, d.crowdRiders, d.crowdRide, d.crowdWaitKm, d.crowdRidersKm) : null;
  const cls = c ? crowdClass(c.score) : "";
  // T-1255: the wait at the platforms where trains come full
  const tip = d && c
    ? (d.crowdRiders > 0 ? `${period}: where trains come full, riders face a wait of ${waitText(c.wait)} on average for a train with room` : `${period}: riders get on the first train with room`)
      + (d.trainsNow > 0 ? `\n${count(d.fullNow)}/${count(d.trainsNow)} trains running now are full on their trip` : "")
    : undefined;
  return <span class={"val w-trains " + (cls || "muted")} data-tip={tip}>{n}</span>;
}

function TotalToday() {
  const t = today.value;
  if (!t.size) return null;
  let n = 0;
  for (const d of t.values()) n += d.riders;
  return <span class="muted">{count(n)} riders in the past 24 hours</span>;
}

export function LinesTab() {
  const [creating, setCreating] = useState(false);
  useToday(); // T-446: riders in the past 24 hours, kept fresh while the tab is open
  void accountVersion.value; // re-sorted each game hour; the counts themselves move on their own
  void todayLoaded.value;
  const w = world.value;
  if (!w) return null;
  const sel = selection.value;
  // T-309: each line's demand level now in its own local time
  const levOf = (id: string) => DEMAND_INDEX[demandAt(lineSeconds(w.lineStats[id]))];
  const t = today.peek();
  const newLine = async () => {
    if (creating) return;
    setCreating(true);
    try { await createLine(); } finally { setCreating(false); }
  };
  // a line that is not running has no train count and no riders: it sorts as zero
  const trains = (id: string) => (w.lineStats[id]?.status === "running" ? w.lineStats[id].trainsNeeded[levOf(id)] : 0);
  const { key, dir } = lineSort.value;
  // T-332: only the viewed place unless the world is chosen
  const sc = scopedLists(w.save.stations, w.save.lines, (k, id) => (k === "line" && (sel?.kind === "line" || sel?.kind === "train") && sel.line === id));
  // T-670: on the whole world, a city column once lines start in two or more cities (a line's city is where its first stop is)
  const cityOf = new Map<string, string>();
  if (sc.zone === null) {
    const at = stationsById(w);
    for (const l of sc.lines) {
      const s = at.get(l.stops[0]), z = s ? stationZone(s) : null;
      cityOf.set(l.id, z ? cityById(z).name : "");
    }
  }
  const showCity = new Set([...cityOf.values()].filter(Boolean)).size >= 2;
  const lines = [...sc.lines].sort((a, b) =>
    (key === "city" ? natural(cityOf.get(a.id) ?? "", cityOf.get(b.id) ?? "") * dir
      : key === "riders" ? ((t.get(a.id)?.riders ?? 0) - (t.get(b.id)?.riders ?? 0)) * dir
      : key === "money" ? (lineNet(t.get(a.id), a, w.lineStats[a.id]) - lineNet(t.get(b.id), b, w.lineStats[b.id])) * dir
      : key === "trains" ? (trains(a.id) - trains(b.id)) * dir
      : key === "near" ? byNear(sc.lineDist.get(a.id), sc.lineDist.get(b.id), dir) : 0)
    || natural(a.name, b.name) * (key === "name" ? dir : 1) || a.num - b.num);
  return (
    <>
      <div class="line-list-actions">
        <button class="btn text" data-tip="Make a new line (N)" onClick={newLine} disabled={creating}><Icon.plus />New line<span class="key muted">N</span></button>
        <ScopeToggle />
      </div>
      {lines.length === 0 && <p class="muted">{w.save.lines.length === 0 ? "No lines yet." : "No lines here."}</p>}
      <div class={"lines-box" + (showCity ? " with-city" : "") + (key === "near" ? " near-on" : "")}>
      {w.save.lines.length > 0 && <SortHead sort={lineSort} cols={[
        { key: "name", label: "Name", tip: "Sort by name", first: 1 },
        ...(showCity ? [{ key: "city" as const, label: "City", tip: "Sort by the city of the first stop", first: 1 as const, cls: "w-city", left: true }] : []),
        { key: "riders", label: "Riders", tip: "Sort by riders in the past 24 hours", first: -1, cls: "w-riders" },
        { key: "money", label: "Money", tip: "Sort by money in the past 24 hours, fares less running cost and upkeep", first: -1, cls: "w-linemoney" },
        { key: "trains", label: "Trains", tip: "Sort by trains needed", first: -1, cls: "w-trains" },
        { key: "near", label: "Near", tip: "Sort by distance from the centre of the screen, by the nearest stop", first: 1, cls: key === "near" ? "w-near" : "near-off" },
      ]} />}
      <ul class="rows pick">
        {lines.map((l) => {
          const st = w.lineStats[l.id];
          const on = (sel?.kind === "line" || sel?.kind === "train") && sel.line === l.id;
          return (
            <li class={on ? "on" : ""} onClick={() => (selection.value = { kind: "line", line: l.id })} onDblClick={() => locateLine(l.stops)}>
              <Chip line={l} />
              <span class="grow line-name">{l.name}</span>
              {showCity && <span class="val muted w-city" data-tip={cityOf.get(l.id) ? `The first stop is in ${cityOf.get(l.id)}` : undefined}>{cityOf.get(l.id)}</span>}
              {st.status !== "running" ? (
                <span class="val muted">{STATUS[st.status]}</span>
              ) : (
                <>
                  <RidersToday line={l.id} />
                  <MoneyToday l={l} st={st} />
                  <TrainsNow line={l.id} n={st.trainsNeeded[levOf(l.id)]} period={PERIODS[periodAt(lineSeconds(st))]?.name ?? ""} />
                </>
              )}
              {key === "near" && <Near m={sc.lineDist.get(l.id)} />}
            </li>
          );
        })}
      </ul>
      </div>
      <div class="pane-foot">
        <TotalToday />
      </div>
    </>
  );
}
