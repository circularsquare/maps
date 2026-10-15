// The line inspector: name and colour, whether it runs (and what constructing the rest costs), the
// schedule by demand level with headway and trains needed (SPEC 6.3), its day so far (riders,
// fares, running cost and its trains, T-293), the stops with each stop's time from the start of
// the line and boardings today, and the round
// trip split into running, dwell, turning and waiting, with where it waits (T-031).

import { useEffect, useRef, useState } from "preact/hooks";
import { nearestStation } from "../../game/places";
import { queryClock } from "../../workers/clockClient";
import { clock, demandAt } from "../../game/clock";
import { clockTime, count, distance, duration, headway, money } from "../../game/format";
import { LINE_PALETTE } from "../../game/palette";
import { levelHours, PERIODS, periodAt } from "../../game/periods";
import { signal } from "@preact/signals";
import { today, useToday, useYesterday, yesterday } from "../../game/today";
import { DEMAND_INDEX, DEMANDS, type Demand, type LineInput, type LineStats } from "../../game/types";
import { ask, lineDraft, renameLinePrompt, selection, tool } from "../../game/ui";
import { moneyState } from "../../game/money";
import { edit, linesAt, renameLine, say, setLineColour, stepTrainCount, trainsShown, world } from "../../game/world";
import { live } from "../../map/live";
import { Chip, Icon, Row, Segmented, Stepper } from "../widgets";
import { PurposeSplit } from "./PurposeSplit";

const DEMAND_LABEL: Record<Demand, string> = { high: "High", medium: "Medium", low: "Low" };

/** Which day the Today block shows (T-408), kept while the game is open. */
const dayShown = signal<"today" | "yesterday">("today");

/** When a demand level runs, in the line's local time (the hour spans of game/periods.ts): "High
 * demand: 06:00 to 10:00, 16:00 to 20:00". */
function levelTimes(d: Demand): string {
  const t = (h: number) => clockTime((h * 60) % 1440);
  return `${DEMAND_LABEL[d]} demand: ${levelHours(d).map(([a, b]) => `${t(a)} to ${t(b)}`).join(", ")}`;
}

/** Where a line waits, in words (T-031): from `TrackApi.line_delays`. */
function spotsFrom(d: Float64Array): { text: string; s: number }[] {
  const out: { text: string; s: number }[] = [];
  const st = new Map((world.value?.save.stations ?? []).map((s) => [s.num, s.name]));
  for (let i = 0; i + 4 < d.length; i += 5) {
    const [kind, node, x, y, s] = [d[i], d[i + 1], d[i + 2], d[i + 3], d[i + 4]];
    const near = nearestStation(x, y)?.name;
    const at = node >= 0 ? st.get(node) : undefined;
    const text =
      (kind === 2 || kind === 3) && at
        ? `At ${at}`
        : kind === 4
          ? `At the junction${near ? " near " + near : ""}`
          : kind === 5
            ? `At the crossing${near ? " near " + near : ""}`
            : `On the track${near ? " near " + near : ""}`;
    const same = out.find((o) => o.text === text); // both directions, or two stretches by one station
    if (same) same.s += s;
    else out.push({ text, s });
  }
  return out.sort((a, b) => b.s - a.s).slice(0, 3);
}

/** The round trip at the level in force, split into running, dwell, turnaround and delay (T-031),
 * and where the delay is. */
function RoundTrip({ st, lev, stops, spots }: { st: LineStats; lev: number; stops: number; spots: { text: string; s: number }[] }) {
  const total = st.roundTripByLevel[lev] ?? st.roundTripS;
  const dwell = 2 * Math.max(0, stops - 2) * st.dwellS;
  const turn = 2 * st.turnaroundS;
  const delay = st.delayS[lev] ?? 0;
  const running = Math.max(0, total - dwell - turn - delay);
  return (
    <>
      <ul class="rows">
        <Row label="Round trip" bold>
          {duration(total)}
        </Row>
        <Row label="Running">{duration(running)}</Row>
        <Row label="Dwell at stops">{duration(dwell)}</Row>
        <Row label="Turning at the ends">{duration(turn)}</Row>
        {delay > 0 && <Row label="Waiting at busy track">{duration(delay)}</Row>}
      </ul>
      {delay > 0 && spots.length > 0 && (
        <ul class="rows sub-rows">
          {spots.map((p) => (
            <Row label={p.text}>
              <span class="muted">{duration(p.s)}</span>
            </Row>
          ))}
        </ul>
      )}
    </>
  );
}

/** Times of day for the full-train spans: "07:40 to 08:50, 17:35 to 18:45". */
function spansText(spans: [number, number][]): string {
  const t = (s: number) => clockTime(Math.round(s / 60) % 1440);
  return spans.map(([a, b]) => (b - a < 60 ? t(a) : `${t(a)} to ${t(b)}`)).join(", ");
}

/** US$ of running cost since midnight (T-293): each hour's trains at its demand level, the hour
 * now in part, as the worker drains the cash (T-284). */
function runningToday(l: LineInput, st: LineStats): number {
  const m = moneyState.value;
  if (!m) return 0;
  // T-309: in the line's own local time, from its midnight
  const now = clock.now() + st.shift, h0 = Math.floor(now / 86400) * 24, t = now / 3600;
  const tph = { high: l.tph.high, medium: l.tph.medium, low: l.tph.low };
  let cost = 0;
  for (let h = h0; h < t; h++) {
    const lev = PERIODS[periodAt(h * 3600)].demand;
    cost += tph[lev] * 2 * st.km * st.cars * m.runCostCarKm * m.economy * Math.min(1, t - h);
  }
  return cost;
}

/** The line's day so far (T-293, game/today.ts): riders, fares and running cost, and its trains
 * from the boarding account (T-230): how many left packed full and when, the fullest, and who
 * was left on the platform. Its own component: it re-renders as the counts move. */
function LineToday({ l, st }: { l: LineInput; st: LineStats }) {
  const yest = dayShown.value === "yesterday";
  useYesterday(yest);
  const d = (yest ? yesterday : today).value.get(l.id);
  if (!d && !yest) return null;
  const fullest = d ? Math.round(100 * d.fullest) : 0;
  return (
    <>
      <div class="h">
        <Segmented<"today" | "yesterday">
          options={[
            ["today", <span data-tip="Since midnight, counted train by train. Whoever does not fit on a full train waits for the next one.">Today</span>],
            ["yesterday", <span data-tip="The last complete day, midnight to midnight in the line's local time">Yesterday</span>],
          ]}
          value={dayShown.value}
          onChange={(v) => { dayShown.value = v; }}
        />
      </div>
      {d && <ul class="rows">
        <Row label="Riders">{count(d.riders)}</Row>
        <PurposeSplit scope={[{ line: l.id }]} unit="riders" />
        <Row label="Fares">{money(d.fares)}</Row>
        <Row label="Running trains">{money(-(yest ? d.running : runningToday(l, st)))}</Row>
        <Row label="Fullest train">{d.fullTrains > 0 ? "100%" : `${fullest}% full`}</Row>
        {d.fullTrains > 0 && <Row label="Trains leaving full">{count(d.fullTrains)}</Row>}
        {d.spans.length > 0 && <Row label="When">{spansText(d.spans)}</Row>}
        <Row label="Riders left on the platform">{count(d.leftBehind)}</Row>
      </ul>}
    </>
  );
}

/** One stop's boardings today (T-293), in the stops list. */
function StopToday({ line, i }: { line: string; i: number }) {
  const d = today.value.get(line);
  return <span class="n num" data-tip={d ? `${count(d.stops[i] ?? 0)} boardings today` : undefined}>{d ? count(d.stops[i] ?? 0) : ""}</span>;
}

export function LineInspector({ id }: { id: string }) {
  const w = world.value;
  const [editing, setEditing] = useState(false);
  const nameInput = useRef<HTMLInputElement>(null);
  const nameCommitted = useRef(false);
  const [nameDraft, setNameDraft] = useState("");
  useEffect(() => { if (editing) { nameCommitted.current = false; nameInput.current?.focus(); nameInput.current?.select(); } }, [editing]);
  const [colours, setColours] = useState(false);
  const [copying, setCopying] = useState(false);
  const [spots, setSpots] = useState<{ text: string; s: number }[]>([]);
  const l = w?.save.lines.find((l) => l.id === id);
  const [symbol, setSymbol] = useState("");
  useEffect(() => { setSymbol(l?.letter ?? ""); }, [l?.letter]);
  // T-309: the demand level now in the line's own local time
  const now = demandAt(clock.hour.value * 3600 + (w?.lineStats[id]?.shift ?? 0));
  const lev = DEMAND_INDEX[now];
  useEffect(() => {
    if (renameLinePrompt.value === id) {
      setNameDraft(l?.name ?? "");
      setEditing(true);
      renameLinePrompt.value = null;
    }
  }, [id, renameLinePrompt.value]);
  // where it loses time to busy track, asked again after every edit and when the level turns
  useEffect(() => {
    if (!l) return;
    let on = true;
    void queryClock({ what: "lineDelays", line: l.num, level: lev }).then((d) => on && d && setSpots(spotsFrom(d)));
    return () => {
      on = false;
    };
  }, [l?.num, w?.version, lev]);
  // its day so far (T-293), kept fresh while the panel is open
  useToday();
  if (!w || !l) return null;
  const st = w.lineStats[l.id];
  const name = new Map(w.save.stations.map((s) => [s.id, s.name]));
  const times = st.fromStartByLevel[lev] ?? st.fromStartS;
  const editingStops = tool.value === "line" && lineDraft.value.line === l.num;

  const commit = (e: Event) => {
    if (nameCommitted.current) return;
    nameCommitted.current = true;
    renameLine(l.id, (e.target as HTMLInputElement).value);
    setEditing(false);
  };
  const build = async () => {
    const a = await edit({ op: "constructLine", line: l.num });
    if (a.ok) say(`${l.name} constructed for ${money(a.charge * 1e6)}. Its trains start with the next departure.`);
  };
  const addStops = () => {
    if (editingStops) { live.tools?.exit(); return; }
    live.tools?.exit();
    lineDraft.value = { line: l.num, stops: l.stops.map(Number) };
    tool.value = "line";
    selection.value = { kind: "line", line: l.id };
  };
  const removeStop = async (i: number) => {
    const stops = l.stops.filter((_, k) => k !== i).map(Number);
    const a = await edit({ op: "setStops", line: l.num, stops });
    if (a.ok && lineDraft.peek().line === l.num) lineDraft.value = { line: l.num, stops };
  };
  const remove = () => {
    ask.value = { text: `Delete ${l.name}? Track and stations stay.`, yes: "Delete line", no: "Keep line", run: () => {
      ask.value = null;
      void edit({ op: "removeLine", line: l.num }).then(a => {
        if (!a.ok) return;
        if (lineDraft.peek().line === l.num) live.tools?.exit();
        const selected = selection.peek();
        if (selected?.kind === "line" && selected.line === l.id) selection.value = null;
      });
    }};
  };
  /** A copy to edit; an express copy (T-255) runs on the express pair of quad track where there
   * is one and goes straight to stop editing, so the player clicks off the stops it skips. */
  const copy = async (express = false) => {
    if (copying) return;
    setCopying(true);
    try {
      const a = await edit({ op: "copyLine", line: l.num, express });
      if (!a.ok || a.line === undefined) return;
      live.tools?.exit();
      lineDraft.value = { line: a.line, stops: [] };
      if (!express) renameLinePrompt.value = String(a.line);
      selection.value = { kind: "line", line: String(a.line) };
      tool.value = "line";
    } finally { setCopying(false); }
  };

  return (
    <>
      <div class="title">
        <button class={"badge-edit" + (colours ? " on" : "")} data-tip="Edit line colour and symbol" aria-label="Edit line colour and symbol" onClick={() => setColours(!colours)}><Chip line={l} size="lg" /></button>
        {editing ? (
          <input
            class="name-edit"
            value={nameDraft}
            onInput={e => setNameDraft(e.currentTarget.value)}
            ref={nameInput}
            onBlur={commit}
            onKeyDown={(e) => {
              if (e.key === "Enter") commit(e);
              if (e.key === "Escape") { nameCommitted.current = true; setEditing(false); }
            }}
          />
        ) : (
          <span class="name">{l.name}</span>
        )}
        <button class="btn icon" data-tip="Rename" aria-label="Rename" onClick={() => { setNameDraft(l.name); setEditing(true); }}>
          <Icon.pencil />
        </button>
      </div>
      {colours && (
        <div class="line-identity">
          <label>Symbol <input aria-label="Line symbol" value={symbol} onInput={e => setSymbol(e.currentTarget.value)} onChange={e => void edit({op:"lineLook",line:l.num,name:l.name,colour:l.colour,letter:e.currentTarget.value.trim()})} /></label>
          <div class="palette">
          {LINE_PALETTE.map((c) => (
            <button class={"swatch" + (c === l.colour ? " on" : "")} style={{ "--c": c }} data-tip={c} aria-label={c} onClick={() => setLineColour(l.id, c)} />
          ))}
          <label class="swatch custom" data-tip="Any colour" style={{ "--c": l.colour }}>
            <input type="color" value={l.colour} onChange={(e) => setLineColour(l.id, (e.target as HTMLInputElement).value)} />
          </label>
          </div>
        </div>
      )}
      <div class="sub muted">
        {l.stops.length} stops, {distance(st.km * 1000)}{st.cars ? `, ${st.cars} car trains` : ""}
      </div>
      {l.stops.length < 2 && <p class="muted small">No route: fewer than two stops.</p>}
      {st.status === "planned" && l.stops.length >= 2 && (
        <div class="callout">
          <p>Planned</p>
          <button class="btn text" data-tip={`Track and stations ${money(st.buildCost)}${st.trainCost > 0 ? `, trains ${money(st.trainCost)}` : ""}. Trains run once it is constructed.`} onClick={build}>
            Construct for {money(st.buildCost + st.trainCost)}
          </button>
        </div>
      )}
      {st.status === "broken" && (
        <div class="callout bad">
          <p>Broken: track it used is gone.</p>
        </div>
      )}

      <div class="h">Schedule</div>
      <table class="sched">
        <tr>
          <th>Demand</th>
          <th>Trains on line</th>
          <th>Headway</th>
        </tr>
        {DEMANDS.map((d) => (
          <tr class={d === now ? "now" : ""}>
            <td><span data-tip={levelTimes(d)}>{DEMAND_LABEL[d]}</span></td>
            <td>
              <Stepper value={(trainsShown(l.id) ?? st.trainsNeeded)[DEMAND_INDEX[d]]} label="trains on the entire line" onStep={(s) => stepTrainCount(l.id, d, s)} />
            </td>
            <td>{headway(l.tph[d])}</td>
          </tr>
        ))}
      </table>

      {st.status === "running" && <LineToday l={l} st={st} />}

      <div class="h">
        Stops<span class="right muted small">boardings today</span>
      </div>
      <div class="line-stop-actions">
        <button class={"btn text" + (editingStops ? " on" : "")} onClick={addStops}><Icon.pencil />{editingStops ? "Done editing stops" : "Edit stops"}</button>
        <button class="btn text" onClick={() => void copy()} disabled={copying}>Copy line</button>
        <button class="btn text" onClick={() => void copy(true)} disabled={copying} data-tip="Copy onto the express tracks where there are any, then click the stops it skips">Express copy</button>
        <button class="btn text danger" onClick={remove}>Delete line</button>
      </div>
      <ul class="stops" style={{ "--c": l.colour }}>
        {l.stops.map((s, i) => {
          const others = linesAt(s).filter((o) => o.id !== l.id);
          return (
            <li class={others.length ? "xfer" : ""}>
              <span class="grow">
                <span class="stop-name" onClick={() => (selection.value = { kind: "station", station: s })}>
                  {name.get(s)}
                </span>
                {i > 0 && times[i] !== undefined && <span class="t muted num">{duration(times[i])}</span>}
              </span>
              {others.map((o) => (
                <Chip line={o} size="sm" />
              ))}
              <StopToday line={l.id} i={i} />
              {editingStops && (
                <button class="btn icon mini danger" data-tip="Take this stop out of the line" aria-label={`Remove stop ${name.get(s)}`} onClick={() => removeStop(i)}>
                  <Icon.cross />
                </button>
              )}
            </li>
          );
        })}
      </ul>
      {st.status !== "broken" && l.stops.length >= 2 && <RoundTrip st={st} lev={lev} stops={l.stops.length} spots={spots} />}
    </>
  );
}
