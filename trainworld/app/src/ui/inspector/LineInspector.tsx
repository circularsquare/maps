// The line inspector: name and colour, whether it runs (and what constructing the rest costs), the
// schedule by demand level with headway and trains needed (SPEC 6.3), riders, fares and running
// cost a day (T-028), the stops with each stop's time from the start of the line, and the round
// trip split into running, dwell, turning and waiting, with where it waits (T-031).

import { useEffect, useRef, useState } from "preact/hooks";
import { nearestStation } from "../../game/places";
import { queryClock } from "../../workers/clockClient";
import { clock } from "../../game/clock";
import { lineDemand } from "../../game/demand";
import { count, duration, headway, money } from "../../game/format";
import { LINE_PALETTE } from "../../game/palette";
import { DEMAND_INDEX, DEMANDS, type Demand, type LineStats } from "../../game/types";
import { ask, lineDraft, renameLinePrompt, selection, tool } from "../../game/ui";
import { income } from "../../game/money";
import { edit, linesAt, renameLine, say, setTrainCount, setLineColour, world } from "../../game/world";
import { live } from "../../map/live";
import { Chip, Icon, Row, Stepper } from "../widgets";

const DEMAND_LABEL: Record<Demand, string> = { high: "High", medium: "Medium", low: "Low" };

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
  const now = clock.demand.value;
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
  if (!w || !l) return null;
  const st = w.lineStats[l.id];
  const dm = lineDemand(l.id); // riders: T-026's demand workers; blank until the first solve lands
  const lm = income.value.lines.get(l.id); // fares and running cost a day (T-028)
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
    ask.value = { text: `Delete ${l.name}? This removes the line and its schedule. Track and stations stay.`, yes: "Delete line", no: "Keep line", run: () => {
      ask.value = null;
      void edit({ op: "removeLine", line: l.num }).then(a => {
        if (!a.ok) return;
        if (lineDraft.peek().line === l.num) live.tools?.exit();
        const selected = selection.peek();
        if (selected?.kind === "line" && selected.line === l.id) selection.value = null;
      });
    }};
  };
  const copy = async () => {
    if (copying) return;
    setCopying(true);
    try {
      const a = await edit({ op: "copyLine", line: l.num });
      if (!a.ok || a.line === undefined) return;
      live.tools?.exit();
      lineDraft.value = { line: a.line, stops: l.stops.map(Number) };
      renameLinePrompt.value = String(a.line);
      selection.value = { kind: "line", line: String(a.line) };
      tool.value = "line";
    } finally { setCopying(false); }
  };

  return (
    <>
      <div class="title">
        <button class={"badge-edit" + (colours ? " on" : "")} title="Edit line colour and symbol" aria-label="Edit line colour and symbol" onClick={() => setColours(!colours)}><Chip line={l} size="lg" /></button>
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
        <button class="btn icon" title="Rename" onClick={() => { setNameDraft(l.name); setEditing(true); }}>
          <Icon.pencil />
        </button>
      </div>
      {colours && (
        <div class="line-identity">
          <label>Symbol <input aria-label="Line symbol" value={symbol} onInput={e => setSymbol(e.currentTarget.value)} onChange={e => void edit({op:"lineLook",line:l.num,name:l.name,colour:l.colour,letter:e.currentTarget.value.trim()})} /></label>
          <div class="palette">
          {LINE_PALETTE.map((c) => (
            <button class={"swatch" + (c === l.colour ? " on" : "")} style={{ "--c": c }} title={c} onClick={() => setLineColour(l.id, c)} />
          ))}
          <label class="swatch custom" title="Any colour" style={{ "--c": l.colour }}>
            <input type="color" value={l.colour} onChange={(e) => setLineColour(l.id, (e.target as HTMLInputElement).value)} />
          </label>
          </div>
        </div>
      )}
      <div class="sub muted">
        {l.stops.length} stops, {st.km.toFixed(1)} km{st.cars ? `, ${st.cars} car trains` : ""}
      </div>
      {l.stops.length < 2 && <p class="muted small">Choose at least two stations to give this line a route.</p>}
      {st.status === "planned" && l.stops.length >= 2 && (
        <div class="callout">
          <p>
            Planned. Trains run once its track and stations are constructed.
            {st.trainCost > 0 && ` The price includes ${money(st.trainCost)} for its trains.`}
          </p>
          <button class="btn text" onClick={build}>
            Construct for {money(st.buildCost + st.trainCost)}
          </button>
        </div>
      )}
      {st.status === "broken" && (
        <div class="callout bad">
          <p>Broken. Track it used is gone. Draw track between its stops again, or change its stops.</p>
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
            <td>{DEMAND_LABEL[d]}</td>
            <td>
              <Stepper value={st.trainsNeeded[DEMAND_INDEX[d]]} label="trains on the entire line" onStep={(s) => setTrainCount(l.id, d, st.trainsNeeded[DEMAND_INDEX[d]] + s)} />
            </td>
            <td>{headway(l.tph[d])}</td>
          </tr>
        ))}
      </table>

      <div class="h">Demand forecast</div>
      <p class="muted small">Steady state for the current network and schedule.</p>
      <ul class="rows">
        <Row label="Riders a day">{dm ? count(dm.ridersPerDay) : ""}</Row>
        <Row label="Local trips">{dm ? count(Math.max(0,dm.ridersPerDay-(dm.intercityRidersPerDay ?? 0))) : ""}</Row>
        <Row label="Longer trips">{dm ? count(dm.intercityRidersPerDay ?? 0) : ""}</Row>
        <Row label="Fullest train at high demand">{dm ? dm.fullestPct + "%" : ""}</Row>
      </ul>
      {st.status === "running" && (
        <>
          <div class="h">Finances</div>
          <ul class="rows">
            <Row label="Fares">{dm && lm ? money(lm.faresPerDay) : ""}</Row>
            <Row label="Running trains">{lm ? money(-lm.runningPerDay) : ""}</Row>
          </ul>
        </>
      )}

      <div class="h">
        Stops<span class="right muted small">forecast boardings per day</span>
      </div>
      <div class="line-stop-actions">
        <button class={"btn text" + (editingStops ? " on" : "")} onClick={addStops}><Icon.pencil />{editingStops ? "Done editing stops" : "Edit stops"}</button>
        <button class="btn text" onClick={copy} disabled={copying}>Copy line</button>
        <button class="btn text danger" onClick={remove}>Delete line</button>
      </div>
      {editingStops && <p class="muted small">Click stations to insert stops. Red crosses remove stops; Enter or Escape finishes editing.</p>}
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
              <span class="n num">{dm ? count(dm.boardings[i]) : ""}</span>
              {editingStops && (
                <button class="btn icon mini danger" title="Take this stop out of the line" aria-label={`Remove stop ${name.get(s)}`} onClick={() => removeStop(i)}>
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
