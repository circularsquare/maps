// Inspectors for a station, a train, a stretch of track and a junction, and placeholders for cell
// and zone (T-026's).

import { useEffect, useState } from "preact/hooks";
import { clock, SPEEDS } from "../../game/clock";
import { demandView, lineDemand } from "../../game/demand";
import { loadFailure } from "../../game/loading";
import { PERIODS, periodAt } from "../../game/periods";
import { count, distance, distanceOf, duration, length, levelLabel, money, roughDistance, speed, upkeepRule } from "../../game/format";
import { compass, stationTowards } from "../../game/places";
import { DEMAND_INDEX, type Level, type LineInput, type Selection, type StationInput } from "../../game/types";
import { complexOf, inComplexView, openAlone } from "../../game/complexView";
import { complexNow, type ComplexNow } from "../../game/complexLive";
import { LINK_MAX_M } from "../../game/complexes";
import { cancelLink, linking, linksAmong, startLink, unlink } from "../../game/links";
import { queryClock } from "../../workers/clockClient";
import { parseEdgePis, type EdgePi, type EdgePis } from "../../map/edgeEdit";
import { display, selection, setDisplay, tool } from "../../game/ui";
import { stationPoint } from "../../game/trip";
import { pickTool } from "../tabs";
import { edit, linesAt, say, world } from "../../game/world";
import { createLine } from "../../game/lineActions";
import { live } from "../../map/live";
import { removeThing } from "../../game/remove";
import { describeIssues } from "../../game/issues";
import { tripAt } from "../../map/network";
import { nextLeavingAt, platformsAt, tripLoad } from "../../game/boarding";
import { today, useToday, type LineToday } from "../../game/today";
import { moneyState } from "../../game/money";
import { edgeUpkeep, stationUpkeep } from "../../game/regionMoney";
import { Actions, Chip, Icon, Row, Segmented, Stepper } from "../widgets";
import { nextSuggestion } from "../../map/stationNames";
import { NameEdit } from "./NameEdit";
import { locateStation } from "../locate";
import { StationRiders } from "./StationRiders";
import { StationLongerTrips } from "./LongerTrips";
import { ComplexMap } from "./ComplexMap";
import { ClosestStations } from "./ClosestStations";
import { PurposeSplit } from "./PurposeSplit";
import { TrainStops } from "./TrainStops";

function StationGlyph() {
  return <span class="st-glyph" />;
}

function Built({ built, what }: { built: boolean; what: string }) {
  return built ? null : <p class="muted" data-tip={`${what} is built when you construct it`}>Blueprint</p>;
}

/** Open station `m` on its own, in station view, from its complex (T-246, T-249). */
function openStation(m: string) {
  openAlone(m);
  live.overlay?.invalidate();
}

/** One row of the complex view's list (T-250): a line name and every station of the complex where a
 * line of that name stops (the save's branches and patterns, "subway 6" to Parkchester and to Pelham
 * Bay Park, summed), or a station no line stops at. */
interface ComplexRow {
  name: string;
  line: LineInput | null;
  stations: string[];
  /** the busiest of them, opened by a click */
  open: string;
  built: boolean;
  waiting: number | null;
  full: boolean;
  /** the forecast's boardings a day: only the rows' order and the station a click opens */
  boardings: number;
  /** boardings in the past 24 hours (T-446), null before the first count */
  today: number | null;
  tip: string;
  waitTip: string;
}

function complexRows(members: string[], lines: LineInput[], byId: Map<string, StationInput>, now: ComplexNow | null, today: Map<string, LineToday>): ComplexRow[] {
  const inside = new Set(members);
  const groups = new Map<string, LineInput[]>();
  for (const l of lines) groups.set(l.name, [...(groups.get(l.name) ?? []), l]);
  const rows: ComplexRow[] = [];
  for (const [name, ls] of groups) {
    const ids = new Set(ls.map((l) => l.id));
    const per = new Map<string, number>();
    let todayB: number | null = null;
    for (const l of ls) {
      const t = today.get(l.id);
      l.stops.forEach((st, k) => {
        if (!inside.has(st)) return;
        per.set(st, (per.get(st) ?? 0) + (lineDemand(l.id)?.boardings[k] ?? 0));
        if (t) todayB = (todayB ?? 0) + (t.stops[k] ?? 0);
      });
    }
    const stations = [...per.keys()];
    const boardings = [...per.values()].reduce((a, b) => a + b, 0);
    const open = stations.reduce((a, b) => ((per.get(b) ?? 0) > (per.get(a) ?? 0) ? b : a), stations[0]);
    const qs = (now?.queues ?? []).filter((q) => ids.has(q.line));
    const waiting = now && now.platforms.size ? qs.reduce((a, q) => a + q.waiting, 0) : null;
    const full = qs.some((q) => q.leftBehind) || stations.some((m) => {
      const p = now?.platforms.get(m);
      return !!p && Math.max(p.side[0], p.side[1]) >= p.room;
    });
    const names = [...new Set(stations.map((m) => byId.get(m)?.name ?? ""))];
    const ends = [...new Set(ls.flatMap((l) => [l.stops[0], l.stops[l.stops.length - 1]]).filter((s) => !inside.has(s)).map((s) => byId.get(s)?.name ?? ""))];
    rows.push({
      name, line: ls[0], stations, open, built: stations.some((m) => byId.get(m)?.built), waiting, full, boardings, today: todayB,
      tip: `${name} at ${names.join(" and ")}` + (ends.length ? `\nTo ${ends.join(", ")}` : ""),
      waitTip: waiting === null ? "Nobody waiting: no train calls here this hour"
        : `${count(waiting)} riders waiting for ${name} now` + (full ? "\nA platform is full, or the last train left riders behind" : ""),
    });
  }
  rows.sort((a, b) => b.boardings - a.boardings || a.name.localeCompare(b.name));
  for (const m of members) {
    if (lines.some((l) => l.stops.includes(m))) continue;
    const s = byId.get(m);
    if (s) rows.push({ name: s.name, line: null, stations: [m], open: m, built: s.built, waiting: null, full: false, boardings: 0, today: null, tip: `${s.name}\nNo line stops here`, waitTip: "" });
  }
  return rows;
}

/** The complex now (game/complexLive.ts): worked out again every 10 game seconds or so (at most
 * every 150 ms of real time), and re-rendered only when something shown changed. T-1096: the same
 * trains keep the object, their times updated in place, as the complex map moves them itself; it
 * looks ahead twice the game time between workings-out at the speed set. */
function useComplexNow(members: string[]): ComplexNow | null {
  const [now, setNow] = useState<ComplexNow | null>(() => complexNow(members));
  const scope = members.join(",");
  useEffect(() => {
    let key = "", t = NaN, board: unknown = null, shown: ComplexNow | null = null;
    const tick = () => {
      const r = live.overlay?.renderer, at = clock.now();
      if (Math.abs(at - t) < 10 && r?.board === board) return;
      t = at;
      board = r?.board ?? null;
      const c = complexNow(members, 2 * Math.max(10, 0.15 * SPEEDS[clock.speed.peek()]) + 10);
      const k = c?.key ?? "";
      if (k !== key || !shown || !c) {
        key = k;
        shown = c;
        setNow(c);
      } else shown.trains = c.trains;
    };
    tick();
    const h = setInterval(tick, 150);
    return () => clearInterval(h);
  }, [scope, world.value]);
  return now;
}

/** A station complex as one (T-146): its stations, the lines serving any of them and the
 * boardings at all of them. */
function ComplexSummary({ id, members }: { id: string; members: string[] }) {
  const w = world.value;
  if (!w) return null;
  const inside = new Set(members);
  const lines = w.save.lines.filter((l) => l.stops.some((st) => inside.has(st)));
  const byId = new Map(w.save.stations.map((s) => [s.id, s]));
  const now = useComplexNow(members);
  return (
    <>
      <ComplexLines id={id} members={members} lines={lines} byId={byId} now={now} />
      <ComplexMap id={id} members={members} stations={byId} lines={lines} now={now} onOpen={openStation} />
      {lines.length > 0 && <BoardingsToday lines={lines} at={(l) => l.stops.flatMap((st, k) => (inside.has(st) ? [k] : []))} />}
    </>
  );
}

/** The complex's lines (T-246, T-250, Anita: "show on top the component stations ... one row per
 * line"), one row per line name with its branches and directions summed: riders waiting now and
 * boardings in the past 24 hours (T-293, T-446). A click opens the line's busiest station there on its own. Its own
 * component, re-rendered as today's counts move. */
function ComplexLines({ id, members, lines, byId, now }: { id: string; members: string[]; lines: LineInput[]; byId: Map<string, StationInput>; now: ComplexNow | null }) {
  const rows = complexRows(members, lines, byId, now, today.value);
  return (
    <ul class="rows pick complex-stations">
      <li class="cols muted small">
        <span class="grow">Lines</span>
        <span class="val">Waiting</span>
        <span class="val">24 h</span>
      </li>
      {rows.map((r) => (
        <li class={r.stations.includes(id) ? "on" : ""} onClick={() => openStation(r.open)} data-tip={r.tip}>
          {r.line && <Chip line={r.line} size="sm" />}
          <span class={"grow" + (r.built ? "" : " muted")}>{r.name}</span>
          <span class={"val" + (r.full ? " bad" : "")} data-tip={r.waitTip}>{r.waiting !== null ? count(r.waiting) : ""}</span>
          <span class="val" data-tip={r.today !== null ? `${count(r.today)} boardings in the past 24 hours` : undefined}>{r.today !== null ? count(r.today) : ""}</span>
        </li>
      ))}
    </ul>
  );
}

/** A station's or complex's boardings in the past 24 hours (T-446), with the purpose
 * shares; `at`: the stops of each line counted. Its own component, re-rendered as counts move. */
function BoardingsToday({ lines, at }: { lines: LineInput[]; at: (l: LineInput) => number[] }) {
  const t = today.value;
  const known = lines.some((l) => t.has(l.id));
  const total = lines.reduce((a, l) => a + at(l).reduce((b, k) => b + (t.get(l.id)?.stops[k] ?? 0), 0), 0);
  return (
    <>
      <ul class="rows pick">
        <Row label="Boardings in past 24 h" bold>
          {known ? count(total) : ""}
        </Row>
        <PurposeSplit scope={lines.map((l) => ({ line: l.id, at: at(l) }))} unit="boardings" />
      </ul>
    </>
  );
}

export function StationInspector({ id }: { id: string }) {
  const w = world.value;
  const [cycling, setCycling] = useState(false);
  useToday(); // T-446: boardings in the past 24 hours, kept fresh while the panel is open
  const s = w?.save.stations.find((s) => s.id === id);
  if (!w || !s) return null;
  // a station complex (T-146): the switch shows only when there is more than this station
  const members = complexOf(id);
  const asComplex = members.length > 1 && inComplexView(id);
  const scope = asComplex ? members.join(",") : id;
  const lines = linesAt(id);
  const props = (platform: number, name = s.name) => void edit({ op: "stationProps", node: s.num, platform, name });
  // T-534: the next of the station's suggested names, worked out now from where it stands
  const cycle = async () => {
    if (cycling || !live.map) return;
    setCycling(true);
    try {
      const others = () => new Set(world.peek()?.save.stations.filter((o) => o.id !== id).map((o) => o.name) ?? []);
      const next = await nextSuggestion(live.map, s.lng, s.lat, s.name, others, live.overlay?.renderer.net ?? null);
      const now = world.peek()?.save.stations.find((o) => o.id === id);
      if (!now) return;
      if (next) void edit({ op: "stationProps", node: now.num, platform: now.length, name: next });
      else say("No other name is suggested here.");
    } finally {
      setCycling(false);
    }
  };
  const construct = async () => {
    const a = await edit({ op: "construct", edges: [], stations: [s.num] });
    if (a.ok) say(`${s.name} constructed for ${money(a.charge * 1e6)}.`);
  };
  // constructed: asks first above the bottom bar (T-096)
  const remove = () => void removeThing({ kind: "station", node: s.num });
  return (
    <>
      <div class="title">
        <StationGlyph />
        <NameEdit key={id} name={s.name} onRename={(n) => props(s.length, n)}>
          <button class="btn icon" data-tip="Next suggested name" aria-label="Next suggested name" onClick={() => void cycle()}>
            <Icon.cycle />
          </button>
          <button class="btn icon" data-tip="Show on the map" aria-label="Show on the map" onClick={() => locateStation(s.lng, s.lat)}>
            <Icon.locate />
          </button>
        </NameEdit>
      </div>
      {/* a click on a station always shows its complex (T-249); a station opened on its own from
          the complex's list goes back to it from here, the Station / Complex switch is gone */}
      <div class="sub muted">
        {asComplex ? `Complex of ${members.length} stations` : `Elevation ${levelLabel(s.level)}`}
        {!asComplex && members.length > 1 && (
          <>
            {" · "}
            <button class="btn text inline" onClick={() => { selection.value = { kind: "station", station: id }; live.overlay?.invalidate(); }} data-tip="Show the whole complex">
              Complex of {members.length} stations
            </button>
          </>
        )}
      </div>
      <StationTripLink id={id} members={asComplex ? members : [id]} />
      {!asComplex && <ConstructRemove built={s.built} construct={construct} constructLabel="Construct station" remove={remove} removeLabel="Remove station" removeTip={s.built ? "Removing a constructed station refunds nothing" : ""} />}
      {asComplex ? (
        <ComplexSummary id={id} members={members} />
      ) : (
        <StationDetails s={s} lines={lines} props={props} />
      )}
      <ClosestStations lng={s.lng} lat={s.lat} exclude={new Set(members)} own={s.name} />
      <StationLinks id={id} members={asComplex ? members : [id]} />
      <StationRiders id={id} scope={scope} />
      <StationLongerTrips id={id} scope={scope} />
    </>
  );
}

/** Start a trip at a station (T-221): the trip tool on, the station as where the trip starts; the
 * next click (a station or a place) is where it ends. */
function tripFrom(id: string) {
  const at = stationPoint(id);
  if (!at) return;
  if (linking.peek() !== null) cancelLink();
  if (tool.peek() !== "trip") pickTool("trip");
  selection.value = { kind: "trip", from: at, to: null, fromStation: id };
}

/** Construct (while a blueprint) on the left and Remove on the right of one row. */
function ConstructRemove({ built, construct, constructLabel, remove, removeLabel, removeTip }: {
  built: boolean; construct: () => void; constructLabel: string; remove: () => void; removeLabel: string; removeTip: string;
}) {
  return (
    <Actions>
      {!built ? <button class="btn text" onClick={construct}>{constructLabel}</button> : <span />}
      <button class="btn text" onClick={remove} data-tip={removeTip}>{removeLabel}</button>
    </Actions>
  );
}

/** A trip from the station, and the link to another (T-221, T-197), with what linking asks. */
function StationTripLink({ id, members }: { id: string; members: string[] }) {
  const picking = linking.value === id;
  return (
    <>
      <Actions>
        {/* T-221: a trip from this station to the next station or place clicked */}
        <button class="btn text" onClick={() => tripFrom(id)} data-tip="Then click another station or a place to see how the demand model's people would travel there">
          Trip from here
        </button>
        <button class="btn text" onClick={() => (picking ? cancelLink() : startLink(id))} data-tip="Linked stations are one complex, and riders change between them on foot">
          {picking ? "Stop linking" : "Link to another station"}
        </button>
      </Actions>
      {picking && <p class="muted">Click another station on the map to link it with this one. Stations up to {roughDistance(LINK_MAX_M)} apart can be linked, and riders change between them on foot. Esc stops.</p>}
    </>
  );
}

/** The player's links (T-197, game/links.ts): those touching the station (or, in complex view,
 * any member), each with Unlink. Stations the distance rule groups are not listed: they cannot be
 * split. */
function StationLinks({ id, members }: { id: string; members: string[] }) {
  const w = world.value;
  if (!w) return null;
  const pairs = linksAmong(members);
  const byId = new Map(w.save.stations.map((s) => [s.id, s]));
  const end = (x: string) => (
    <>
      <span>{byId.get(x)?.name ?? ""}</span>
      {linesAt(x).map((l) => <Chip line={l} size="sm" />)}
    </>
  );
  return (
    <>
      {pairs.length > 0 && <div class="h">Linked by you</div>}
      {pairs.length > 0 && (
        <ul class="rows">
          {pairs.map(([a, b]) => (
            <li>
              <span class="grow" style={{ display: "flex", flexWrap: "wrap", alignItems: "center", gap: "4px" }}>{members.length > 1 ? <>{end(a)}<span>and</span>{end(b)}</> : end(a === id ? b : a)}</span>
              <button class="btn text" onClick={() => void unlink(a, b)}>Unlink</button>
            </li>
          ))}
        </ul>
      )}
    </>
  );
}

/** The station view's own rows and actions. */
function StationDetails({ s, lines, props }: {
  s: StationInput; lines: LineInput[];
  props: (platform: number) => void;
}) {
  // T-446: each line's boardings here in the past 24 hours (re-rendered as they move: a small part of the panel)
  const t = today.value;
  // a line may stop here twice (T-797: on both sides of a reversal): both visits count
  const visits = (l: LineInput) => l.stops.flatMap((x, k) => (x === s.id ? [k] : []));
  const here = lines.map((l) => [l, visits(l).reduce((n, k) => n + (t.get(l.id)?.stops[k] ?? 0), 0)] as const);
  const m = moneyState.value;
  return (
    <>
      <Built built={s.built} what="The station" />
      <ul class="rows">
        <li>
          <span class="grow">Platforms</span>
          <Stepper value={length(s.length)} wide label="platform length" onStep={(d) => props(Math.max(60, Math.min(400, s.length + d * 20)))} />
        </li>
        {s.built && m && (
          <Row label="Upkeep">
            <span data-tip={`Each constructed station costs ${money(stationUpkeep(m))} a day to keep up, at any elevation, whether trains stop or not. The lines stopping here share it by their trains a day.`}>{money(stationUpkeep(m))} a day</span>
          </Row>
        )}
      </ul>
      {lines.length > 0 && <div class="h">Lines<span class="right muted small">boardings in past 24 h</span></div>}
      <ul class="rows pick">
        {here.map(([l, b]) => (
          <li onClick={() => (selection.value = { kind: "line", line: l.id })}>
            <Chip line={l} />
            <span class="grow">{l.name}</span>
            <span class="val" data-tip={t.has(l.id) ? `${count(b)} boardings on ${l.name} in the past 24 hours` : undefined}>{t.has(l.id) ? count(b) : ""}</span>
          </li>
        ))}
      </ul>
      {lines.length > 0 && <BoardingsToday lines={lines} at={visits} />}
      <Waiting node={s.num} lines={lines} />
    </>
  );
}

/** Riders on the platforms now (T-164): per line and direction, from the boarding account, with
 * how many a full train left behind this hour and the longest wait for a train. */
function Waiting({ node, lines }: { node: number; lines: LineInput[] }) {
  clock.minute.value; // re-render each game minute: the crowds change
  // and the moment a train leaves one of its platforms with whoever fitted (T-238)
  const [, bump] = useState(0);
  const speed = clock.speed.value, running = clock.running.value, held = clock.held.value;
  const w = world.value;
  const r = live.overlay?.renderer;
  const epoch = r?.tripsEpoch ?? NaN;
  const next = r?.board ? nextLeavingAt(r.board, node, clock.now() - epoch) : Infinity;
  useEffect(() => {
    if (!Number.isFinite(next) || !clock.ticking) return;
    const ms = ((next - (clock.now() - epoch)) / SPEEDS[speed]) * 1000 + 5;
    const id = setTimeout(() => bump((x) => x + 1), Math.max(0, ms));
    return () => clearTimeout(id);
  }, [next, epoch, speed, running, held]);
  if (!w || !r?.board) return null;
  const byId = new Map(lines.map((l) => [l.id, l]));
  const rows = platformsAt(r.board, node, clock.now() - r.tripsEpoch).filter((p) => byId.has(p.line));
  if (!rows.length) return null;
  const name = new Map(w.save.stations.map((s) => [s.id, s.name]));
  return (
    <>
      <div class="h">
        Waiting now
      </div>
      <ul class="rows waiting">
        {rows.map((p) => {
          const l = byId.get(p.line)!;
          const to = name.get(p.run === 0 ? l.stops[l.stops.length - 1] : l.stops[0]);
          return (
            <li>
              <Chip line={l} />
              <span class="grow what">
                <span>To {to}</span>
                {p.leftBehind >= 1 ? (
                  <span class="bad small">{count(p.leftBehind)} left behind, longest wait {duration(p.longestWait)}</span>
                ) : (
                  <span class="muted small">{p.arriving >= 1 ? `${count(p.arriving)} arriving, longest wait ${duration(p.longestWait)}` : "Nobody arriving"}</span>
                )}
              </span>
              <span class="val num">{count(p.waiting)}</span>
            </li>
          );
        })}
      </ul>
    </>
  );
}

export function TrainInspector({ line, profile, dep }: { line: string; profile: number; dep: number }) {
  const w = world.value;
  clock.minute.value; // re-render each game minute: the train moves
  const l = w?.save.lines.find((l) => l.id === line);
  const net = live.overlay?.renderer.net;
  if (!w || !l || !net) return null;
  const st = w.lineStats[l.id];
  // a train standing at a terminal for its next departure is drawn past its trip (T-206)
  const ren = live.overlay!.renderer, ti = ren.tripIndex(profile, dep - ren.tripsEpoch);
  const at = tripAt(net, profile, dep, clock.now(), ti >= 0 && ren.tripEnds ? ren.tripEnds[ti] + ren.tripsEpoch : -Infinity);
  const name = new Map(w.save.stations.map((s) => [s.id, s.name]));
  const run = Math.floor(profile / 3) % 2;
  const stops = run === 0 ? l.stops : [...l.stops].reverse();
  const stopS = st.stopS[run] ?? [];
  const next = at ? stopS.findIndex((s) => s > at.s + 1) : -1;
  const len = net.meta[profile * 4 + 3];
  return (
    <>
      <div class="title">
        <Chip line={l} size="lg" />
        <span class="name">{l.name} train</span>
      </div>
      <div class="sub muted">Towards {name.get(stops[stops.length - 1])}</div>
      {at ? (
        <ul class="rows">
          <Row label="Next stop">{next >= 0 ? name.get(stops[next]) : "Terminus"}</Row>
          <Row label="Along the line">
            {distanceOf(at.s, len)}
          </Row>
        </ul>
      ) : (
        <p class="muted">This train finished its trip and left service.</p>
      )}
      {at && <OnBoard line={l} run={run} s={at.s} stopS={stopS} cars={st.cars} profile={profile} dep={dep} />}
    </>
  );
}

/** Seats and riders at crush load per 20 m car: a copy of `SEATS_PER_CAR` and `CRUSH_PER_CAR` in
 * sim/src/demand/api.rs (a New York subway car). */
const SEATS_PER_CAR = 44;
const CRUSH_PER_CAR = 160;

/**
 * Who is on board (T-083, T-164, SPEC 8): this train's own riders on the stretch it is running,
 * from the boarding account (game/boarding.ts), against seats and crush load; and the typical
 * train of the period, demand's riders on the stretch shared among the trains then.
 */
function OnBoard({ line, run, s, stopS, cars, profile, dep }: { line: LineInput; run: number; s: number; stopS: number[]; cars: number; profile: number; dep: number }) {
  const dm = lineDemand(line.id);
  const n = line.stops.length;
  // T-309: the period in the line's own local time
  const p = periodAt(clock.minute.value * 60 + (world.value?.lineStats[line.id]?.shift ?? 0));
  const tph = line.tph[PERIODS[p].demand];
  const trains = tph * (PERIODS[p].to - PERIODS[p].from);
  let seg = 0;
  while (seg + 1 < n - 1 && stopS[seg + 1] <= s + 1) seg++;
  const loads = dm?.loads[p];
  const r = live.overlay?.renderer;
  const i = r?.board ? r.tripIndex(profile, dep - r.tripsEpoch) : -1;
  const own = r?.board && i >= 0 ? tripLoad(r.board, i, run === 0 ? seg : n - 2 - seg) : null;
  if (!demandView.value || own === null || !loads || loads.length !== 2 * (n - 1) || trains <= 0)
    return (
      <ul class="rows">
        <Row label="Riders aboard">
          {/* T-324: a short state while the riders are being worked out (the train is drawn striped) */}
          <span class="muted">{loadFailure.value ? "Not available" : "Calculating"}</span>
        </Row>
      </ul>
    );
  const atEnd = s >= stopS[n - 1] - 1;
  // the typical train this hour on this stretch, from the same account as this one (T-230): its
  // line's trains in this direction drawn this hour
  let sum = 0, many = 0;
  const b = r!.board!, stretch = run === 0 ? seg : n - 2 - seg;
  for (let k = 0; k < r!.tripCount; k++) {
    const pk = r!.trips[k * 2];
    if (Math.floor(pk / 6) !== Math.floor(profile / 6) || Math.floor(pk / 3) % 2 !== run) continue;
    const v = tripLoad(b, k, stretch);
    if (v !== null) { sum += v; many++; }
  }
  const average = atEnd || !many ? 0 : sum / many;
  const aboard = atEnd ? 0 : own;
  const seats = SEATS_PER_CAR * cars, crush = CRUSH_PER_CAR * cars;
  const state = aboard <= 0.5 ? "Empty" : aboard <= seats ? "Seats free" : aboard < crush - 0.5 ? "Standing" : "Full";
  return (
    <>
      <div class="h">
        <span data-tip="Counted train by train from the demand forecast: riders board at the forecast's rate until the train is full">On board</span><span class="right muted small">{PERIODS[p].name.toLowerCase()}{dm?.estimate && <span data-tip="First estimate for this new or changed line, refined when demand finishes updating">, estimate</span>}</span>
      </div>
      <ul class="rows">
        <Row label="Riders aboard" bold>{Math.round(aboard).toLocaleString("en-US")}</Row>
        <Row label="Typical train this hour">{Math.round(average).toLocaleString("en-US")}</Row>
        <li>
          <span class="grow">
            <span class="load-bar" data-tip={`${Math.round(aboard).toLocaleString("en-US")} riders aboard. The bar is full at ${crush.toLocaleString("en-US")} riders, and the tick marks the ${seats.toLocaleString("en-US")} seats.`}>
              <i class="fill" style={{ width: Math.min(100, (aboard / crush) * 100) + "%" }} />
              <i class="seats" style={{ left: (seats / crush) * 100 + "%" }} />
            </span>
          </span>
          <span class="val">{state}</span>
        </li>
        <Row label="Seats">{seats.toLocaleString("en-US")}</Row>
        <Row label="Full at">{crush.toLocaleString("en-US")}</Row>
      </ul>
      {!atEnd && dm && <TrainStops line={line} dm={dm} board={b} trip={i} run={run} seg={seg} stopS={stopS} profile={profile} dep={dep} />}
    </>
  );
}

/** Radii the curve stepper walks through, m; past the last is auto (SPEC 6.1). */
const RADII = [100, 150, 200, 250, 300, 400, 500, 600, 800, 1000, 1200, 1500, 1796];

/** A blueprint stretch's shape (T-064, T-092): a row per corner (PI) with its curve's speed and a
 * radius stepper (auto, or set), and the stretch's level. Dragging the corners is on the map
 * (map/edgeEdit.ts). */
function Shape({ edge, version }: { edge: number; version: number }) {
  const [shape, setShape] = useState<EdgePis | null>(null);
  const [busy, setBusy] = useState(false);
  useEffect(() => {
    let on = true;
    setShape(null);
    void queryClock({ what: "edgePis", edge }).then((d) => on && d && setShape(parseEdgePis(d)));
    return () => {
      on = false;
    };
  }, [edge, version]);
  if (!shape) return null;
  const pis = shape.pis;
  const w = world.value;
  const track = w?.edges.find(e => e.id === edge);
  const from = track ? w?.nodes.get(track.a) : undefined, to = track ? w?.nodes.get(track.b) : undefined;
  if (!from || !to) return null;
  const put = async (next: EdgePi[]) => {
    if (busy) return;
    setBusy(true);
    try { await live.edgeEditor?.setPis(edge, next); }
    finally { setBusy(false); }
  };
  const add = (i: number) => {
    const a = i === 0 ? from : pis[i - 1], b = i === pis.length ? to : pis[i];
    const x = (a.x + b.x) / 2, y = (a.y + b.y) / 2;
    const p: EdgePi = { x, y, level: Math.round((a.level + b.level) / 2), radius: 0, used: 0, kmh: 0, mx: x, my: y };
    void put([...pis.slice(0, i), p, ...pis.slice(i)]);
  };
  const step = (i: number, d: number) => {
    const p = pis[i];
    const cur = p.radius > 0 ? p.radius : Infinity;
    let r: number;
    if (d < 0) r = [...RADII].reverse().find((x) => x < Math.min(cur, p.used > 0 ? p.used + 1 : cur)) ?? RADII[0];
    else r = RADII.find((x) => x > cur) ?? 0;
    void put(pis.map((q, k) => (k === i ? { ...q, radius: r >= RADII[RADII.length - 1] ? 0 : r } : q)));
  };
  // T-841: a straight piece has no corners, so its elevation is its ends', moved by edgesLevel as in the selection bar
  const level = pis.length ? (pis.every((p) => p.level === pis[0].level) ? pis[0].level : null) : track && track.levelMin === track.levelMax ? track.levelMin : null;
  const setLevel = async (l: Level) => {
    if (pis.length) return void edit({ op: "edgeLevel", edge, level: l });
    const a = await edit({ op: "edgesLevel", edges: [edge], level: l }, true);
    if (!a.ok) say(`This track cannot go to elevation ${levelLabel(l)}.`, true);
  };
  return (
    <>
      <div class="h">
        <span data-tip={"Add a corner between points, then drag its square on the map." + (shape.legacy ? " This stretch runs through its squares; once you change it, they become corners like on any other track." : "")}>Corners</span>
      </div>
      <div class="blueprint-corners">
        {Array.from({ length: pis.length + 1 }, (_, i) => <>
          <button class="btn text add-corner" disabled={busy} onClick={() => add(i)}
            aria-label={pis.length ? `Add corner ${i + 1}` : "Add corner"}>Add corner</button>
          {i < pis.length && <fieldset class="corner" disabled={busy}>
            <legend>Corner {i + 1}</legend>
            <div class="corner-row">
              <span class="grow">
                {pis[i].used > 0 ? speed(pis[i].kmh) : "No curve"}
                {pis[i].radius > 0 ? "" : <span class="muted"> auto</span>}
              </span>
              <Stepper value={pis[i].used > 0 ? length(pis[i].used) : "none"} wide label={`corner ${i + 1} radius`} onStep={(d) => step(i, d)} />
            </div>
            <div class="corner-row">
              <span class="grow">Elevation</span>
              <Stepper value={levelLabel(pis[i].level)} label={`corner ${i + 1} elevation`} onStep={d => {
                const next = Math.max(-3, Math.min(3, pis[i].level + d));
                if (next !== pis[i].level) void put(pis.map((p, k) => k === i ? { ...p, level: next } : p));
              }} />
              <button class="btn text" aria-label={`Remove corner ${i + 1}`} onClick={() => void put(pis.filter((_, k) => k !== i))}>Remove</button>
            </div>
          </fieldset>}
        </>)}
      </div>
      <div class="h">Elevation of the stretch</div>
      <div class="group levels">
        {[3, 2, 1, 0, -1, -2, -3].map((l) => (
          <button disabled={busy} class={"btn" + (l === level ? " on" : "")} onClick={() => void setLevel(l as Level)}>
            {levelLabel(l)}
          </button>
        ))}
      </div>
    </>
  );
}

/** Double up (T-708): an express pair beside this double track as far as its run goes, as
 * blueprint. The hover gives the price or why not. */
function DoubleUp({ edge, version }: { edge: number; version: number }) {
  const [answer, setAnswer] = useState<{ price: number } | { why: string } | null>(null);
  useEffect(() => {
    let on = true;
    setAnswer(null);
    void queryClock({ what: "doubleUp", edge }).then((d) => {
      if (!on || !d) return;
      setAnswer(d[0] >= 0 ? { price: d[0] } : { why: describeIssues(String.fromCharCode(...d.subarray(1)).split("\n").filter(Boolean)) });
    });
    return () => {
      on = false;
    };
  }, [edge, version]);
  if (!answer) return null;
  const tip = "price" in answer ? `Double up to quad track for ${money(answer.price * 1e6)} (blueprint)\nThe express pair runs beside it to the next junction or track end each way\nThe price includes a crossover past each station where one fits` : answer.why;
  return (
    <button class="btn icon" disabled={"why" in answer} data-tip={tip} aria-label="Double up" onClick={() => void edit({ op: "doubleUp", edge })}>
      <svg viewBox="0 0 16 16"><path d="M1.5 3h13M1.5 6h13M8 9v6M5 12h6" /></svg>
    </button>
  );
}

export function TrackInspector({ edge }: { edge: number }) {
  const w = world.value;
  const e = w?.edges.find((e) => e.id === edge);
  if (!w || !e) return null;
  const lines = w.save.lines.filter((l) => e.lines.includes(l.id));
  const lv = e.levelMin === e.levelMax ? `Elevation ${levelLabel(e.levelMin)}` : `Elevation ${levelLabel(e.levelMin)} to ${levelLabel(e.levelMax)}`;
  // a crossover is one thing: both diagonals are priced, built and removed together (T-391)
  const other = e.crossover ? w.edges.find((x) => x.id === e.partner) : undefined;
  const cost = e.cost + (other?.cost ?? 0);
  const m = moneyState.value;
  const construct = async () => {
    const a = await edit({ op: "construct", edges: [e.id], stations: [] });
    if (a.ok) say(`${e.crossover ? "Crossover" : "Track"} constructed for ${money(a.charge * 1e6)}.`);
  };
  // constructed: asks first above the bottom bar (T-096)
  const remove = () => void removeThing({ kind: "track", edge: e.id });
  return (
    <>
      <div class="title">
        <span class="name">{e.crossover ? "Crossover" : e.hsr ? "High-speed track" : e.tracks === 2 ? "Double track" : "Single track"}</span>
        {e.tracks === 2 && !e.crossover && !e.hsr && <DoubleUp edge={e.id} version={w.version} />}
      </div>
      <div class="sub muted">
        {distance(e.lengthM)}, {lv}
      </div>
      <ConstructRemove built={e.built} construct={construct} constructLabel={`Construct for ${money(cost)}`} remove={remove}
        removeLabel={`Remove ${e.crossover ? "crossover" : "track"}`} removeTip={e.built ? `Removing a constructed ${e.crossover ? "crossover" : "track"} refunds nothing` : ""} />
      <Built built={e.built} what={e.crossover ? "This crossover" : "This track"} />
      {!e.built && e.off && <p class="muted" data-tip={`Construct leaves ${e.crossover ? "this crossover" : "this track"} out until it is enabled`}>Disabled</p>}
      <ul class="rows">
        {!e.crossover && (
          <li data-tip={e.hsr ? "High-speed trains run up to 320 km/h here and ordinary trains up to 160 km/h" : "Trains run up to 160 km/h here. Over 160 needs high-speed track."}>
            <span class="grow">Speed limit</span>
            <span class="val">{speed(e.hsr ? 320 : 160)}</span>
          </li>
        )}
        <Row label="Slowest curve">{speed(e.vminKmh)}</Row>
        <Row label={e.built ? "Cost to build" : "Cost to construct"}>{money(cost)}</Row>
        {!e.crossover && m && (
          <Row label="Upkeep">
            <span data-tip={`${e.built ? "Upkeep" : "Once constructed, upkeep"} is ${upkeepRule(m)}, whether trains run or not. High-speed track costs ${m.upkeepHsr} times as much and single track ${m.upkeepSingle} times. The lines here share it by their trains a day.`}>
              {money(edgeUpkeep({ ...e, built: true }, m))} a day
            </span>
          </Row>
        )}
      </ul>
      {!e.built && !e.crossover && <Shape edge={e.id} version={w.version} />}
      {lines.length > 0 && <div class="h">Lines here</div>}
      {lines.length === 0 && (
        <div class="h">
          Lines here
          <span class="right">
            <button class="btn icon" data-tip="New line, starting at the stations on this track (N)" aria-label="New line" onClick={() => void createLine(e)}><Icon.plus /></button>
          </span>
        </div>
      )}
      <ul class="rows pick">
        {lines.map((l) => (
          <li onClick={() => (selection.value = { kind: "line", line: l.id })}>
            <Chip line={l} />
            <span class="grow">{l.name}</span>
          </li>
        ))}
      </ul>
    </>
  );
}

interface JPort {
  edge: number;
  end: number;
  dx: number;
  dy: number;
  name: string;
}
interface JMove {
  from: number;
  to: number;
  tph: number[];
  delay: number[];
  rho: number[];
  lines: number[];
  crosses: number[];
}

/** `TrackApi.junction_info` (T-031) in objects, with each edge end named by the station it leads
 * to, else its compass direction. */
function parseJunction(d: Float64Array): { ports: JPort[]; moves: JMove[] } {
  const ports: JPort[] = [];
  let i = 1;
  for (let k = 0; k < d[0]; k++, i += 4) {
    const [edge, end, dx, dy] = [d[i], d[i + 1], d[i + 2], d[i + 3]];
    ports.push({ edge, end, dx, dy, name: stationTowards(edge, end) ?? compass(dx, dy) });
  }
  // the same name twice (two ends towards one station): add the direction
  for (const p of ports) if (ports.some((q) => q !== p && q.name === p.name)) p.name += ` (${compass(p.dx, p.dy)})`;
  const moves: JMove[] = [];
  const nm = d[i++] ?? 0;
  for (let k = 0; k < nm; k++) {
    const m: JMove = { from: d[i], to: d[i + 1], tph: [d[i + 2], d[i + 3], d[i + 4]], delay: [d[i + 5], d[i + 6], d[i + 7]], rho: [d[i + 8], d[i + 9], d[i + 10]], lines: [], crosses: [] };
    i += 11;
    const nl = d[i++];
    for (let j = 0; j < nl; j++) m.lines.push(d[i++]);
    const nc = d[i++];
    for (let j = 0; j < nc; j++) m.crosses.push(d[i++]);
    moves.push(m);
  }
  return { ports, moves };
}

/** The junction from above: each track leaving it, and the moves as curves through the middle,
 * in the first line's colour; a move that crosses another is drawn thicker. */
function JunctionMap({ ports, moves, colourOf, tip }: { ports: JPort[]; moves: JMove[]; colourOf: (line: number) => string; tip: (m: JMove) => string }) {
  const R = 46, c = 56;
  // Branches leave almost along the main line; spread close ends to at least 30 degrees apart so
  // each can be told apart (a diagram, not a map).
  const ang = ports.map((p) => Math.atan2(p.dy, p.dx));
  const MIN = Math.PI / 6;
  for (let pass = 0; pass < 4; pass++)
    for (let i = 0; i < ang.length; i++)
      for (let j = i + 1; j < ang.length; j++) {
        let d = ang[j] - ang[i];
        d = Math.atan2(Math.sin(d), Math.cos(d));
        if (Math.abs(d) < MIN) {
          const push = (MIN - Math.abs(d)) / 2 * (d >= 0 ? 1 : -1);
          ang[i] -= push;
          ang[j] += push;
        }
      }
  const end = (p: JPort): [number, number] => {
    const a = ang[ports.indexOf(p)];
    return [c + Math.cos(a) * R, c - Math.sin(a) * R];
  };
  return (
    <svg class="junction-map" width={c * 2} height={c * 2} viewBox={`0 0 ${c * 2} ${c * 2}`}>
      {ports.map((p) => {
        const [x, y] = end(p);
        return <line x1={c} y1={c} x2={x} y2={y} stroke="var(--muted)" stroke-opacity="0.3" stroke-width="7" stroke-linecap="round" />;
      })}
      {moves.map((m) => {
        const [x1, y1] = end(ports[m.from]);
        const [x2, y2] = end(ports[m.to]);
        return <path d={`M${x1},${y1} Q${c},${c} ${x2},${y2}`} fill="none" stroke={colourOf(m.lines[0])} stroke-width={m.crosses.length ? 3 : 1.5} data-tip={tip(m)} />;
      })}
      <circle cx={c} cy={c} r="3" fill="var(--text)" />
    </svg>
  );
}

export function JunctionInspector({ node }: { node: number }) {
  const w = world.value;
  const [info, setInfo] = useState<{ ports: JPort[]; moves: JMove[] } | null>(null);
  const [price, setPrice] = useState<number | null>(null);
  const demand = clock.demand.value;
  const lev = DEMAND_INDEX[demand];
  useEffect(() => {
    let on = true;
    void queryClock({ what: "junction", node }).then((d) => on && d && setInfo(parseJunction(d)));
    void queryClock({ what: "flyover", node }).then((d) => on && d && setPrice(d[0]));
    return () => {
      on = false;
    };
  }, [node, w?.version]);
  const n = w?.nodes.get(node);
  if (!w || !n) return null;
  const built = n.builtPorts >= 3;
  // a crossover's end keeps its shape, so no flyover there (T-391)
  const atCrossover = w.edges.some((e) => e.crossover && (e.a === node || e.b === node));
  const setFlying = async (flying: boolean) => {
    const a = await edit({ op: "flying", node, flying });
    if (a.ok && flying) say(a.charge > 0 ? `Flyover built for ${money(a.charge * 1e6)}.` : "Flyover added to the blueprint.");
  };
  const lineOf = new Map(w.save.lines.map((l) => [l.num, l]));
  const colourOf = (id: number) => lineOf.get(id)?.colour ?? "#888888";
  const moves = info?.moves ?? [];
  const label = (m: JMove) => `${info!.ports[m.from].name} to ${info!.ports[m.to].name}`;
  const worst = moves.reduce((a, m) => Math.max(a, m.rho[lev]), 0);
  const crossing = moves.some((m) => m.crosses.length);
  // The waiting a flyover saves: a flying junction has no crossing paths, so all the waiting at
  // the junction goes (SPEC 6.2). Per train through it, at the demand level in force.
  const trains = moves.reduce((a, m) => a + m.tph[lev], 0);
  const waitPerHour = moves.reduce((a, m) => a + m.tph[lev] * m.delay[lev], 0);
  const perTrain = trains > 0 ? waitPerHour / trains : 0;
  return (
    <>
      <div class="title">
        <span class="name">Junction</span>
      </div>
      <div class="sub muted">
        Elevation {levelLabel(n.level)}, {n.ports} tracks meet{n.flying ? ", with a flyover" : ""}
      </div>
      {n.flying ? (!built && (
        <Actions>
          <span />
          <button class="btn text" onClick={() => setFlying(false)}>
            Remove the flyover
          </button>
        </Actions>
      )) : !atCrossover && (
        <Actions>
          <span class="muted small">{built ? "Paid at once" : "Added to the blueprint"}</span>
          <button class="btn text" data-tip="Take one track over the other, so trains never wait to cross" onClick={() => setFlying(true)}>
            Build a flyover
          </button>
        </Actions>
      )}
      <Built built={built} what="The junction" />
      {!n.flying && (
        <>
          <p class="note muted">{crossing ? "Trains cross on the level here and wait for each other." : "No trains cross here now."}</p>
          {!atCrossover && (
            <ul class="rows">
              <Row label="Flyover price">{price === null ? "" : money(price * 1e6)}</Row>
              <Row label="Waiting it saves">{perTrain >= 1 ? `${duration(perTrain)} a train at ${demand} demand` : "none now"}</Row>
            </ul>
          )}
        </>
      )}
      {info && moves.length > 0 && (
        <>
          <JunctionMap ports={info.ports} moves={moves} colourOf={colourOf}
            tip={(m) => `${label(m)}: ${Math.round(m.tph[lev])} trains an hour at ${demand} demand${m.crosses.length ? "\nIt crosses another route on the level, so it is drawn thicker" : ""}`} />
          <div class="h">
            Routes through it<span class="right muted small">trains an hour, wait per train</span>
          </div>
          <ul class="rows moves">
            {moves.map((m) => (
              <li>
                <span class="grow what">
                  <span>
                    {m.lines.map((id) => lineOf.get(id) && <Chip line={lineOf.get(id)!} size="sm" />)} {label(m)}
                  </span>
                  {m.crosses.length > 0 && <span class="muted small">Crosses {m.crosses.map((k) => label(moves[k])).join(", ")}</span>}
                </span>
                <span class="val num">{Math.round(m.tph[lev])}</span>
                <span class="val num w-num">{m.delay[lev] > 0 ? "+" + duration(m.delay[lev]) : ""}</span>
              </li>
            ))}
          </ul>
          {crossing && (
            <ul class="rows">
              <Row label="Busiest crossing">{Math.round(worst * 100)}% of capacity</Row>
            </ul>
          )}
        </>
      )}
      {info && moves.length === 0 && <p class="note muted">No running line goes through it.</p>}
    </>
  );
}

function Placeholder({ title, sub, text }: { title: string; sub: string; text: string }) {
  return (
    <>
      <div class="title">
        <span class="name">{title}</span>
      </div>
      <div class="sub muted">{sub}</div>
      <p class="note muted">{text}</p>
    </>
  );
}

export function AreaInspector({ sel }: { sel: Exclude<Selection, null> }) {
  if (sel.kind === "cell") return <Placeholder title="Cell" sub={sel.cell} text="People, jobs and trips by train, once the city pack is in the game." />;
  if (sel.kind === "zone") return <Placeholder title="Zone" sub={`Zone ${sel.zone}`} text="Trips to and from this zone, once the city pack is in the game." />;
  return null;
}
