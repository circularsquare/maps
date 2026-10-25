// The station inspector (SPEC 8): a station on its own, or its complex.

import { useEffect, useState } from "preact/hooks";
import { clock, SPEEDS } from "../../game/clock";
import { lineDemand } from "../../game/demand";
import { count, duration, length, levelLabel, money, roughDistance } from "../../game/format";
import type { LineInput, StationInput } from "../../game/types";
import { complexOf, inComplexView, openAlone } from "../../game/complexView";
import { complexNow, type ComplexNow } from "../../game/complexLive";
import { LINK_MAX_M } from "../../game/complexes";
import { cancelLink, linking, linksAmong, startLink, unlink } from "../../game/links";
import { selection, tool } from "../../game/ui";
import { stationPoint } from "../../game/trip";
import { pickTool } from "../toolKeys";
import { edit, linesAt, say, stationNames, stationsById, world } from "../../game/world";
import { live } from "../../map/live";
import { removeThing } from "../../game/remove";
import { nextLeavingAt, platformsAt, stepPlatform } from "../../game/boarding";
import { today, useToday, type LineToday } from "../../game/today";
import { moneyState } from "../../game/money";
import { stationUpkeep } from "../../game/regionMoney";
import { Actions, Built, Chip, ConstructRemove, Icon, Row, Stepper } from "../widgets";
import { nextSuggestion } from "../../map/stationNames";
import { NameEdit } from "./NameEdit";
import { locateStation } from "../locate";
import { StationRiders } from "./StationRiders";
import { StationLongerTrips } from "./LongerTrips";
import { ComplexMap } from "./ComplexMap";
import { ClosestStations } from "./ClosestStations";
import { PurposeSplit } from "./PurposeSplit";

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
  const byId = stationsById(w);
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
      <ul class="rows pick">
        <Row label="Boardings in past 24 h" bold>
          {known ? count(total) : ""}
        </Row>
        <PurposeSplit scope={lines.map((l) => ({ line: l.id, at: at(l) }))} unit="boardings" />
      </ul>
  );
}

export function StationInspector({ id }: { id: string }) {
  const w = world.value;
  const [cycling, setCycling] = useState(false);
  useToday(); // T-446: boardings in the past 24 hours, kept fresh while the panel is open
  const s = w && stationsById(w).get(id);
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
      const w = world.peek();
      const now = w && stationsById(w).get(id);
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
        <span class="st-glyph" />
        <NameEdit key={id} name={s.name} onRename={(n) => props(s.length, n)}>
          <button class="btn icon" data-tip="Next suggested name" aria-label="Next suggested name" onClick={() => void cycle()}>
            <Icon.cycle />
          </button>
          <button class="btn icon" data-tip="Show on the map" aria-label="Show on the map" onClick={() => locateStation(s.lng, s.lat)}>
            <Icon.locate />
          </button>
        </NameEdit>
      </div>
      {/* a click on a station always shows its complex; a station opened on its own from the
          complex's list goes back to it from here */}
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
      {!asComplex && <ConstructRemove built={s.built} construct={construct} constructLabel="Construct station" remove={remove} removeLabel="Remove station" removeTip={s.built ? "Nothing is refunded" : ""} />}
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
  const byId = stationsById(w);
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
              <span class="grow chip-line">{members.length > 1 ? <>{end(a)}<span>and</span>{end(b)}</> : end(a === id ? b : a)}</span>
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
      <Built built={s.built} />
      <ul class="rows">
        <li>
          <span class="grow">Platforms</span>
          <Stepper value={length(s.length)} wide label="platform length" onStep={(d) => props(stepPlatform(s.length, d))} />
        </li>
        {s.built && m && (
          <Row label="Upkeep">
            <span data-tip={`Each built station costs ${money(stationUpkeep(m))} a day to keep up, at any elevation, whether trains stop or not. The lines stopping here share it by their trains a day.`}>{money(stationUpkeep(m))} a day</span>
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
  void clock.minute.value; // re-render each game minute: the crowds change
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
  const name = stationNames(w);
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
