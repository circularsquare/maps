// The station inspector (SPEC 8): a station on its own, or its complex.

import { useEffect, useState } from "preact/hooks";
import { clock } from "../../game/clock";
import { t as tr, tList, tNumber, tSeries } from "../../i18n";
import { lineDemand } from "../../game/demand";
import { area, count, distance, duration, length, levelLabel, money, roughDistance } from "../../game/format";
import { groundDist, groundScale } from "../../game/coords";
import { outlineArea } from "../../game/buildings";
import { buildingMembersById, buildingOf, hallWaiting } from "../../game/buildingView";
import { hallRoom } from "../../game/buildingLayout";
import { platformOffsets } from "../../game/complexes";
import { passagePrice } from "../../game/upgrades";
import type { LineInput, StationInput } from "../../game/types";
import { complexOf, inComplexView, openAlone } from "../../game/complexView";
import { complexNow, type ComplexNow } from "../../game/complexLive";
import { LINK_MAX_M } from "../../game/complexes";
import { linking, linksAmong, unlink } from "../../game/links";
import { selection } from "../../game/ui";
import { edit, linesAt, say, stationNames, stationsById, world } from "../../game/world";
import { live } from "../../map/live";
import { removeThing } from "../../game/remove";
import { nextLeavingAt, platformsAt, stepPlatform } from "../../game/boarding";
import { today, useToday, type LineToday } from "../../game/today";
import { moneyState } from "../../game/money";
import { stationUpkeep } from "../../game/regionMoney";
import { Actions, Built, Chip, Icon, Row, Stepper } from "../widgets";
import { nextSuggestion } from "../../map/stationNames";
import { NameEdit } from "./NameEdit";
import { locateStation } from "../locate";
import { ComplexMap } from "./ComplexMap";
import { ClosestStations } from "./ClosestStations";
import { PurposeSplit } from "./PurposeSplit";
import { StationUpgrades } from "./StationUpgrades";

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
    const w = world.peek(), stopNames = w ? stationNames(w) : null; // a bus line may end at a bus stop (T-1464)
    const ends = [...new Set(ls.flatMap((l) => [l.stops[0], l.stops[l.stops.length - 1]]).filter((s) => !inside.has(s)).map((s) => stopNames?.get(s) ?? byId.get(s)?.name ?? ""))];
    rows.push({
      name, line: ls[0], stations, open, built: stations.some((m) => byId.get(m)?.built), waiting, full, boardings, today: todayB,
      tip: tr("stationInspector.rowTip", { name, stations: tList(names) }) + (ends.length ? "\n" + tr("stationInspector.rowTo", { ends: tSeries(ends) }) : ""),
      waitTip: waiting === null ? tr("stationInspector.noTrain")
        : tr("stationInspector.waitingFor", { count: count(waiting), name }) + (full ? "\n" + tr("stationInspector.leftBehind") : ""),
    });
  }
  rows.sort((a, b) => b.boardings - a.boardings || a.name.localeCompare(b.name));
  for (const m of members) {
    if (lines.some((l) => l.stops.includes(m))) continue;
    const s = byId.get(m);
    if (s) rows.push({ name: s.name, line: null, stations: [m], open: m, built: s.built, waiting: null, full: false, boardings: 0, today: null, tip: `${s.name}\n${tr("complexMap.noLines")}`, waitTip: "" });
  }
  return rows;
}

/** The complex now (game/complexLive.ts): worked out again every 10 game seconds or so (at most
 * every 150 ms of real time), and re-rendered only when something shown changed. */
function useComplexNow(members: string[]): ComplexNow | null {
  const [now, setNow] = useState<ComplexNow | null>(() => complexNow(members));
  const scope = members.join(",");
  useEffect(() => {
    let key = "", t = NaN, board: unknown = null;
    const tick = () => {
      const r = live.overlay?.renderer, at = clock.now();
      if (Math.abs(at - t) < 10 && r?.board === board) return;
      t = at;
      board = r?.board ?? null;
      const c = complexNow(members);
      const k = c?.key ?? "";
      if (k !== key || !c) {
        key = k;
        setNow(c);
      }
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
        <span class="grow">{tr("stationInspector.lines")}</span>
        <span class="val">{tr("stationInspector.waiting")}</span>
        <span class="val">{tr("stationInspector.today")}</span>
      </li>
      {rows.map((r) => (
        <li class={r.stations.includes(id) ? "on" : ""} onClick={() => openStation(r.open)} data-tip={r.tip}>
          {r.line && <Chip line={r.line} size="sm" />}
          <span class={"grow" + (r.built ? "" : " muted")}>{r.name}</span>
          <span class={"val" + (r.full ? " bad" : "")} data-tip={r.waitTip}>{r.waiting !== null ? count(r.waiting) : ""}</span>
          <span class="val" data-tip={r.today !== null ? tr("stationInspector.boardings24", { count: count(r.today) }) : undefined}>{r.today !== null ? count(r.today) : ""}</span>
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
        <Row label={tr("stationInspector.boardingsToday")} bold>
          {known ? count(total) : ""}
        </Row>
        <PurposeSplit scope={lines.map((l) => ({ line: l.id, at: at(l) }))} unit={tr("stationInspector.boardingsUnit")} />
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
  const lines = linesAt(id);
  const props = (platform: number, name = s.name) => edit({ op: "stationProps", node: s.num, platform, name }).then((a) => a.ok);
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
      else say(tr("stationInspector.noName"));
    } finally {
      setCycling(false);
    }
  };
  const construct = async () => {
    const a = await edit({ op: "construct", edges: [], stations: [s.num] });
    if (a.ok) say(tr("stationInspector.constructed", { name: s.name, price: money(a.charge * 1e6) }));
  };
  // constructed: asks first above the bottom bar (T-096)
  const remove = () => void removeThing({ kind: "station", node: s.num });
  return (
    <>
      <div class="title">
        <span class="st-glyph" />
        <NameEdit key={id} name={s.name} onRename={(n) => props(s.length, n)}>
          <button class="btn icon" data-tip={tr("stationInspector.nextName")} aria-label={tr("stationInspector.nextName")} onClick={() => void cycle()}>
            <Icon.cycle />
          </button>
          <button class="btn icon" data-tip={tr("lineTitle.show")} aria-label={tr("lineTitle.show")} onClick={() => locateStation(s.lng, s.lat)}>
            <Icon.locate />
          </button>
        </NameEdit>
      </div>
      {/* a click on a station always shows its complex; a station opened on its own from the
          complex's list goes back to it from here */}
      {!asComplex && (
        <div class="sub muted">
          {s.port ? tr("stationInspector.port") : tr("trackInspector.level", { level: levelLabel(s.level) })}
          {members.length > 1 && (
            <>
              {" · "}
              <button class="btn text inline" onClick={() => { selection.value = { kind: "station", station: id }; live.overlay?.invalidate(); }} data-tip={tr("stationInspector.wholeComplex")}>
                {tr("stationInspector.complexOf", { count: members.length })}
              </button>
            </>
          )}
        </div>
      )}
      {linking.value === id && <p class="muted">{tr("stationInspector.linking", { distance: roughDistance(LINK_MAX_M) })}</p>}
      {!asComplex && !s.built && <Actions><button class="btn text" onClick={construct}>{tr(s.port ? "stationInspector.constructPort" : "stationInspector.constructStation")}</button></Actions>}
      {/* T-2194: a port has no platform to upgrade */}
      {!s.port && <div class="h">{tr("stationUpgrades.group")}</div>}
      {!s.port && <StationUpgrades s={s} members={asComplex ? members : undefined} />}
      {asComplex ? (
        <ComplexSummary id={id} members={members} />
      ) : (
        <StationDetails s={s} lines={lines} props={props} />
      )}
      <ClosestStations lng={s.lng} lat={s.lat} exclude={new Set(members)} own={s.name} />
      <StationLinks id={id} members={asComplex ? members : [id]} />
      {!s.port && <StationBuilding id={id} members={members} />}
      {!asComplex && <Actions><button class="btn text danger" onClick={remove}>{tr(s.port ? "stationInspector.removePort" : "stationInspector.removeStation")}</button></Actions>}
    </>
  );
}

/** The passages (the player's links, T-197, T-1281; game/links.ts) touching the station (or, in
 * complex view, any member): length and price, each with Remove. Stations the distance rule
 * groups are not listed: they cannot be split. */
function StationLinks({ id, members }: { id: string; members: string[] }) {
  const w = world.value;
  if (!w) return null;
  const pairs = linksAmong(members);
  const byId = stationsById(w);
  const end = (x: string) => (
    <>
      <span>{byId.get(x)?.name ?? ""}</span>
      <span class="bullets">{linesAt(x).map((l) => <Chip line={l} size="sm" />)}</span>
    </>
  );
  const length = (a: string, b: string) => {
    const sa = byId.get(a), sb = byId.get(b);
    return sa && sb ? groundDist(sa.x, sa.y, sb.x, sb.y) : 0;
  };
  return (
    <>
      {pairs.length > 0 && <div class="h">{tr("stationInspector.passages")}</div>}
      {pairs.length > 0 && (
        <ul class="rows">
          {pairs.map(([a, b]) => {
            const m = length(a, b), built = byId.get(a)?.built && byId.get(b)?.built;
            return (
              <li>
                <span class="grow chip-line">{members.length > 1 ? <>{end(a)}<span>{tr("stationInspector.and")}</span>{end(b)}</> : end(a === id ? b : a)}</span>
                <span class="val muted" data-tip={tr("stationInspector.linkTip", { distance: distance(m), price: money(passagePrice(m) * 1e6), built: !!built })}>{distance(m)}</span>
                <button class="btn text" data-tip={built ? tr("stationInspector.noRefund") : undefined} onClick={() => void unlink(a, b)}>{tr("stationInspector.removeLink")}</button>
              </li>
            );
          })}
        </ul>
      )}
    </>
  );
}

/** T-2218: the station building the complex belongs to: its name, floor, Edit outline and Remove building. */
function StationBuilding({ id, members }: { id: string; members: string[] }) {
  const b = buildingOf(members);
  if (!b) return null;
  const n = b.corners.length / 2;
  let my = 0;
  for (let i = 0; i < n; i++) my += b.corners[2 * i + 1] / n;
  const k = groundScale(my), m2 = outlineArea(b.corners) / (k * k);
  // T-2219: the hall's room and, while platform crowds are drawn, the riders waiting in it
  const w = world.value, byId = w ? stationsById(w) : new Map<string, StationInput>(), inside = buildingMembersById.value.get(b.id) ?? [];
  const room = roomOf(b.corners, k, inside.map((m) => byId.get(m)).filter((s): s is StationInput => !!s));
  const waiting = hallWaiting.value.get(b.id) ?? 0;
  const whole = (n: number) => tNumber(Math.round(n));
  return (
    <>
      <div class="h">{tr("stationUpgrades.building")}</div>
      <ul class="rows">
        <li class="building-name"><NameEdit key={b.id} name={b.name} onRename={(name) => void edit({ op: "renameBuilding", id: b.id, name })} /></li>
        <li><span class="grow">{tr("stationInspector.area")}</span><span class="val">{area(m2)}</span></li>
        {inside.length > 0 && <li><span class="grow">{tr("stationInspector.room")}</span><span class="val" data-tip={tr("stationInspector.roomTip")}>{tr("stationInspector.roomFigure", { figure: waiting >= 1 ? `${whole(waiting)}/${whole(room)}` : whole(room) })}</span></li>}
      </ul>
      <Actions>
        <button class="btn text" onClick={() => live.tools?.building.begin(b.corners, b.id, id)}>{tr("stationInspector.editOutline")}</button>
        <button class="btn text danger" onClick={() => void removeThing({ kind: "building", id: b.id })}>{tr("stationInspector.removeBuilding")}</button>
      </Actions>
    </>
  );
}

/** A building's hall room (game/buildingLayout.ts) from its members' platforms, kept for one outline and
 * set of platforms. */
let roomKept = { key: "", room: 0 };
function roomOf(corners: number[], k: number, members: StationInput[]): number {
  const lines = members.map((s) => {
    const o = platformOffsets(s, groundScale(s.y)), out: number[] = [];
    for (let i = 0; i + 1 < o.length; i += 2) out.push(s.x + o[i], s.y + o[i + 1]);
    return out;
  });
  const key = corners.join(",") + "|" + lines.map((l) => l.join(",")).join("|");
  if (key !== roomKept.key) roomKept = { key, room: Math.round(hallRoom(corners, k, lines) / 100) * 100 };
  return roomKept.room;
}

/** The station view's own rows and actions. */
function StationDetails({ s, lines, props }: {
  s: StationInput; lines: LineInput[];
  props: (platform: number) => Promise<boolean>;
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
        {!s.port && <li>
          <span class="grow">{tr("stationInspector.platforms")}</span>
          <Stepper value={length(s.length)} wide label={tr("stationInspector.platformLength")} onStep={(d) => {
            const to = stepPlatform(s.length, d);
            if (to === s.length) return false;
            return props(to);
          }} />
        </li>}
        {s.built && m && (
          <Row label={tr("trackInspector.upkeep")}>
            {s.port
              ? <span class="money-neg" data-tip={tr("stationInspector.portUpkeep", { price: money(m.upkeepPort * m.economy) })}>{tr("inspector.perDay", { value: money(m.upkeepPort * m.economy) })}</span>
              : <span class="money-neg" data-tip={tr("stationInspector.stationUpkeep", { under: money(m.upkeepStationUnder * m.economy), other: money(m.upkeepStation * m.economy) })}>{tr("inspector.perDay", { value: money(stationUpkeep(m, s.level)) })}</span>}
          </Row>
        )}
      </ul>
      {lines.length > 0 && <div class="h">{tr("stationInspector.lines")}<span class="right muted small">{tr("cityDetails.boardingsToday")}</span></div>}
      <ul class="rows pick">
        {here.map(([l, b]) => (
          <li onClick={() => (selection.value = { kind: "line", line: l.id })}>
            <Chip line={l} />
            <span class="grow">{l.name}</span>
            <span class="val" data-tip={t.has(l.id) ? tr("stationInspector.lineBoardings", { count: count(b), name: l.name }) : undefined}>{t.has(l.id) ? count(b) : ""}</span>
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
  const speed = clock.rateNow.value, running = clock.running.value, held = clock.held.value;
  const w = world.value;
  const r = live.overlay?.renderer;
  const epoch = r?.tripsEpoch ?? NaN;
  const next = r?.board ? nextLeavingAt(r.board, node, clock.now() - epoch) : Infinity;
  useEffect(() => {
    if (!Number.isFinite(next) || !clock.ticking) return;
    const ms = ((next - (clock.now() - epoch)) / speed) * 1000 + 5;
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
        {tr("stationInspector.waitingNow")}
      </div>
      <ul class="rows waiting">
        {rows.map((p) => {
          const l = byId.get(p.line)!;
          const to = name.get(p.run === 0 ? l.stops[l.stops.length - 1] : l.stops[0]);
          return (
            <li>
              <Chip line={l} />
              <span class="grow what">
                <span>{tr("stationInspector.to", { name: to ?? "" })}</span>
                {p.leftBehind >= 1 ? (
                  <span class="bad small">{tr("stationInspector.behind", { count: count(p.leftBehind), time: duration(p.longestWait) })}</span>
                ) : (
                  <span class="muted small">{p.arriving >= 1 ? tr("stationInspector.arriving", { count: count(p.arriving), time: duration(p.longestWait) }) : tr("stationInspector.nobody")}</span>
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
