// Inspectors for a station, a train, a stretch of track and a junction, and placeholders for cell
// and zone (T-026's).

import { useEffect, useState } from "preact/hooks";
import { clock } from "../../game/clock";
import { demandView, lineDemand } from "../../game/demand";
import { PERIODS, periodAt } from "../../game/periods";
import { count, duration, levelLabel, money } from "../../game/format";
import { compass, stationTowards } from "../../game/places";
import { DEMAND_INDEX, type Level, type LineInput, type Selection } from "../../game/types";
import { queryClock } from "../../workers/clockClient";
import { parseEdgePis, type EdgePi, type EdgePis } from "../../map/edgeEdit";
import { display, selection } from "../../game/ui";
import { edit, linesAt, say, world } from "../../game/world";
import { live } from "../../map/live";
import { removeThing } from "../../game/remove";
import { tripAt } from "../../map/network";
import { Chip, Row, Stepper } from "../widgets";
import { StationRiders } from "./StationRiders";

function StationGlyph() {
  return <span class="st-glyph" />;
}

function Built({ built, what }: { built: boolean; what: string }) {
  return built ? null : <p class="muted">Blueprint. {what} is built when you construct it.</p>;
}

export function StationInspector({ id }: { id: string }) {
  const w = world.value;
  const [editing, setEditing] = useState(false);
  const s = w?.save.stations.find((s) => s.id === id);
  if (!w || !s) return null;
  const lines = linesAt(id);
  const here = lines.map((l) => [l, lineDemand(l.id)?.boardings[l.stops.indexOf(id)] ?? 0] as const);
  const total = here.reduce((a, [, b]) => a + b, 0);
  const props = (platform: number, name = s.name) => void edit({ op: "stationProps", node: s.num, platform, name });
  const commit = (e: Event) => {
    const n = (e.target as HTMLInputElement).value.trim();
    if (n && n !== s.name) props(s.length, n);
    setEditing(false);
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
        {editing ? (
          <input
            class="name-edit"
            value={s.name}
            ref={(el) => el?.focus()}
            onBlur={commit}
            onKeyDown={(e) => {
              if (e.key === "Enter") commit(e);
              if (e.key === "Escape") setEditing(false);
            }}
          />
        ) : (
          <span class="name" onDblClick={() => setEditing(true)} title="Double click to rename">
            {s.name}
          </span>
        )}
      </div>
      <div class="sub muted">Level {levelLabel(s.level)}</div>
      <Built built={s.built} what="The station" />
      <ul class="rows">
        <li>
          <span class="grow">Platforms</span>
          <Stepper value={s.length + " m"} wide label="platform length" onStep={(d) => props(Math.max(60, Math.min(400, s.length + d * 20)))} />
        </li>
      </ul>
      {lines.length > 0 && <div class="h">Lines</div>}
      <ul class="rows pick">
        {here.map(([l, b]) => (
          <li onClick={() => (selection.value = { kind: "line", line: l.id })}>
            <Chip line={l} />
            <span class="grow">{l.name}</span>
            <span class="val">{lineDemand(l.id) ? count(b) : ""}</span>
          </li>
        ))}
        {lines.length > 0 && (
          <Row label="Boardings a day" bold>
            {lines.some((l) => lineDemand(l.id)) ? count(total) : ""}
          </Row>
        )}
      </ul>
      <div class="pane-foot">
        {!s.built ? (
          <button class="btn text" onClick={construct}>
            Construct station
          </button>
        ) : (
          <span />
        )}
        <button class="btn text" onClick={remove} title={s.built ? "Removing a constructed station refunds nothing" : ""}>
          Remove station
        </button>
      </div>
      <StationRiders id={id} />
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
  const at = tripAt(net, profile, dep, clock.now());
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
            {(at.s / 1000).toFixed(1)} of {(len / 1000).toFixed(1)} km
          </Row>
        </ul>
      ) : (
        <p class="muted">This train has finished its trip.</p>
      )}
      {at && <OnBoard line={l} run={run} s={at.s} stopS={stopS} cars={st.cars} />}
    </>
  );
}

/** Seats and riders at crush load per 20 m car: a copy of `SEATS_PER_CAR` and `CRUSH_PER_CAR` in
 * sim/src/demand/api.rs (a New York subway car). */
const SEATS_PER_CAR = 44;
const CRUSH_PER_CAR = 160;

/**
 * Who is on a train now (T-083, SPEC 8): the riders demand puts on the segment it is running
 * (`LineDemand.loads`, riders over the whole period) shared among the trains the line runs in
 * that period, with both the average and busiest hour; against seats and crush load.
 */
function OnBoard({ line, run, s, stopS, cars }: { line: LineInput; run: number; s: number; stopS: number[]; cars: number }) {
  const dm = lineDemand(line.id);
  const n = line.stops.length;
  const p = periodAt(clock.minute.value * 60);
  const tph = line.tph[PERIODS[p].demand];
  const trains = tph * (PERIODS[p].to - PERIODS[p].from);
  let seg = 0;
  while (seg + 1 < n - 1 && stopS[seg + 1] <= s + 1) seg++;
  const loads = dm?.loads[p];
  if (!demandView.value || !loads || loads.length !== 2 * (n - 1) || trains <= 0)
    return (
      <ul class="rows">
        <Row label="On board">
          <span class="muted">{demandView.value ? "worked out after the next demand update" : "worked out once demand is ready"}</span>
        </Row>
      </ul>
    );
  const atEnd = s >= stopS[n - 1] - 1;
  const average = atEnd ? 0 : loads[run === 0 ? seg : n - 1 + seg] / trains;
  const busiest = average * (demandView.value.peakHourFactor[p] ?? 1);
  const basis = display.value.trainLoadBasis;
  const aboard = basis === "average" ? average : busiest;
  const seats = SEATS_PER_CAR * cars, crush = CRUSH_PER_CAR * cars;
  const state = aboard <= 0.5 ? "Empty" : aboard <= seats ? "Seats free" : aboard <= crush ? "Standing" : "Over full";
  return (
    <>
      <div class="h">
        On board<span class="right muted small">{PERIODS[p].name.toLowerCase()}, {basis === "average" ? "average" : "busiest hour"}</span>
      </div>
      <ul class="rows">
        <Row label="Average riders" bold={basis === "average"}>{Math.round(average).toLocaleString("en-US")}</Row>
        <Row label="Busiest hour" bold={basis === "busiest"}>{Math.round(busiest).toLocaleString("en-US")}</Row>
        <li>
          <span class="grow">
            <span class="load-bar">
              <i class="fill" style={{ width: Math.min(100, (aboard / crush) * 100) + "%" }} />
              <i class="seats" style={{ left: (seats / crush) * 100 + "%" }} />
            </span>
          </span>
          <span class="val">{state}</span>
        </li>
        <Row label="Seats">{seats.toLocaleString("en-US")}</Row>
        <Row label="Packed full">{crush.toLocaleString("en-US")}</Row>
      </ul>
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
  useEffect(() => {
    let on = true;
    void queryClock({ what: "edgePis", edge }).then((d) => on && d && setShape(parseEdgePis(d)));
    return () => {
      on = false;
    };
  }, [edge, version]);
  if (!shape) return null;
  const pis = shape.pis;
  if (!pis.length) return <p class="note muted">A straight stretch.</p>;
  const put = (next: EdgePi[]) => void live.edgeEditor?.setPis(edge, next);
  const step = (i: number, d: number) => {
    const p = pis[i];
    const cur = p.radius > 0 ? p.radius : Infinity;
    let r: number;
    if (d < 0) r = [...RADII].reverse().find((x) => x < Math.min(cur, p.used > 0 ? p.used + 1 : cur)) ?? RADII[0];
    else r = RADII.find((x) => x > cur) ?? 0;
    put(pis.map((q, k) => (k === i ? { ...q, radius: r >= RADII[RADII.length - 1] ? 0 : r } : q)));
  };
  const level = pis.every((p) => p.level === pis[0].level) ? pis[0].level : null;
  return (
    <>
      <div class="h">
        Curves<span class="right muted small">drag the squares on the map</span>
      </div>
      {shape.legacy && <p class="note muted">This stretch runs through its squares. Once you change it, they become corners like on any other track.</p>}
      <ul class="rows">
        {pis.map((p, i) => (
          <li>
            <span class="grow">
              {p.used > 0 ? `${Math.round(p.kmh)} km/h` : "No curve"}
              {p.radius > 0 ? "" : <span class="muted"> auto</span>}
            </span>
            <Stepper value={p.used > 0 ? Math.round(p.used).toLocaleString("en-US") + " m" : "none"} wide label="radius" onStep={(d) => step(i, d)} />
          </li>
        ))}
      </ul>
      <div class="h">Level of the stretch</div>
      <div class="group levels">
        {[3, 2, 1, 0, -1, -2, -3].map((l) => (
          <button class={"btn" + (l === level ? " on" : "")} onClick={() => void edit({ op: "edgeLevel", edge, level: l as Level })}>
            {levelLabel(l)}
          </button>
        ))}
      </div>
    </>
  );
}

export function TrackInspector({ edge }: { edge: number }) {
  const w = world.value;
  const e = w?.edges.find((e) => e.id === edge);
  if (!w || !e) return null;
  const lines = w.save.lines.filter((l) => e.lines.includes(l.id));
  const lv = e.levelMin === e.levelMax ? `Level ${levelLabel(e.levelMin)}` : `Levels ${levelLabel(e.levelMin)} to ${levelLabel(e.levelMax)}`;
  const construct = async () => {
    const a = await edit({ op: "construct", edges: [e.id], stations: [] });
    if (a.ok) say(`Track constructed for ${money(a.charge * 1e6)}.`);
  };
  // constructed: asks first above the bottom bar (T-096)
  const remove = () => void removeThing({ kind: "track", edge: e.id });
  return (
    <>
      <div class="title">
        <span class="name">{e.tracks === 2 ? "Double track" : "Single track"}</span>
      </div>
      <div class="sub muted">
        {(e.lengthM / 1000).toFixed(2)} km, {lv}
      </div>
      <Built built={e.built} what="This track" />
      <ul class="rows">
        <Row label="Slowest curve">{Math.round(e.vminKmh)} km/h</Row>
        <Row label={e.built ? "Cost to build" : "Cost to construct"}>{money(e.cost)}</Row>
      </ul>
      {!e.built && <Shape edge={e.id} version={w.version} />}
      {lines.length > 0 && <div class="h">Lines on it</div>}
      <ul class="rows pick">
        {lines.map((l) => (
          <li onClick={() => (selection.value = { kind: "line", line: l.id })}>
            <Chip line={l} />
            <span class="grow">{l.name}</span>
          </li>
        ))}
      </ul>
      <div class="pane-foot">
        {!e.built ? (
          <button class="btn text" onClick={construct}>
            Construct for {money(e.cost)}
          </button>
        ) : (
          <span />
        )}
        <button class="btn text" onClick={remove} title={e.built ? "Removing constructed track refunds nothing" : ""}>
          Remove track
        </button>
      </div>
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
function JunctionMap({ ports, moves, colourOf }: { ports: JPort[]; moves: JMove[]; colourOf: (line: number) => string }) {
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
        return <path d={`M${x1},${y1} Q${c},${c} ${x2},${y2}`} fill="none" stroke={colourOf(m.lines[0])} stroke-width={m.crosses.length ? 3 : 1.5} />;
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
        Level {levelLabel(n.level)}, {n.ports} tracks meet{n.flying ? ", with a flyover" : ""}
      </div>
      <Built built={built} what="The junction" />
      {n.flying ? (
        <>
          <p class="note muted">One track goes over the other here, so trains never wait for each other to cross.</p>
          {!built && (
            <div class="pane-foot">
              <span />
              <button class="btn text" onClick={() => setFlying(false)}>
                Remove the flyover
              </button>
            </div>
          )}
        </>
      ) : (
        <>
          <p class="note muted">
            {crossing
              ? "Some trains cross the path of others here on the same level, so they wait for each other. A flyover takes one track over the other."
              : "No trains cross each other's path here now. A flyover takes one track over the other, for when they do."}
          </p>
          <ul class="rows">
            <Row label="Flyover price">{price === null ? "" : money(price * 1e6)}</Row>
            <Row label="Waiting it saves">{perTrain >= 1 ? `${duration(perTrain)} a train at ${demand} demand` : "none now"}</Row>
          </ul>
          <div class="pane-foot">
            <span class="muted small">{built ? "Paid at once" : "Added to the blueprint"}</span>
            <button class="btn text" onClick={() => setFlying(true)}>
              Build a flyover
            </button>
          </div>
        </>
      )}
      {info && moves.length > 0 && (
        <>
          <JunctionMap ports={info.ports} moves={moves} colourOf={colourOf} />
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
