// The junction inspector (SPEC 8): the routes through a junction, its flyover and what it saves.

import { clock } from "../../game/clock";
import { duration, levelLabel, money } from "../../game/format";
import { compass, stationTowards } from "../../game/places";
import { DEMAND_INDEX } from "../../game/types";
import { edit, linesByNum, say, world } from "../../game/world";
import { Actions, Built, Chip, Row } from "../widgets";
import { useClockQuery } from "../useClockQuery";

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
  const info = useClockQuery({ what: "junction", node }, [node, w?.version], parseJunction);
  const price = useClockQuery({ what: "flyover", node }, [node, w?.version], (d) => d[0]);
  const demand = clock.demand.value;
  const lev = DEMAND_INDEX[demand];
  const n = w?.nodes.get(node);
  if (!w || !n) return null;
  const built = n.builtPorts >= 3;
  // a crossover's end keeps its shape, so no flyover there
  const atCrossover = w.edges.some((e) => e.crossover && (e.a === node || e.b === node));
  const setFlying = async (flying: boolean) => {
    const a = await edit({ op: "flying", node, flying });
    if (a.ok && flying) say(a.charge > 0 ? `Flyover built for ${money(a.charge * 1e6)}.` : "Flyover added to the blueprint.");
  };
  const lineOf = linesByNum(w);
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
