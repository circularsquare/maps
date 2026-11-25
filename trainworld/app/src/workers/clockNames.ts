// T-4214 stage A: a station is sent at its click with no name and a ticket (game/world.ts edits from map/stationTool.ts,
// stationFree.ts, stationPort.ts), and its name follows as "stationName" once the page's lookup ends (place lists, real
// stations: 10-250 ms). One that arrives before its station is applied is taken at the apply; one that arrives after names
// the stations the edit placed, outside the undo history (`name_new_stations`). notes/T-4214.md "Stage A".

import type { TrackApi } from "../wasm/track/sim";
import type { EditOp } from "./protocol";

/** tickets whose name came, and those whose stations were placed before it: at most this many of each are kept */
const KEEP = 64;
const known = new Map<number, { name: string; game: number }>();
const owed = new Map<number, { nodes: number[]; game: number }>();

const trim = <V>(m: Map<number, V>) => { while (m.size > KEEP) m.delete(m.keys().next().value!); };

type StationOp = Extract<EditOp, { op: "addStation" | "addBrtStation" | "addPort" }>;
const ticketed = (op: EditOp): op is StationOp & { ticket: number } =>
  (op.op === "addStation" || op.op === "addBrtStation" || op.op === "addPort") && op.ticket !== undefined;

/** A name arrived (outside the message queue, so a station edit still waiting in it takes the name when applied). */
export function nameArrived(ticket: number, name: string, game: number) {
  known.set(ticket, { name, game });
  trim(known);
}

/** `op` as it is applied in game `game`: a ticketed station op with its name if that came already. */
export function withName(op: EditOp, game: number): EditOp {
  if (!ticketed(op)) return op;
  const k = known.get(op.ticket);
  return k && k.game === game ? { ...op, name: k.name } : op;
}

/** A ticketed station op `op` was applied with no name: the stations it placed take the name when it comes. */
export function placedNameless(api: TrackApi, op: EditOp, game: number) {
  if (!ticketed(op) || op.name) return;
  const nodes = op.op === "addStation" ? [...api.last_stations()] : [op.op === "addBrtStation" ? api.last_brt_station() : api.last_port()];
  owed.set(op.ticket, { nodes: nodes.filter((n) => n !== 0xffffffff), game });
  trim(owed);
}

/** Whether a name for `ticket` has stations to name in game `game`. */
export const nameOwed = (ticket: number, game: number) => owed.get(ticket)?.game === game;

/** Name the stations placed for `ticket` before its name came; how many took it. */
export function nameOwedStations(api: TrackApi, ticket: number, name: string, game: number): number {
  const o = owed.get(ticket);
  owed.delete(ticket);
  known.delete(ticket);
  return o && o.game === game && o.nodes.length ? api.name_new_stations(Uint32Array.from(o.nodes), name) : 0;
}
