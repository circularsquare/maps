// T-4147: the line tool's stop click drawn at the click. While the pointer is over a station of the line or one it could
// take, its path once taken out or added is read (`clickPath`, T-4097: the worker's own toggle, read again after each
// edit sent); a click on it then draws that path before the worker has applied the edit (its word follows a moment
// later and replaces it). T-4170: with the line edits' other guesses from their op (game/editGuess.ts): a removed end
// stop's path cut back, a line deleted, its colour, the panel's switches.

import { stopsQueued, world } from "../game/world";
import { editsSentCount, type Show } from "../game/pending";
import type { EditOp } from "../workers/protocol";
import { gameStamp, queryClock } from "../workers/clockClient";
import { edgesInStretches, edgesWithoutEnd } from "./lineAhead";
import { live } from "./live";

/** the last path read: the game and the count of edits sent when it was asked (the worker reads in message order, so it
 * is of the model those edits left), so a click after any later edit draws nothing from it */
const read = { game: -1, after: -1, line: -1, station: -1, edges: null as number[] | null };
const readFor = (line: number, station: number) => read.game === gameStamp.peek() && read.after === editsSentCount() && read.line === line && read.station === station;

/** The pointer is over `station`'s marker while line `line` is edited. */
export function hoverStop(line: number, station: number) {
  if (readFor(line, station)) return;
  const game = gameStamp.peek(), after = editsSentCount();
  Object.assign(read, { game, after, line, station, edges: null });
  void queryClock({ what: "clickPath", line, station }).then((d) => {
    if (read.game === game && read.after === after && read.line === line && read.station === station) read.edges = d && d.length ? Array.from(d) : null;
  });
}

/** T-4114: a plain stop click's path at the click: the path the pointer read over the station since the last edit sent
 * (added or taken out; T-4097: so a click in a burst has its own), else an end stop taken out cuts the strokes back, only
 * with no other click out and the network drawn from the world's stops */
function stopPath(lineNum: number, station: number): Iterable<number> | null {
  if (readFor(lineNum, station) && read.edges) return read.edges;
  const r = live.overlay?.renderer, n = r?.net, w = world.peek();
  const line = w?.save.lines.find((l) => Number(l.id) === lineNum);
  // (a network older than the world has an earlier click's stops: its stretches are not these stops')
  if (!r || !n || !w || !line || r.netVersion !== w.version || line.planStops || stopsQueued.peek() > 1) return null;
  const stops = line.stops.map(Number), k = stops.indexOf(station);
  if (k < 0 || stops[0] === stops[stops.length - 1] || stops.indexOf(station, k + 1) >= 0) return null;
  return edgesWithoutEnd(n, lineNum, k, stops.length);
}

/** T-4147: the group panel's stop edits, which keep the line's path: an end stop dropped (or the line ended at a stop) cuts the
 * strokes back to the stretches left, from the network drawn; a stop dropped between two keeps every stroke (null). Only a
 * network that holds the world's stops, as for a stop click. */
function cutBack(lineNum: number, index: number, drop: boolean): Set<number> | null {
  const r = live.overlay?.renderer, n = r?.net, w = world.peek();
  const line = w?.save.lines.find((l) => Number(l.id) === lineNum);
  if (!r || !n || !w || !line || r.netVersion !== w.version || line.planStops || stopsQueued.peek() > 1) return null;
  const len = line.stops.length;
  if (len < 3 || line.stops[0] === line.stops[len - 1] || index < 0 || index >= len) return null;
  if (drop) return index === 0 || index === len - 1 ? edgesWithoutEnd(n, lineNum, index, len) : null;
  // the stops past the new end, toward the nearer end of the line, go (at least two stops stay)
  return index * 2 >= len - 1 ? (index >= 1 ? edgesInStretches(n, lineNum, 0, index) : null) : (index <= len - 2 ? edgesInStretches(n, lineNum, index, len - 1) : null);
}

const fields = (line: number, f: Record<string, unknown>): Show => ({ fields: new Map([[`line:${line}`, f]]) });

/** What a line edit shows at its click (game/editGuess.ts), or null. */
export function lineGuess(op: EditOp): Show | null {
  const bus = (line: number) => !!world.peek()?.lineStats[String(line)]?.bus;
  switch (op.op) {
    case "toggleStop": {
      if (bus(op.line)) return null;
      const edges = stopPath(op.line, op.station);
      return edges ? { lines: new Map([[op.line, edges]]) } : null;
    }
    case "dropStop":
    case "endLineAt": {
      if (bus(op.line)) return null;
      const edges = cutBack(op.line, op.index, op.op === "dropStop");
      return edges ? { lines: new Map([[op.line, edges]]) } : null;
    }
    // (a bus or ferry line has no strokes)
    case "removeLine": return bus(op.line) ? null : { lines: new Map([[op.line, null]]) };
    case "lineLook": {
      if (op.colour === undefined) return null;
      const n = parseInt(op.colour.slice(1), 16);
      return { colours: new Map([[op.line, [(n >> 16) & 255, (n >> 8) & 255, n & 255]]]) };
    }
    case "lineFast": return fields(op.line, { fast: op.fast });
    case "lineOneWay": return fields(op.line, { oneWay: op.oneWay });
    case "linePairs": return fields(op.line, { pairs: op.on ? 1 : 0 });
    case "lineFareDist": return fields(op.line, { fareDist: op.dist });
    default: return null;
  }
}
