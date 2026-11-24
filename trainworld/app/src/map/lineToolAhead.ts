// T-4147: the line tool's added stop drawn at the click. While the pointer is over a station the line could take, its
// path once added is read (`insertPath`, the line tool's own search); a click on it then draws that path before the
// worker has applied the edit (its word follows a moment later and replaces it). T-4170: with the line edits' other
// guesses from their op (game/editGuess.ts): a removed end stop's path cut back, a line deleted, its colour, the
// panel's switches.

import { stopsQueued, world } from "../game/world";
import type { Show } from "../game/pending";
import type { EditOp } from "../workers/protocol";
import { queryClock } from "../workers/clockClient";
import { edgesWithoutEnd } from "./lineAhead";
import { live } from "./live";

/** the last path read: the world version it was read in, so a click after any other edit draws nothing from it */
const read = { version: -1, line: -1, station: -1, edges: null as number[] | null };

/** The pointer is over `station`'s marker while line `line` is edited. */
export function hoverStop(line: number, station: number) {
  const version = world.peek()?.version ?? -1;
  if (read.version === version && read.line === line && read.station === station) return;
  Object.assign(read, { version, line, station, edges: null });
  void queryClock({ what: "insertPath", line, station }).then((d) => {
    if (read.version === version && read.line === line && read.station === station) read.edges = d && d.length ? Array.from(d) : null;
  });
}

/** T-4114: a plain stop click's path at the click: an end stop taken out cuts the strokes back, a station added draws
 * the path the pointer read over it; only with no other click out and the network drawn from the world's stops */
function stopPath(lineNum: number, station: number): Iterable<number> | null {
  const r = live.overlay?.renderer, n = r?.net, w = world.peek();
  const line = w?.save.lines.find((l) => Number(l.id) === lineNum);
  // (a network older than the world has an earlier click's stops: its stretches are not these stops')
  if (!r || !n || !w || !line || r.netVersion !== w.version || line.planStops || stopsQueued.peek() > 1) return null;
  const stops = line.stops.map(Number), k = stops.indexOf(station);
  if (k < 0) return read.version === w.version && read.line === lineNum && read.station === station ? read.edges : null;
  if (stops[0] === stops[stops.length - 1] || stops.indexOf(station, k + 1) >= 0) return null;
  return edgesWithoutEnd(n, lineNum, k, stops.length);
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
