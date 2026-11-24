// T-4147: the line tool's added stop drawn at the click. While the pointer is over a station the line could take, its
// path once added is read (`insertPath`, the line tool's own search); a click on it then draws that path before the
// worker has applied the edit (its word follows a moment later and replaces it).

import { world } from "../game/world";
import type { LineInput } from "../game/types";
import { queryClock } from "../workers/clockClient";
import { showAhead } from "./aheadShow";

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

/** A click that adds `station` to `line`: its path now, when the pointer read it over the same stops. */
export function guessAdded(line: LineInput, station: number) {
  if (read.version !== world.peek()?.version || read.line !== line.num || read.station !== station || !read.edges) return;
  showAhead(Number(line.id), read.edges, false);
}
