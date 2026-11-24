// T-4145, T-4147: the lines' part of an applied edit's word (workers/clockWord.ts): what the page draws of a line edit
// before the network that holds it.

import type { TrackApi } from "../wasm/track/sim";
import type { Applied } from "./clockEdits";
import type { EditOp, EditWord } from "./protocol";
import { lineEdges, type Posted } from "./clockWord";

const ROWS_FIRST = new Set<EditOp["op"]>(["copyLine", "lineLook", "lineFast", "lineOneWay", "linePairs", "lineLimit", "lineFareDist", "lineCars", "groupMembers", "groupName", "groupLook"]);

/** An edit whose panel and list entries are all in the rows (the world and its money): they go before the full state,
 * which the drawn network follows with, and the answer goes with them. A line's name alone is rows-only already
 * (`nameOnly`), a new line with no stops too. */
export function rowsFirst(op: EditOp): boolean {
  return ROWS_FIRST.has(op.op);
}

const rgbOf = (c: number): [number, number, number] => [(c >> 16) & 255, (c >> 8) & 255, c & 255];
const colourOfLine = (api: TrackApi, line: number): number | undefined => {
  const info = api.lines_info();
  for (let i = 0; i < info.length; i += 22) if (info[i] === line) return info[i + 1];
  return undefined;
};

/** A copy drawn on its source's edges in its source's colour, and a new colour on the line, before the network. */
export function lineWord(api: TrackApi, op: EditOp, answer: Applied, _posted: Posted): Partial<EditWord> | undefined {
  if (answer.r !== 0) return undefined;
  if (op.op === "lineLook" && op.colour !== undefined) return { colours: [{ line: op.line, rgb: rgbOf(parseInt(op.colour.slice(1), 16)) }] };
  if (op.op === "copyLine" && answer.line !== undefined) {
    const id = answer.line, c = colourOfLine(api, id);
    if (api.is_bus(id) || api.is_ferry(id)) return undefined;
    return { lines: [{ line: id, edges: lineEdges(api, id) }], ...(c !== undefined ? { colours: [{ line: id, rgb: rgbOf(c) }] } : {}) };
  }
  return undefined;
}
