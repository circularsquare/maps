// T-1208, T-1210: demand's sums off the main thread (workers/demandSums.ts); no WASM. Answers in
// the order asked.
import { driveLevels } from "../game/drives";
import { curveRank, dealtCells, type DealtCells } from "../game/farChunks";
import type { PurposeFilter } from "../game/purposes";
import { dayViews, flowsBody, packDays, selectionFlows, stationFlows, sumParts, type FlowsAnswer, type FromSums, type ToSums } from "./demandSums";

/** the cells of the zones last asked about, by key (`ToSums` "flows"), and their curve ranks */
const cells = new Map<string, Uint32Array>();

/** the answer, and the arrays it hands over */
function answer(msg: ToSums): [unknown, Transferable[]] {
  if (msg.kind === "icSel") return [selectionFlows(sumParts("icSel", msg.parts, msg.ctx.places), msg.ctx), []];
  if (msg.kind === "icStation") return [stationFlows(msg.members.map((parts) => sumParts("icStation", parts, msg.ctx.places)), msg.ctx), []];
  if (msg.kind === "drives") return [driveLevels(msg.input), []];
  if (msg.kind === "days") {
    const p = packDays(dayViews(msg.input), msg.input.layout, msg.input.res.length);
    return [p, [p.f32.buffer, p.f64.buffer, p.idx.buffer]];
  }
  // flows
  const body = flowsBody(msg.packed, msg.parts);
  if (!body) return [null, []];
  let rank: Uint32Array | undefined;
  if (msg.cells) {
    rank = cells.get(msg.cells.key);
    if (!rank && msg.cells.xy) {
      rank = curveRank(msg.cells.xy);
      cells.set(msg.cells.key, rank);
      if (cells.size > 4) cells.delete(cells.keys().next().value!);
    }
  }
  const dealt: [PurposeFilter, DealtCells | null][] | null = rank ? [["all", dealtCells(body, "all", rank)], ...[...(body.byPurpose?.keys() ?? [])].map((k): [PurposeFilter, DealtCells | null] => [k, dealtCells(body, k, rank!)])] : null;
  return [{ body, dealt } satisfies FlowsAnswer, []];
}

self.onmessage = (e: MessageEvent<ToSums>) => {
  const msg = e.data;
  let reply: FromSums, transfer: Transferable[] = [];
  try {
    const [result, t] = answer(msg);
    reply = { id: msg.id, result };
    transfer = t;
  } catch (err) {
    reply = { id: msg.id, result: null, error: String(err) };
  }
  (self as unknown as Worker).postMessage(reply, transfer);
};
