// T-4145: Stations and deletes's part of an applied edit's word (workers/clockWord.ts), stream B (T-4146). A stub from the base: its stream owns the body.

import type { TrackApi } from "../wasm/track/sim";
import type { Applied } from "./clockEdits";
import type { EditOp, EditWord } from "./protocol";
import type { Posted } from "./clockWord";

/** What an accepted edit changed that the page draws ahead of the network; undefined for nothing. `before`: what
 * `stationWordBefore` read before the apply, when there is one. */
export function stationWord(_api: TrackApi, _op: EditOp, _answer: Applied, _posted: Posted, _before?: unknown): Partial<EditWord> | undefined {
  return undefined;
}
