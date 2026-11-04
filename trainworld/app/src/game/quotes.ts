// T-1171: what the blueprint and a planned line would cost, asked of the clock worker only when a
// panel shows it. Each runs a trial construction there (about 10-100 ms a planned line on the real
// save), which every clock state used to pay for every planned line.

import { effect } from "@preact/signals";
import { useLatestAnswer } from "../ui/useClockQuery";
import { queryClock } from "../workers/clockClient";
import { world } from "./world";
import type { BlueprintItem } from "./types";

/** one answer per world version, shared by every panel asking */
let blueprint: { v: number; p: Promise<BlueprintItem[] | null> } | null = null;
const lines = new Map<number, { v: number; p: Promise<{ build: number; trains: number } | null> }>();

/** A panel's quote waits this long after an edit, so a run of clicks pays for one trial, not one each. */
const SETTLE_MS = 250;
/** `ask` after SETTLE_MS, or null when the world has moved past version `v` by then */
function settled<T>(v: number, ask: () => Promise<T | null>): Promise<T | null> {
  const wait = changedAt + SETTLE_MS - performance.now();
  if (wait <= 0) return ask();
  return new Promise((done) => setTimeout(() => done(world.peek()?.version === v ? ask() : null), wait));
}
/** when the world last changed */
let changedAt = 0;
effect(() => { void world.value; changedAt = performance.now(); });

/** The construct-all quote's rows at world version `v`, US$; null when an edit came first. */
export function blueprintQuote(v: number, now = false): Promise<BlueprintItem[] | null> {
  if (blueprint?.v !== v) {
    const ask = () => queryClock({ what: "blueprintQuote" });
    blueprint = {
      v,
      p: (now ? ask() : settled(v, ask)).then((d) => d && Array.from({ length: d.length / 7 }, (_, i) => {
        const r = d.subarray(i * 7, i * 7 + 7);
        return { kind: r[0], level: r[1] as BlueprintItem["level"], tracks: r[2], wet: !!r[3], ground: r[4], quantity: r[5], cost: r[6] * 1e6 };
      })),
    };
  }
  return blueprint.p;
}

/** What constructing a planned line (`num`) and the trains it needs would cost at world version
 * `v`, US$. */
export function lineQuote(num: number, v: number): Promise<{ build: number; trains: number } | null> {
  let q = lines.get(num);
  if (q?.v !== v) {
    if (lines.size > 50) lines.clear();
    lines.set(num, (q = { v, p: settled(v, () => queryClock({ what: "lineCost", line: num })).then((d) => d && { build: d[0] * 1e6, trains: d[1] * 1e6 }) }));
  }
  return q.p;
}

let pieces: { v: number; p: Promise<Float64Array | null> } | null = null;
/** T-2269: the blueprint priced piece by piece at world version `v` (`blueprint_piece_costs`: kind,
 * two ids, US$M, counted), the switched-off pieces too; null when an edit came first. */
export function blueprintPieces(v: number): Promise<Float64Array | null> {
  if (pieces?.v !== v) pieces = { v, p: settled(v, () => queryClock({ what: "blueprintPieces" })) };
  return pieces.p;
}
/** `blueprintPieces` for a panel while `shown`: the last answer meanwhile. */
export function useBlueprintPieces(shown: boolean): Float64Array | null {
  const v = world.value?.version ?? -1;
  return useLatestAnswer(shown ? () => blueprintPieces(v) : null, [v, shown]);
}

/** The blueprint's quote rows for a panel, asked again as the world changes; the last ones meanwhile. */
export function useBlueprintQuote(): BlueprintItem[] {
  const v = world.value?.version ?? -1;
  return useLatestAnswer(() => blueprintQuote(v), [v]) ?? [];
}

/** A planned line's build and train costs for its panel; null until the first answer, or for `num`
 * null (a running line). */
export function useLineQuote(num: number | null): { build: number; trains: number } | null {
  const v = world.value?.version ?? -1;
  const a = useLatestAnswer(num === null ? null : () => lineQuote(num, v).then((q) => q && { num, ...q }), [num, v]);
  return a && a.num === num ? a : null;
}
