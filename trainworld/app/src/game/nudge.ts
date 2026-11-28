// T-3742: nudging a rail line's departures a row at a time (notes/T-3742.md), from the schedule panel's Offset
// column: an undoable service edit that buys nothing, one in flight per row.

import { signal } from "@preact/signals";
import type { LineId } from "./types";
import { edit, world } from "./world";

/** A row's offset as stepped and not yet shown by the worker's readout (`lineNudges`): `target` the offset (s) the
 * cell shows meanwhile, `sent` what the edits sent reach, `from` the world version holding them (Infinity while
 * one is out). */
interface Pending { target: number; sent: number; busy: boolean; from: number }
export const pendingNudge = signal<ReadonlyMap<string, Pending>>(new Map());

/** T-4510: weekend rows are nudged on their own, keyed apart. */
export const nudgeKey = (id: LineId, row: number, weekend = false) => `${id}:${weekend ? "w" : ""}${row}`;

function put(k: string, p: Partial<Pending> | null) {
  const was = pendingNudge.peek().get(k);
  if (!was) return;
  const m = new Map(pendingNudge.peek());
  if (p === null) m.delete(k);
  else m.set(k, { ...was, ...p });
  pendingNudge.value = m;
}

/** Step row `row` of line `id` from the offset it shows (`now`, s) to `to`. Steps while an edit is out add up into
 * the next edit, sent when it lands; a refused one puts the row back to the world's. */
export function stepNudge(id: LineId, row: number, now: number, to: number, weekend = false): void {
  const k = nudgeKey(id, row, weekend);
  const m = new Map(pendingNudge.peek());
  const p = m.get(k) ?? { target: now, sent: now, busy: false, from: Infinity };
  m.set(k, { ...p, target: to });
  pendingNudge.value = m;
  if (!p.busy) void send(id, row, k, weekend);
}

async function send(id: LineId, row: number, k: string, weekend: boolean) {
  put(k, { busy: true, from: Infinity });
  for (;;) {
    const p = pendingNudge.peek().get(k);
    if (!p) return;
    const by = p.target - p.sent;
    if (Math.abs(by) < 1e-6) break;
    const a = await edit({ op: "nudgeRow", line: Number(id), row, by, weekend }, false, true);
    if (!a.ok) return put(k, null);
    put(k, { sent: p.sent + by });
  }
  put(k, { busy: false, from: world.peek()?.version ?? 0 });
}

/** A readout of line `id`'s offsets taken at world version `version`: the rows it already holds show it. */
export function nudgesSeen(id: LineId, version: number): void {
  const m = pendingNudge.peek();
  const done = [...m].filter(([k, p]) => k.startsWith(id + ":") && !p.busy && version >= p.from).map(([k]) => k);
  if (!done.length) return;
  const out = new Map(m);
  for (const k of done) out.delete(k);
  pendingNudge.value = out;
}
