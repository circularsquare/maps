// T-4114: a line's strokes drawn ahead of the network (map/lineAhead.ts), from the worker's word or a guess at the click.

import { world } from "../game/world";
import type { AheadLook } from "./lineAhead";
import { live } from "./live";

function lookOf(line: number): AheadLook {
  const st = world.peek()?.lineStats[String(line)];
  return { running: st?.status === "running", fast: !!st?.fast, tram: !!st?.tram };
}

/** Line `line`'s strokes on `edges` from now (null: the line is gone) until a network holding the edit; `armed`: the
 * worker's word, not a guess. */
export function showAhead(line: number, edges: Iterable<number> | null, armed: boolean) {
  const o = live.overlay;
  // (a bus or ferry line has no strokes, and the worker sends none)
  if (!o || world.peek()?.lineStats[String(line)]?.bus) return;
  o.renderer.lineAhead(line, edges, armed, lookOf(line));
  o.invalidate();
}

/** The line's edit was refused: its network strokes again. */
export function dropAhead(line: number) {
  const o = live.overlay;
  if (!o) return;
  o.renderer.dropAhead(line);
  o.invalidate();
}
