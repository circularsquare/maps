// T-4114, T-4145: what is drawn ahead of the network (map/lineAhead.ts), from the edits waiting to show (game/pending.ts).

import { world } from "../game/world";
import { setAheadSink, setCashSource, setOpGuess, setUndoSource, type Composed } from "../game/pending";
import { liveMoney, moneyState } from "../game/money";
import { editGuess } from "../game/editGuess";
import { undoAhead, warmUndo, type AheadLook, type AheadModel } from "./lineAhead";
import { live } from "./live";

function lookOf(line: number): AheadLook {
  const st = world.peek()?.lineStats[String(line)];
  return { running: st?.status === "running", fast: !!st?.fast, tram: !!st?.tram };
}

/** The renderer draws `c` ahead from now; `build` false: a network is about to be set, which builds it. */
function draw(c: Composed, build: boolean) {
  const o = live.overlay;
  if (!o) return;
  // (the bus layer takes its routes from the world: they are drawn whether or not a network is about to come)
  if (o.renderer.buses.setAhead(c.buses.size ? c.buses : null)) o.invalidate();
  const stats = world.peek()?.lineStats;
  const m: AheadModel = { ...c, lines: new Map() };
  // (a bus or ferry line has no strokes)
  for (const [l, edges] of c.lines) if (!stats?.[String(l)]?.bus) m.lines.set(l, { edges, look: lookOf(l) });
  o.renderer.setAhead(m, build);
  o.invalidate();
}
setAheadSink(draw);
// T-4170: every edit's guess at its click, from its op
setOpGuess(editGuess);
// T-4181: an edit the cash cannot pay is refused, so it shows nothing at its click
setCashSource(() => { const m = moneyState.peek(); return m ? { cash: liveMoney.peek().cash, sandbox: m.mode === "sandbox" } : null; });
// T-4145: an undo of the newest edit drawn at its click from the network before it
setUndoSource(() => live.overlay?.renderer.net ?? null, (was, now) => {
  const u = undoAhead(was, now);
  return { hideEdges: u.hideEdges, hideLines: u.hideLines, hideStations: u.hideStations, track: u.track, lineStrokes: u.lineStrokes, rawStations: u.stations ? [u.stations] : [], colours: u.colours };
}, warmUndo);
