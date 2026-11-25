// T-4114, T-4145: what is drawn ahead of the network (map/lineAhead.ts), from the edits waiting to show (game/pending.ts).

import { world } from "../game/world";
import { clickWorld, openUnder } from "./clickMap";
import { edgeOf } from "../game/worldDiff";
import { strokesFromRows } from "./network";
import { setAheadSink, setCashSource, setOpGuess, setUndoSource, type Composed } from "../game/pending";
import { liveMoney, moneyState } from "../game/money";
import { editGuess } from "../game/editGuess";
import { undoAhead, warmUndo, type AheadLook, type AheadModel } from "./lineAhead";
import { live } from "./live";

function lookOf(line: number): AheadLook {
  const st = world.peek()?.lineStats[String(line)];
  return { running: st?.status === "running", fast: !!st?.fast, tram: !!st?.tram };
}

/** what was drawn last, and whether part of it was read off the tools' map (`openUnder`) */
let last: { c: Composed; open: boolean } | null = null;

/** The renderer draws `c` ahead from now; `build` false: a network is about to be set, which builds it. */
function draw(c: Composed, build: boolean) {
  const o = live.overlay;
  if (!o) return;
  // (the bus layer takes its routes from the world: they are drawn whether or not a network is about to come)
  if (o.renderer.buses.setAhead(c.buses.size ? c.buses : null)) o.invalidate();
  const stats = world.peek()?.lineStats;
  const m: AheadModel = { ...c, lines: new Map() };
  // T-4214 stage E: segments drawn from the tools' map once their patch is in, and track hidden at a click that the drawn
  // network or a stand-in has only as part of an edge since cut (the far half of an edge a station just split): that edge
  // goes, the tools' map's pieces of it that stay drawn in its place (map/clickMap.ts `openUnder`)
  const open = openUnder(o.renderer.net, c.hideEdges, c.pieces, c.opened);
  last = { c, open: !!open || c.pieces.size > 0 || c.opened.size > 0 };
  if (open?.hide.length) m.hideEdges = new Set([...c.hideEdges, ...open.hide]);
  const pieces = open?.pieces ?? c.pieces, w = clickWorld();
  if (pieces.size && w) {
    const laid = [...pieces].flatMap(([id, pts]) => { const row = edgeOf(w, id); return row && pts.length >= 6 ? [{ pts, row }] : []; });
    if (laid.length) m.track = [...c.track, strokesFromRows(laid)];
  }
  // (a bus or ferry line has no strokes)
  for (const [l, edges] of c.lines) if (!stats?.[String(l)]?.bus) m.lines.set(l, { edges, look: lookOf(l) });
  o.renderer.setAhead(m, build);
  o.invalidate();
}
setAheadSink(draw);

/** T-4214 stage E, for checks: what the last drawing ahead hid of the drawn network and drew from the tools' map (edge ids) */
export function aheadNow() {
  const c = last?.c, o = live.overlay;
  if (!c || !o) return null;
  const open = openUnder(o.renderer.net, c.hideEdges, c.pieces, c.opened);
  return { hidden: [...c.hideEdges, ...(open?.hide ?? [])], pieces: [...(open?.pieces ?? c.pieces).keys()], stations: c.stations.length };
}

/** T-4214 stage E: the map the tools pick on changed (a patch laid, a network set): what is drawn ahead from it again. */
export function aheadMapMoved() {
  if (!last || !live.overlay) return;
  const was = last;
  // (cheap when nothing drawn ahead is read off that map or hides track a patch changed, the case nearly always)
  if (was.open || openUnder(live.overlay.renderer.net, was.c.hideEdges, was.c.pieces, was.c.opened)?.hide.length) draw(was.c, true);
}
// T-4170: every edit's guess at its click, from its op
setOpGuess(editGuess);
// T-4181: an edit the cash cannot pay is refused, so it shows nothing at its click
setCashSource(() => { const m = moneyState.peek(); return m ? { cash: liveMoney.peek().cash, sandbox: m.mode === "sandbox" } : null; });
// T-4145: an undo of the newest edit drawn at its click from the network before it
setUndoSource(() => live.overlay?.renderer.net ?? null, (was, now) => {
  const u = undoAhead(was, now);
  return { hideEdges: u.hideEdges, hideLines: u.hideLines, hideStations: u.hideStations, track: u.track, lineStrokes: u.lineStrokes, rawStations: u.stations ? [u.stations] : [], colours: u.colours };
}, warmUndo);
