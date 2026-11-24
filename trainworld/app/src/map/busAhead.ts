// T-4147: a bus or ferry line's route drawn at its click: the route the edit sends, as longitude and latitude, in the bus
// layer (map/busLayer.ts `setAhead`) until the rows that hold it arrive (game/pending.ts, a part of the entry that clears
// with the rows). The stops, which the sim places, are left out until then. Spacing, Manual, BRT and a ferry's boat show in
// their panel (BusInspector.tsx) through `fields`.

import type { EditOp, FerryPortIn } from "../workers/protocol";
import type { Show } from "../game/pending";
import { packToLngLat } from "../game/coords";
import { world } from "../game/world";

/** a bus or ferry line drawn ahead: its route and, for a ferry, each port's landing and berth (longitude, latitude flat) */
export interface BusAhead {
  route: number[];
  /** a line the world does not hold yet */
  colour?: string;
  ferry?: { spurs: number[]; marks: number[] };
}

/** the longitude and latitude of `route`'s points, `stride` numbers per point (x, y first) */
function routeLL(route: readonly number[], stride: number): number[] {
  const out: number[] = [];
  for (let i = 0; i + 1 < route.length; i += stride) out.push(...packToLngLat(route[i], route[i + 1]));
  return out;
}

/** a ferry's ports as the edit names them: each one's landing and berth, flat */
function portsLL(line: number | null, ports: readonly FerryPortIn[]): { spurs: number[]; marks: number[] } | null {
  const w = world.peek();
  const had = line === null ? [] : w?.lineStats[String(w.save.lines.find((l) => l.num === line)?.id)]?.bus?.ferry?.ports ?? [];
  const at: { landing: number[]; berth: number[] }[] = [];
  for (const p of ports) {
    if ("again" in p) { const a = at[p.again]; if (!a) return null; at.push(a); }
    else if ("node" in p) {
      const h = had.find((x) => x.node === p.node);
      if (!h) return null;
      at.push({ landing: h.landing, berth: h.berth });
    } else at.push({ landing: packToLngLat(p.x, p.y), berth: packToLngLat(p.bx, p.by) });
  }
  return { spurs: at.flatMap((a) => [...a.landing, ...a.berth]), marks: at.flatMap((a) => a.berth) };
}

/** The new line a drawn-ahead new bus or ferry line takes while the world has none: a number no line has */
export const NEW_AHEAD = 1_000_000;

const record = (line: number) => "line:" + line;

/** What a bus or ferry edit shows at its click (game/editGuess.ts), or null. */
export function busGuess(op: EditOp): Show | null {
  switch (op.op) {
    case "busRoute": return { buses: new Map([[op.line, { route: routeLL(op.route, 3) }]]) };
    case "addBusLine": return { buses: new Map([[NEW_AHEAD, { route: routeLL(op.route, 3), colour: op.colour }]]) };
    case "ferryRoute": {
      const f = portsLL(op.line, op.ports);
      return f ? { buses: new Map([[op.line, { route: routeLL(op.route, 4), ferry: f }]]) } : null;
    }
    case "addFerryLine": {
      const f = portsLL(null, op.ports);
      return f ? { buses: new Map([[NEW_AHEAD, { route: routeLL(op.route, 4), colour: op.colour, ferry: f }]]) } : null;
    }
    case "busSpacing": return { fields: new Map([[record(op.line), { hand: false, spacing: op.spacing }]]) };
    case "busHand": return { fields: new Map([[record(op.line), { hand: true, spacing: 0 }]]) };
    case "busBrt": return { fields: new Map([[record(op.line), { brt: op.on }]]) };
    case "ferryBoat": return { fields: new Map([[record(op.line), { boat: op.boat }]]) };
    default: return null;
  }
}
