// T-4146: what a station or delete edit shows at the click (stream B of T-4144), until the network holding it is set
// (game/pending.ts draws it). Each guess is made from what the page holds: the drawn network, the world, the station
// tool's ghost. The worker's word (workers/clockWordStations.ts) replaces a guess's lines where they differ.

import { linesByNum, world } from "../game/world";
import { shown, type Show } from "../game/pending";
import { planStopsOf, type StationInput } from "../game/types";
import type { EditOp } from "../workers/protocol";
import { live } from "./live";
import { edgesWithoutEnd } from "./lineAhead";
import { stationIndexOf, type NetworkBuffers } from "./network";
import { reversed, ST_ONE, ST_ONE_MID, ST_ONE_NEG, ST_TRAM } from "./stationGeometry";

/** the network's station flag for a blueprint station */
const ST_BLUEPRINT = 2;

/** The record the panels read a station's pending figures by. */
export const stationRecord = (node: number) => `station:${node}`;

/** Station `node` removed: it is hidden, and a line it ended loses the stretch to it (the drawn network's
 * `edgesWithoutEnd`); a line stopping there part way keeps its strokes (it is routed on the same track, T-859) until
 * the word says otherwise. */
export function stationGoneShow(n: NetworkBuffers | null, node: number): Show {
  const w = world.peek(), lines = new Map<number, Iterable<number> | null>();
  if (w && n) for (const l of linesByNum(w).values()) {
    const stops = planStopsOf(l).map(Number), k = stops.indexOf(node);
    if (k < 0 || (k !== 0 && k !== stops.length - 1) || w.lineStats[l.id]?.bus) continue;
    const keep = edgesWithoutEnd(n, l.num, k, stops.length);
    if (keep) lines.set(l.num, keep);
  }
  return { hideStations: [node], lines };
}

/** Track `edge` removed (a crossover with its other diagonal): hidden, with the lines over it, until the word says where
 * they are routed instead. */
export function trackGoneShow(edge: number): Show {
  const e = world.peek()?.edges.find((x) => x.id === edge);
  return { hideEdges: e?.crossover && e.partner !== undefined && e.partner >= 0 ? [edge, e.partner] : [edge] };
}

/** Station `node` as drawn in `n` with `changes` to its box (`plat` metres, `side`), laid along the drawn track at its node
 * as the station tool's ghost is; null where it is not drawn. */
function drawnBox(n: NetworkBuffers, node: number, changes: { plat?: number; side?: number }) {
  const w = world.peek(), i = stationIndexOf(n, String(node)), nd = w?.nodes.get(node);
  if (!w || !nd || i < 0) return null;
  const ways = w.edges.filter((e) => e.a === node || e.b === node).flatMap((e) => {
    const p = n.edgePts.get(e.id);
    return p ? [e.a === node ? p : reversed(p)] : [];
  });
  if (!ways.length) return null;
  const f = Math.round(n.stations[i * 3 + 2]);
  const side = (f & ST_ONE) === 0 ? 0 : (f & ST_ONE_NEG) !== 0 ? -1 : 1;
  return { lx: n.stations[i * 3], ly: n.stations[i * 3 + 1], ways, perM: 1 / Math.cos((nd.lat * Math.PI) / 180), plat: changes.plat ?? nd.platform, side: changes.side ?? side, tram: (f & ST_TRAM) !== 0, mid: (f & ST_ONE_MID) !== 0, built: (f & ST_BLUEPRINT) === 0 };
}

/** Station `node`'s platform set to `platform` m for `cost` (US$M): the panel's figure, and its box drawn at that length
 * in place of the drawn one. */
export function platformShow(n: NetworkBuffers | null, node: number, platform: number, cost: number): Show {
  const out: Show = { fields: new Map([[stationRecord(node), { platform }]]), cost };
  const box = n ? drawnBox(n, node, { plat: platform }) : null;
  if (box) Object.assign(out, { stations: [box], hideStations: [node] });
  return out;
}

/** The station panel's other figures at the click, by station node (one track or both, the way of a one-track station,
 * upgrades: `StationInput`'s fields), for `cost` (US$M). */
export function fieldsShow(fields: [node: number, fields: { one?: number; oneHeading?: number; upgrades?: number }][], cost = 0): Show {
  return { fields: new Map(fields.map(([node, f]) => [stationRecord(node), f])), cost };
}

/** Station `s` as the panels show it: its figures still on their way to the world (`fieldsShow`, `platformShow`) in place. */
export function shownStation(s: StationInput): StationInput {
  const r = stationRecord(s.num);
  const length = shown(r, "platform", s.length), one = shown(r, "one", s.one), oneHeading = shown(r, "oneHeading", s.oneHeading), upgrades = shown(r, "upgrades", s.upgrades);
  return length === s.length && one === s.one && oneHeading === s.oneHeading && upgrades === s.upgrades ? s : { ...s, length, one, oneHeading, upgrades };
}

/** What a station or delete edit shows at its click (game/editGuess.ts), or null. Its price, where the panel quoted one,
 * is the caller's (`ahead({ cost })`); a platform's comes with the worker's word. */
export function stationGuess(op: EditOp): Show | null {
  const net = () => live.overlay?.renderer.net ?? null;
  switch (op.op) {
    case "removeStation": return stationGoneShow(net(), op.node);
    case "removeEdge": return trackGoneShow(op.edge);
    case "stationProps": {
      // (a rename alone is answered with its rows and no network, which a box drawn ahead would wait for)
      const nd = world.peek()?.nodes.get(op.node);
      return nd && op.platform !== shown(stationRecord(op.node), "platform", nd.platform) ? platformShow(net(), op.node, op.platform, 0) : null;
    }
    case "stationTracks": return fieldsShow([[op.node, { one: op.one ? 2 : 0 }]]);
    case "stationWay": return fieldsShow([[op.node, { one: 1, oneHeading: op.heading }]]);
    case "stationUpgrade": {
      const byNum = new Map((world.peek()?.save.stations ?? []).map((s) => [s.num, s]));
      return fieldsShow(op.nodes.flatMap((num) => {
        const s = byNum.get(num);
        if (!s) return [];
        const was = shown(stationRecord(num), "upgrades", s.upgrades) ?? 0;
        return [[num, { upgrades: op.on ? was | op.bit : was & ~op.bit }]];
      }));
    }
    default: return null;
  }
}
