// T-4411: sort a station's lines by top-level line group (in the Lines tab's order), ungrouped
// lines after them, and within a group by riders (most first).

import { cityById } from "../game/cities";
import { demandAt, lineSeconds } from "../game/clock";
import { byHand, handOrder } from "../game/handOrder";
import { groupChain, groupName, groupTree, lineGroups } from "../game/lineGroups";
import { trainsNowAt } from "../game/lineRunning";
import { linesView } from "../game/linesView";
import { distFromCentre, stationZone, viewCentre } from "../game/listScope";
import { profitKnown, runningToday, today, upkeepToday, type LineToday } from "../game/today";
import { DEMAND_INDEX, type LineInput, type LineStats, type WorldView } from "../game/types";
import { lineSort } from "../game/ui";
import { vehicleOf } from "../game/vehicle";
import { listedLines, stationsById, stopsById } from "../game/world";
import { tPlace } from "../i18n/places";
import { byNear, natural } from "./listParts";

const lineTypeRank = (st: LineStats | undefined) => {
  if (!st) return 0;
  const v = vehicleOf(st);
  return v === "train" ? (st.fast ? 1 : 0) : v === "tram" ? 2 : v === "bus" ? 3 : 4;
};
const lineNet = (d: LineToday | undefined, l: LineInput, st: LineStats) =>
  (d && profitKnown(l) ? d.dayFares - runningToday(l, st) - upkeepToday(l, st) : 0);

export interface StationLineOrder {
  topGroup: (l: LineInput) => number;
  groups: Map<number, LineInput[]>;
  sortLines: (ls: readonly LineInput[], riders: (l: LineInput) => number) => LineInput[];
}

/** Order of top-level line groups as the Lines tab shows them, and a comparator for a station's lines. */
export function stationLineOrder(w: WorldView, lineDist?: Map<string, number>): StationLineOrder {
  const listed = listedLines(w);
  const groups = lineGroups(listed), tree = groupTree(listed);
  const parentOf = (g: number) => {
    const p = tree.get(g)?.parent ?? 0;
    return tree.has(p) ? p : 0;
  };
  const homeOf = (l: LineInput) => {
    const c = groupChain(l);
    const g = c[0]?.group ?? l.group ?? 0;
    return g && tree.has(g) ? g : 0;
  };
  const topGroup = (l: LineInput) => {
    let g = homeOf(l);
    while (g && parentOf(g)) g = parentOf(g);
    return g;
  };

  const rank = new Map<number, number>();
  if (tree.size > 0) {
    const hand = handOrder.value.lines;
    const view = linesView.value;
    const sortOff = (view.cols as readonly string[]).includes(lineSort.value.key);
    const key = hand.length ? null : sortOff ? "name" : lineSort.value.key;
    const dir = sortOff ? 1 : lineSort.value.dir;
    const tod = today.value;
    const lsOf = (g: number) => groups.get(g) ?? tree.get(g)!.lines;

    const stAt = key === "city" ? stationsById(w) : null;
    const cityOf = (l: LineInput) => {
      const s = stAt?.get(l.stops[0]);
      const z = s ? stationZone(s) : null;
      return z ? tPlace(z, cityById(z).name) : "";
    };
    const sum = (ls: LineInput[], f: (l: LineInput) => number) => ls.reduce((n, l) => n + f(l), 0);
    const riders = (ls: LineInput[]) => sum(ls, (l) => tod.get(l.id)?.riders ?? 0);
    const net = (ls: LineInput[]) => sum(ls, (l) => lineNet(tod.get(l.id), l, w.lineStats[l.id]));
    const levOf = (id: string) => DEMAND_INDEX[demandAt(lineSeconds(w.lineStats[id]))];
    const trains = (ls: LineInput[]) => sum(ls, (l) => (w.lineStats[l.id]?.status === "running" ? trainsNowAt(l.id, levOf(l.id)) : 0));
    const typeOf = (ls: LineInput[]) => {
      const n = [0, 0, 0, 0, 0];
      for (const l of ls) n[lineTypeRank(w.lineStats[l.id])]++;
      return n.indexOf(Math.max(...n));
    };

    let distMap = lineDist;
    if (key === "near" && !distMap) {
      const c = viewCentre.value;
      if (c) {
        const at = stopsById(w);
        distMap = new Map();
        for (const l of listed) {
          let best = Infinity;
          for (const id of l.stops) {
            const s = at.get(id);
            if (s) best = Math.min(best, distFromCentre(s.lng, s.lat, c));
          }
          if (best < Infinity) distMap.set(l.id, best);
        }
      }
    }
    const near = (ls: LineInput[]) =>
      ls.reduce<number | undefined>((m, l) => {
        const d = distMap?.get(l.id);
        return d === undefined ? m : m === undefined ? d : Math.min(m, d);
      }, undefined);

    const order = (a: number, b: number) => {
      const la = lsOf(a), lb = lsOf(b);
      return (
        (key === "kind" ? (typeOf(la) - typeOf(lb)) * dir
          : key === "city" ? natural(cityOf(la[0]), cityOf(lb[0])) * dir
          : key === "riders" ? (riders(la) - riders(lb)) * dir
          : key === "money" ? (net(la) - net(lb)) * dir
          : key === "trains" ? (trains(la) - trains(lb)) * dir
          : key === "near" ? byNear(near(la), near(lb), dir)
          : 0)
        || natural(groupName(la, a), groupName(lb, b)) * (key === "name" ? dir : 1)
        || la[0].num - lb[0].num
      );
    };

    const topIds = [...tree.keys()].filter((g) => !parentOf(g)).sort(order);
    const ordered = hand.length ? byHand(topIds, (g) => lsOf(g).map((l) => l.id), hand) : topIds;
    ordered.forEach((g, i) => rank.set(g, i));
  }

  const sortLines = (ls: readonly LineInput[], riders: (l: LineInput) => number): LineInput[] =>
    [...ls].sort((a, b) => {
      const ga = topGroup(a), gb = topGroup(b);
      const ra = ga ? (rank.get(ga) ?? Infinity) : Infinity;
      const rb = gb ? (rank.get(gb) ?? Infinity) : Infinity;
      if (ra !== rb) return ra < rb ? -1 : 1;
      return riders(b) - riders(a) || a.num - b.num;
    });

  return { topGroup, groups, sortLines };
}
