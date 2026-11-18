// One-way single track (T-3094): setting it sends the lines that ran the other way through the same
// stops another way, in the same edit. The player confirms the lines it changes first; a line with
// no other way refuses it.

import { queryClock } from "../workers/clockClient";
import { t } from "../i18n";
import { noWay } from "./issues";
import type { WorldView } from "./types";
import { ask } from "./ui";
import { edit, linesByNum, say, world } from "./world";

/** A broken line whose track is all still there but has a piece set one way against the line's run
 * (`path` entries are `edge << 1 | dir`, dir 0 running a to b); a deleted piece makes it plain gone. */
export function runsAgainstOneWay(w: WorldView, path: readonly number[]): boolean {
  const edges = new Map(w.edges.map((e) => [e.id, e]));
  let against = false;
  for (const p of path) {
    const e = edges.get(p >> 1);
    if (!e) return false;
    if (e.oneWay && e.oneWay !== (p & 1) + 1) against = true;
  }
  return against;
}

/** Run an edge one way (`dir` 1 a to b, 2 b to a), both ways pinned (3), or the way its run takes (0). */
export async function setOneWay(edge: number, dir: 0 | 1 | 2 | 3, reverse = false, off = false) {
  const d = await queryClock({ what: "oneWay", edge, dir });
  const w = world.peek();
  if (!w) return;
  const rows: { line: number; ok: boolean }[] = [];
  for (let k = 0; d && k + 1 < d.length; k += 2) rows.push({ line: d[k], ok: d[k + 1] === 1 });
  const stuck = rows.find((r) => !r.ok);
  if (stuck) return say(noWay(stuck.line), true);
  const go = () => void edit({ op: "oneWay", edge, dir });
  if (!rows.length) return go();
  const byNum = linesByNum(w);
  ask.value = {
    text: off ? t("oneWay.questionOff") : t("oneWay.question", { reverse }),
    yes: off ? t("oneWay.yesOff") : t("oneWay.yes", { reverse }),
    no: t("oneWay.no"),
    run: go,
    lines: rows.map((r) => ({ line: r.line, text: t("oneWay.reroutes", { line: byNum.get(r.line)?.name ?? "" }) })),
  };
}
