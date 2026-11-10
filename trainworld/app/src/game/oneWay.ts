// One-way single track (T-3094): setting it sends the lines that ran the other way through the same
// stops another way, in the same edit. The player confirms the lines it changes first; a line with
// no other way refuses it.

import { queryClock } from "../workers/clockClient";
import { t } from "../i18n";
import { noWay } from "./issues";
import { ask } from "./ui";
import { edit, linesByNum, say, world } from "./world";

/** Run an edge one way (`dir` 1 a to b, 2 b to a) or both ways (0). */
export async function setOneWay(edge: number, dir: 0 | 1 | 2, reverse = false) {
  const d = dir ? await queryClock({ what: "oneWay", edge, dir }) : null;
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
    text: t("oneWay.question", { reverse }),
    yes: t("oneWay.yes", { reverse }),
    no: t("oneWay.no"),
    run: go,
    lines: rows.map((r) => ({ line: r.line, text: t("oneWay.reroutes", { line: byNum.get(r.line)?.name ?? "" }) })),
  };
}
