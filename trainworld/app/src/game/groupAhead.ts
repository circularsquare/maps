// T-4147: a line group edit at its click: each line's chain of groups (game/lineGroups.ts `groupChain`) as the track
// model will leave it, worked out as sim/src/track/wasm/lines.rs does (`set_group_members`, `edit_group`,
// `settle_groups`). Groups are a list and panel matter only, so nothing is drawn on the map.

import type { EditOp } from "../workers/protocol";
import type { OuterGroup } from "./types";
import { groupChain } from "./lineGroups";
import type { Show } from "./pending";
import { world } from "./world";

type Chains = Map<number, OuterGroup[]>;

/** a group with fewer than two members (lines straight in it, groups straight under it) stops being one */
function settle(chains: Chains) {
  for (;;) {
    const members = new Map<number, Set<string>>();
    const add = (g: number, m: string) => members.set(g, (members.get(g) ?? new Set()).add(m));
    for (const [l, c] of chains) {
      if (c[0]) add(c[0].group, "l" + l);
      for (let k = 0; k + 1 < c.length; k++) add(c[k + 1].group, "g" + c[k].group);
    }
    const lone = new Set([...members].filter(([, m]) => m.size < 2).map(([g]) => g));
    if (!lone.size) return;
    for (const [l, c] of chains) chains.set(l, c.filter((o) => !lone.has(o.group)));
  }
}

/** What a group edit shows at its click: the new chain of every line it changes, or null (refused, or no change). */
export function groupGuess(op: EditOp): Show | null {
  if (op.op !== "groupName" && op.op !== "groupLook" && op.op !== "groupMembers") return null;
  const lines = world.peek()?.save.lines;
  if (!lines) return null;
  const was: Chains = new Map(lines.map((l) => [l.num, groupChain(l)]));
  const chains: Chains = new Map([...was].map(([l, c]) => [l, c.map((o) => ({ ...o }))]));
  if (op.op === "groupName" || op.op === "groupLook") {
    let any = false;
    for (const c of chains.values()) {
      const o = op.group !== 0 ? c.find((x) => x.group === op.group) : undefined;
      if (!o) continue;
      any = true;
      if (op.op === "groupName") o.name = op.name;
      else { o.symbol = op.symbol.trim(); o.colour = op.colour ?? ""; }
    }
    if (!any) return null;
  } else {
    const from = (g: number): OuterGroup[] => {
      if (g === 0) return [];
      for (const c of chains.values()) { const k = c.findIndex((o) => o.group === g); if (k >= 0) return c.slice(k); }
      return [];
    };
    const group = op.group === "new" ? Math.max(0, ...[...chains.values()].flat().map((o) => o.group)) + 1 : op.group;
    let target: OuterGroup[];
    if (group === 0) target = [];
    else if (from(group).length) target = from(group);
    else {
      const up = from(op.parent);
      if (op.parent === group || (op.parent !== 0 && !up.length)) return null;
      target = [{ group, name: "", symbol: "", colour: "" }, ...up];
    }
    for (const g of op.groups) if (g === 0 || target.some((o) => o.group === g) || !from(g).length) return null;
    for (const [l, c] of chains) {
      if (op.lines.includes(l)) chains.set(l, target.map((o) => ({ ...o })));
      else {
        const k = c.findIndex((o) => op.groups.includes(o.group));
        if (k >= 0) chains.set(l, [...c.slice(0, k + 1), ...target.map((o) => ({ ...o }))]);
      }
    }
    settle(chains);
  }
  const fields = new Map<string, Record<string, unknown>>();
  for (const [l, c] of chains) if (JSON.stringify(c) !== JSON.stringify(was.get(l))) fields.set("line:" + l, { chain: c });
  return fields.size ? { fields } : null;
}
