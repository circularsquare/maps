// T-1791: the line panel's group row, when the line is in a group: the group's lines, its name
// (a click opens the group view) and Leave group.

import { groupName, lineGroups } from "../../game/lineGroups";
import type { LineInput } from "../../game/types";
import { edit, world } from "../../game/world";
import { selection } from "../../game/ui";
import { Chip, Icon } from "../widgets";

export function LineGroup({ l }: { l: LineInput }) {
  const w = world.value;
  const ls = l.group && w ? lineGroups(w.save.lines).get(l.group) : undefined;
  if (!ls) return null;
  return (
    <div class="line-group">
      <span class="group-chips">{ls.map((m) => <Chip line={m} plain={m.id === l.id} />)}</span>
      <span class="name link" data-tip="Open the group" onClick={() => (selection.value = { kind: "group", group: l.group! })}>{groupName(ls)}</span>
      <button class="btn icon" data-tip="Leave group" aria-label="Leave group" onClick={() => void edit({ op: "lineGroup", lines: [l.num], group: 0 })}>
        <Icon.leaveGroup />
      </button>
    </div>
  );
}
