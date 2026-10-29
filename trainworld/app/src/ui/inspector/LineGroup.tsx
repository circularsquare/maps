// T-1791, T-1813: the line panel's group row, when the line is in a group: the group's bullet and
// name (a click opens the group view) and Leave group.

import { groupBullet, groupName, lineGroups } from "../../game/lineGroups";
import type { LineInput } from "../../game/types";
import { edit, world } from "../../game/world";
import { selection } from "../../game/ui";
import { Icon } from "../widgets";
import { GroupPill } from "../GroupPill";

export function LineGroup({ l }: { l: LineInput }) {
  const w = world.value;
  const ls = l.group && w ? lineGroups(w.save.lines).get(l.group) : undefined;
  if (!ls) return null;
  const open = () => (selection.value = { kind: "group", group: l.group! });
  // the name only where the bullet does not read the same
  const name = groupName(ls), same = groupBullet(ls, () => 0).text.toLowerCase() === name.toLowerCase();
  return (
    <div class="line-group">
      <GroupPill ls={ls} tip="Open the group" onClick={open} />
      <span class="name link" data-tip="Open the group" onClick={open}>{same ? "" : name}</span>
      <button class="btn icon" data-tip="Leave group" aria-label="Leave group" onClick={() => void edit({ op: "lineGroup", lines: [l.num], group: 0 })}>
        <Icon.leaveGroup />
      </button>
    </div>
  );
}
