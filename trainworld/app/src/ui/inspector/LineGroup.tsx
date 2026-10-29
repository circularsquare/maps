// T-1791, T-1813, T-1818: the line panel's group row, when the line is in a group: "Part of", the
// group's bullet (a click opens the group view), "group", and Leave group.

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
  // the hover carries the full name where the bullet reads it shorter
  const name = groupName(ls), same = groupBullet(ls, () => 0).text.toLowerCase() === name.toLowerCase();
  return (
    <div class="line-group">
      <span class="muted">Part of</span>
      <GroupPill ls={ls} tip={same ? "Open the group" : `${name}\nOpen the group`} onClick={open} />
      <span class="muted">group</span>
      <button class="btn icon" data-tip="Leave group" aria-label="Leave group" onClick={() => void edit({ op: "lineGroup", lines: [l.num], group: 0 })}>
        <Icon.leaveGroup />
      </button>
    </div>
  );
}
