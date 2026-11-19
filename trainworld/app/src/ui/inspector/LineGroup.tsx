// T-1791, T-1813, T-1818: the line panel's group row, when the line is in a group: "Part of", the
// group's bullet (a click opens the group view), "group", and Leave group. T-3423: the group view
// has the same row for the group around it; leaving moves out one level.

import { groupBullet, groupName, groupTree, lineGroups } from "../../game/lineGroups";
import type { LineInput } from "../../game/types";
import { edit, world } from "../../game/world";
import { tRich, t } from "../../i18n";
import { selection } from "../../game/ui";
import { Icon } from "../widgets";
import { GroupPill } from "../GroupPill";

/** The row for group `g`, which the line `line` or the group `of` sits straight in. */
function PartOf({ g, line, of }: { g: number; line?: number; of?: number }) {
  const w = world.value;
  const ls = w ? lineGroups(w.save.lines).get(g) : undefined;
  if (!w || !ls) return null;
  const up = groupTree(w.save.lines).get(g)?.parent ?? 0;
  const open = () => (selection.value = { kind: "group", group: g });
  const leave = () => void edit(line !== undefined ? { op: "lineGroup", lines: [line], group: up } : { op: "groupMembers", lines: [], groups: [of!], group: up, parent: 0 });
  // the hover carries the full name where the bullet reads it shorter
  const name = groupName(ls, g), same = groupBullet(ls, g, () => 0).text.toLowerCase() === name.toLowerCase();
  return (
    <div class="line-group">
      {tRich("lineGroup.partOf", { group: <GroupPill ls={ls} g={g} tip={same ? t("lineGroup.open") : t("lineGroup.openNamed", { name })} onClick={open} /> })}
      <button class="btn icon" data-tip={t("lineGroup.leave")} aria-label={t("lineGroup.leave")} onClick={leave}>
        <Icon.leaveGroup />
      </button>
    </div>
  );
}

export function LineGroup({ l }: { l: LineInput }) {
  return l.group ? <PartOf g={l.group} line={l.num} /> : null;
}

/** T-3423: the group view's row for the group around it. */
export function GroupParent({ group }: { group: number }) {
  const w = world.value;
  const up = w ? groupTree(w.save.lines).get(group)?.parent ?? 0 : 0;
  return up ? <PartOf g={up} of={group} /> : null;
}
