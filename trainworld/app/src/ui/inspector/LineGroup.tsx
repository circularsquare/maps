// T-1791, T-1813, T-1818: the line panel's group row, when the line is in a group: "Part of", the
// group's bullet (a click opens the group view), "group". T-3423: the group view has the same row
// for the group around it. A line leaves a group by dragging it out in the Lines tab (T-3810).

import { groupBullet, groupName, groupTree, lineGroups } from "../../game/lineGroups";
import type { LineInput } from "../../game/types";
import { world } from "../../game/world";
import { tRich, t } from "../../i18n";
import { selection } from "../../game/ui";
import { GroupPill } from "../GroupPill";

/** The row for group `g`, which a line or a group sits straight in. */
function PartOf({ g }: { g: number }) {
  const w = world.value;
  const ls = w ? lineGroups(w.save.lines).get(g) : undefined;
  if (!w || !ls) return null;
  const open = () => (selection.value = { kind: "group", group: g });
  // the hover carries the full name where the bullet reads it shorter
  const name = groupName(ls, g), same = groupBullet(ls, g, () => 0).text.toLowerCase() === name.toLowerCase();
  return (
    <div class="line-group">
      {tRich("lineGroup.partOf", { group: <GroupPill ls={ls} g={g} tip={same ? t("lineGroup.open") : t("lineGroup.openNamed", { name })} onClick={open} /> })}
    </div>
  );
}

export function LineGroup({ l }: { l: LineInput }) {
  return l.group ? <PartOf g={l.group} /> : null;
}

/** T-3423: the group view's row for the group around it. */
export function GroupParent({ group }: { group: number }) {
  const w = world.value;
  const up = w ? groupTree(w.save.lines).get(group)?.parent ?? 0 : 0;
  return up ? <PartOf g={up} /> : null;
}
