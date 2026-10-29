// T-1813: a line group's bullet, an outlined pill lettered in one colour with the group's name.

import { groupBullet } from "../game/lineGroups";
import { today } from "../game/today";
import type { LineInput } from "../game/types";

export function GroupPill({ ls, size = "", tip, onClick, look }: {
  /** the group's lines, oldest first */
  ls: readonly LineInput[];
  size?: "" | "lg";
  tip?: string;
  onClick?: (e: MouseEvent) => void;
  /** a colour being picked, shown before the edit lands */
  look?: { colour?: string };
}) {
  const t = today.value;
  const b = groupBullet(ls, (id) => t.get(id)?.riders ?? 0);
  return (
    <span class={"group-pill " + size + (onClick ? " link" : "")} style={{ "--c": look?.colour ?? b.colour }} data-tip={tip} onClick={onClick}>
      {b.text}
    </span>
  );
}
