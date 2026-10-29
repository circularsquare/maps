// T-1813: a line group's bullet, an outlined pill lettered in one colour with the group's name.

import { groupBullet } from "../game/lineGroups";
import { legibleOn } from "../game/palette";
import { today } from "../game/today";
import type { LineInput } from "../game/types";
import { shade } from "../game/ui";
import { bulletEpoch, inkShift } from "./bullet";

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
  // T-1817: kept readable on the page's shade
  const c = legibleOn(look?.colour ?? b.colour, shade.value === "dark");
  // the text's ink centred in the pill, as a line bullet's symbol (line box and font as style.css)
  void bulletEpoch.value;
  const dy = size === "lg" ? inkShift(b.text, 16, 26) : inkShift(b.text, 12, 16);
  return (
    <span class={"group-pill " + size + (onClick ? " link" : "")} style={{ "--c": c }} data-tip={tip} onClick={onClick}>
      <span class="group-pill-text" style={dy ? { top: dy + "px" } : undefined}>{b.text}</span>
    </span>
  );
}
