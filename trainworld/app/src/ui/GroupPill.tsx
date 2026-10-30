// T-1813: a line group's bullet, an outlined pill lettered in one colour with the group's name.

import { useLayoutEffect, useRef } from "preact/hooks";
import { groupBullet } from "../game/lineGroups";
import { legibleOn } from "../game/palette";
import { today } from "../game/today";
import type { LineInput } from "../game/types";
import { shade } from "../game/ui";
import { bulletEpoch, inkBox } from "./bullet";

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
  // T-1835: the text's capitals (or, in other scripts, its ink) centred in the pill. Measured where the
  // pill is: the dock is zoomed, which rounds the line box, so the font's metrics alone land 1-2 px high.
  void bulletEpoch.value;
  const pill = useRef<HTMLSpanElement>(null), text = useRef<HTMLSpanElement>(null);
  useLayoutEffect(() => {
    const p = pill.current, t = text.current;
    const box = inkBox(b.text, size === "lg" ? 16 : 12);
    if (!p || !t || !box || !p.offsetHeight) return;
    t.style.top = "0px";
    const probe = document.createElement("span");
    probe.style.cssText = "display:inline-block;width:0;height:0;vertical-align:baseline";
    t.appendChild(probe);
    const base = probe.getBoundingClientRect().bottom;
    probe.remove();
    const r = p.getBoundingClientRect(), s = r.height / p.offsetHeight;
    const dy = (r.top + r.height / 2 + ((box.up - box.down) / 2) * s - base) / s;
    if (Number.isFinite(dy)) t.style.top = Math.round(dy * 4) / 4 + "px";
  });
  return (
    <span ref={pill} class={"group-pill " + size + (onClick ? " link" : "")} style={{ "--c": c }} data-tip={tip} onClick={onClick}>
      <span ref={text} class="group-pill-text">{b.text}</span>
    </span>
  );
}
