// Parts the Lines and Stations tabs share: the scope toggle, the distance column, the name order.

import { shortDistance } from "../game/format";
import { cityById, viewedZone } from "../game/cities";
import { listScope, scopeEmpty } from "../game/listScope";
import { Icon } from "./widgets";

/** Line names in natural order ("2" before "10"). One collator for every comparison: localeCompare
 * with options builds a new one each call, which made sorting 109 lines take 15-400 ms (T-348). */
export const natural = new Intl.Collator(undefined, { numeric: true, sensitivity: "base" }).compare;

/** T-332: the viewed place or the whole world. Over dead space only the world is on offer. `tip`
 * adds a line to the place's hover (T-531). */
export function ScopeToggle({ tip }: { tip?: string }) {
  const zone = viewedZone.value;
  const name = zone ? cityById(zone).name : null;
  const empty = scopeEmpty();
  const here = listScope.value === "here" && !!zone && !empty;
  return (
    <div class="group scope">
      <button class={"btn" + (here ? " on" : "")} disabled={!zone} data-tip={name ? (empty ? `No lines in ${name} yet, so the whole world is shown` : `Only ${name}` + (tip ? "\n" + tip : "")) : "Nothing modelled here"}
        onClick={() => (listScope.value = "here")}>
        <Icon.pin />{name}
      </button>
      <button class={"btn" + (here ? "" : " on")} data-tip="The whole world" aria-label="The whole world" onClick={() => (listScope.value = "world")}>
        <Icon.globe />
      </button>
    </div>
  );
}

/** T-332: a row's distance from the centre of the screen. */
export function Near({ m }: { m: number | undefined }) {
  return <span class="val muted w-near" data-tip={m !== undefined ? `${shortDistance(m)} from the centre of the screen` : undefined}>{m !== undefined ? shortDistance(m) : ""}</span>;
}

/** Rows with no distance (a line with no stops) sort last whichever way the column runs. */
export const byNear = (a: number | undefined, b: number | undefined, dir: number) =>
  a === undefined || b === undefined ? (a === undefined ? 1 : 0) - (b === undefined ? 1 : 0) : (a - b) * dir;
