// Parts the Lines and Stations tabs share: the scope toggle, the name filter, the distance column, the name order.

import type { Signal } from "@preact/signals";
import { signal } from "@preact/signals";
import { shortDistance } from "../game/format";
import { cityById, viewedZone } from "../game/cities";
import { listScope, scopeEmpty } from "../game/listScope";
import { normalise } from "../game/nameMatch";
import { FindInput, SearchIcon } from "./FindInput";
import { composing } from "./keys";
import { Icon } from "./widgets";
import { t } from "../i18n";
import { tPlace } from "../i18n/places";

/** Line names in natural order ("2" before "10"). One collator for every comparison: localeCompare
 * with options builds a new one each call, which made sorting 109 lines take 15-400 ms (T-348). */
export const natural = new Intl.Collator(undefined, { numeric: true, sensitivity: "base" }).compare;

/** T-332: the viewed place or the whole world. Over dead space only the world is on offer. `tip`
 * adds a line to the place's hover (T-531). */
export function ScopeToggle({ tip }: { tip?: string }) {
  const zone = viewedZone.value;
  const name = zone ? tPlace(zone, cityById(zone).name) : null;
  const empty = scopeEmpty();
  const here = listScope.value === "here" && !!zone && !empty;
  return (
    <div class="group scope">
      <button class={"btn" + (here ? " on" : "")} disabled={!zone} data-tip={name ? (empty ? t("listParts.noLines", { name }) : t("listParts.only", { name }) + (tip ? "\n" + tip : "")) : t("listParts.nothingHere")}
        onClick={() => (listScope.value = "here")}>
        <Icon.pin />{name}
      </button>
      <button class={"btn" + (here ? "" : " on")} data-tip={t("listParts.world")} aria-label={t("listParts.world")} onClick={() => (listScope.value = "world")}>
        <Icon.globe />
      </button>
    </div>
  );
}

/** What each list is filtered by, as typed. Kept while the tab is closed, so going back to a station
 * picked from a filtered list finds the list as it was. */
export const linesFind = signal(""), stationsFind = signal("");

/** Whether `name` holds the typed filter `q` (already `normalise`d), read as the search box reads names: case,
 * accents and punctuation aside ("alcala" finds "Alcalá"), spaces optional. */
export function nameHas(name: string, q: string): boolean {
  let n = normed.get(name);
  if (n === undefined) {
    if (normed.size > 50_000) normed.clear();
    normed.set(name, (n = normalise(name)));
  }
  return n.includes(q) || n.replace(/ /g, "").includes(q.replace(/ /g, ""));
}
/** names as `normalise` reads them, kept: a typed filter reads every row's name on each render */
const normed = new Map<string, string>();

/** The box over a list that keeps the rows whose name holds what is typed, alongside the scope, the gear's
 * filters and the sort; Esc empties it. */
export function ListFind({ text }: { text: Signal<string> }) {
  return (
    <FindInput type="search" wrap="list-find" class="new-game-find" icon={<SearchIcon />} onClear={() => (text.value = "")} aria-label={t("listParts.find")}
      maxLength={60} spellcheck={false} value={text.value} onInput={(e) => (text.value = (e.currentTarget as HTMLInputElement).value)}
      onKeyDown={(e) => { if (e.key === "Escape" && !composing(e) && text.peek()) { e.stopPropagation(); text.value = ""; } }} />
  );
}

/** T-332: a row's distance from the center of the screen. */
export function Near({ m }: { m: number | undefined }) {
  return <span class="val muted w-near" data-tip={m !== undefined ? t("listParts.near", { distance: shortDistance(m) }) : undefined}>{m !== undefined ? shortDistance(m) : ""}</span>;
}

/** Rows with no distance (a line with no stops) sort last whichever way the column runs. */
export const byNear = (a: number | undefined, b: number | undefined, dir: number) =>
  a === undefined || b === undefined ? (a === undefined ? 1 : 0) - (b === undefined ? 1 : 0) : (a - b) * dir;
