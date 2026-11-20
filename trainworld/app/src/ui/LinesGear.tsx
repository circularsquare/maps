// T-3816, T-3799: the Lines tab's gear and its popover: the line types and cities to list, and the columns to show.

import { useEffect, useRef, useState } from "preact/hooks";
import type { JSX } from "preact";
import { t } from "../i18n";
import { LINE_COLS, linesView, setOff, type LineCol } from "../game/linesView";
import { Icon } from "./widgets";

/** a line type's icon and name, by the Type column's rank: train, high speed, tram, bus, ferry */
export const TYPE_ICONS = [Icon.train, Icon.highSpeed, Icon.tram, Icon.bus, Icon.ferry];
export const typeName = (rank: number) => [t("settingsTab.shapeTrain"), t("toolBar.highSpeed"), t("toolBar.tram"), t("buildTab.bus"), t("buildTab.ferry")][rank];
const colName = (c: LineCol) => ({ city: t("linesTab.colCity"), riders: t("linesTab.colRiders"), money: t("linesTab.colMoney"), trains: t("linesTab.colTrains"), kind: t("linesTab.colType") })[c];

/** `modes` and `cities` are the choices the game has (a section with fewer than two is left out); `leftOut` the
 * lines the filters keep off the list, in the whole game. */
export function LinesGear({ modes, cities, leftOut }: { modes: number[]; cities: [string, string][]; leftOut: number }) {
  const [open, setOpen] = useState(false);
  const box = useRef<HTMLSpanElement>(null);
  useEffect(() => {
    if (!open) return;
    // before main.ts's Escape, which would clear the selection too
    const key = (e: KeyboardEvent) => { if (e.key !== "Escape") return; setOpen(false); e.preventDefault(); e.stopImmediatePropagation(); };
    const away = (e: PointerEvent) => { if (!box.current?.contains(e.target as Node)) setOpen(false); };
    window.addEventListener("keydown", key, true);
    window.addEventListener("pointerdown", away, true);
    return () => { window.removeEventListener("keydown", key, true); window.removeEventListener("pointerdown", away, true); };
  }, [open]);
  const v = linesView.value;
  const offModes = new Set(v.modes), offCities = new Set(v.cities), offCols = new Set(v.cols);
  const tip = t("linesTab.gear") + (leftOut ? "\n" + t("linesTab.leftOut", { count: leftOut }) : "");
  const cityOff = cities.filter(([id]) => offCities.has(id)).length;
  const check = (on: boolean, label: string, flip: () => void, some = false, k?: string): JSX.Element => (
    <label class="check" key={k}>
      <input type="checkbox" class={some ? "some" : undefined} checked={on} onChange={flip} />
      <span>{label}</span>
    </label>
  );
  return (
    <span class="lines-gear" ref={box}>
      <button class={"btn icon" + (open ? " on" : "") + (leftOut ? " filtered" : "")} data-tip={tip} aria-label={t("linesTab.gear")} aria-expanded={open}
        onClick={() => setOpen(!open)}><Icon.gear /></button>
      {open && (
        <div class="lines-gear-pop" role="group" aria-label={t("linesTab.gear")}>
          {modes.length >= 2 && (
            <div class="gear-sec">
              <div class="h">{t("linesTab.gearTypes")}</div>
              <div class="group">
                {modes.map((m) => {
                  const I = TYPE_ICONS[m], on = !offModes.has(m);
                  return <button key={m} class={"btn icon" + (on ? " on" : "")} data-tip={typeName(m)} aria-label={typeName(m)} aria-pressed={on} onClick={() => setOff("modes", [m], on)}><I /></button>;
                })}
              </div>
            </div>
          )}
          <div class="gear-sec">
            <div class="h">{t("linesTab.gearColumns")}</div>
            <div class="checks">
              {LINE_COLS.map((c) => check(!offCols.has(c), colName(c), () => setOff("cols", [c], !offCols.has(c)), false, c))}
            </div>
          </div>
          {cities.length >= 2 && (
            <div class="gear-sec">
              <div class="h">{check(cityOff === 0, t("linesTab.gearCities"), () => setOff("cities", cities.map(([id]) => id), cityOff === 0), cityOff > 0 && cityOff < cities.length)}</div>
              <div class="checks gear-cities">
                {cities.map(([id, name]) => check(!offCities.has(id), name, () => setOff("cities", [id], !offCities.has(id)), false, id))}
              </div>
            </div>
          )}
        </div>
      )}
    </span>
  );
}
