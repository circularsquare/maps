// T-3739: the Advanced fares panel, opened from the Money tab's Fare heading; centred like the
// schedule panel. Everything the Money tab's fare rows set, plus points added to and removed from the
// curve (1 to 12, "Add point" then a click on the graph), and every country's multiple and base fare.
// Room is left below the countries for fare zones (T-3740) and a fare per stop (T-3744).
// docs/ui-panels.md "Advanced fares".

import { signal } from "@preact/signals";
import { useEffect, useRef, useState } from "preact/hooks";
import { busFare, fareAt, ferryFare, HSR_FARE, hsrFare, isCustomCurve, MAX_NODES, NODE_KM, scaleNodes, transferFare, type FareNode, type Fares } from "../game/fareCurve";
import { roundedDefault } from "../game/fareDefault";
import { CITIES } from "../game/cities";
import { defaultScale, fareScale, SHOWN_STEP, shownScale, storedScale } from "../game/income";
import { moneyState, stationCountry } from "../game/money";
import { redo, undo, changeFares } from "../game/fareUndo";
import { ask } from "../game/ui";
import { setFareRef, setFareScale, world } from "../game/world";
import { price, unitMetres, unitShort } from "../game/format";
import { playBuild } from "../map/buildSound";
import { t, tCountry, tNumber, tRich } from "../i18n";
import { useModalDialog } from "./Modal";
import { advancedFares, fareAdding, FareButtons, FareGraph, FareNodeRows } from "./FareGraph";
import { BusFareRow, FareEffect, fareRef, FerryFareRow, HsrFareRow, TransferFareRow } from "./MoneyTab";
import { Check, Icon, Stepper } from "./widgets";

/** The panel's own drag preview: its country's scale may differ from the dock's graph's. */
const panelPreview = signal<FareNode[] | null>(null);
/** "Show all countries", kept while the game is open. */
const allCountries = signal(false);

const fix2 = (v: number) => tNumber(v, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
const mulText = (v: number) => `${fix2(v)}x`;

/** The curve as charged in `country` (its round default while untouched, T-1662), US$. */
const nodesIn = (f: Fares, country: string): FareNode[] => roundedDefault(f, country)?.nodes ?? scaleNodes(f.nodes, fareScale(country, f.scales));

/** Every country with a built station, by name. */
function builtCountries(): string[] {
  const w = world.value, out = new Set<string>();
  for (const st of w?.save.stations ?? []) {
    if (!st.built) continue;
    const c = stationCountry(st);
    if (c) out.add(c);
  }
  return [...out];
}
let everyCountry: string[] | null = null;
const allCountryCodes = () => (everyCountry ??= [...new Set(CITIES.flatMap((c) => (c.country ? [c.country] : [])))]);
const byName = (a: string, b: string) => tCountry(a).localeCompare(tCountry(b));

/** One country's row: its multiple of the reference's fares (a plain 1.00x for the reference), its base
 * fare, and the Money tab's anchor and undo buttons. */
function CountryRow({ f, country, refCountry }: { f: Fares; country: string; refCountry: string }) {
  const name = tCountry(country);
  const base = <span class="af-base">{price(nodesIn(f, country)[0][1])}</span>;
  if (country === refCountry) {
    return (
      <li>
        <span class="fr-what">{name}</span>
        <span class="fr-ref" data-tip={t("moneyTab.referenceTip", { name })}>{mulText(1)}</span>
        {base}
        <span />
      </li>
    );
  }
  const scales = f.scales, refScale = fareScale(refCountry, scales), scale = fareScale(country, scales);
  const set = (v: number) => void setFareScale(country, storedScale(v, country, scales, refScale));
  const shown = shownScale(scale, refScale);
  return (
    <li>
      <span class="fr-what">{name}</span>
      <Stepper value={mulText(shown)} wide label={t("moneyTab.fareScale")} onStep={(d) => set((d > 0 ? Math.floor(shown / SHOWN_STEP + 1e-6) + d : Math.ceil(shown / SHOWN_STEP - 1e-6) + d) * SHOWN_STEP)}
        type={{ text: shown.toFixed(2), set, tip: t("moneyTab.countryFaresTip", { name, ref: tCountry(refCountry) }) }} />
      {base}
      <span class="group">
        <button class="btn icon" aria-label={t("moneyTab.makeRef")} data-tip={`${t("moneyTab.makeRef")}\n${t("moneyTab.makeRefSame")}`} onClick={() => void setFareRef(country)}><Icon.anchor /></button>
        {scales?.[country] !== undefined && <button class="btn icon" aria-label={t("moneyTab.backToDefault")} data-tip={t("moneyTab.backTo", { value: mulText(shownScale(defaultScale(country, scales), refScale)) })} onClick={() => void setFareScale(country, null)}><Icon.undo /></button>}
      </span>
    </li>
  );
}

function Panel({ start }: { start: string }) {
  const dialog = useRef<HTMLDialogElement>(null);
  useModalDialog(dialog, true);
  const [picked, setPicked] = useState(start);
  const w = world.value, m = moneyState.value;
  useEffect(() => () => { fareAdding.value = false; panelPreview.value = null; }, []);
  if (!w || !m) return null;
  const f = m.fares;
  const close = () => (advancedFares.value = null);
  // Esc first lets go of "Add point"
  const escape = () => { if (fareAdding.peek()) fareAdding.value = false; else close(); };
  const ref = fareRef(f);
  const built = builtCountries().filter((c) => c !== ref).sort(byName);
  const listed = [ref, ...built];
  const rest = allCountries.value ? allCountryCodes().filter((c) => !listed.includes(c)).sort(byName) : [];
  const country = picked;
  const options = [...listed, ...(listed.includes(country) ? [] : [country]), ...rest.filter((c) => c !== country)];
  const scale = fareScale(country, f.scales);
  const shown = nodesIn(f, country);
  const nodes = panelPreview.value ?? shown;
  const custom = isCustomCurve(f.nodes);
  const kpu = unitMetres() / 1000, unit = unitShort();
  const lineKind = (k: "fast" | "bus" | "ferry") => w.save.lines.some((l) => {
    const st = w.lineStats[l.id];
    return k === "fast" ? !!st?.fast : k === "ferry" ? !!st?.bus?.ferry : !!st?.bus && !st.bus.ferry;
  });
  const hsr = lineKind("fast") ? hsrFare(f) : null, bus = lineKind("bus") ? busFare(f) : null, ferry = lineKind("ferry") ? ferryFare(f) : null;
  const full = shown.length >= MAX_NODES;
  const simple = () => {
    const km = (k: number) => tNumber(Math.round((k / kpu) * 10) / 10);
    ask.value = {
      text: t("advancedFares.simpleAsk", { a: km(NODE_KM[1]), b: km(NODE_KM[2]), unit }),
      yes: t("advancedFares.simple"),
      no: t("toolNotes.cancel"),
      run: () => { playBuild("pop"); void changeFares(scaleNodes(NODE_KM.map((k) => [k, fareAt(shown, k)] as FareNode), 1 / scale)); },
    };
  };
  return (
    <dialog class="saves-panel sched-panel fares-panel" ref={dialog} aria-label={t("advancedFares.title")}
      onCancel={(e) => { e.preventDefault(); escape(); }}
      onClick={(e) => { if (e.target === e.currentTarget) close(); }}
      onKeyDown={(e) => {
        e.stopPropagation();
        if (e.key === "Escape") { e.preventDefault(); escape(); return; }
        // the game's undo keys, which stop at the dialog: fare changes and then the track's (T-1130)
        const typing = e.target instanceof HTMLInputElement || e.target instanceof HTMLSelectElement;
        if ((e.ctrlKey || e.metaKey) && !e.altKey && !typing) {
          const k = e.key.toLowerCase();
          if (k === "z" && !e.shiftKey) { e.preventDefault(); void undo(); }
          else if (k === "y" || (k === "z" && e.shiftKey)) { e.preventDefault(); void redo(); }
        }
      }}>
      <div class="saves-inner">
        <div class="saves-head sp-title">
          <h2>{t("advancedFares.title")}</h2>
          <button class="btn icon sp-close" autofocus aria-label={t("modal.close")} data-tip={t("modal.closeTip")} onClick={close}><Icon.cross /></button>
        </div>
        <div class="af-tools">
          <span class="af-country">{tRich("advancedFares.faresIn", {
            country: <select aria-label={t("advancedFares.country")} value={country} onChange={(e) => { panelPreview.value = null; setPicked(e.currentTarget.value); }}>
              {options.map((c) => <option value={c}>{tCountry(c)}</option>)}
            </select>,
          })}</span>
          <span class="right"><FareButtons nodes={nodes} /></span>
        </div>
        <FareGraph nodes={shown} scale={scale} big preview={panelPreview} />
        <FareNodeRows nodes={nodes} scale={scale} remove
          last={<>
            <li class="af-add">
              <span class="fr-what">
                <button class={"btn text mini" + (fareAdding.value ? " on" : "")} aria-pressed={fareAdding.value} disabled={full}
                  data-tip={full ? t("advancedFares.full", { n: MAX_NODES }) : undefined}
                  onClick={() => (fareAdding.value = !fareAdding.peek())}><Icon.plus />{t("advancedFares.addPoint")}</button>
                {custom && <button class="btn text mini" onClick={simple}>{t("advancedFares.simple")}</button>}
              </span>
              <span /><span />
            </li>
            {hsr !== null && <HsrFareRow mul={hsr} own={hsr !== HSR_FARE} />}
            {bus !== null && <BusFareRow mul={bus} />}
            {ferry !== null && <FerryFareRow mul={ferry} />}
            <TransferFareRow fare={transferFare(f)} base={f.nodes[0]?.[1] ?? f.base} scale={scale} />
          </>} />
        <FareEffect preview={panelPreview.value} scale={scale} />
        <div class="h">{t("advancedFares.byCountry")}</div>
        <ul class="rows fare-rows af-countries">
          <li class="af-head"><span /><span>{t("advancedFares.multiple")}</span><span>{t("fareGraph.baseFare")}</span><span /></li>
          {[...listed, ...rest].map((c) => <CountryRow f={f} country={c} refCountry={ref} />)}
        </ul>
        <Check checked={allCountries.value} onChange={(v) => (allCountries.value = v)}>{t("advancedFares.showAll")}</Check>
      </div>
    </dialog>
  );
}

/** The panel while `advancedFares` holds the country it opened on. */
export function AdvancedFares() {
  const start = advancedFares.value;
  return start ? <Panel start={start} /> : null;
}
