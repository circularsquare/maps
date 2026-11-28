// T-3739: the Advanced fares panel, opened from the Money tab's Fare heading; centred like the
// schedule panel. Everything the Money tab's fare rows set, plus points added to and removed from the
// curve (1 to 12, "Add point" then a click on the graph), and every country's (T-3766: or every city's) multiple
// and base fare.
// Room is left below the countries for fare zones (T-3740) and a fare per stop (T-3744).
// docs/ui-panels.md "Advanced fares".

import { signal } from "@preact/signals";
import { useEffect, useRef, useState } from "preact/hooks";
import type { ComponentChildren } from "preact";
import { brtFare, busFare, fareAt, HSR_FARE, hsrFare, isCustomCurve, MAX_NODES, NODE_KM, scaleNodes, transferFare, type FareArea, type FareNode, type Fares } from "../game/fareCurve";
import { chargedPrices, MAX_ZONES } from "../game/fareZones";
import { newArea, removeArea, renameArea, setAreaOff, setZonePrice, startPainting, topZone } from "../game/fareZonePaint";
import { roundTo } from "../game/currency";
import { countryFareScales, faresIn } from "../game/fareDefault";
import { cityById, isCity, zoneCountry } from "../game/cities";
import { goToCity } from "./Cities";
import { moneyState, stationCity, stationCountry } from "../game/money";
import { redo, undo, changeFares } from "../game/fareUndo";
import { ask, selection } from "../game/ui";
import { setFareByCity, setFareRef, world } from "../game/world";
import { currencySpec, fromShown, price, priceTyped, toShown, unitMetres, unitShort } from "../game/format";
import { playBuild } from "../map/buildSound";
import { t, tCountry, tNumber, tRich } from "../i18n";
import { tipMuted } from "./tip";
import type { StationInput } from "../game/types";
import { useModalDialog } from "./Modal";
import { advancedFares, fareAdding, FareButtons, FareGraph, FareNodeRows } from "./FareGraph";
import { BusFareRow, chargedAt, fareCity, FareEffect, fareKindsInUse, fareRef, FerryFareRows, HsrFareRow, placeMultiple, placeName, stepMultiple, TransferFareRow } from "./MoneyTab";
import { Check, Icon, Segmented, Stepper } from "./widgets";
import { composing } from "./keys";

/** The panel's own drag preview: its country's scale may differ from the dock's graph's. */
const panelPreview = signal<FareNode[] | null>(null);

const fix2 = (v: number) => tNumber(v, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
const mulText = (v: number) => `${fix2(v)}x`;

/** Every country (T-3766: or city, `of`) with a built station. */
function builtIn(of: (st: StationInput) => string | null): string[] {
  const w = world.value, out = new Set<string>();
  for (const st of w?.save.stations ?? []) {
    if (!st.built) continue;
    const c = of(st);
    if (c) out.add(c);
  }
  return [...out];
}

const openCity = (id: string) => {
  const s = selection.peek();
  if (s?.kind !== "city" || s.city !== id) selection.value = { kind: "city", city: id };
};

/** One country's or (T-3766) city's row: its multiple of the reference's fares (a plain 1.00x for the reference
 * country), its base fare, and the Money tab's anchor (countries only) and undo buttons. */
function PlaceRow({ f, place }: { f: Fares; place: string }) {
  const name = placeName(f, place);
  // T-3811: a city's name opens its view (a click) and centres the map on it (a double click) behind the panel
  const label = f.byCity && isCity(place)
    ? <span class="place-link" onMouseDown={(e) => { if (e.detail > 1) e.preventDefault(); }} onClick={() => openCity(place)} onDblClick={() => { goToCity(cityById(place)); openCity(place); }}>{name}</span>
    : name;
  const base = <span class="af-base">{price(chargedAt(f, place).nodes[0][1])}</span>;
  const m = placeMultiple(f, place);
  if (m.isRef) {
    return (
      <li>
        <span class="fr-what">{label}</span>
        <span class="fr-ref" data-tip={t("moneyTab.referenceTip", { name })}>{mulText(1)}</span>
        {base}
        <span />
      </li>
    );
  }
  const set = (v: number) => void m.set(v);
  return (
    <li>
      <span class="fr-what">{label}</span>
      <Stepper value={mulText(m.shown)} wide label={t("moneyTab.fareScale")} onStep={(d) => set(stepMultiple(m.shown, d))}
        type={{ text: m.shown.toFixed(2), set, tip: t("moneyTab.countryFaresTip", { name, ref: tCountry(m.ref) }) }} />
      {base}
      <span class="group">
        {!f.byCity && <button class="btn icon" aria-label={t("moneyTab.makeRef")} data-tip={`${t("moneyTab.makeRef")}\n${t("moneyTab.makeRefSame")}`} onClick={() => void setFareRef(place)}><Icon.anchor /></button>}
        {m.own && <button class="btn icon" aria-label={t("moneyTab.backToDefault")} data-tip={t("moneyTab.backTo", { value: mulText(m.usual) })} onClick={() => void m.set(null)}><Icon.undo /></button>}
      </span>
    </li>
  );
}

/** T-3740: the fare areas, each its name, the brush that starts the Fare zones tool on it, its bin, and its price
 * for 1, 2, 3 ... fare zones (as many as its highest zone painted, at least three) as charged in the panel's
 * country or city (times its rate multiple `scale`, as demand charges them: `cityFareScales`); then New fare area,
 * which starts the tool on a new one. */
function FareAreas({ f, scale }: { f: Fares; scale: number }) {
  const areas = f.zones ?? [];
  const sp = currencySpec();
  const paint = (id: number) => { advancedFares.value = null; startPainting(id); };
  const remove = (a: FareArea, now: boolean) => {
    if (now) return removeArea(a.id);
    ask.value = { text: t("advancedFares.removeAreaAsk", { name: a.name }), yes: t("advancedFares.removeArea"), no: t("toolNotes.cancel"), run: () => removeArea(a.id) };
  };
  return (
    <ul class="rows fare-rows af-zones">
      <li class="af-head"><span class="af-title" data-tip={t("advancedFares.zonesTip")}>{t("advancedFares.zones")}</span><span /><span /></li>
      {areas.map((a) => {
        const n = Math.min(MAX_ZONES, Math.max(3, topZone(a.id)));
        const charged = chargedPrices(f, a);
        const onOff = a.off ? t("advancedFares.enableArea", { name: a.name }) : t("advancedFares.disableArea", { name: a.name });
        return (
          <>
            <li class="af-area">
              <span class="fr-what">
                <input class="af-name" value={a.name} aria-label={t("advancedFares.areaLabel")} maxLength={80}
                  onChange={(e) => renameArea(a.id, e.currentTarget.value)}
                  onKeyDown={(e) => { if (!composing(e) && e.key === "Enter") e.currentTarget.blur(); }} />
              </span>
              <button class="btn text mini" onClick={() => paint(a.id)}><Icon.brush />{t("advancedFares.paint")}</button>
              <span class="group">
                <Check checked={!a.off} onChange={(on) => setAreaOff(a.id, !on)} label={onOff} tip={onOff} />
                <button class="btn icon mini" aria-label={t("advancedFares.removeArea")} data-tip={t("advancedFares.removeArea") + tipMuted(t("common.shiftSkip"))} onClick={(e) => remove(a, e.shiftKey)}><Icon.cross /></button>
              </span>
            </li>
            {Array.from({ length: n }, (_, k) => {
              const shown = (charged[Math.min(k, charged.length - 1)] ?? 0) * scale;
              const set = (v: number) => setZonePrice(a.id, k + 1, Math.max(0, v) / scale);
              return (
                <li class={a.off ? "af-off" : undefined}>
                  <span class="fr-what" data-tip={k ? t("advancedFares.zonePriceTip", { count: k + 1 }) : t("advancedFares.zonePriceOneTip")}>{t("advancedFares.zonePrice", { count: k + 1 })}</span>
                  <Stepper value={price(shown)} wide label={t("advancedFares.zonePriceLabel", { count: k + 1 })}
                    onStep={(d) => set(fromShown(roundTo(toShown(shown), sp.grain) + d * sp.fareStep))}
                    type={{ text: priceTyped(shown), set: (v) => set(fromShown(v)) }} />
                  <span />
                </li>
              );
            })}
          </>
        );
      })}
      <li class="af-add">
        <span class="fr-what">
          <button class="btn text mini" onClick={() => void newArea().then((id) => { if (id !== null) paint(id); })}><Icon.plus />{t("advancedFares.newArea")}</button>
        </span>
        <span /><span />
      </li>
    </ul>
  );
}

/** T-3915: the kinds' multiples, each only while one of its lines exists, and the transfer fare: high speed, bus and
 * BRT, then the transfer fare, on the left, the boat sizes on the right, where the panel has the room (`.af-cols`);
 * one column, the transfer fare last, where it hasn't, or when there are no boats. */
function KindFareRows({ f, transfer }: { f: Fares; transfer: ComponentChildren }) {
  const w = world.value;
  if (!w) return null;
  const k = fareKindsInUse(w);
  const left = <>
    {k.fast && <HsrFareRow mul={hsrFare(f)} own={hsrFare(f) !== HSR_FARE} />}
    {k.bus && <BusFareRow mul={busFare(f)} />}
    {k.brt && <BusFareRow mul={brtFare(f)} brt />}
  </>;
  if (!k.boats.length) return <>{left}{transfer}</>;
  // Medium's row ("Ferry") with any ferry line, as before T-3915; Small's and Large's with a line of theirs
  const sizes = ([0, 1, 2] as const).filter((b) => b === 1 || k.boats.includes(b));
  return (
    <li class="af-cols">
      <ul class="rows fare-rows">{left}{transfer}</ul>
      <ul class="rows fare-rows"><FerryFareRows fares={f} sizes={sizes} /></ul>
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
  // T-3766: with fares by city the panel lists and shows cities, else countries; a pick of the other kind turns into
  // its country, or the first listed city of its country
  const byName = (a: string, b: string) => placeName(f, a).localeCompare(placeName(f, b));
  const listed = f.byCity ? builtIn(stationCity).sort(byName) : [ref, ...builtIn(stationCountry).filter((c) => c !== ref).sort(byName)];
  const place = !f.byCity ? (isCity(picked) ? zoneCountry(picked) ?? ref : picked)
    : isCity(picked) ? picked : listed.find((c) => zoneCountry(c) === picked) ?? listed[0] ?? fareCity();
  // the Money tab's country or city may have no station built: the picker still offers it
  const options = [...listed, ...(listed.includes(place) ? [] : [place])];
  const { scale, nodes: shown, city, country } = chargedAt(f, place);
  const rate = countryFareScales(country, faresIn(f, city)).rate;
  const nodes = panelPreview.value ?? shown;
  const custom = isCustomCurve(f.nodes);
  const kpu = unitMetres() / 1000, unit = unitShort();
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
        // Esc in a typed value's box only cancels the typing (the box's own handler)
        if (e.key === "Escape") { e.preventDefault(); if (!(e.target instanceof HTMLInputElement)) escape(); return; }
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
            country: <select aria-label={f.byCity ? t("advancedFares.city") : t("advancedFares.country")} value={place} onChange={(e) => { panelPreview.value = null; setPicked(e.currentTarget.value); }}>
              {options.map((c) => <option value={c}>{placeName(f, c)}</option>)}
            </select>,
          })}</span>
          <span class="right"><FareButtons nodes={nodes} /></span>
        </div>
        <FareGraph nodes={shown} scale={scale} big preview={panelPreview} />
        <FareNodeRows nodes={nodes} scale={scale} remove
          last={<>
            {/* T-3915: a thin rule under it, over the multiples and the transfer fare */}
            <li class="af-add af-rule">
              <span class="fr-what">
                <button class={"btn text mini" + (fareAdding.value ? " on" : "")} aria-pressed={fareAdding.value} disabled={full}
                  data-tip={full ? t("advancedFares.full", { n: MAX_NODES }) : undefined}
                  onClick={() => (fareAdding.value = !fareAdding.peek())}><Icon.plus />{t("advancedFares.addPoint")}</button>
                {custom && <button class="btn text mini" onClick={simple}>{t("advancedFares.simple")}</button>}
              </span>
            </li>
            <KindFareRows f={f} transfer={<TransferFareRow fare={transferFare(f)} base={f.nodes[0]?.[1] ?? f.base} scale={scale} />} />
          </>} />
        <FareEffect preview={panelPreview.value} scale={scale} />
        <FareAreas f={f} scale={rate} />
        <ul class="rows fare-rows af-countries">
          <li class="af-head">
            {/* T-3766: fares by country or by city, never both */}
            <span class="af-title af-by">{tRich("advancedFares.faresBy", {
              mode: <Segmented<boolean> options={[[false, t("advancedFares.byCountryMode")], [true, t("advancedFares.byCityMode")]]} value={!!f.byCity}
                onChange={(on) => { if (on !== !!f.byCity) { panelPreview.value = null; void setFareByCity(on); } }} />,
            })}</span>
            <span>{t("advancedFares.multiple")}</span><span>{t("fareGraph.baseFare")}</span><span />
          </li>
          {listed.map((c) => <PlaceRow f={f} place={c} />)}
        </ul>
      </div>
    </dialog>
  );
}

/** The panel while `advancedFares` holds the country it opened on. */
export function AdvancedFares() {
  const start = advancedFares.value;
  return start ? <Panel start={start} /> : null;
}
