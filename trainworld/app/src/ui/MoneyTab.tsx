// The Money tab: cash, the ledger, the fare curve and the fleet.

import { boatRunRule, boatsText, busesText, busRouteRule, busRunRule, currencySpec, fromShown, money, moneyCash, price, priceTyped, runCostHere, runCostLine, runCostRule, toShown, upkeepLines } from "../game/format";
import { roundTo } from "../game/currency";
import { borrow, repay, setFareBus, setFareFerry, setFareHsr, setFareRef, setFareScale, setFareTransfer, stationsById, world } from "../game/world";
import { LOAN_AMOUNTS, LOAN_RATE, LOAN_TERM, loanAllowed, loanBalance, loanPayment, loanProfit, type Loan } from "../game/loan";
import { liveMoney, moneyState, stationCountry } from "../game/money";
import { cityById, HOME_CITY, isCity, viewedZone } from "../game/cities";
import { BUS_FARE, busFare, FERRY_FARE, ferryFare, HSR_FARE, hsrFare, scaleNodes, tidyHsr, tidyTransfer, transferFare, type FareNode, type Fares } from "../game/fareCurve";
import { shownEstimate } from "../game/fareEstimate";
import { roundedDefault } from "../game/fareDefault";
import { defaultScale, FARE_BASELINE, fareScale, runScale, SHOWN_STEP, shownScale, storedScale } from "../game/income";
import { gameStartZone } from "../game/realms";
import { selection } from "../game/ui";
import { scopeZone, viewCentre } from "../game/listScope";
import { regionLedger } from "../game/regionMoney";
import type { DayLedger, MoneyState } from "../workers/protocol";
import { Icon, Row, Stepper } from "./widgets";
import { Money } from "./MoneyText";
import { START_CASH } from "../workers/saveFile";
import { FareButtons, FareGraph, FareNodeRows, farePreview } from "./FareGraph";
import { MoneyHistory } from "./MoneyHistory";
import { ScopeToggle } from "./listParts";
import { t, tCountry, tNumber } from "../i18n";

/** two decimals, in the language's digits */
const fix2 = (v: number) => tNumber(v, { minimumFractionDigits: 2, maximumFractionDigits: 2 });

const ledgerNet = (d: DayLedger) => d.fares - d.running - (d.upkeep ?? 0) - (d.loan ?? 0) - d.trains - d.build;

/** T-867: the Upkeep row's hover: the rule and the network's track and stations behind it. */
function upkeepTip(m: MoneyState): string {
  return upkeepLines(m); // T-1838: Anita's three lines, nothing else
}

/** The New trains row's hover once the network has buses or boats (T-1464, T-2194). */
function vehiclesTip(m: MoneyState): string | undefined {
  const bus = m.fleetBus > 0 || m.busesNeeded > 0, boat = m.fleetFerry > 0 || m.boatsNeeded > 0;
  return bus && boat ? t("moneyTab.boughtAll") : bus ? t("moneyTab.boughtBuses") : boat ? t("moneyTab.boughtBoats") : undefined;
}

/** T-284: the cash as fares land, to the dollar, as in the top bar. */
function LiveCash() {
  return <Money dollars={liveMoney.value.cash * 1e6} cash full />;
}

/** T-284: one figure of today's column as fares land; the network's or a region's (T-531). */
function TodayCell({ v, zone }: { v: (d: DayLedger) => number; zone: string | null }) {
  const d = regionLedger(zone).today;
  return <td class="num">{d ? <Money dollars={v(d) * 1e6} /> : ""}</td>;
}

/** T-531: how a region's money is counted, in the scope toggle's hover. */
const regionSplit = () => t("moneyTab.regionSplit");

/** T-1690: the country of the built station nearest the view's centre, null with none built. */
function nearestBuilt(w: NonNullable<typeof world.value>): string | null {
  const at = viewCentre.value;
  if (!at) return null;
  const [lng, lat] = at, k = Math.cos((lat * Math.PI) / 180);
  let best = Infinity, country: string | null = null;
  for (const st of w.save.stations) {
    if (!st.built) continue;
    const d = ((st.lng - lng + 540) % 360 - 180) ** 2 * k * k + (st.lat - lat) ** 2;
    if (d >= best) continue;
    const c = stationCountry(st);
    if (c) { best = d; country = c; }
  }
  return country;
}

/** T-1277: the country whose fares the panel shows: the selected station's or city's, else the
 * nearest country the player has built in (T-1690), else the zone in view's, else the home city's. */
function fareCountry(): string {
  const s = selection.value, w = world.value;
  let c: string | null | undefined = null;
  if (s?.kind === "station" && w) {
    const st = stationsById(w).get(s.station);
    c = st && stationCountry(st);
  } else if (s?.kind === "city" && isCity(s.city)) c = cityById(s.city).country;
  if (!c && w) c = nearestBuilt(w);
  if (!c) {
    const z = viewedZone.value;
    c = z && isCity(z) ? cityById(z).country : null;
  }
  return c ?? cityById(HOME_CITY).country ?? FARE_BASELINE;
}


/** T-2679: the reference country, whose fares the panel's multiples are of: the game's (its first
 * built station's, or the player's pick), else the start zone's, else the home city's. */
function fareRef(f: Fares): string {
  const z = gameStartZone();
  return f.ref ?? (z && isCity(z) ? cityById(z).country : undefined) ?? cityById(HOME_CITY).country ?? FARE_BASELINE;
}

/** T-1277: the fare scale where the panel looks, as a multiple of the reference country's (T-2679):
 * the reference itself a plain 1.00x; any other a stepper to the next 0.05x up or down (a default is
 * an odd multiple like 0.63x), typed on a click, a button that
 * makes it the reference, and one back to its default once the player has set one. */
function FareScaleRow({ country, scale, own, scales, refCountry: ref }: { country: string; scale: number; own: boolean; scales: Record<string, number> | undefined; refCountry: string }) {
  const name = tCountry(country);
  const text = (v: number) => `${fix2(v)}x`;
  if (country === ref) {
    return (
      <li>
        <span class="fr-what">{t("moneyTab.countryFares", { name })}</span>
        <span class="fr-ref" data-tip={t("moneyTab.referenceTip", { name })}>{text(1)}</span>
        <span />
      </li>
    );
  }
  const refScale = fareScale(ref, scales);
  const set = (v: number) => void setFareScale(country, storedScale(v, country, scales, refScale));
  const shown = shownScale(scale, refScale);
  return (
    <li>
      <span class="fr-what">{t("moneyTab.countryFares", { name })}</span>
      <Stepper value={text(shown)} wide label={t("moneyTab.fareScale")} onStep={(d) => set((d > 0 ? Math.floor(shown / SHOWN_STEP + 1e-6) + d : Math.ceil(shown / SHOWN_STEP - 1e-6) + d) * SHOWN_STEP)}
        type={{ text: shown.toFixed(2), set, tip: t("moneyTab.countryFaresTip", { name, ref: tCountry(ref) }) }} />
      <span class="group">
        <button class="btn icon" aria-label={t("moneyTab.makeRef")} data-tip={`${t("moneyTab.makeRef")}\n${t("moneyTab.makeRefSame")}`} onClick={() => void setFareRef(country)}><Icon.anchor /></button>
        {own && <button class="btn icon" aria-label={t("moneyTab.backToDefault")} data-tip={t("moneyTab.backTo", { value: text(shownScale(defaultScale(country, scales), refScale)) })} onClick={() => void setFareScale(country, null)}><Icon.undo /></button>}
      </span>
    </li>
  );
}

/** T-1626: the fare multiplier on high-speed lines: a stepper by 0.05, typed on a click, and a
 * button back to the default once the player has set another. */
function HsrFareRow({ mul, own }: { mul: number; own: boolean }) {
  const set = (v: number) => void setFareHsr(tidyHsr(v));
  const text = (v: number) => `${fix2(v)}x`;
  return (
    <li>
      <span class="fr-what" data-tip={t("moneyTab.hsrTip")}>{t("moneyTab.hsr")}</span>
      <Stepper value={text(mul)} wide label={t("moneyTab.hsrFare")} onStep={(d) => set(Math.round(mul * 100) / 100 + d * 0.05)}
        type={{ text: mul.toFixed(2), set, tip: t("moneyTab.hsrTip") }} />
      {own
        ? <button class="btn icon" aria-label={t("moneyTab.backToUsual")} data-tip={t("moneyTab.backTo", { value: text(HSR_FARE) })} onClick={() => set(HSR_FARE)}><Icon.undo /></button>
        : <span />}
    </li>
  );
}

/** T-3028: the fare multiplier on bus lines, as the ferry one. */
function BusFareRow({ mul }: { mul: number }) {
  const set = (v: number) => void setFareBus(tidyHsr(v));
  const text = (v: number) => `${fix2(v)}x`;
  return (
    <li>
      <span class="fr-what" data-tip={t("moneyTab.busFareTip")}>{t("moneyTab.busFareRow")}</span>
      <Stepper value={text(mul)} wide label={t("moneyTab.busFare")} onStep={(d) => set(Math.round(mul * 100) / 100 + d * 0.05)}
        type={{ text: mul.toFixed(2), set, tip: t("moneyTab.busFareTip") }} />
      {mul !== BUS_FARE
        ? <button class="btn icon" aria-label={t("moneyTab.backToUsual")} data-tip={t("moneyTab.backTo", { value: text(BUS_FARE) })} onClick={() => set(BUS_FARE)}><Icon.undo /></button>
        : <span />}
    </li>
  );
}

/** T-2194: the fare multiplier on ferry lines, as the high-speed one (notes/T-2177.md 3.7). */
function FerryFareRow({ mul }: { mul: number }) {
  const set = (v: number) => void setFareFerry(tidyHsr(v));
  const text = (v: number) => `${fix2(v)}x`;
  return (
    <li>
      <span class="fr-what" data-tip={t("moneyTab.ferryTip")}>{t("moneyTab.ferry")}</span>
      <Stepper value={text(mul)} wide label={t("moneyTab.ferryFare")} onStep={(d) => set(Math.round(mul * 100) / 100 + d * 0.05)}
        type={{ text: mul.toFixed(2), set, tip: t("moneyTab.ferryTip") }} />
      {mul !== FERRY_FARE
        ? <button class="btn icon" aria-label={t("moneyTab.backToUsual")} data-tip={t("moneyTab.backTo", { value: text(FERRY_FARE) })} onClick={() => set(FERRY_FARE)}><Icon.undo /></button>
        : <span />}
    </li>
  );
}

/** T-1801: the transfer fare as charged where the panel looks (times its scale): a stepper by
 * $0.25, typed on a click, as the base fare. */
function TransferFareRow({ fare, scale }: { fare: number; scale: number }) {
  const shown = fare * scale;
  const set = (v: number) => void setFareTransfer(tidyTransfer(Math.max(0, v) / scale));
  return (
    <li>
      <span class="fr-what">{t("moneyTab.transferFare")}</span>
      {/* T-2798: in the shown currency, stepped from the figure shown, back to US$ for the game */}
      <Stepper value={price(shown)} wide label={t("moneyTab.transferFareLabel")} onStep={(d) => set(fromShown(roundTo(toShown(shown), currencySpec().grain) + d * currencySpec().fareStep))}
        type={{ text: priceTyped(shown), set: (v) => set(fromShown(v)), tip: t("moneyTab.transferTip") }} />
      <span />
    </li>
  );
}

/** T-1300: what the fare being dragged, or one demand has yet to solve, would do to rail trips and
 * fare income a day; nothing otherwise. */
function FareEffect({ preview, scale }: { preview: FareNode[] | null; scale: number }) {
  const e = shownEstimate(preview && scaleNodes(preview, 1 / scale));
  if (!e) return null;
  const dr = e.riders / e.ridersNow - 1, df = e.fares - e.faresNow;
  const pct = Math.round(100 * dr);
  return (
    <ul class="rows">
      <Row label={t("moneyTab.ridersPerDay")}>
        <span data-tip={t("moneyTab.fareImpact")}>{pct > 0 ? "+" : pct < 0 ? "−" : ""}{tNumber(Math.abs(pct))}%</span>
      </Row>
      <Row label={t("moneyTab.faresPerDay")}>
        <span data-tip={t("moneyTab.fareImpact")}>{df > 0 ? "+" : ""}<Money dollars={df} /></span>
      </Row>
    </ul>
  );
}

export function MoneyTab() {
  const w = world.value;
  const m = moneyState.value;
  if (!w || !m) return null;
  // T-284: the cash and today's column follow fares as they land (`liveMoney`, own components)
  // T-531: the ledger is the viewed region's unless the whole world is chosen, as in the lists
  const zone = scopeZone();
  const todayDay = m.ledger[m.ledger.length - 1]?.day ?? 0;
  const prevAll = m.ledger.find((d) => d.day === todayDay - 1) ?? null;
  const prev = zone === null || !prevAll ? prevAll
    : regionLedger(zone).ledger.find((d) => d.day === todayDay - 1) ?? { day: todayDay - 1, fares: 0, running: 0, trains: 0, build: 0 };
  const cell = (d: DayLedger | null, v: (d: DayLedger) => number) => <td class="num">{d ? <Money dollars={v(d) * 1e6} /> : ""}</td>;
  // T-1277: the curve as charged in the country of the selected station or city, else the one in view
  const country = fareCountry();
  const scale = fareScale(country, m.fares.scales);
  const shown = roundedDefault(m.fares, country)?.nodes ?? scaleNodes(m.fares.nodes, scale);
  const f = farePreview.value ?? shown;
  // T-1626: shown once a line runs high-speed trains
  const hsr = w.save.lines.some((l) => w.lineStats[l.id]?.fast) ? hsrFare(m.fares) : null;
  // T-2194: shown once a ferry line exists
  const ferry = w.save.lines.some((l) => w.lineStats[l.id]?.bus?.ferry) ? ferryFare(m.fares) : null;
  // T-3028: shown once a bus line exists
  const busLine = w.save.lines.some((l) => { const b = w.lineStats[l.id]?.bus; return !!b && !b.ferry; }) ? busFare(m.fares) : null;
  return (
    <>
      <ul class="rows">
        {m.mode === "sandbox"
          // T-728: a sandbox's cash is not shown; its money since the start is
          ? <Row label={<span data-tip={t("moneyTab.sinceDay1Tip")}>{t("moneyTab.sinceDay1")}</span>} bold><Money dollars={(liveMoney.value.cash - START_CASH) * 1e6} /></Row>
          : <Row label={t("moneyTab.cash")} bold><LiveCash /></Row>}
      </ul>
      <table class="ledger">
        <tr>
          <th style={{ textAlign: "left" }}><ScopeToggle tip={regionSplit()} /></th>
          <th>{t("moneyTab.yesterday")}</th>
          <th>{t("moneyTab.today")}</th>
        </tr>
        <tr>
          <td>{t("moneyTab.fares")}</td>
          {cell(prev, (d) => d.fares)}
          <TodayCell zone={zone} v={(d) => d.fares} />
        </tr>
        <tr>
          <td data-tip={runCostLine(m) + (runScale(country) !== 1 ? "\n" + runCostHere(tCountry(country), runScale(country)) : "") + (m.fleetBus > 0 || m.busesNeeded > 0 ? "\n" + t("moneyTab.busesCost", { rule: busRunRule(m) }) : "") + (m.fleetFerry > 0 || m.boatsNeeded > 0 ? "\n" + t("moneyTab.boatsCost", { rule: boatRunRule(m) }) : "")}>{t("moneyTab.runningTrains")}</td>
          {cell(prev, (d) => -d.running)}
          <TodayCell zone={zone} v={(d) => -d.running} />
        </tr>
        <tr>
          <td data-tip={upkeepTip(m)}>{t("moneyTab.upkeep")}</td>
          {cell(prev, (d) => -(d.upkeep ?? 0))}
          <TodayCell zone={zone} v={(d) => -(d.upkeep ?? 0)} />
        </tr>
        {zone === null && (m.loan || (prev?.loan ?? 0) > 0) && (
          <tr>
            <td data-tip={t("moneyTab.loanTerm", { days: LOAN_TERM })}>{t("moneyTab.ledgerLoan")}</td>
            {cell(prev, (d) => -(d.loan ?? 0))}
            <TodayCell zone={zone} v={(d) => -(d.loan ?? 0)} />
          </tr>
        )}
        <tr>
          <td data-tip={vehiclesTip(m)}>{t("moneyTab.newTrains")}</td>
          {cell(prev, (d) => -d.trains)}
          <TodayCell zone={zone} v={(d) => -d.trains} />
        </tr>
        <tr>
          <td>{t("moneyTab.construction")}</td>
          {cell(prev, (d) => -d.build)}
          <TodayCell zone={zone} v={(d) => -d.build} />
        </tr>
        <tr class="b">
          <td>{t("moneyTab.net")}</td>
          {cell(prev, ledgerNet)}
          <TodayCell zone={zone} v={ledgerNet} />
        </tr>
      </table>
      <div class="h">{t("moneyTab.fare")}<span class="right"><FareButtons nodes={f} /></span></div>
      <FareGraph nodes={shown} scale={scale} />
      <FareNodeRows nodes={f} scale={scale}
        last={<>
          {hsr !== null && <HsrFareRow mul={hsr} own={hsr !== HSR_FARE} />}
          {busLine !== null && <BusFareRow mul={busLine} />}
          {ferry !== null && <FerryFareRow mul={ferry} />}
          <TransferFareRow fare={transferFare(m.fares)} scale={scale} />
          <FareScaleRow country={country} scale={scale} own={m.fares.scales?.[country] !== undefined} scales={m.fares.scales} refCountry={fareRef(m.fares)} />
        </>} />
      <FareEffect preview={farePreview.value} scale={scale} />
      <MoneyHistory zone={zone} />
      <div class="h">{t("moneyTab.trains")}</div>
      <ul class="rows">
        <Row label={<span data-tip={t("moneyTab.carTip", { price: money(m.carPrice * 1e6), rule: runCostRule(m) })}>{t("moneyTab.carsOwned")}</span>}>
          <span data-tip={t("moneyTab.carsBought", { count: m.fleet, price: money(m.carPrice * 1e6), total: money(m.fleet * m.carPrice * 1e6) })}>{m.fleet}</span>
        </Row>
        <Row label={t("moneyTab.carsNeeded")}>
          {m.carsNeeded}
        </Row>
        <Row label={t("moneyTab.carsSpare")}>
          {Math.max(0, m.fleet - m.carsNeeded)}
        </Row>
        {(m.fleetFast > 0 || m.carsNeededFast > 0) && (
          <>
            <Row label={<span data-tip={t("moneyTab.hsCarTip", { price: money(m.hsCarPrice * 1e6), rule: runCostRule({ runCostTrainKm: m.hsRunCostTrainKm, runCostCarKm: m.hsRunCostCarKm }) })}>{t("moneyTab.hsCarsOwned")}</span>}>
              <span data-tip={t("moneyTab.hsCarsRunning", { count: tNumber(m.carsNeededFast) })}>{m.fleetFast}</span>
            </Row>
            <Row label={t("moneyTab.hsCarsSpare")}>
              {Math.max(0, m.fleetFast - m.carsNeededFast)}
            </Row>
          </>
        )}
        {(m.fleetTram > 0 || m.tramsNeeded > 0) && (
          <>
            <Row label={<span data-tip={t("moneyTab.tramCarTip", { price: money(m.tramPrice * 1e6), rule: runCostRule({ runCostTrainKm: m.runCostTrainKm, runCostCarKm: m.runCostCarKm }) })}>{t("moneyTab.tramCarsOwned")}</span>}>
              <span data-tip={t("moneyTab.hsCarsRunning", { count: tNumber(m.tramsNeeded) })}>{m.fleetTram}</span>
            </Row>
            <Row label={t("moneyTab.tramCarsSpare")}>
              {Math.max(0, m.fleetTram - m.tramsNeeded)}
            </Row>
          </>
        )}
        {(m.fleetBus > 0 || m.busesNeeded > 0) && (
          <>
            <Row label={<span data-tip={t("moneyTab.busTip", { price: money(m.busPrice * 1e6), rule: busRunRule(m), route: busRouteRule(m) })}>{t("moneyTab.busesOwned")}</span>}>
              <span data-tip={t("moneyTab.busesBought", { buses: busesText(m.fleetBus), price: money(m.busPrice * 1e6), total: money(m.fleetBus * m.busPrice * 1e6) })}>{m.fleetBus}</span>
            </Row>
            <Row label={t("moneyTab.busesNeeded")}>
              {m.busesNeeded}
            </Row>
            <Row label={t("moneyTab.busesSpare")}>
              {Math.max(0, m.fleetBus - m.busesNeeded)}
            </Row>
          </>
        )}
        {(m.fleetFerry > 0 || m.boatsNeeded > 0) && (
          <>
            <Row label={<span data-tip={t("moneyTab.boatTip", { price: money(m.boatPrice * 1e6), rule: boatRunRule(m), port: money(m.portPrice * 1e6) })}>{t("moneyTab.boatsOwned")}</span>}>
              <span data-tip={t("moneyTab.boatsBought", { boats: boatsText(m.fleetFerry), price: money(m.boatPrice * 1e6), total: money(m.fleetFerry * m.boatPrice * 1e6) })}>{m.fleetFerry}</span>
            </Row>
            <Row label={t("moneyTab.boatsNeeded")}>
              {m.boatsNeeded}
            </Row>
            <Row label={t("moneyTab.boatsSpare")}>
              {Math.max(0, m.fleetFerry - m.boatsNeeded)}
            </Row>
          </>
        )}
      </ul>
      <ul class="rows">
        <Row label={t("moneyTab.built")}><Money dollars={w.money.builtValue} /></Row>
      </ul>
      {m.mode !== "sandbox" && <LoanRows loan={m.loan} ledger={m.ledger} />}
    </>
  );
}

/** T-1906: why a loan of `amount` US$M is refused, null if it is not. */
function loanRefusal(amount: number, profit: ReturnType<typeof loanProfit>): string | null {
  if (loanAllowed(amount, profit)) return null;
  if (!profit) return t("moneyTab.loanFirst", { amount: money(LOAN_AMOUNTS[0] * 1e6) });
  return t("moneyTab.loanRefused", { payment: money(loanPayment(amount) * 1e6), yesterday: profit.days === 1, days: profit.days, profit: money(profit.perDay * 1e6) });
}

/** T-1661: the loan at the bottom of the tab: three sizes to borrow while none is out, each refused
 * past what profits pay (T-1906); the loan, its payment a day, the days left and Repay while one is (T-2022). */
function LoanRows({ loan, ledger }: { loan: Loan | null; ledger: DayLedger[] }) {
  if (!loan) {
    const profit = loanProfit(ledger, liveMoney.value.today?.day ?? ledger[ledger.length - 1]?.day ?? 0);
    return (
      <>
        <div class="h">{t("moneyTab.loan")}</div>
        <ul class="rows">
          <Row label={t("moneyTab.borrow")}>
            <span class="group">
              {LOAN_AMOUNTS.map((a) => {
                const no = loanRefusal(a, profit);
                return <button class="btn text" disabled={no !== null} data-tip={no ?? t("moneyTab.borrowTip", { amount: money(a * 1e6), payment: money(loanPayment(a) * 1e6), days: LOAN_TERM })} onClick={() => void borrow(a)}>{moneyCash(a * 1e6)}</button>;
              })}
            </span>
          </Row>
        </ul>
      </>
    );
  }
  const left = loanBalance(loan);
  const canPay = liveMoney.value.cash >= left;
  return (
    <>
      <div class="h">{t("moneyTab.loan")}</div>
      <ul class="rows">
        <Row label={t("moneyTab.borrowed")}><Money dollars={loan.amount * 1e6} /></Row>
        <Row label={t("moneyTab.paymentPerDay")}><span data-tip={t("moneyTab.interest", { rate: tNumber(LOAN_RATE * 100) })}><Money dollars={loan.pay * 1e6} /></span></Row>
        <Row label={t("moneyTab.daysLeft")}>{Math.ceil(loan.days)}</Row>
        <Row label={<button class="btn text" disabled={!canPay} data-tip={canPay ? t("moneyTab.repayTip", { amount: money(left * 1e6) }) : t("moneyTab.repayNeeds", { amount: money(left * 1e6) })} onClick={() => void repay()}>{t("moneyTab.repay")}</button>} />
      </ul>
    </>
  );
}
