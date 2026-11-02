// The Money tab: cash, the ledger, the fare curve and the fleet.

import { busesText, busRouteRule, busRunRule, money, runCostLine, runCostRule, upkeepLines } from "../game/format";
import { borrow, repay, setFareHsr, setFareScale, setFareTransfer, stationsById, world } from "../game/world";
import { LOAN_AMOUNTS, LOAN_RATE, loanAllowed, loanProfit } from "../game/loan";
import { liveMoney, moneyState, stationCountry } from "../game/money";
import { cityById, HOME_CITY, isCity, viewedZone } from "../game/cities";
import { HSR_FARE, hsrFare, scaleNodes, tidyHsr, tidyTransfer, transferFare, type FareNode } from "../game/fareCurve";
import { shownEstimate } from "../game/fareEstimate";
import { roundedDefault } from "../game/fareDefault";
import { FARE_BASELINE, fareScale, SHOWN_STEP, shownScale, storedScale } from "../game/income";
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

const ledgerNet = (d: DayLedger) => d.fares - d.running - (d.upkeep ?? 0) - (d.interest ?? 0) - d.trains - d.build;

/** T-867: the Upkeep row's hover: the rule and the network's track and stations behind it. */
function upkeepTip(m: MoneyState): string {
  return upkeepLines(m); // T-1838: Anita's three lines, nothing else
}

/** T-284: the cash as fares land, to the dollar, as in the top bar. */
function LiveCash() {
  return <Money dollars={liveMoney.value.cash * 1e6} cash />;
}

/** T-284: one figure of today's column as fares land; the network's or a region's (T-531). */
function TodayCell({ v, zone }: { v: (d: DayLedger) => number; zone: string | null }) {
  const d = regionLedger(zone).today;
  return <td class="num">{d ? <Money dollars={v(d) * 1e6} /> : ""}</td>;
}

/** T-531: how a region's money is counted, in the scope toggle's hover. */
const REGION_SPLIT = "A line's fares count where its riders board and travel, its running costs where its trains run. Construction and upkeep count where the track and stations are.";

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

let regionNames: Intl.DisplayNames | null = null;
function countryName(iso: string): string {
  try {
    regionNames ??= new Intl.DisplayNames(["en"], { type: "region" });
    return regionNames.of(iso) ?? iso;
  } catch {
    return iso;
  }
}

/** T-1277: the fare scale where the panel looks, as a multiple of the country's default (T-1906:
 * 1.00x until set): a stepper by 0.05, typed on a click, and a button back to the default once the
 * player has set one. */
function FareScaleRow({ country, scale, own, scales }: { country: string; scale: number; own: boolean; scales: Record<string, number> | undefined }) {
  const set = (v: number) => void setFareScale(country, storedScale(v, country, scales));
  const name = countryName(country);
  const shown = shownScale(scale, country, scales);
  const text = (v: number) => `${v.toFixed(2)}x`;
  return (
    <li>
      <span class="fr-what">{`${name} fares`}</span>
      <Stepper value={text(shown)} wide label="fare scale" onStep={(d) => set(Math.round(shown * 100) / 100 + d * SHOWN_STEP)}
        type={{ text: shown.toFixed(2), set, tip: `Fares in ${name} as a multiple of the default fares.` }} />
      {own
        ? <button class="btn icon" aria-label="Back to the default fares" data-tip={`Back to ${text(1)}`} onClick={() => void setFareScale(country, null)}><Icon.undo /></button>
        : <span />}
    </li>
  );
}

/** T-1626: the fare multiplier on high-speed lines: a stepper by 0.05, typed on a click, and a
 * button back to the default once the player has set another. */
function HsrFareRow({ mul, own }: { mul: number; own: boolean }) {
  const set = (v: number) => void setFareHsr(tidyHsr(v));
  const text = (v: number) => `${v.toFixed(2)}x`;
  return (
    <li>
      <span class="fr-what" data-tip="Fare multiple for rides on high speed lines.">High speed</span>
      <Stepper value={text(mul)} wide label="high-speed fare" onStep={(d) => set(Math.round(mul * 100) / 100 + d * 0.05)}
        type={{ text: mul.toFixed(2), set, tip: "Fare multiple for rides on high speed lines." }} />
      {own
        ? <button class="btn icon" aria-label="Back to the usual multiple" data-tip={`Back to ${text(HSR_FARE)}`} onClick={() => set(HSR_FARE)}><Icon.undo /></button>
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
      <span class="fr-what">Transfer fare</span>
      <Stepper value={`$${shown.toFixed(2)}`} wide label="transfer fare" onStep={(d) => set(Math.round(shown * 100) / 100 + d * 0.25)}
        type={{ text: shown.toFixed(2), set, tip: "Paid each time a trip changes lines." }} />
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
      <Row label="Riders per day">
        <span data-tip="Estimated impact of fare change">{pct > 0 ? "+" : pct < 0 ? "−" : ""}{Math.abs(pct)}%</span>
      </Row>
      <Row label="Fares per day">
        <span data-tip="Estimated impact of fare change">{df > 0 ? "+" : ""}<Money dollars={df} /></span>
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
  return (
    <>
      <ul class="rows">
        {m.mode === "sandbox"
          // T-728: a sandbox's cash is not shown; its money since the start is
          ? <Row label={<span data-tip="Fares minus running costs, upkeep, trains and construction">Since day 1</span>} bold><Money dollars={(liveMoney.value.cash - START_CASH) * 1e6} /></Row>
          : <Row label="Cash" bold><LiveCash /></Row>}
      </ul>
      <table class="ledger">
        <tr>
          <th style={{ textAlign: "left" }}><ScopeToggle tip={REGION_SPLIT} /></th>
          <th>Yesterday</th>
          <th>Today</th>
        </tr>
        <tr>
          <td>Fares</td>
          {cell(prev, (d) => d.fares)}
          <TodayCell zone={zone} v={(d) => d.fares} />
        </tr>
        <tr>
          <td data-tip={runCostLine(m) + (m.fleetBus > 0 || m.busesNeeded > 0 ? `\nBuses cost ${busRunRule(m)}` : "")}>Running trains</td>
          {cell(prev, (d) => -d.running)}
          <TodayCell zone={zone} v={(d) => -d.running} />
        </tr>
        <tr>
          <td data-tip={upkeepTip(m)}>Upkeep</td>
          {cell(prev, (d) => -(d.upkeep ?? 0))}
          <TodayCell zone={zone} v={(d) => -(d.upkeep ?? 0)} />
        </tr>
        {zone === null && (m.loan > 0 || (prev?.interest ?? 0) > 0) && (
          <tr>
            <td data-tip={`Interest on the loan: ${LOAN_RATE * 100}% of it per game day`}>Interest</td>
            {cell(prev, (d) => -(d.interest ?? 0))}
            <TodayCell zone={zone} v={(d) => -(d.interest ?? 0)} />
          </tr>
        )}
        <tr>
          <td data-tip={m.fleetBus > 0 || m.busesNeeded > 0 ? "Cars and buses bought" : undefined}>New trains</td>
          {cell(prev, (d) => -d.trains)}
          <TodayCell zone={zone} v={(d) => -d.trains} />
        </tr>
        <tr>
          <td>Construction</td>
          {cell(prev, (d) => -d.build)}
          <TodayCell zone={zone} v={(d) => -d.build} />
        </tr>
        <tr class="b">
          <td>Net</td>
          {cell(prev, ledgerNet)}
          <TodayCell zone={zone} v={ledgerNet} />
        </tr>
      </table>
      <div class="h">Fare<span class="right"><FareButtons nodes={f} /></span></div>
      <FareGraph nodes={shown} scale={scale} />
      <FareNodeRows nodes={f} scale={scale}
        last={<>
          {hsr !== null && <HsrFareRow mul={hsr} own={hsr !== HSR_FARE} />}
          <TransferFareRow fare={transferFare(m.fares)} scale={scale} />
          <FareScaleRow country={country} scale={scale} own={m.fares.scales?.[country] !== undefined} scales={m.fares.scales} />
        </>} />
      <FareEffect preview={farePreview.value} scale={scale} />
      <MoneyHistory zone={zone} />
      <div class="h">Trains</div>
      <ul class="rows">
        <Row label={<span data-tip={`A car costs ${money(m.carPrice * 1e6)}. Running costs ${runCostRule(m)}.`}>Cars owned</span>}>
          <span data-tip={`${m.fleet.toLocaleString("en-US")} cars bought at ${money(m.carPrice * 1e6)} each, ${money(m.fleet * m.carPrice * 1e6)} in all`}>{m.fleet}</span>
        </Row>
        <Row label="Cars the lines need">
          {m.carsNeeded}
        </Row>
        <Row label="Spare cars">
          {Math.max(0, m.fleet - m.carsNeeded)}
        </Row>
        {(m.fleetFast > 0 || m.carsNeededFast > 0) && (
          <>
            <Row label={<span data-tip={`A high-speed car costs ${money(m.hsCarPrice * 1e6)}. Running costs ${runCostRule({ runCostTrainKm: m.hsRunCostTrainKm, runCostCarKm: m.hsRunCostCarKm })}. Lines with high-speed trains use only these.`}>High-speed cars owned</span>}>
              <span data-tip={`${m.carsNeededFast.toLocaleString("en-US")} on the running lines`}>{m.fleetFast}</span>
            </Row>
            <Row label="Spare high-speed cars">
              {Math.max(0, m.fleetFast - m.carsNeededFast)}
            </Row>
          </>
        )}
        {(m.fleetBus > 0 || m.busesNeeded > 0) && (
          <>
            <Row label={<span data-tip={`A bus costs ${money(m.busPrice * 1e6)}. Running costs ${busRunRule(m)}. A route costs ${busRouteRule(m)} to construct.`}>Buses owned</span>}>
              <span data-tip={`${busesText(m.fleetBus)} bought at ${money(m.busPrice * 1e6)} each, ${money(m.fleetBus * m.busPrice * 1e6)} in all`}>{m.fleetBus}</span>
            </Row>
            <Row label="Buses the lines need">
              {m.busesNeeded}
            </Row>
            <Row label="Spare buses">
              {Math.max(0, m.fleetBus - m.busesNeeded)}
            </Row>
          </>
        )}
      </ul>
      <ul class="rows">
        <Row label="Track and stations built"><Money dollars={w.money.builtValue} /></Row>
      </ul>
      {m.mode !== "sandbox" && <LoanRows loan={m.loan} ledger={m.ledger} />}
    </>
  );
}

/** T-1906: why a loan of `amount` US$M is refused, null if it is not. */
function loanRefusal(amount: number, profit: ReturnType<typeof loanProfit>): string | null {
  if (loanAllowed(amount, profit)) return null;
  if (!profit) return `Until there is a full day of fares, the most you can borrow is $${LOAN_AMOUNTS[0] / 1000}B.`;
  const over = profit.days === 1 ? "yesterday" : `over the last ${profit.days} days`;
  return `Interest would be ${money(amount * LOAN_RATE * 1e6)} per day. Fares less running costs and upkeep averaged ${money(profit.perDay * 1e6)} per day ${over}.`;
}

/** T-1661: the loan at the bottom of the tab: three sizes to borrow while none is out, each refused
 * past what profits pay (T-1906); the loan, its interest a day and Repay while one is. */
function LoanRows({ loan, ledger }: { loan: number; ledger: DayLedger[] }) {
  const rule = `Interest is ${LOAN_RATE * 100}% of the loan per game day. One loan at a time.`;
  if (loan <= 0) {
    const profit = loanProfit(ledger, liveMoney.value.today?.day ?? ledger[ledger.length - 1]?.day ?? 0);
    return (
      <>
        <div class="h">Loan</div>
        <ul class="rows">
          <Row label="Borrow">
            <span class="group">
              {LOAN_AMOUNTS.map((a) => {
                const no = loanRefusal(a, profit);
                return <button class="btn text" disabled={no !== null} data-tip={no ?? `Borrow ${money(a * 1e6)}. ${rule}`} onClick={() => void borrow(a)}>{`$${a / 1000}B`}</button>;
              })}
            </span>
          </Row>
        </ul>
      </>
    );
  }
  const canPay = liveMoney.value.cash >= loan;
  return (
    <>
      <div class="h">Loan</div>
      <ul class="rows">
        <Row label="Borrowed"><Money dollars={loan * 1e6} /></Row>
        <Row label="Interest per day"><span data-tip={rule}><Money dollars={loan * LOAN_RATE * 1e6} /></span></Row>
        <Row label={<button class="btn text" disabled={!canPay} data-tip={canPay ? `Pay back ${money(loan * 1e6)} from the cash` : `Needs ${money(loan * 1e6)} in cash`} onClick={() => void repay()}>Repay</button>} />
      </ul>
    </>
  );
}
