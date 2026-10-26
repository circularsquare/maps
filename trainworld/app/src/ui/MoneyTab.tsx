// The Money tab: cash, the ledger, the fare curve and the fleet.

import { money, runCostRule, shortDistance, upkeepText } from "../game/format";
import { setFareScale, stationsById, world } from "../game/world";
import { liveMoney, moneyState, stationCountry } from "../game/money";
import { cityById, HOME_CITY, isCity, viewedZone } from "../game/cities";
import { scaleNodes, type FareNode } from "../game/fareCurve";
import { shownEstimate } from "../game/fareEstimate";
import { FARE_BASELINE, fareScale, incomeScale, SCALE_STEP, tidyScale } from "../game/income";
import { selection } from "../game/ui";
import { scopeZone } from "../game/listScope";
import { regionLedger } from "../game/regionMoney";
import type { DayLedger, MoneyState } from "../workers/protocol";
import { Icon, Row, Stepper } from "./widgets";
import { Money } from "./MoneyText";
import { START_CASH } from "../workers/saveFile";
import { FareButtons, FareGraph, FareNodeRows, farePreview } from "./FareGraph";
import { MoneyHistory } from "./MoneyHistory";
import { ScopeToggle } from "./listParts";

const ledgerNet = (d: DayLedger) => d.fares - d.running - (d.upkeep ?? 0) - d.trains - d.build;

/** T-867: the Upkeep row's hover: the rule and the network's track and stations behind it. */
function upkeepTip(m: MoneyState): string {
  const u = m.upkeep;
  return `${upkeepText(m, "Upkeep")}\n`
    + `The network has ${shortDistance(u.km * 1000)} of track and ${u.count.toLocaleString("en-US")} stations: ${money((u.track + u.stations) * 1e6)} a day.`;
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

/** T-1277: the country whose fares the panel shows: the selected station's or city's, else the
 * zone in view's, else the home city's. */
function fareCountry(): string {
  const s = selection.value, w = world.value;
  let c: string | null | undefined = null;
  if (s?.kind === "station" && w) {
    const st = stationsById(w).get(s.station);
    c = st && stationCountry(st);
  } else if (s?.kind === "city" && isCity(s.city)) c = cityById(s.city).country;
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

/** T-1277: the fare scale where the panel looks: a stepper by tenths, typed on a click, and a button
 * back to the income's scale once the player has set one. */
function FareScaleRow({ country, scale, own }: { country: string; scale: number; own: boolean }) {
  const set = (v: number) => void setFareScale(country, tidyScale(v));
  const income = incomeScale(country);
  const name = countryName(country);
  return (
    <li>
      <span class="fr-what" data-tip={`Fares shown for ${name}: ${scale.toFixed(1)} times those in ${countryName(FARE_BASELINE)}, where the curve is set. By income it would be ${income.toFixed(1)}.`}>Fares here</span>
      <Stepper value={`${scale.toFixed(1)}x`} wide label="fares here" onStep={(d) => set(scale + d * SCALE_STEP)}
        type={{ text: scale.toFixed(1), set, tip: `Fares in ${name} as a multiple of those in ${countryName(FARE_BASELINE)}. Click to type one.` }} />
      {own
        ? <button class="btn icon" aria-label="Back to the income's scale" data-tip={`Back to ${income.toFixed(1)}x, the scale by income`} onClick={() => void setFareScale(country, null)}><Icon.undo /></button>
        : <span />}
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
  const count = (v: number) => (Math.round(v / 1e4) * 1e4).toLocaleString("en-US");
  return (
    <ul class="rows">
      <Row label="Riders a day">
        <span data-tip={`About ${count(e.riders)} at these fares, ${count(e.ridersNow)} now. A guess until demand catches up.`}>{pct > 0 ? "+" : pct < 0 ? "−" : ""}{Math.abs(pct)}%</span>
      </Row>
      <Row label="Fares a day">
        <span data-tip={`About ${money(e.fares)} at these fares, ${money(e.faresNow)} now. A guess until demand catches up.`}>{df > 0 ? "+" : ""}<Money dollars={df} /></span>
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
  const shown = scaleNodes(m.fares.nodes, scale);
  const f = farePreview.value ?? shown;
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
          <td data-tip={`Running costs ${runCostRule(m)}`}>Running trains</td>
          {cell(prev, (d) => -d.running)}
          <TodayCell zone={zone} v={(d) => -d.running} />
        </tr>
        <tr>
          <td data-tip={upkeepTip(m)}>Upkeep</td>
          {cell(prev, (d) => -(d.upkeep ?? 0))}
          <TodayCell zone={zone} v={(d) => -(d.upkeep ?? 0)} />
        </tr>
        <tr>
          <td>New trains</td>
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
      <MoneyHistory zone={zone} />
      <div class="h">Fare<span class="right"><FareButtons nodes={f} /></span></div>
      <FareGraph nodes={shown} scale={scale} />
      <FareNodeRows nodes={f} scale={scale} first={<FareScaleRow country={country} scale={scale} own={m.fares.scales?.[country] !== undefined} />} />
      <FareEffect preview={farePreview.value} scale={scale} />
      <div class="h">Trains</div>
      <ul class="rows">
        <Row label={<span data-tip={`A car costs ${money(m.carPrice * 1e6)}. Running costs ${runCostRule(m)}. Cars are bought when a line needs more than the spare ones.`}>Cars owned</span>}>
          <span data-tip={`${m.fleet.toLocaleString("en-US")} cars bought at ${money(m.carPrice * 1e6)} each, ${money(m.fleet * m.carPrice * 1e6)} in all`}>{m.fleet}</span>
        </Row>
        <Row label="Cars the lines need">
          <span data-tip="Cars on the running lines at the demand level that needs most">{m.carsNeeded}</span>
        </Row>
        <Row label="Spare cars">
          <span data-tip="Cars kept in the depot after a line shrank. A line that grows uses them before any are bought.">{Math.max(0, m.fleet - m.carsNeeded)}</span>
        </Row>
        {(m.fleetFast > 0 || m.carsNeededFast > 0) && (
          <>
            <Row label={<span data-tip={`A high-speed car costs ${money(m.hsCarPrice * 1e6)}. Running costs ${runCostRule({ runCostTrainKm: m.hsRunCostTrainKm, runCostCarKm: m.hsRunCostCarKm })}. Lines with high-speed trains use only these.`}>High-speed cars owned</span>}>
              <span data-tip={`${m.carsNeededFast.toLocaleString("en-US")} on the running lines`}>{m.fleetFast}</span>
            </Row>
            <Row label="Spare high-speed cars">
              <span data-tip="High-speed cars kept in the depot. A line with high-speed trains uses them before any are bought.">{Math.max(0, m.fleetFast - m.carsNeededFast)}</span>
            </Row>
          </>
        )}
      </ul>
      <ul class="rows">
        <Row label="Track and stations built"><Money dollars={w.money.builtValue} /></Row>
      </ul>
    </>
  );
}
