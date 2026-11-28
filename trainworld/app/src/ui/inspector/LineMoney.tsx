// T-2787, T-3566: a line's past month of money at the bottom of its panel: each of the last 30 whole days its
// fares (blue) and running cost (red) side by side, the Money tab history's chart (`FareCostBars`), a young line's
// few days at the right. The clock worker keeps each line's days (`MoneyState.lineDays`) and saves them; upkeep is
// not in them, it is not kept per line per day. Days the line did not run have no bars.

import { useEffect, useState } from "preact/hooks";
import { dateName, shortDate } from "../../game/days";
import { FARE_MAX, LINE_FARE_MAX, lineFareMul, ownLineFare, tidyLineFare } from "../../game/fareCurve";
import { baseFareIn } from "../../game/fareDefault";
import { money, moneyWhole } from "../../game/format";
import { liveMoney, moneyState, stationCity } from "../../game/money";
import { boatOf } from "../../game/boats";
import { busStopsById, edit, setLineFare, stationsById, world } from "../../game/world";
import { shown as pendingShown } from "../../game/pending";
import type { LineInput } from "../../game/types";
import type { Vehicle } from "../../game/vehicle";
import { t, tNumber, tRich } from "../../i18n";
import { barTop, FareCostBars, type BarCol } from "../MoneyHistory";
import { runningRow } from "../MoneyText";
import { Icon, Segmented, Stepper } from "../widgets";

const H = 64, DAYS = 30, CAP_SHOWN_MS = 5000;

/** T-3567: the line's own fare multiple, on top of the Money tab's fares: a stepper by 0.05, typed on a click, as
 * the Money tab's multiples, and a button back to 1.00x once set. T-3758: above it, while the game has a fare area,
 * whether the line's rides pay zone fares or the distance fare. T-4253: up to 10x, refused when the line's base fare
 * times it would exceed `FARE_MAX`. */
export function LineFare({ l }: { l: LineInput }) {
  const m = moneyState.value, w = world.value;
  const [capAt, setCapAt] = useState(0);
  useEffect(() => { setCapAt(0); }, [l.id]);
  useEffect(() => {
    if (!capAt) return;
    const id = setTimeout(() => setCapAt(0), CAP_SHOWN_MS);
    return () => clearTimeout(id);
  }, [capAt]);
  if (!m) return null;
  const mul = ownLineFare(m.fares, l.id);
  const text = (v: number) => `${tNumber(v, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}x`;
  // highest base fare charged across the line's stops, US$ (its own country's, or city's with fares by city)
  let baseUsd = 0;
  if (w && l.stops.length) {
    const stAt = stationsById(w), busAt = busStopsById(w);
    for (const id of l.stops) {
      const s = stAt.get(id) ?? busAt.get(id);
      baseUsd = Math.max(baseUsd, baseFareIn(m.fares, s ? stationCity(s) : null));
    }
  } else baseUsd = baseFareIn(m.fares, null);
  const set = (v: number): boolean => {
    const next = tidyLineFare(v);
    if (next > mul && baseUsd * next > FARE_MAX + 1e-6) { setCapAt(Date.now()); return false; }
    setCapAt(0);
    if (next !== mul) void setLineFare(l.num, next);
    return true;
  };
  const st = w?.lineStats[l.id];
  const dist = !!pendingShown(`line:${l.num}`, "fareDist", st?.fareDist);
  // the kind's multiple from the fares, read by the same function the fare uses; the line's own is the stepper
  const kindMul = st ? lineFareMul({ id: l.id, fast: st.fast, ferry: !!st.bus?.ferry, boat: boatOf(st.bus), bus: !!st.bus && !st.bus.ferry, brt: !!st.bus?.brt }, { ...m.fares, lines: undefined }) : 1;
  const kind = st?.bus?.ferry ? "ferry" : st?.bus?.brt ? "brt" : st?.bus ? "bus" : "hsr";
  return (
    <ul class="rows">
      {!!m.fares.zones?.length && (
        <li>
          <span class="grow">{t("lineMoney.fareMode")}</span>
          <Segmented<boolean> options={[[false, t("lineMoney.fareZones")], [true, t("lineMoney.fareDistance")]]} value={dist}
            onChange={(to) => { if (to !== dist) void edit({ op: "lineFareDist", line: l.num, dist: to }); }} />
        </li>
      )}
      <li>
        <span class="grow" data-tip={t("lineMoney.fareTip")}>{t("lineMoney.fare")}</span>
        {kindMul !== 1 && <span class="muted small" data-tip={t("lineMoney.farePaid", { value: text(kindMul * mul) })}>{t("lineMoney.fareKind", { kind, value: text(kindMul) })}</span>}
        {mul !== 1 && <button class="btn icon" aria-label={t("lineMoney.fareBack", { value: text(1) })} data-tip={t("lineMoney.fareBack", { value: text(1) })} onClick={() => set(1)}><Icon.undo /></button>}
        <Stepper value={text(mul)} wide label={t("lineMoney.fareLabel")} onStep={(d) => { if (d < 0 ? mul <= 0 : mul >= LINE_FARE_MAX) return false; return set(Math.round(mul * 100) / 100 + d * 0.05); }}
          type={{ text: mul.toFixed(2), set, tip: t("lineMoney.fareTip") }} />
      </li>
      {capAt > 0 && <li class="line-fare-cap" role="alert">{t("lineMoney.fareCap", { max: moneyWhole(FARE_MAX) })}</li>}
    </ul>
  );
}

export function LineMoney({ num, vehicle }: { num: number; vehicle: Vehicle }) {
  const m = moneyState.value;
  const v = m?.lineDays?.[String(num)];
  if (!m || !v) return null;
  // the day now is not whole yet: it is left out, and the month ends the day before it
  const today = liveMoney.value.today?.day ?? m.ledger[m.ledger.length - 1]?.day ?? 0;
  const first = today - DAYS;
  const byDay = new Map<number, [number, number]>();
  for (let i = 0; i + 2 < v.length; i += 3) if (v[i] >= first && v[i] < today) byDay.set(v[i], [v[i + 1], v[i + 2]]);
  if (!byDay.size) return null;
  const usd = (x: number) => money(x * 1e6);
  // every day from the line's first kept one, a day it did not run an empty column
  const from = Math.min(...byDay.keys());
  const cols: BarCol[] = [];
  for (let d = from; d < today; d++) {
    const [fares, running] = byDay.get(d) ?? [0, 0];
    cols.push({ fares, costs: running, tip: [dateName(d), t("lineMoney.fares", { amount: usd(fares) }), runningRow(vehicle, usd(-running)), t("lineMoney.net", { amount: usd(fares - running) })].join("\n") });
  }
  return (
    <>
      <div class="h">
        <span>{tRich("lineMoney.title", { revenue: <span class="hist-fares">{t("moneyHistory.revenue")}</span>, expenses: <span class="hist-costs">{t("moneyHistory.expenses")}</span> })}</span>
        <span class="right muted small">{usd(barTop(cols))}</span>
      </div>
      <FareCostBars cols={cols} slots={DAYS} h={H} now={false} cls="line-money" />
      <div class="hist-x small muted"><span>{shortDate(first)}</span><span>{shortDate(today - 1)}</span></div>
    </>
  );
}
