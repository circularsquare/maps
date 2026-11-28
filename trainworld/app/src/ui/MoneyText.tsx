// A money figure in the panels, in the alert red while it is negative (T-722). `dollars` is in
// dollars, as `money` takes it; `cash` writes it to four figures (`moneyCash`) instead of three.

import { money, moneyCash, moneyNegative, withFullSymbol } from "../game/format";
import { t, tNumber } from "../i18n";
import type { Vehicle } from "../game/vehicle";

/** T-1851: the hover of a line's or bus line's money, in the Lines tab and in the line and bus panels
 * alike: the net, then fares, running cost and upkeep (negative amounts are red in the hover, ui/tip.ts). */
/** T-4331: the hover's last money row: fares divided by running and upkeep, as a percent; none while there are no costs to divide by. */
export const profitPercentRow = (fares: number, costs: number): string[] =>
  costs > 0 ? [t("moneyText.profitPercent", { percent: tNumber(fares / costs, { style: "percent", maximumFractionDigits: 0 }) })] : [];

/** A money hover's running cost row, named for what the line runs: "Running buses -$900". */
const RUNNING_KIND: Record<Vehicle, string> = { boat: "boats", bus: "buses", tram: "trams", train: "trains" };
export const runningRow = (v: Vehicle, amount: string): string => t("moneyText.running", { kind: RUNNING_KIND[v], amount });

export const lineMoneyTip = (fares: number, run: number, upkeep: number, v: Vehicle, dayNote?: string | null): string =>
  [t("moneyText.profit", { amount: money(fares - run - upkeep) }), t("moneyText.fares", { amount: money(fares) }),
    runningRow(v, money(-run)), t("moneyText.upkeep", { amount: money(-upkeep) }), ...profitPercentRow(fares, run + upkeep), ...(dayNote ? [dayNote] : [])].join("\n");

export function Money({ dollars, cash, full }: { dollars: number; cash?: boolean; full?: boolean }) {
  const shown = cash ? moneyCash(dollars) : money(dollars);
  const text = full ? withFullSymbol(shown) : shown;
  return moneyNegative(dollars) ? <span class="money-neg">{text}</span> : <>{text}</>;
}
