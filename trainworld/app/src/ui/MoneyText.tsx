// A money figure in the panels, in the alert red while it is negative (T-722). `dollars` is in
// dollars, as `money` takes it; `cash` writes it to four figures (`moneyCash`) instead of three.

import { money, moneyCash, moneyNegative, withFullSymbol } from "../game/format";
import { t, tNumber } from "../i18n";

/** T-1851: the hover of a line's or bus line's money, in the Lines tab and in the line and bus panels
 * alike: the net, then fares, running cost and upkeep (negative amounts are red in the hover, ui/tip.ts). */
/** T-4331: the hover's last money row: fares divided by running and upkeep, as a percent; none while there are no costs to divide by. */
export const profitPercentRow = (fares: number, costs: number): string[] =>
  costs > 0 ? [t("moneyText.profitPercent", { percent: tNumber(fares / costs, { style: "percent", maximumFractionDigits: 0 }) })] : [];

export const lineMoneyTip = (fares: number, run: number, upkeep: number, bus: boolean | "boats" | "trams", dayNote?: string | null): string =>
  [t("moneyText.profit", { amount: money(fares - run - upkeep) }), t("moneyText.fares", { amount: money(fares) }),
    t("moneyText.running", { kind: bus === "boats" || bus === "trams" ? bus : bus ? "buses" : "trains", amount: money(-run) }), t("moneyText.upkeep", { amount: money(-upkeep) }), ...profitPercentRow(fares, run + upkeep), ...(dayNote ? [dayNote] : [])].join("\n");

export function Money({ dollars, cash, full }: { dollars: number; cash?: boolean; full?: boolean }) {
  const shown = cash ? moneyCash(dollars) : money(dollars);
  const text = full ? withFullSymbol(shown) : shown;
  return moneyNegative(dollars) ? <span class="money-neg">{text}</span> : <>{text}</>;
}
