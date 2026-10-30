// A money figure in the panels, in the alert red while it is negative (T-722). `dollars` is in
// dollars, as `money` takes it; `cash` writes it to four figures (`moneyCash`) instead of three.

import { money, moneyCash, moneyNegative } from "../game/format";

/** T-1851: the hover of a line's or bus line's money, in the Lines tab and in the line and bus panels
 * alike: the net, then fares, running cost and upkeep (negative amounts are red in the hover, ui/tip.ts). */
export const lineMoneyTip = (fares: number, run: number, upkeep: number, bus: boolean): string =>
  [`${money(fares - run - upkeep)} net in the past 24 hours`, `Fares ${money(fares)}`, `Running ${bus ? "buses" : "trains"} ${money(-run)}`, `Upkeep ${money(-upkeep)}`].join("\n");

export function Money({ dollars, cash }: { dollars: number; cash?: boolean }) {
  const text = cash ? moneyCash(dollars) : money(dollars);
  return moneyNegative(dollars) ? <span class="money-neg">{text}</span> : <>{text}</>;
}
