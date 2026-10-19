// A money figure in the panels, in the alert red while it is negative (T-722). `dollars` is in
// dollars, as `money` takes it; `cash` writes it to four figures (`moneyCash`) instead of three.

import { money, moneyCash, moneyNegative } from "../game/format";

export function Money({ dollars, cash }: { dollars: number; cash?: boolean }) {
  const text = cash ? moneyCash(dollars) : money(dollars);
  return moneyNegative(dollars) ? <span class="money-neg">{text}</span> : <>{text}</>;
}
