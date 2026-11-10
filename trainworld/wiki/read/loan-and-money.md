# The loan and the money scale

**Game days earn a hundred real days.** Fares and running costs are scaled up by 100 so that a game day earns and spends what that many real days would. Building prices are not scaled. The fare you set and the per-km running costs you see in the Money tab are the real ones, before the scale. [upkeep](upkeep.md) is shown as the game day's figure.

**One loan at a time**, of US$1B, US$5B or US$10B, at the bottom of the Money tab. It pays itself off in 60 equal daily payments, whatever the size, with interest of 1% a game day on what is left. Payments come out through the day like upkeep and show as "Loan" in the ledger. They go out even when the cash is short, so the cash can go below zero.

**Repay** pays off what is left once the cash covers it. After the last payment, or a repay, you can take another.

**A loan can be refused.** It is turned down while its daily payment is more than the profit a day (fares minus running costs and upkeep) averaged over the last 7 whole game days. Before a whole day with fares has passed, only the smallest loan can be taken. A sandbox game has no loan.

Exact rule: `docs/track-and-service.md` "Starting money, loan and the economy's scale".
In the code: **main** `app/src/game/loan.ts` `loanPayment`, `loanAllowed`, `loanProfit` `sim/src/track/params.rs` `ECONOMY`
