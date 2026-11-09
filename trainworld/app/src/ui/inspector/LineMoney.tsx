// T-2787: a line's past week of money in its panel: fares minus running cost for each of the last 7
// whole days, as bars up (blue) or down (red) from zero, in the Money tab history's style. The clock
// worker keeps each line's days (`MoneyState.lineDays`) and saves them; upkeep is not in them, it is
// not kept per line per day. Days the line did not run have no bar.

import { dateName, shortDate } from "../../game/days";
import { money } from "../../game/format";
import { liveMoney, moneyState } from "../../game/money";
import { t } from "../../i18n";

const W = 300, H = 64, PAD = 3, DAYS = 7, GAP = 1.5;

export function LineMoney({ num }: { num: number }) {
  const m = moneyState.value;
  const v = m?.lineDays?.[String(num)];
  if (!m || !v) return null;
  // the day now is not whole yet: it is left out, and the week ends the day before it
  const today = liveMoney.value.today?.day ?? m.ledger[m.ledger.length - 1]?.day ?? 0;
  const first = today - DAYS;
  const days: { day: number; fares: number; running: number; net: number }[] = [];
  for (let i = 0; i + 2 < v.length; i += 3) {
    if (v[i] >= first && v[i] < today) days.push({ day: v[i], fares: v[i + 1], running: v[i + 2], net: v[i + 1] - v[i + 2] });
  }
  if (!days.length) return null;
  const hi = Math.max(0, ...days.map((d) => d.net)), lo = Math.min(0, ...days.map((d) => d.net));
  const span = hi - lo || 1;
  const y = (x: number) => PAD + ((hi - x) / span) * (H - 2 * PAD);
  const col = W / DAYS, bw = col - GAP;
  const x0 = (d: number) => (d - first) * col;
  const usd = (x: number) => money(x * 1e6);
  return (
    <>
      <div class="h">{t("lineMoney.heading")}</div>
      <svg class="hist line-money" viewBox={`0 0 ${W} ${H}`}>
        <line class="hist-axis" x1={0} x2={W} y1={y(0)} y2={y(0)} />
        {days.map((d) => {
          const top = Math.min(y(0), y(d.net)), h = Math.max(0.5, Math.abs(y(d.net) - y(0)));
          return <rect class={d.net < 0 ? "hist-costs" : "hist-fares"} x={x0(d.day) + GAP / 2} y={top} width={bw} height={h} />;
        })}
        {days.map((d) => (
          <rect class="hist-col" x={x0(d.day)} y={0} width={col} height={H}
            data-tip={[dateName(d.day), t("lineMoney.fares", { amount: usd(d.fares) }), t("lineMoney.running", { amount: usd(-d.running) }), t("lineMoney.net", { amount: usd(d.net) })].join("\n")} />
        ))}
      </svg>
      <div class="hist-x small muted"><span>{shortDate(first)}</span><span>{shortDate(today - 1)}</span></div>
    </>
  );
}
