import { money } from "../../game/format";
import { useLineQuote } from "../../game/quotes";
import { t } from "../../i18n";

/** A planned line's Construct button with its price, asked of the clock worker while the panel is
 * open (T-1171); the price is left off until the first answer. `kind`: what the line runs. */
export function ConstructButton({ num, build, kind }: { num: number; build: () => void; kind: "line" | "bus" | "boat" }) {
  const q = useLineQuote(num);
  const tip = !q ? t("constructButton.tipNone", { kind })
    : q.trains > 0 ? t("constructButton.tipBoth", { kind, build: money(q.build), trains: money(q.trains) })
    : t("constructButton.tipBuild", { kind, build: money(q.build) });
  return (
    <button class="btn text" data-tip={tip} onClick={build}>
      {q ? t("constructButton.priced", { price: money(q.build + q.trains) }) : t("constructButton.plain")}
    </button>
  );
}
