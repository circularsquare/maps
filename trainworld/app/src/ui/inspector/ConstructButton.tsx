import { money } from "../../game/format";
import { useLineQuote } from "../../game/quotes";

/** A planned line's Construct button with its price, asked of the clock worker while the panel is
 * open (T-1171); the price is left off until the first answer. */
export function ConstructButton({ num, build, what, vehicles, runs }: { num: number; build: () => void; what: string; vehicles: string; runs: string }) {
  const q = useLineQuote(num);
  return (
    <button class="btn text" data-tip={q ? `${what} ${money(q.build)}${q.trains > 0 ? `, ${vehicles} ${money(q.trains)}` : ""}. ${runs} once it is constructed.` : `${runs} once it is constructed.`} onClick={build}>
      {q ? `Construct for ${money(q.build + q.trains)}` : "Construct"}
    </button>
  );
}
