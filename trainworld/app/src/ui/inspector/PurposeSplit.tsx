// Riders by trip purpose under a line's riders and a station's boardings (T-165, SPEC 8): a thin
// stacked bar in the purpose colours (game/purposes.ts, the purpose pie's) and a small key with
// shares; counts on hover. Every purpose is always shown. While the map's purpose filter is on,
// the purpose it draws is ringed in the bar and bold in the key (DECISIONS 2026-10-12).
//
// The figures are `DemandView.purposes` (T-158), summed over exactly the stops the inspector's own
// total sums, so the parts add up to the number shown above them.

import { demandView, type DemandView } from "../../game/demand";
import { count } from "../../game/format";
import { PURPOSES, purposeColour, type PurposeKey } from "../../game/purposes";
import { purposeFilter } from "../../game/ui";

/** What to count: a whole line (`at` left out, its riders a day), or some of its stops' boardings
 * (indices into the line's stops). */
export type SplitScope = { line: string; at?: number[] }[];

function sum(dv: DemandView, scope: SplitScope, key?: PurposeKey): number {
  let s = 0;
  for (const { line, at } of scope) {
    const d = key ? dv.purposes?.get(key)?.lines.get(line) : dv.lines.get(line);
    if (!d) continue;
    if (!at) s += d.ridersPerDay;
    else for (const k of at) s += d.boardings[k] ?? 0;
  }
  return s;
}

/** Each purpose's share of `scope`'s riders (or boardings), and the total; null before demand. */
export function purposeSplit(dv: DemandView | null, scope: SplitScope): { parts: { key: PurposeKey; n: number }[]; total: number } | null {
  if (!dv?.purposes || !scope.some(({ line }) => dv.lines.has(line))) return null;
  return { parts: PURPOSES.map(({ key }) => ({ key, n: sum(dv, scope, key) })), total: sum(dv, scope) };
}

const pct = (n: number, of: number) => {
  const p = of > 0 ? (100 * n) / of : 0;
  return n >= 0.5 && p < 0.5 ? "<1%" : Math.round(p) + "%";
};

/** A list item for a `.rows` list. `unit`: "riders" for a line, "boardings" for a station. */
export function PurposeSplit({ scope, unit }: { scope: SplitScope; unit: string }) {
  const split = purposeSplit(demandView.value, scope);
  if (!split) return null;
  const all = split.parts.reduce((a, p) => a + p.n, 0);
  if (all <= 0) return null;
  const shown = purposeFilter.value;
  const words = (p: { key: PurposeKey; n: number }) =>
    `${PURPOSES.find((q) => q.key === p.key)!.name}: ${count(p.n)} ${unit} a day, ${pct(p.n, all)}`;
  return (
    <li class="purpose-split">
      <div class="bar split-bar" role="img" aria-label={`By purpose: ${split.parts.map(words).join("; ")}`}>
        {split.parts.map((p) =>
          p.n > 0 ? (
            <i class={"seg" + (p.key === shown ? " shown" : "")} style={{ width: (100 * p.n) / all + "%", background: purposeColour(p.key) }} data-tip={words(p)} />
          ) : null,
        )}
      </div>
      <ul class="split-key">
        {split.parts.map((p) => (
          <li class={p.key === shown ? "b" : p.n > 0 ? "" : "muted"} data-tip={words(p)}>
            <i class="sw" style={{ background: purposeColour(p.key) }} />
            {PURPOSES.find((q) => q.key === p.key)!.name} <span class="num">{pct(p.n, all)}</span>
          </li>
        ))}
      </ul>
    </li>
  );
}
