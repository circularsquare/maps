// T-3329: the question before opening a saved game bigger than the device can give (game/memoryEstimate.ts).
import { useState } from "preact/hooks";
import { deviceGBOrBudget, estimateGB, gbText, overBudget } from "../game/memoryEstimate";
import { t } from "../i18n";

/** The warning for a game of `stations`, with Open anyway and Cancel. */
export function MemoryWarn({ stations, onGo, onCancel }: { stations: number; onGo: () => void; onCancel: () => void }) {
  return <div class="callout bad memory-warn" role="alert">
    <p>{t("savesPanel.memoryWarn", { need: gbText(estimateGB(stations)), have: gbText(deviceGBOrBudget()) })}</p>
    <div class="pane-foot">
      <button class="btn text danger" onClick={onGo}>{t("savesPanel.openAnyway")}</button>
      <button class="btn text" onClick={onCancel}>{t("savesPanel.openCancel")}</button>
    </div>
  </div>;
}

/** An Open button's guard: `ask(go)` runs `go` now, or after Open anyway when the game is over budget. `warning` is
 * the callout to show (null while there is none). */
export function useMemoryGuard(stations: number | undefined) {
  const [pending, setPending] = useState<(() => void) | null>(null);
  return {
    ask: (go: () => void) => { if (overBudget(stations)) setPending(() => go); else go(); },
    warning: pending ? <MemoryWarn stations={stations!} onGo={() => { const g = pending; setPending(null); g(); }} onCancel={() => setPending(null)} /> : null,
  };
}
