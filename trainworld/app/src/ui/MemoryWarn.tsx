// T-3329: the question before opening a saved game bigger than the device can give (game/memoryEstimate.ts).
import { useState } from "preact/hooks";
import { bigForTab, deviceText, estimateGB, gbText, overBudget } from "../game/memoryEstimate";
import { t } from "../i18n";

/** The warning for a game of `stations`, with Open anyway and Cancel. T-3940: `died`, the game's last load in this
 * browser never finished; `onLight`, a third button that opens it without calculating trips. */
export function MemoryWarn({ stations, died, onGo, onLight, onCancel }: { stations: number; died?: boolean; onGo: () => void; onLight?: () => void; onCancel: () => void }) {
  const text = died ? t("savesPanel.memoryWarnDied")
    : bigForTab(stations) ? t("savesPanel.memoryWarnPhone")
    : t("savesPanel.memoryWarn", { need: gbText(estimateGB(stations)), have: deviceText() });
  return <div class="callout bad memory-warn" role="alert">
    <p>{text}</p>
    <div class="pane-foot">
      <button class="btn text danger" onClick={onGo}>{t("savesPanel.openAnyway")}</button>
      {onLight && <button class="btn text memory-warn-light" onClick={onLight}>{t("common.openWithoutTrips")}</button>}
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
