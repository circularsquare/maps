// T-757, T-2657: a demand worker failed after startup (out of memory on a very big network, or an error):
// the game is paused once (game/loading.ts) and this notice sits at the top centre of the map with a Reload
// button; the technical error is the small line under it. The figures shown stay.

import { t } from "../i18n";
import { demandStopped, loadFailure } from "../game/loading";

/** `save` writes the autosave and resolves when it is safe to leave the page. */
export function DemandStopped({ save }: { save: () => Promise<void> }) {
  const why = demandStopped.value;
  if (!why) return null;
  const error = loadFailure.value;
  const reload = () => { void save().finally(() => location.reload()); };
  return <div class="demand-stopped-note" role="alert">
    <div class="dsn-row">
      <span>{t("demandStopped.note", { memory: why === "memory" })}</span>
      <button class="btn" onClick={reload}>{t("demandStopped.reload")}</button>
    </div>
    {why === "error" && error && <div class="dsn-detail muted" data-tip={error}>{error}</div>}
  </div>;
}
