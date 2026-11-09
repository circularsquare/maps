// T-757, T-2657: a demand worker failed after startup (out of memory on a very big network, or an error):
// the game is paused once (game/loading.ts) and this notice sits at the top centre of the map with a Reload
// button; the technical error is the small line under it. The figures shown stay.

import { t } from "../i18n";
import { demandStopped, loadFailure } from "../game/loading";
import { CHOICE_PARAMS } from "../workers/clockClient";
import { dropMarket, marketOff } from "../game/ui";

/** T-2867: intercity trips off for this tab and the page back in the same game (a page with no game switch
 * in its address would come back to the start menu), after `save` where there is a game to save. */
export function playWithoutIntercity(save: () => Promise<void> = () => Promise.resolve()) {
  dropMarket();
  void save().finally(() => {
    const u = new URL(location.href);
    if (!CHOICE_PARAMS.some((k) => u.searchParams.has(k))) u.searchParams.set("continue", "1");
    if (u.href === location.href) location.reload();
    else location.replace(u.href);
  });
}

/** `save` writes the autosave and resolves when it is safe to leave the page. */
export function DemandStopped({ save }: { save: () => Promise<void> }) {
  const why = demandStopped.value;
  if (!why) return null;
  const error = loadFailure.value;
  const reload = () => { void save().finally(() => location.reload()); };
  return <div class="demand-stopped-note" role="alert">
    <div class="dsn-row">
      <span>{why === "graphics" ? t("demandStopped.graphics") : t("demandStopped.note", { memory: why === "memory" })}</span>
      <button class="btn" onClick={reload}>{t("demandStopped.reload")}</button>
      {why === "memory" && !marketOff() && <button class="btn" onClick={() => playWithoutIntercity(save)}>{t("loadingScreen.playWithoutIntercity")}</button>}
    </div>
    {why === "error" && error && <div class="dsn-detail muted" data-tip={error}>{error}</div>}
  </div>;
}
