// T-2962: the notice at the top centre of the map when a file of this page could not be loaded because the
// game was published again since the tab opened (game/staleBuild.ts). Reload writes the autosave first.

import { t } from "../i18n";
import { staleBuild } from "../game/staleBuild";

/** `save` writes the autosave and resolves when it is safe to leave the page. */
export function NewVersion({ save }: { save: () => Promise<void> }) {
  if (!staleBuild.value) return null;
  return <div class="demand-stopped-note new-version-note" role="alert">
    <div class="dsn-row">
      <span>{t("newVersion.note")}</span>
      <button class="btn" onClick={() => { void save().finally(() => location.reload()); }}>{t("newVersion.reload")}</button>
    </div>
  </div>;
}
