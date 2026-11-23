// T-4109: the notice at the top centre of the map once the track model has stopped. Reload goes straight away: the
// model cannot save, and the last autosave is what stays.

import { t } from "../i18n";
import { modelStopped } from "../game/modelStopped";

export function ModelStopped() {
  if (!modelStopped.value) return null;
  return <div class="demand-stopped-note" role="alert">
    <div class="dsn-row">
      <span>{t("clockWorker.modelDown")}</span>
      <button class="btn" onClick={() => location.reload()}>{t("demandStopped.reload")}</button>
    </div>
  </div>;
}
