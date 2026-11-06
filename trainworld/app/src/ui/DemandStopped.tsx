// T-757: a demand worker failed after startup (out of memory on a very big network, or an error):
// the alert under the demand buttons, in red. The figures shown stay.

import { t } from "../i18n";
import { demandStopped, loadFailure } from "../game/loading";

export function DemandStopped() {
  const why = demandStopped.value;
  if (!why) return null;
  const tip = t("demandStopped.tip", { memory: why === "memory", error: loadFailure.value ?? t("demandStopped.unknown") });
  return <div class="btn demand-stopped" role="alert" data-tip={tip}>
    {why === "memory" ? t("demandStopped.memory") : t("demandStopped.stopped")}
  </div>;
}
