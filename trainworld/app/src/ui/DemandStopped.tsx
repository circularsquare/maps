// T-757: a demand worker failed after startup (out of memory on a very big network, or an error):
// the alert under the demand buttons, in red. The figures shown stay.

import { demandStopped, loadFailure } from "../game/loading";

export function DemandStopped() {
  const why = demandStopped.value;
  if (!why) return null;
  const tip = `${why === "memory" ? "The demand model needs more memory than the browser gives it for this network." : `The demand model stopped with an error: ${loadFailure.value ?? "unknown"}.`} The figures shown are from before it stopped. Reload the page to start it again.`;
  return <div class="btn demand-stopped" role="alert" data-tip={tip}>
    {why === "memory" ? "Demand ran out of memory" : "Demand stopped"}
  </div>;
}
