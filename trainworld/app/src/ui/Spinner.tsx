// T-647: the small spinner shown while demand loads (the one animation in the game, Anita's
// exception to "no animations"), and whether the figures it stands in for are still to come.

import { computed } from "@preact/signals";
import { demandView, viewFinal, viewUpdating } from "../game/demand";
import { demandIdle, loadFailure } from "../game/loading";

/** The zone in view has no final figures yet or a solve is still changing them; never over a view
 * that has nothing to load, or after a failure. */
export const demandPending = computed(() => !loadFailure.value && !demandIdle.value && (!demandView.value || !viewFinal.value || viewUpdating.value));

export function Spinner({ tip, class: cls }: { tip?: string; class?: string }) {
  return <i class={"spin" + (cls ? " " + cls : "")} data-tip={tip} role="progressbar" aria-label="Loading" />;
}
