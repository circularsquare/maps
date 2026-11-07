// T-647: the small spinner shown while demand loads, and whether the figures it stands in for are
// still to come.
// T-2693: the pill ring for the loading card and the map's statuses (its keyframes are the `pill-ring`
// rules in style.css).

import { computed } from "@preact/signals";
import { demandView, viewFinal, viewUpdating } from "../game/demand";
import { demandIdle, loadFailure } from "../game/loading";
import { t } from "../i18n";

/** The zone in view has no final figures yet or a solve is still changing them; never over a view
 * that has nothing to load, or after a failure. */
export const demandPending = computed(() => !loadFailure.value && !demandIdle.value && (!demandView.value || !viewFinal.value || viewUpdating.value));

export function Spinner({ tip, class: cls }: { tip?: string; class?: string }) {
  return <i class={"spin" + (cls ? " " + cls : "")} data-tip={tip} role="progressbar" aria-label={t("spinner.loading")} />;
}

const RING_R = 9, RING_LEN = 2 * Math.PI * RING_R;
const PILL_DASH = `${(RING_LEN * 0.42).toFixed(2)} ${(RING_LEN * 0.58).toFixed(2)}`;

/** A pill running round a ring, drawn as the map draws a train: a rim in a darker shade of the line
 * colour round a fill of it, bent along its track. `big`: the loading card's size. */
export function PillRing({ big }: { big?: boolean }) {
  return <svg class={"pill-ring" + (big ? " big" : "")} viewBox="0 0 24 24" aria-hidden="true">
    <circle class="pr-track" cx="12" cy="12" r={RING_R} fill="none" />
    <g class="pr-run">
      <circle class="pr-rim" cx="12" cy="12" r={RING_R} fill="none" stroke-linecap="round" stroke-dasharray={PILL_DASH} />
      <circle class="pr-fill" cx="12" cy="12" r={RING_R} fill="none" stroke-linecap="round" stroke-dasharray={PILL_DASH} />
    </g>
  </svg>;
}
