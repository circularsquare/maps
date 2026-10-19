// T-647: the small spinner shown while demand loads (the one animation in the game, Anita's
// exception to "no animations"), and whether the figures it stands in for are still to come.
// T-677: the bunny in her train, for the loading card and the map's demand status; a draft drawing,
// swapped by replacing `BunnyTrain` (its keyframes are the `bt-` rules in style.css).

import { computed } from "@preact/signals";
import { demandView, viewFinal, viewUpdating } from "../game/demand";
import { demandIdle, loadFailure } from "../game/loading";

/** The zone in view has no final figures yet or a solve is still changing them; never over a view
 * that has nothing to load, or after a failure. */
export const demandPending = computed(() => !loadFailure.value && !demandIdle.value && (!demandView.value || !viewFinal.value || viewUpdating.value));

export function Spinner({ tip, class: cls }: { tip?: string; class?: string }) {
  return <i class={"spin" + (cls ? " " + cls : "")} data-tip={tip} role="progressbar" aria-label="Loading" />;
}

const INK = "#5b4a45", WHITE = "#fffdfb", PINK = "#f5a9bb";

/** A wheel at (x, y): it turns. */
const Wheel = ({ x, y }: { x: number; y: number }) => <g transform={`translate(${x} ${y})`}>
  <g class="bt-wheel">
    <circle r="6" fill={WHITE} stroke={INK} stroke-width="2" />
    <path d="M-3.5 0H3.5M0 -3.5V3.5" stroke={INK} stroke-width="1.6" stroke-linecap="round" />
  </g>
</g>;

/** A bunny riding a toy train, the track running by under it. `small`: the status tag's size. */
export function BunnyTrain({ small, tip }: { small?: boolean; tip?: string }) {
  return <svg class={"bunny-train" + (small ? " small" : "")} viewBox="0 -5 100 69" data-tip={tip}
    role={small ? "progressbar" : undefined} aria-label={small ? "Loading" : undefined} aria-hidden={small ? undefined : "true"}>
    <line x1="0" y1="56" x2="100" y2="56" class="bt-rail" stroke-width="2" />
    <line x1="-12" y1="60" x2="112" y2="60" class="bt-ties" stroke-width="3" stroke-dasharray="4 8" />
    <g class="bt-puffs" fill={WHITE} stroke={INK} stroke-width="1.4" opacity=".75">
      <circle class="bt-puff" cx="76" cy="14" r="3.2" />
      <circle class="bt-puff late" cx="76" cy="14" r="3.2" />
    </g>
    <g class="bt-train">
      <g class="bt-ears">
        <ellipse cx="34" cy="11" rx="4.6" ry="11" transform="rotate(-14 34 11)" fill={WHITE} stroke={INK} stroke-width="2" />
        <ellipse cx="34" cy="12" rx="2" ry="7" transform="rotate(-14 34 12)" fill={PINK} />
        <ellipse cx="46" cy="11" rx="4.6" ry="11" transform="rotate(14 46 11)" fill={WHITE} stroke={INK} stroke-width="2" />
        <ellipse cx="46" cy="12" rx="2" ry="7" transform="rotate(14 46 12)" fill={PINK} />
      </g>
      <ellipse cx="40" cy="27" rx="13" ry="11" fill={WHITE} stroke={INK} stroke-width="2" />
      <circle cx="35" cy="26" r="1.7" fill={INK} />
      <circle cx="45" cy="26" r="1.7" fill={INK} />
      <ellipse cx="31.5" cy="30" rx="2.6" ry="1.5" fill={PINK} />
      <ellipse cx="48.5" cy="30" rx="2.6" ry="1.5" fill={PINK} />
      <path d="M38 29.4q1 1.4 2 0q1 1.4 2 0" fill="none" stroke={INK} stroke-width="1.2" stroke-linecap="round" />
      <rect x="69" y="22" width="9" height="12" rx="2" class="bt-body" stroke={INK} stroke-width="2" />
      <rect x="67" y="19" width="13" height="5" rx="2.5" class="bt-body" stroke={INK} stroke-width="2" />
      <rect x="16" y="32" width="70" height="17" rx="6" class="bt-body" stroke={INK} stroke-width="2" />
      <rect x="18" y="38" width="66" height="3" fill={WHITE} opacity=".55" />
      <ellipse cx="34" cy="32.5" rx="3.2" ry="2.3" fill={WHITE} stroke={INK} stroke-width="1.5" />
      <ellipse cx="46" cy="32.5" rx="3.2" ry="2.3" fill={WHITE} stroke={INK} stroke-width="1.5" />
      <circle cx="86" cy="41" r="2.6" fill="#ffe39a" stroke={INK} stroke-width="1.5" />
    </g>
    <Wheel x={32} y={50} />
    <Wheel x={70} y={50} />
  </svg>;
}
