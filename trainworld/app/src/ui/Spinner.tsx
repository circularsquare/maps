// T-647: the small spinner shown while demand loads (the one animation in the game, Anita's
// exception to "no animations"), and whether the figures it stands in for are still to come.
// T-677, T-1057: the bunny in her train, for the loading card and the map's demand status (its
// keyframes are the `bt-` rules in style.css).

import { computed } from "@preact/signals";
import { demandView, viewFinal, viewUpdating } from "../game/demand";
import { demandIdle, loadFailure } from "../game/loading";

/** The zone in view has no final figures yet or a solve is still changing them; never over a view
 * that has nothing to load, or after a failure. */
export const demandPending = computed(() => !loadFailure.value && !demandIdle.value && (!demandView.value || !viewFinal.value || viewUpdating.value));

export function Spinner({ tip, class: cls }: { tip?: string; class?: string }) {
  return <i class={"spin" + (cls ? " " + cls : "")} data-tip={tip} role="progressbar" aria-label="Loading" />;
}

const INK = "#5b4a45", WHITE = "#fffdfb";

/** A wheel at (x, y): it turns. */
const Wheel = ({ x, y }: { x: number; y: number }) => <g transform={`translate(${x} ${y})`}>
  <g class="bt-wheel">
    <circle r="4.6" fill={WHITE} stroke={INK} stroke-width="1.8" />
    <path d="M-2.6 0H2.6M0 -2.6V2.6" stroke={INK} stroke-width="1.4" stroke-linecap="round" />
  </g>
</g>;

/** A bogie centred on x: its frame and two wheels. */
const Bogie = ({ x }: { x: number }) => <g>
  <rect x={x - 12} y="46.5" width="24" height="4.5" rx="2.2" fill={INK} />
  <Wheel x={x - 6.5} y={51} />
  <Wheel x={x + 6.5} y={51} />
</g>;

/** A door at x: a tall panel with a window in its upper half. */
const Door = ({ x }: { x: number }) => <g>
  <rect x={x} y="27.5" width="8.5" height="19" rx="1.2" class="bt-body" stroke={INK} stroke-width="1.5" />
  <rect x={x + 2} y="29.5" width="4.5" height="7.5" rx="1" fill={WHITE} stroke={INK} stroke-width="1.2" />
</g>;

const WINDOWS = [25, 36, 47, 58, 69];

/** A passenger car running right, the bunny inside facing the way it goes, only its ear tips above
 * the roof (seen edge on, the far one a step behind); the track runs by under it. `small`: the
 * status tag's size. */
export function BunnyTrain({ small, tip }: { small?: boolean; tip?: string }) {
  return <svg class={"bunny-train" + (small ? " small" : "")} viewBox="0 -5 100 69" data-tip={tip}
    role={small ? "progressbar" : undefined} aria-label={small ? "Loading" : undefined} aria-hidden={small ? undefined : "true"}>
    <line x1="0" y1="56" x2="100" y2="56" class="bt-rail" stroke-width="2" />
    <line x1="-12" y1="60" x2="112" y2="60" class="bt-ties" stroke-width="3" stroke-dasharray="4 8" />
    <g class="bt-train">
      <g class="bt-ears" stroke={INK} stroke-width="1.6">
        <ellipse cx="55" cy="21" rx="2.3" ry="8" transform="rotate(-12 55 25)" fill="#e9e1dd" />
        <ellipse cx="59.5" cy="20" rx="2.5" ry="8.5" transform="rotate(-12 59.5 25)" fill={WHITE} />
      </g>
      <rect x="9" y="18.5" width="82" height="6" rx="3" class="bt-roof" stroke={INK} stroke-width="1.8" />
      <rect x="5" y="22.5" width="90" height="26" rx="4.5" class="bt-body" stroke={INK} stroke-width="2" />
      <rect x="6" y="40" width="88" height="2.6" fill={WHITE} opacity=".55" />
      {WINDOWS.map((x) => <rect key={x} x={x} y="28.5" width="8" height="8" rx="1.6" fill={WHITE} stroke={INK} stroke-width="1.4" />)}
      <Door x={12} />
      <Door x={79.5} />
      <circle cx="93" cy="43" r="1.9" fill="#ffe39a" stroke={INK} stroke-width="1.2" />
    </g>
    <Bogie x={24} />
    <Bogie x={76} />
  </svg>;
}
