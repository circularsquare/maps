// T-647: the small spinner shown while demand loads, and whether the figures it stands in for are
// still to come.
// T-2693: the pill ring for the loading card and the map's statuses (its keyframes are the `pill-ring`
// rules in style.css). The bunny's train (T-677, T-1057; `bt-` rules) is the loading card's alone
// (T-2693 follow-up).

import type { ComponentChildren } from "preact";
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
const BUNNY_WINDOW = 58;
const GLASS = "#efe7e2";

/** A window at x: dim glass, a curtain drawn back to each side and tied. */
const Window = ({ x, children }: { x: number; children?: ComponentChildren }) => {
  const a = x + 0.7, b = x + 7.3;
  return <g>
    <rect x={x} y="28.5" width="8" height="8" rx="1.6" fill={GLASS} stroke={INK} stroke-width="1.4" />
    {children}
    <path class="bt-curtain" stroke={INK} stroke-width=".55" stroke-linejoin="round"
      d={`M${a} 29.2H${a + 2}C${a + 2} 31 ${a + 1.2} 32 ${a + 1.2} 33.2C${a + 1.2} 34.2 ${a + 2} 35 ${a + 2.3} 35.8H${a}Z`} />
    <path class="bt-curtain" stroke={INK} stroke-width=".55" stroke-linejoin="round"
      d={`M${b} 29.2H${b - 2}C${b - 2} 31 ${b - 1.2} 32 ${b - 1.2} 33.2C${b - 1.2} 34.2 ${b - 2} 35 ${b - 2.3} 35.8H${b}Z`} />
    <path d={`M${a} 33.2h1.2M${b} 33.2h-1.2`} stroke={INK} stroke-width=".8" stroke-linecap="round" />
  </g>;
};

/** A passenger car running right. The bunny looks out of the fourth window from behind its sill,
 * small and between the curtains, and only its ear tips show above the roof (seen edge on, the far
 * one a step behind); the track runs by under it. */
export function BunnyTrain() {
  return <svg class="bunny-train" viewBox="0 -5 100 69" aria-hidden="true">
    <line x1="0" y1="56" x2="100" y2="56" class="bt-rail" stroke-width="2" />
    <line x1="-12" y1="60" x2="112" y2="60" class="bt-ties" stroke-width="3" stroke-dasharray="4 8" />
    <g class="bt-train">
      <g class="bt-ears" stroke={INK} stroke-width="1.6">
        <ellipse cx="61.5" cy="23.2" rx="2.3" ry="8" transform="rotate(-12 61.5 25)" fill="#e9e1dd" />
        <ellipse cx="64.7" cy="22.8" rx="2.5" ry="8.5" transform="rotate(-12 64.7 25)" fill={WHITE} />
      </g>
      <rect x="9" y="18.5" width="82" height="6" rx="3" class="bt-roof" stroke={INK} stroke-width="1.8" />
      <rect x="5" y="22.5" width="90" height="26" rx="4.5" class="bt-body" stroke={INK} stroke-width="2" />
      <rect x="6" y="40" width="88" height="2.6" fill={WHITE} opacity=".55" />
      {WINDOWS.map((x) => <Window key={x} x={x}>{x === BUNNY_WINDOW && <>
        <circle cx="62" cy="34.4" r="2.4" fill={WHITE} stroke={INK} stroke-width=".9" />
        <circle cx="62.9" cy="33.5" r=".55" fill={INK} />
        <circle cx="64.2" cy="34.3" r=".5" fill="#e8909f" />
        <rect x="58.7" y="35.4" width="6.6" height="1.3" fill={INK} />
      </>}</Window>)}
      <Door x={12} />
      <Door x={79.5} />
      <circle cx="93" cy="43" r="1.9" fill="#ffe39a" stroke={INK} stroke-width="1.2" />
    </g>
    <Bogie x={24} />
    <Bogie x={76} />
  </svg>;
}
