// T-2264: the share of trips taking the train against the car, drawn in the effective time explainer
// (HelpPanel.tsx) from the demand rule itself.

import { t } from "../i18n";
import { modeColours } from "../game/palette";

/** sim/src/demand/kernel.rs `Params::ratio_share` and its defaults (`ratio_fade_lo`, `ratio_fade_hi`): the
 * train's share of the trips between the train and the car, from q = the train's effective time over the car's. */
const FADE_LO = 3, FADE_HI = 4;
export function ratioShare(q: number): number {
  q = Math.max(q, 0.01);
  const s = 1 / (1 + q * q * q);
  const k = 1 / (FADE_HI - FADE_LO);
  const fr = Math.min(Math.max((FADE_HI - q) * k, 0), 1);
  const fc = Math.min(Math.max((FADE_HI - 1 / q) * k, 0), 1);
  return 1 - (1 - s * fr) * fc;
}

const W = 402, H = 168, X0 = 46, X1 = 396, Y0 = 8, Y1 = 128, Q_MAX = 4.5;
const sx = (q: number) => X0 + (q / Q_MAX) * (X1 - X0);
const sy = (share: number) => Y1 - share * (Y1 - Y0);

export function TripSplitGraph() {
  const c = modeColours();
  const qs: number[] = [];
  for (let q = 0; q <= Q_MAX + 1e-9; q += 0.025) qs.push(q);
  const top = qs.map((q) => `${sx(q).toFixed(1)},${sy(ratioShare(q)).toFixed(1)}`).join(" L");
  const trainArea = `M${sx(0)},${Y1} L${top} L${sx(Q_MAX)},${Y1} Z`;
  const carArea = `M${top} L${sx(Q_MAX)},${Y0} L${sx(0)},${Y0} Z`;
  return (
    <svg class="split-graph" viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`${t("helpTopic.graphTrain")}, ${t("helpTopic.graphCar")}`}>
      <path class="tg-area" style={{ fill: c.train }} d={trainArea} />
      <path class="tg-area" style={{ fill: c.drive }} d={carArea} />
      <text class="sg-in" x={sx(0.12)} y={sy(0.3)}>{t("helpTopic.graphTrain")}</text>
      <text class="sg-in" x={sx(3.55)} y={sy(0.62)}>{t("helpTopic.graphCar")}</text>
      {[0, 0.5, 1].map((v) => (
        <text class="tg-label" x={X0 - 5} y={sy(v) + 4} text-anchor="end">{Math.round(v * 100)}%</text>
      ))}
      {[1, 2, 3, 4].map((q) => (
        <text class="tg-label" x={sx(q)} y={Y1 + 14} text-anchor="middle">{q}x</text>
      ))}
      <line class="tg-axis" x1={X0} y1={Y1} x2={X1} y2={Y1} />
      <line class="sg-guide" x1={sx(1)} y1={sy(0.5)} x2={sx(1)} y2={Y1} />
      <circle class="sg-dot" cx={sx(1)} cy={sy(0.5)} r="4" />
      <text class="sg-in sg-equal" x={sx(1) + 9} y={sy(0.5) - 6}>{t("helpTopic.graphEqual")}</text>
      <text class="tg-label" x={(X0 + X1) / 2} y={H - 4} text-anchor="middle">{t("helpTopic.graphX")}</text>
      <text class="tg-label" transform={`rotate(-90 12 ${(Y0 + Y1) / 2})`} x="12" y={(Y0 + Y1) / 2 + 4} text-anchor="middle">{t("helpTopic.graphY")}</text>
    </svg>
  );
}
