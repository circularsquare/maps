// The commuter bubbles' legend (T-084, T-097, SPEC 8), in the map's top left corner while they are
// on: where commuters live or work, the three mode colours with commuters and shares, and the
// mixes between them a bubble's colour runs through. Its colours are the bubbles'
// (game/palette.ts MODE_COLOURS, modeMix). The switch itself is in the settings pane.

import { demandUpdating, demandView } from "../game/demand";
import { count } from "../game/format";
import { BUBBLE_ALPHA, MODE_COLOURS, modeMix } from "../game/palette";
import { display } from "../game/ui";
import { commuterLegend } from "../map/commuterView";

const ROWS: [keyof typeof MODE_COLOURS, string][] = [
  ["train", "Train"],
  ["walk", "Walking"],
  ["drive", "Driving"],
];

const css = (c: [number, number, number]) => `rgba(${c[0]}, ${c[1]}, ${c[2]}, ${BUBBLE_ALPHA})`;
/** Mixes from all driving to all train, in quarters. */
const STRIP = [0, 0.25, 0.5, 0.75, 1].map((t) => css(modeMix(t, 0, 1 - t)));

export function CommuterLegend() {
  const d = display.value;
  if (!d.commuters) return null;
  const l = commuterLegend.value;
  const shown = l && l.end === d.commuterEnd ? l : null;
  const all = shown ? shown.totals[0] + shown.totals[1] + shown.totals[2] : 0;
  return (
    <div id="commuter-legend" style={{ opacity: demandUpdating.value && shown ? 0.6 : 1 }}>
      <div class="lg-title">{d.commuterEnd === "home" ? "Where commuters live" : "Where commuters work"}</div>
      {shown ? (
        <>
          <ul class="rows">
            {ROWS.map(([k, name], i) => (
              <li>
                <i class="dot" style={{ background: css(modeMix(k === "train" ? 1 : 0, k === "walk" ? 1 : 0, k === "drive" ? 1 : 0)) }} />
                <span class="grow">{name}</span>
                <span class="val">{count(shown.totals[i])}</span>
                <span class="val w-num">{all > 0 ? Math.round((100 * shown.totals[i]) / all) + "%" : ""}</span>
              </li>
            ))}
          </ul>
          <div class="mix">
            <span class="muted small">Driving</span>
            {STRIP.map((c) => (
              <i style={{ background: c }} />
            ))}
            <span class="muted small">Train</span>
          </div>
          <p class="muted small">A bubble's area is its number of commuters.</p>
          <p class="muted small">Click a bubble, or hold and drag to pick several.</p>
        </>
      ) : (
        <p class="muted small">{demandView.value ? "Adding up the commuters" : "Shown once demand is worked out"}</p>
      )}
    </div>
  );
}
