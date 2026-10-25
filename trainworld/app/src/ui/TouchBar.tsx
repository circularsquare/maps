// The touch control bar (T-1260; docs/ui-building.md "Building by touch"): beside the point a build
// tool is placing on a touch screen, big buttons to confirm, take back the last point, stop, and
// change the elevation and track type. map/tools.ts says what it offers and moves it.

import type { JSX } from "preact";
import { batch } from "@preact/signals";
import { useLayoutEffect, useState } from "preact/hooks";
import { levelLabel, levelName } from "../game/format";
import { inkOn, LEVEL_COLOURS } from "../game/palette";
import { autoLevel, buildLevel, setTrackType, trackType, type TrackType } from "../game/ui";
import { touchActions, touchCtl } from "../map/touch";
import { Icon, LevelButtons } from "./widgets";

const TRACK_TYPES: [TrackType, string, () => JSX.Element][] = [
  [2, "Double track", () => <svg viewBox="0 0 16 16"><path d="M2 6h12M2 10h12" /></svg>],
  [4, "Quad track", () => <svg viewBox="0 0 16 16"><path d="M2 3.5h12M2 6.5h12M2 9.5h12M2 12.5h12" /></svg>],
  ["hsr", "High-speed track", () => <svg viewBox="0 0 16 16"><path d="M1.5 5h7M1.5 11h7M10 4.5l3.5 3.5-3.5 3.5M12.5 4.5L16 8l-3.5 3.5" /></svg>],
];

export function TouchBar() {
  const ctl = touchCtl.value;
  const [levels, setLevels] = useState(false);
  // placed again at its new size
  useLayoutEffect(() => touchActions.drawn());
  if (!ctl) return null;
  const lv = buildLevel.value, auto = autoLevel.value;
  const tt = TRACK_TYPES.findIndex(([t]) => t === trackType.value);
  const [, ttName, TrackIcon] = TRACK_TYPES[Math.max(0, tt)];
  return (
    <div class="touch-bar">
      {ctl.level && levels && (
        <div class="touch-levels">
          <LevelButtons value={auto ? null : lv} tip={levelName} onPick={(l) => { batch(() => { buildLevel.value = l; autoLevel.value = false; }); setLevels(false); }} />
        </div>
      )}
      <div class="touch-row">
        {ctl.confirm !== null && (
          <button class="btn icon tb ok" disabled={!ctl.confirm} data-tip={ctl.confirmTip} aria-label={ctl.confirmTip} onClick={() => touchActions.confirm()}><Icon.check /></button>
        )}
        {ctl.back && <button class="btn icon tb" data-tip="Take back the last point" aria-label="Take back the last point" onClick={() => touchActions.back()}><Icon.undo /></button>}
        <button class="btn icon tb" data-tip={ctl.stopTip} aria-label={ctl.stopTip} onClick={() => touchActions.stop()}><Icon.cross /></button>
        {ctl.level && (
          <button class={"btn icon tb" + (auto ? "" : " lv on")} style={auto ? undefined : { "--lc": LEVEL_COLOURS[lv], "--lk": inkOn(LEVEL_COLOURS[lv]) }}
            data-tip={auto ? "Elevation of the track you extend" : `Elevation ${levelName(lv)}`} aria-label="Elevation" aria-expanded={levels}
            onClick={() => setLevels(!levels)}>{auto ? <Icon.autoLevel /> : levelLabel(lv)}</button>
        )}
        {ctl.track && (
          <button class="btn icon tb" data-tip={ttName} aria-label={ttName}
            onClick={() => setTrackType(TRACK_TYPES[(tt + 1) % TRACK_TYPES.length][0])}><TrackIcon /></button>
        )}
      </div>
    </div>
  );
}
