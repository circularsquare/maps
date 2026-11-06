// The touch control bar (T-1260; docs/ui-building.md "Building by touch"): beside the point a build
// tool is placing on a touch screen, big buttons to confirm, take back and stop. The elevation and
// track type stay in the Build tab (T-1706). map/tools.ts says what it offers and moves it.

import { useLayoutEffect } from "preact/hooks";
import { touchActions, touchCtl } from "../map/touch";
import { Icon } from "./widgets";
import { t } from "../i18n";

export function TouchBar() {
  const ctl = touchCtl.value;
  // placed again at its new size
  useLayoutEffect(() => touchActions.drawn());
  if (!ctl) return null;
  return (
    <div class="touch-bar">
      <div class="touch-row">
        {ctl.confirm !== null && (
          <button class="btn icon tb ok" disabled={!ctl.confirm} data-tip={ctl.confirmTip} aria-label={ctl.confirmTip} onClick={() => touchActions.confirm()}><Icon.check /></button>
        )}
        {ctl.back && <button class="btn icon tb back" data-tip={t("touchBar.undo")} aria-label={t("touchBar.undo")} onClick={() => touchActions.back()}><Icon.undo /></button>}
        <button class="btn icon tb stop" data-tip={ctl.stopTip} aria-label={ctl.stopTip} onClick={() => touchActions.stop()}><Icon.cross /></button>
      </div>
    </div>
  );
}
