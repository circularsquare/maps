// T-1200: what the page is working out while it freezes (game/busy.ts), as a small tag on the map's
// top row after the demand status; it comes and goes in one step. Not while the loading card is up,
// which already says the game is loading.

import { busyText } from "../game/busy";
import { loadingCardShown } from "./LoadingScreen";

export function BusyTag() {
  const text = busyText.value;
  if (!text || loadingCardShown()) return null;
  return <div class="status-tag busy-tag" role="status">
    <span class="demand-busy">{text}</span>
  </div>;
}
