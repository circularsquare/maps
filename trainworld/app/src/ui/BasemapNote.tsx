// T-4275: OpenFreeMap's style or tiles refused by every route (map/openMapTiles.ts `basemapBlocked`), so the map has no
// roads, water or labels: said at the bottom of the map once the game is ready, as DeviceNote is, until its close
// button hides it for the rest of the load; taps go through the words. Nothing shows for a player who isn't blocked.

import { computed, signal } from "@preact/signals";
import { t } from "../i18n";
import { gameReady } from "../game/loading";
import { basemapBlocked } from "../map/openMapTiles";
import { deviceNoteShown } from "./DeviceNote";
import { retryNoteShown } from "./RetryNote";
import { Icon } from "./widgets";

const closed = signal(false);

/** the note is on show (ui/InstallBanner.tsx waits for it; it waits for the device and retry notes: they share a place) */
export const basemapNoteShown = computed(() =>
  basemapBlocked.value && gameReady.value && !closed.value && !deviceNoteShown.value && !retryNoteShown.value);

export function BasemapNote() {
  if (!basemapNoteShown.value) return null;
  return <div class="no-market-note" role="status">
    <span>{t("basemapNote.text")}</span>
    <button class="btn icon" aria-label={t("modal.close")} onClick={() => { closed.value = true; }}><Icon.cross /></button>
  </div>;
}
