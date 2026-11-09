// T-2802, T-2839, T-2861: what a small device is told, once ever per browser, at the bottom of the map: taps go through the
// words, so it never stands between the player and the map or the tutorial; it has a close button and goes by
// itself after a while. Two sentences, memory first: the browser reports 4 GB or less (T-2839), and a phone or
// tablet has no trips between cities unless the Settings toggle is on (T-2802, T-2838). Mounted only where one
// of them can apply (main.ts).

import { effect, signal } from "@preact/signals";
import { t } from "../i18n";
import { gameReady } from "../game/loading";
import { display } from "../game/ui";
import { mobileDevice } from "../workers/deviceMemory";
import { Icon } from "./widgets";

const SHOWN_MS = 45_000;
const shown = signal(false);
let timer = 0;

/** The browser says the device has 4 GB or less (never a guess: Firefox and Safari do not say). */
export const reportsLowMemory = () => {
  const gb = (navigator as Navigator & { deviceMemory?: number }).deviceMemory;
  return typeof gb === "number" && gb <= 4;
};
/** Whether this device can ever be told something (main.ts mounts the note by it) */
export const deviceNoteApplies = () => mobileDevice() || reportsLowMemory();

/** T-2861: the note shows once ever in a browser, then stays away (remembered in localStorage) */
const STORE = "anitabuilder-device-note";
const seenBefore = () => { try { return localStorage.getItem(STORE) === "1"; } catch { return false; } };
const markSeen = () => { try { localStorage.setItem(STORE, "1"); } catch { /* this page load only */ } };
const noteParts = () => {
  const parts: string[] = [];
  if (reportsLowMemory()) parts.push(t("noMarketNote.memory"));
  if (mobileDevice() && !display.peek().mobileIntercity) parts.push(t("noMarketNote.text"));
  return parts;
};

const hide = () => { clearTimeout(timer); timer = 0; shown.value = false; };

let wasReady = false;
effect(() => {
  const ready = gameReady.value;
  if (ready === wasReady) return;
  wasReady = ready;
  hide();
  if (ready && !seenBefore() && noteParts().length) { markSeen(); shown.value = true; timer = window.setTimeout(hide, SHOWN_MS); }
});

export function DeviceNote() {
  if (!shown.value) return null;
  void display.value; // re-render when the intercity toggle changes
  const parts = noteParts();
  if (!parts.length) return null;
  return <div class="no-market-note" role="status">
    <span>{parts.join(" ")}</span>
    <button class="btn icon" aria-label={t("modal.close")} onClick={hide}><Icon.cross /></button>
  </div>;
}
