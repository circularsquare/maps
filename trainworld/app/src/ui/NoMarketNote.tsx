// T-2802: a small device opens no market, so a game there has no trips between cities. Said once per game
// load, at the bottom of the map: taps go through the words, so it never stands between the player and the map or
// the tutorial; it has a close button and goes by itself after a while. Mounted only on a small device (main.ts).

import { effect, signal } from "@preact/signals";
import { t } from "../i18n";
import { gameReady } from "../game/loading";
import { Icon } from "./widgets";

const SHOWN_MS = 45_000;
const shown = signal(false);
let timer = 0;

const hide = () => { clearTimeout(timer); timer = 0; shown.value = false; };

let wasReady = false;
effect(() => {
  const ready = gameReady.value;
  if (ready === wasReady) return;
  wasReady = ready;
  hide();
  if (ready) { shown.value = true; timer = window.setTimeout(hide, SHOWN_MS); }
});

export function NoMarketNote() {
  if (!shown.value) return null;
  return <div class="no-market-note" role="status">
    <span>{t("noMarketNote.text")}</span>
    <button class="btn icon" aria-label={t("modal.close")} onClick={hide}><Icon.cross /></button>
  </div>;
}
