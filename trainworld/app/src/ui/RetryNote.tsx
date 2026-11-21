// T-3933: a load that ran out of memory opened again with fewer solving workers (game/oomRetry.ts): said at the bottom
// of the map from the start of that load until a while after it is ready, as DeviceNote is; taps go through the words.

import { effect, signal } from "@preact/signals";
import { t } from "../i18n";
import { demandIdle, gameReady, loadFailure, loadRetried } from "../game/loading";
import { Icon } from "./widgets";

const AFTER_READY_MS = 45_000;
const closed = signal(false);
let timer = 0;

effect(() => {
  if (!loadRetried.value || !(gameReady.value || demandIdle.value) || timer) return;
  timer = window.setTimeout(() => { closed.value = true; }, AFTER_READY_MS);
});

export function RetryNote() {
  if (!loadRetried.value || closed.value || loadFailure.value) return null;
  return <div class="no-market-note" role="status">
    <span>{t("retryNote.text")}</span>
    <button class="btn icon" aria-label={t("modal.close")} onClick={() => { closed.value = true; }}><Icon.cross /></button>
  </div>;
}
