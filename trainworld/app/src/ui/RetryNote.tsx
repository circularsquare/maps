// T-3933, T-4396: a load that ran out of memory opened again with less (game/oomRetry.ts): said at the bottom
// of the map from the start of that load until a while after it is ready, as DeviceNote is; taps go through the words.

import { computed, effect, signal } from "@preact/signals";
import { t, type MessageKey } from "../i18n";
import { demandIdle, gameReady, loadFailure, loadRetried, type RetryKind } from "../game/loading";
import { Icon } from "./widgets";

const AFTER_READY_MS = 45_000;
const closed = signal(false);
let timer = 0;

const RETRY_TEXT: Record<RetryKind, MessageKey> = {
  cores: "retryNote.text",
  intercity: "retryNote.intercity",
  detail: "retryNote.detail",
};

effect(() => {
  if (!loadRetried.value || !(gameReady.value || demandIdle.value) || timer) return;
  timer = window.setTimeout(() => { closed.value = true; }, AFTER_READY_MS);
});

/** the note is on show (ui/InstallBanner.tsx waits for it: they share a place) */
export const retryNoteShown = computed(() => !!loadRetried.value && !closed.value && !loadFailure.value);

export function RetryNote() {
  const kind = loadRetried.value;
  if (!retryNoteShown.value || !kind) return null;
  return <div class="no-market-note" role="status">
    <span>{t(RETRY_TEXT[kind])}</span>
    <button class="btn icon" aria-label={t("modal.close")} onClick={() => { closed.value = true; }}><Icon.cross /></button>
  </div>;
}
