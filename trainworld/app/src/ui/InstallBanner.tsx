// T-4106 (notes/T-4106.md): on a phone or tablet, after 2 hours of play in this browser (game/hoursPlayed.ts, which
// counts across visits), a note at the bottom of the map on how to install the game as an app. Installing gives no
// more memory (notes/T-4086.md); on iOS it keeps saves past Safari's 7-day clearing. The close button hides it for good; left
// alone it goes after a minute and comes back on a later visit. Never in the installed app, never while a notice or a
// question is up, never beside the device or retry note that share its place.

import { computed, effect, signal } from "@preact/signals";
import { t } from "../i18n";
import { playedSeconds } from "../game/hoursPlayed";
import { demandStopped, gameReady, loadFailure } from "../game/loading";
import { figuresStopped } from "../game/figuresStopped";
import { modelStopped } from "../game/modelStopped";
import { staleBuild } from "../game/staleBuild";
import { ask } from "../game/ui";
import { deviceNoteShown } from "./DeviceNote";
import { retryNoteShown } from "./RetryNote";
import { Icon } from "./widgets";

const AFTER_S = 2 * 3600;
const SHOWN_MS = 60_000;
const STORE = "trainworld.install-banner";

type Platform = "ios" | "iosSafari" | "android";
/** How this device installs a web app; null on a computer and in apps' own browsers (Facebook, Instagram, an Android
 * web view), which cannot add to the home screen. iPadOS gives a Mac's user agent, with touch. */
function platform(): Platform | null {
  const ua = navigator.userAgent;
  if (/FBAN|FBAV|Instagram|Line\/|; wv\)/.test(ua)) return null;
  if (/iPhone|iPad|iPod/.test(ua) || (/Macintosh/.test(ua) && navigator.maxTouchPoints > 1))
    return /CriOS|FxiOS|EdgiOS|OPiOS|GSA\//.test(ua) ? "ios" : "iosSafari";
  return /Android/.test(ua) ? "android" : null;
}
const installed = () =>
  ["standalone", "fullscreen", "minimal-ui"].some((m) => matchMedia(`(display-mode: ${m})`).matches) ||
  (navigator as Navigator & { standalone?: boolean }).standalone === true;

const dismissed = signal((() => { try { return localStorage.getItem(STORE) === "1"; } catch { return false; } })());
function dismiss() {
  dismissed.value = true;
  try { localStorage.setItem(STORE, "1"); } catch { /* this page load only */ }
}

/** Chrome's own install prompt (Android), when it offers one; kept to show from the banner's button */
interface InstallPrompt extends Event { prompt(): Promise<void>; userChoice: Promise<{ outcome: "accepted" | "dismissed" }> }
const prompt = signal<InstallPrompt | null>(null);
const where = typeof navigator === "undefined" ? null : platform();
if (where === "android") {
  addEventListener("beforeinstallprompt", (e) => { e.preventDefault(); prompt.value = e as InstallPrompt; });
  addEventListener("appinstalled", dismiss);
}
async function install() {
  const p = prompt.peek();
  if (!p) return;
  prompt.value = null; // a prompt shows once
  await p.prompt();
  await p.userChoice; // installed, or the player said no to the browser's own question: either way, asked
  dismiss();
}

const due = computed(() => !!where && !dismissed.value && gameReady.value && playedSeconds.value >= AFTER_S);
const timedOut = signal(false);
let timer = 0;
effect(() => {
  if (due.value && !timer) timer = window.setTimeout(() => { timedOut.value = true; }, SHOWN_MS);
});

export const installBannerApplies = () => !!where && !installed();

export function InstallBanner() {
  if (!due.value || timedOut.value) return null;
  if (ask.value || demandStopped.value || loadFailure.value || staleBuild.value || figuresStopped.value || modelStopped.value) return null;
  if (deviceNoteShown.value || retryNoteShown.value) return null;
  const android = where === "android", p = android && prompt.value;
  const how = android ? (p ? t("installBanner.android") : t("installBanner.androidMenu"))
    : where === "iosSafari" ? t("installBanner.iosSafari") : t("installBanner.ios");
  return <div class="no-market-note install-banner" role="status">
    <span>{how}</span>
    <span class="install-acts">
      {p && <button class="btn text" onClick={() => void install()}>{t("installBanner.install")}</button>}
      <button class="btn icon" aria-label={t("modal.close")} onClick={dismiss}><Icon.cross /></button>
    </span>
  </div>;
}
