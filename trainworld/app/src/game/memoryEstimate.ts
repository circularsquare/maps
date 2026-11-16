// T-3329: the memory a game takes, shown to the player: an estimate for a saved game from its station count
// (the saves list and the warning on Open), and a reading of the running game (Settings). Fitted on 13 real-network
// saves and a fresh game in headless Chrome, all processes of the tab, notes/T-3329.md. The all-China save is left
// out of the fit (it is a hand-drawn network of long straight lines, not a reference).
// memoryCheck.ts judges a save before it loads from its zones and must stay cautious; this is the plain figure.

import { workersFor, demandWorkersMB } from "../workers/demandClient";
import { addressLimited, budgetKnown, deviceGB, memBudget } from "../workers/deviceMemory";
import { t } from "../i18n";

const GB = 2 ** 30;
/** GB all of the browser's processes take with a small saved game open, one solving worker */
const BASE_GB = 1.8;
/** GB more for each 1,000 stations (R2 0.85 over the 13 saves, error about 0.2 GB) */
const PER_1000_STATIONS_GB = 0.65;
/** GB a solving worker takes, and more per 1,000 stations: each past the first adds it, and its size sets how many
 * run (T-3373; New York 0.55 measured 0.49, notes/T-3373.md) */
const PER_EXTRA_WORKER_GB = 0.2, PER_EXTRA_WORKER_1000_GB = 0.2;
/** The running game: GB outside the workers' own WASM and heap, and GB of browser per GB of them (R2 0.90, error about 0.2 GB) */
const LIVE_BASE_GB = 1.15, LIVE_PER_WORKER_GB = 1.76;

/** GB a saved game of `stations` will take on this device (its solving workers counted). */
export function estimateGB(stations: number): number {
  const worker = PER_EXTRA_WORKER_GB + (PER_EXTRA_WORKER_1000_GB * stations) / 1000;
  return BASE_GB + (PER_1000_STATIONS_GB * stations) / 1000 + worker * (workersFor(worker * GB) - 1);
}

/** GB the running game takes now, from what its demand workers report; null before they have. */
export function liveGB(): number | null {
  const mb = demandWorkersMB();
  return mb === null ? null : LIVE_BASE_GB + (LIVE_PER_WORKER_GB * mb) / 1024;
}

/** GB the game may take here: Settings' limit, and what a saved game is compared with. */
export const budgetGB = () => memBudget() / GB;

/** True when a saved game of `stations` is more than this device can give, where the browser says what the
 * device has; elsewhere never, as the budget is a guess there. */
export const overBudget = (stations: number | undefined): boolean => budgetKnown() && !!stations && estimateGB(stations) > budgetGB();

/** GB the device has, for the warning: the browser's own figure, else the budget (a 32-bit browser). */
export const deviceGBOrBudget = (): number => (deviceGB() !== undefined && budgetKnown() ? deviceGB()! : budgetGB());

/** T-3494: how far the budget is the device\'s own figure. "cap": the browser reports 8 GB, the most it ever reports,
 * so the device has 8 or more; "known": a lower figure it reported, or a 32-bit browser\'s limit; "unknown": the
 * browser does not say (Safari, Firefox; a phone or tablet is assumed 4 GB), so the budget is only a guess and
 * is not shown as a total. Display only: `memBudget` does not read it. */
export type BudgetKind = "cap" | "known" | "unknown";
export function budgetKind(): BudgetKind {
  if (!budgetKnown()) return "unknown";
  return !addressLimited() && (deviceGB() ?? 0) >= 8 ? "cap" : "known";
}

/** T-3494: the device\'s memory for the saves warning, as far as the browser says it ("8 GB or more", "about 4 GB") */
export const deviceText = (): string => budgetKind() === "cap" ? t("savesPanel.haveCap", { gb: gbText(deviceGBOrBudget()) }) : t("savesPanel.haveAbout", { gb: gbText(deviceGBOrBudget()) });

/** One decimal, none when it is a whole number or from 10 up ("2.4", "8", "12") */
export const gbText = (n: number): string => (n < 10 ? String(Math.round(n * 10) / 10) : n.toFixed(0));
