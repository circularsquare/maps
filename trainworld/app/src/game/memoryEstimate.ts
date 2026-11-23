// T-3329: the memory a game takes, shown to the player: an estimate for a saved game from its station count
// (the saves list and the warning on Open), and a reading of the running game (Settings). Fitted on 13 real-network
// saves and a fresh game in headless Chrome, all processes of the tab, notes/T-3329.md, refitted with players'
// largest networks and 3 solving workers in notes/T-3757.md. The all-China save is left out of the fit (it is a
// hand-drawn network of long straight lines, not a reference).
// memoryCheck.ts judges a save before it loads from its zones and must stay cautious; this is the plain figure.

import { workersFor, demandWorkersWasmMB } from "../workers/demandClient";
import { addressLimited, budgetKnown, deviceGB, memBudget, tabLimited } from "../workers/deviceMemory";
import { liveBytes } from "../workers/liveMemory";
import { t } from "../i18n";

const GB = 2 ** 30;
/** GB all of the browser's processes take with a small saved game open, one solving worker, and GB more for each
 * 1,000 stations (T-3757 refit with two 53-64 city saves of 1,174 and 7,070 stations: error within about 20%) */
const BASE_GB = 1.64, PER_1000_STATIONS_GB = 0.8;
/** GB a solving worker takes, and more per 1,000 stations: each past the first adds it, and its size sets how many
 * run (T-3373; T-3757: 7,070 stations measured 2.2 a worker) */
const PER_EXTRA_WORKER_GB = 0.1, PER_EXTRA_WORKER_1000_GB = 0.3;

/** GB a saved game of `stations` will take on this device (its solving workers counted). Past one, solving workers
 * count only while they fit in the budget, as the pool cuts them there (`capPool`). */
export function estimateGB(stations: number): number {
  const worker = PER_EXTRA_WORKER_GB + (PER_EXTRA_WORKER_1000_GB * stations) / 1000;
  const one = BASE_GB + (PER_1000_STATIONS_GB * stations) / 1000;
  const fit = Math.max(0, Math.floor((budgetGB() - one) / worker));
  return one + worker * Math.min(workersFor(worker * GB) - 1, fit);
}

/** GB the running game takes now, from what its demand workers report; null before they have. */
export function liveGB(): number | null {
  const w = demandWorkersWasmMB();
  return w === null ? null : liveBytes(w.solving.map((m) => m * 2 ** 20), w.aux * 2 ** 20) / GB;
}

/** GB the game may take here: Settings' limit, and what a saved game is compared with. */
export const budgetGB = () => memBudget() / GB;

/** T-3940: the stations from which a saved game is big for a phone or tablet that reports no memory (`tabLimited`).
 * On an emulated iPhone the shipped Tokyo start (1,454 stations) peaks at 1.36 GB in the tab; every crashed player game
 * whose count is known had 1,646-2,090, peaking 1.27-1.71 GB against iOS's kill at 1.4-2 GB (notes/T-3940.md). */
const TAB_LIMITED_STATIONS = 1600;
/** T-3940: a saved game of `stations` is big for this phone or tablet (one that reports no memory) */
export const bigForTab = (stations: number | undefined): boolean => tabLimited() && !!stations && stations >= TAB_LIMITED_STATIONS;

/** True when a saved game of `stations` is more than this device can give, where the browser says what the
 * device has, or (T-3940) big for a phone or tablet that does not; elsewhere never, as the budget is a guess there. */
export const overBudget = (stations: number | undefined): boolean =>
  tabLimited() ? bigForTab(stations) : budgetKnown() && !!stations && estimateGB(stations) > budgetGB();

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
