// T-2810: what each published update brought, newest first, for the start menu's row and its changelog.
// Adding an update is one entry here, with its strings in i18n/messages/en.json and context.json. The row's line is
// at most 80 characters in English (checks/i18n.mjs).
// The row shows the newest entry for two days from its `at`, unless this browser has dismissed that version.

import type { MessageKey } from "../i18n";

export interface Update {
  version: string;
  /** ISO time it went out */
  at: string;
  /** the row's one line */
  line: MessageKey;
  /** the changelog's list */
  changes: MessageKey[];
}

/** When v0.1 went out: set at publish. */
const V0_1_AT = "2026-11-09T15:00:00Z";

/** When v0.2 went out: set at publish. */
const V0_2_AT = "2026-11-10T17:30:00Z";

/** When v0.2.1 went out: set at publish. */
const V0_2_1_AT = "2026-11-10T20:50:00Z";

/** When v0.2.2 went out: set at publish. */
const V0_2_2_AT = "2026-11-11T03:00:00Z";

export const UPDATES: Update[] = [
  {
    version: "v0.2.2",
    at: V0_2_2_AT,
    line: "updates.v022Line",
    changes: [
      "updates.v022Flyovers",
      "updates.v022Single",
      "updates.v022RightClick",
      "updates.v022BusLoops",
      "updates.v022LoopButton",
      "updates.v022Rebuild",
      "updates.v022BlueprintTram",
      "updates.v022Turning",
      "updates.v022StreetTram",
      "updates.v022Curves",
      "updates.v022Catalan",
      "updates.v022Fixes",
    ],
  },
  {
    version: "v0.2.1",
    at: V0_2_1_AT,
    line: "updates.v021Line",
    changes: [
      "updates.v021Hsr",
      "updates.v021Terminals",
      "updates.v021VehicleNames",
      "updates.v021Transit",
      "updates.v021TramCrossing",
      "updates.v021Memory",
      "updates.v021Fixes",
    ],
  },
  {
    version: "v0.2",
    at: V0_2_AT,
    line: "updates.v02Line",
    changes: [
      "updates.v02Trams",
      "updates.v02TramCrossing",
      "updates.v02Loops",
      "updates.v02BusLoops",
      "updates.v02Colors",
      "updates.v02Limit",
      "updates.v02BusFare",
      "updates.v02Places",
      "updates.v02Languages",
      "updates.v02Hidden",
      "updates.v02ColorBlind",
      "updates.v02BusManual",
      "updates.v02Crowding",
      "updates.v02Purpose",
      "updates.v02StopOrder",
      "updates.v02Terrain",
      "updates.v02Fixes",
    ],
  },
  {
    version: "v0.1.1",
    at: "2026-11-09T16:40:00Z",
    line: "updates.v011Line",
    changes: [
      "updates.v011Feedback",
      "updates.v011Buses",
      "updates.v011Currencies",
      "updates.v011Discord",
      "updates.v011Memory",
      "updates.v011Fixes",
    ],
  },
  {
    version: "v0.1",
    at: V0_1_AT,
    line: "updates.v01Line",
    changes: [
      "updates.v01Currency",
      "updates.v01Spanish",
      "updates.v01Curves",
      "updates.v01DeleteKeys",
      "updates.v01StationsApart",
      "updates.v01Rename",
      "updates.v01DeleteBig",
      "updates.v01Cars",
      "updates.v01MainMenu",
      "updates.v01MapDetail",
      "updates.v01Phones",
      "updates.v01Fixes",
    ],
  },
];

const SHOW_MS = 2 * 24 * 3600_000;
const STORE = "trainworld-update-dismissed";

/** The update the start menu's row should show, or null. */
export function updateToShow(now = Date.now()): Update | null {
  const u = UPDATES[0];
  if (!u || now > Date.parse(u.at) + SHOW_MS) return null;
  try { if (localStorage.getItem(STORE) === u.version) return null; } catch { /* storage unavailable */ }
  return u;
}

/** Remember that this browser has dismissed `version`. */
export function dismissUpdate(version: string) {
  try { localStorage.setItem(STORE, version); } catch { /* this page only */ }
}
