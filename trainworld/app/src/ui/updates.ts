// T-2810: what each published update brought, newest first, for the start menu's row and its changelog.
// Adding an update is one entry here, with its strings in i18n/messages/en.json and context.json. The row's line is
// at most 80 characters in English (checks/i18n.mjs).
// The row shows the newest entry for two days from its `at`, unless this browser has dismissed that version.

import type { MessageKey } from "../i18n";

/** What a change is about, shown as its icon (Changelog.tsx); the list is sorted in this order (T-3466). "other"
 * fits none of Anita's areas and gets a plain dot; "bug" is a fix that fits no area. */
export const AREAS = ["track", "demand", "scheduling", "fares", "schematic", "community", "performance", "other", "bug"] as const;
export type Area = typeof AREAS[number];
/** one item: its text, its area, and "fix" for a bug fix (drawn a little lighter, after the area's other items) */
export type Change = [key: MessageKey, area: Area, fix?: "fix"];

export interface Update {
  version: string;
  /** ISO time it went out */
  at: string;
  /** the row's one line */
  line: MessageKey;
  /** the changelog's list, shown sorted by area */
  changes: Change[];
  /** T-3591: those of `changes` listed first, above the rest, in this order (a big update's 1-8 biggest changes);
   * left out, the list is as before */
  highlights?: MessageKey[];
  /** those of `changes` that make something harder, each with the key of its new figure, which is filled into the
   * line's `{figure}` and shown red (T-3133; only the figure is red, Anita 2026-11-11) */
  nerfs?: Partial<Record<MessageKey, MessageKey>>;
  /** sub-items, shown indented under their parent item of `changes` (a group, like "track bug fixes"), with its area and
   * fix; a sub key is not also in `changes` */
  sub?: Partial<Record<MessageKey, MessageKey[]>>;
}

/** When v0.1 went out: set at publish. */
const V0_1_AT = "2026-11-09T15:00:00Z";

/** When v0.2 went out: set at publish. */
const V0_2_AT = "2026-11-10T17:30:00Z";

/** When v0.2.1 went out: set at publish. */
const V0_2_1_AT = "2026-11-10T20:50:00Z";

/** When v0.3 went out: set at publish. */
const V0_3_AT = "2026-11-11T17:15:00Z";

/** When v0.3.1 went out: set at publish. */
const V0_3_1_AT = "2026-11-12T03:30:00Z";

/** When v0.3.2 went out: set at publish. */
const V0_3_2_AT = "2026-11-12T18:00:00Z";

/** When v0.3.3 went out: set at publish. */
const V0_3_3_AT = "2026-11-13T14:30:00Z";

export const UPDATES: Update[] = [
  {
    version: "v0.3.5",
    at: "2026-11-15T22:45:00Z",
    line: "updates.v035Line",
    changes: [["updates.v035Fixes", "bug", "fix"]],
  },
  {
    version: "v0.3.4",
    at: "2026-11-15T14:15:00Z",
    line: "updates.v034Line",
    changes: [["updates.v034Fixes", "performance", "fix"]],
  },
  {
    version: "v0.3.3",
    at: V0_3_3_AT,
    line: "updates.v033Line",
    changes: [
      ["updates.v033Satellite", "schematic"],
      ["updates.v033OpenSource", "community"],
      ["updates.v033OneWay", "track"],
      ["updates.v033SingleTrack", "track"],
      ["updates.v033Follow", "other"],
      ["updates.v033Bullets", "schematic"],
      ["updates.v033LeftHand", "track"],
      ["updates.v033SBahn", "track"],
      ["updates.v033BusReroute", "track"],
      ["updates.v033RealTime", "other"],
      ["updates.v033Memory", "performance"],
      ["updates.v033Screenshots", "community"],
      ["updates.v033Zones", "demand"],
      ["updates.v033LongTrips", "demand"],
      ["updates.v033Resorts", "demand"],
      ["updates.v033Schematic", "schematic"],
      ["updates.v033Terrain", "schematic"],
      ["updates.v033StationNames", "track"],
      ["updates.v033Networks", "track"],
      ["updates.v033Tunnels", "track"],
      ["updates.v033ZoomButtons", "other"],
      ["updates.v033ZoneNames", "schematic"],
      ["updates.v033HsrFare", "fares"],
      ["updates.v033Fixes", "bug", "fix"],
    ],
    sub: {
      "updates.v033Zones": ["updates.v033NorthKorea"],
      "updates.v033LongTrips": ["updates.v033Airports"],
    },
  },
  {
    version: "v0.3.2",
    at: V0_3_2_AT,
    line: "updates.v032Line",
    changes: [
      ["updates.v032FareWait", "fares", "fix"],
      ["updates.v032FareCap", "fares"],
      ["updates.v032Downloads", "bug", "fix"],
    ],
  },
  {
    version: "v0.3.1",
    at: V0_3_1_AT,
    line: "updates.v031Line",
    changes: [
      ["updates.v031Hsr", "fares", "fix"],
      ["updates.v031Saves", "demand", "fix"],
      ["updates.v031Search", "community"],
      ["updates.v031Played", "community"],
      ["updates.v031Button", "community"],
    ],
  },
  {
    version: "v0.3",
    at: V0_3_AT,
    line: "updates.v03Line",
    changes: [
      ["updates.v03Flyovers", "track"],
      ["updates.v03Single", "track"],
      ["updates.v03Share", "community"],
      ["updates.v03Rebuild", "track"],
      ["updates.v03Boards", "community"],
      ["updates.v03Roads", "track"],
      ["updates.v03Quad", "track"],
      ["updates.v03Complexes", "track"],
      ["updates.v03Loops", "track"],
      ["updates.v03Ferries", "track"],
      ["updates.v03Curves", "track"],
      ["updates.v03Catalan", "other"],
      ["updates.v03NerfTerminals", "scheduling"],
      ["updates.v03NerfTrams", "scheduling"],
      ["updates.v03NerfJunctions", "scheduling"],
      ["updates.v03Fixes", "bug", "fix"],
    ],
    nerfs: {
      "updates.v03NerfTerminals": "updates.v03NerfTerminalsFigure",
      "updates.v03NerfTrams": "updates.v03NerfTramsFigure",
      "updates.v03NerfJunctions": "updates.v03NerfJunctionsFigure",
    },
  },
  {
    version: "v0.2.1",
    at: V0_2_1_AT,
    line: "updates.v021Line",
    changes: [
      ["updates.v021Hsr", "track", "fix"],
      ["updates.v021Terminals", "track", "fix"],
      ["updates.v021VehicleNames", "bug", "fix"],
      ["updates.v021Transit", "demand"],
      ["updates.v021TramCrossing", "track"],
      ["updates.v021Memory", "performance"],
      ["updates.v021Fixes", "bug", "fix"],
    ],
  },
  {
    version: "v0.2",
    at: V0_2_AT,
    line: "updates.v02Line",
    changes: [
      ["updates.v02Trams", "track"],
      ["updates.v02TramCrossing", "track"],
      ["updates.v02Loops", "track"],
      ["updates.v02BusLoops", "track"],
      ["updates.v02Colors", "schematic"],
      ["updates.v02Limit", "scheduling"],
      ["updates.v02BusFare", "fares"],
      ["updates.v02Places", "demand"],
      ["updates.v02Languages", "other"],
      ["updates.v02Hidden", "other"],
      ["updates.v02ColorBlind", "demand"],
      ["updates.v02BusManual", "track"],
      ["updates.v02Crowding", "demand"],
      ["updates.v02Purpose", "demand"],
      ["updates.v02StopOrder", "track"],
      ["updates.v02Terrain", "schematic"],
      ["updates.v02Fixes", "bug", "fix"],
    ],
  },
  {
    version: "v0.1.1",
    at: "2026-11-09T16:40:00Z",
    line: "updates.v011Line",
    changes: [
      ["updates.v011Feedback", "community"],
      ["updates.v011Buses", "track"],
      ["updates.v011Currencies", "fares"],
      ["updates.v011Discord", "community"],
      ["updates.v011Memory", "performance"],
      ["updates.v011Fixes", "bug", "fix"],
    ],
  },
  {
    version: "v0.1",
    at: V0_1_AT,
    line: "updates.v01Line",
    changes: [
      ["updates.v01Currency", "fares"],
      ["updates.v01Spanish", "other"],
      ["updates.v01Curves", "track"],
      ["updates.v01DeleteKeys", "track"],
      ["updates.v01StationsApart", "track"],
      ["updates.v01Rename", "other"],
      ["updates.v01DeleteBig", "other"],
      ["updates.v01Cars", "track"],
      ["updates.v01MainMenu", "other"],
      ["updates.v01MapDetail", "performance"],
      ["updates.v01Phones", "performance"],
      ["updates.v01Fixes", "bug", "fix"],
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
