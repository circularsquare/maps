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
    version: "v0.4.4",
    at: "2026-11-19T22:52:58Z",
    line: "updates.v044Line",
    changes: [
      ["updates.v044Change1", "track"],
      ["updates.v044Change2", "track", "fix"],
      ["updates.v044Change3", "community"],
    ],
  },
  {
    version: "v0.4.3",
    at: "2026-11-19T19:55:21Z",
    line: "updates.v043Line",
    changes: [
      ["updates.v043Change1", "track"],
      ["updates.v043Change2", "track"],
      ["updates.v043Change3", "track"],
      ["updates.v043Change4", "track"],
      ["updates.v043Change5", "track", "fix"],
      ["updates.v043Change6", "demand", "fix"],
      ["updates.v043Change7", "scheduling"],
      ["updates.v043Change8", "scheduling", "fix"],
      ["updates.v043Change9", "fares"],
      ["updates.v043Change10", "fares"],
      ["updates.v043Change11", "schematic"],
      ["updates.v043Change12", "community"],
      ["updates.v043Change13", "community", "fix"],
      ["updates.v043Change14", "performance", "fix"],
      ["updates.v043Change15", "performance", "fix"],
      ["updates.v043Change16", "other"],
      ["updates.v043Change17", "other"],
      ["updates.v043Change18", "other"],
    ],
    highlights: ["updates.v043Change7", "updates.v043Change9", "updates.v043Change10", "updates.v043Change16", "updates.v043Change17"],
  },
  {
    version: "v0.4.2",
    at: "2026-11-18T21:30:43Z",
    line: "updates.v042Line",
    changes: [
      ["updates.v042Change1", "track"],
      ["updates.v042Change2", "track"],
      ["updates.v042Change3", "track"],
      ["updates.v042Change4", "track", "fix"],
      ["updates.v042Change5", "track", "fix"],
      ["updates.v042Change6", "track", "fix"],
      ["updates.v042Change7", "track", "fix"],
      ["updates.v042Change8", "track", "fix"],
      ["updates.v042Change9", "demand"],
      ["updates.v042Change10", "demand"],
      ["updates.v042Change11", "demand"],
      ["updates.v042Change12", "demand", "fix"],
      ["updates.v042Change13", "demand", "fix"],
      ["updates.v042Change14", "scheduling"],
      ["updates.v042Change15", "scheduling"],
      ["updates.v042Change16", "scheduling", "fix"],
      ["updates.v042Change17", "scheduling", "fix"],
      ["updates.v042Change18", "scheduling", "fix"],
      ["updates.v042Change19", "fares"],
      ["updates.v042Change20", "fares"],
      ["updates.v042Change21", "schematic", "fix"],
      ["updates.v042Change22", "schematic", "fix"],
      ["updates.v042Change23", "community"],
      ["updates.v042Change24", "community", "fix"],
      ["updates.v042Change25", "community", "fix"],
      ["updates.v042Change26", "community", "fix"],
      ["updates.v042Change27", "performance"],
      ["updates.v042Change28", "performance"],
      ["updates.v042Change29", "performance"],
      ["updates.v042Change30", "performance", "fix"],
      ["updates.v042Change31", "other"],
      ["updates.v042Change32", "other"],
      ["updates.v042Change33", "other"],
      ["updates.v042Change34", "other"],
      ["updates.v042Change35", "other"],
      ["updates.v042Change36", "other", "fix"],
      ["updates.v042Change37", "other", "fix"],
      ["updates.v042Change38", "other", "fix"],
      ["updates.v042Change39", "other", "fix"],
      ["updates.v042Change40", "other", "fix"],
    ],
    highlights: ["updates.v042Change14", "updates.v042Change16", "updates.v042Change31"],
  },
  {
    version: "v0.4.1",
    at: "2026-11-18T04:53:47Z",
    line: "updates.v041Line",
    changes: [
      ["updates.v041Change1", "track"],
      ["updates.v041Change2", "track"],
      ["updates.v041Change3", "track", "fix"],
      ["updates.v041Change4", "track", "fix"],
      ["updates.v041Change5", "demand", "fix"],
      ["updates.v041Change6", "fares", "fix"],
      ["updates.v041Change7", "schematic"],
      ["updates.v041Change9", "schematic", "fix"],
      ["updates.v041Change10", "schematic", "fix"],
      ["updates.v041Change12", "community", "fix"],
      ["updates.v041Change13", "performance"],
      ["updates.v041Change14", "performance"],
      ["updates.v041Change15", "performance", "fix"],
    ],
  },
  {
    version: "v0.4.0",
    at: "2026-11-18T00:50:09Z",
    line: "updates.v040Line",
    changes: [
      ["updates.v040Change1", "track"],
      ["updates.v040Change5", "track"],
      ["updates.v040Change8", "track"],
      ["updates.v040Change9", "track"],
      ["updates.v040Change10", "track"],
      ["updates.v040Change11", "track"],
      ["updates.v040Change12", "track"],
      ["updates.v040Change13", "track"],
      ["updates.v040Change14", "track"],
      ["updates.v040Change15", "track", "fix"],
      ["updates.v040Change16", "track", "fix"],
      ["updates.v040Change23", "demand"],
      ["updates.v040Change24", "demand"],
      ["updates.v040Change25", "demand"],
      ["updates.v040Change26", "demand"],
      ["updates.v040Change27", "demand", "fix"],
      ["updates.v040Change28", "scheduling"],
      ["updates.v040Change32", "scheduling"],
      ["updates.v040Change33", "scheduling"],
      ["updates.v040Change34", "scheduling"],
      ["updates.v040Change35", "scheduling", "fix"],
      ["updates.v040Change39", "fares"],
      ["updates.v040Change40", "fares"],
      ["updates.v040Change41", "fares", "fix"],
      ["updates.v040Change45", "schematic"],
      ["updates.v040Change46", "schematic"],
      ["updates.v040Change47", "schematic"],
      ["updates.v040Change48", "schematic"],
      ["updates.v040Change49", "schematic"],
      ["updates.v040Change50", "schematic"],
      ["updates.v040Change51", "schematic", "fix"],
      ["updates.v040Change55", "schematic", "fix"],
      ["updates.v040Change61", "community"],
      ["updates.v040Change62", "community"],
      ["updates.v040Change63", "community", "fix"],
      ["updates.v040Change64", "performance"],
      ["updates.v040Change69", "performance"],
      ["updates.v040Change73", "performance"],
      ["updates.v040Change74", "other"],
      ["updates.v040Change75", "other"],
      ["updates.v040Change76", "other"],
      ["updates.v040Change77", "other"],
      ["updates.v040Change78", "bug", "fix"],
    ],
    highlights: ["updates.v040Change74", "updates.v040Change27", "updates.v040Change28"],
    sub: {
      "updates.v040Change1": ["updates.v040Change2", "updates.v040Change3", "updates.v040Change4"],
      "updates.v040Change5": ["updates.v040Change6", "updates.v040Change7"],
      "updates.v040Change16": ["updates.v040Change17", "updates.v040Change18", "updates.v040Change19", "updates.v040Change20", "updates.v040Change21", "updates.v040Change22"],
      "updates.v040Change28": ["updates.v040Change29", "updates.v040Change30", "updates.v040Change31"],
      "updates.v040Change35": ["updates.v040Change36", "updates.v040Change37", "updates.v040Change38"],
      "updates.v040Change41": ["updates.v040Change42", "updates.v040Change43", "updates.v040Change44"],
      "updates.v040Change51": ["updates.v040Change52", "updates.v040Change53", "updates.v040Change54"],
      "updates.v040Change55": ["updates.v040Change56", "updates.v040Change57", "updates.v040Change58", "updates.v040Change59", "updates.v040Change60"],
      "updates.v040Change64": ["updates.v040Change65", "updates.v040Change66", "updates.v040Change67", "updates.v040Change68"],
      "updates.v040Change69": ["updates.v040Change70", "updates.v040Change71", "updates.v040Change72"],
    },
  },
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
