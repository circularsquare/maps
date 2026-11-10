// What limits each kind of capacity resource, in words, for the map's dots, the Trouble list and
// the line panel's alert (T-627). The kind numbers are `Capacity::wire_kind` in sim/src/track/capacity.rs.

import { effect } from "@preact/signals";
import { t, type MessageKey } from "../i18n";

export interface CapacityKindText {
  /** its name in the Trouble list ("Junction at 94% of capacity") */
  readonly name: string;
  /** the map dot's hover, lowercase ("shared track at 81% of capacity", T-1772) */
  readonly short: string;
  /** what is full, and how it sits by its station ("Junction", "near"): "Junction near X is full" */
  readonly title: string;
  readonly prep: "at" | "near";
  /** whose trains the count covers: each way, both ways together, or none said */
  readonly ways: "each" | "both" | "";
  /** what limits it, one sentence */
  readonly cause: string;
  /** what to do about it, one sentence of advice */
  readonly help: string;
  /** the same advice in a few words, for the Issues list's hover (T-2933) */
  readonly fix: string;
}

type KindBase = "track" | "single" | "platform" | "terminus" | "junction" | "flat" | "turnPlatform" | "shared" | "turnAcross" | "reversing" | "sidings" | "merge";
// the texts read their messages when read, so the list follows the language
const kind = (base: KindBase, prep: "at" | "near", ways: "each" | "both" | ""): CapacityKindText => ({
  get name() { return t(`capacityKinds.${base}Name` as MessageKey); }, // i18n-ignore
  get short() { return t(`capacityKinds.${base}Short` as MessageKey); }, // i18n-ignore
  get title() { return t(`capacityKinds.${base}Title` as MessageKey); }, // i18n-ignore
  prep, ways, // i18n-ignore
  get cause() { return t(`capacityKinds.${base}Cause` as MessageKey); }, // i18n-ignore
  get help() { return t(`capacityKinds.${base}Help` as MessageKey); }, // i18n-ignore
  get fix() { return t(`capacityKinds.${base}Fix` as MessageKey); }, // i18n-ignore
});

/** Indexed by kind: 0 track, 1 single track, 2 platform, 3 terminus, 4 junction, 5 flat crossing,
 * 6 a line turning on a through station's platform, 7 track shared by several lines, 8 a line
 * turning across the other track at a through station, 9 lines reversing partway at a station (T-797),
 * 10 a terminal's sidings (T-1281), 12 shared track from where lines join it, a merge (T-2185).
 * 11, behind slower trains, and 13, waiting at passing tracks for faster ones (T-2231), are never a
 * dot or a limit. */
export const CAPACITY_KINDS: CapacityKindText[] = [
  kind("track", "near", "each"),
  kind("single", "near", "both"),
  kind("platform", "at", ""),
  kind("terminus", "at", ""),
  kind("junction", "near", ""),
  kind("flat", "near", ""),
  kind("turnPlatform", "at", ""),
  kind("shared", "near", "each"),
  kind("turnAcross", "at", ""),
  kind("reversing", "at", ""),
  kind("sidings", "at", ""),
];
CAPACITY_KINDS[12] = kind("merge", "near", "");

const kindText = (kind: number): CapacityKindText => CAPACITY_KINDS[kind] ?? CAPACITY_KINDS[0];

/** "Turning on the platform at Nerul Village is full" */
export const fullTitle = (kind: number, place: string) => { const p = fullParts(kind, place); return p.head + place + p.tail; };

/** `fullTitle` around the station's name, so a panel can make the name a link. */
export function fullParts(kind: number, place: string): { head: string; tail: string } {
  const k = kindText(kind);
  if (!place) return { head: t("capacityKinds.fullNoPlace", { title: k.title }), tail: "" };
  // the message is made with a stand-in for the place and cut there, so a language puts the name where it likes
  const [head, tail] = t(k.prep === "at" ? "capacityKinds.fullAt" : "capacityKinds.fullNear", { title: k.title, place: "\u0000" }).split("\u0000");
  return { head, tail };
}

/** The hover of a full resource: what limits it, how many trains an hour it takes, what helps. */
export function fullTip(kind: number, most: number): string {
  const k = kindText(kind);
  return t("capacityKinds.tip", { cause: k.cause, ways: k.ways || "none", most: Math.round(most), help: k.help });
}

/** Whole trains out of service from `TrackApi.line_held`'s share of a fleet: a part train still runs some of the time (T-2935). */
export const heldTrains = (wait: number) => Math.floor(wait + 0.01);

/** Trains the line owns but cannot run, in words: "2 trains held out of service". */
export const heldText = (wait: number) => t("capacityKinds.held", { n: heldTrains(wait) });

/** What holding them means, for the hover. */
export let heldTip = "";
effect(() => { heldTip = t("capacityKinds.heldTip"); });
