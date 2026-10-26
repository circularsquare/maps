// What limits each kind of capacity resource, in words, for the map's tags and the line panel's
// alert (T-627). The kind numbers are `Capacity::wire_kind` in sim/src/track/capacity.rs.

import { count } from "./format";

export interface CapacityKindText {
  /** the tag's name for it ("Junction at 94% of capacity") */
  name: string;
  /** what is full, and how it sits by its station ("Junction", "near"): "Junction near X is full" */
  title: string;
  prep: "at" | "near";
  /** whose trains the count covers */
  ways: string;
  /** what limits it, one sentence */
  cause: string;
  /** what to do about it, one sentence of advice */
  help: string;
}

const EACH_WAY = " each way";

/** Indexed by kind: 0 track, 1 single track, 2 platform, 3 terminus, 4 junction, 5 flat crossing,
 * 6 a line turning on a through station's platform, 7 track shared by several lines, 8 a line
 * turning across the other track at a through station, 9 lines reversing partway at a station (T-797),
 * 10 a terminal's sidings (T-1281). */
export const CAPACITY_KINDS: CapacityKindText[] = [
  { name: "Track", title: "Track", prep: "near", ways: EACH_WAY, cause: "Each train needs a gap behind it for braking and signals, longer on faster track.", help: "Run fewer trains, or make the track slower so each train needs a shorter gap." },
  { name: "Single track", title: "Single track", prep: "near", ways: " both ways together", cause: "Trains from both directions share one track, and each holds it from one passing point to the next.", help: "Add another track, or a station along it where trains can pass." },
  { name: "Platform", title: "Platform", prep: "at", ways: "", cause: "Each stopping train holds the platform for its dwell and a minute more.", help: "Add another platform track, or stop fewer trains here." },
  { name: "Terminus", title: "Terminus", prep: "at", ways: "", cause: "Each train turning here holds a platform for its turnaround and a minute more.", help: "Add more platform tracks at the terminus." },
  { name: "Junction", title: "Junction", prep: "near", ways: "", cause: "Trains cross each other's path here and take turns.", help: "Add a flyover so trains pass without crossing." },
  { name: "Flat crossing", title: "Flat crossing", prep: "near", ways: "", cause: "Trains on one route wait for the other route to clear.", help: "Take one track over or under the other." },
  { name: "Turning on a platform", title: "Turning on the platform", prep: "at", ways: "", cause: "A line ends at this through station and turns on its platform, which each train holds for the turnaround and a minute more.", help: "Build a terminal at the station from its panel, so trains turn in sidings past the platform." },
  { name: "Shared track", title: "Track shared with other lines", prep: "near", ways: EACH_WAY, cause: "Trains of several lines use this track, and each needs a gap behind it for braking and signals.", help: "Give some of the lines their own track beside it." },
  { name: "Turning across the track", title: "Turning across the track", prep: "at", ways: "", cause: "A train turning at this through station crosses the other track and waits for trains on it.", help: "End the line at a dead-end terminus." },
  { name: "Reversing partway", title: "Reversing", prep: "at", ways: "", cause: "Trains reverse here partway along their line, at most 12 an hour on each side of a station or at a track end, or 30 at a station with a terminal.", help: "Build a terminal at the station, route the line through without reversing, or split it into two lines that end here." },
  { name: "Terminal sidings", title: "Terminal sidings", prep: "at", ways: "", cause: "Each train turning here stands in one of two sidings past the platform for its turnaround and a minute more.", help: "Turn some trains at another station, or end the line at a dead end with more platform tracks." },
];

const kindText = (kind: number): CapacityKindText => CAPACITY_KINDS[kind] ?? CAPACITY_KINDS[0];

/** "Turning on the platform at Nerul Village is full" */
export const fullTitle = (kind: number, place: string) => { const p = fullParts(kind, place); return p.head + place + p.tail; };

/** `fullTitle` around the station's name, so a panel can make the name a link. */
export function fullParts(kind: number, place: string): { head: string; tail: string } {
  const k = kindText(kind);
  return { head: place ? `${k.title} ${k.prep} ` : k.title, tail: " is full" };
}

/** The hover of a full resource: what limits it, how many trains an hour it takes, what helps. */
export function fullTip(kind: number, most: number): string {
  const k = kindText(kind);
  return `${k.cause}\nIt takes at most ${Math.round(most)} trains an hour${k.ways}.\n${k.help}`;
}

/** Trains the line owns but cannot run, in words: "2 trains held out of service". */
export const heldText = (wait: number) => (Math.round(wait) > 1 ? `${count(Math.round(wait))} trains` : "1 train") + " held out of service";

/** What holding them means, for the hover. */
export const heldTip = "These trains wait at the ends of the line instead of running, because no more trains an hour fit through the full spot.";
