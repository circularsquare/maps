// The track model's problem kinds (`TrackApi.issue_kinds`, sim/src/track/wasm.rs) in plain
// words for the player. Player-facing text: copy rules in Anita's CLAUDE.md.

import { length, money } from "./format";

const TEXT: Record<string, string | (() => string)> = {
  "Geom:ZeroLeg": "Two points are in the same place.",
  "Geom:Reversal": "The track would turn back on itself.",
  "Geom:RadiusTooSmall": () => `The curve is too tight. Curves need a radius of at least ${length(100)}.`,
  "Geom:ArcDoesNotFit": "The curve does not fit between its neighbours.",
  "Geom:RampTooShort": () => `Not enough room to change elevation. Each step up or down needs ${length(200)} of ramp.`,
  "Geom:LevelRange": "Elevation goes from −3 to +3.",
  GroundOverWater: "Track at ground level cannot cross water. Pick a viaduct or tunnel elevation.",
  BridgeNoRoom: () => `There is not enough land for the bridge ramps, which are ${length(200)} long.`,
  Crossing: "The new track would cross other track less than one elevation step above or below it.",
  CrossingOverlap: "The new track would run on top of existing track.",
  NodeHeading: "Track has to leave a junction along the existing track.",
  NodeOneSided: "Track has to run through a station.",
  TooManyPorts: "Too many tracks meet here.",
  StationPorts: "A station has to be on plain track, and there is one here already.",
  PlatformLength: () => `Platforms are ${length(60)} to ${length(400)} long.`,
  PlatformTooLong: "The platform would reach a junction, the end of the track or another station. Move it or shorten the platform.",
  PlatformJoint: "The platform would reach the end of a piece of track. Move it or shorten the platform.",
  PlatformNotLevel: "The platform would be on a ramp.",
  PlatformCrossing: "Another track crosses here on the same level. Move the platform or put one of the tracks on another level.",
  StationGroundOverWater: "A station at ground level cannot stand in water.",
  StationInUse: "A line stops here. Take the station out of that line first.",
  LineStops: "A line needs at least two different stations.",
  LinePath: "No track joins these stations in this order.",
  LineSchedule: "Trains an hour cannot be negative.",
  SplitPoint: "Too close to the end of the track, or on a ramp.",
  NodeInUse: "Constructed track cannot be moved. Remove it and draw it again.",
  NoSuchEdge: "That track is not there any more.",
  NoSuchNode: "That is not there any more.",
  NoSuchLine: "That line is not there any more.",
  EdgeLoop: "Track cannot start and end at the same place.",
  NodeLevel: "Elevation goes from −3 to +3.",
  Tracks: "Track is single or double.",
  OutsideWorld: "Track cannot be built within 5 degrees of a pole or across the 180th meridian.",
  Crossover: "A crossover keeps its shape. Delete it and place it again.",
  NoParallel: () => `A crossover needs other track alongside within ${length(15)}, with ${length(60)} of plain track each way.`,
};

export function describeIssue(kind: string): string {
  if (kind.startsWith("Money:")) {
    const [, need, have] = kind.split(":").map(Number);
    return `Not enough money. This costs ${money(need * 1e6)} and there is ${money(have * 1e6)}.`;
  }
  const t = TEXT[kind];
  return typeof t === "function" ? t() : t ?? "That cannot be built here.";
}

export function describeIssues(kinds: string[]): string {
  return [...new Set(kinds.map(describeIssue))].join(" ");
}
