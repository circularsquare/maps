// The track model's problem kinds (`TrackApi.issue_kinds`, sim/src/track/wasm.rs) in plain
// words for the player. Player-facing text: copy rules in Anita's CLAUDE.md.

import { money } from "./format";

const TEXT: Record<string, string> = {
  "Geom:ZeroLeg": "Two points are in the same place.",
  "Geom:Reversal": "The track would turn back on itself.",
  "Geom:RadiusTooSmall": "The curve is too tight. Curves need a radius of at least 100 m.",
  "Geom:ArcDoesNotFit": "The curve does not fit between its neighbours.",
  "Geom:RampTooShort": "Not enough room to change level. Each level up or down needs 200 m of ramp.",
  "Geom:LevelRange": "Levels go from −3 to +3.",
  GroundOverWater: "Track at ground level cannot cross water. Pick a viaduct or tunnel level.",
  Crossing: "The new track would cross other track less than one level above or below it.",
  CrossingOverlap: "The new track would run on top of existing track.",
  NodeHeading: "Track has to leave a junction along the existing track.",
  NodeOneSided: "Track has to run through a station.",
  TooManyPorts: "Too many tracks meet here.",
  StationPorts: "A station has to be on plain track, and there is one here already.",
  PlatformLength: "Platforms are 60 to 400 m long.",
  PlatformTooLong: "The platform would reach a junction or another station. Move it or shorten the platform.",
  PlatformNotLevel: "The platform would be on a ramp.",
  PlatformCrossing: "The platform would sit on a crossing.",
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
  NodeLevel: "Levels go from −3 to +3.",
  Tracks: "Track is single or double.",
};

export function describeIssue(kind: string): string {
  if (kind.startsWith("Money:")) {
    const [, need, have] = kind.split(":").map(Number);
    return `Not enough money. This costs ${money(need * 1e6)} and there is ${money(have * 1e6)}.`;
  }
  return TEXT[kind] ?? "That cannot be built here.";
}

export function describeIssues(kinds: string[]): string {
  return [...new Set(kinds.map(describeIssue))].join(" ");
}
