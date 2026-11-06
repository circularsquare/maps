// The track model's problem kinds (`TrackApi.issue_kinds`, sim/src/track/wasm.rs) in plain
// words for the player. Player-facing text: copy rules in Anita's CLAUDE.md.

import { area, length, money, roughDistance } from "./format";

const TEXT: Record<string, string | (() => string)> = {
  "Geom:ZeroLeg": "Two points are in the same place.",
  "Geom:Reversal": "The track would turn back on itself.",
  "Geom:RadiusTooSmall": () => `The curve is too tight. Curves need a radius of at least ${length(100)}.`,
  "Geom:RampTooShort": () => `Not enough room to change level. Each level up or down needs ${length(200)} of ramp.`,
  "Geom:LevelRange": "Levels go from −3 to +3.",
  GroundOverWater: "Track at ground level cannot cross water. Pick a viaduct or tunnel.",
  BridgeNoRoom: "Not enough space for bridge ramps",
  BridgeNoRoomStation: "Not enough space for a bridge ramp between the station and the water",
  BridgeNoRoomJunction: "Not enough space for a bridge ramp between the junction and the water",
  BridgeNoRoomBuilt: "Not enough space for a bridge ramp between the built track and the water",
  WaterTooLong: () => `Viaducts and cut and cover tunnels cannot cross more than ${roughDistance(30000)} of open water. Use a bored tunnel.`,
  Crossing: "Tracks cannot cross on a ramp unless they are a full level apart.",
  CrossingOverlap: "The new track would run on top of existing track.",
  NodeHeading: "Track has to leave a junction along the existing track.",
  NodeOneSided: "Track has to run through a station.",
  TooManyPorts: "Too many tracks meet here.",
  StationPorts: "There is already a station here.",
  StationJunction: "A station cannot be on a junction.",
  PlatformLength: () => `Platforms are ${length(60)} to ${length(400)} long.`,
  PlatformTooLong: "The track here is too short for this platform.",
  PlatformStation: "There is already a station here.",
  PlatformJunction: "A station cannot be on a junction.",
  PlatformJoint: "The platform would reach the end of a piece of track. Move it or shorten the platform.",
  PlatformNotLevel: "The platform would be on a ramp.",
  PlatformCrossing: "Another track crosses here on the same level. Move the platform or put one of the tracks on another level.",
  StationGroundOverWater: "A station at ground level cannot stand in water.",
  StationInUse: "A line stops here. Take the station out of that line first.",
  LineStops: "A line needs at least two different stations.",
  LinePath: "No track joins these stations in this order.",
  InsertStop: "No track joins these stations with that stop inserted. Check the track connections.",
  DropStop: "The line has to keep this stop.",
  LineSchedule:"Trains an hour cannot be negative.",
  SplitPoint: "Too close to the end of the track, or on a ramp.",
  NodeInUse: "Built track cannot be moved. Remove it and draw it again.",
  NoSuchEdge: "That track is not there any more.",
  NoSuchNode: "That is not there any more.",
  NoSuchLine: "That line is not there any more.",
  EdgeLoop: "Track cannot start and end at the same place.",
  NodeLevel: "Levels go from −3 to +3.",
  Tracks: "Track has to be double or quad.",
  OutsideWorld: "Track cannot go within 5 degrees of a pole or across the 180th meridian.",
  Crossover: "A crossover cannot be reshaped.",
  Alongside: "This track has a pair beside it already.",
  HighSpeed: "High-speed track cannot be doubled up to quad.",
  NothingToConstruct: "Everything here is built already.",
  // T-2613: the clock worker's own refusals of a Construct whose ground it could not fetch
  TerrainMissing: "The terrain here did not load. Try again.",
  RoadsMissing: "The roads here did not load. Try again.",
  BranchNoRoom: "Track cannot branch off just past this platform, because a junction or another station is too close.",
  Internal: "That edit failed and was undone.",
  NoParallel: () => `A crossover needs other track alongside within ${length(15)}, with ${length(60)} of plain track each way.`,
  // T-2218: station buildings (sim/src/track/building.rs)
  BuildingCorners: "A station building has 3 to 24 corners.",
  BuildingEdgesCross: "The outline cannot cross itself.",
  BuildingTooWide: () => `Keep the corners within ${roughDistance(1000)} of each other.`,
  BuildingTooSmall: () => `A station building needs at least ${area(1000)}.`,
  BuildingOverlap: "Station buildings cannot overlap.",
  BuildingNoStation: "Draw the building round a built station.",
  NoSuchBuilding: "That station building is not there any more.",
};

/** What a refused edit would have bought (T-1696): the money refusal names it. */
export type Buys = "trains" | "construction";

export function describeIssue(kind: string, buys?: Buys): string {
  if (kind.startsWith("Money:")) {
    const [, need, have] = kind.split(":").map(Number);
    const cost = `${money(need * 1e6)} and there is ${money(have * 1e6)}`;
    if (buys === "trains") return `Not enough money to buy trains. They cost ${cost}.`;
    if (buys === "construction") return `Not enough money to build this. It costs ${cost}.`;
    return `Not enough money. This costs ${cost}.`;
  }
  const t = TEXT[kind];
  return typeof t === "function" ? t() : t ?? "That cannot be placed here.";
}

/** a platform short of room at both ends says only the first of these it meets (T-1870) */
const ROOM = ["PlatformStation", "PlatformJunction", "PlatformJoint", "PlatformTooLong"];

export function describeIssues(kinds: string[], buys?: Buys): string {
  const room = ROOM.find((k) => kinds.includes(k));
  return [...new Set(kinds.filter((k) => !ROOM.includes(k) || k === room).map((k) => describeIssue(k, buys)))].join(" ");
}
