// The track model's problem kinds (`TrackApi.issue_kinds`, sim/src/track/wasm.rs) in plain
// words for the player. Player-facing text: copy rules in Anita's CLAUDE.md.

import { area, length, money, roughDistance } from "./format";
import { t } from "../i18n";
import type { Vehicle } from "./vehicle";
import { TRACK_SPECS } from "./trackKinds";
import { highSpeed, stationOnTram, tool, tramTrack, TRAM_PLATFORM } from "./ui";
import { linesByNum, world } from "./world";

/** T-2926: the track kind the player is working on, for the text of a limit that differs by kind */
const onTram = () => (tool.peek() === "station" ? stationOnTram.peek() : tool.peek() === "track" && tramTrack.peek());
const onHsr = () => tool.peek() === "track" && highSpeed.peek(); // T-3153: other tools do not read the track tool's type

const TEXT: Record<string, () => string> = {
  "Geom:ZeroLeg": () => t("issues.zeroLeg"),
  "Geom:Reversal": () => t("issues.reversal"),
  "Geom:RadiusTooSmall": () => t("issues.radiusTooSmall", { radius: length(TRACK_SPECS[onTram() ? "tram" : onHsr() ? "highSpeed" : "regular"].minRadius) }),
  "Geom:RampTooShort": () => t("issues.rampTooShort", { ramp: length(200) }),
  "Geom:LevelRange": () => t("issues.levelRange"),
  NodeLevel: () => t("issues.levelRange"),
  GroundOverWater: () => t("issues.groundOverWater"),
  BridgeNoRoom: () => t("issues.bridgeNoRoom"),
  BridgeNoRoomStation: () => t("issues.bridgeNoRoomStation"),
  BridgeNoRoomJunction: () => t("issues.bridgeNoRoomJunction"),
  BridgeNoRoomBuilt: () => t("issues.bridgeNoRoomBuilt"),
  WaterTooLong: () => t("issues.waterTooLong", { distance: roughDistance(30000) }),
  Crossing: () => t("issues.crossing"),
  CrossingOverlap: () => t("issues.crossingOverlap"),
  NodeHeading: () => t("issues.nodeHeading"),
  NodeOneSided: () => t("issues.nodeOneSided"),
  TooManyPorts: () => t("issues.tooManyPorts"),
  StationPorts: () => t("issues.stationHere"),
  PlatformStation: () => t("issues.stationHere"),
  StationJunction: () => t("issues.stationOnJunction"),
  PlatformJunction: () => t("issues.stationOnJunction"),
  PlatformLength: () => t("issues.platformLength", onTram() ? { tram: true, min: length(TRAM_PLATFORM.min), max: length(TRAM_PLATFORM.max) } : { tram: false, min: length(60), max: length(400) }),
  PlatformTooLong: () => t("issues.platformTooLong", { tram: onTram() }),
  PlatformJoint: () => t("issues.platformJoint", { tram: onTram() }),
  PlatformNotLevel: () => t("issues.platformNotLevel", { tram: onTram() }),
  PlatformCrossing: () => t("issues.platformCrossing", { tram: onTram() }),
  StationGroundOverWater: () => t("issues.stationInWater"),
  StationInUse: () => t("issues.stationInUse"),
  LineStops: () => t("issues.lineStops"),
  LinePath: () => t("issues.linePath"),
  InsertStop: () => t("issues.insertStop"),
  DropStop: () => t("issues.dropStop"),
  LineSchedule: () => t("issues.lineSchedule"),
  SplitPoint: () => t("issues.splitPoint"),
  NodeInUse: () => t("issues.nodeInUse"),
  NoSuchEdge: () => t("issues.noSuchEdge"),
  StationTrack: () => t("issues.stationTrack"),
  NoSuchNode: () => t("issues.noSuchNode"),
  NoSuchLine: () => t("issues.noSuchLine"),
  EdgeLoop: () => t("issues.edgeLoop"),
  Tracks: () => t("issues.tracks"),
  OutsideWorld: () => t("issues.outsideWorld"),
  Crossover: () => t("issues.crossover"),
  Alongside: () => t("issues.alongside"),
  HighSpeed: () => t("issues.highSpeed"),
  // T-3321: what keeps double track from being rebuilt as single (`single_refusal` in sim/src/track/wasm/rebuild.rs)
  "NoSingle:HighSpeed": () => t("toolBar.singleHsr"),
  "NoSingle:Tram": () => t("issues.singleTram"),
  "NoSingle:Quad": () => t("issues.singleQuad"),
  "NoSingle:Crossover": () => t("issues.singleCrossover"),
  "NoSingle:OneWay": () => t("issues.singleOneWay"),
  "NoSingle:Flyover": () => t("issues.singleFlyover"),
  "NoSingle:Passing": () => t("issues.singlePassing"),
  TrackFamily: () => t("issues.trackFamily"),
  NothingToConstruct: () => t("issues.nothingToConstruct"),
  TerrainMissing: () => t("issues.terrainMissing"),
  RoadsMissing: () => t("issues.roadsMissing"),
  BranchNoRoom: () => t("issues.branchNoRoom", { tram: onTram() }),
  Internal: () => t("issues.internal"),
  NoParallel: () => t("issues.noParallel", { near: length(15), plain: length(60) }),
  BuildingCorners: () => t("issues.buildingCorners"),
  BuildingEdgesCross: () => t("issues.buildingEdgesCross"),
  BuildingTooWide: () => t("issues.buildingTooWide", { distance: roughDistance(1000) }),
  BuildingTooSmall: () => t("issues.buildingTooSmall", { area: area(1000) }),
  BuildingOverlap: () => t("issues.buildingOverlap"),
  BuildingNoStation: () => t("issues.buildingNoStation"),
  NoSuchBuilding: () => t("issues.noSuchBuilding"),
};

/** T-3094: one-way track a line cannot go round; `line` is the line's number */
export function noWay(line: number): string {
  const w = world.peek();
  return t("issues.oneWayNoWay", { line: (w && linesByNum(w).get(line)?.name) ?? "" });
}

/** What a refused edit would have bought (T-1696): the money refusal names it. */
export type Buys = "trains" | "construction";

export function describeIssue(kind: string, buys?: Buys, vehicle: Vehicle = "train"): string {
  if (kind.startsWith("Money:")) {
    const [, need, have] = kind.split(":").map(Number);
    const cost = t("issues.moneyCost", { need: money(need * 1e6), have: money(have * 1e6) });
    if (buys === "trains") return t("issues.moneyTrains", { cost, vehicle });
    if (buys === "construction") return t("issues.moneyBuild", { cost });
    return t("issues.moneyOther", { cost });
  }
  // T-3094: the line that has no other way round one-way track, by its number
  if (kind.startsWith("OneWayNoWay:")) return noWay(Number(kind.split(":")[1]));
  const text = TEXT[kind];
  return text ? text() : t("issues.fallback");
}

/** a platform short of room at both ends says only the first of these it meets (T-1870) */
const ROOM = ["PlatformStation", "PlatformJunction", "PlatformJoint", "PlatformTooLong"]; // i18n-ignore

export function describeIssues(kinds: string[], buys?: Buys, vehicle: Vehicle = "train"): string {
  const room = ROOM.find((k) => kinds.includes(k));
  return [...new Set(kinds.filter((k) => !ROOM.includes(k) || k === room).map((k) => describeIssue(k, buys, vehicle)))].join(" ");
}
