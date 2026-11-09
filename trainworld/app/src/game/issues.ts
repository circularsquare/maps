// The track model's problem kinds (`TrackApi.issue_kinds`, sim/src/track/wasm.rs) in plain
// words for the player. Player-facing text: copy rules in Anita's CLAUDE.md.

import { area, length, money, roughDistance } from "./format";
import { t } from "../i18n";

const TEXT: Record<string, () => string> = {
  "Geom:ZeroLeg": () => t("issues.zeroLeg"),
  "Geom:Reversal": () => t("issues.reversal"),
  "Geom:RadiusTooSmall": () => t("issues.radiusTooSmall", { radius: length(50) }),
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
  PlatformLength: () => t("issues.platformLength", { min: length(60), max: length(400) }),
  PlatformTooLong: () => t("issues.platformTooLong"),
  PlatformJoint: () => t("issues.platformJoint"),
  PlatformNotLevel: () => t("issues.platformNotLevel"),
  PlatformCrossing: () => t("issues.platformCrossing"),
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
  NothingToConstruct: () => t("issues.nothingToConstruct"),
  TerrainMissing: () => t("issues.terrainMissing"),
  RoadsMissing: () => t("issues.roadsMissing"),
  BranchNoRoom: () => t("issues.branchNoRoom"),
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

/** What a refused edit would have bought (T-1696): the money refusal names it. */
export type Buys = "trains" | "construction";

export function describeIssue(kind: string, buys?: Buys): string {
  if (kind.startsWith("Money:")) {
    const [, need, have] = kind.split(":").map(Number);
    const cost = t("issues.moneyCost", { need: money(need * 1e6), have: money(have * 1e6) });
    if (buys === "trains") return t("issues.moneyTrains", { cost });
    if (buys === "construction") return t("issues.moneyBuild", { cost });
    return t("issues.moneyOther", { cost });
  }
  const text = TEXT[kind];
  return text ? text() : t("issues.fallback");
}

/** a platform short of room at both ends says only the first of these it meets (T-1870) */
const ROOM = ["PlatformStation", "PlatformJunction", "PlatformJoint", "PlatformTooLong"]; // i18n-ignore

export function describeIssues(kinds: string[], buys?: Buys): string {
  const room = ROOM.find((k) => kinds.includes(k));
  return [...new Set(kinds.filter((k) => !ROOM.includes(k) || k === room).map((k) => describeIssue(k, buys)))].join(" ");
}
