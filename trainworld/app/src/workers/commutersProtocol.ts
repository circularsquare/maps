// Messages between the main thread (map/demandViews.ts) and the commuter bubble worker
// (workers/commuters.worker.ts), T-097.

import type { FrameSpec } from "../game/frame";
import type { Anchor } from "../map/geo";
/** H3 resolutions the bubbles are aggregated at, finest first (cells are resolution 9). */
export const BUBBLE_RES = [9, 8, 7, 6] as const;
/** Average H3 cell area per resolution in BUBBLE_RES, m² (H3's own table). */
export const RES_AREA_M2 = [105333, 737327, 5161293, 36129062];

/** Floats per disc: x, y (local units), radius (local units), r, g, b, a (0-1), edge (0 none,
 * 1 white edge, 2 dark outline). */
export const DISC = 8;

/**
 * Bubble area per commuter, per end: a bubble covers its commuters at this density (per m²), so
 * the bubbles' total area is the same at every aggregation level. Homes at 25,000 a km², about
 * Manhattan's: dense home areas touch. Jobs at 80,000: job centres are several times denser than
 * any home area, and at the homes' scale Midtown's bubbles covered half of Manhattan.
 */
// Two trip legs per former commuter: retain the old commute bubble area when converting units.
export const BUBBLE_DENSITY = { home: 50000 / 1e6, work: 160000 / 1e6 } as const;

export type ToCommuters =
  /** `anchor`: the draw anchor positions are measured from (T-317, map/geo.ts); every answer says
   * which one its positions are relative to */
  | { kind: "cells"; xy: Float32Array; h3: BigUint64Array; frame: FrameSpec; anchor: Anchor }
  /** the draw anchor moved: measure the cells from the new one and send them back (`Moved`) */
  | { kind: "anchor"; id: number; anchor: Anchor }
  | { kind: "build"; id: number; end: "home" | "work"; train: Float32Array; walk: Float32Array; drive: Float32Array }
  /** a set of cells with their own counts (a selection's far end, up to ~80k cells), summed
   * into every level */
  | { kind: "sum"; id: number; cells: Uint32Array; train: Float32Array; walk: Float32Array; drive: Float32Array };

/** Groups at one level (map/bubbles.ts `Groups`), as the worker sends them. */
export interface Summed {
  kind: "summed";
  id: number;
  /** one per BUBBLE_RES: group, x, y, train, walk, drive per group with anyone in it */
  levels: { group: Uint32Array; x: Float32Array; y: Float32Array; train: Float32Array; walk: Float32Array; drive: Float32Array }[];
  /** the anchor `x` and `y` are relative to */
  anchor: Anchor;
  ms: number;
}

/** The answer to `anchor`: the cells' positions from the new anchor. */
export interface Moved {
  kind: "moved";
  id: number;
  anchor: Anchor;
  cellX: Float32Array;
  cellY: Float32Array;
}

export interface Grouping {
  res: number;
  /** per cell: its group (bubble) at this resolution */
  groupOf: Uint32Array;
  n: number;
  /** cells of group g: members[off[g] .. off[g + 1]) */
  off: Uint32Array;
  members: Uint32Array;
}

export interface Ready {
  kind: "ready";
  ms: number;
  /** per cell, local units of `anchor` (map/geo.ts); the main thread updates the three when a
   * `Moved` arrives */
  cellX: Float32Array;
  cellY: Float32Array;
  anchor: Anchor;
  levels: Grouping[];
}

export interface BubbleLevel {
  /** discs, DISC floats each, biggest first (so smaller ones draw on top) */
  discs: Float32Array;
  /** per disc: its group, and its commuters by train, walking, driving */
  group: Uint32Array;
  train: Float32Array;
  walk: Float32Array;
  drive: Float32Array;
}

export interface Bubbles {
  kind: "bubbles";
  id: number;
  end: "home" | "work";
  /** one per BUBBLE_RES */
  levels: BubbleLevel[];
  /** commuters by mode, the whole city */
  totals: [number, number, number];
  /** the anchor the discs' positions are relative to */
  anchor: Anchor;
  ms: number;
}

export type FromCommuters = Ready | Bubbles | Summed | Moved;
