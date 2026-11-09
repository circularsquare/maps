// Messages between the main thread (map/bubbles.ts, commuterView.ts, demandHover.ts, demandOverlay.ts,
// intercityView.ts) and the commuter bubble worker (workers/commuters.worker.ts), T-097.

import type { FrameSpec } from "../game/frame";
import type { BubbleEnd } from "../game/types";
import type { Anchor } from "../map/geo";
/** H3 resolutions the bubbles are aggregated at, finest first (cells are resolution 9). */
export const BUBBLE_RES = [9, 8, 7, 6] as const;
/** Average H3 cell area per resolution in BUBBLE_RES, m² (H3's own table). */
export const RES_AREA_M2 = [105333, 737327, 5161293, 36129062];

/** Floats per disc: x, y (local units), radius (local units), r, g, b, a (0-1), edge (0 none,
 * 1 white edge, 2 dark outline). */
export const DISC = 8;

/**
 * Trip legs per m² of bubble, per end, so the bubbles' total area is the same at every aggregation
 * level. Destinations at 160,000 a km²: job centres are several times denser than any home area.
 * Homes at 50,000 a km² with the radius then cut by a quarter (T-1294, Anita), so about 88,900.
 * Total (both ends summed, T-1903) at the destinations' density.
 */
export const BUBBLE_DENSITY = { home: 50000 / 0.75 ** 2 / 1e6, work: 160000 / 1e6, both: 160000 / 1e6 } as const;

export type ToCommuters =
  /** `anchor`: the draw anchor positions are measured from (T-317, map/geo.ts); every answer says
   * which one its positions are relative to */
  | { kind: "cells"; xy: Float32Array; h3: BigUint64Array; frame: FrameSpec; anchor: Anchor }
  /** the draw anchor moved: measure the cells from the new one and send them back (`Moved`) */
  | { kind: "anchor"; id: number; anchor: Anchor }
  /** `none`: T-2215's trips no way serves (No route) */
  | { kind: "build"; id: number; end: BubbleEnd; train: Float32Array; walk: Float32Array; drive: Float32Array; none: Float32Array }
  /** a set of cells with their own counts (a selection's far end, up to ~80k cells), summed
   * into every level */
  | { kind: "sum"; id: number; cells: Uint32Array; train: Float32Array; walk: Float32Array; drive: Float32Array; none?: Float32Array };

/** Groups at one level (map/bubbles.ts `Groups`), as the worker sends them. */
export interface Summed {
  kind: "summed";
  id: number;
  /** one per BUBBLE_RES: group, x, y, train, walk, drive, no route per group with anyone in it */
  levels: { group: Uint32Array; x: Float32Array; y: Float32Array; train: Float32Array; walk: Float32Array; drive: Float32Array; none: Float32Array }[];
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
  /** per disc: its group, and its commuters by train, walking, driving and no route (T-2215) */
  group: Uint32Array;
  train: Float32Array;
  walk: Float32Array;
  drive: Float32Array;
  none: Float32Array;
  /** per disc, the same smoothed among nearby cells (T-900): its colour only */
  shade?: { train: Float32Array; walk: Float32Array; drive: Float32Array; none: Float32Array };
  /** T-2886: per disc, trips for work, errands and leisure, and the rest (and the same smoothed), only
   * while bubbles are coloured by purpose; the main thread puts it there */
  purp?: { work: Float32Array; errands: Float32Array; other: Float32Array; shade?: { work: Float32Array; errands: Float32Array; other: Float32Array } };
}

export interface Bubbles {
  kind: "bubbles";
  id: number;
  end: BubbleEnd;
  /** one per BUBBLE_RES */
  levels: BubbleLevel[];
  /** commuters by mode, the whole city: train, walk, drive, no route */
  totals: [number, number, number, number];
  /** the anchor the discs' positions are relative to */
  anchor: Anchor;
  ms: number;
  /** the colour scheme the main thread last painted the discs in (game/palette.ts
   * `bubbleSchemeKey`); the worker builds them as the mode mix, "mode" */
  coloured?: string;
}

export type FromCommuters = Ready | Bubbles | Summed | Moved;
