// T-1270: the terrain classes the track price adds for (sim `TERRAIN_SURCHARGE`, params.rs), with
// their names and the colour the track tool's tag and the blueprint breakdown give them.

import { lazyTexts } from "./lazyText";
import { signal } from "@preact/signals";
import { BLOCK_CELLS, blockKey, indexHas, parseBlockIndex, type BlockIndex } from "./terrainBlocks";

export { BLOCK_CELLS, blockKey, blocksTouching } from "./terrainBlocks";

// T-1547: classes 1 and 2 are both "Hills"; the surcharge shown beside the name tells them apart
export const TERRAIN_NAMES = lazyTexts(["terrain.flat", "terrain.hills", "terrain.hills", "terrain.mountains"]);
/** what each class adds to the level's multiplier (sim `TERRAIN_SURCHARGE`, T-1890) */
export const TERRAIN_SURCHARGES = [0, 0.5, 1.2, 3] as const;
/** what water adds to track's multiplier at any level (sim `WATER_SURCHARGE`) */
export const WATER_SURCHARGE = 2.5;
export const TERRAIN_COLOURS = ["#d8d4c8", "#d9c08a", "#c99a5b", "#a8703e"] as const;

/** A block's terrain grid as the track model holds it (sim `cost::ClassGrid`): classes on the world
 * lattice of `perDeg` cells a degree, row-major from the south-west cell at `col0`, `row0`, two bits a
 * cell, the first in the low bits (`classAt`). */
export interface ClassGrid { perDeg: number; col0: number; row0: number; w: number; h: number; cls: Uint8Array }

/** The class of cell `i` (row-major) of a grid. */
export const classAt = (g: ClassGrid, i: number) => (g.cls[i >> 2] >> ((i & 3) << 1)) & 3;

/** The latitude of a mercator y, degrees. */
export const mercLat = (y: number) => (Math.atan(Math.sinh(Math.PI * (1 - 2 * y))) * 180) / Math.PI;

/** The blocks the clock worker sent, by `blockKey` (its "terrain" message), so the map's tint and
 * hillshade show what the price reads. Kept across games: the lattice is the world's. */
export const terrainGrids = new Map<number, ClassGrid>();

/** `index.bin`, once the worker has it: which blocks have a file; null until then. */
let terrainIndex: BlockIndex | null = null;
export function setTerrainIndex(bytes: ArrayBuffer) {
  terrainIndex = parseBlockIndex(new Uint8Array(bytes));
  terrainVersion.value++;
}
/** Whether a block has terrain data: false only when the index says so (open sea). */
export const blockListed = (key: number) => !terrainIndex || indexHas(terrainIndex, key);
/** Bumped whenever a grid arrives. */
export const terrainVersion = signal(0);

/** T-2613: the clock worker's terrain grids and water masks on their way, and those whose last fetch
 * failed (tried again while still needed) */
export const groundNet = signal({ pending: 0, failed: 0 });
/** T-2613: the hillshade's tiles being downloaded now (`map/demFlat.ts`) */
export const shadeDownloads = signal(0);

export function addTerrainGrid(key: number, bytes: ArrayBuffer) {
  if (terrainGrids.has(key)) return;
  const g = decodeClassGrid(new Uint8Array(bytes));
  if (!g || g.perDeg !== BLOCK_CELLS || g.w !== BLOCK_CELLS || g.h !== BLOCK_CELLS) return;
  terrainGrids.set(key, g);
  terrainVersion.value++;
}

/** Blocks the camera has left, past what the page keeps. */
export function dropTerrainGrids(keys: number[]) {
  for (const k of keys) terrainGrids.delete(k);
}

/** A `.terrain` file (`TWT1`, u32 w, h, i32 col0, row0, u32 cells a degree, then class u8 and
 * LEB128 run pairs), as `ClassGrid::decode` reads it; null if it is not one. */
export function decodeClassGrid(b: Uint8Array): ClassGrid | null {
  if (b.length < 24 || b[0] !== 0x54 || b[1] !== 0x57 || b[2] !== 0x54 || b[3] !== 0x31) return null;
  const v = new DataView(b.buffer, b.byteOffset, b.byteLength);
  const w = v.getUint32(4, true), h = v.getUint32(8, true);
  const col0 = v.getInt32(12, true), row0 = v.getInt32(16, true), perDeg = v.getUint32(20, true);
  const n = w * h;
  if (perDeg <= 0 || n === 0 || n > 256 << 20) return null;
  const cls = new Uint8Array((n + 3) >> 2);
  let i = 24, k = 0;
  while (i < b.length && k < n) {
    const c = Math.min(3, b[i++]);
    let run = 0, shift = 0;
    for (;;) {
      if (i >= b.length) return null;
      const x = b[i++];
      run += (x & 0x7f) * 2 ** shift;
      if (!(x & 0x80)) break;
      shift += 7;
      if (shift > 35) return null;
    }
    const end = Math.min(n, k + run);
    if (c) {
      // whole bytes of a run at once: a big zone is tens of millions of cells
      let j = k;
      const whole = end & ~3;
      for (; j < end && j & 3; j++) cls[j >> 2] |= c << ((j & 3) << 1);
      if (j < whole) cls.fill(c * 0x55, j >> 2, whole >> 2), j = whole;
      for (; j < end; j++) cls[j >> 2] |= c << ((j & 3) << 1);
    }
    k = end;
  }
  return k === n ? { perDeg, col0, row0, w, h, cls } : null;
}
