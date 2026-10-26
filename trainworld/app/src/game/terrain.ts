// T-1270: the terrain classes the track price multiplies by (sim `TERRAIN_FACTOR`, params.rs), with
// their names and the colour the track tool's tag and the blueprint breakdown give them.

import { signal } from "@preact/signals";

// T-1394: plain names (Anita suggested "slanted / hills / mountain"), DECISIONS.md
export const TERRAIN_NAMES = ["Flat", "Gentle slope", "Hills", "Mountains"] as const;
export const TERRAIN_FACTORS = [1, 1.25, 1.6, 2.5] as const;
export const TERRAIN_COLOURS = ["#d8d4c8", "#d9c08a", "#c99a5b", "#a8703e"] as const;

/** "Hills, track costs 1.6x" for a class 0-3. */
export function terrainText(c: number): string {
  const k = Math.max(0, Math.min(3, Math.round(c)));
  return k ? `${TERRAIN_NAMES[k]}, track costs ${TERRAIN_FACTORS[k]}x` : TERRAIN_NAMES[0];
}

/** A zone's terrain grid as the track model holds it (sim `cost::ClassGrid`): classes on the world
 * lattice of `perDeg` cells a degree, row-major from the south-west cell at `col0`, `row0`. */
export interface ClassGrid { perDeg: number; col0: number; row0: number; w: number; h: number; cls: Uint8Array }

/** The grids the clock worker has put in the track model, by zone (its "terrain" message), so the
 * map's tint shows what the price reads. Kept across games: the lattice is the world's. */
export const terrainGrids = new Map<string, ClassGrid>();
/** Bumped whenever a grid arrives. */
export const terrainVersion = signal(0);

export function addTerrainGrid(id: string, bytes: ArrayBuffer) {
  if (terrainGrids.has(id)) return;
  const g = decodeClassGrid(new Uint8Array(bytes));
  if (!g) return;
  terrainGrids.set(id, g);
  terrainVersion.value++;
}

/** A `.terrain` file (`TWT1`, u32 w, h, i32 col0, row0, u32 cells a degree, then class u8 and
 * LEB128 run pairs), as `ClassGrid::decode` reads it; null if it is not one. */
export function decodeClassGrid(b: Uint8Array): ClassGrid | null {
  if (b.length < 24 || b[0] !== 0x54 || b[1] !== 0x57 || b[2] !== 0x54 || b[3] !== 0x31) return null;
  const v = new DataView(b.buffer, b.byteOffset, b.byteLength);
  const w = v.getUint32(4, true), h = v.getUint32(8, true);
  const col0 = v.getInt32(12, true), row0 = v.getInt32(16, true), perDeg = v.getUint32(20, true);
  const n = w * h;
  if (perDeg <= 0 || n === 0 || n > 64 << 20) return null;
  const cls = new Uint8Array(n);
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
    if (c) cls.fill(c, k, end);
    k = end;
  }
  return k === n ? { perDeg, col0, row0, w, h, cls } : null;
}
