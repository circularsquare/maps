// T-2901: the world's terrain as one file per degree block (pipeline/world_terrain.py), read by the
// clock worker and the page alike. No imports, so a worker can use it.

/** a block's lattice cells a side (`.terrain` files are 960 x 960) */
export const BLOCK_CELLS = 960;

const M_PER_DEG = 111_320;

/** A degree block as one number: `(la + 90) * 360 + (lo + 180)`, `la` and `lo` the block's south-west
 * corner in whole degrees. Must match sim `cost::block_key`. */
export function blockKey(lon: number, lat: number): number {
  const lo = Math.min(359, Math.floor((((lon + 180) % 360) + 360) % 360));
  const la = Math.max(0, Math.min(179, Math.floor(lat + 90)));
  return la * 360 + lo;
}

/** The file of a block, `<la>_<lo>.terrain` by its south-west corner. */
export const blockFile = (key: number) => `${Math.floor(key / 360) - 90}_${(key % 360) - 180}.terrain`;

/** The blocks a lon/lat box reaches, `marginM` metres round it. */
export function blocksTouching(w: number, s: number, e: number, n: number, marginM = 0): number[] {
  const dy = marginM / M_PER_DEG;
  const dx = dy / Math.cos((Math.min(85, Math.max(Math.abs(s), Math.abs(n))) * Math.PI) / 180);
  const la0 = Math.max(-90, Math.floor(s - dy)), la1 = Math.min(89, Math.floor(n + dy));
  const lo0 = Math.floor(w - dx), lo1 = Math.min(lo0 + 359, Math.floor(e + dx));
  const out: number[] = [];
  for (let la = la0; la <= la1; la++) for (let lo = lo0; lo <= lo1; lo++) out.push(blockKey(lo + 0.5, la + 0.5));
  return out;
}

/** Which blocks have a file: `index.bin`, as sim `cost::BlockIndex`. */
export interface BlockIndex { south: number; rows: number; west: number; cols: number; bits: Uint8Array }

/** `TWI1`, i16 south, u16 rows, i16 west, u16 columns, u32 blocks listed, then a bit a block row-major
 * from the south-west, the first in the low bit of each byte; null if it is not one. */
export function parseBlockIndex(b: Uint8Array): BlockIndex | null {
  if (b.length < 16 || b[0] !== 0x54 || b[1] !== 0x57 || b[2] !== 0x49 || b[3] !== 0x31) return null;
  const v = new DataView(b.buffer, b.byteOffset, b.byteLength);
  const south = v.getInt16(4, true), rows = v.getUint16(6, true), west = v.getInt16(8, true), cols = v.getUint16(10, true);
  const bits = b.subarray(16);
  return rows > 0 && cols > 0 && Math.ceil((rows * cols) / 8) === bits.length ? { south, rows, west, cols, bits } : null;
}

export function indexHas(ix: BlockIndex, key: number): boolean {
  const la = Math.floor(key / 360) - 90 - ix.south, lo = (key % 360) - 180 - ix.west;
  if (la < 0 || lo < 0 || la >= ix.rows || lo >= ix.cols) return false;
  const i = la * ix.cols + lo;
  return ((ix.bits[i >> 3] >> (i & 7)) & 1) === 1;
}
